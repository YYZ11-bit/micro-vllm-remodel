"""Compare W4A16 layer latency for the reference and fused int4 kernels.

The reference path unpacks and dequantizes the whole weight on every forward,
then runs an fp16 matmul. The fused path calls ``_weight_int4pack_mm`` on a
cached tiled weight. This benchmark measures complete layer forwards, which is
the comparison the engine actually makes when it selects a kernel, across the
projection shapes of one model and a sweep of token counts.

A dense fp16 linear on the same weights is measured for context. Timings are
CUDA-event medians over interleaved repetitions; the output includes the
smallest token count at which the fused path stops being slower per shape.

Example:
    python benchmarks/bench_w4a16_kernel.py \
        --model /home/lenovo/huggingface/Qwen3-0.6B \
        --output benchmarks/results/w4a16-kernel-sweep-qwen3-0.6b.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from statistics import median

import torch
import torch.nn.functional as F

from micro_vllm.layers.quantization.w4a16 import W4A16Linear


SCHEMA_VERSION = "micro-vllm.w4a16-kernel-sweep.v1"
DTYPE = torch.float16
GROUP_SIZE = 128
M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
WARMUP_REPETITIONS = 5
REPETITIONS = 30
RANDOM_SEED = 0
WEIGHT_STD = 0.02


@dataclass(frozen=True, slots=True)
class ProjectionShape:
    name: str
    in_features: int
    out_features: int


def projection_shapes(model_config: dict[str, object]) -> tuple[ProjectionShape, ...]:
    """Derive Llama/Qwen-style projection shapes from a config document."""

    hidden_size = int(model_config["hidden_size"])
    intermediate_size = int(model_config["intermediate_size"])
    num_heads = int(model_config["num_attention_heads"])
    num_kv_heads = int(model_config.get("num_key_value_heads", num_heads))
    head_dim = int(
        model_config.get("head_dim")
        or hidden_size // num_heads
    )
    query_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim
    return (
        ProjectionShape("q_proj", hidden_size, query_width),
        ProjectionShape("k_proj", hidden_size, kv_width),
        ProjectionShape("v_proj", hidden_size, kv_width),
        ProjectionShape("o_proj", query_width, hidden_size),
        ProjectionShape("gate_proj", hidden_size, intermediate_size),
        ProjectionShape("up_proj", hidden_size, intermediate_size),
        ProjectionShape("down_proj", intermediate_size, hidden_size),
    )


def _build_layer(
    shape: ProjectionShape,
    *,
    kernel: str,
    group_size: int,
    device: torch.device,
) -> tuple[W4A16Linear, torch.Tensor]:
    torch.manual_seed(RANDOM_SEED)
    weight = torch.randn(shape.out_features, shape.in_features) * WEIGHT_STD
    layer = W4A16Linear(
        shape.in_features,
        shape.out_features,
        bias=False,
        group_size=group_size,
        kernel=kernel,
    )
    layer.load_weight(weight)
    layer = layer.to(device=device, dtype=DTYPE)
    return layer, weight.to(device=device, dtype=DTYPE)


def measure(
    shape: ProjectionShape,
    *,
    group_size: int,
    device: torch.device,
) -> dict[str, object]:
    reference, dense_weight = _build_layer(
        shape, kernel="reference", group_size=group_size, device=device
    )
    fused, _ = _build_layer(
        shape, kernel="int4pack", group_size=group_size, device=device
    )
    calls = {
        "dense": lambda tensor: F.linear(tensor, dense_weight),
        "reference": reference,
        "int4pack": fused,
    }
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    measurements: dict[str, dict[str, float]] = {
        "dense": {},
        "reference": {},
        "int4pack": {},
    }

    with torch.no_grad():
        for num_tokens in M_VALUES:
            x = torch.randn(
                num_tokens, shape.in_features, dtype=DTYPE, device=device
            )
            for call in calls.values():
                for _ in range(WARMUP_REPETITIONS):
                    call(x)
            torch.cuda.synchronize(device)

            # Interleave variants and flip the order every repetition so
            # thermal drift does not systematically favor one path.
            samples: dict[str, list[float]] = {
                "dense": [],
                "reference": [],
                "int4pack": [],
            }
            order = ("dense", "reference", "int4pack")
            for repetition in range(REPETITIONS):
                names = order if repetition % 2 == 0 else tuple(reversed(order))
                for name in names:
                    start.record()
                    calls[name](x)
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end))
            for name, values in samples.items():
                measurements[name][str(num_tokens)] = median(values)

    return {
        "projection": shape.name,
        "in_features": shape.in_features,
        "out_features": shape.out_features,
        "latency_ms": measurements,
    }


def _environment(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": list(
            torch.cuda.get_device_capability(device)
        ),
        "dtype": str(DTYPE),
        "group_size": GROUP_SIZE,
        "repetitions": REPETITIONS,
        "warmup_repetitions": WARMUP_REPETITIONS,
    }


def _aggregate(
    records: Sequence[dict[str, object]],
    num_hidden_layers: int,
) -> dict[str, object]:
    """Sum the seven projections into one decoder layer and one full model."""

    totals: dict[str, dict[str, float]] = {
        "dense": {},
        "reference": {},
        "int4pack": {},
    }
    for record in records:
        latency = record["latency_ms"]
        assert isinstance(latency, dict)
        for name, values in latency.items():
            for num_tokens, milliseconds in values.items():
                totals[name][num_tokens] = (
                    totals[name].get(num_tokens, 0.0) + milliseconds
                )
    per_layer = totals
    per_model = {
        name: {
            num_tokens: milliseconds * num_hidden_layers
            for num_tokens, milliseconds in values.items()
        }
        for name, values in totals.items()
    }
    speedup = {
        str(num_tokens): (
            per_layer["reference"][str(num_tokens)]
            / per_layer["int4pack"][str(num_tokens)]
        )
        for num_tokens in M_VALUES
        if per_layer["int4pack"][str(num_tokens)] > 0
    }
    return {
        "decoder_layer_ms": per_layer,
        "model_ms": per_model,
        "fused_speedup_over_reference": speedup,
    }


def _crossover(
    records: Sequence[dict[str, object]],
) -> dict[str, dict[str, int | None]]:
    """Largest fused win and first fused loss per projection."""

    crossovers: dict[str, dict[str, int | None]] = {}
    for record in records:
        latency = record["latency_ms"]
        assert isinstance(latency, dict)
        faster_through = None
        first_slowdown = None
        for num_tokens in M_VALUES:
            key = str(num_tokens)
            if latency["int4pack"][key] <= latency["reference"][key]:
                faster_through = num_tokens
            elif first_slowdown is None:
                first_slowdown = num_tokens
        crossovers[str(record["projection"])] = {
            "faster_through_tokens": faster_through,
            "first_slowdown_tokens": first_slowdown,
        }
    return crossovers


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/lenovo/huggingface/Qwen3-0.6B"),
        help="local checkpoint directory whose config defines the shapes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/results/w4a16-kernel-sweep-qwen3-0.6b.json"
        ),
        help="JSON result path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    model_path = args.model.expanduser().resolve()
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("the int4pack kernel requires a CUDA GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    shapes = projection_shapes(config)
    records = []
    for shape in shapes:
        print(
            f"measuring {shape.name}: "
            f"{shape.in_features} -> {shape.out_features}",
            flush=True,
        )
        records.append(measure(shape, group_size=GROUP_SIZE, device=device))

    num_hidden_layers = int(config["num_hidden_layers"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "environment": _environment(device),
        "model": {
            "path": str(model_path),
            "num_hidden_layers": num_hidden_layers,
            "m_values": list(M_VALUES),
        },
        "projections": records,
        "aggregate": _aggregate(records, num_hidden_layers),
        "fused_crossover_tokens": _crossover(records),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    aggregate = result["aggregate"]
    assert isinstance(aggregate, dict)
    print()
    print(f"{'tokens':>7} {'dense ms':>10} {'ref ms':>10} {'fused ms':>10} "
          f"{'fused/ref':>10}")
    for num_tokens in M_VALUES:
        key = str(num_tokens)
        layer = aggregate["decoder_layer_ms"]
        speedup = aggregate["fused_speedup_over_reference"][key]
        print(
            f"{num_tokens:>7} "
            f"{layer['dense'][key]:>10.3f} "
            f"{layer['reference'][key]:>10.3f} "
            f"{layer['int4pack'][key]:>10.3f} "
            f"{speedup:>9.2f}x"
        )
    print()
    print("fused crossover tokens:", result["fused_crossover_tokens"])
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
