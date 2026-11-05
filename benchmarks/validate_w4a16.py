"""Validate W4A16 weight-only quantization on a local Qwen3 checkpoint.

One run produces one JSON document covering:

- storage: dense vs packed bytes on CPU, with a per-component breakdown;
- GPU memory: KV-cache block capacity, steady allocation, and peak allocation;
- numeric parity: greedy-token agreement, logit cosine/error/top-k agreement,
  and perplexity on a fixed passage;
- throughput: prefill and decode tokens per second at batch size one.

``--kernel`` selects the W4A16 matmul path: ``reference`` dequantizes to fp16
before every matmul, ``int4pack`` always calls the fused GPU kernel, and
``auto`` uses the fused kernel only up to a decode-sized token budget.

The script uses only public micro-vLLM components plus the model runner's
initialized KV buffers for the logit pass. It never mutates source code.

Example:
    python benchmarks/validate_w4a16.py \
        --model /home/lenovo/huggingface/Qwen3-0.6B \
        --kernel int4pack \
        --output benchmarks/results/w4a16-validation-qwen3-0.6b-int4pack.json
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import platform
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from micro_vllm.attention.attention import map_cache_slots
from micro_vllm.config import (
    CacheConfig,
    LLMConfig,
    ModelConfig,
    QuantizationConfig,
    SchedulerConfig,
)
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.engine.engine import Engine
from micro_vllm.executor.single_process import SingleProcessExecutor
from micro_vllm.layers.quantization.w4a16 import W4A16Linear
from micro_vllm.model_loader import load_model
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.worker import Worker


SCHEMA_VERSION = "micro-vllm.w4a16-validation.v2"
KERNELS = ("reference", "int4pack", "auto")
DTYPE = torch.float16
BLOCK_SIZE = 16
MAX_MODEL_LEN = 1024
MAX_BATCHED_TOKENS = 512
MAX_NUM_SEQS = 4
GPU_MEMORY_UTILIZATION = 0.7
PPL_TOKENS = 1024
THROUGHPUT_PROMPT_TOKENS = 256
THROUGHPUT_NEW_TOKENS = 16
PARITY_NEW_TOKENS = 32

PROJECTIONS = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)

PARITY_PROMPTS = (
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The three primary colors are red, green, and",
    "In a transformer, self-attention allows each token to",
)

DEFAULT_PPL_TEXT = Path(__file__).resolve().parents[1] / "README_EN.md"


@dataclass(frozen=True, slots=True)
class StorageMeasurement:
    dense_bytes: int
    w4a16_bytes: int
    dense_breakdown: dict[str, int]
    w4a16_breakdown: dict[str, int]

    @property
    def ratio(self) -> float:
        return self.w4a16_bytes / self.dense_bytes


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Build models directly in ``dtype`` to bound host-memory peak."""

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _build_llm_config(
    model_path: Path,
    quantization: QuantizationConfig,
    *,
    device: str,
    dtype: torch.dtype,
) -> LLMConfig:
    return LLMConfig(
        model_config=ModelConfig(
            model=model_path,
            local_files_only=True,
            device=device,
            dtype=dtype,
            quantization=quantization,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=MAX_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
            max_model_len=MAX_MODEL_LEN,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        ),
    )


def _component(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if leaf in PROJECTIONS:
        return "projections"
    if leaf == "embed_tokens":
        return "embedding"
    if leaf == "lm_head":
        return "lm_head"
    return "other"


def _breakdown(model: torch.nn.Module) -> dict[str, int]:
    """Count each stored tensor once, including tied embeddings."""

    totals = {
        "projections": 0,
        "embedding": 0,
        "lm_head": 0,
        "other": 0,
    }
    seen: set[int] = set()
    for name, module in model.named_modules():
        if name == "":
            continue
        tensors = list(module.parameters(recurse=False))
        tensors += list(module.buffers(recurse=False))
        for tensor in tensors:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            totals[_component(name)] += (
                tensor.numel() * tensor.element_size()
            )
    return totals


def measure_storage(
    model_path: Path,
    *,
    group_size: int,
) -> StorageMeasurement:
    """Load both variants on CPU in fp16 and count their stored bytes."""

    with _default_dtype(torch.float16):
        dense = load_model(
            _build_llm_config(
                model_path,
                QuantizationConfig(),
                device="cpu",
                dtype=torch.float16,
            )
        ).to(dtype=torch.float16)
    dense_breakdown = _breakdown(dense)
    dense_bytes = sum(dense_breakdown.values())
    del dense
    gc.collect()

    with _default_dtype(torch.float16):
        quantized = load_model(
            _build_llm_config(
                model_path,
                QuantizationConfig(method="w4a16", group_size=group_size),
                device="cpu",
                dtype=torch.float16,
            )
        ).to(dtype=torch.float16)
    quantized_breakdown = _breakdown(quantized)
    w4a16_bytes = sum(quantized_breakdown.values())
    del quantized
    gc.collect()

    return StorageMeasurement(
        dense_bytes=dense_bytes,
        w4a16_bytes=w4a16_bytes,
        dense_breakdown=dense_breakdown,
        w4a16_breakdown=quantized_breakdown,
    )


def _kv_cache_bytes(worker: Worker) -> int:
    runner = worker._model_runner
    assert runner is not None
    buffers = runner._kv_cache_buffers
    assert buffers is not None
    return sum(
        buffer.numel() * buffer.element_size() for buffer in buffers
    )


def _logits_for_text(
    worker: Worker,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Run one full prefill through the runner model and return logits."""

    runner = worker._model_runner
    assert runner is not None
    model = runner._model
    assert model is not None
    device = input_ids.device
    num_tokens = input_ids.numel()
    values = worker.config.model_config.values
    num_attention_heads = int(values["num_attention_heads"])
    num_key_value_heads = int(
        values.get("num_key_value_heads", num_attention_heads)
    )
    head_dim = values.get("head_dim")
    if head_dim is None:
        head_dim = int(values["hidden_size"]) // num_attention_heads
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache_shape = (
        2,
        num_blocks,
        BLOCK_SIZE,
        num_key_value_heads,
        int(head_dim),
    )
    topology = runner._kv_cache_topology
    assert topology is not None
    buffers = tuple(
        torch.zeros(cache_shape, dtype=DTYPE, device=device)
        for _ in range(topology.page_layer_count)
    )
    layer_caches = runner._bind_layer_kv_caches(buffers)

    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    block_table = torch.arange(
        num_blocks, dtype=torch.int64, device=device
    ).unsqueeze(0)
    slot_mapping = map_cache_slots(positions, block_table[0], BLOCK_SIZE)
    attention = model.layer_attentions[0]
    metadata = attention.build_metadata(
        query_positions=positions,
        query_start_locations=torch.tensor(
            [0, num_tokens], dtype=torch.int64, device=device
        ),
        context_lengths=torch.tensor(
            [num_tokens], dtype=torch.int64, device=device
        ),
        block_table=block_table,
        slot_mapping=slot_mapping,
        block_size=BLOCK_SIZE,
    )
    layer_metadata = (metadata,) * len(model.layers)
    with torch.inference_mode():
        hidden_states = model(
            input_ids,
            positions,
            layer_caches,
            layer_metadata,
        )
        logits = model.compute_logits(hidden_states)
    return logits.cpu()


def run_mode(
    model_path: Path,
    quantization: QuantizationConfig,
    *,
    ppl_token_ids: torch.Tensor,
    throughput_token_ids: list[int],
    parity_token_ids: list[list[int]],
    device: torch.device,
) -> dict[str, object]:
    """Initialize one worker, run all GPU measurements, and release it."""

    config = _build_llm_config(
        model_path,
        quantization,
        device=str(device),
        dtype=DTYPE,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with _default_dtype(DTYPE):
        worker = Worker(config)
    runner = worker._model_runner
    assert runner is not None
    layout = worker.kv_cache_layout
    print(
        f"  initialized {quantization.method}: "
        f"blocks={layout.num_blocks} "
        f"kv={_kv_cache_bytes(worker) / 2**20:.0f} MiB "
        f"allocated={torch.cuda.memory_allocated(device) / 2**20:.0f} MiB",
        flush=True,
    )

    steady_allocated_bytes = torch.cuda.memory_allocated(device)
    kv_cache_bytes = _kv_cache_bytes(worker)
    logits = _logits_for_text(
        worker,
        ppl_token_ids.to(device),
    )

    cache_manager = KVCacheManager(config, layout)
    scheduler = Scheduler(config, cache_manager)
    engine = Engine(scheduler, SingleProcessExecutor(worker))

    generated: list[dict[str, object]] = []
    decode_times: list[float] = []

    def run_request(
        request_id: str,
        token_ids: list[int],
        max_tokens: int,
    ) -> dict[str, object]:
        engine.add_request(
            request_id,
            token_ids,
            SamplingParams(
                max_tokens=max_tokens,
                ignore_eos=True,
            ),
        )
        step_seconds: list[float] = []
        while engine.has_unfinished_requests():
            _synchronize(device)
            started = time.perf_counter()
            engine.step()
            _synchronize(device)
            step_seconds.append(time.perf_counter() - started)
        request = scheduler.get_request(request_id)
        assert request is not None
        return {
            "request_id": request_id,
            "prompt_tokens": len(token_ids),
            "output_token_ids": list(request.output_token_ids),
            "step_seconds": step_seconds,
        }

    throughput = run_request(
        "throughput",
        throughput_token_ids,
        THROUGHPUT_NEW_TOKENS,
    )
    for index, prompt in enumerate(parity_token_ids):
        record = run_request(f"parity-{index}", prompt, PARITY_NEW_TOKENS)
        generated.append(record)

    _synchronize(device)
    # The fused kernel builds its tiled weight lazily on the first fused
    # forward, so this includes that cache when the kernel was selected.
    post_run_allocated_bytes = torch.cuda.memory_allocated(device)
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    worker.shutdown()
    del worker, runner, engine, scheduler, cache_manager
    gc.collect()
    torch.cuda.empty_cache()

    throughput_steps: list[float] = throughput["step_seconds"]  # type: ignore[assignment]
    prefill_seconds = throughput_steps[0]
    decode_seconds = sum(throughput_steps[1:])
    return {
        "num_blocks": layout.num_blocks,
        "block_size": layout.block_size,
        "kv_cache_bytes": kv_cache_bytes,
        "steady_allocated_bytes": steady_allocated_bytes,
        "post_run_allocated_bytes": post_run_allocated_bytes,
        "peak_allocated_bytes": peak_allocated_bytes,
        "logits": logits,
        "throughput": {
            "prompt_tokens": throughput["prompt_tokens"],
            "new_tokens": THROUGHPUT_NEW_TOKENS,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "prefill_tokens_per_second": (
                throughput["prompt_tokens"] / prefill_seconds
            ),
            "decode_tokens_per_second": (
                (THROUGHPUT_NEW_TOKENS - 1) / decode_seconds
            ),
            "step_seconds": throughput_steps,
        },
        "generation": generated,
    }


def _logits_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    """Compare two [tokens, vocab] fp16 logit tensors row by row."""

    assert reference.shape == candidate.shape
    num_tokens = reference.shape[0]
    cosine_sum = 0.0
    cosine_min = 1.0
    top1_matches = 0
    top5_overlap = 0.0
    max_abs_diff = 0.0
    abs_diff_sum = 0.0
    chunk_size = 32

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        left = reference[start:end].to(torch.float32)
        right = candidate[start:end].to(torch.float32)
        cosine = F.cosine_similarity(left, right, dim=-1)
        cosine_sum += float(cosine.sum())
        cosine_min = min(cosine_min, float(cosine.min()))
        top1_matches += int((left.argmax(dim=-1) == right.argmax(dim=-1)).sum())
        left_top5 = left.topk(5, dim=-1).indices
        right_top5 = right.topk(5, dim=-1).indices
        overlap = (
            left_top5.unsqueeze(-1) == right_top5.unsqueeze(-2)
        ).any(dim=-1).sum(dim=-1)
        top5_overlap += float(overlap.to(torch.float32).sum())
        difference = (left - right).abs()
        max_abs_diff = max(max_abs_diff, float(difference.max()))
        abs_diff_sum += float(difference.sum())

    return {
        "tokens": float(num_tokens),
        "cosine_mean": cosine_sum / num_tokens,
        "cosine_min": cosine_min,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": abs_diff_sum / (num_tokens * reference.shape[1]),
        "top1_agreement": top1_matches / num_tokens,
        "top5_overlap": top5_overlap / num_tokens,
    }


def _perplexity(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    """Perplexity of token_ids under next-token logits."""

    assert logits.shape[0] == token_ids.numel()
    if token_ids.numel() < 2:
        raise ValueError("perplexity requires at least two tokens")
    total_loss = 0.0
    chunk_size = 32
    for start in range(0, token_ids.numel() - 1, chunk_size):
        end = min(start + chunk_size, token_ids.numel() - 1)
        chunk = logits[start:end].to(torch.float32)
        targets = token_ids[start + 1 : end + 1].to(torch.int64)
        total_loss += float(
            F.cross_entropy(chunk, targets, reduction="sum")
        )
    return float(torch.exp(torch.tensor(total_loss / (token_ids.numel() - 1))))


def _token_agreement(
    left: Sequence[int],
    right: Sequence[int],
) -> dict[str, object]:
    matches = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matches += 1
    return {
        "match_tokens": matches,
        "total_tokens": len(left),
        "agreement": matches / len(left),
        "first_divergence": None if matches == len(left) else matches,
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
        "attention_operator": "reference",
    }


def run_validation(
    model_path: Path,
    device: torch.device,
    *,
    group_size: int,
    kernel: str,
    ppl_text_path: Path,
) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    ppl_token_ids = tokenizer(
        ppl_text_path.read_text(encoding="utf-8"),
        add_special_tokens=False,
    )["input_ids"][:PPL_TOKENS]
    if len(ppl_token_ids) < PPL_TOKENS:
        raise ValueError(
            f"{ppl_text_path} contains only {len(ppl_token_ids)} tokens; "
            f"need {PPL_TOKENS}"
        )
    ppl_ids = torch.tensor(ppl_token_ids, dtype=torch.int64)
    throughput_ids = ppl_ids[:THROUGHPUT_PROMPT_TOKENS].tolist()
    parity_ids = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for prompt in PARITY_PROMPTS
    ]

    quantized_config = QuantizationConfig(
        method="w4a16", group_size=group_size, kernel=kernel
    )

    print("measuring CPU storage...", flush=True)
    storage = measure_storage(model_path, group_size=group_size)
    print(
        f"  dense={storage.dense_bytes / 2**20:.0f} MiB "
        f"w4a16={storage.w4a16_bytes / 2**20:.0f} MiB",
        flush=True,
    )
    print("running dense fp16...", flush=True)
    dense = run_mode(
        model_path,
        QuantizationConfig(),
        ppl_token_ids=ppl_ids,
        throughput_token_ids=throughput_ids,
        parity_token_ids=parity_ids,
        device=device,
    )
    print(f"running w4a16 (kernel={kernel})...", flush=True)
    quantized = run_mode(
        model_path,
        quantized_config,
        ppl_token_ids=ppl_ids,
        throughput_token_ids=throughput_ids,
        parity_token_ids=parity_ids,
        device=device,
    )

    logits_metrics = _logits_metrics(dense["logits"], quantized["logits"])
    dense_ppl = _perplexity(dense["logits"], ppl_ids)
    quantized_ppl = _perplexity(quantized["logits"], ppl_ids)
    generation = []
    for dense_record, quantized_record, prompt in zip(
        dense["generation"],
        quantized["generation"],
        PARITY_PROMPTS,
        strict=True,
    ):
        generation.append(
            {
                "prompt": prompt,
                "dense_tokens": dense_record["output_token_ids"],
                "w4a16_tokens": quantized_record["output_token_ids"],
                **_token_agreement(
                    dense_record["output_token_ids"],
                    quantized_record["output_token_ids"],
                ),
            }
        )

    for mode in (dense, quantized):
        del mode["logits"]

    return {
        "schema_version": SCHEMA_VERSION,
        "environment": _environment(device),
        "model": {
            "path": str(model_path),
            "quantization": asdict(quantized_config),
            "ppl_text": str(ppl_text_path),
            "ppl_tokens": int(ppl_ids.numel()),
            "throughput_prompt_tokens": THROUGHPUT_PROMPT_TOKENS,
            "throughput_new_tokens": THROUGHPUT_NEW_TOKENS,
            "parity_new_tokens": PARITY_NEW_TOKENS,
        },
        "storage": {
            "dense_bytes": storage.dense_bytes,
            "w4a16_bytes": storage.w4a16_bytes,
            "ratio": storage.ratio,
            "dense_breakdown": storage.dense_breakdown,
            "w4a16_breakdown": storage.w4a16_breakdown,
        },
        "memory": {
            "dense": {
                key: dense[key]
                for key in (
                    "num_blocks",
                    "block_size",
                    "kv_cache_bytes",
                    "steady_allocated_bytes",
                    "post_run_allocated_bytes",
                    "peak_allocated_bytes",
                )
            },
            "w4a16": {
                key: quantized[key]
                for key in (
                    "num_blocks",
                    "block_size",
                    "kv_cache_bytes",
                    "steady_allocated_bytes",
                    "post_run_allocated_bytes",
                    "peak_allocated_bytes",
                )
            },
            "block_gain": (
                quantized["num_blocks"] / dense["num_blocks"] - 1.0
            ),
        },
        "parity": {
            "logits": logits_metrics,
            "perplexity": {
                "dense": dense_ppl,
                "w4a16": quantized_ppl,
                "delta": quantized_ppl - dense_ppl,
                "delta_percent": (quantized_ppl / dense_ppl - 1.0) * 100.0,
            },
            "generation": generation,
        },
        "throughput": {
            "dense": dense["throughput"],
            "w4a16": quantized["throughput"],
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/lenovo/huggingface/Qwen3-0.6B"),
        help="local Hugging Face checkpoint directory",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="symmetric group size along in_features",
    )
    parser.add_argument(
        "--kernel",
        choices=KERNELS,
        default="reference",
        help="W4A16 matmul kernel used by the quantized run",
    )
    parser.add_argument(
        "--ppl-text",
        type=Path,
        default=DEFAULT_PPL_TEXT,
        help="UTF-8 text file whose first ppl-tokens are scored",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/results/w4a16-validation-qwen3-0.6b.json"
        ),
        help="JSON result path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("W4A16 validation requires a CUDA GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    result = run_validation(
        model_path,
        device,
        group_size=args.group_size,
        kernel=args.kernel,
        ppl_text_path=args.ppl_text.expanduser().resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    storage = result["storage"]
    memory = result["memory"]
    parity = result["parity"]
    throughput = result["throughput"]
    print(
        f"storage: dense={storage['dense_bytes'] / 2**20:.1f} MiB "
        f"w4a16={storage['w4a16_bytes'] / 2**20:.1f} MiB "
        f"ratio={storage['ratio']:.3f}"
    )
    print(
        f"kv blocks: dense={memory['dense']['num_blocks']} "
        f"w4a16={memory['w4a16']['num_blocks']} "
        f"gain={memory['block_gain'] * 100:.1f}%"
    )
    print(
        f"logits: cosine={parity['logits']['cosine_mean']:.5f} "
        f"top1={parity['logits']['top1_agreement']:.4f} "
        f"max_abs_diff={parity['logits']['max_abs_diff']:.4f}"
    )
    print(
        f"perplexity: dense={parity['perplexity']['dense']:.4f} "
        f"w4a16={parity['perplexity']['w4a16']:.4f} "
        f"delta={parity['perplexity']['delta_percent']:.2f}%"
    )
    print(
        f"decode tokens/s: dense={throughput['dense']['decode_tokens_per_second']:.2f} "
        f"w4a16={throughput['w4a16']['decode_tokens_per_second']:.2f}"
    )
    print(
        f"prefill tokens/s: dense={throughput['dense']['prefill_tokens_per_second']:.2f} "
        f"w4a16={throughput['w4a16']['prefill_tokens_per_second']:.2f}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
