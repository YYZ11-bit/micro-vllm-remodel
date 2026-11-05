"""Cross-check micro-vLLM W4A16 quality with a Hugging Face forward.

The script loads a dense HF causal LM in fp16, scores a fixed token prefix,
then replaces every projection weight (q/k/v/o/gate/up/down) with the same
groupwise round-to-nearest int4 weights used by micro-vLLM, and scores again.
If the quantized perplexity matches micro-vLLM's engine measurement, the
runtime dequantization path is correct and the remaining error is inherent to
the quantization scheme.

Example:
    python benchmarks/crosscheck_w4a16_hf.py \
        --model /home/lenovo/huggingface/Qwen3-0.6B --group-size 128
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


ATTENTION_PROJECTIONS = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj"}
)
MLP_PROJECTIONS = frozenset({"gate_proj", "up_proj", "down_proj"})
PROJECTIONS = ATTENTION_PROJECTIONS | MLP_PROJECTIONS
SCOPES = {
    "all": PROJECTIONS,
    "attention": ATTENTION_PROJECTIONS,
    "mlp": MLP_PROJECTIONS,
}


def quantize_dequantize(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply micro-vLLM's symmetric groupwise int4 round-to-nearest scheme."""

    shape = weight.shape
    segments = weight.float().view(-1, group_size)
    max_abs = segments.abs().max(dim=1, keepdim=True).values
    scale = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs / 7.0,
    )
    quantized = torch.round(segments / scale).clamp(-8, 7)
    return (quantized * scale).view(shape).to(weight.dtype)


def perplexity(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    return float(
        torch.exp(
            F.cross_entropy(logits[:-1].float(), token_ids[1:], reduction="mean")
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/lenovo/huggingface/Qwen3-0.6B"),
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--scope",
        choices=tuple(SCOPES),
        default="all",
        help="which projection families to quantize",
    )
    parser.add_argument(
        "--ppl-text",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "README_EN.md",
    )
    parser.add_argument("--ppl-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("this cross-check requires a CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    model_path = args.model.expanduser().resolve()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    token_ids = torch.tensor(
        tokenizer(
            args.ppl_text.expanduser().read_text(encoding="utf-8"),
            add_special_tokens=False,
        )["input_ids"][: args.ppl_tokens],
        dtype=torch.int64,
    )
    if token_ids.numel() < args.ppl_tokens:
        raise ValueError(
            f"text has only {token_ids.numel()} tokens; "
            f"need {args.ppl_tokens}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
    ).eval().to(device)

    with torch.inference_mode():
        dense_logits = model(
            token_ids.unsqueeze(0).to(device),
            use_cache=False,
        ).logits[0].cpu()

    selected = SCOPES[args.scope]
    with torch.no_grad():
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if isinstance(module, torch.nn.Linear) and leaf in selected:
                module.weight.copy_(
                    quantize_dequantize(module.weight, args.group_size)
                )

    with torch.inference_mode():
        quantized_logits = model(
            token_ids.unsqueeze(0).to(device),
            use_cache=False,
        ).logits[0].cpu()

    dense_ppl = perplexity(dense_logits, token_ids)
    quantized_ppl = perplexity(quantized_logits, token_ids)
    del model
    torch.cuda.empty_cache()

    result = {
        "model": str(model_path),
        "scope": args.scope,
        "group_size": args.group_size,
        "ppl_text": str(args.ppl_text),
        "ppl_tokens": int(token_ids.numel()),
        "dense_perplexity": dense_ppl,
        "w4a16_perplexity": quantized_ppl,
        "delta": quantized_ppl - dense_ppl,
        "delta_percent": (quantized_ppl / dense_ppl - 1.0) * 100.0,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
