"""Validate n-gram speculative decoding on a local model checkpoint.

The script runs the same greedy prompts twice:

1. speculative decoding disabled (baseline);
2. n-gram drafting + greedy acceptance enabled.

It checks token-level parity, records draft acceptance statistics, and measures
end-to-end generation latency.  The model must already exist locally; this
script never downloads weights.

Example:
    python benchmarks/validate_speculative.py \
        --model /root/autodl-tmp/models/Qwen2.5-0.5B \
        --output benchmarks/results/speculative-validation-qwen2.5-0.5b.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from micro_vllm.config import (
    CacheConfig,
    LLMConfig,
    ModelConfig,
    QuantizationConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from micro_vllm.llm import LLM
import micro_vllm.core.scheduler as scheduler_module
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.speculative.ngram import NGramDrafter


PROMPTS = (
    (
        "repeated-prose",
        (
            "The quick brown fox jumps over the lazy dog. "
            "The quick brown fox jumps over the lazy dog. "
            "The quick brown fox jumps over the lazy dog. "
            "The quick brown fox jumps over the lazy dog. "
            "The quick brown fox jumps over the lazy dog. "
            "The quick brown fox jumps over the lazy dog. "
            "The quick brown fox"
        ),
    ),
    (
        "repeated-code",
        (
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def add(a, b):\n"
            "    return a + b\n"
        ),
    ),
    (
        "repeated-numbers",
        (
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10. "
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10. "
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10. "
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10. "
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10. "
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10. "
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10. "
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10. "
        ),
    ),
    (
        "general-explanation",
        (
            "Explain in a few sentences why the sky appears blue during the "
            "day and red at sunset."
        ),
    ),
    (
        "general-story",
        (
            "Write a short paragraph about a robot learning to paint. "
            "Use clear, natural language."
        ),
    ),
)


@dataclass
class GenerationStats:
    engine_steps: int = 0
    propose_calls: int = 0
    draft_tokens_proposed: int = 0
    verifications: int = 0
    draft_tokens_verified: int = 0
    draft_tokens_accepted: int = 0

    @property
    def acceptance_rate(self) -> float | None:
        if self.draft_tokens_verified == 0:
            return None
        return self.draft_tokens_accepted / self.draft_tokens_verified

    @property
    def accepted_per_verification(self) -> float | None:
        if self.verifications == 0:
            return None
        return self.draft_tokens_accepted / self.verifications


@dataclass
class PromptResult:
    name: str
    output_token_ids: tuple[int, ...]
    elapsed_seconds: float
    generated_tokens: int
    stats: GenerationStats
    elapsed_seconds_all: tuple[float, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/speculative-validation.json"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--max-draft-tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="run all prompts in one generate call to exercise mixed batches",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "w4a16"),
        default="none",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--kernel",
        choices=("reference", "int4pack", "auto"),
        default="reference",
    )
    return parser.parse_args()


def make_config(
    model_path: Path,
    *,
    speculative: bool,
    args: argparse.Namespace,
) -> LLMConfig:
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    return LLMConfig(
        model_config=ModelConfig(
            model=model_path,
            local_files_only=True,
            device="cuda",
            dtype=dtype,
            attention_operator="sdpa",
            quantization=QuantizationConfig(
                method=args.quantization,
                group_size=args.group_size,
                kernel=args.kernel,
            ),
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=512,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
        ),
        cache_config=CacheConfig(
            block_size=args.block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ),
        speculative_config=(
            SpeculativeConfig(
                enabled=True,
                drafter_type="ngram",
                ngram_size=args.ngram_size,
                max_draft_tokens=args.max_draft_tokens,
                acceptance_method="greedy",
            )
            if speculative
            else SpeculativeConfig()
        ),
    )


def instrument_llm(llm: LLM, stats: GenerationStats) -> None:
    original_step = llm._engine.step

    def counting_step():
        stats.engine_steps += 1
        return original_step()

    llm._engine.step = counting_step


def run_prompts(
    llm: LLM,
    *,
    max_new_tokens: int,
    repeat: int,
    total_stats: GenerationStats,
    prompt_proposal_stats: GenerationStats | None = None,
) -> tuple[list[PromptResult], GenerationStats]:
    params = SamplingParams(
        max_tokens=max_new_tokens,
        ignore_eos=True,
    )
    results: list[PromptResult] = []

    for name, prompt in PROMPTS:
        prompt_stats = GenerationStats()
        output_token_ids: tuple[int, ...] | None = None
        latencies: list[float] = []

        for _ in range(repeat):
            engine_steps_before = total_stats.engine_steps
            if prompt_proposal_stats is not None:
                prompt_proposal_stats.__init__()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = llm.generate(prompt, params)[0]
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)

            if output_token_ids is None:
                output_token_ids = output.output_token_ids
            elif output_token_ids != output.output_token_ids:
                raise RuntimeError(
                    f"greedy output changed across repeats for {name}"
                )

            prompt_stats.engine_steps += (
                total_stats.engine_steps - engine_steps_before
            )
            if prompt_proposal_stats is not None:
                prompt_stats.propose_calls += (
                    prompt_proposal_stats.propose_calls
                )
                prompt_stats.draft_tokens_proposed += (
                    prompt_proposal_stats.draft_tokens_proposed
                )
                prompt_stats.verifications += (
                    prompt_proposal_stats.verifications
                )
                prompt_stats.draft_tokens_verified += (
                    prompt_proposal_stats.draft_tokens_verified
                )
                prompt_stats.draft_tokens_accepted += (
                    prompt_proposal_stats.draft_tokens_accepted
                )

        assert output_token_ids is not None
        total_stats.propose_calls += prompt_stats.propose_calls
        total_stats.draft_tokens_proposed += prompt_stats.draft_tokens_proposed
        total_stats.verifications += prompt_stats.verifications
        total_stats.draft_tokens_verified += prompt_stats.draft_tokens_verified
        total_stats.draft_tokens_accepted += prompt_stats.draft_tokens_accepted
        results.append(
            PromptResult(
                name=name,
                output_token_ids=output_token_ids,
                elapsed_seconds=statistics.median(latencies),
                generated_tokens=len(output_token_ids),
                stats=prompt_stats,
                elapsed_seconds_all=tuple(latencies),
            )
        )
    return results, total_stats


def run_batch(
    llm: LLM,
    *,
    max_new_tokens: int,
    repeat: int,
    total_stats: GenerationStats,
    prompt_proposal_stats: GenerationStats | None = None,
) -> tuple[list[PromptResult], GenerationStats]:
    params = SamplingParams(
        max_tokens=max_new_tokens,
        ignore_eos=True,
    )
    names = [name for name, _ in PROMPTS]
    prompts = [prompt for _, prompt in PROMPTS]
    output_token_ids: tuple[tuple[int, ...], ...] | None = None
    latencies: list[float] = []
    batch_stats = GenerationStats()

    for _ in range(repeat):
        engine_steps_before = total_stats.engine_steps
        if prompt_proposal_stats is not None:
            prompt_proposal_stats.__init__()
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(prompts, params)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)

        current = tuple(output.output_token_ids for output in outputs)
        if output_token_ids is None:
            output_token_ids = current
        elif output_token_ids != current:
            raise RuntimeError("greedy batch output changed across repeats")

        batch_stats.engine_steps += (
            total_stats.engine_steps - engine_steps_before
        )
        if prompt_proposal_stats is not None:
            batch_stats.propose_calls += (
                prompt_proposal_stats.propose_calls
            )
            batch_stats.draft_tokens_proposed += (
                prompt_proposal_stats.draft_tokens_proposed
            )
            batch_stats.verifications += (
                prompt_proposal_stats.verifications
            )
            batch_stats.draft_tokens_verified += (
                prompt_proposal_stats.draft_tokens_verified
            )
            batch_stats.draft_tokens_accepted += (
                prompt_proposal_stats.draft_tokens_accepted
            )

    assert output_token_ids is not None
    total_stats.propose_calls += batch_stats.propose_calls
    total_stats.draft_tokens_proposed += batch_stats.draft_tokens_proposed
    total_stats.verifications += batch_stats.verifications
    total_stats.draft_tokens_verified += batch_stats.draft_tokens_verified
    total_stats.draft_tokens_accepted += batch_stats.draft_tokens_accepted

    median_latency = statistics.median(latencies)
    results = [
        PromptResult(
            name=name,
            output_token_ids=tokens,
            elapsed_seconds=median_latency,
            generated_tokens=len(tokens),
            stats=GenerationStats(**asdict(batch_stats)),
            elapsed_seconds_all=tuple(latencies),
        )
        for name, tokens in zip(names, output_token_ids, strict=True)
    ]
    return results, total_stats


def install_proposal_instrumentation(stats: GenerationStats) -> None:
    original_propose = NGramDrafter.propose

    def counting_propose(
        self: NGramDrafter,
        token_ids: tuple[int, ...],
        max_tokens: int,
    ):
        result = original_propose(self, token_ids, max_tokens)
        stats.propose_calls += 1
        stats.draft_tokens_proposed += len(result.tokens_ids)
        return result

    NGramDrafter.propose = counting_propose


def install_acceptance_instrumentation(stats: GenerationStats) -> None:
    original_accept = scheduler_module.greedy_accept

    def counting_accept(
        proposals: list[int],
        predictions: list[int],
    ):
        result = original_accept(proposals, predictions)
        stats.verifications += 1
        stats.draft_tokens_verified += len(proposals)
        stats.draft_tokens_accepted += len(result.accepted)
        return result

    scheduler_module.greedy_accept = counting_accept


def summarize(
    baseline: list[PromptResult],
    speculative: list[PromptResult],
) -> dict[str, object]:
    parity = []
    for base, spec in zip(baseline, speculative, strict=True):
        first_mismatch = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(base.output_token_ids, spec.output_token_ids)
                )
                if left != right
            ),
            None,
        )
        parity.append(
            {
                "prompt": base.name,
                "match": base.output_token_ids == spec.output_token_ids,
                "baseline_tokens": list(base.output_token_ids),
                "speculative_tokens": list(spec.output_token_ids),
                "first_mismatch_index": first_mismatch,
            }
        )

    baseline_latencies = [result.elapsed_seconds for result in baseline]
    speculative_latencies = [result.elapsed_seconds for result in speculative]
    baseline_median = statistics.median(baseline_latencies)
    speculative_median = statistics.median(speculative_latencies)
    return {
        "all_tokens_match": all(item["match"] for item in parity),
        "parity": parity,
        "baseline_latency_seconds": baseline_latencies,
        "speculative_latency_seconds": speculative_latencies,
        "baseline_median_seconds": baseline_median,
        "speculative_median_seconds": speculative_median,
        "speedup": (
            baseline_median / speculative_median
            if speculative_median > 0
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if args.batch:
        args.max_num_seqs = max(args.max_num_seqs, len(PROMPTS))

    print(f"model={args.model}")
    print(
        f"dtype={args.dtype} block_size={args.block_size} "
        f"mode={'batch' if args.batch else 'single'}"
    )
    print(
        f"quantization={args.quantization} "
        f"group_size={args.group_size} kernel={args.kernel}"
    )
    print(
        f"ngram_size={args.ngram_size} "
        f"max_draft_tokens={args.max_draft_tokens}"
    )
    print(f"gpu={torch.cuda.get_device_name(0)}")
    runner = run_batch if args.batch else run_prompts

    baseline_config = make_config(args.model, speculative=False, args=args)
    baseline_llm = LLM(baseline_config)
    baseline_stats = GenerationStats()
    instrument_llm(baseline_llm, baseline_stats)
    torch.cuda.synchronize()
    baseline_llm.generate("warmup", SamplingParams(max_tokens=2, ignore_eos=True))
    baseline_stats.__init__()
    torch.cuda.reset_peak_memory_stats()
    baseline, baseline_stats = runner(
        baseline_llm,
        max_new_tokens=args.max_new_tokens,
        repeat=args.repeat,
        total_stats=baseline_stats,
    )
    baseline_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    baseline_llm.shutdown()
    del baseline_llm
    gc.collect()
    torch.cuda.empty_cache()

    spec_total_stats = GenerationStats()
    spec_prompt_stats = GenerationStats()
    install_proposal_instrumentation(spec_prompt_stats)
    install_acceptance_instrumentation(spec_prompt_stats)
    speculative_config = make_config(
        args.model,
        speculative=True,
        args=args,
    )
    speculative_llm = LLM(speculative_config)
    instrument_llm(speculative_llm, spec_total_stats)
    speculative_llm.generate(
        "warmup",
        SamplingParams(max_tokens=2, ignore_eos=True),
    )
    # Warmup proposals are not part of the measured prompts.
    spec_total_stats.__init__()
    spec_prompt_stats.__init__()
    torch.cuda.reset_peak_memory_stats()
    speculative, total_spec_stats = runner(
        speculative_llm,
        max_new_tokens=args.max_new_tokens,
        repeat=args.repeat,
        total_stats=spec_total_stats,
        prompt_proposal_stats=spec_prompt_stats,
    )
    speculative_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    speculative_llm.shutdown()
    del speculative_llm
    gc.collect()
    torch.cuda.empty_cache()

    summary = summarize(baseline, speculative)
    document = {
        "schema": "micro-vllm.speculative-validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "mode": "batch" if args.batch else "single",
        "dtype": args.dtype,
        "attention_operator": "sdpa",
        "block_size": args.block_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_new_tokens": args.max_new_tokens,
        "repeat": args.repeat,
        "ngram_size": args.ngram_size,
        "max_draft_tokens": args.max_draft_tokens,
        "quantization": {
            "method": args.quantization,
            "group_size": args.group_size,
            "kernel": args.kernel,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "baseline": {
            "peak_memory_mb": baseline_peak_mb,
            "stats": asdict(baseline_stats),
            "results": [asdict(result) for result in baseline],
        },
        "speculative": {
            "peak_memory_mb": speculative_peak_mb,
            "stats": asdict(total_spec_stats),
            "acceptance_rate": total_spec_stats.acceptance_rate,
            "accepted_per_verification": (
                total_spec_stats.accepted_per_verification
            ),
            "results": [asdict(result) for result in speculative],
        },
        "comparison": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")

    for item in summary["parity"]:
        print(
            f"{item['prompt']}: match={item['match']} "
            f"first_mismatch={item['first_mismatch_index']}"
        )
    print(
        f"baseline median={summary['baseline_median_seconds']:.4f}s "
        f"speculative median={summary['speculative_median_seconds']:.4f}s "
        f"speedup={summary['speedup']:.3f}x"
    )
    print(
        f"proposals={total_spec_stats.draft_tokens_proposed} "
        f"verified={total_spec_stats.draft_tokens_verified} "
        f"accepted={total_spec_stats.draft_tokens_accepted} "
        f"acceptance_rate={total_spec_stats.acceptance_rate}"
    )
    print(
        f"engine_steps baseline={baseline_stats.engine_steps} "
        f"speculative={total_spec_stats.engine_steps}"
    )
    print(f"wrote {args.output}")

    if not summary["all_tokens_match"]:
        raise SystemExit("token parity failed")


if __name__ == "__main__":
    main()
