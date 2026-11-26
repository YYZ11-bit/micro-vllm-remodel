"""Sweep n-gram size and draft count for speculative decoding.

Each grid point invokes ``validate_speculative.py`` in a fresh process, then
collects the JSON summary.  A fresh process is intentional: each configuration
needs a new Scheduler/LLM, and process isolation keeps GPU memory and monkeypatch
state clean.

Example:
    PYTHONPATH=src python benchmarks/sweep_speculative.py \
        --model /root/autodl-tmp/models/Qwen2.5-0.5B \
        --output-dir benchmarks/results/sweep-qwen2.5-0.5b
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().with_name("validate_speculative.py")
REPO_ROOT = SCRIPT.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/sweep-speculative"),
    )
    parser.add_argument(
        "--ngram-sizes",
        type=int,
        nargs="+",
        default=[2, 3, 4],
    )
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        nargs="+",
        default=[2, 4, 6],
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--quantization", choices=("none", "w4a16"), default="none")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--kernel",
        choices=("reference", "int4pack", "auto"),
        default="reference",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def build_command(
    args: argparse.Namespace,
    *,
    ngram_size: int,
    max_draft_tokens: int,
    output_path: Path,
) -> list[str]:
    command = [
        str(args.python),
        str(SCRIPT),
        "--model",
        str(args.model),
        "--output",
        str(output_path),
        "--ngram-size",
        str(ngram_size),
        "--max-draft-tokens",
        str(max_draft_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--repeat",
        str(args.repeat),
        "--dtype",
        args.dtype,
        "--quantization",
        args.quantization,
        "--group-size",
        str(args.group_size),
        "--kernel",
        args.kernel,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
    ]
    if args.batch:
        command.append("--batch")
    return command


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        src if not env.get("PYTHONPATH") else src + os.pathsep + env["PYTHONPATH"]
    )

    rows: list[dict[str, object]] = []
    failures = 0

    for ngram_size, max_draft_tokens in itertools.product(
        args.ngram_sizes,
        args.max_draft_tokens,
    ):
        stem = f"speculative-ngram{ngram_size}-draft{max_draft_tokens}"
        if args.batch:
            stem += "-batch"
        if args.quantization != "none":
            stem += f"-{args.quantization}-g{args.group_size}-{args.kernel}"
        output_path = args.output_dir / f"{stem}.json"
        output_path.unlink(missing_ok=True)
        command = build_command(
            args,
            ngram_size=ngram_size,
            max_draft_tokens=max_draft_tokens,
            output_path=output_path,
        )
        print(
            f"[sweep] ngram_size={ngram_size} "
            f"max_draft_tokens={max_draft_tokens}",
            flush=True,
        )
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        error = None
        document = None
        if output_path.exists():
            document = json.loads(output_path.read_text())
        if document is None or completed.returncode != 0:
            failures += 1
            error = (completed.stderr or completed.stdout)[-2000:]

        row: dict[str, object] = {
            "ngram_size": ngram_size,
            "max_draft_tokens": max_draft_tokens,
            "returncode": completed.returncode,
            "output": str(output_path),
            "error": error,
        }
        if document is not None:
            comparison = document["comparison"]
            row.update(
                {
                    "all_tokens_match": comparison["all_tokens_match"],
                    "speedup": comparison["speedup"],
                    "baseline_median_seconds": comparison[
                        "baseline_median_seconds"
                    ],
                    "speculative_median_seconds": comparison[
                        "speculative_median_seconds"
                    ],
                    "acceptance_rate": document["speculative"][
                        "acceptance_rate"
                    ],
                    "engine_steps": {
                        "baseline": document["baseline"]["stats"][
                            "engine_steps"
                        ],
                        "speculative": document["speculative"]["stats"][
                            "engine_steps"
                        ],
                    },
                }
            )
            if (
                completed.returncode == 0
                and not comparison["all_tokens_match"]
            ):
                failures += 1
        rows.append(row)
        print(
            "  "
            + (
                f"match={row.get('all_tokens_match')} "
                f"speedup={row.get('speedup')} "
                f"acceptance={row.get('acceptance_rate')}"
                if document is not None
                else "FAILED"
            ),
            flush=True,
        )

    summary = {
        "model": str(args.model),
        "mode": "batch" if args.batch else "single",
        "dtype": args.dtype,
        "quantization": {
            "method": args.quantization,
            "group_size": args.group_size,
            "kernel": args.kernel,
        },
        "max_new_tokens": args.max_new_tokens,
        "repeat": args.repeat,
        "rows": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[sweep] wrote {summary_path}")
    if failures:
        raise SystemExit(f"{failures} sweep run(s) failed")


if __name__ == "__main__":
    main()
