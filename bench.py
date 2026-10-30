"""Streaming OpenAI-compatible benchmark for vLLM / micro-vLLM baselines.

The script measures the client-visible request lifecycle:

- TTFT: request send -> first non-empty content delta
- E2E: request send -> stream end
- TPOT: (E2E - TTFT) / (completion_tokens - 1)
- ITL: inter-chunk arrival intervals (vLLM normally streams one token per chunk)
- Throughput: successful requests / wall time, output tokens / wall time

It also records prompt/completion token counts from the final `usage` chunk when
the server supports `stream_options.include_usage`.

Example:
    python bench.py --num-requests 100 --concurrency 10 --warmup 10 --repeats 3

Raw per-request results are written to
`benchmarks/results/<label>-<timestamp>.json`.

If the server rejects `stream_options.include_usage`, rerun with
`--no-include-usage`; prompt tokens will then be missing and completion tokens
will fall back to counting content chunks.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen-merged"
DEFAULT_PROMPT = "你好，"


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _mem_total_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return round(kib / 1024**2, 2)
    except OSError:
        pass
    return None


def _gpu_snapshot() -> list[dict[str, str]]:
    """Read GPU facts through nvidia-smi; do not initialize a CUDA context."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,"
                "driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []

    gpus: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        gpus.append(
            {
                "name": parts[0],
                "memory_total_mib": parts[1],
                "memory_used_mib": parts[2],
                "utilization_gpu_percent": parts[3],
                "driver_version": parts[4],
                "compute_capability": parts[5],
            }
        )
    return gpus


def collect_environment() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "mem_total_gib": _mem_total_gib(),
        "packages": {
            name: _package_version(name)
            for name in ("torch", "vllm", "triton", "aiohttp", "numpy")
        },
        "gpu": _gpu_snapshot(),
    }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def _stats(values: list[float | int | None]) -> dict[str, float | int | None]:
    array = np.asarray(
        [value for value in values if value is not None],
        dtype=np.float64,
    )
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize(
    results: list[dict[str, Any]],
    wall_time_s: float,
) -> dict[str, Any]:
    successful = [result for result in results if result["success"]]
    completion_tokens = [
        result["completion_tokens"]
        for result in successful
        if result["completion_tokens"] is not None
    ]
    prompt_tokens = [
        result["prompt_tokens"]
        for result in successful
        if result["prompt_tokens"] is not None
    ]
    usage_count = sum(
        1
        for result in successful
        if result["prompt_tokens"] is not None
        and result["completion_tokens"] is not None
    )

    total_output_tokens = sum(completion_tokens)
    total_prompt_tokens = sum(prompt_tokens)
    total_tokens = total_prompt_tokens + total_output_tokens

    itl = [
        interval
        for result in successful
        for interval in result.get("itl_ms", [])
    ]

    return {
        "attempted": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "wall_time_s": wall_time_s,
        "req_per_s": len(successful) / wall_time_s if wall_time_s > 0 else None,
        "output_tokens_per_s": (
            total_output_tokens / wall_time_s if wall_time_s > 0 else None
        ),
        "total_tokens_per_s": (
            total_tokens / wall_time_s
            if wall_time_s > 0 and usage_count == len(successful)
            else None
        ),
        "usage_available_requests": usage_count,
        "prompt_tokens_sum": total_prompt_tokens if prompt_tokens else 0,
        "completion_tokens_sum": total_output_tokens,
        "content_chunk_count_sum": sum(
            result["content_chunk_count"] for result in successful
        ),
        "suspect_total_tokens_requests": sum(
            1
            for result in successful
            if result["prompt_tokens"] is not None
            and result["completion_tokens"] is not None
            and result["completion_tokens"]
            == result["prompt_tokens"] + result["content_chunk_count"]
        ),
        "ttft_ms": _stats([result["ttft_ms"] for result in successful]),
        "e2e_ms": _stats([result["e2e_ms"] for result in successful]),
        "tpot_ms": _stats([result["tpot_ms"] for result in successful]),
        "itl_ms": _stats(itl),
        "client_wait_ms": _stats(
            [result["client_wait_ms"] for result in successful]
        ),
        "ttft_from_submit_ms": _stats(
            [result["ttft_from_submit_ms"] for result in successful]
        ),
        "e2e_from_submit_ms": _stats(
            [result["e2e_from_submit_ms"] for result in successful]
        ),
        "ttft_method": "first non-empty content delta",
        "tpot_method": "(e2e_ms - ttft_ms) / (completion_tokens - 1)",
        "itl_method": "content-chunk inter-arrival; assumes one token per chunk",
    }


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------


def build_prompt(args: argparse.Namespace, global_id: int) -> str:
    prompt = args.prompt * args.prompt_repeat
    if args.unique_prompt:
        return f"[request {global_id:06d}] {prompt}"
    return prompt


def _extract_text(delta: dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    reasoning = delta.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else ""


async def fetch(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    *,
    global_id: int,
    submitted_at: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "global_id": global_id,
        "success": False,
        "error": None,
        "prompt": build_prompt(args, global_id),
        "prompt_tokens": None,
        "completion_tokens": None,
        "chunk_count": 0,
        "content_chunk_count": 0,
        "finish_reason": None,
        "client_wait_ms": None,
        "ttft_ms": None,
        "ttft_from_submit_ms": None,
        "e2e_ms": None,
        "e2e_from_submit_ms": None,
        "tpot_ms": None,
        "itl_ms": [],
        "response_text_prefix": None,
    }

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": result["prompt"]}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.include_usage:
        payload["stream_options"] = {"include_usage": True}

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    try:
        async with semaphore:
            send_start = time.perf_counter()
            result["client_wait_ms"] = (send_start - submitted_at) * 1000.0

            async with session.post(
                args.url,
                json=payload,
                headers=headers,
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    result["error"] = (
                        f"HTTP {response.status}: {body[:1000]}"
                    )
                    result["response_text_prefix"] = body[:1000]
                    result["e2e_ms"] = (time.perf_counter() - send_start) * 1000.0
                    result["e2e_from_submit_ms"] = (
                        time.perf_counter() - submitted_at
                    ) * 1000.0
                    return result

                token_times: list[float] = []
                usage: dict[str, Any] = {}

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage = chunk_usage

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    result["chunk_count"] += 1
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None:
                        result["finish_reason"] = finish_reason

                    delta = choice.get("delta") or {}
                    text = _extract_text(delta)
                    if not text:
                        continue

                    now = time.perf_counter()
                    token_times.append(now)
                    result["content_chunk_count"] += 1
                    if result["ttft_ms"] is None:
                        result["ttft_ms"] = (now - send_start) * 1000.0
                        result["ttft_from_submit_ms"] = (
                            now - submitted_at
                        ) * 1000.0

                end_time = time.perf_counter()
                result["e2e_ms"] = (end_time - send_start) * 1000.0
                result["e2e_from_submit_ms"] = (
                    end_time - submitted_at
                ) * 1000.0
                result["success"] = True

                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    if isinstance(prompt_tokens, int):
                        result["prompt_tokens"] = prompt_tokens
                    if isinstance(completion_tokens, int):
                        result["completion_tokens"] = completion_tokens

                if result["completion_tokens"] is None:
                    # Fallback only: one content chunk is assumed to be one token.
                    result["completion_tokens"] = len(token_times)

                if len(token_times) >= 2:
                    intervals = np.diff(np.asarray(token_times)) * 1000.0
                    result["itl_ms"] = [float(value) for value in intervals]

                if (
                    result["ttft_ms"] is not None
                    and result["completion_tokens"] is not None
                    and result["completion_tokens"] > 1
                ):
                    result["tpot_ms"] = (
                        result["e2e_ms"] - result["ttft_ms"]
                    ) / (result["completion_tokens"] - 1)

                return result

    except Exception as error:  # noqa: BLE001 - benchmark must record failures
        result["error"] = f"{type(error).__name__}: {error}"
        if result["e2e_from_submit_ms"] is None:
            result["e2e_from_submit_ms"] = (
                time.perf_counter() - submitted_at
            ) * 1000.0
        return result


async def run_round(
    args: argparse.Namespace,
    *,
    round_index: int,
    num_requests: int,
    phase: str,
) -> tuple[list[dict[str, Any]], float]:
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(
        limit=max(args.concurrency * 2, 20),
        force_close=False,
    )

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
    ) as session:
        start = time.perf_counter()
        if phase == "warmup":
            id_base = 10_000_000
        else:
            id_base = round_index * args.num_requests
        tasks = [
            asyncio.create_task(
                fetch(
                    session,
                    semaphore,
                    args,
                    global_id=id_base + index,
                    submitted_at=start,
                )
            )
            for index in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
        wall_time_s = time.perf_counter() - start

    if phase == "measured":
        print(
            f"  round {round_index + 1}/{args.repeats}: "
            f"{num_requests} requests in {wall_time_s:.2f}s"
        )
    return results, wall_time_s


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_summary(label: str, summary: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}")
    print(label)
    print(f"{'=' * 72}")
    print(
        f"requests: {summary['attempted']} attempted, "
        f"{summary['successful']} successful, {summary['failed']} failed"
    )
    if summary["successful"]:
        print(
            f"wall: {summary['wall_time_s']:.2f}s | "
            f"throughput: {_fmt(summary['req_per_s'])} req/s, "
            f"{_fmt(summary['output_tokens_per_s'])} output tok/s, "
            f"{_fmt(summary['total_tokens_per_s'])} total tok/s"
        )
        print(
            f"tokens: {summary['prompt_tokens_sum']} prompt, "
            f"{summary['completion_tokens_sum']} completion | "
            f"usage available: {summary['usage_available_requests']}/"
            f"{summary['successful']}"
        )

        for key, name in (
            ("ttft_ms", "TTFT"),
            ("e2e_ms", "E2E"),
            ("tpot_ms", "TPOT"),
            ("itl_ms", "ITL"),
            ("client_wait_ms", "client wait"),
            ("ttft_from_submit_ms", "TTFT from submit"),
            ("e2e_from_submit_ms", "E2E from submit"),
        ):
            stats = summary[key]
            print(
                f"{name:>18}: "
                f"mean {_fmt(stats['mean'])} | "
                f"p50 {_fmt(stats['median'])} | "
                f"p90 {_fmt(stats['p90'])} | "
                f"p99 {_fmt(stats['p99'])} "
                f"(n={stats['count']})"
            )
    print(f"{'=' * 72}\n")


def _json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    values.pop("api_key", None)
    return values


def _default_output_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.label).strip("-")
    return Path(args.output_dir) / f"{label}-{timestamp}.json"


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming OpenAI-compatible benchmark with TTFT/TPOT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--label", default="vllm-baseline")
    parser.add_argument("--notes", default="", help="Free-form server/config notes")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sleep-between-rounds", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-repeat", type=int, default=50)
    parser.add_argument(
        "--unique-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a unique request id to avoid prefix-cache reuse.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ask the server to generate exactly max_tokens when supported.",
    )
    parser.add_argument(
        "--include-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request stream_options.include_usage for real token counts.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "benchmarks" / "results"),
        help="Directory for the raw JSON result.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit JSON output path; overrides --output-dir.",
    )
    args = parser.parse_args()

    if args.num_requests <= 0:
        parser.error("--num-requests must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return args


async def async_main(args: argparse.Namespace) -> None:
    environment = collect_environment()
    print("environment:")
    print(json.dumps(environment, ensure_ascii=False, indent=2))
    print(
        f"\nworkload: {args.num_requests} requests, "
        f"concurrency {args.concurrency}, warmup {args.warmup}, "
        f"repeats {args.repeats}, max_tokens {args.max_tokens}"
    )
    if not args.unique_prompt:
        print(
            "WARNING: --no-unique-prompt means every request shares the same "
            "prefix; with prefix caching enabled, TTFT will be biased low."
        )
    if not args.include_usage:
        print(
            "WARNING: --no-include-usage means prompt_tokens and "
            "completion_tokens may be missing or estimated from chunks."
        )

    if args.warmup:
        warmup_results, warmup_wall = await run_round(
            args,
            round_index=-1,
            num_requests=args.warmup,
            phase="warmup",
        )
        warmup_summary = summarize(warmup_results, warmup_wall)
        print_summary("warmup (excluded from measurements)", warmup_summary)
        failures = warmup_summary["failed"]
        if failures:
            print(
                f"WARNING: {failures}/{args.warmup} warmup requests failed; "
                "the measured run may also fail."
            )
        await asyncio.sleep(args.sleep_between_rounds)

    rounds: list[dict[str, Any]] = []
    all_measured_results: list[dict[str, Any]] = []
    measured_wall_time = 0.0

    for round_index in range(args.repeats):
        results, wall_time_s = await run_round(
            args,
            round_index=round_index,
            num_requests=args.num_requests,
            phase="measured",
        )
        summary = summarize(results, wall_time_s)
        rounds.append(
            {
                "round": round_index,
                "summary": summary,
                "requests": results,
            }
        )
        all_measured_results.extend(results)
        measured_wall_time += wall_time_s

        print_summary(f"measured round {round_index + 1}/{args.repeats}", summary)
        if round_index + 1 < args.repeats:
            await asyncio.sleep(args.sleep_between_rounds)

    overall = summarize(all_measured_results, measured_wall_time)
    print_summary("overall (all measured rounds)", overall)

    overshoot = [
        result
        for result in all_measured_results
        if result["completion_tokens"] is not None
        and result["completion_tokens"] > args.max_tokens
    ]
    if overshoot:
        suspicious = overall["suspect_total_tokens_requests"]
        print(
            f"\nWARNING: {len(overshoot)}/{overall['successful']} requests "
            f"report completion_tokens > max_tokens={args.max_tokens}; "
            f"{suspicious} of them look like prompt_tokens + content chunks "
            "(i.e. total_tokens was recorded as completion_tokens). "
            "Use content_chunk_count or a fixed --max-tokens for throughput "
            "until the usage field is understood."
        )

    output_path = (
        Path(args.output)
        if args.output is not None
        else _default_output_path(args)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "schema_version": 1,
        "label": args.label,
        "notes": args.notes,
        "environment": environment,
        "server": {
            "url": args.url,
            "model": args.model,
            "api_key_set": bool(args.api_key),
        },
        "arguments": _json_safe_args(args),
        "measurement": {
            "streaming": True,
            "warmup_requests_excluded": args.warmup,
            "ttft": "first non-empty content delta",
            "tpot": "(e2e_ms - ttft_ms) / (completion_tokens - 1)",
            "itl": "content-chunk inter-arrival; assumes one token per chunk",
        },
        "rounds": rounds,
        "summary": overall,
    }
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved raw results to: {output_path.resolve()}")


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
