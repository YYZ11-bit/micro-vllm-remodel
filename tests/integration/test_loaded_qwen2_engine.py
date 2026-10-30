"""End-to-end Engine semantics for a loaded HF-format Qwen2 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from micro_vllm.config import CacheConfig, LLMConfig, ModelConfig, SchedulerConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.request import FinishReason
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.engine.engine import Engine
from micro_vllm.engine.output import RequestOutput
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.executor.single_process import SingleProcessExecutor
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.worker import Worker


class RecordingSingleProcessExecutor(SingleProcessExecutor):
    """Record execution plans for chunking and preemption assertions."""

    __slots__ = ("plans",)

    def __init__(self, worker: Worker) -> None:
        super().__init__(worker)
        self.plans: list[SchedulerOutput] = []

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        output = super().execute(plan)
        self.plans.append(plan)
        return output


@dataclass
class LoadedQwen2Harness:
    engine: Engine
    scheduler: Scheduler
    cache_manager: KVCacheManager
    worker: Worker
    executor: RecordingSingleProcessExecutor


def make_loaded_engine(
    checkpoint_path: Path,
    *,
    token_budget: int = 8,
    sequence_budget: int = 2,
    block_size: int = 2,
    num_blocks: int = 16,
    max_model_len: int = 32,
    enable_prefix_cache: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LoadedQwen2Harness:
    config = LLMConfig(
        model_config=ModelConfig(
            model=checkpoint_path,
            device=device,
            dtype=dtype,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=token_budget,
            max_num_seqs=sequence_budget,
            max_model_len=max_model_len,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            num_blocks=num_blocks + 1,
            enable_prefix_cache=enable_prefix_cache,
        ),
    )
    worker = Worker(config)
    cache_manager = KVCacheManager(config)
    scheduler = Scheduler(config, cache_manager)
    executor = RecordingSingleProcessExecutor(worker)
    return LoadedQwen2Harness(
        engine=Engine(scheduler, executor),
        scheduler=scheduler,
        cache_manager=cache_manager,
        worker=worker,
        executor=executor,
    )


def run_to_completion(
    harness: LoadedQwen2Harness,
) -> dict[str, RequestOutput]:
    latest_outputs: dict[str, RequestOutput] = {}
    for _ in range(64):
        if not harness.engine.has_unfinished_requests():
            return latest_outputs
        for output in harness.engine.step():
            latest_outputs[output.request_id] = output
    raise AssertionError("loaded Qwen2 engine did not finish within 64 steps")


def request_output_tokens(
    harness: LoadedQwen2Harness,
    request_id: str,
) -> tuple[int, ...]:
    request = harness.scheduler.get_request(request_id)
    assert request is not None
    return tuple(request.output_token_ids)


@torch.no_grad()
def reference_greedy_tokens(
    reference_model: Qwen2ForCausalLM,
    prompt_token_ids: list[int],
    *,
    max_tokens: int,
) -> tuple[int, ...]:
    sequence = list(prompt_token_ids)
    generated: list[int] = []
    device = next(reference_model.parameters()).device
    for _ in range(max_tokens):
        input_ids = torch.tensor([sequence], dtype=torch.int64, device=device)
        logits = reference_model(input_ids, use_cache=False).logits[0, -1]
        token_id = int(logits.argmax(dim=-1).item())
        generated.append(token_id)
        sequence.append(token_id)
    return tuple(generated)


def test_Qwen2_full_chunked_prefill_and_decode_match_transformers(
    tiny_hf_qwen2_checkpoint: tuple[Qwen2ForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_qwen2_checkpoint
    prompt = [1, 3, 5, 7, 9]
    expected = reference_greedy_tokens(reference_model, prompt, max_tokens=4)
    full = make_loaded_engine(
        checkpoint_path,
        token_budget=8,
        sequence_budget=1,
    )
    chunked = make_loaded_engine(
        checkpoint_path,
        token_budget=2,
        sequence_budget=1,
    )
    for harness in (full, chunked):
        harness.engine.add_request(
            "request",
            prompt,
            SamplingParams(max_tokens=4),
        )
        run_to_completion(harness)

    assert request_output_tokens(full, "request") == expected
    assert request_output_tokens(chunked, "request") == expected
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in chunked.executor.plans[:3]
    ] == [2, 2, 1]


def test_Qwen2_mixed_batch_and_prefix_cache_preserve_tokens(
    tiny_hf_qwen2_checkpoint: tuple[Qwen2ForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_qwen2_checkpoint
    mixed = make_loaded_engine(
        checkpoint_path,
        token_budget=4,
        sequence_budget=2,
    )
    mixed.engine.add_request("decode", [1], SamplingParams(max_tokens=2))
    mixed.engine.step()
    mixed.engine.add_request(
        "prefill",
        [3, 4, 5],
        SamplingParams(max_tokens=1),
    )
    run_to_completion(mixed)

    assert request_output_tokens(mixed, "decode") == reference_greedy_tokens(
        reference_model,
        [1],
        max_tokens=2,
    )
    assert request_output_tokens(mixed, "prefill") == reference_greedy_tokens(
        reference_model,
        [3, 4, 5],
        max_tokens=1,
    )
    assert [
        request.request_id
        for request in mixed.executor.plans[1].scheduled_requests
    ] == ["decode", "prefill"]

    prompt = [1, 2, 3, 4, 5]
    prefix = make_loaded_engine(checkpoint_path, enable_prefix_cache=True)
    for request_id in ("first", "second"):
        prefix.engine.add_request(
            request_id,
            prompt,
            SamplingParams(max_tokens=1),
        )
        run_to_completion(prefix)
    first = prefix.executor.plans[0].scheduled_requests[0]
    second = prefix.executor.plans[1].scheduled_requests[0]
    assert second.num_computed_tokens == 4
    assert (
        second.block_tables[0].block_ids[:2]
        == first.block_tables[0].block_ids[:2]
    )
    expected = reference_greedy_tokens(reference_model, prompt, max_tokens=1)
    assert request_output_tokens(prefix, "first") == expected
    assert request_output_tokens(prefix, "second") == expected


def test_Qwen2_preemption_recompute_matches_unconstrained_execution(
    tiny_hf_qwen2_checkpoint: tuple[Qwen2ForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_qwen2_checkpoint
    prompts = {"first": [1, 2], "second": [3, 4]}
    preempted = make_loaded_engine(
        checkpoint_path,
        token_budget=4,
        sequence_budget=2,
        num_blocks=2,
    )
    baseline = make_loaded_engine(
        checkpoint_path,
        token_budget=4,
        sequence_budget=2,
        num_blocks=4,
    )
    for harness in (preempted, baseline):
        for request_id, prompt in prompts.items():
            harness.engine.add_request(
                request_id,
                prompt,
                SamplingParams(max_tokens=2),
            )
        run_to_completion(harness)

    assert preempted.executor.plans[1].preempted_request_ids == ("second",)
    resumed = preempted.executor.plans[2].scheduled_requests[0]
    assert resumed.request_id == "second"
    assert resumed.num_computed_tokens == 0
    for request_id, prompt in prompts.items():
        expected = reference_greedy_tokens(
            reference_model,
            prompt,
            max_tokens=2,
        )
        assert request_output_tokens(preempted, request_id) == expected
        assert request_output_tokens(baseline, request_id) == expected


def test_Qwen2_reused_block_and_request_termination_release_cache(
    tiny_hf_qwen2_checkpoint: tuple[Qwen2ForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_qwen2_checkpoint
    reused = make_loaded_engine(
        checkpoint_path,
        token_budget=4,
        sequence_budget=1,
        block_size=4,
        num_blocks=1,
    )
    reused.engine.add_request("old", [1, 2, 3], SamplingParams(max_tokens=1))
    run_to_completion(reused)
    assert reused.cache_manager.num_free_blocks == 1
    reused.engine.add_request("new", [7], SamplingParams(max_tokens=1))
    output = run_to_completion(reused)["new"]

    assert output.output_token_ids == reference_greedy_tokens(
        reference_model,
        [7],
        max_tokens=1,
    )
    assert output.finish_reason is FinishReason.LENGTH
    assert (
        reused.executor.plans[0]
        .scheduled_requests[0]
        .block_tables[0]
        .block_ids
        == (0,)
    )
    assert (
        reused.executor.plans[1]
        .scheduled_requests[0]
        .block_tables[0]
        .block_ids
        == (0,)
    )
    assert reused.cache_manager.num_free_blocks == 1

    aborting = make_loaded_engine(
        checkpoint_path,
        token_budget=2,
        sequence_budget=1,
        block_size=2,
        num_blocks=2,
    )
    aborting.engine.add_request(
        "aborted",
        [1, 2, 3],
        SamplingParams(max_tokens=1),
    )
    assert aborting.engine.step() == ()
    assert aborting.cache_manager.num_free_blocks == 1
    assert aborting.engine.abort_request("aborted") is True
    assert aborting.cache_manager.num_free_blocks == 2


def test_Qwen2_eos_and_model_length_termination(
    tiny_hf_qwen2_checkpoint: tuple[Qwen2ForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_qwen2_checkpoint
    prompt = [5]
    first_token = reference_greedy_tokens(
        reference_model,
        prompt,
        max_tokens=1,
    )[0]
    eos = make_loaded_engine(checkpoint_path)
    eos.engine.add_request(
        "eos",
        prompt,
        SamplingParams(max_tokens=4, eos_token_id=first_token),
    )
    eos_output = run_to_completion(eos)["eos"]

    model_length = make_loaded_engine(checkpoint_path, max_model_len=2)
    model_length.engine.add_request(
        "model-length",
        prompt,
        SamplingParams(max_tokens=4),
    )
    model_length_output = run_to_completion(model_length)["model-length"]

    assert eos_output.output_token_ids == (first_token,)
    assert eos_output.finish_reason is FinishReason.EOS
    assert model_length_output.finish_reason is FinishReason.MODEL_LENGTH


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_loaded_Qwen2_runs_through_cuda_engine_fp16(
    tiny_hf_qwen2_checkpoint: tuple[Qwen2ForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_qwen2_checkpoint
    reference_model.to(device="cuda", dtype=torch.float16)
    prompt = [1, 2, 3]
    harness = make_loaded_engine(
        checkpoint_path,
        sequence_budget=1,
        device="cuda",
        dtype=torch.float16,
    )
    harness.engine.add_request(
        "cuda-loaded",
        prompt,
        SamplingParams(max_tokens=2),
    )

    output = run_to_completion(harness)["cuda-loaded"]
    model_runner = harness.worker._model_runner
    assert model_runner is not None
    kv_cache_buffers = model_runner._kv_cache_buffers
    assert kv_cache_buffers is not None
    assert output.output_token_ids == reference_greedy_tokens(
        reference_model,
        prompt,
        max_tokens=2,
    )
    assert kv_cache_buffers[0].shape[-1] == 4
    assert kv_cache_buffers[0].device.type == "cuda"
    assert kv_cache_buffers[0].dtype is torch.float16

@pytest.mark.model
@torch.no_grad()
def test_real_qwen2_engine_greedy_matches_hf() -> None:
    path = os.getenv("MICRO_VLLM_QWEN2_MODEL_PATH")
    if path is None:
        pytest.skip("MICRO_VLLM_QWEN2_MODEL_PATH is not set")
    model_path = Path(path).expanduser()
    assert model_path.is_dir()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = tokenizer("The capital of France is", add_special_tokens=False)["input_ids"]

    reference = Qwen2ForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="eager",
    ).eval()
    expected: list[int] = []
    sequence = list(prompt)
    for _ in range(4):
        logits = reference(torch.tensor([sequence])).logits[0, -1]
        token_id = int(logits.argmax())
        expected.append(token_id)
        sequence.append(token_id)
    del reference                     # 这里才是最后一个引用
    gc.collect()

    harness = make_loaded_engine(model_path, sequence_budget=1)
    harness.engine.add_request("request", prompt, SamplingParams(max_tokens=4))
    run_to_completion(harness)
    request = harness.scheduler.get_request("request")
    assert tuple(request.output_token_ids) == tuple(expected)
    assert request.finish_reason is FinishReason.LENGTH
