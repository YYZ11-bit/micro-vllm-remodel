"""Tests for speculative scheduling, acceptance, and KV rollback."""

from dataclasses import dataclass

import pytest

from micro_vllm.config import (
    CacheConfig,
    LLMConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.request import FinishReason, Request, RequestStatus
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.executor.interface import (
    ModelRunnerOutput,
    RequestVerification,
    SampledToken,
)
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.speculative.drafter import DraftOutput, Drafter

REQUEST_ID = "request-1"


@dataclass
class FixedDrafter:
    token_ids: tuple[int, ...]
    calls: list[tuple[tuple[int, ...], int]]

    def __init__(self, token_ids: tuple[int, ...]) -> None:
        self.token_ids = token_ids
        self.calls = []

    def propose(
        self,
        token_ids: tuple[int, ...],
        max_tokens: int,
    ) -> DraftOutput:
        self.calls.append((tuple(token_ids), max_tokens))
        return DraftOutput(tokens_ids=self.token_ids[:max_tokens])


@dataclass
class MappingDrafter:
    """Return different drafts based on the last known token."""

    proposals: dict[int, tuple[int, ...]]
    calls: list[tuple[tuple[int, ...], int]]

    def __init__(self, proposals: dict[int, tuple[int, ...]]) -> None:
        self.proposals = proposals
        self.calls = []

    def propose(
        self,
        token_ids: tuple[int, ...],
        max_tokens: int,
    ) -> DraftOutput:
        self.calls.append((tuple(token_ids), max_tokens))
        return DraftOutput(
            tokens_ids=self.proposals.get(token_ids[-1], ())[:max_tokens]
        )


def make_scheduler(
    drafter: Drafter | None,
    *,
    max_draft_tokens: int = 4,
    token_budget: int = 16,
    max_num_seqs: int = 2,
    max_model_len: int = 128,
    block_size: int = 4,
    num_data_blocks: int = 16,
    enable_prefix_cache: bool = False,
) -> tuple[Scheduler, KVCacheManager]:
    config = LLMConfig(
        model_config=ModelConfig(model="unused"),
        scheduler_config=SchedulerConfig(
            token_budget,
            max_num_seqs,
            max_model_len,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            num_blocks=num_data_blocks + 1,
            enable_prefix_cache=enable_prefix_cache,
        ),
        speculative_config=SpeculativeConfig(
            enabled=drafter is not None,
            max_draft_tokens=max_draft_tokens,
        ),
    )
    manager = KVCacheManager(config, cache_namespace="spec")
    return Scheduler(config, manager, drafter), manager


def make_request(
    request_id: str = REQUEST_ID,
    *,
    prompt: tuple[int, ...] = (1, 2, 3),
    max_tokens: int = 8,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
) -> Request:
    return Request(
        request_id,
        list(prompt),
        SamplingParams(
            max_tokens=max_tokens,
            eos_token_id=eos_token_id,
            temperature=temperature,
        ),
    )


def prefill(
    scheduler: Scheduler,
    request: Request,
    token_id: int = 7,
) -> None:
    plan = scheduler.schedule()
    scheduled = plan.scheduled_requests[0]
    assert scheduled.requires_sampling is True
    scheduler.update(
        plan,
        ModelRunnerOutput((SampledToken(request.request_id, token_id),)),
    )


def prepare_spec_step(
    *,
    drafter_tokens: tuple[int, ...] = (10, 11),
    prompt: tuple[int, ...] = (1, 2, 3),
    max_tokens: int = 8,
    eos_token_id: int | None = None,
    block_size: int = 4,
    num_data_blocks: int = 16,
    enable_prefix_cache: bool = False,
) -> tuple[Scheduler, KVCacheManager, Request, SchedulerOutput]:
    drafter = FixedDrafter(drafter_tokens)
    scheduler, manager = make_scheduler(
        drafter,
        block_size=block_size,
        num_data_blocks=num_data_blocks,
        enable_prefix_cache=enable_prefix_cache,
    )
    request = make_request(
        prompt=prompt,
        max_tokens=max_tokens,
        eos_token_id=eos_token_id,
    )
    scheduler.add_request(request)
    prefill(scheduler, request)
    plan = scheduler.schedule()
    assert plan.scheduled_requests[0].num_draft_tokens == len(drafter_tokens)
    return scheduler, manager, request, plan


def test_proposal_schedules_pending_token_and_all_drafts() -> None:
    drafter = FixedDrafter((10, 11))
    scheduler, _ = make_scheduler(drafter)
    request = make_request()
    scheduler.add_request(request)
    prefill(scheduler, request)

    plan = scheduler.schedule()
    scheduled = plan.scheduled_requests[0]

    assert scheduled.input_token_ids == (7, 10, 11)
    assert scheduled.num_scheduled_tokens == 3
    assert scheduled.num_draft_tokens == 2
    assert scheduled.requires_sampling is True
    assert request.draft_token_ids == [10, 11]
    assert drafter.calls == [((1, 2, 3, 7), 4)]


@pytest.mark.parametrize("max_draft_tokens", [1, 2])
def test_max_draft_tokens_caps_proposal(max_draft_tokens: int) -> None:
    drafter = FixedDrafter((10, 11, 12, 13))
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=max_draft_tokens,
    )
    request = make_request()
    scheduler.add_request(request)
    prefill(scheduler, request)

    scheduler.schedule()

    assert request.draft_token_ids == list(
        drafter.token_ids[:max_draft_tokens]
    )
    assert drafter.calls == [((1, 2, 3, 7), max_draft_tokens)]


def test_remaining_token_budget_caps_proposal() -> None:
    drafter = FixedDrafter((10, 11, 12, 13))
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=4,
        token_budget=3,
    )
    request = make_request()
    scheduler.add_request(request)
    prefill(scheduler, request)

    plan = scheduler.schedule()

    assert request.draft_token_ids == [10, 11]
    assert plan.scheduled_requests[0].num_scheduled_tokens == 3
    assert drafter.calls == [((1, 2, 3, 7), 2)]


def test_no_proposal_when_budget_only_fits_pending_token() -> None:
    drafter = FixedDrafter((10, 11))
    scheduler, _ = make_scheduler(drafter, token_budget=1)
    request = make_request(prompt=(1,))
    scheduler.add_request(request)
    prefill(scheduler, request)

    plan = scheduler.schedule()
    scheduled = plan.scheduled_requests[0]

    assert request.draft_token_ids == []
    assert scheduled.input_token_ids == (7,)
    assert scheduled.num_draft_tokens == 0
    assert drafter.calls == []


def test_no_proposal_for_non_greedy_request() -> None:
    drafter = FixedDrafter((10, 11))
    scheduler, _ = make_scheduler(drafter)
    request = make_request(temperature=1.0)
    scheduler.add_request(request)
    prefill(scheduler, request)

    scheduler.schedule()

    assert request.draft_token_ids == []
    assert drafter.calls == []


def test_no_proposal_while_prefill_still_has_pending_tokens() -> None:
    drafter = FixedDrafter((10, 11))
    scheduler, _ = make_scheduler(drafter, token_budget=4)
    request = make_request(prompt=tuple(range(10)))
    scheduler.add_request(request)
    first_plan = scheduler.schedule()
    scheduler.update(first_plan, ModelRunnerOutput())

    second_plan = scheduler.schedule()

    assert second_plan.scheduled_requests[0].num_draft_tokens == 0
    assert drafter.calls == []


def test_all_drafts_accepted_appends_bonus_and_commits_prefix() -> None:
    scheduler, manager, request, plan = prepare_spec_step(
        drafter_tokens=(10, 11)
    )

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, (10, 11, 12)),
            )
        ),
    )

    assert request.output_token_ids == [7, 10, 11, 12]
    assert request.draft_token_ids == []
    assert request.num_computed_tokens == 6
    assert request.finish_reason is None
    assert len(manager.get_block_table(REQUEST_ID)) == 2


def test_first_draft_rejected_keeps_only_replacement_and_truncates_kv() -> None:
    scheduler, manager, request, plan = prepare_spec_step(
        drafter_tokens=(10, 11)
    )

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, (9, 11, 12)),
            )
        ),
    )

    assert request.output_token_ids == [7, 9]
    assert request.draft_token_ids == []
    assert request.num_computed_tokens == 4
    assert request.finish_reason is None
    assert len(manager.get_block_table(REQUEST_ID)) == 1


def test_mid_rejection_commits_accepted_prefix_and_replacement() -> None:
    scheduler, manager, request, plan = prepare_spec_step(
        drafter_tokens=(10, 11, 12)
    )

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, (10, 9, 12, 13)),
            )
        ),
    )

    assert request.output_token_ids == [7, 10, 9]
    assert request.draft_token_ids == []
    assert request.num_computed_tokens == 5
    assert request.finish_reason is None
    assert len(manager.get_block_table(REQUEST_ID)) == 2


def test_eos_in_accepted_drafts_stops_before_bonus() -> None:
    scheduler, _, request, plan = prepare_spec_step(
        drafter_tokens=(10, 11),
        eos_token_id=11,
    )

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, (10, 11, 12)),
            )
        ),
    )

    assert request.output_token_ids == [7, 10, 11]
    assert request.draft_token_ids == []
    assert request.num_computed_tokens == 6
    assert request.finish_reason is FinishReason.EOS


def test_eos_in_replacement_finishes_without_bonus() -> None:
    scheduler, _, request, plan = prepare_spec_step(
        drafter_tokens=(10, 11),
        eos_token_id=9,
    )

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, (9, 11, 12)),
            )
        ),
    )

    assert request.output_token_ids == [7, 9]
    assert request.draft_token_ids == []
    assert request.num_computed_tokens == 4
    assert request.finish_reason is FinishReason.EOS


def test_eos_in_bonus_finishes_after_all_drafts() -> None:
    scheduler, _, request, plan = prepare_spec_step(
        drafter_tokens=(10, 11),
        eos_token_id=12,
    )

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, (10, 11, 12)),
            )
        ),
    )

    assert request.output_token_ids == [7, 10, 11, 12]
    assert request.draft_token_ids == []
    assert request.num_computed_tokens == 6
    assert request.finish_reason is FinishReason.EOS


def test_max_tokens_stops_inside_accepted_drafts() -> None:
    scheduler, _, request, plan = prepare_spec_step(
        drafter_tokens=(10, 11),
        max_tokens=3,
    )

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, (10, 11, 12)),
            )
        ),
    )

    assert request.output_token_ids == [7, 10, 11]
    assert request.draft_token_ids == []
    assert request.num_computed_tokens == 6
    assert request.finish_reason is FinishReason.LENGTH


def test_rejected_draft_is_not_committed_to_prefix_cache() -> None:
    scheduler, manager, request, plan = prepare_spec_step(
        drafter_tokens=(10, 11, 12),
        prompt=(1, 2, 3, 4),
        enable_prefix_cache=True,
    )

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, (10, 11, 9, 13)),
            )
        ),
    )

    assert request.output_token_ids == [7, 10, 11, 9]
    assert request.draft_token_ids == []
    assert request.num_computed_tokens == 7
    assert len(manager.get_block_table(REQUEST_ID)) == 2
    assert manager._block_pool.num_cached_blocks == 1


def test_speculative_plan_requires_verification_result() -> None:
    scheduler, _, _, plan = prepare_spec_step(drafter_tokens=(10, 11))

    with pytest.raises(ValueError, match="verification"):
        scheduler.update(plan, ModelRunnerOutput())


def test_non_speculative_plan_rejects_verification_result() -> None:
    scheduler, _ = make_scheduler(None)
    request = make_request()
    scheduler.add_request(request)
    plan = scheduler.schedule()
    assert plan.scheduled_requests[0].num_draft_tokens == 0

    with pytest.raises(ValueError, match="verification"):
        scheduler.update(
            plan,
            ModelRunnerOutput(
                verifications=(
                    RequestVerification(REQUEST_ID, (7,)),
                )
            ),
        )


def test_mixed_batch_updates_two_spec_requests_and_one_normal_decode() -> None:
    drafter = MappingDrafter({10: (11, 12), 20: (21, 22)})
    scheduler, manager = make_scheduler(
        drafter,
        max_draft_tokens=2,
        token_budget=32,
        max_num_seqs=3,
    )
    first = make_request("first", prompt=(1, 2, 3))
    second = make_request("second", prompt=(4, 5, 6))
    normal = make_request("normal", prompt=(7, 8, 9))
    for request in (first, second, normal):
        scheduler.add_request(request)

    prefill_plan = scheduler.schedule()
    assert [
        request.request_id for request in prefill_plan.scheduled_requests
    ] == ["first", "second", "normal"]
    scheduler.update(
        prefill_plan,
        ModelRunnerOutput(
            (
                SampledToken("first", 10),
                SampledToken("second", 20),
                SampledToken("normal", 30),
            )
        ),
    )

    plan = scheduler.schedule()
    scheduled = {
        request.request_id: request for request in plan.scheduled_requests
    }
    assert scheduled["first"].input_token_ids == (10, 11, 12)
    assert scheduled["second"].input_token_ids == (20, 21, 22)
    assert scheduled["normal"].input_token_ids == (30,)
    assert scheduled["first"].num_draft_tokens == 2
    assert scheduled["second"].num_draft_tokens == 2
    assert scheduled["normal"].num_draft_tokens == 0

    scheduler.update(
        plan,
        ModelRunnerOutput(
            sampled_tokens=(SampledToken("normal", 31),),
            verifications=(
                RequestVerification("first", (11, 12, 13)),
                RequestVerification("second", (9, 22, 23)),
            ),
        ),
    )

    assert first.output_token_ids == [10, 11, 12, 13]
    assert second.output_token_ids == [20, 9]
    assert normal.output_token_ids == [30, 31]
    assert first.draft_token_ids == []
    assert second.draft_token_ids == []
    assert normal.draft_token_ids == []
    assert first.num_computed_tokens == 6
    assert second.num_computed_tokens == 4
    assert normal.num_computed_tokens == 4
    assert len(manager.get_block_table("first")) == 2
    assert len(manager.get_block_table("second")) == 1
    assert len(manager.get_block_table("normal")) == 1


def test_mixed_batch_supports_different_draft_lengths_per_request() -> None:
    drafter = MappingDrafter({10: (11,), 20: (21, 22, 23)})
    scheduler, manager = make_scheduler(
        drafter,
        max_draft_tokens=3,
        token_budget=32,
        max_num_seqs=2,
    )
    first = make_request("first", prompt=(1, 2, 3))
    second = make_request("second", prompt=(4, 5, 6))
    scheduler.add_request(first)
    scheduler.add_request(second)

    prefill_plan = scheduler.schedule()
    scheduler.update(
        prefill_plan,
        ModelRunnerOutput(
            (
                SampledToken("first", 10),
                SampledToken("second", 20),
            )
        ),
    )

    plan = scheduler.schedule()
    scheduled = {
        request.request_id: request for request in plan.scheduled_requests
    }
    assert scheduled["first"].input_token_ids == (10, 11)
    assert scheduled["first"].num_draft_tokens == 1
    assert scheduled["second"].input_token_ids == (20, 21, 22, 23)
    assert scheduled["second"].num_draft_tokens == 3

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification("first", (11, 12)),
                RequestVerification("second", (9, 22, 23, 24)),
            )
        ),
    )

    assert first.output_token_ids == [10, 11, 12]
    assert second.output_token_ids == [20, 9]
    assert first.num_computed_tokens == 5
    assert second.num_computed_tokens == 4
    assert len(manager.get_block_table("first")) == 2
    assert len(manager.get_block_table("second")) == 1


def test_speculative_decode_and_waiting_prefill_share_one_plan() -> None:
    drafter = MappingDrafter({10: (11, 12)})
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=2,
        token_budget=6,
        max_num_seqs=2,
    )
    first = make_request("first", prompt=(1, 2, 3))
    scheduler.add_request(first)
    prefill_first = scheduler.schedule()
    scheduler.update(
        prefill_first,
        ModelRunnerOutput((SampledToken("first", 10),)),
    )

    second = make_request("second", prompt=tuple(range(10, 18)))
    scheduler.add_request(second)
    plan = scheduler.schedule()

    assert [
        request.request_id for request in plan.scheduled_requests
    ] == ["first", "second"]
    first_scheduled, second_scheduled = plan.scheduled_requests
    assert first_scheduled.num_draft_tokens == 2
    assert first_scheduled.num_scheduled_tokens == 3
    assert second_scheduled.num_draft_tokens == 0
    assert second_scheduled.requires_sampling is False
    assert second_scheduled.num_scheduled_tokens == 3

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification("first", (11, 12, 13)),
            )
        ),
    )

    assert first.output_token_ids == [10, 11, 12, 13]
    assert first.num_computed_tokens == 6
    assert second.output_token_ids == []
    assert second.num_computed_tokens == 3
    assert second.status is RequestStatus.RUNNING


def test_committed_speculative_prefix_is_reused_by_waiting_request() -> None:
    drafter = MappingDrafter({10: (11, 12)})
    scheduler, manager = make_scheduler(
        drafter,
        max_draft_tokens=2,
        token_budget=8,
        max_num_seqs=2,
        block_size=4,
        num_data_blocks=8,
        enable_prefix_cache=True,
    )
    first = make_request("first", prompt=(1, 2, 3, 4))
    scheduler.add_request(first)
    prefill_plan = scheduler.schedule()
    scheduler.update(
        prefill_plan,
        ModelRunnerOutput((SampledToken("first", 10),)),
    )

    spec_plan = scheduler.schedule()
    scheduler.update(
        spec_plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification("first", (11, 12, 13)),
            )
        ),
    )
    assert first.output_token_ids == [10, 11, 12, 13]
    assert first.num_computed_tokens == 7

    second = make_request("second", prompt=(1, 2, 3, 4, 5), max_tokens=4)
    scheduler.add_request(second)
    plan = scheduler.schedule()
    scheduled = {
        request.request_id: request for request in plan.scheduled_requests
    }

    assert scheduled["second"].input_token_ids == (5,)
    assert scheduled["second"].num_computed_tokens == 4
    assert scheduled["second"].num_scheduled_tokens == 1
    assert scheduled["second"].requires_sampling is True

    scheduler.update(
        plan,
        ModelRunnerOutput(
            (
                SampledToken("first", 14),
                SampledToken("second", 6),
            )
        ),
    )

    first_blocks = manager.get_block_table("first")
    second_blocks = manager.get_block_table("second")
    assert first.output_token_ids == [10, 11, 12, 13, 14]
    assert first.num_computed_tokens == 8
    assert second.output_token_ids == [6]
    assert second.num_computed_tokens == 5
    assert first_blocks[0] == second_blocks[0]
    assert len(first_blocks) == 2
    assert len(second_blocks) == 2
    assert first_blocks[1] != second_blocks[1]


def run_spec_step(
    scheduler: Scheduler,
    predictions: tuple[int, ...],
) -> None:
    plan = scheduler.schedule()
    scheduled = plan.scheduled_requests[0]
    assert scheduled.num_draft_tokens > 0
    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, predictions),
            )
        ),
    )


def run_normal_step(
    scheduler: Scheduler,
    token_id: int = 50,
) -> None:
    plan = scheduler.schedule()
    scheduled = plan.scheduled_requests[0]
    assert scheduled.num_draft_tokens == 0
    scheduler.update(
        plan,
        ModelRunnerOutput((SampledToken(REQUEST_ID, token_id),)),
    )


def test_adaptive_disables_low_acceptance_after_window() -> None:
    drafter = FixedDrafter((10, 11, 12, 13))
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=4,
        token_budget=32,
        max_num_seqs=1,
        num_data_blocks=64,
    )
    request = make_request(max_tokens=100)
    scheduler.add_request(request)
    prefill(scheduler, request)

    # 4 verified drafts per step; two steps fill the 8-draft window.
    run_spec_step(scheduler, (9, 9, 9, 9, 9))
    run_spec_step(scheduler, (9, 9, 9, 9, 9))

    state = scheduler._speculation_states[REQUEST_ID]
    assert state.disabled is True
    assert state.cooldown == 8

    calls_before = len(drafter.calls)
    plan = scheduler.schedule()

    assert plan.scheduled_requests[0].num_draft_tokens == 0
    assert len(drafter.calls) == calls_before


def test_adaptive_keeps_high_acceptance_enabled() -> None:
    drafter = FixedDrafter((10, 11, 12, 13))
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=4,
        token_budget=32,
        max_num_seqs=1,
        num_data_blocks=64,
    )
    request = make_request(max_tokens=100)
    scheduler.add_request(request)
    prefill(scheduler, request)

    run_spec_step(scheduler, (10, 11, 12, 13, 14))
    run_spec_step(scheduler, (10, 11, 12, 13, 14))

    state = scheduler._speculation_states[REQUEST_ID]
    assert state.disabled is False

    calls_before = len(drafter.calls)
    plan = scheduler.schedule()

    assert plan.scheduled_requests[0].num_draft_tokens == 4
    assert len(drafter.calls) == calls_before + 1


def test_adaptive_probe_after_cooldown_rejects_empty_probe() -> None:
    drafter = FixedDrafter((10, 11, 12, 13))
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=4,
        token_budget=32,
        max_num_seqs=1,
        num_data_blocks=64,
    )
    request = make_request(max_tokens=100)
    scheduler.add_request(request)
    prefill(scheduler, request)

    run_spec_step(scheduler, (9, 9, 9, 9, 9))
    run_spec_step(scheduler, (9, 9, 9, 9, 9))
    state = scheduler._speculation_states[REQUEST_ID]
    assert state.disabled is True
    assert state.cooldown == 8

    for _ in range(8):
        run_normal_step(scheduler)
    assert state.cooldown == 0
    assert state.probe is False

    calls_before = len(drafter.calls)
    plan = scheduler.schedule()
    assert plan.scheduled_requests[0].num_draft_tokens == 4
    assert len(drafter.calls) == calls_before + 1
    assert state.probe is True

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, (9, 9, 9, 9, 9)),
            )
        ),
    )

    assert state.disabled is True
    assert state.probe is False
    assert state.backoff == 16
    assert state.cooldown == 16


def test_adaptive_probe_with_empty_proposal_backs_off() -> None:
    drafter = MappingDrafter({7: (10, 11, 12, 13), 9: (10, 11, 12, 13)})
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=4,
        token_budget=32,
        max_num_seqs=1,
        num_data_blocks=64,
    )
    request = make_request(max_tokens=100)
    scheduler.add_request(request)
    prefill(scheduler, request)

    run_spec_step(scheduler, (9, 9, 9, 9, 9))
    run_spec_step(scheduler, (9, 9, 9, 9, 9))
    state = scheduler._speculation_states[REQUEST_ID]
    assert state.disabled is True

    for _ in range(8):
        run_normal_step(scheduler)
    assert state.cooldown == 0

    calls_before = len(drafter.calls)
    plan = scheduler.schedule()

    assert plan.scheduled_requests[0].num_draft_tokens == 0
    assert len(drafter.calls) == calls_before + 1
    assert state.disabled is True
    assert state.probe is False
    assert state.backoff == 16
    assert state.cooldown == 16


def test_adaptive_probe_after_cooldown_can_reenable() -> None:
    drafter = FixedDrafter((10, 11, 12, 13))
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=4,
        token_budget=32,
        max_num_seqs=1,
        num_data_blocks=64,
    )
    request = make_request(max_tokens=100)
    scheduler.add_request(request)
    prefill(scheduler, request)

    run_spec_step(scheduler, (9, 9, 9, 9, 9))
    run_spec_step(scheduler, (9, 9, 9, 9, 9))
    state = scheduler._speculation_states[REQUEST_ID]
    assert state.disabled is True

    for _ in range(8):
        run_normal_step(scheduler)

    plan = scheduler.schedule()
    assert plan.scheduled_requests[0].num_draft_tokens == 4
    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification(REQUEST_ID, (10, 11, 12, 13, 14)),
            )
        ),
    )

    assert state.disabled is False
    assert state.cooldown == 0
    assert state.probe is False


def test_adaptive_state_is_per_request_in_mixed_batch() -> None:
    drafter = MappingDrafter(
        {
            10: (11, 12, 13, 14),
            9: (11, 12, 13, 14),
            20: (21, 22, 23, 24),
            25: (26, 27, 28, 29),
            30: (31, 32, 33, 34),
        }
    )
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=4,
        token_budget=32,
        max_num_seqs=2,
        num_data_blocks=64,
    )
    low = make_request("low", prompt=(1, 2, 3), max_tokens=100)
    high = make_request("high", prompt=(4, 5, 6), max_tokens=100)
    scheduler.add_request(low)
    scheduler.add_request(high)

    prefill_plan = scheduler.schedule()
    scheduler.update(
        prefill_plan,
        ModelRunnerOutput(
            (
                SampledToken("low", 10),
                SampledToken("high", 20),
            )
        ),
    )

    first_spec = scheduler.schedule()
    assert all(
        request.num_draft_tokens == 4
        for request in first_spec.scheduled_requests
    )
    scheduler.update(
        first_spec,
        ModelRunnerOutput(
            verifications=(
                RequestVerification("low", (9, 9, 9, 9, 9)),
                RequestVerification("high", (21, 22, 23, 24, 25)),
            )
        ),
    )

    second_spec = scheduler.schedule()
    scheduler.update(
        second_spec,
        ModelRunnerOutput(
            verifications=(
                RequestVerification("low", (9, 9, 9, 9, 9)),
                RequestVerification("high", (26, 27, 28, 29, 30)),
            )
        ),
    )

    low_state = scheduler._speculation_states["low"]
    high_state = scheduler._speculation_states["high"]
    assert low_state.disabled is True
    assert high_state.disabled is False

    calls_before = len(drafter.calls)
    plan = scheduler.schedule()
    scheduled = {
        request.request_id: request for request in plan.scheduled_requests
    }

    assert scheduled["low"].num_draft_tokens == 0
    assert scheduled["high"].num_draft_tokens == 4
    assert scheduled["high"].input_token_ids == (30, 31, 32, 33, 34)
    assert len(drafter.calls) == calls_before + 1
    assert drafter.calls[-1][0][-1] == 30


def test_adaptive_state_removed_when_request_finishes() -> None:
    drafter = FixedDrafter((10, 11, 12, 13))
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=4,
        token_budget=32,
        max_num_seqs=1,
        num_data_blocks=64,
    )
    request = make_request(max_tokens=2)
    scheduler.add_request(request)
    prefill(scheduler, request)

    run_spec_step(scheduler, (10, 11, 12, 13, 14))

    assert request.is_finished
    assert request.finish_reason is FinishReason.LENGTH
    assert REQUEST_ID not in scheduler._speculation_states
    assert scheduler._pending_plan is None


def test_adaptive_state_removed_when_request_aborts() -> None:
    drafter = FixedDrafter((10, 11, 12, 13))
    scheduler, _ = make_scheduler(
        drafter,
        max_draft_tokens=4,
        token_budget=32,
        max_num_seqs=1,
        num_data_blocks=64,
    )
    request = make_request(max_tokens=100)
    scheduler.add_request(request)
    prefill(scheduler, request)

    plan = scheduler.schedule()
    assert REQUEST_ID in scheduler._speculation_states

    assert scheduler.abort_request(REQUEST_ID) is True

    assert request.finish_reason is FinishReason.ABORTED
    assert REQUEST_ID not in scheduler._speculation_states
    assert scheduler._pending_plan is None


def test_speculative_request_preempts_victim_and_victim_resumes_normally() -> None:
    drafter = MappingDrafter({10: (11, 12), 20: ()})
    scheduler, manager = make_scheduler(
        drafter,
        max_draft_tokens=2,
        token_budget=8,
        max_num_seqs=2,
        block_size=2,
        num_data_blocks=3,
    )
    current = make_request("current", prompt=(1, 2))
    victim = make_request("victim", prompt=(3, 4))
    scheduler.add_request(current)
    scheduler.add_request(victim)

    initial_plan = scheduler.schedule()
    assert [
        request.request_id for request in initial_plan.scheduled_requests
    ] == ["current", "victim"]
    scheduler.update(
        initial_plan,
        ModelRunnerOutput(
            (
                SampledToken("current", 10),
                SampledToken("victim", 20),
            )
        ),
    )

    plan = scheduler.schedule()

    assert plan.preempted_request_ids == ("victim",)
    assert [
        request.request_id for request in plan.scheduled_requests
    ] == ["current"]
    scheduled = plan.scheduled_requests[0]
    assert scheduled.input_token_ids == (10, 11, 12)
    assert scheduled.num_draft_tokens == 2
    assert victim.status is RequestStatus.WAITING
    assert victim.num_computed_tokens == 0
    assert victim.output_token_ids == [20]
    assert victim.draft_token_ids == []
    assert manager.get_block_table("victim") == ()

    scheduler.update(
        plan,
        ModelRunnerOutput(
            verifications=(
                RequestVerification("current", (11, 12, 13)),
            )
        ),
    )

    assert current.output_token_ids == [10, 11, 12, 13]
    assert current.num_computed_tokens == 5
    assert len(manager.get_block_table("current")) == 3
