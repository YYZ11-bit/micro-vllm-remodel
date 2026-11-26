"""Tests for immutable scheduler execution-plan contracts."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.core.scheduler_output import (
    CacheGroupBlockTable,
    ScheduledRequest,
    SchedulerOutput,
)
from micro_vllm.sampling.params import SamplingParams


def make_scheduled_request() -> ScheduledRequest:
    return ScheduledRequest(
        request_id="request-1",
        input_token_ids=(1, 2),
        num_computed_tokens=0,
        num_scheduled_tokens=2,
        requires_sampling=True,
        block_tables=(CacheGroupBlockTable((0,)),),
        sampling_params=SamplingParams(max_tokens=2),
        sample_index=0,
    )


def make_speculative_request(
    *,
    num_draft_tokens: int,
    num_scheduled_tokens: int | None = None,
    requires_sampling: bool = True,
) -> ScheduledRequest:
    if num_scheduled_tokens is None:
        num_scheduled_tokens = num_draft_tokens + 1
    return ScheduledRequest(
        request_id="request-1",
        input_token_ids=tuple(range(num_scheduled_tokens)),
        num_computed_tokens=4,
        num_scheduled_tokens=num_scheduled_tokens,
        requires_sampling=requires_sampling,
        block_tables=(CacheGroupBlockTable((0,)),),
        sampling_params=(
            SamplingParams(max_tokens=4) if requires_sampling else None
        ),
        sample_index=0 if requires_sampling else None,
        num_draft_tokens=num_draft_tokens,
    )


def test_num_sampling_rows_is_zero_without_sampling() -> None:
    request = make_speculative_request(
        num_draft_tokens=0,
        num_scheduled_tokens=2,
        requires_sampling=False,
    )

    assert request.num_sampling_rows == 0


def test_num_sampling_rows_is_one_for_ordinary_sampling() -> None:
    assert make_scheduled_request().num_sampling_rows == 1


@pytest.mark.parametrize("num_draft_tokens", [1, 2, 4])
def test_num_sampling_rows_counts_pending_token_and_all_drafts(
    num_draft_tokens: int,
) -> None:
    request = make_speculative_request(
        num_draft_tokens=num_draft_tokens,
    )

    assert request.num_sampling_rows == num_draft_tokens + 1


def test_rejects_negative_draft_token_count() -> None:
    with pytest.raises(
        ValueError,
        match="num_draft_tokens must be non-negative",
    ):
        make_speculative_request(num_draft_tokens=-1)


def test_rejects_draft_tokens_without_sampling() -> None:
    with pytest.raises(
        ValueError,
        match="draft tokens require sampling in the same step",
    ):
        make_speculative_request(
            num_draft_tokens=2,
            requires_sampling=False,
        )


def test_rejects_speculative_step_that_omits_draft_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="one pending token plus all drafts",
    ):
        make_speculative_request(
            num_draft_tokens=2,
            num_scheduled_tokens=2,
        )


@pytest.mark.parametrize(
    ("field_name", "new_value"),
    [
        ("request_id", "request-2"),
        ("input_token_ids", (3,)),
        ("num_computed_tokens", 1),
        ("num_scheduled_tokens", 1),
        ("requires_sampling", False),
        ("block_tables", (CacheGroupBlockTable((1,)),)),
        ("sampling_params", None),
        ("sample_index", 1),
        ("num_draft_tokens", 1),
    ],
)
def test_scheduled_request_is_frozen(field_name: str, new_value: object) -> None:
    scheduled_request = make_scheduled_request()

    with pytest.raises(FrozenInstanceError):
        setattr(scheduled_request, field_name, new_value)

    assert isinstance(scheduled_request.input_token_ids, tuple)


def test_scheduler_output_is_frozen() -> None:
    preempted_request_ids = ["request-2"]
    output = SchedulerOutput(
        (make_scheduled_request(),),
        preempted_request_ids,
    )
    preempted_request_ids.append("request-3")

    with pytest.raises(FrozenInstanceError):
        output.scheduled_requests = ()

    assert isinstance(output.scheduled_requests, tuple)
    assert output.preempted_request_ids == ("request-2",)


def test_original_token_list_does_not_change_plan_snapshot() -> None:
    input_token_ids = [1, 2]

    scheduled_request = ScheduledRequest(
        request_id="request-1",
        input_token_ids=input_token_ids,
        num_computed_tokens=0,
        num_scheduled_tokens=2,
        requires_sampling=True,
        block_tables=(CacheGroupBlockTable((0,)),),
        sampling_params=SamplingParams(max_tokens=2),
        sample_index=0,
    )
    input_token_ids[0] = 9
    input_token_ids.append(3)

    assert scheduled_request.input_token_ids == (1, 2)


def test_original_block_table_list_does_not_change_plan_snapshot() -> None:
    block_table = [0, 7]
    group_block_table = CacheGroupBlockTable(block_table)
    block_tables = [group_block_table]
    scheduled_request = ScheduledRequest(
        request_id="request-1",
        input_token_ids=(1,),
        num_computed_tokens=0,
        num_scheduled_tokens=1,
        requires_sampling=True,
        block_tables=block_tables,
        sampling_params=SamplingParams(max_tokens=1),
        sample_index=0,
    )

    block_table[0] = 9
    block_table.append(2)
    block_tables.clear()

    assert scheduled_request.block_tables == (group_block_table,)
    assert group_block_table.block_ids == (0, 7)
    with pytest.raises(FrozenInstanceError):
        group_block_table.block_ids = ()
