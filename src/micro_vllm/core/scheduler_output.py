"""Immutable execution-plan contracts produced by the scheduler."""

from dataclasses import dataclass

from micro_vllm.sampling.params import SamplingParams


@dataclass(frozen=True, slots=True)
class CacheGroupBlockTable:
    """One cache group's immutable logical-to-physical block snapshot."""

    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_ids", tuple(self.block_ids))


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """Immutable request snapshot with tables in cache-topology order."""

    request_id: str
    input_token_ids: tuple[int, ...]
    num_computed_tokens: int
    num_scheduled_tokens: int
    requires_sampling: bool
    block_tables: tuple[CacheGroupBlockTable, ...]
    sampling_params: SamplingParams | None = None
    sample_index: int | None = None
    num_draft_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_token_ids", tuple(self.input_token_ids))
        object.__setattr__(self, "block_tables", tuple(self.block_tables))
        if self.num_draft_tokens < 0:
            raise ValueError("num_draft_tokens must be non-negative")
        if self.num_draft_tokens > 0:
            if not self.requires_sampling:
                raise ValueError("draft tokens require sampling in the same step")
            if self.num_scheduled_tokens != self.num_draft_tokens + 1:
                raise ValueError(
                    "speculative step must schedule one pending token plus all drafts"
                )
            
    @property
    def num_sampling_rows(self) -> int:
        """Number of logits rows that produce predictions for this request."""

        if not self.requires_sampling:
            return 0
        return self.num_draft_tokens + 1 if self.num_draft_tokens > 0 else 1

@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """Immutable execution plan produced for one scheduler step."""

    scheduled_requests: tuple[ScheduledRequest, ...]
    preempted_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scheduled_requests",
            tuple(self.scheduled_requests),
        )
        object.__setattr__(
            self,
            "preempted_request_ids",
            tuple(self.preempted_request_ids),
        )
