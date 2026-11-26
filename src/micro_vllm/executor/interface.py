"""Stable executor contracts consumed by the engine."""

from dataclasses import dataclass
from typing import Protocol

from micro_vllm.core.scheduler_output import SchedulerOutput


@dataclass(frozen=True, slots=True)
class SampledToken:
    """One sampled token explicitly associated with its request."""

    request_id: str
    token_id: int


@dataclass(frozen=True, slots=True)
class ModelRunnerOutput:
    """Minimal result returned after executing one scheduler plan."""

    sampled_tokens: tuple[SampledToken, ...] = ()
    verifications: tuple["RequestVerification", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampled_tokens", tuple(self.sampled_tokens))
        object.__setattr__(self, "verifications", tuple(self.verifications))


class Executor(Protocol):
    """Execution boundary required by the engine control loop."""

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """Execute one immutable scheduler plan."""

        ...

    def shutdown(self) -> None:
        """Release resources owned by the executor."""

        ...

@dataclass(frozen=True, slots=True)
class RequestVerification:
    request_id: str
    predicted_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "predicted_token_ids", tuple(self.predicted_token_ids)
        )