
from typing import Protocol
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class DraftOutput:
    tokens_ids: tuple[int, ...]

class Drafter(Protocol):
    def propose(self, token_ids: Sequence[int], max_tokens: int) -> DraftOutput:
        ...
