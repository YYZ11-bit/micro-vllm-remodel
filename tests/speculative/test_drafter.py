"""Tests for speculative decoding contracts and package exports."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.speculative import (
    AcceptanceResult,
    Drafter,
    DraftOutput,
    NGramDrafter,
    SpeculativeConfig,
    greedy_accept,
)


def test_draft_output_preserves_token_ids() -> None:
    output = DraftOutput(tokens_ids=(1, 2, 3))

    assert output.tokens_ids == (1, 2, 3)


def test_draft_output_is_frozen() -> None:
    output = DraftOutput(tokens_ids=(1,))

    with pytest.raises(FrozenInstanceError):
        output.tokens_ids = ()


def test_public_package_exports_contracts() -> None:
    assert Drafter is not None
    assert NGramDrafter is not None
    assert SpeculativeConfig().enabled is False
    assert AcceptanceResult([], [], [1]).extra == [1]
    assert greedy_accept([], [1]).extra == [1]
