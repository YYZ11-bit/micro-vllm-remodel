"""Tests for greedy speculative acceptance."""

import pytest

from micro_vllm.speculative.acceptance import greedy_accept


def test_accepts_all_proposals_and_returns_bonus_token() -> None:
    result = greedy_accept([4, 5], [4, 5, 6])

    assert result.accepted == [4, 5]
    assert result.replacement == []
    assert result.extra == [6]


def test_first_rejection_replaces_and_stops_before_bonus() -> None:
    result = greedy_accept([4, 5], [9, 5, 6])

    assert result.accepted == []
    assert result.replacement == [9]
    assert result.extra == []


def test_mid_rejection_keeps_accepted_prefix() -> None:
    result = greedy_accept([4, 5, 6], [4, 9, 6, 7])

    assert result.accepted == [4]
    assert result.replacement == [9]
    assert result.extra == []


def test_empty_proposal_returns_only_the_bonus_token() -> None:
    result = greedy_accept([], [7])

    assert result.accepted == []
    assert result.replacement == []
    assert result.extra == [7]


@pytest.mark.parametrize(
    ("proposals", "predictions"),
    [
        ([4], []),
        ([4], [4, 5, 6]),
        ([4, 5], [4]),
        ([], []),
    ],
)
def test_rejects_prediction_length_other_than_proposals_plus_one(
    proposals: list[int],
    predictions: list[int],
) -> None:
    with pytest.raises(ValueError, match="predictions must contain"):
        greedy_accept(proposals, predictions)
