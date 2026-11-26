"""Tests for suffix-match n-gram speculative drafting."""

import pytest

from micro_vllm.speculative.ngram import NGramDrafter


def propose(
    token_ids: list[int] | tuple[int, ...],
    *,
    ngram_size: int = 3,
    max_tokens: int = 4,
) -> tuple[int, ...]:
    output = NGramDrafter(ngram_size).propose(token_ids, max_tokens)
    return tuple(output.tokens_ids)


def test_returns_empty_when_sequence_is_shorter_than_ngram() -> None:
    assert propose([1, 2], ngram_size=3) == ()


def test_returns_empty_when_suffix_has_no_earlier_match() -> None:
    assert propose([1, 2, 3, 4], ngram_size=3) == ()


def test_matches_occurrence_at_sequence_start() -> None:
    assert propose([1, 2, 3, 1, 2, 3], ngram_size=3, max_tokens=3) == (
        1,
        2,
        3,
    )


def test_prefers_latest_earlier_occurrence() -> None:
    token_ids = [1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3]

    assert propose(token_ids, ngram_size=3, max_tokens=3) == (5, 1, 2)


def test_truncates_proposal_to_max_tokens() -> None:
    assert propose([1, 2, 3, 1, 2, 3], ngram_size=3, max_tokens=2) == (1, 2)


def test_returns_available_tokens_when_fewer_than_max() -> None:
    assert propose([1, 2, 1, 2], ngram_size=2, max_tokens=8) == (1, 2)


def test_returns_empty_for_zero_or_negative_max_tokens() -> None:
    assert propose([1, 2, 3, 1, 2, 3], max_tokens=0) == ()
    assert propose([1, 2, 3, 1, 2, 3], max_tokens=-1) == ()


def test_returns_empty_when_ngram_covers_entire_sequence() -> None:
    assert propose([1, 2, 3], ngram_size=3) == ()


@pytest.mark.parametrize("ngram_size", [0, -1])
def test_rejects_non_positive_ngram_size(ngram_size: int) -> None:
    with pytest.raises(ValueError, match="ngram_size must be a positive integer"):
        NGramDrafter(ngram_size)


def test_does_not_mutate_input_tokens() -> None:
    token_ids = [1, 2, 3, 1, 2, 3]

    propose(token_ids)

    assert token_ids == [1, 2, 3, 1, 2, 3]
