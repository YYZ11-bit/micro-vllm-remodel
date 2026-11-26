"""Tests for the executor input and output contracts."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.executor.interface import (
    ModelRunnerOutput,
    RequestVerification,
    SampledToken,
)


def test_model_runner_output_allows_no_sampled_tokens() -> None:
    assert ModelRunnerOutput().sampled_tokens == ()
    assert ModelRunnerOutput().verifications == ()


def test_request_verification_owns_predicted_token_snapshot() -> None:
    predicted_token_ids = [1, 2, 3]

    verification = RequestVerification("request-1", predicted_token_ids)
    predicted_token_ids.append(4)

    assert verification.predicted_token_ids == (1, 2, 3)
    with pytest.raises(FrozenInstanceError):
        verification.predicted_token_ids = ()


def test_model_runner_output_owns_verification_snapshot() -> None:
    verification = RequestVerification("request-1", (1, 2))
    source = [verification]

    output = ModelRunnerOutput(verifications=source)
    source.clear()

    assert output.verifications == (verification,)


def test_model_runner_output_owns_an_immutable_snapshot() -> None:
    sampled_token = SampledToken("sampling", 3)
    source = [sampled_token]

    output = ModelRunnerOutput(source)
    source.clear()

    assert output.sampled_tokens == (sampled_token,)
    with pytest.raises(FrozenInstanceError):
        output.sampled_tokens = ()
