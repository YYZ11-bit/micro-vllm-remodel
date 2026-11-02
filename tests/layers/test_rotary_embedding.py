"""Tests for reusable rotary embeddings."""

from collections.abc import Mapping

import pytest
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from micro_vllm.layers.rotary_embedding import RotaryEmbedding
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from micro_vllm.layers.rotary_embedding import _inverse_frequencies


def transformers_rotary_reference(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = query.shape[-1]
    frequency_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inverse_frequencies = 1.0 / (base ** (frequency_indices / head_dim))
    frequencies = torch.outer(positions.to(torch.float32), inverse_frequencies)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    cosine = angles.cos().unsqueeze(0)
    sine = angles.sin().unsqueeze(0)
    reference_query, reference_key = apply_rotary_pos_emb(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        cosine,
        sine,
    )
    return (
        reference_query.squeeze(0).transpose(0, 1),
        reference_key.squeeze(0).transpose(0, 1),
    )


def test_rotary_embedding_matches_transformers_for_flattened_positions() -> None:
    positions = torch.tensor([5, 1, 127], dtype=torch.int64)
    query = torch.linspace(-1.0, 1.0, steps=3 * 4 * 8).reshape(3, 4, 8)
    key = torch.linspace(1.0, -1.0, steps=3 * 2 * 8).reshape(3, 2, 8)
    rotary = RotaryEmbedding(head_dim=8, base=50_000.0)

    rotated_query, rotated_key = rotary(positions, query, key)
    expected_query, expected_key = transformers_rotary_reference(
        positions,
        query,
        key,
        base=50_000.0,
    )

    torch.testing.assert_close(rotated_query, expected_query, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(rotated_key, expected_key, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_embedding_preserves_qk_dtype(dtype: torch.dtype) -> None:
    positions = torch.tensor([0, 3], dtype=torch.int64)
    query = torch.ones((2, 2, 4), dtype=dtype)
    key = torch.ones((2, 1, 4), dtype=dtype)

    rotated_query, rotated_key = RotaryEmbedding(head_dim=4)(
        positions,
        query,
        key,
    )

    assert rotated_query.dtype is dtype
    assert rotated_key.dtype is dtype


def test_rotary_embedding_does_not_modify_value() -> None:
    positions = torch.tensor([0, 1], dtype=torch.int64)
    query = torch.ones((2, 2, 4))
    key = torch.ones((2, 1, 4))
    value = torch.linspace(-1.0, 1.0, steps=8).reshape(2, 1, 4)
    original_value = value.clone()

    RotaryEmbedding(head_dim=4)(positions, query, key)

    assert torch.equal(value, original_value)

LLAMA32_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

def test_inverse_frequencies_with_llama3_rope_scaling() -> None:
    hf_config = LlamaConfig(
        head_dim=64, hidden_size=2048, num_attention_heads=32,
        rope_theta=500_000.0, rope_scaling=LLAMA32_ROPE_SCALING,  
        max_position_embeddings=131072
    )
    expected = LlamaRotaryEmbedding(hf_config).inv_freq
    actual = _inverse_frequencies(
        head_dim=64,
        base=500_000.0,
        rope_scaling=LLAMA32_ROPE_SCALING,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-8)

    raw = _inverse_frequencies(64, 500_000.0, None)
    assert torch.equal(actual[:15], raw[:15])                       # 高频不动
    torch.testing.assert_close(actual[18:], raw[18:] / 32.0)        # 低频 /32
    assert torch.all(actual[15:18] > raw[15:18] / 32.0)             # 过渡带夹在中间
    assert torch.all(actual[15:18] < raw[15:18])

    torch.testing.assert_close(
        _inverse_frequencies(8, 50_000.0, None),
        _inverse_frequencies(8, 50_000.0, {"rope_type": "default"}),
        rtol=0.0, atol=0.0,
    )
