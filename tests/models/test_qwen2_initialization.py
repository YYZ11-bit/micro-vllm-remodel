"""Tests for initializing Qwen2 from the generic model config."""

import pytest

from micro_vllm.attention.spec import FullAttentionSpec
from micro_vllm.config import ModelConfig
from micro_vllm.models.qwen2 import Qwen2Model


def make_model(**overrides: object) -> Qwen2Model:
    values = {
        "architectures": ["Qwen2ForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        #"head_dim": 24,
    }
    values.update(overrides)
    return Qwen2Model(ModelConfig(model="unused", metadata=values))


def test_qwen2_initializes_fixed_attention_biases_without_qk_norm() -> None:
    model = make_model(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        # head_dim=32,
        rms_norm_eps=1e-5,
        rope_theta=500_000.0,
        tie_word_embeddings=False,
    )

    attention = model.layers[0].self_attn
    assert model.embed_tokens.weight.shape == (32, 32)
    assert len(model.layers) == 2
    assert attention.q_proj.weight.shape == (32, 32)
    assert attention.k_proj.weight.shape == (16, 32)
    assert attention.v_proj.weight.shape == (16, 32)
    assert attention.o_proj.weight.shape == (32, 32)
    assert attention.q_proj.bias is not None
    assert attention.k_proj.bias is not None
    assert attention.v_proj.bias is not None
    assert attention.o_proj.bias is None
    assert not hasattr(attention, 'q_norm')
    assert not hasattr(attention, 'k_norm')
    assert 'q_norm.weight' not in dict(model.named_parameters())
    assert 'k_norm.weight' not in dict(model.named_parameters())
    assert attention.rotary_embedding.base == 500_000.0
    assert model.embed_tokens.weight is not model.lm_head.weight
    runtime_attention = attention.attention
    assert model.layer_attentions == (
        runtime_attention,
        runtime_attention,
    )
    assert model.layer_attentions[0].spec == FullAttentionSpec()
    assert all(
        layer.self_attn.attention is model.layer_attentions[layer_index]
        for layer_index, layer in enumerate(model.layers)
    )


def test_qwen2_uses_simple_defaults() -> None:
    model = make_model(head_dim=8, num_key_value_heads=4)
    attention = model.layers[0].self_attn

    # assert attention.head_dim == 16
    assert attention.head_dim == 8
    assert attention.num_key_value_heads == 4
    assert attention.rotary_embedding.base == 1_000_000.0
    assert model.norm.variance_epsilon == 1e-6
    assert model.embed_tokens.weight is not model.lm_head.weight


def test_qwen2_always_has_qkv_biases_but_no_o_proj_bias() -> None:
    model = make_model()
    attention = model.layers[0].self_attn

    assert attention.q_proj.bias is not None
    assert attention.k_proj.bias is not None
    assert attention.v_proj.bias is not None
    assert attention.o_proj.bias is None


def test_qwen2_rejects_incompatible_attention_head_counts() -> None:
    with pytest.raises(
        ValueError,
        match="num_attention_heads must be divisible by num_key_value_heads",
    ):
        make_model(num_attention_heads=3, num_key_value_heads=2)
