"""Real Llama-3.2-1B checks: loading, llama3 RoPE, and HF parity."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from micro_vllm.attention.attention import map_cache_slots
from micro_vllm.config import LLMConfig, ModelConfig
from micro_vllm.layers.rotary_embedding import RotaryEmbedding
from micro_vllm.model_loader import load_model
from micro_vllm.models.llama import LlamaModel

pytestmark = pytest.mark.model


def _model_path() -> Path:
    path = os.getenv("MICRO_VLLM_LLAMA32_MODEL_PATH")
    if path is None:
        pytest.skip("MICRO_VLLM_LLAMA32_MODEL_PATH is not set")
    model_path = Path(path).expanduser()
    assert model_path.is_dir()
    return model_path

def test_llama32_load_weight() -> None:
    model_path = _model_path()
    model = load_model(
        LLMConfig(
            model_config=ModelConfig(
                model=model_path, device="cpu", dtype=torch.float32
            )
        )
    )
    assert isinstance(model, LlamaModel)
    assert model.training is False
    assert model.config.architectures == ("LlamaForCausalLM",)

    # config 里的 rope 字段被正确解析
    assert model.config.values["rope_theta"] == 500_000.0
    scaling = model.config.values["rope_scaling"]
    assert scaling["rope_type"] == "llama3"
    assert scaling["original_max_position_embeddings"] == 8192

    attention = model.layers[0].self_attn
    assert attention.head_dim == 64
    assert attention.rotary_embedding.base == 500_000.0
    assert attention.q_proj.weight.shape == (2048, 2048)
    assert attention.k_proj.weight.shape == (8 * 64, 2048)
    assert attention.v_proj.weight.shape == (512, 2048)
    assert attention.o_proj.weight.shape == (2048, 2048)
    assert model.embed_tokens.weight is model.lm_head.weight  # tied

    with safe_open(
        model_path / "model.safetensors", framework="pt", device="cpu"
    ) as checkpoint:
        tensor = checkpoint.get_tensor(
            "model.layers.0.self_attn.q_proj.weight"
        ).reshape(-1)
    flat = attention.q_proj.weight.reshape(-1)
    count = min(4096, flat.numel())
    torch.testing.assert_close(
        flat[:count], tensor[:count].float(), rtol=1e-5, atol=1e-5
    )
    del model
    gc.collect()

def test_llama32_rope_scaling_matches_transformers() -> None:
    model_path = _model_path()
    config = LlamaConfig.from_pretrained(model_path)
    head_dim = config.head_dim or config.hidden_size // config.num_attention_heads

    reference = LlamaRotaryEmbedding(config)
    rotary = RotaryEmbedding(
        head_dim=head_dim,
        base=config.rope_theta,
        rope_scaling=config.rope_scaling,
    )
    torch.testing.assert_close(
        rotary._inv_freq, reference.inv_freq, atol=1e-9, rtol=1e-7
    )

    positions = torch.tensor([0, 1, 127, 8192, 131071], dtype=torch.int64)
    query = torch.randn(5, config.num_attention_heads, head_dim)
    key = torch.randn(5, config.num_key_value_heads, head_dim)
    rotated_query, rotated_key = rotary(positions, query, key)

    cos, sin = reference(query, positions.unsqueeze(0))
    expected_query, expected_key = apply_rotary_pos_emb(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        cos,
        sin,
    )
    torch.testing.assert_close(
        rotated_query, expected_query.squeeze(0).transpose(0, 1),
        atol=1e-6, rtol=1e-6,
    )
    torch.testing.assert_close(
        rotated_key, expected_key.squeeze(0).transpose(0, 1),
        atol=1e-6, rtol=1e-6,
    )

def test_llama32_parity_matches_transformers() -> None:
    model_path = _model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = torch.tensor(
        tokenizer("The capital of France is", add_special_tokens=False)["input_ids"],
        dtype=torch.int64,
    )
    positions = torch.arange(input_ids.numel(), dtype=torch.int64)

    model = load_model(
        LLMConfig(
            model_config=ModelConfig(
                model=model_path, device="cpu", dtype=torch.float32
            )
        )
    )
    block_size = 2
    num_blocks = (input_ids.numel() + block_size - 1) // block_size
    physical_blocks = torch.arange(num_blocks - 1, -1, -1)
    metadata = model.layer_attentions[0].build_metadata(
        query_positions=positions,
        query_start_locations=torch.tensor([0, input_ids.numel()], dtype=torch.int64),
        context_lengths=torch.tensor([input_ids.numel()], dtype=torch.int64),
        block_table=physical_blocks.unsqueeze(0),
        slot_mapping=map_cache_slots(positions, physical_blocks, block_size),
        block_size=block_size,
    )
    kv_heads = model.config.require_positive_int("num_key_value_heads")
    num_layers = model.config.require_positive_int("num_hidden_layers")
    head_dim = (
        model.config.require_positive_int("hidden_size")
        // model.config.require_positive_int("num_attention_heads")
    )
    layer_caches = tuple(
        torch.zeros((2, num_blocks, block_size, kv_heads, head_dim))
        for _ in range(num_layers)
    )
    with torch.no_grad():
        hidden = model(input_ids, positions, layer_caches, (metadata,) * num_layers)
        logits = model.compute_logits(hidden)
    del model, layer_caches, metadata
    gc.collect()

    reference = LlamaForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        reference_hidden = reference.model(
            input_ids.unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            use_cache=False,
        ).last_hidden_state.squeeze(0)
        reference_logits = reference.lm_head(reference_hidden)

    torch.testing.assert_close(hidden, reference_hidden, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits, reference_logits, atol=1e-5, rtol=1e-5)
    assert logits.argmax(-1).equal(reference_logits.argmax(-1))

