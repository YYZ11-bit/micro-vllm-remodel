from pathlib import Path

from safetensors import safe_open
import torch
import pytest
from micro_vllm.models.qwen2 import Qwen2Model
from micro_vllm.config import LLMConfig, ModelConfig
from micro_vllm.model_loader import load_model

import os


@pytest.mark.model
def test_qwen2_load_weight() -> None:
    path = os.getenv("MICRO_VLLM_QWEN2_MODEL_PATH")
    # assert Path(path).is_dir(), f"MICRO_VLLM_QWEN2_MODEL_PATH={path} is not a directory"
    if path is None:
        pytest.skip("MICRO_VLLM_QWEN2_MODEL_PATH is not set")
    model_config = ModelConfig(model=path, device='cpu', dtype=torch.float32)
    model= load_model(LLMConfig(model_config=model_config))

    assert isinstance(model, Qwen2Model)
    assert model.training is False
    assert model.config.architectures == ("Qwen2ForCausalLM", )
    assert model.embed_tokens.weight is model.lm_head.weight
    attn = model.layers[0].self_attn
    assert not hasattr(attn, "q_norm")
    assert not hasattr(attn, "k_norm")
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.o_proj.bias is None

    hidden = model.config.require_positive_int("hidden_size")
    heads = model.config.require_positive_int("num_attention_heads")
    kv_heads = model.config.positive_int("num_key_value_heads", heads)
    head_dim = model.config.positive_int("head_dim", hidden // heads)
    assert attn.head_dim == head_dim
    kv_dim = head_dim * kv_heads
    assert attn.q_proj.weight.shape == (hidden, hidden)
    assert attn.k_proj.weight.shape == (kv_dim, hidden)
    assert attn.v_proj.weight.shape == (kv_dim, hidden)
    assert attn.o_proj.weight.shape == (hidden, hidden)
    assert attn.q_proj.bias.shape == (hidden,)
    assert attn.k_proj.bias.shape == (kv_dim,)
    assert attn.v_proj.bias.shape == (kv_dim,)
    assert model.norm.weight.shape == (hidden, )

    with safe_open(Path(path) / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        tensor = checkpoint.get_tensor("model.layers.0.self_attn.q_proj.weight")
        flat_model = attn.q_proj.weight.reshape(-1)
        flat_tensor = tensor.reshape(-1)
        n = min(4096, flat_model.numel())
        torch.testing.assert_close(flat_model[:n], flat_tensor[:n].to(flat_model.dtype), rtol=1e-5, atol=1e-5)
