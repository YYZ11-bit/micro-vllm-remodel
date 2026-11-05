"""End-to-end loader tests for W4A16 quantized checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from micro_vllm.config import LLMConfig, ModelConfig, QuantizationConfig
from micro_vllm.layers.embedding import LMHead, TokenEmbedding
from micro_vllm.layers.normalization import RMSNorm
from micro_vllm.layers.quantization.w4a16 import W4A16Linear
from micro_vllm.model_loader import load_model


_MODEL_FIXTURES = (
    "tiny_hf_llama_checkpoint",
    "tiny_hf_qwen2_checkpoint",
    "tiny_hf_qwen3_checkpoint",
    "tiny_hf_mistral_checkpoint",
)


def _config(model_path: Path, *, group_size: int = 8) -> LLMConfig:
    return LLMConfig(
        model_config=ModelConfig(
            model=model_path,
            device="cpu",
            dtype=torch.float32,
            quantization=QuantizationConfig(
                method="w4a16",
                group_size=group_size,
            ),
        ),
    )


def _dense_config(model_path: Path) -> LLMConfig:
    return LLMConfig(
        model_config=ModelConfig(
            model=model_path,
            device="cpu",
            dtype=torch.float32,
        ),
    )


def _byte_count(model: torch.nn.Module) -> int:
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


@pytest.mark.parametrize("fixture_name", _MODEL_FIXTURES)
def test_loader_quantizes_every_projection(
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    _, model_path = request.getfixturevalue(fixture_name)

    model = load_model(_config(model_path))

    for layer in model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert type(getattr(layer.self_attn, name)) is W4A16Linear
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert type(getattr(layer.mlp, name)) is W4A16Linear
    assert type(model.embed_tokens) is TokenEmbedding
    assert type(model.norm) is RMSNorm
    assert type(model.lm_head) is LMHead


def test_quantized_buffers_survive_dtype_conversion(
    tiny_hf_llama_checkpoint: tuple[torch.nn.Module, Path],
) -> None:
    _, model_path = tiny_hf_llama_checkpoint

    model = load_model(_config(model_path)).to(dtype=torch.float16)

    projection = model.layers[0].self_attn.q_proj
    assert projection.weight.dtype is torch.uint8
    assert projection.scale.dtype is torch.float16
    assert projection.weight.shape == (16, 8)
    assert projection.scale.shape == (16, 2)


def test_quantization_reduces_parameter_bytes(
    tiny_hf_llama_checkpoint: tuple[torch.nn.Module, Path],
) -> None:
    _, model_path = tiny_hf_llama_checkpoint

    dense = load_model(_dense_config(model_path))
    quantized = load_model(_config(model_path))

    assert _byte_count(quantized) < _byte_count(dense) * 0.5


def test_qwen2_quantized_layer_keeps_float_bias(
    tiny_hf_qwen2_checkpoint: tuple[torch.nn.Module, Path],
) -> None:
    _, model_path = tiny_hf_qwen2_checkpoint

    model = load_model(_config(model_path))

    attention = model.layers[0].self_attn
    assert type(attention.q_proj) is W4A16Linear
    assert isinstance(attention.q_proj.bias, torch.nn.Parameter)
    assert attention.q_proj.bias.dtype is torch.float32
    assert attention.q_proj.bias.shape == (attention.num_attention_heads * attention.head_dim,)
