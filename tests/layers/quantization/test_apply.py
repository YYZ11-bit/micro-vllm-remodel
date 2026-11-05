"""Tests for rewriting dense Linear modules as W4A16 layers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from micro_vllm.config import QuantizationConfig
from micro_vllm.layers.embedding import LMHead, TokenEmbedding
from micro_vllm.layers.linear import Linear
from micro_vllm.layers.normalization import RMSNorm
from micro_vllm.layers.quantization.apply import apply_quantization
from micro_vllm.layers.quantization.w4a16 import W4A16Linear


class TinyBlock(nn.Module):
    """Nested dense projections, matching the decoder-layer shape."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = Linear(16, 16, bias=True)
        self.mlp = nn.Sequential(
            Linear(16, 32, bias=False),
            Linear(32, 16, bias=False),
        )


class TinyModel(nn.Module):
    """Model-shaped module containing quantizable and excluded layers."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = TokenEmbedding(32, 16)
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.norm = RMSNorm(16)
        self.lm_head = LMHead(16, 32, tied_embedding=self.embed_tokens)


def test_apply_quantization_replaces_only_dense_linear_layers() -> None:
    model = TinyModel()

    apply_quantization(model, QuantizationConfig(method="w4a16", group_size=8))

    for block in model.blocks:
        assert type(block.proj) is W4A16Linear
        assert type(block.mlp[0]) is W4A16Linear
        assert type(block.mlp[1]) is W4A16Linear
    assert type(model.lm_head) is LMHead
    assert type(model.embed_tokens) is TokenEmbedding
    assert type(model.norm) is RMSNorm
    assert model.lm_head.weight is model.embed_tokens.weight


def test_apply_quantization_preserves_bias_configuration() -> None:
    model = TinyModel()

    apply_quantization(model, QuantizationConfig(method="w4a16", group_size=8))

    assert model.blocks[0].proj.bias is not None
    assert model.blocks[0].mlp[0].bias is None
    assert model.blocks[0].mlp[1].bias is None


def test_apply_quantization_allocates_packed_buffers() -> None:
    model = TinyModel()

    apply_quantization(model, QuantizationConfig(method="w4a16", group_size=8))

    projection = model.blocks[0].proj
    assert "weight" not in dict(projection.named_parameters())
    assert projection.weight.dtype is torch.uint8
    assert projection.weight.shape == (16, 8)
    assert projection.scale.dtype is torch.float32
    assert projection.scale.shape == (16, 2)


def test_apply_quantization_none_is_a_no_op() -> None:
    model = TinyModel()
    projection = model.blocks[0].proj

    apply_quantization(model, QuantizationConfig())

    assert model.blocks[0].proj is projection
    assert type(model.blocks[0].proj) is Linear


class MismatchedModel(nn.Module):
    """Projection whose input width cannot be split into even groups."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = Linear(12, 16, bias=False)


def test_apply_quantization_rejects_incompatible_group_size() -> None:
    model = MismatchedModel()

    with pytest.raises(ValueError, match="not divisible by group_size"):
        apply_quantization(
            model,
            QuantizationConfig(method="w4a16", group_size=8),
        )


class KernelCompatibleModel(nn.Module):
    """Projection shaped for the fused kernel requirements."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = Linear(128, 8, bias=False)


@pytest.mark.parametrize("kernel", ["reference", "int4pack", "auto"])
def test_apply_quantization_propagates_kernel_choice(kernel: str) -> None:
    model = KernelCompatibleModel()

    apply_quantization(
        model,
        QuantizationConfig(method="w4a16", group_size=32, kernel=kernel),
    )

    assert type(model.proj) is W4A16Linear
    assert model.proj.kernel == kernel
