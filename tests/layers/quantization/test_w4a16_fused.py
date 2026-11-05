"""Parity and validation tests for the fused ``_weight_int4pack_mm`` path."""

from __future__ import annotations

import pytest
import torch

from micro_vllm.layers.quantization.w4a16 import (
    FUSED_DECODE_MAX_TOKENS,
    W4A16Linear,
)


def _cuda_supports_int4pack() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8


requires_int4pack = pytest.mark.skipif(
    not _cuda_supports_int4pack(),
    reason="int4pack kernel requires CUDA with compute capability >= 8.0",
)


def _make_layers(
    in_features: int,
    out_features: int,
    *,
    group_size: int,
    kernel: str,
    dtype: torch.dtype = torch.float16,
    with_bias: bool = True,
    seed: int = 0,
) -> W4A16Linear:
    torch.manual_seed(seed)
    linear = W4A16Linear(
        in_features,
        out_features,
        bias=with_bias,
        group_size=group_size,
        kernel=kernel,
    )
    weight = torch.randn(out_features, in_features) * 0.03
    if with_bias:
        linear.bias.data.copy_(torch.randn(out_features) * 0.05)
    linear.load_weight(weight)
    return linear.to(device="cuda", dtype=dtype)


def _reference_and_fused(
    in_features: int,
    out_features: int,
    *,
    group_size: int,
    dtype: torch.dtype = torch.float16,
    with_bias: bool = True,
) -> tuple[W4A16Linear, W4A16Linear]:
    # load_weight is deterministic, so both layers see identical qweight/scale.
    reference = _make_layers(
        in_features,
        out_features,
        group_size=group_size,
        kernel="reference",
        dtype=dtype,
        with_bias=with_bias,
    )
    fused = _make_layers(
        in_features,
        out_features,
        group_size=group_size,
        kernel="int4pack",
        dtype=dtype,
        with_bias=with_bias,
    )
    return reference, fused


@pytest.mark.gpu
@requires_int4pack
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
@pytest.mark.parametrize(
    ("in_features", "out_features"),
    [(1024, 16), (2048, 128), (3072, 1024)],
)
@pytest.mark.parametrize("num_tokens", [1, 33])
def test_fused_int4pack_matches_reference_dequant(
    group_size: int,
    in_features: int,
    out_features: int,
    num_tokens: int,
) -> None:
    reference, fused = _reference_and_fused(
        in_features, out_features, group_size=group_size
    )
    x = torch.randn(num_tokens, in_features, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        expected = reference(x)
        actual = fused(x)

    assert actual.dtype is torch.float16
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


@pytest.mark.gpu
@requires_int4pack
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_int4pack_preserves_leading_dimensions_and_dtype(
    dtype: torch.dtype,
) -> None:
    reference, fused = _reference_and_fused(
        1024, 128, group_size=128, dtype=dtype, with_bias=False
    )
    x = torch.randn(2, 3, 1024, dtype=dtype, device="cuda")

    with torch.no_grad():
        expected = reference(x)
        actual = fused(x)

    assert actual.shape == (2, 3, 128)
    assert actual.dtype is dtype
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


@pytest.mark.gpu
@requires_int4pack
def test_auto_uses_fused_only_within_decode_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_layers(
        1024, 128, group_size=128, kernel="auto", with_bias=False
    )
    calls: list[int] = []
    original = W4A16Linear._ensure_int4pack

    def counting_ensure(self: W4A16Linear) -> None:
        calls.append(1)
        original(self)

    monkeypatch.setattr(W4A16Linear, "_ensure_int4pack", counting_ensure)

    with torch.no_grad():
        layer(
            torch.randn(
                FUSED_DECODE_MAX_TOKENS, 1024, dtype=torch.float16, device="cuda"
            )
        )
        assert len(calls) == 1

        # Above the threshold the reference dequant path must be used instead.
        layer(
            torch.randn(
                FUSED_DECODE_MAX_TOKENS + 1,
                1024,
                dtype=torch.float16,
                device="cuda",
            )
        )
        assert len(calls) == 1


@pytest.mark.gpu
@requires_int4pack
def test_fused_packing_is_cached_and_invalidated_by_load_weight() -> None:
    layer = _make_layers(
        1024, 128, group_size=128, kernel="int4pack", with_bias=False
    )
    assert layer._int4pack_weight is None
    assert layer._int4pack_scale_and_zero is None

    x = torch.randn(2, 1024, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        layer(x)
    packed = layer._int4pack_weight
    scale_and_zero = layer._int4pack_scale_and_zero
    assert packed is not None
    assert scale_and_zero is not None
    assert packed.dtype is torch.int32
    assert scale_and_zero.dtype is torch.bfloat16
    assert scale_and_zero.shape == (1024 // 128, 128, 2)

    with torch.no_grad():
        layer(x)
    assert layer._int4pack_weight is packed
    assert layer._int4pack_scale_and_zero is scale_and_zero

    layer.load_weight(torch.randn(128, 1024) * 0.02)
    assert layer._int4pack_weight is None
    assert layer._int4pack_scale_and_zero is None


def test_fused_forward_rejects_cpu_device() -> None:
    layer = W4A16Linear(
        128, 8, bias=False, group_size=32, kernel="int4pack"
    )
    layer.load_weight(torch.randn(8, 128) * 0.03)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        layer(torch.randn(1, 128))


@pytest.mark.parametrize("kernel", ["int4pack", "auto"])
@pytest.mark.parametrize(
    ("in_features", "out_features", "group_size"),
    [
        (96, 8, 32),    # in_features 不能被 inner_k_tiles * 16 整除
        (128, 12, 32),  # out_features 不能被 8 整除
        (128, 8, 16),   # kernel 不支持的 group size
    ],
)
def test_fused_kernel_rejects_unsupported_layer_shapes(
    kernel: str,
    in_features: int,
    out_features: int,
    group_size: int,
) -> None:
    with pytest.raises(ValueError):
        W4A16Linear(
            in_features,
            out_features,
            bias=False,
            group_size=group_size,
            kernel=kernel,
        )


def test_w4a16_linear_rejects_unknown_kernel() -> None:
    with pytest.raises(ValueError, match="unsupported kernel"):
        W4A16Linear(128, 8, bias=False, group_size=32, kernel="triton")
