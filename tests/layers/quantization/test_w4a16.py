import torch
import pytest
from micro_vllm.layers.quantization.reference import (
    dequantize_weight,
    pack_int4,
    unpack_int4,
    quantize_groupwise,
)
from micro_vllm.layers.quantization.w4a16 import W4A16Linear

def test_compare_load_weight_and_manual():
    in_features = 8
    out_features = 4
    group_size = 4
    weight = torch.rand((out_features, in_features)) - 0.5
    linear = W4A16Linear(in_features, out_features, group_size=group_size, bias=False)
    linear.load_weight(weight)
    input_tensor = torch.rand((2, in_features))
    model_output = linear(input_tensor)

    # Manually quantize and dequantize the weight
    quantized, scale = quantize_groupwise(weight, group_size=group_size)
    packed = pack_int4(quantized)
    unpacked = unpack_int4(packed)
    dequantized_weight = dequantize_weight(unpacked, scale, group_size=group_size)
    manual_output = torch.nn.functional.linear(
        input_tensor, dequantized_weight.to(input_tensor.dtype), linear.bias
    )
    torch.testing.assert_close(model_output, manual_output, rtol=0, atol=0)

def test_w4a16_linear_bias_matches_dense():
    in_features = 8
    out_features = 4
    group_size = 4
    weight = torch.rand((out_features, in_features)) - 0.5
    bias = torch.rand(out_features) - 0.5

    # Create W4A16Linear layer and load weight and bias
    linear = W4A16Linear(in_features, out_features, group_size=group_size)
    linear.load_weight(weight)
    linear.bias.data.copy_(bias)

    quantized, scale = quantize_groupwise(weight, group_size=group_size)
    dequantized_weight = dequantize_weight(unpack_int4(pack_int4(quantized)), scale, group_size=group_size)
    dense_linear = torch.nn.Linear(in_features, out_features)
    dense_linear.weight.data.copy_(dequantized_weight)
    dense_linear.bias.data.copy_(bias)

    input_tensor = torch.rand((2, in_features))
    w4a16_output = linear(input_tensor)
    dense_output = dense_linear(input_tensor)

    torch.testing.assert_close(w4a16_output, dense_output, rtol=0, atol=0)   

def test_w4a16_linear_dtype_invariance():
    in_features = 8
    out_features = 4
    group_size = 4
    weight = torch.rand((out_features, in_features)) - 0.5
    bias = torch.rand(out_features) - 0.5

    # Create W4A16Linear layer and load weight and bias
    linear = W4A16Linear(in_features, out_features, group_size=group_size)
    linear.load_weight(weight)
    linear.bias.data.copy_(bias)

    input_tensor_fp16 = torch.rand((2, in_features), dtype=torch.float16)
    input_tensor_fp32 = input_tensor_fp16.to(torch.float32)

    output_fp16 = linear(input_tensor_fp16)
    output_fp32 = linear(input_tensor_fp32)

    torch.testing.assert_close(output_fp16.to(torch.float32), output_fp32, rtol=1e-3, atol=1e-3)                                                                 

def test_w4a16_linear_weight_shape_and_memory_savings():
    in_features = 8
    out_features = 4
    group_size = 4
    weight = torch.rand((out_features, in_features)) - 0.5

    # Create W4A16Linear layer and load weight
    linear = W4A16Linear(in_features, out_features, group_size=group_size, bias=False)
    linear.load_weight(weight)

    # Check the shape of the packed weight
    expected_shape = (out_features, in_features // 2)
    assert linear.weight.shape == expected_shape, f"Expected shape {expected_shape}, but got {linear.weight.shape}"

    # Check memory savings
    original_memory = weight.numel() * weight.element_size()
    packed_memory = linear.weight.numel() * linear.weight.element_size()
    assert packed_memory < original_memory, "Packed weight should use less memory than original weight"

def test_w4a16_linear_zero_weight_yields_zero_output():
    in_features = 8
    out_features = 4
    group_size = 4
    weight = torch.zeros((out_features, in_features))
    bias = torch.rand(out_features) - 0.5

    # Create W4A16Linear layer and load weight and bias
    linear = W4A16Linear(in_features, out_features, group_size=group_size)
    linear.load_weight(weight)
    linear.bias.data.copy_(bias)

    input_tensor = torch.rand((2, in_features))
    output = linear(input_tensor)

    # The output should be equal to the bias since the weight is zero
    expected_output = bias.unsqueeze(0).expand_as(output)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

def test_w4a16_linear_forward_shape_preservation():
    in_features = 8
    out_features = 4
    group_size = 4
    weight = torch.rand((out_features, in_features)) - 0.5
    bias = torch.rand(out_features) - 0.5

    # Create W4A16Linear layer and load weight and bias
    linear = W4A16Linear(in_features, out_features, group_size=group_size)
    linear.load_weight(weight)
    linear.bias.data.copy_(bias)

    input_tensor = torch.rand((2, in_features))
    output = linear(input_tensor)

    # Check that the output shape is correct
    expected_shape = (input_tensor.shape[0], out_features)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_w4a16_linear_supports_group_sizes(group_size: int) -> None:
    """每组一个 scale；输出必须与按同一 group_size 手算的结果完全一致。"""

    in_features, out_features = 256, 8
    weight = torch.randn((out_features, in_features))
    linear = W4A16Linear(
        in_features,
        out_features,
        bias=False,
        group_size=group_size,
    )
    linear.load_weight(weight)

    # packed 权重只取决于 in_features；scale 的列数才是 group_size 的体现
    assert linear.weight.shape == (out_features, in_features // 2)
    assert linear.scale.shape == (out_features, in_features // group_size)

    # 期望值用模块实际存储的 scale dtype 计算，避免 fp16 舍入差异干扰比较
    quantized, scale = quantize_groupwise(weight, group_size=group_size)
    expected_weight = dequantize_weight(
        unpack_int4(pack_int4(quantized)),
        scale.to(linear.scale.dtype),
        group_size=group_size,
    )
    input_tensor = torch.randn((3, in_features), dtype=torch.float16)
    expected = torch.nn.functional.linear(
        input_tensor,
        expected_weight.to(input_tensor.dtype),
        None,
    )

    torch.testing.assert_close(linear(input_tensor), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("in_features", "group_size"),
    [
        (8, 3),   # group_size 必须是偶数，否则无法两两打包
        (12, 8),  # 12 不能被 8 整除
        (8, 0),   # 非正数
    ],
)
def test_w4a16_linear_rejects_invalid_group_size(
    in_features: int,
    group_size: int,
) -> None:
    with pytest.raises(ValueError):
        W4A16Linear(
            in_features,
            8,
            bias=False,
            group_size=group_size,
        )