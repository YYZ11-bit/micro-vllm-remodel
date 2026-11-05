import torch
import pytest
from micro_vllm.layers.quantization.reference import (
    dequantize_weight,
    pack_int4,  
    unpack_int4,
    quantize_groupwise,
)

def test_pack_unpack_int4():
    weight = torch.rand((3, 8)) - 0.5
    quantized, scale = quantize_groupwise(weight, group_size=4)
    packed = pack_int4(quantized)
    unpacked = unpack_int4(packed)
    assert torch.equal(quantized, unpacked)
    assert (-8 in unpacked or 7 in unpacked)

def test_quantize_error():
    group_size = 4
    weight = torch.rand((3, 8))
    quantized, scale = quantize_groupwise(weight, group_size=group_size)
    packed = pack_int4(quantized)
    unpacked = unpack_int4(packed)
    dequantized = dequantize_weight(unpacked, scale, group_size=group_size)
    per_element_scale = scale.view(weight.shape[0], -1).repeat_interleave(group_size, dim=1)
    error = torch.abs(weight - dequantized)
    assert torch.all(error < per_element_scale / 2.0)

def test_scale_equal_0():
    weight = torch.zeros((3, 8))
    quantized, scale = quantize_groupwise(weight, group_size=4)
    assert torch.all(scale == 1.0)
    packed = pack_int4(quantized)
    unpacked = unpack_int4(packed)
    dequantized = dequantize_weight(unpacked, scale, group_size=4)
    assert torch.all(dequantized == 0)

def test_invalid_group_size_1():
    weight = torch.rand((3, 7))
    with pytest.raises(ValueError):
        quantize_groupwise(weight, group_size=4)   

def test_invalid_group_size_2():
    weight = torch.rand((3, 8))
    with pytest.raises(ValueError):
        quantize_groupwise(weight, group_size=3)

def test_scale_shape():
    weight = torch.rand((3, 8))
    quantized, scale = quantize_groupwise(weight, group_size=4)
    assert scale.shape == (weight.shape[0] * weight.shape[1] // 4,)

def test_unpack_negative_extremes():
    q = torch.tensor([[-8, -7, -1, 0, 1, 6, 7, -8]], dtype=torch.int8)
    assert torch.equal(unpack_int4(pack_int4(q)), q)
