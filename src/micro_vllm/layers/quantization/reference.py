import torch

def quantize_groupwise(weight: torch.Tensor, group_size: int):
    if group_size % 2 != 0:
        raise ValueError(f"group_size must be even, got {group_size}.")
    if(weight.shape[-1] % group_size != 0):
        raise ValueError(f"Last dimension of weight must be divisible by group_size {group_size}.")
    rows, cols = weight.shape
    seg_weight = weight.view(-1, group_size)
    max_abs = seg_weight.abs().max(dim=1, keepdim=True).values
    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / 7.0)
    quantized = torch.round(seg_weight / scale).clamp(-8, 7).to(torch.int8)
    return quantized.view(rows, cols), scale.squeeze(1)

def pack_int4(quantized: torch.Tensor):
    assert quantized.dtype == torch.int8
    assert quantized.ndim == 2
    assert quantized.shape[1] % 2 == 0, "Last dimension of quantized must be even."
    packed = ((quantized[:, ::2] & 0x0F).to(torch.uint8) << 4) | (quantized[:, 1::2] & 0x0F).to(torch.uint8)
    return packed

def unpack_int4(packed: torch.Tensor):
    assert packed.dtype == torch.uint8
    assert packed.ndim == 2
    unpacked = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = ((packed >> 4).to(torch.int16) ^ 8) - 8
    unpacked[:, 1::2] = ((packed & 0x0F).to(torch.int16) ^ 8) - 8
    return unpacked

def dequantize_weight(quantized: torch.Tensor, scale: torch.Tensor, group_size: int):
    assert quantized.ndim == 2
    assert scale.ndim == 1
    seg_weight = quantized.view(-1, group_size)
    assert quantized.numel() == scale.numel() * group_size
    dequantized = seg_weight.to(torch.float32) * scale.unsqueeze(1)
    return dequantized.view(quantized.shape)