"""Rotary position embedding implementations."""

from collections.abc import Mapping

import torch
from torch import nn


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the final dimension."""

    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary embedding for flattened absolute positions."""

    def __init__(self, head_dim: int, base: float = 10_000.0, rope_scaling: Mapping[str, object] | None = None) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._inv_freq = _inverse_frequencies(head_dim, base, rope_scaling)   

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequency_indices = torch.arange(
            0,
            self.head_dim,
            2,
            dtype=torch.float32,
            device=positions.device,
        )
        inverse_frequencies = self._inv_freq.to(positions.device)
        frequencies = torch.outer(
            positions.to(torch.float32),
            inverse_frequencies,
        )
        angles = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
        cosine = angles.cos()
        sine = angles.sin()

        float_query = query.to(torch.float32)
        float_key = key.to(torch.float32)
        rotated_query = (
            float_query * cosine + rotate_half(float_query) * sine
        )
        rotated_key = float_key * cosine + rotate_half(float_key) * sine
        return rotated_query.to(query.dtype), rotated_key.to(key.dtype)

def _inverse_frequencies(head_dim, base, rope_scaling):
    frequency_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (frequency_indices / head_dim))
    if rope_scaling is None:
        return inv_freq
    rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or "default"
    if rope_type == "default":
        return inv_freq
    if rope_type != "llama3":
        raise NotImplementedError(...)
    # 上面的 llama3 三段式，全部用 float32
    factor = rope_scaling["factor"]
    low_freq_factor  = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]

    low_wavelen  = original_max_position_embeddings / low_freq_factor
    high_wavelen = original_max_position_embeddings / high_freq_factor

    wavelen = 2 * torch.pi / inv_freq

    # 第一段：只有“波长太长”的那批频率被压缩
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)

    # 中间过渡带的线性权重
    smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)

    smoothed = (1 - smooth) * scaled / factor + smooth * scaled

    # 第二段：只在过渡带内替换（等价于 high_wavelen <= wavelen <= low_wavelen）
    medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(medium, smoothed, scaled)
