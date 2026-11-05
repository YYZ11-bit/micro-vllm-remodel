from micro_vllm.layers.quantization.reference import (
    dequantize_weight,
    pack_int4,
    unpack_int4,
    quantize_groupwise,
)
import torch
from torch import nn

INT4PACK_INNER_K_TILES = 8
INT4PACK_GROUP_SIZES = frozenset({32, 64, 128, 256})
FUSED_DECODE_MAX_TOKENS = 512
_TORCH_USES_UINT8_PAIRS = tuple(
    int(part) for part in torch.__version__.split(".")[:2]
) >= (2, 5)

class W4A16Linear(nn.Module):
    def __init__(self, in_features, out_features, *, bias=True, group_size=128, kernel="reference"):
        if group_size <= 0 or group_size % 2 != 0:
            raise ValueError("group_size must be a positive even integer")
        if in_features % group_size != 0:
            raise ValueError("in_features must be divisible by group_size")
        if kernel not in {"reference", "int4pack", "auto"}:
            raise ValueError(f"unsupported kernel {kernel!r}")
        if kernel in {"int4pack", "auto"}:
            if group_size not in INT4PACK_GROUP_SIZES:
                raise ValueError(
                    f"group_size must be one of {sorted(INT4PACK_GROUP_SIZES)} "
                    f"when using kernel={kernel!r}"
                )
            if in_features % (INT4PACK_INNER_K_TILES * 16) != 0:
                raise ValueError(
                    f"in_features={in_features} must be divisible by "
                    f"{INT4PACK_INNER_K_TILES * 16} when using kernel={kernel!r}"
                )
            if out_features % 8 != 0:
                raise ValueError(
                    f"out_features={out_features} must be divisible by 8 "
                    f"when using kernel={kernel!r}"
                )
        super().__init__()
        self._int4pack_weight = None
        self._int4pack_scale_and_zero = None
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.kernel = kernel
        self.register_buffer(
            "weight",
            torch.empty((out_features, in_features // 2), dtype=torch.uint8),
        )
        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.empty(out_features, dtype=torch.float32))
            )
        else:
            self.register_parameter("bias", None)
        self.register_buffer(
            "scale",
            torch.empty(out_features, in_features // group_size, dtype=torch.float32)
        )
        # self.reset_parameters()

    def load_weight(self, weight: torch.Tensor) -> None:
        assert weight.shape == (self.out_features, self.in_features)
        weight = weight.to(torch.float32)
        quantized, scale = quantize_groupwise(weight, group_size=self.group_size)
        packed = pack_int4(quantized)
        self.weight.data.copy_(packed)
        self.scale.data.copy_(scale.to(torch.float32).view(-1, self.in_features // self.group_size))
        self._int4pack_scale_and_zero = None
        self._int4pack_weight = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_fused = self.kernel == "int4pack"
        if self.kernel == "auto":
            num_tokens = x.numel() // self.in_features
            use_fused = num_tokens <= FUSED_DECODE_MAX_TOKENS

        if not use_fused:
            unpacked = unpack_int4(self.weight)
            final_weight = dequantize_weight(
                unpacked, self.scale.view(-1), group_size=self.group_size
            )
            bias = None if self.bias is None else self.bias.to(x.dtype)
            return torch.nn.functional.linear(x, final_weight.to(x.dtype), bias)

        self._ensure_int4pack()
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).to(torch.bfloat16)
        y = torch.ops.aten._weight_int4pack_mm(
            x_2d,
            self._int4pack_weight,
            self.group_size,
            self._int4pack_scale_and_zero,
        ).to(x.dtype)
        y = y.reshape(*original_shape[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)
        return y
        
    def _ensure_int4pack(self) -> None:
        """Build the tiled int4 layout expected by _weight_int4pack_mm, once."""
        if (
            self._int4pack_weight is not None
            and self._int4pack_weight.device == self.weight.device
        ):
            return
        if self.weight.device.type != "cuda":
            raise RuntimeError(
                "kernel='int4pack' requires CUDA (compute capability >= 8.0); "
                f"got device {self.weight.device}. Use kernel='reference'."
            )
        if torch.cuda.get_device_capability(self.weight.device)[0] < 8:
            raise RuntimeError(
                "kernel='int4pack' requires compute capability >= 8.0, "
                f"got {torch.cuda.get_device_capability(self.weight.device)}"
            )

        # reference layout stores signed nibbles; the kernel wants unsigned 0..15.
        quantized = unpack_int4(self.weight)
        unsigned = (quantized.to(torch.int16) + 8).to(torch.uint8)
        if _TORCH_USES_UINT8_PAIRS:
            # 2.5+ 版本：两个数字挤进一个 uint8
            pairs = ((unsigned[:, ::2] << 4) | unsigned[:, 1::2]).to(torch.uint8).contiguous()
        else:
            # 2.4 版本：把每个数字展开成一个独立的 int32
            pairs = unsigned.to(torch.int32).contiguous()
        packed = torch.ops.aten._convert_weight_to_int4pack(
            pairs, INT4PACK_INNER_K_TILES
        )

        scales = self.scale.to(torch.float32).view(
            self.out_features, self.in_features // self.group_size
        )
        zeros = torch.zeros_like(scales)
        scale_and_zero = (
            torch.cat((scales.unsqueeze(-1), zeros.unsqueeze(-1)), dim=-1)
            .transpose(-3, -2)
            .contiguous()
            .to(torch.bfloat16)
        )
        self._int4pack_weight = packed
        self._int4pack_scale_and_zero = scale_and_zero
            
