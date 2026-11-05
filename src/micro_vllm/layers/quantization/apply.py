from micro_vllm.config import QuantizationConfig
from micro_vllm.layers.linear import Linear
from micro_vllm.layers.quantization.w4a16 import W4A16Linear
import torch.nn as nn

def apply_quantization(model: nn.Module, config: QuantizationConfig) -> None:
    if config.method == "none":
        return
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if type(module) is Linear
    ]
    for name, leaf in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        if leaf.in_features % config.group_size != 0:
            raise ValueError(
                f"Linear layer {name} has in_features={leaf.in_features}, "
                f"which is not divisible by group_size={config.group_size}. "
            )
        replacement = W4A16Linear(
            in_features=leaf.in_features,
            out_features=leaf.out_features,
            bias=leaf.bias is not None,
            group_size=config.group_size,
            kernel=config.kernel,
        )
        setattr(parent, attr, replacement)