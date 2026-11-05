from micro_vllm.layers.quantization.w4a16 import W4A16Linear
import torch
from torch import nn

def load_checkpoint_tensor(
    model: nn.Module,
    parameter_name: str,
    tensor: torch.Tensor,
) -> bool:
    """Route one checkpoint tensor; return True when a quantized layer consumed it."""
    module_name, separator, leaf = parameter_name.rpartition(".")
    if separator and leaf == "weight":
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            return False   
        if isinstance(module, W4A16Linear):
            module.load_weight(tensor)
            return True
    return False
