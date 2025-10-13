import torch
from .manager_modules import LinearLayerMemoryManager, ConvLayerMemoryManager
import random

LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
    "QLinear",
]
CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
    "QConv2d",
]

UNMANAGED_MODULES = [
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNN",
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm"]


class MemoryManager:
    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device = torch.device("cpu"),
    ):
        self.module: torch.nn.Module = module
        self.process_device: torch.device = process_device
        self.unmanaged_modules: list[torch.nn.Module] = []

    def memory_managed_to(self, *args, **kwargs):
        # first move all the unmanaged modules
        for module in self.unmanaged_modules:
            module.to(*args, **kwargs)
        # check for a dtype argument
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif len(args) > 0:
            for i, arg in enumerate(args):
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            return self.module._mm_to(dtype=dtype)
        return self.module

    @classmethod
    def attach(
        cls, module: torch.nn.Module, device: torch.device, offload_percent: float = 1.0
    ):
        if hasattr(module, "_memory_manager"):
            # already attached
            return

        module._memory_manager = cls(module, device)

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        modules_processed = []
        # attach to all modules
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if (
                    child_module.__class__.__name__ in LINEAR_MODULES
                    and child_module not in modules_processed
                ):
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # linear
                        LinearLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                    modules_processed.append(child_module)
                elif (
                    child_module.__class__.__name__ in CONV_MODULES
                    and child_module not in modules_processed
                ):
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # conv
                        ConvLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                    modules_processed.append(child_module)
                elif child_module.__class__.__name__ in UNMANAGED_MODULES or any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_MODULES_INCLUDES
                ):
                    # unmanaged
                    module._memory_manager.unmanaged_modules.append(child_module)
                else:
                    continue
