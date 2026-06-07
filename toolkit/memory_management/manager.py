import torch
from .manager_modules import LinearLayerMemoryManager, ConvLayerMemoryManager, _DEVICE_STATE
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
    "Conv3d"
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]


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
            if isinstance(module, torch.nn.Parameter):
                # Parameter cannot move this way
                module.data = module.data.to(*args, **kwargs)
            else:
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
        cls,
        module: torch.nn.Module,
        device: torch.device,
        offload_percent: float = 1.0,
        ignore_modules: list[torch.nn.Module] = []
    ):
        if hasattr(module, "_memory_manager"):
            # already attached
            return

        module._memory_manager = cls(module, device)

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        # add ignore modules to unmanaged list
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)

        # count ignore modules as processed
        modules_processed = [x for x in ignore_modules]
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
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
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
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
                                )
                            modules_processed.append(ara)
                    modules_processed.append(child_module)
                elif child_module.__class__.__name__ in UNMANAGED_MODULES or any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_MODULES_INCLUDES
                ):
                    # unmanaged
                    module._memory_manager.unmanaged_modules.append(child_module)
                else:
                    continue

    @classmethod
    def detach(cls, module: torch.nn.Module):
        """
        Reverse of attach(). Moves unmanaged modules back to CPU, restores the
        original .to() and forward methods on all child layers, unpins CPU weight
        tensors, and clears the global CUDA device state.

        Call this before unloading/replacing a module that had attach() applied.
        """
        if not hasattr(module, "_memory_manager"):
            return

        for unmanaged in module._memory_manager.unmanaged_modules:
            try:
                if isinstance(unmanaged, torch.nn.Parameter):
                    unmanaged.data = unmanaged.data.to('cpu')
                else:
                    unmanaged.to('cpu')
            except Exception:
                pass

        if hasattr(module, "_mm_to"):
            module.to = module._mm_to
            del module._mm_to

        del module._memory_manager

        for child in module.modules():
            lmm = getattr(child, "_layer_memory_manager", None)
            if lmm is None:
                continue

            original_forward = getattr(lmm, "_original_forward", None)
            if original_forward is not None:
                if hasattr(child, "ara_lora_ref"):
                    ara = child.ara_lora_ref()
                    if ara is not None:
                        ara.org_forward = original_forward
                else:
                    child.forward = original_forward

            for param_name in ("weight", "bias"):
                param = getattr(child, param_name, None)
                if param is None or not isinstance(param, torch.nn.Parameter):
                    continue
                try:
                    if param.data.is_pinned():
                        object.__setattr__(
                            child,
                            param_name,
                            torch.nn.Parameter(
                                param.data.clone(),
                                requires_grad=param.requires_grad,
                            ),
                        )
                except Exception:
                    pass

            del child._layer_memory_manager
            if hasattr(child, "_memory_management_device"):
                del child._memory_management_device
            if hasattr(child, "_is_memory_managed"):
                del child._is_memory_managed

        keys_to_delete = [
            dev for dev in _DEVICE_STATE
            if isinstance(dev, torch.device) and dev.type == "cuda"
        ]
        for key in keys_to_delete:
            del _DEVICE_STATE[key]

        torch.cuda.empty_cache()
