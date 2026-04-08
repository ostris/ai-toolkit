from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        global_accelerator = Accelerator()
    return global_accelerator


def reset_accelerator(fsdp_plugin=None, **kwargs) -> Accelerator:
    """Recreate the global accelerator with new configuration.

    Must be called before any accelerator.prepare() calls. Used to switch
    from the default bare Accelerator to one configured with FSDP.

    The previous accelerator must not have been used for prepare() calls.
    Only read-only operations (is_main_process, device) are safe before reset.
    """
    global global_accelerator
    if global_accelerator is not None:
        # Release the old accelerator's resources. The process group is shared
        # and will be reused by the new Accelerator instance.
        del global_accelerator
    global_accelerator = Accelerator(fsdp_plugin=fsdp_plugin, **kwargs)
    return global_accelerator


def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model
