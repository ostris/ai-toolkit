"""ComfyUI-style handling for mixed-precision checkpoints (e.g. fp32
scale_shift tables next to bf16 linears in one file).

Two pieces, used together on a loaded module tree:

- attach_per_op_casting(root): every weighted module casts its floating-point
  inputs to its own weight dtype at forward (what comfy's manual-cast ops do),
  so activations promoted to fp32 by a stored-fp32 tensor drop back to the
  layer's dtype at the next op instead of erroring in a bf16 matmul.
- pin_stored_fp32(root): tensors that are fp32 after load stay fp32 through
  any parent-level dtype cast (`module.to(dtype=...)`, `.half()`, ...) —
  device moves still apply. Without this, a holder's blanket `.to(dtype)`
  would silently downcast the deliberately-fp32 pieces.
"""

import types

import torch
from torch import nn

from toolkit.util.ostris_quant import OstrisLinear


def _cast_args(args, kwargs, dtype):
    def cast(t):
        if torch.is_tensor(t) and t.is_floating_point() and t.dtype != dtype:
            return t.to(dtype)
        return t

    return tuple(cast(a) for a in args), {k: cast(v) for k, v in kwargs.items()}


def attach_per_op_casting(root: nn.Module) -> int:
    """Register forward-pre-hooks casting inputs to each weighted module's own
    weight dtype. Covers Linear/Conv/Norm-style modules that own a floating
    ``weight`` parameter, and OstrisLinear via its stored orig dtype (its
    ``weight`` property would materialize the dequantized tensor). Returns the
    number of modules hooked."""
    hooked = 0
    for module in root.modules():
        if isinstance(module, OstrisLinear):
            def hook(mod, args, kwargs):
                return _cast_args(args, kwargs, mod.ostris_orig_dtype)

        else:
            weight = module._parameters.get("weight", None)
            if weight is None or not weight.is_floating_point():
                continue

            def hook(mod, args, kwargs):
                return _cast_args(args, kwargs, mod._parameters["weight"].dtype)

        module.register_forward_pre_hook(hook, with_kwargs=True)
        hooked += 1
    return hooked


def _tag_fp32(root: nn.Module):
    for t in list(root.parameters()) + list(root.buffers()):
        if t.is_floating_point() and t.dtype == torch.float32:
            t._pin_dtype = True
            if isinstance(t, nn.Parameter):
                t.data._pin_dtype = True


def pin_stored_fp32(root: nn.Module):
    """Mark every currently-fp32 param/buffer in ``root`` and wrap the root's
    ``_apply`` so parent dtype casts skip them (device moves still apply).
    The wrapper probes the cast function with a scalar to detect whether it
    changes float dtypes, so ``.to(device)`` passes through untouched."""
    _tag_fp32(root)
    orig_apply = root._apply

    def _apply(self, fn, *args, **kwargs):
        probe = fn(torch.zeros((), dtype=torch.float32))
        if probe.dtype == torch.float32:
            return orig_apply(fn, *args, **kwargs)

        def fn_pinned(t):
            if getattr(t, "_pin_dtype", False):
                out = t if t.device == probe.device else t.to(probe.device)
                out._pin_dtype = True
                return out
            return fn(t)

        result = orig_apply(fn_pinned, *args, **kwargs)
        # _apply may rewrap tensors (dropping attributes); the invariant is
        # simple — the pinned set IS the fp32 set — so re-tag
        _tag_fp32(root)
        return result

    root._apply = types.MethodType(_apply, root)
    return root
