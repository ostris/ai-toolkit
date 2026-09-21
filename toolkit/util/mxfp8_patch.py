"""
Stopgap shim for mxfp8 frozen-weight training (ai-toolkit LoRA / frozen-layer).

Two bugfix-grade patches to torchao's mxfp8 training op `mx_mm` (NOT an RFC):
  ① grad_output.contiguous() before the Triton dim0 cast -> torchao's own eager
     backward otherwise crashes on `assert x.is_contiguous()`.
  ② when the weight is frozen (ctx.needs_input_grad[1] is False) skip the wgrad
     GEMM + its 2 dim1 casts -> ~1.45x faster fwd+bwd, zero behavior change
     (autograd discards grad_weight for frozen weights anyway). Idiom already in
     torchao's int8_mixed_precision.py / bitnet.py; mx_mm just omitted it.

Plus a guarded config factory so callers can feature-detect mxfp8 without breaking
on older torchao / non-Blackwell GPUs.

Designed to drop into ai-toolkit as e.g. toolkit/util/mxfp8_patch.py. Reversible,
idempotent. Apply once at import-time of the quant path.
"""
from typing import Optional, Tuple
import torch

_PATCHED = False


def apply_mxfp8_frozen_wgrad_patch() -> bool:
    """Monkeypatch torchao mx_mm.backward with ①contiguous + ②skip-wgrad-when-frozen.
    Idempotent. Returns True if patched (or already patched), False if torchao lacks mx_mm."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from torchao.prototype.moe_training import mxfp8_linear as mxl
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
        from torchao.prototype.mx_formats.config import MXFP8Dim1CastKernelChoice
        from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    except Exception:
        return False

    _orig_backward = mxl.mx_mm.backward

    def _patched_backward(ctx, grad_output_hp):
        # ② weight needs grad (trainable) -> full original backward (+ ① contiguous)
        if ctx.needs_input_grad[1]:
            return _orig_backward(ctx, grad_output_hp.contiguous())

        # frozen weight -> compute grad_input (dgrad) only, skip wgrad GEMM + casts
        _, weight_hp = ctx.saved_tensors
        bs, kp, scm = ctx.block_size, ctx.kernel_preference, ctx.scale_calculation_mode
        d0, d1 = ctx.mxfp8_dim0_cast_kernel_choice, ctx.mxfp8_dim1_cast_kernel_choice
        ge, we = ctx.grad_elem_dtype, ctx.w_elem_dtype

        sh = grad_output_hp.shape
        go_r = grad_output_hp.reshape(-1, sh[-1]).contiguous()  # ① contiguous
        go_mx0 = MXTensor.to_mx(go_r, ge, bs, scm, kp, mxfp8_dim0_cast_kernel_choice=d0)

        if kp == KernelPreference.EMULATED or d1 == MXFP8Dim1CastKernelChoice.TORCH:
            w_mx1 = MXTensor.to_mx(
                weight_hp.t().contiguous(), we, bs,
                kernel_preference=kp, scaling_mode=scm,
                mxfp8_dim0_cast_kernel_choice=d0,
            )
        else:
            w_mx1 = _to_mxfp8_dim1_kernel_wrapper(
                weight_hp, bs, we, weight_hp.dtype, kp, d1, scm
            )

        grad_input = torch.mm(go_mx0, w_mx1.t()).reshape(*sh[:-1], -1)
        # forward sig has 11 inputs -> 11 grads; only grad_input is non-None
        return (grad_input, None) + (None,) * 9

    mxl.mx_mm.backward = staticmethod(_patched_backward)
    mxl._mxfp8_frozen_patch_orig_backward = _orig_backward  # keep for revert
    _PATCHED = True
    return True


def revert_mxfp8_frozen_wgrad_patch() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    from torchao.prototype.moe_training import mxfp8_linear as mxl
    mxl.mx_mm.backward = staticmethod(mxl._mxfp8_frozen_patch_orig_backward)
    _PATCHED = False


def mxfp8_config_or_none(recipe: str = "MXFP8_RCEIL") -> Tuple[Optional[object], Optional[str]]:
    """Guarded factory. Returns (config, None) if mxfp8 training is usable here,
    else (None, reason). Backward-compatible: never raises on old torchao / old HW."""
    # ① API/version guard (lazy import)
    try:
        from torchao.prototype.moe_training.config import (
            MXFP8TrainingOpConfig, MXFP8TrainingRecipe,
        )
    except Exception as e:
        return None, f"torchao lacks MXFP8TrainingOpConfig ({type(e).__name__})"
    # ② hardware guard
    if not torch.cuda.is_available():
        return None, "no CUDA device"
    if torch.cuda.get_device_capability() < (10, 0):
        cap = torch.cuda.get_device_capability()
        return None, f"mxfp8 needs sm_100+ (Blackwell); got sm_{cap[0]}{cap[1]}"
    # ③ native kernel guard (the dim1 CUDA cast op must be registered/loadable).
    # torchao>=0.19 defines the op lazily in mx_formats.kernels (gated there on
    # sm_100+ and CUDA>=12.8); older torchao registered it at `import torchao`.
    if not hasattr(torch.ops.torchao, "mxfp8_quantize"):
        try:
            import torchao.prototype.mx_formats.kernels  # noqa: F401
        except Exception:
            pass
    if not hasattr(torch.ops.torchao, "mxfp8_quantize"):
        return None, "native mxfp8 kernel not built (need source-built torchao)"
    try:
        cfg = MXFP8TrainingOpConfig.from_recipe(getattr(MXFP8TrainingRecipe, recipe))
    except Exception as e:
        return None, f"recipe {recipe} unavailable ({type(e).__name__})"
    return cfg, None


# ===================== static frozen-weight mxfp8 (memory win for LoRA) =====================
# The dynamic MXFP8TrainingOpConfig keeps the bf16 master weight (and caches an fp8 copy),
# so a FROZEN LoRA base costs >= bf16 -- the opposite of what LoRA wants. For a base that is
# never updated we can store ONLY the fp8 cast(s) and drop the bf16 master:
#   store_dim1=False -> fp8 dim0 only  (~0.5x bf16 weight memory); dgrad via dequant->bf16 matmul
#   store_dim1=True  -> fp8 dim0+dim1  (~1.0x bf16, == parity);    all-fp8 dgrad (fastest)
# Both keep the module class nn.Linear (LoRA-safe: only the weight tensor is swapped, in place),
# do fp8 fprop, and skip wgrad. The fp8 cast is computed ONCE (static weight), so accuracy is
# identical to the dynamic path (which re-casts the same unchanged weight every step).
# NOTE: cast kernels are CUDA/Triton -> swap while the weight is on cuda. Cross-device .to(cpu)
# degrades to a dequantized bf16 tensor (offload not optimized; use low_vram=false).

_STATIC_CASTS = None


def _move_mxtensor(mxt, device):
    """Move an MXTensor (or plain tensor) across devices via its flattened components.
    MXTensor.to(cpu) trips torchao's storage-aliasing check, so move qdata+scale directly
    and rebuild -> keeps the fp8 representation (no dequantize) on either device."""
    if not hasattr(mxt, "__tensor_flatten__"):
        return mxt.to(device)
    names, meta = mxt.__tensor_flatten__()
    inner = {n: getattr(mxt, n).to(device) for n in names}
    return type(mxt).__tensor_unflatten__(inner, meta, mxt.size(), mxt.stride())


def _static_casts():
    """Lazy (cast_dim0, cast_dim1) using the torchao Triton/CUDA mxfp8 kernels.

    torchao>=0.19 routes MXTensor mm through torch's F.scaled_mm v2, which on
    some GPUs (sm_120 consumer Blackwell) only accepts SWIZZLE_32_4_4 scales.
    Probe once with the fast unswizzled casts (Triton dim0 / native CUDA dim1)
    and fall back to swizzled TORCH casts when the unswizzled mm is rejected."""
    global _STATIC_CASTS
    if _STATIC_CASTS is not None:
        return _STATIC_CASTS
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
    from torchao.prototype.mx_formats.config import (
        MXFP8Dim0CastKernelChoice, MXFP8Dim1CastKernelChoice, ScaleCalculationMode,
    )
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    e4m3 = torch.float8_e4m3fn
    BS = 32
    KP = KernelPreference.AUTO
    SM = ScaleCalculationMode.RCEIL

    def make_casts(swizzled: bool):
        if swizzled:
            T0 = MXFP8Dim0CastKernelChoice.TORCH

            def cast_dim0(t):  # -> swizzled MXTensor (TORCH cast + to_blocked scales)
                return MXTensor.to_mx(t.contiguous(), e4m3, BS, SM, KP,
                                      is_swizzled_scales=True,
                                      mxfp8_dim0_cast_kernel_choice=T0)

            def cast_dim1(W):  # -> swizzled MXTensor shaped like W.t() (blocks along N)
                return MXTensor.to_mx(W.t().contiguous(), e4m3, BS, SM, KP,
                                      is_swizzled_scales=True,
                                      mxfp8_dim0_cast_kernel_choice=T0)
        else:
            T0 = MXFP8Dim0CastKernelChoice.TRITON
            C1 = MXFP8Dim1CastKernelChoice.CUDA

            def cast_dim0(t):  # activation/dim0 cast -> MXTensor (fast Triton kernel)
                return MXTensor.to_mx(t.contiguous(), e4m3, BS, SM, KP, mxfp8_dim0_cast_kernel_choice=T0)

            def cast_dim1(W):  # weight dim1 cast -> MXTensor (native CUDA kernel)
                return _to_mxfp8_dim1_kernel_wrapper(W, BS, e4m3, W.dtype, KP, C1, SM)

        return cast_dim0, cast_dim1

    casts = make_casts(swizzled=False)
    if torch.cuda.is_available():
        try:
            # probe whether an unswizzled MX mm is accepted on this torch/GPU
            _a = casts[0](torch.randn(128, 64, device="cuda", dtype=torch.bfloat16))
            _b = casts[1](torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
            torch.mm(_a, _b.t())
        except Exception:
            casts = make_casts(swizzled=True)
    _STATIC_CASTS = casts
    return _STATIC_CASTS


class _StaticMXFp8LinearFn(torch.autograd.Function):
    """fp8 fprop with a pre-quantized static weight; dgrad only (frozen -> no wgrad)."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        cast0, _ = _static_casts()
        xr = x.reshape(-1, x.shape[-1])
        out = torch.mm(cast0(xr), weight._w_dim0.t())          # fp8 _scaled_mm
        return out.reshape(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, gy):
        w = ctx.weight
        cast0, _ = _static_casts()
        gyr = gy.reshape(-1, gy.shape[-1])
        if w._w_dim1 is not None:
            gx = torch.mm(cast0(gyr), w._w_dim1.t())            # all-fp8 dgrad
        else:
            gx = gyr.contiguous() @ w._w_dim0.dequantize().to(gyr.dtype)  # bf16 dgrad (low-mem)
        return gx.reshape(*gy.shape[:-1], gx.shape[-1]), None


class MXFp8StaticFrozenWeight(torch.Tensor):
    """Weight tensor holding only the fp8 cast(s) of a frozen weight (no bf16 master).

    Dispatches F.linear to a custom autograd Function (fp8 fprop + dgrad, no wgrad);
    supports .dequantize() for save; cross-device move degrades to dequantized bf16."""

    @staticmethod
    def __new__(cls, w_dim0, w_dim1, shape, dtype, device):
        return torch.Tensor._make_wrapper_subclass(
            cls, shape, dtype=dtype, device=device, requires_grad=False)

    def __init__(self, w_dim0, w_dim1, shape, dtype, device):
        self._w_dim0 = w_dim0
        self._w_dim1 = w_dim1
        self._orig_dtype = dtype

    def __repr__(self):
        return f"MXFp8StaticFrozenWeight({tuple(self.shape)}, dim1={'y' if self._w_dim1 is not None else 'n'})"

    def dequantize(self):
        return self._w_dim0.dequantize().to(self._orig_dtype)

    def __tensor_flatten__(self):
        names = ["_w_dim0"] + (["_w_dim1"] if self._w_dim1 is not None else [])
        return names, {"shape": tuple(self.shape), "dtype": self._orig_dtype}

    @staticmethod
    def __tensor_unflatten__(inner, meta, sz, st):
        return MXFp8StaticFrozenWeight(
            inner["_w_dim0"], inner.get("_w_dim1"),
            torch.Size(meta["shape"]), meta["dtype"], inner["_w_dim0"].device)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            x, weight = args[0], args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            out = _StaticMXFp8LinearFn.apply(x, weight)
            return out if bias is None else out + bias
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        aten = torch.ops.aten
        if func in (aten.detach.default, aten.alias.default, aten.clone.default):
            s = args[0]
            return cls(s._w_dim0, s._w_dim1, s.shape, s._orig_dtype, s._w_dim0.device)
        if func is aten._to_copy.default:
            s = args[0]
            tgt = kwargs.get("device")
            if tgt is None:
                return cls(s._w_dim0, s._w_dim1, s.shape, s._orig_dtype, s._w_dim0.device)
            d = torch.device(tgt)
            # move the fp8 components (keeps the static representation on either device)
            d0 = _move_mxtensor(s._w_dim0, d)
            d1 = _move_mxtensor(s._w_dim1, d) if s._w_dim1 is not None else None
            return cls(d0, d1, s.shape, s._orig_dtype, d)
        def uw(a):
            return a.dequantize() if isinstance(a, cls) else a
        return func(*[uw(a) for a in args], **{k: uw(v) for k, v in kwargs.items()})


def make_mxfp8_static_frozen_weight(weight: torch.Tensor, store_dim1: bool = False) -> "MXFp8StaticFrozenWeight":
    """Build a static fp8 weight from a (cuda, floating) bf16/fp16 weight tensor."""
    cast0, cast1 = _static_casts()
    W = weight.detach()
    return MXFp8StaticFrozenWeight(
        cast0(W), cast1(W) if store_dim1 else None, W.shape, weight.dtype, W.device)


def mxfp8_static_swap_linear(module, store_dim1: bool = False) -> bool:
    """Swap a single nn.Linear's frozen weight to the static fp8 representation, in place.

    Accepts a plain bf16 weight OR an already-dynamic MXFP8TrainingWeightWrapperTensor
    (whose bf16 master `_data` is extracted). No-op (returns False) unless `module` is an
    nn.Linear whose weight is a cuda floating tensor. Module class is unchanged (LoRA-safe)."""
    import torch.nn as nn
    if not isinstance(module, nn.Linear):
        return False
    w = module.weight
    if isinstance(w, MXFp8StaticFrozenWeight):
        return False  # already static
    src = getattr(w, "_data", w)  # dynamic wrapper keeps the bf16 master in _data
    if (not torch.is_tensor(src)) or src.device.type != "cuda" or src.dtype not in (
        torch.bfloat16, torch.float16, torch.float32,
    ):
        return False
    static = make_mxfp8_static_frozen_weight(src, store_dim1=store_dim1)
    module.weight = nn.Parameter(static, requires_grad=False)
    return True


def apply_mxfp8_static_frozen_weight(root, store_dim1: bool = False) -> int:
    """Walk `root` and static-swap every eligible (cuda, frozen) nn.Linear weight.
    Returns the number of modules swapped."""
    n = 0
    for m in root.modules():
        if mxfp8_static_swap_linear(m, store_dim1=store_dim1):
            n += 1
    return n


# ===================== no-op move guard (compile + sampling) =====================
# A static-mxfp8 weight is a traceable wrapper subclass, so nn.Module._apply forces the
# swap_tensors path on ANY `.to()`. Once torch.compile attaches a weakref to the param,
# swap_tensors refuses to swap it ("_apply(): Couldn't swap Linear.weight"). Training never
# moves the unet, but sampling does several redundant no-op `.to(device)` moves (set device
# state, pipeline.to, teardown) -> crash. A tensor-level `.to` override CANNOT help (the swap
# happens above us, in _apply, on the OLD param). So we override `.to` on the MODULE: skip a
# move whose target == the current device/dtype for every param (a true no-op). A GENUINE move
# (e.g. cpu offload) still runs normally -- so pre-compile quantize-time offload is unaffected;
# offload AFTER compile is unsupported (the weakref'd subclass param can't be swapped) and is
# irrelevant in the low_vram=false speed path. Install on BOTH the OptimizedModule and its
# _orig_mod, since the pipeline uses the unwrapped inner module. Idempotent per instance.
import weakref as _weakref
_NOOP_GUARDED = _weakref.WeakSet()


def install_noop_move_guard(module) -> None:
    if module is None or module in _NOOP_GUARDED:
        return

    orig_to = module.to

    def _already_on(device, dtype) -> bool:
        tgt = device if isinstance(device, torch.device) else torch.device(device)
        for p in module.parameters():
            if p.device.type != tgt.type or (tgt.index is not None and p.device.index != tgt.index):
                return False
            if dtype is not None and p.is_floating_point() and p.dtype != dtype:
                return False
        return True

    def guarded_to(*args, **kwargs):
        device, dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and _already_on(device, dtype):
            return module          # redundant move -> skip (would crash _apply/swap_tensors)
        return orig_to(*args, **kwargs)

    module.to = guarded_to
    _NOOP_GUARDED.add(module)


def install_noop_move_guard_recursive(unet) -> None:
    """Install the guard on a (possibly torch.compiled) unet and its inner _orig_mod."""
    install_noop_move_guard(unet)
    install_noop_move_guard(getattr(unet, "_orig_mod", None))
