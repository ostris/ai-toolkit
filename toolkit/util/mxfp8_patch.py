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

    torchao>=0.19 routes MXTensor mm through torch's F.scaled_mm v2, which
    unconditionally requires SWIZZLE_32_4_4 scales on NVIDIA (aten
    _scaled_mxfp8_mxfp8; only ROCm gfx950 has a NO_SWIZZLE variant). torchao's
    inference workflows adapted (is_swizzled_scales=True) but the training path
    never did. Probe the available paths once, fastest first:
      1. unswizzled: Triton dim0 cast + native CUDA dim1 cast (torchao<=0.18)
      2. swizzled:   Triton cast + to_blocked scale rearrange (fast eager)
      3. swizzled:   TORCH cast (to_mx swizzles internally; slow eager, but
                     torch.compile fuses it back to parity)
    """
    global _STATIC_CASTS
    if _STATIC_CASTS is not None:
        return _STATIC_CASTS
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.prototype.mx_formats.utils import (
        _to_mxfp8_dim1_kernel_wrapper,
        hp_data_dims_to_swizzled_scale_dims_mx,
        to_blocked,
    )
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    from torchao.prototype.mx_formats.config import (
        MXFP8Dim0CastKernelChoice, MXFP8Dim1CastKernelChoice, ScaleCalculationMode,
    )
    e4m3 = torch.float8_e4m3fn
    BS = 32
    KP = KernelPreference.AUTO
    SM = ScaleCalculationMode.RCEIL

    def _triton_mx(t):  # fast cast, row-major scales
        return MXTensor.to_mx(t.contiguous(), e4m3, BS, SM, KP,
                              mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON)

    def _with_swizzled_scales(m, M, K):
        # keep the fast-cast qdata/scale values; only rearrange the scale layout
        sc = to_blocked(m.scale.view(M, K // BS))
        sM, sK = hp_data_dims_to_swizzled_scale_dims_mx(M, K)
        return MXTensor(m.qdata, sc.view(sM, sK), m.elem_dtype, m.block_size,
                        m.orig_dtype, m.kernel_preference, m.act_quant_kwargs, True)

    def make_casts(mode):
        if mode == "unswizzled":
            C1 = MXFP8Dim1CastKernelChoice.CUDA

            def cast_dim0(t):  # activation/dim0 cast -> MXTensor (fast Triton kernel)
                return _triton_mx(t)

            def cast_dim1(W):  # weight dim1 cast -> MXTensor (native CUDA kernel)
                return _to_mxfp8_dim1_kernel_wrapper(W, BS, e4m3, W.dtype, KP, C1, SM)
        elif mode == "swizzled_fast":

            def cast_dim0(t):
                return _with_swizzled_scales(_triton_mx(t), t.shape[0], t.shape[-1])

            def cast_dim1(W):
                Wt = W.t().contiguous()
                return _with_swizzled_scales(_triton_mx(Wt), Wt.shape[0], Wt.shape[1])
        else:  # swizzled_torch
            T0 = MXFP8Dim0CastKernelChoice.TORCH

            def cast_dim0(t):  # TORCH cast + to_blocked swizzle inside to_mx
                return MXTensor.to_mx(t.contiguous(), e4m3, BS, SM, KP,
                                      is_swizzled_scales=True,
                                      mxfp8_dim0_cast_kernel_choice=T0)

            def cast_dim1(W):
                return MXTensor.to_mx(W.t().contiguous(), e4m3, BS, SM, KP,
                                      is_swizzled_scales=True,
                                      mxfp8_dim0_cast_kernel_choice=T0)

        return cast_dim0, cast_dim1

    casts = make_casts("unswizzled")
    if torch.cuda.is_available():
        def _probe(c):
            _a = c[0](torch.randn(128, 64, device="cuda", dtype=torch.bfloat16))
            _b = c[1](torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
            torch.mm(_a, _b.t())

        # The probe MUST run on real tensors: if the first caller is a torch.compile
        # AOT trace (fake tensors), fake-mode mm does not enforce the swizzle
        # requirement and the probe would wrongly pick "unswizzled" -> downstream
        # is_swizzled_scales mismatch (assert in MXTensor.mm).
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        with unset_fake_temporarily():
            for mode in ("unswizzled", "swizzled_fast", "swizzled_torch"):
                try:
                    c = make_casts(mode)
                    _probe(c)
                    casts = c
                    break
                except Exception:
                    continue
    _STATIC_CASTS = casts
    return _STATIC_CASTS


# ===================== runtime-transpose mxfp8 (32x32 cast) =====================
# sm_90/sm_120 cuBLAS block-scaled FP8 GEMM is TN-only, so an all-fp8 dgrad normally needs a
# second, transposed fp8 copy of the weight (store_dim1=True, ~1.0x bf16). torchao#4777's
# standalone 32x32 Triton cast emits ONE qdata plus BOTH swizzled scale frames (dim-K and
# dim-M) from a single square-block quantization: the block scale is shared, so the dim-M
# frame can reuse the same qdata transposed. At dgrad time we materialize qdata.t()
# (~40us under torch.compile's fused transpose, ~220us eager for 3072x3072) and run cuBLAS
# TN with the dim-M frame -> all-fp8 dgrad at ~0.53x bf16 weight memory.

_RT_OP = "triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale"

_NN_OK = None


def _nn_mm_op():
    """torch._scaled_mm NN (row-major mat2) wrapped as a custom op. torch 2.14's META
    registration for _scaled_mm still enforces mat2 col-major for block-scaled recipes
    even though cuBLAS >= 13.6 accepts NN at runtime - so under torch.compile the NN call
    must hide behind an opaque op whose fake impl skips the stale stride check."""
    if not hasattr(torch.ops.mxfp8_patch, "nn_mm"):
        from torch.library import custom_op

        @custom_op("mxfp8_patch::nn_mm", mutates_args=(), device_types="cuda")
        def _nn_mm(a: torch.Tensor, b: torch.Tensor,
                   scale_a: torch.Tensor, scale_b: torch.Tensor,
                   out_dtype: torch.dtype) -> torch.Tensor:
            return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)

        @_nn_mm.register_fake
        def _(a, b, scale_a, scale_b, out_dtype):
            return a.new_empty(a.shape[0], b.shape[1], dtype=out_dtype)

    return torch.ops.mxfp8_patch.nn_mm


def _nn_blockscaled_ok() -> bool:
    """Probe once whether this cuBLAS accepts a row-major mat2 for block-scaled fp8
    (_scaled_mm "NN"). cuBLAS < 13.6 (torch cu130 wheels ship 13.1) is TN-only on
    sm_90/sm_120; 13.6 (CUDA 13.3 Update 1) lifts the restriction, which makes the
    runtime transpose unnecessary (and is even faster than TN). Cached."""
    global _NN_OK
    if _NN_OK is None:
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        try:
            # probe on real tensors even when first called from an AOT fake-tensor trace
            with unset_fake_temporarily():
                from torchao.prototype.mx_formats.utils import to_blocked
                a = torch.zeros(128, 64, device="cuda").to(torch.float8_e4m3fn)
                b = torch.zeros(64, 128, device="cuda").to(torch.float8_e4m3fn)
                sa = to_blocked(torch.ones(128, 2, device="cuda").to(torch.float8_e8m0fnu))
                sb = to_blocked(torch.ones(128, 2, device="cuda").to(torch.float8_e8m0fnu))
                torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            _NN_OK = True
        except Exception:
            _NN_OK = False
    return _NN_OK


def _rt_available() -> bool:
    try:
        import torchao.prototype.mx_formats.kernels  # noqa: F401
    except Exception:
        pass
    return hasattr(torch.ops.torchao, _RT_OP)


def _rt_cast(W):
    """32x32-cast a contiguous 2D weight -> (w_dim0 MXTensor, dim-M swizzled scale frame).
    The returned frame pairs with w_dim0.qdata.t().contiguous() for the dgrad TN gemm."""
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.prototype.mx_formats.utils import hp_data_dims_to_swizzled_scale_dims_mx
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    N, K = W.shape
    if N % 32 or K % 32:
        raise ValueError(f"mxfp8_static_rt needs weight dims % 32 == 0, got {tuple(W.shape)}")
    qk, sk, sm = getattr(torch.ops.torchao, _RT_OP)(W.contiguous())
    sk = sk.reshape(*hp_data_dims_to_swizzled_scale_dims_mx(N, K))
    sm = sm.reshape(*hp_data_dims_to_swizzled_scale_dims_mx(K, N))
    w_dim0 = MXTensor(qk, sk, torch.float8_e4m3fn, 32, W.dtype, KernelPreference.AUTO, None, True)
    return w_dim0, sm


def make_mxfp8_static_rt_weight(weight: torch.Tensor) -> "MXFp8StaticFrozenWeight":
    """Build a static fp8 weight using the 32x32 runtime-transpose representation."""
    w_dim0, sm = _rt_cast(weight.detach())
    return MXFp8StaticFrozenWeight(
        w_dim0, None, weight.shape, weight.dtype, weight.device, sf_rt=sm)


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
        if w._sf_rt is not None:
            gy_mx = cast0(gyr)
            if _nn_blockscaled_ok():
                # cuBLAS >= 13.6: block-scaled NN accepted -> zero-copy dgrad,
                # mat2 = the stored row-major qdata directly (+ dim-M scale frame).
                # Via a custom op so torch.compile's stale TN-only meta check is bypassed.
                out_dt = gyr.dtype if gyr.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
                gx = _nn_mm_op()(gy_mx.qdata, w._w_dim0.qdata,
                                 gy_mx.scale, w._sf_rt, out_dt)
            else:
                # runtime transpose: materialize qdata.t() once per dgrad (fused under
                # torch.compile), reuse the stored dim-M scale frame -> TN gemm.
                from torchao.prototype.mx_formats.mx_tensor import MXTensor
                wt = MXTensor(w._w_dim0.qdata.t().contiguous(), w._sf_rt,
                              torch.float8_e4m3fn, 32, w._orig_dtype,
                              w._w_dim0.kernel_preference, None, True)
                gx = torch.mm(gy_mx, wt.t())
        elif w._w_dim1 is not None:
            gx = torch.mm(cast0(gyr), w._w_dim1.t())            # all-fp8 dgrad
        else:
            gx = gyr.contiguous() @ w._w_dim0.dequantize().to(gyr.dtype)  # bf16 dgrad (low-mem)
        return gx.reshape(*gy.shape[:-1], gx.shape[-1]), None


class MXFp8StaticFrozenWeight(torch.Tensor):
    """Weight tensor holding only the fp8 cast(s) of a frozen weight (no bf16 master).

    Dispatches F.linear to a custom autograd Function (fp8 fprop + dgrad, no wgrad);
    supports .dequantize() for save; cross-device move degrades to dequantized bf16."""

    @staticmethod
    def __new__(cls, w_dim0, w_dim1, shape, dtype, device, sf_rt=None):
        return torch.Tensor._make_wrapper_subclass(
            cls, shape, dtype=dtype, device=device, requires_grad=False)

    def __init__(self, w_dim0, w_dim1, shape, dtype, device, sf_rt=None):
        self._w_dim0 = w_dim0
        self._w_dim1 = w_dim1
        self._sf_rt = sf_rt  # dim-M swizzled scale frame for runtime-transpose dgrad
        self._orig_dtype = dtype

    def __repr__(self):
        mode = "rt" if self._sf_rt is not None else ("d1" if self._w_dim1 is not None else "d0")
        return f"MXFp8StaticFrozenWeight({tuple(self.shape)}, {mode})"

    def dequantize(self):
        return self._w_dim0.dequantize().to(self._orig_dtype)

    def __tensor_flatten__(self):
        names = ["_w_dim0"]
        names += ["_w_dim1"] if self._w_dim1 is not None else []
        names += ["_sf_rt"] if self._sf_rt is not None else []
        return names, {"shape": tuple(self.shape), "dtype": self._orig_dtype}

    @staticmethod
    def __tensor_unflatten__(inner, meta, sz, st):
        return MXFp8StaticFrozenWeight(
            inner["_w_dim0"], inner.get("_w_dim1"),
            torch.Size(meta["shape"]), meta["dtype"], inner["_w_dim0"].device,
            sf_rt=inner.get("_sf_rt"))

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
        c10d = torch.ops._c10d_functional
        # FSDP2 / collectives (torchao MXTensor pattern, but gathering the raw
        # bytes ourselves: torchao's MXTensor-level dispatch segfaults on gloo).
        # Ships compressed bytes (qdata uint8 + scale frames), never a
        # high-precision master -> ~2x less comm volume than a bf16 gather.
        if func in (c10d.all_gather_into_tensor.default, c10d.wait_tensor.default):
            from torchao.prototype.mx_formats.mx_tensor import MXTensor

            def run_comp(mxt):
                if mxt is None:
                    return None
                qd = func(mxt.qdata.view(torch.uint8), *args[1:], **kwargs)
                sc = func(mxt.scale.view(torch.uint8), *args[1:], **kwargs)
                return MXTensor(qd.view(mxt.elem_dtype), sc.view(torch.float8_e8m0fnu),
                                mxt.elem_dtype, mxt.block_size, mxt.orig_dtype,
                                mxt.kernel_preference, mxt.act_quant_kwargs,
                                mxt.is_swizzled_scales)

            s = args[0]

            def run_sf(sf):
                if sf is None:
                    return None
                # e8m0 scale frame: gather as raw bytes (collectives reject fp8 dtypes)
                return func(sf.view(torch.uint8), *args[1:], **kwargs).view(sf.dtype)

            g = {"_w_dim0": run_comp(s._w_dim0),
                 "_w_dim1": run_comp(s._w_dim1),
                 "_sf_rt": run_sf(s._sf_rt)}
            if func is c10d.wait_tensor.default:
                meta = {"shape": tuple(s.shape), "dtype": s._orig_dtype}
            else:
                world = g["_w_dim0"].shape[0] // s._w_dim0.shape[0]
                meta = {"shape": (s.shape[0] * world, *s.shape[1:]), "dtype": s._orig_dtype}
            return cls.__tensor_unflatten__(g, meta, None, None)
        if func in (aten.detach.default, aten.alias.default, aten.clone.default):
            s = args[0]
            return cls(s._w_dim0, s._w_dim1, s.shape, s._orig_dtype, s._w_dim0.device, sf_rt=s._sf_rt)
        if func is aten._to_copy.default:
            s = args[0]
            tgt = kwargs.get("device")
            if tgt is None:
                return cls(s._w_dim0, s._w_dim1, s.shape, s._orig_dtype, s._w_dim0.device, sf_rt=s._sf_rt)
            d = torch.device(tgt)
            # move the fp8 components (keeps the static representation on either device)
            d0 = _move_mxtensor(s._w_dim0, d)
            d1 = _move_mxtensor(s._w_dim1, d) if s._w_dim1 is not None else None
            sf = s._sf_rt.to(d) if s._sf_rt is not None else None
            return cls(d0, d1, s.shape, s._orig_dtype, d, sf_rt=sf)
        def uw(a):
            return a.dequantize() if isinstance(a, cls) else a
        return func(*[uw(a) for a in args], **{k: uw(v) for k, v in kwargs.items()})

def make_mxfp8_static_frozen_weight(weight: torch.Tensor, store_dim1: bool = False,
                                    runtime_transpose: bool = False) -> "MXFp8StaticFrozenWeight":
    """Build a static fp8 weight from a (cuda, floating) bf16/fp16 weight tensor."""
    if runtime_transpose:
        try:
            return make_mxfp8_static_rt_weight(weight)
        except Exception as e:
            print(f"[mxfp8_patch] runtime-transpose cast failed ({e}); "
                  f"falling back to bf16 dgrad for {tuple(weight.shape)}")
    cast0, cast1 = _static_casts()
    W = weight.detach()
    return MXFp8StaticFrozenWeight(
        cast0(W), cast1(W) if store_dim1 else None, W.shape, weight.dtype, W.device)


def mxfp8_static_swap_linear(module, store_dim1: bool = False, runtime_transpose: bool = False) -> bool:
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
    static = make_mxfp8_static_frozen_weight(
        src, store_dim1=store_dim1, runtime_transpose=runtime_transpose)
    module.weight = nn.Parameter(static, requires_grad=False)
    return True


def apply_mxfp8_static_frozen_weight(root, store_dim1: bool = False,
                                     runtime_transpose: bool = False) -> int:
    """Walk `root` and static-swap every eligible (cuda, frozen) nn.Linear weight.
    Returns the number of modules swapped."""
    n = 0
    for m in root.modules():
        if mxfp8_static_swap_linear(m, store_dim1=store_dim1, runtime_transpose=runtime_transpose):
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
