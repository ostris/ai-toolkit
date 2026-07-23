"""Krea 2 (K2) single-stream MMDiT backbone.

Vendored from the reference ``mmdit.py`` for ai-toolkit. This is a single-stream
MMDiT: Qwen3-VL text features are fused by a small ``TextFusionTransformer`` and
then concatenated with the patchified image latent tokens into one sequence that
flows through ``SingleStreamBlock`` layers. The model predicts the flow-matching
velocity on the image tokens.

Differences from the reference (all training-driven, numerically equivalent):
  - ``torch.compile`` decorators are dropped (they fight gradient checkpointing,
    LoRA module swapping and variable shapes during training).
  - Attention uses a plain ``F.scaled_dot_product_attention`` instead of forcing
    the cuDNN SDPA backend, so it works across dtypes / masks / backward.
  - ``enable_gradient_checkpointing`` / ``disable_gradient_checkpointing`` and a
    per-block ``torch.utils.checkpoint`` wrapper are added (gated on
    ``torch.is_grad_enabled()`` so eval/sampling never pays for it).
  - ``patch`` renamed to ``patch_size`` (diffusers-style name, matching the
    toolkit's other archs) so the generic trainer's timestep-shift setup can
    discover the token patch size via ``unet.config.patch_size``.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
) -> Tensor:
    # cuDNN attention is NVIDIA-only, so hardcoding SDPBackend.CUDNN_ATTENTION
    # raises "No available kernel" on non-NVIDIA backends (AMD ROCm, Intel XPU,
    # Apple MPS). Pass a priority list instead: cuDNN is still preferred on
    # NVIDIA, and the dispatcher falls back to flash/efficient/math elsewhere.
    # (On ROCm gfx11xx the flash path needs
    # TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1; masked attention uses math.)
    with sdpa_kernel(
        [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ],
        set_priority=True,
    ):
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(x, "B H L D -> B L (H D)")


def _mask(mask: Tensor) -> Tensor:
    """Expand a (B, L) key-padding mask into a (B, 1, L, L) attention mask."""
    return mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(period)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    # t: (B,) -> args: (B, 1, half), so the embedding broadcasts as a per-sample vec.
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch_size: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20


class SimpleModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    # vec (b d)
    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))

    # vec (b (6 d))
    def forward(self, vec: Tensor):
        out = vec + self.lin
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(
            6, dim=-1
        )
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims  # how to split the head dimension across the position axes
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [
                rope(pos[..., i], d, self.theta, self.ntk)
                for i, d in enumerate(self.axdims)
            ],
            dim=-3,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class RMSNorm(torch.nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.zeros(features, device=device, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(
            t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0)
        )
        return t.to(dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self, features: int, multiplier: int, bias: bool = False, multiple: int = 128
    ):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)

        self.gate = torch.nn.Linear(features, mlpdim, bias=bias)
        self.up = torch.nn.Linear(features, mlpdim, bias=bias)
        self.down = torch.nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(torch.nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads

        self.wq = torch.nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = torch.nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = torch.nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        qkv: Tensor,
        freqs: Tensor | None = None,
        mask: Tensor | None = None,
        ref_span: tuple[int, int] | None = None,
        kv_capture: list | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)

        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        if kv_capture is not None and ref_span is not None:
            # Stash this block's post-RoPE ref K/V so later denoising steps can
            # run without the ref tokens in the sequence (clone: drop the view
            # into the full-sequence K/V so only the ref span stays alive).
            kv_capture.append(
                (
                    k[:, :, ref_span[0] : ref_span[1]].clone(),
                    v[:, :, ref_span[0] : ref_span[1]].clone(),
                )
            )
        if kv_cache is not None:
            # Cached ref K/V are already RoPE'd at their original positions.
            k = torch.cat((k, kv_cache[0]), dim=2)
            v = torch.cat((v, kv_cache[1]), dim=2)
        out = self.wo(attention(q, k, v, mask=mask, gqa=self.gqa) * F.sigmoid(gate))

        return out


class LastLayer(torch.nn.Module):
    def __init__(self, features: int, patch_size: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = torch.nn.Linear(features, patch_size * patch_size * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        x = self.linear(x)
        return x


class TextFusionBlock(torch.nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))

        return x


class TextFusionTransformer(torch.nn.Module):
    # num_txt_layers is the number of selected encoder hidden-state layers fed in
    # (projected down to 1), NOT the transformer depth — that's fixed at 2 + 2 blocks.
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        # Collapse to 3D for the projector: a quantized (quanto) Linear's matmul
        # kernel only accepts 2D/3D activations, and this layer-axis projection
        # (n -> 1) otherwise feeds it a 4D (b, l, d, n) tensor.
        x = self.projector(x.reshape(b * l, d, n))
        x = x.reshape(b, l, d)

        for block in self.refiner_blocks:
            x = block(x, mask=mask)

        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
        ref_span: tuple[int, int] | None = None,
        kv_capture: list | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        attn_kwargs = dict(ref_span=ref_span, kv_capture=kv_capture, kv_cache=kv_cache)
        # ``vec`` is the (B, 1, 6*features) modulation input, or a tuple
        # ``(vec, refvec, split)`` for reference-image conditioning: tokens
        # ``[:split]`` (text + noisy image) are modulated with ``vec`` while
        # tokens ``[split:]`` (clean reference tokens) use ``refvec`` built from
        # t=0 (ComfyUI Kontext "index_timestep_zero"). Applied per span rather
        # than materializing a per-token (B, L, 6*features) tensor.
        if isinstance(vec, tuple):
            vec, refvec, split = vec
            m = self.mod(vec)
            r = self.mod(refvec)

            def mod(h, scale, shift):
                return torch.cat(
                    (
                        (1 + m[scale]) * h[:, :split] + m[shift],
                        (1 + r[scale]) * h[:, split:] + r[shift],
                    ),
                    dim=1,
                )

            def gate(h, g):
                return torch.cat((m[g] * h[:, :split], r[g] * h[:, split:]), dim=1)

            x = x + gate(
                self.attn(mod(self.prenorm(x), 0, 1), freqs, mask, **attn_kwargs), 2
            )
            x = x + gate(self.mlp(mod(self.postnorm(x), 3, 4)), 5)
            return x

        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn(
            (1 + prescale) * self.prenorm(x) + preshift, freqs, mask, **attn_kwargs
        )
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)

        return x


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        headdim = config.features // config.heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"

        self.posemb = PositionalEncoding(
            config.features, axes, theta=config.theta, ntk=1.0
        )
        self.first = nn.Linear(
            config.channels * config.patch_size**2, config.features, bias=True
        )

        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features,
                    config.heads,
                    config.multiplier,
                    config.bias,
                    config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch_size, config.channels)

        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        reflen: int = 0,
        isolate_refs: bool = False,
        ref_kv_capture: list | None = None,
        ref_kv_cache: tuple[list, Tensor] | None = None,
    ) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        txtmask = _mask(mask[:, : context.shape[1]])

        context = self.txtfusion(context, mask=txtmask)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        # Pad combined sequence to a multiple of 256 to stabilize compiled kernel shapes.
        fulllen = combined.shape[1]
        _padlen = (-fulllen) % 256
        if _padlen > 0:
            combined = F.pad(combined, (0, 0, 0, _padlen))
            mask = F.pad(mask, (0, _padlen), value=False)
            pos = F.pad(pos, (0, 0, 0, _padlen))

        blockvec = tvec
        if reflen > 0:
            # The last ``reflen`` image tokens are clean reference tokens: they
            # get t=0 modulation (ComfyUI Kontext "index_timestep_zero") while
            # text + noisy image tokens keep the real t. Padding tokens fall in
            # the t=0 span, but they are masked from attention and sliced off
            # the output, so their values never matter.
            t0 = self.tmlp(
                temb(
                    torch.zeros_like(t[:, 0, 0]),
                    self.config.tdim,
                    device=img.device,
                    dtype=img.dtype,
                )
            )
            blockvec = (tvec, self.tproj(t0), txtlen + imglen - reflen)

        padmask = mask  # (B, L) key-padding mask, incl. the 256-alignment pad
        mask = _mask(mask)

        if reflen > 0 and isolate_refs:
            # Asymmetric attention (OminiControl2-style "feature reuse"): ref
            # queries attend only to ref keys, while text + noisy queries still
            # see everything. Combined with the t=0 modulation above, ref hidden
            # states become independent of t and of the noisy tokens, so their
            # per-layer K/V can be computed once and cached across denoising
            # steps at inference. Changes attention flow vs the base model, so
            # it needs to be trained in.
            split = txtlen + imglen - reflen
            is_ref = torch.zeros(
                combined.shape[1], dtype=torch.bool, device=combined.device
            )
            is_ref[split : split + reflen] = True
            mask = mask & (~is_ref[:, None] | is_ref[None, :])

        # Ref K/V caching (inference-only; requires isolate_refs so the cached
        # features are step-invariant). Capture mode: this pass has the refs in
        # the sequence and records each block's post-RoPE ref K/V. Reuse mode:
        # the refs are dropped from the sequence (reflen == 0) and the cached
        # K/V are appended as extra attention keys instead.
        ref_span = None
        if ref_kv_capture is not None and reflen > 0:
            assert isolate_refs, "ref K/V capture requires isolate_refs"
            split = txtlen + imglen - reflen
            ref_span = (split, split + reflen)

        blockcaches = [None] * len(self.blocks)
        if ref_kv_cache is not None:
            blockcaches, refmask = ref_kv_cache
            # live queries may attend a cached ref key wherever that ref token
            # is real (refmask right-pads samples with fewer ref tokens)
            extra = padmask.unsqueeze(1).unsqueeze(3) & refmask.unsqueeze(1).unsqueeze(2)
            mask = torch.cat((mask, extra), dim=3)

        freqs = self.posemb(pos)

        for block, blockkv in zip(self.blocks, blockcaches):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                combined = checkpoint(
                    block,
                    combined,
                    blockvec,
                    freqs,
                    mask,
                    use_reentrant=False,
                )
            else:
                combined = block(
                    combined,
                    blockvec,
                    freqs,
                    mask,
                    ref_span=ref_span,
                    kv_capture=ref_kv_capture,
                    kv_cache=blockkv,
                )

        final = self.last(combined, t)
        output = final[:, txtlen : txtlen + imglen - reflen, :]

        return output
