"""Flux3Transformer -- the FLUX.3 joint single-sequence DiT.

Every content mode is a token stream: ``x_image`` is the denoised stream and
``x_image_cond`` carries clean (t=0) reference tokens. Each stream is a
``(B, S, C_mode)`` token sequence with 4-axis RoPE position ids ``(B, S, 4)``
and per-token timesteps ``(B, S)``. Text context ``ctx`` is ``(B, L, 20480)``
(stacked Qwen3-VL hidden layers); FLUX.3 has no pooled text vector.

Each mode gets its own input projection, ``depth`` early self-attention
blocks and modulation tables; text gets the same. The active modes and text
are then concatenated into one sequence for ``depth_single_blocks`` joint
blocks, split back apart, and the output modes go through their own final
layer. A mode with no tokens runs a 2-token zero dummy through its early
blocks (keeps the graph uniform) and never enters the joint sequence.

The module tree matches the shipped ``model.safetensors`` layout so the state
dict loads without any key renaming.

Attention defaults to torch's native ``scaled_dot_product_attention`` so it
runs on any GPU; ``set_attention_backend`` can opt into the cuDNN-pinned
kernel (Blackwell) or flash-attn 3 when installed.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from toolkit.models.v2._mixin import OstrisModelMixin

from .checkpoints import COMFY_REPO, COMFY_TRANSFORMER_FILES

try:
    from flash_attn_interface import flash_attn_func

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    _FLASH_ATTN_AVAILABLE = False

ATTENTION_BACKENDS = ("native", "cudnn", "flash")


@dataclass
class Flux3TransformerParams:
    # content modes -> latent channels (text is separate: context_in_dim)
    in_channels: dict = field(
        default_factory=lambda: {"image": 128, "image_cond": 128}
    )
    # forward kwarg name -> content mode
    sequence: dict = field(
        default_factory=lambda: {"x_image": "image", "x_image_cond": "image_cond"}
    )
    # pooled-vector embedder width; None = no vector_in
    vec_in_dim: int | None = None
    context_in_dim: int = 20480
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 28
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 10000
    mlp_ratio: float = 3.0
    qkv_bias: bool = False
    attn_mode: str = "native"
    # modes with a final_layer (velocity outputs); None = every content mode
    out_modes: list[str] | None = field(default_factory=lambda: ["image"])


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class Attention(nn.Module):
    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mode: str = "native"
    ) -> Tensor:
        if attn_mode == "flash":
            if flash_attn_func is None:
                raise RuntimeError("flash_attn_func is unavailable")
            b, h, n, d = q.shape
            q, k, v = (rearrange(t, "b h n d -> b n h d") for t in (q, k, v))
            x = flash_attn_func(q, k, v)[0]
            return x.reshape(b, n, h * d)
        if attn_mode == "cudnn":
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v)
        else:  # native
            x = F.scaled_dot_product_attention(q, k, v)
        return rearrange(x, "b h n d -> b n (h d)")


def _bf16_if_required(module: nn.Module, x: Tensor) -> Tensor:
    if getattr(module, "_requires_bf16_input", False) and x.dtype != torch.bfloat16:
        return x.to(torch.bfloat16)
    return x


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [
                rope(ids[..., i], self.axes_dim[i], self.theta)
                for i in range(len(self.axes_dim))
            ],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(
    t: Tensor,
    dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(out_dtype if out_dtype is not None else t.dtype)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec: Tensor):
        out = self.lin(F.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        s = self.adaLN_modulation[0](vec)  # type: ignore
        shift_w, scale_w = self.adaLN_modulation[1].weight.chunk(2)

        # x = (1 + scale) * self.norm_final(x) + shift
        # avoid materializing the full (seq, 2*hidden) projection to reduce memory
        x = self.norm_final(x)
        x = x * (F.linear(s, scale_w) + 1)
        x = x + F.linear(s, shift_w)
        return self.linear(x)


class ModeBlock(nn.Module):
    """Per-mode (early) block: self attention within one stream."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        qkv_bias: bool = False,
        use_gated_mlp: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2 if use_gated_mlp else 1

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.mlp_in = nn.Linear(
            hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor, bias=False
        )

        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)

        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = (
            SiLUActivation() if use_gated_mlp else nn.GELU(approximate="tanh")
        )
        self.attention = Attention()
        self.attn_mode = "native"

    def forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, ...]) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        q = rearrange(
            self.q_proj(_bf16_if_required(self.q_proj, x_mod)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        k = rearrange(
            self.k_proj(_bf16_if_required(self.k_proj, x_mod)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        v = rearrange(
            self.v_proj(_bf16_if_required(self.v_proj, x_mod)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        mlp = self.mlp_in(_bf16_if_required(self.mlp_in, x_mod))

        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        attn = self.attention(q, k, v, self.attn_mode)

        output = self.attn_out(_bf16_if_required(self.attn_out, attn)) + self.mlp_out(
            _bf16_if_required(self.mlp_out, self.mlp_act(mlp))
        )
        return x + mod_gate * output


class SingleStreamBlock(nn.Module):
    """Joint block: text + all active content modes in one attention sequence."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        use_gated_mlp: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2 if use_gated_mlp else 1

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(
            hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor, bias=False
        )
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)

        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = (
            SiLUActivation() if use_gated_mlp else nn.GELU(approximate="tanh")
        )
        self.attention = Attention()
        self.attn_mode = "native"

    def forward(
        self,
        x: Tensor,
        ctx: Tensor,
        pe: Tensor,
        mod_x: tuple[Tensor, ...],
        mod_ctx: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        mod_shift_x, mod_scale_x, mod_gate_x = mod_x
        mod_shift_ctx, mod_scale_ctx, mod_gate_ctx = mod_ctx

        x_mod = (1 + mod_scale_x) * self.pre_norm(x) + mod_shift_x
        ctx_mod = (1 + mod_scale_ctx) * self.pre_norm(ctx) + mod_shift_ctx
        combined = torch.cat((ctx_mod, x_mod), dim=1)

        q = rearrange(
            self.q_proj(_bf16_if_required(self.q_proj, combined)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        k = rearrange(
            self.k_proj(_bf16_if_required(self.k_proj, combined)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        v = rearrange(
            self.v_proj(_bf16_if_required(self.v_proj, combined)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        mlp = self.mlp_in(_bf16_if_required(self.mlp_in, combined))

        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        attn = self.attention(q, k, v, self.attn_mode)

        output = self.attn_out(_bf16_if_required(self.attn_out, attn)) + self.mlp_out(
            _bf16_if_required(self.mlp_out, self.mlp_act(mlp))
        )

        ctx_out, x_out = torch.split(output, [ctx.shape[1], x.shape[1]], dim=1)
        return ctx + mod_gate_ctx * ctx_out, x + mod_gate_x * x_out


_QKV = ("q_proj", "k_proj", "v_proj")


class Flux3Transformer(nn.Module, OstrisModelMixin):
    aitk_comfy_repo = COMFY_REPO
    aitk_comfy_weight_names = {COMFY_REPO: COMFY_TRANSFORMER_FILES}

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        if any(k.startswith("ref_time_in.") for k in state_dict):
            raise ValueError(
                "this is the step-distilled (flow-map) FLUX.3 checkpoint; use the "
                "base model (flux3_image_klein_* without _sd_)"
            )
        from toolkit.util.comfy_quant_import import split_fused_quantized_keys

        state_dict = dict(state_dict)
        # the int8 repack fuses every block's q_proj/k_proj/v_proj into one qkv
        # entry (rows q, k, v); the module (and the bf16/nvfp4 files) keep them
        # split
        for marker in [k for k in list(state_dict) if k.endswith(".qkv.comfy_quant")]:
            base = marker[: -len(".qkv.comfy_quant")]
            split_fused_quantized_keys(
                state_dict, f"{base}.qkv", [f"{base}.{p}" for p in _QKV]
            )
        for key in [k for k in list(state_dict) if k.endswith(".qkv.weight")]:
            base = key[: -len(".qkv.weight")]
            for p, w in zip(_QKV, torch.chunk(state_dict.pop(key), 3, dim=0)):
                state_dict[f"{base}.{p}.weight"] = w.contiguous()
        # static activation scales (calibration extras) are unused
        for k in [k for k in state_dict if k.endswith(".input_scale")]:
            state_dict.pop(k)
        return state_dict

    @classmethod
    def get_transformer_block_names(cls):
        return [
            "content_mode_blocks.image",
            "content_mode_blocks.image_cond",
            "txt_mode_blocks",
            "single_blocks",
        ]

    def get_offload_ignore_modules(self):
        # modulation tables stay resident on the compute device; LastLayer
        # reads adaLN weights raw (bypasses offload forward hijacks)
        return [m for m in self.modules() if isinstance(m, (Modulation, LastLayer))]

    def __init__(self, params: Flux3TransformerParams | None = None):
        super().__init__()
        if params is None:
            params = Flux3TransformerParams()
        self.params = params

        self.sequence = params.sequence
        for _, v in params.sequence.items():
            assert v in params.in_channels, (params.sequence, params.in_channels)

        self.in_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.depth = params.depth
        self.depth_single_blocks = params.depth_single_blocks
        self.mlp_ratio = params.mlp_ratio
        self.qkv_bias = params.qkv_bias
        self.attn_mode = params.attn_mode

        self.gradient_checkpointing = False

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by "
                f"num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )

        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )

        self.emb_in = nn.ModuleDict(
            {
                m: nn.Linear(c, params.hidden_size, bias=False)
                for m, c in params.in_channels.items()
            }
        )
        self.txt_in = nn.Linear(params.context_in_dim, params.hidden_size, bias=False)

        self.time_in = MLPEmbedder(
            in_dim=256, hidden_dim=params.hidden_size, disable_bias=True
        )
        self.vector_in: nn.Module = (
            MLPEmbedder(params.vec_in_dim, params.hidden_size, disable_bias=True)
            if params.vec_in_dim is not None
            else nn.Identity()
        )

        content_modes = sorted(params.in_channels.keys())
        self.early_stream_modulations = nn.ModuleDict(
            {
                **{
                    m: Modulation(params.hidden_size, double=False, disable_bias=True)
                    for m in content_modes
                },
                "txt": Modulation(params.hidden_size, double=False, disable_bias=True),
            }
        )
        self.single_stream_modulations = nn.ModuleDict(
            {
                **{
                    m: Modulation(params.hidden_size, double=False, disable_bias=True)
                    for m in content_modes
                },
                "txt": Modulation(params.hidden_size, double=False, disable_bias=True),
            }
        )

        self.content_mode_blocks = nn.ModuleDict(
            {
                m: nn.ModuleList(
                    [
                        ModeBlock(
                            params.hidden_size,
                            params.num_heads,
                            mlp_ratio=params.mlp_ratio,
                            qkv_bias=params.qkv_bias,
                        )
                        for _ in range(params.depth)
                    ]
                )
                for m in content_modes
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            [
                ModeBlock(
                    params.hidden_size,
                    params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    params.hidden_size,
                    params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        out_modes = (
            list(params.in_channels) if params.out_modes is None else params.out_modes
        )
        assert set(out_modes) <= set(params.in_channels), (out_modes, params.in_channels)
        self.final_layer = nn.ModuleDict(
            {
                m: LastLayer(params.hidden_size, c)
                for m, c in params.in_channels.items()
                if m in out_modes
            }
        )

        self.set_attention_backend(params.attn_mode)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def set_attention_backend(self, backend: str):
        if backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"Unknown attention backend {backend!r}; choose from {ATTENTION_BACKENDS}"
            )
        if backend == "flash" and not _FLASH_ATTN_AVAILABLE:
            raise RuntimeError(
                "attention backend 'flash' requested but flash_attn_interface "
                "(flash-attn 3) is not installed"
            )
        if backend == "cudnn" and not torch.cuda.is_available():
            raise RuntimeError("attention backend 'cudnn' requires CUDA")
        self.attn_mode = backend
        for module in self.modules():
            if hasattr(module, "attn_mode"):
                module.attn_mode = backend

    def _ckpt(self, module, *args):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(
        self,
        ctx: Tensor,
        ctx_ids: Tensor,
        timesteps_ctx: Tensor,
        vector: Tensor | None = None,
        **kwargs: Tensor,
    ) -> dict[str, Tensor]:
        batch_size = ctx.shape[0]

        x_by_mod: dict[str, list] = defaultdict(list)
        ids_by_mod: dict[str, list] = defaultdict(list)
        ts_by_mod: dict[str, list] = defaultdict(list)
        lens_by_mod: dict[str, list] = defaultdict(list)
        names_by_mod: dict[str, list] = defaultdict(list)
        for k, v in self.sequence.items():
            if k not in kwargs or kwargs[k] is None:
                continue
            assert k + "_ids" in kwargs and k + "_timesteps" in kwargs
            x_by_mod[v].append(kwargs[k])
            ids_by_mod[v].append(kwargs[k + "_ids"])
            ts_by_mod[v].append(kwargs[k + "_timesteps"])
            lens_by_mod[v].append(kwargs[k].shape[1])
            names_by_mod[v].append(k)

        active = list(x_by_mod.keys())
        all_mods = sorted(self.in_channels.keys())
        inactive = [m for m in all_mods if m not in active]
        dummy_len = 2  # attention kernels need >=2

        device = x_by_mod[active[0]][0].device
        dtype = x_by_mod[active[0]][0].dtype
        for k in all_mods:
            if k in active:
                x_by_mod[k] = torch.cat(x_by_mod[k], 1)
                ids_by_mod[k] = torch.cat(ids_by_mod[k], 1)
                ts_by_mod[k] = torch.cat(ts_by_mod[k], 1)
            else:
                x_by_mod[k] = torch.zeros(
                    batch_size,
                    dummy_len,
                    self.in_channels[k],
                    device=device,
                    dtype=dtype,
                )
                ids_by_mod[k] = torch.zeros(
                    batch_size, dummy_len, 4, device=device, dtype=torch.int64
                )
                ts_by_mod[k] = torch.zeros(
                    batch_size, dummy_len, device=device, dtype=dtype
                )

        vec_by_mod: dict[str, Tensor] = {}
        for k in all_mods:
            ts = rearrange(ts_by_mod[k], "b t -> (b t)")
            vec_by_mod[k] = rearrange(
                self.time_in(timestep_embedding(ts, 256, out_dtype=dtype)),
                "(b t) d -> b t d",
                b=batch_size,
            )
        ts_ctx = rearrange(timesteps_ctx, "b t -> (b t)")
        vec_ctx = rearrange(
            self.time_in(timestep_embedding(ts_ctx, 256, out_dtype=dtype)),
            "(b t) d -> b t d",
            b=batch_size,
        )

        if self.params.vec_in_dim is not None:
            assert vector is not None and vector.ndim == 2
            vector_emb = self.vector_in(vector)
            for k in all_mods:
                vec_by_mod[k] = vec_by_mod[k] + vector_emb[:, None, :]
            vec_ctx = vec_ctx + vector_emb[:, None, :]

        early_content_mods: dict[str, tuple] = {}
        for key in self.early_stream_modulations:
            if key == "txt":
                continue
            mod, _ = self.early_stream_modulations[key](vec_by_mod[key])
            early_content_mods[key] = mod
        early_txt_mod, _ = self.early_stream_modulations["txt"](vec_ctx)

        single_content_mods: dict[str, list] = {}
        for key in self.single_stream_modulations:
            if key == "txt":
                continue
            mod, _ = self.single_stream_modulations[key](vec_by_mod[key])
            single_content_mods[key] = list(mod)
        single_txt_mod, _ = self.single_stream_modulations["txt"](vec_ctx)

        content_by_mod = {k: self.emb_in[k](x_by_mod[k]) for k in all_mods}
        txt = self.txt_in(ctx)

        pe_by_mod = {k: self.pe_embedder(ids_by_mod[k]) for k in all_mods}
        pe_ctx = self.pe_embedder(ctx_ids)

        for d in range(self.depth):
            for m in all_mods:
                content_by_mod[m] = self._ckpt(
                    self.content_mode_blocks[m][d],
                    content_by_mod[m],
                    pe_by_mod[m],
                    early_content_mods[m],
                )
            txt = self._ckpt(self.txt_mode_blocks[d], txt, pe_ctx, early_txt_mod)

        img = torch.cat([content_by_mod[k] for k in active], dim=1)
        content_mod = tuple(
            torch.cat([single_content_mods[k][i] for k in active], dim=1)
            for i in range(3)
        )
        pe_content = torch.cat([pe_by_mod[k] for k in active], dim=2)
        pe = torch.cat((pe_ctx, pe_content), dim=2)

        for block in self.single_blocks:
            txt, img = self._ckpt(
                block,
                img,
                txt,
                pe,
                content_mod,
                single_txt_mod,
            )

        img_by_mod: dict[str, Tensor] = {}
        pos = 0
        for k in active:
            length = sum(lens_by_mod[k])
            img_by_mod[k] = img[:, pos : pos + length]
            pos += length
        for k in inactive:
            img_by_mod[k] = content_by_mod[k]

        final = {
            k: self._ckpt(self.final_layer[k], img_by_mod[k], vec_by_mod[k])
            for k in self.final_layer
        }
        out: dict[str, Tensor] = {}
        for k in active:
            if k not in final:
                continue
            pos = 0
            for name, length in zip(names_by_mod[k], lens_by_mod[k]):
                out[name] = final[k][:, pos : pos + length]
                pos += length
        return out
