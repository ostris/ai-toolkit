"""MOSS-Image DiT (OpenMOSS-Team/MOSS-Image-SFT / -Base).

Port of ``MOSSImageTransformer`` from the vendor's ``pipeline_moss_image.py``
in the Comfy-Org/MOSS-Image key layout: the vendor's names with each block's
q/k/v fused into ``proj.qkv`` (vendor shards and split community repacks are
fused on load), so Comfy files load as-is and LoRA keys match ComfyUI. One stream: image tokens followed by text tokens
through 42 blocks that share one AdaLN modulation; the text first goes through
a 2-block refiner. The vendor evaluates every item alone at its true text
length; here a batch runs together with padded text masked out of every
attention (identical results) and the unmasked path when lengths match.
Activations follow the weight dtype exactly like the vendor's bf16 inference:
fp32 timestep sinusoid and RoPE phases, everything else in the compute dtype.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from toolkit.models.v2._mixin import OstrisModelMixin


@dataclass
class MossImageParams:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads, mask):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    out = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, mask):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, mask))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        # q/k/v fused (rows q, k, v) like the ComfyUI files; the vendor keeps
        # them split (fused on load)
        self.proj = nn.ModuleDict(
            {
                "qkv": nn.Linear(dim, 3 * dim, bias=False),
                "o": nn.Linear(dim, dim, bias=False),
                "ffn_up_gate": nn.Linear(dim, 2 * ffn_dim, bias=False),
                "ffn_down": nn.Linear(ffn_dim, dim, bias=False),
            }
        )

    def forward(self, x, rope, modulation, mask):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        q, k, v = self.proj["qkv"](y).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        attn = _attention(q, k, v, rope, self.heads, mask)
        x = x + self.proj["o"](attn) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class MossImageTransformer(nn.Module, OstrisModelMixin):
    @classmethod
    def aitk_config_from_state_dict(cls, state_dict):
        # sniffed from unquantized tensors only (block linears may be int8)
        video = state_dict["video_embedder.weight"]
        dim = video.shape[0]
        return MossImageParams(
            dim=dim,
            ffn_dim=state_dict["text_embedder.blocks.0.ffn_up.weight"].shape[0],
            num_heads=dim // state_dict["layers.0.q_norm.weight"].shape[0],
            num_layers=sum(
                1
                for k in state_dict
                if k.startswith("layers.") and k.endswith(".norm1.weight")
            ),
            text_dim=state_dict["text_embedder.proj.weight"].shape[1],
            freq_dim=state_dict["time_embedding.0.weight"].shape[1],
            latent_channels=video.shape[1],
        )

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        # derived constants; recomputed in fp32 every forward
        state_dict = {k: v for k, v in state_dict.items() if k != "rope.bands"}
        fuse_split_qkv(state_dict)
        return state_dict

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        # the vendor's strict loader expects the (unused) frequency table
        state_dict = dict(state_dict)
        head_dim = state_dict["layers.0.q_norm.weight"].shape[0]
        state_dict["rope.bands"] = _rope_bands("cpu", head_dim, MossImageParams.rope_base)
        return state_dict

    @classmethod
    def get_transformer_block_names(cls):
        return ["layers"]

    @classmethod
    def get_quantization_exclude_modules(cls):
        # embedders, modulation, text refiner and output head stay full precision
        return [
            "video_embedder",
            "text_embedder*",
            "time_embedding*",
            "shared_block_modulation",
            "video_head_modulation",
            "final_linear_video",
        ]

    def __init__(self, params: Optional[MossImageParams] = None):
        super().__init__()
        if params is None:
            params = MossImageParams()
        if params.dim % params.num_heads or (params.dim // params.num_heads) % 8:
            raise ValueError("dim must divide into heads with head_dim divisible by 8")
        self.params = params
        dim, ffn_dim, heads = params.dim, params.ffn_dim, params.num_heads
        self.video_embedder = nn.Linear(params.latent_channels, dim)
        self.text_embedder = _TextProjection(params.text_dim, dim, ffn_dim, heads)
        self.time_embedding = nn.Sequential(
            nn.Linear(params.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList(
            [_Block(dim, ffn_dim, heads) for _ in range(params.num_layers)]
        )
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, params.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.gradient_checkpointing = False
        self._bands = {}

    @property
    def device(self):
        return self.final_norm_video.weight.device

    @property
    def dtype(self):
        # norms are never quantized; the block linears may be
        return self.final_norm_video.weight.dtype

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _ckpt(self, module, *args):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _rope(self, lengths, height, width, text_len, device):
        """(B, H*W + S, head_dim) fp32 rope; coordinates (r, f, h, w): image
        tokens at (0, item text length, centered h, centered w), text token i
        at (0, i, 0, 0)."""
        bands = self._bands.get(device)
        if bands is None:
            bands = _rope_bands(
                device, self.params.dim // self.params.num_heads, self.params.rope_base
            )
            self._bands[device] = bands
        batch = lengths.shape[0]
        h, w = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        image = torch.zeros(batch, height * width, 4, device=device)
        image[..., 1] = lengths.float()[:, None]
        image[..., 2] = (h - (height - 1) / 2).flatten()
        image[..., 3] = (w - (width - 1) / 2).flatten()
        text = torch.zeros(batch, text_len, 4, device=device)
        text[..., 1] = torch.arange(text_len, device=device, dtype=torch.float32)
        phase = torch.cat((image, text), 1).unsqueeze(-1) * bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, latent_channels, H, W)
        timestep: torch.Tensor,  # 0..1000, 1000 = pure noise
        encoder_hidden_states: torch.Tensor,  # (B, S, text_dim)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # (B, S) bool
        skip_layers: Optional[Iterable[int]] = None,
    ):
        skip = frozenset(skip_layers or ())
        batch, _, height, width = hidden_states.shape
        num_image = height * width
        device = hidden_states.device
        dtype = self.dtype

        context = encoder_hidden_states
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(context.shape[:2], dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        lengths = mask.sum(1)
        if bool((lengths == 0).any()):
            raise ValueError("each sample needs at least one valid text token")
        text_len = int(lengths.max())
        padded = bool((lengths < text_len).any())
        if padded or not bool(mask[:, :text_len].all()):
            # valid tokens first, in order (the vendor selects them by mask)
            order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
            context = torch.gather(
                context, 1, order[..., None].expand(-1, -1, context.shape[-1])
            )
        context = context[:, :text_len]
        valid = torch.arange(text_len, device=device)[None] < lengths[:, None]

        times = torch.as_tensor(timestep, device=device, dtype=torch.float32)
        times = times.reshape(-1).expand(batch)
        half = self.params.freq_dim // 2
        freq = 10000 ** (
            -torch.arange(half, device=device, dtype=torch.float64) / half
        )
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(dtype)).float()
        mod = self.shared_block_modulation(time.to(dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(dtype)).view(batch, 2, -1)

        rope = self._rope(lengths, height, width, text_len, device).to(dtype)

        text_mask = joint_mask = None
        if padded:
            text_mask = valid[:, None, None, :]
            joint_mask = torch.cat(
                (torch.ones(batch, num_image, dtype=torch.bool, device=device), valid), 1
            )[:, None, None, :]

        video = hidden_states.flatten(2).transpose(1, 2).to(dtype)
        video = self.video_embedder(video.contiguous())
        text = self.text_embedder.proj(context.to(dtype))
        text_rope = rope[:, num_image:]
        for block in self.text_embedder.blocks:
            text = self._ckpt(block, text, text_rope, text_mask)

        x = torch.cat((video, text), 1)
        for index, layer in enumerate(self.layers):
            if index not in skip:
                x = self._ckpt(layer, x, rope, mod, joint_mask)

        x = self.final_norm_video(x[:, :num_image])
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x)
        return x.transpose(1, 2).reshape(batch, -1, height, width)


def fuse_split_qkv(state_dict):
    """Vendor / community layout (``layers.N.proj.{q,k,v}``, plain or comfy
    quantized) -> the fused ComfyUI ``layers.N.proj.qkv``. In place."""
    from toolkit.util.comfy_quant_import import fuse_split_quantized_keys

    for key in [k for k in state_dict if k.endswith(".proj.q.weight")]:
        base = key[: -len(".q.weight")]
        parts = [f"{base}.{p}" for p in ("q", "k", "v")]
        if f"{base}.q.comfy_quant" in state_dict:
            fuse_split_quantized_keys(state_dict, parts, f"{base}.qkv")
        else:
            state_dict[f"{base}.qkv.weight"] = torch.cat(
                [state_dict.pop(f"{p}.weight") for p in parts], dim=0
            )
    return state_dict


def _rope_bands(device, head_dim: int, base: float):
    # built on cpu in fp32 like the vendor's training, then moved
    num_bands = head_dim // 8
    exponent = torch.arange(num_bands, dtype=torch.float32) / num_bands
    return (1 / (base**exponent)).to(device)
