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

Edit LoRAs (not part of the vendor model): clean reference
latents ride after the text tokens with t=0 modulation, each on its own index
of the RoPE axis the vendor leaves at 0, optionally isolated so their per-block
K/V can be cached across sampling steps.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional

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


def _attention(
    q, k, v, rope, heads, mask, ref_span=None, kv_capture=None, kv_cache=None, isolate_split=None
):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    if kv_capture is not None:
        # post-RoPE reference K/V, reused as extra keys on later steps. Copies:
        # slices would pin the full-sequence K and the whole fused qkv output
        # (~4x the cache) for the rest of sampling
        kv_capture.append(
            tuple(
                t[:, :, ref_span[0] : ref_span[1]].clone(memory_format=torch.contiguous_format)
                for t in (k, v)
            )
        )
    if kv_cache is not None:
        k = torch.cat((k, kv_cache[0].to(k.dtype)), 2)
        v = torch.cat((v, kv_cache[1].to(v.dtype)), 2)
    q, k = q.to(v.dtype), k.to(v.dtype)
    if isolate_split is not None:
        # reference isolation without a mask (a boolean mask forces SDPA off
        # flash attention; its backward is ~4x slower): live queries attend
        # every key, reference queries only reference keys
        live = F.scaled_dot_product_attention(q[:, :, :isolate_split], k, v)
        ref = F.scaled_dot_product_attention(
            q[:, :, isolate_split:], k[:, :, isolate_split:], v[:, :, isolate_split:]
        )
        out = torch.cat((live, ref), 2)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
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

    def forward(
        self, x, rope, modulation, mask, isolate_split=None, ref_span=None, kv_capture=None, kv_cache=None
    ):
        # modulation (B, 6, D) [shift_a, scale_a, gate_a, shift_f, scale_f,
        # gate_f], or (mod, ref_mod, split): tokens [split:] are clean
        # references modulated at t=0
        if isinstance(modulation, tuple):
            mod, ref_mod, split = modulation
            m, r = mod.unbind(1), ref_mod.unbind(1)

            def affine(h, scale, shift):
                return torch.cat(
                    (
                        h[:, :split] * (1 + m[scale][:, None]) + m[shift][:, None],
                        h[:, split:] * (1 + r[scale][:, None]) + r[shift][:, None],
                    ),
                    1,
                )

            def gate(h, g):
                return torch.cat(
                    (h[:, :split] * m[g][:, None], h[:, split:] * r[g][:, None]), 1
                )

        else:
            m = modulation.unbind(1)

            def affine(h, scale, shift):
                return h * (1 + m[scale][:, None]) + m[shift][:, None]

            def gate(h, g):
                return h * m[g][:, None]

        y = affine(self.norm1(x), 1, 0)
        q, k, v = self.proj["qkv"](y).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        attn = _attention(
            q, k, v, rope, self.heads, mask, ref_span, kv_capture, kv_cache, isolate_split
        )
        x = x + gate(self.proj["o"](attn), 2)
        y = affine(self.norm2(x), 4, 3)
        hidden, ffn_gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + gate(self.proj["ffn_down"](_situ(hidden, ffn_gate)), 5)


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
        # ComfyUI-saved checkpoints wrap the DiT under model.diffusion_model;
        # component files may instead use diffusion_model or transformer.
        # Select the component before config sniffing, QKV fusion and quantized
        # import so weights and their scale/marker sidecars stay together.
        for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
            if prefix + "video_embedder.weight" in state_dict:
                state_dict = {
                    k[len(prefix):]: v for k, v in state_dict.items()
                    if k.startswith(prefix)
                }
                break
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

    def _rope_from_coords(self, coords):
        """(B, L, 4) fp32 (r, f, h, w) coordinates -> (B, L, head_dim) rope."""
        device = coords.device
        bands = self._bands.get(device)
        if bands is None:
            bands = _rope_bands(
                device, self.params.dim // self.params.num_heads, self.params.rope_base
            )
            self._bands[device] = bands
        phase = coords.unsqueeze(-1) * bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)

    @staticmethod
    def _image_coords(index, frame, height, width, device):
        """(H*W, 4) coordinates of one latent grid: (index, frame, centered h,
        centered w). The vendor's image has index 0; references use 1, 2, ..."""
        h, w = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.zeros(height * width, 4, device=device)
        coords[:, 0] = index
        coords[:, 1] = frame
        coords[:, 2] = (h - (height - 1) / 2).flatten()
        coords[:, 3] = (w - (width - 1) / 2).flatten()
        return coords

    def _rope(self, lengths, height, width, text_len, device):
        """(B, H*W + S, head_dim) fp32 rope; coordinates (r, f, h, w): image
        tokens at (0, item text length, centered h, centered w), text token i
        at (0, i, 0, 0)."""
        batch = lengths.shape[0]
        image = self._image_coords(0, 0, height, width, device)
        image = image.unsqueeze(0).repeat(batch, 1, 1)
        image[..., 1] = lengths.float()[:, None]
        text = torch.zeros(batch, text_len, 4, device=device)
        text[..., 1] = torch.arange(text_len, device=device, dtype=torch.float32)
        return self._rope_from_coords(torch.cat((image, text), 1))

    def _pack_refs(self, ref_latents, lengths, device, dtype):
        """Per-sample lists of clean (C, h, w) reference latents -> right-padded
        tokens (B, R, C), rope (B, R, head_dim) and valid mask (B, R).
        Reference i sits on RoPE axis r = i + 1 (always 0 for the vendor's
        image and text) with the image's frame coordinate and its own centered
        grid."""
        tokens, coords = [], []
        for refs, length in zip(ref_latents, lengths.tolist()):
            sample_tokens, sample_coords = [], []
            for i, latent in enumerate(refs):
                _, h, w = latent.shape
                sample_tokens.append(latent.flatten(1).transpose(0, 1))
                sample_coords.append(self._image_coords(i + 1, length, h, w, device))
            if sample_tokens:
                tokens.append(torch.cat(sample_tokens).to(device, dtype))
                coords.append(torch.cat(sample_coords))
            else:
                tokens.append(torch.zeros(0, self.params.latent_channels, device=device, dtype=dtype))
                coords.append(torch.zeros(0, 4, device=device))
        num_ref = max(t.shape[0] for t in tokens)
        valid = torch.stack(
            [torch.arange(num_ref, device=device) < t.shape[0] for t in tokens]
        )
        tokens = torch.stack([F.pad(t, (0, 0, 0, num_ref - t.shape[0])) for t in tokens])
        coords = torch.stack([F.pad(c, (0, 0, 0, num_ref - c.shape[0])) for c in coords])
        return tokens, self._rope_from_coords(coords), valid

    def _time_modulation(self, times, dtype):
        half = self.params.freq_dim // 2
        freq = 10000 ** (
            -torch.arange(half, device=times.device, dtype=torch.float64) / half
        )
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(dtype)).float()
        mod = self.shared_block_modulation(time.to(dtype)).view(times.shape[0], 6, -1)
        return mod, time

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, latent_channels, H, W)
        timestep: torch.Tensor,  # 0..1000, 1000 = pure noise
        encoder_hidden_states: torch.Tensor,  # (B, S, text_dim)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # (B, S) bool
        skip_layers: Optional[Iterable[int]] = None,
        ref_latents: Optional[List[List[torch.Tensor]]] = None,  # per-sample (C, h, w)
        isolate_refs: bool = False,
        ref_kv_cache: Optional[dict] = None,
    ):
        """``ref_latents``: clean reference latents appended after the text at
        t=0 (edit LoRAs; the base model never saw them). ``isolate_refs``:
        reference queries attend only to reference keys, so their K/V do not
        depend on the step and can be cached. ``ref_kv_cache`` (inference,
        needs ``isolate_refs``): an empty dict is filled with every block's
        reference K/V on this call; once filled, later calls skip the
        reference tokens and attend to the cached K/V instead. One cache per
        prompt / skip-layer combination (the reference RoPE follows the text
        length)."""
        skip = frozenset(skip_layers or ())
        if ref_kv_cache is not None and not isolate_refs:
            raise ValueError("ref_kv_cache needs isolate_refs: cached K/V are only step-invariant when references attend to themselves")
        reuse_refs = ref_kv_cache is not None and ref_kv_cache.get("kv") is not None
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
        mod, time = self._time_modulation(times, dtype)
        head = self.video_head_modulation(time.to(dtype)).view(batch, 2, -1)

        rope = self._rope(lengths, height, width, text_len, device).to(dtype)
        num_live = num_image + text_len

        num_ref, ref_valid, modulation = 0, None, mod
        if reuse_refs:
            ref_valid = ref_kv_cache["mask"]
        elif ref_latents is not None and any(len(r) > 0 for r in ref_latents):
            ref_tokens, ref_rope, ref_valid = self._pack_refs(
                ref_latents, lengths, device, dtype
            )
            num_ref = ref_tokens.shape[1]
            ref_mod, _ = self._time_modulation(torch.zeros_like(times), dtype)
            modulation = (mod, ref_mod, num_live)
            rope = torch.cat((rope, ref_rope.to(dtype)), 1)

        text_mask = None
        if padded:
            text_mask = valid[:, None, None, :]
        live_valid = torch.cat(
            (torch.ones(batch, num_image, dtype=torch.bool, device=device), valid), 1
        )
        joint_mask = isolate_split = None
        if num_ref and isolate_refs:
            key_valid = torch.cat((live_valid, ref_valid), 1)
            if bool(key_valid.all()):
                # no padding anywhere: isolation as two unmasked flash calls
                isolate_split = num_live
            else:
                is_ref = torch.arange(num_live + num_ref, device=device) >= num_live
                allowed = key_valid[:, None, :] & (~is_ref[:, None] | is_ref[None, :])[None]
                # every query keeps at least itself (padded rows would be all-masked)
                allowed = allowed | torch.eye(num_live + num_ref, dtype=torch.bool, device=device)
                joint_mask = allowed[:, None]
        elif num_ref or reuse_refs:
            # live queries see every real reference key (cached ones included)
            key_valid = torch.cat((live_valid, ref_valid), 1)
            if not bool(key_valid.all()):
                joint_mask = key_valid[:, None, None, :]
        elif padded:
            joint_mask = live_valid[:, None, None, :]

        video = hidden_states.flatten(2).transpose(1, 2).to(dtype)
        video = self.video_embedder(video.contiguous())
        text = self.text_embedder.proj(context.to(dtype))
        text_rope = rope[:, num_image:num_live]
        for block in self.text_embedder.blocks:
            text = self._ckpt(block, text, text_rope, text_mask)

        parts = [video, text]
        if num_ref:
            parts.append(self.video_embedder(ref_tokens))
        x = torch.cat(parts, 1)

        capture = ref_kv_cache is not None and not reuse_refs and num_ref > 0
        captured = {}
        for index, layer in enumerate(self.layers):
            if index in skip:
                continue
            if capture:
                slot = []
                x = layer(
                    x, rope, modulation, joint_mask, isolate_split,
                    (num_live, num_live + num_ref), slot,
                )
                captured[index] = slot[0]
            elif reuse_refs:
                x = layer(x, rope, modulation, joint_mask, kv_cache=ref_kv_cache["kv"][index])
            else:
                x = self._ckpt(layer, x, rope, modulation, joint_mask, isolate_split)
        if capture:
            ref_kv_cache["kv"] = captured
            ref_kv_cache["mask"] = ref_valid

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
