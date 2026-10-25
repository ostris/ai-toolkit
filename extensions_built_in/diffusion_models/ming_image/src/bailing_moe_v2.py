"""BailingMoeV2 (Ling-mini-2.0) decoder for the Ming-Image conditioning stack.

Ported from inclusionAI/Ming `modeling_bailing_moe_v2.py` (MIT) and cut down
to the one thing Ming-Image asks of it: a single prefill pass over an already
embedded sequence that returns the hidden states of selected layers. Kept:

  - video-rope 3D positions (text tokens count along all three axes, image
    and query-token blocks get t/h/w coordinates),
  - partial rotary (half of each 128-wide head),
  - the MultiRouter MoE: tokens that came in through the vision path route
    through `image_gate`, everything else through `gate`,
  - group-limited top-k routing with the expert bias and sigmoid scores.

Dropped: KV cache and generation, the audio router, flash-attention paths.

The routed experts of every MoE layer live in one stacked bank (`FusedExperts`)
and run as grouped GEMMs (`moe_kernels.py`); the checkpoint's per-expert
linears are stacked on load. The rope channel layout is built once per forward
rather than once per layer.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from toolkit.util.convrot_quant import largest_pow4_divisor, rotate

from .moe_kernels import grouped_linear, grouped_swiglu


class BailingMoeV2Config:
    """The `llm_config` block of the checkpoint's mllm/config.json."""

    def __init__(self, **kwargs):
        self.vocab_size = 157184
        self.hidden_size = 2048
        self.intermediate_size = 5120
        self.moe_intermediate_size = 512
        self.num_hidden_layers = 20
        self.num_attention_heads = 16
        self.num_key_value_heads = 4
        self.head_dim = 128
        self.rms_norm_eps = 1e-6
        self.rope_theta = 600000.0
        self.partial_rotary_factor = 0.5
        self.num_experts = 256
        self.num_experts_per_tok = 8
        self.num_shared_experts = 1
        self.n_group = 8
        self.topk_group = 4
        self.routed_scaling_factor = 2.5
        self.first_k_dense_replace = 1
        self.use_qkv_bias = False
        self.use_bias = False
        self.router_type = "MultiRouter"
        self.pad_token_id = 156892
        self.image_patch_token = 157157
        self.image_start_token = 157158
        self.image_end_token = 157159
        self.spatial_merge_size = 2
        self.mrope_section = (8, 12, 12)
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize in fp32, apply the weight in the input dtype (as the original)
        x = F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps)
        return self.weight * x.to(self.weight.dtype)


class VideoRotaryEmbedding(nn.Module):
    """cos/sin tables for 3D position ids, shaped `(3, B, L, rope_dim)`."""

    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        dim = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
            .to(x.device)
        )
        positions = position_ids[:, :, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq.float() @ positions.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def video_rope_layout_index(rotary_dim: int, mrope_section: Sequence[int] = (8, 12, 12)) -> List[int]:
    """Flat (axis * rotary_dim + channel) gather index realizing Ming's
    `video_rope` channel layout: the h/w frequencies interleave channel by
    channel and the t frequencies sit in their own runs, i.e.
    (h,w,h,w,...)x24, t x8, (h,w,...)x24, t x8 for a 64-wide rotary half."""
    section = [mrope_section[0], mrope_section[1] + mrope_section[2]] * 2
    section = section[::-1]
    rows = []
    for i, width in enumerate(section):
        if i % 2 == 0:
            rows += [1 if j % 2 == 0 else 2 for j in range(width)]
        else:
            rows += [0] * width
    # the layout is written for a 64-wide half; a narrower table (tiny test
    # configs) keeps only the channels it has, as the slice loop it replaces did
    return [row * rotary_dim + col for col, row in enumerate(rows) if col < rotary_dim]


def video_rope_layout(cos, sin, flat: torch.Tensor):
    """Apply the layout index once per forward. `cos`/`sin` are `(3, B, L,
    rotary_dim)` tables per axis; returns `(B, 1, L, rotary_dim)` tables."""
    # (3, B, L, D) -> (B, L, 3*D) -> pick (axis, channel) pairs -> (B, 1, L, D)
    cos = cos.permute(1, 2, 0, 3).reshape(*cos.shape[1:3], -1).index_select(-1, flat)
    sin = sin.permute(1, 2, 0, 3).reshape(*sin.shape[1:3], -1).index_select(-1, flat)
    return cos.unsqueeze(1), sin.unsqueeze(1)


def apply_video_rope(q, k, cos, sin):
    """Rotate the first `cos.shape[-1]` channels of each head; `cos`/`sin`
    come from `video_rope_layout`."""
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def get_video_rope_index(
    config: BailingMoeV2Config,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    scale_factor: float = 2.0,
) -> torch.Tensor:
    """3D position ids `(3, B, L)` for a text sequence with image blocks.

    Text tokens advance all three axes together. Each image block (a
    `<image>` start token followed by its patch tokens) gets t/h/w
    coordinates centred on the block's row/column midpoints, offset by the
    text position it follows; the text after it resumes from the block's
    largest t. `image_grid_thw` lists the blocks in order of appearance
    across the whole batch, in pre-merge patch units.
    """
    device = input_ids.device
    if image_grid_thw is None or len(image_grid_thw) == 0:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids.unsqueeze(0).expand(3, -1, -1).to(device)

    merge = config.spatial_merge_size
    position_ids = torch.ones(
        3, input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=device
    )
    image_index = 0
    for b in range(input_ids.shape[0]):
        valid = attention_mask[b] == 1
        ids = input_ids[b][valid]
        starts = torch.argwhere(ids == config.image_start_token).squeeze(1)
        image_nums = int((ids[starts + 1] == config.image_patch_token).sum())
        tokens = ids.tolist()
        pos_list = []
        st = 0
        for _ in range(image_nums):
            ed = tokens.index(config.image_patch_token, st)
            t, h, w = (int(v) for v in image_grid_thw[image_index])
            image_index += 1
            grid_t, grid_h, grid_w = t, h // merge, w // merge
            text_len = ed - st
            st_idx = pos_list[-1][0].max() + 1 if pos_list else 0
            pos_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(grid_t).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            h_index = (
                torch.arange(grid_h).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
                - (grid_h - 1) // 2
            )
            w_index = (
                torch.arange(grid_w).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
                - (grid_w - 1) // 2
            )
            t_index = t_index * scale_factor + text_len + st_idx
            pos_list.append(torch.stack([t_index, h_index + t_index, w_index + t_index]))
            st = ed + grid_t * grid_h * grid_w
        if st < len(tokens):
            st_idx = pos_list[-1][0].max() + 1 if pos_list else 0
            pos_list.append(
                torch.arange(len(tokens) - st).view(1, -1).expand(3, -1) + st_idx
            )
        llm_positions = torch.cat(pos_list, dim=1).reshape(3, -1).long()
        position_ids[:, b, valid] = llm_positions.to(device)
    return position_ids


class MLP(nn.Module):
    def __init__(self, config: BailingMoeV2Config, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    """Sigmoid router with the load-balancing expert bias applied to the
    selection only (weights come from the unbiased scores)."""

    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.expert_bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x.shape[0]
        logits = F.linear(x.float(), self.weight.float())
        scores = torch.sigmoid(logits)
        routing = scores + self.expert_bias.float()

        group_scores = routing.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
            .bool()
        )
        masked = routing.masked_fill(~score_mask, float("-inf"))
        topk_idx = torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

        topk_weight = torch.gather(scores, dim=1, index=topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor


class FusedExperts(nn.Module):
    """All routed experts of one layer as stacked weight banks: `gate_up`
    `(E, 2*inter, hidden)` (gate rows first) and `down` `(E, hidden, inter)`.
    Runs as two grouped GEMMs over tokens sorted by expert.

    Int8 storage is codes plus float32 per-output-channel scales (kept as uint8
    byte views so `.to(dtype=...)` never casts them). `quantize_int8_` derives
    it from the bf16 banks; `attach_int8_` takes it ready-made from a ComfyUI
    checkpoint, whose convrot files hold the rows rotated by a block
    regular-Hadamard along the input dim (`rot_size`). The rotation is
    orthogonal and self-inverse, so the activations are rotated the same way
    ahead of each GEMM instead of un-rotating the weights."""

    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden = hidden
        self.inter = inter
        self.gate_up = nn.Parameter(torch.empty(num_experts, 2 * inter, hidden))
        self.down = nn.Parameter(torch.empty(num_experts, hidden, inter))
        self.is_int8 = False
        self.rot_size = 1

    def _rot(self, in_features: int) -> int:
        # convrot's block: the group size, capped by the largest power of
        # four dividing the input dim
        return min(self.rot_size, largest_pow4_divisor(in_features))

    @staticmethod
    def fuse_state_dict(state: dict, prefix: str, num_experts: int) -> bool:
        """Stack `{prefix}{e}.{gate,up,down}_proj.weight` entries into the bank
        keys in place (the per-expert keys are removed). Returns whether any
        per-expert keys were found."""
        if f"{prefix}0.gate_proj.weight" not in state:
            return False
        gate_up = torch.stack(
            [
                torch.cat(
                    [state.pop(f"{prefix}{e}.gate_proj.weight"), state.pop(f"{prefix}{e}.up_proj.weight")],
                    dim=0,
                )
                for e in range(num_experts)
            ]
        )
        down = torch.stack([state.pop(f"{prefix}{e}.down_proj.weight") for e in range(num_experts)])
        state[f"{prefix}gate_up"] = gate_up
        state[f"{prefix}down"] = down
        return True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if f"{prefix}gate_up_q" in state_dict:
            # ready-made int8 banks (a ComfyUI checkpoint, see
            # MingImageTextEncoder.convert_state_dict_on_load)
            rot = state_dict.pop(f"{prefix}rot_size", None)
            self.attach_int8_(
                state_dict.pop(f"{prefix}gate_up_q"),
                state_dict.pop(f"{prefix}gate_up_scale"),
                state_dict.pop(f"{prefix}down_q"),
                state_dict.pop(f"{prefix}down_scale"),
                rot_size=int(rot.item()) if rot is not None else 1,
            )
        else:
            # accept the checkpoint's per-expert layout directly
            self.fuse_state_dict(state_dict, prefix, self.num_experts)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def _set_int8(self, name: str, q: torch.Tensor, scale: torch.Tensor):
        if name in self._parameters:
            del self._parameters[name]
        self.register_buffer(f"{name}_q", q.contiguous(), persistent=False)
        self.register_buffer(
            f"{name}_scale",
            scale.detach().float().reshape(q.shape[0], q.shape[1]).contiguous().view(torch.uint8),
            persistent=False,
        )

    @torch.no_grad()
    def quantize_int8_(self, device=None):
        """Per-output-channel symmetric int8 weights; the math runs on
        `device` (the banks return to where they were)."""
        if self.is_int8:
            return
        for name in ("gate_up", "down"):
            param = self._parameters[name]
            home = param.device
            w = param.data.to(device or home, torch.float32)
            scale = w.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
            q = torch.round(w / scale.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
            del w
            self._set_int8(name, q.to(home), scale.to(home))
        self.is_int8 = True
        self.rot_size = 1

    @torch.no_grad()
    def attach_int8_(self, gate_up_q, gate_up_scale, down_q, down_scale, rot_size: int = 1):
        """Ready-made int8 banks: codes `(E, N, K)` with float32 scales
        `(E, N)` or `(E, N, 1)`; `rot_size` > 1 means the rows are rotated
        along K in blocks of that size."""
        expected = {
            "gate_up": ((self.num_experts, 2 * self.inter, self.hidden), gate_up_q),
            "down": ((self.num_experts, self.hidden, self.inter), down_q),
        }
        for name, (shape, q) in expected.items():
            if tuple(q.shape) != shape or q.dtype != torch.int8:
                raise ValueError(f"{name}: expected int8 bank {shape}, got {tuple(q.shape)} {q.dtype}")
        self._set_int8("gate_up", gate_up_q, gate_up_scale)
        self._set_int8("down", down_q, down_scale)
        self.is_int8 = True
        self.rot_size = int(rot_size)

    @torch.no_grad()
    def dequantize_(self, dtype=torch.bfloat16, device=None):
        """Back to full-precision parameters (the rotation undone); the math
        runs on `device`."""
        if not self.is_int8:
            return
        for name in ("gate_up", "down"):
            q = self._buffers.pop(f"{name}_q")
            scale = self._buffers.pop(f"{name}_scale").view(torch.float32)
            self._non_persistent_buffers_set.discard(f"{name}_q")
            self._non_persistent_buffers_set.discard(f"{name}_scale")
            home = q.device
            work = device or home
            w = q.to(work, torch.float32) * scale.to(work).unsqueeze(-1)
            rot = self._rot(w.shape[-1])
            if rot > 1:
                w = rotate(w, rot)  # self-inverse
            self.register_parameter(name, nn.Parameter(w.to(dtype).to(home), requires_grad=False))
        self.is_int8 = False
        self.rot_size = 1

    def bank_names(self):
        if self.is_int8:
            return ("gate_up_q", "gate_up_scale", "down_q", "down_scale")
        return ("gate_up", "down")

    def bank_tensors(self) -> Dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.bank_names()}

    def _bank(self, tensors: Dict[str, torch.Tensor], name: str):
        if self.is_int8:
            return tensors[f"{name}_q"], tensors[f"{name}_scale"].view(torch.float32)
        return tensors[name], None

    def forward_with(self, x_sorted: torch.Tensor, offs: torch.Tensor, tensors: Dict[str, torch.Tensor]):
        """The forward on an explicit set of bank tensors (the offload manager
        hands over staged device copies)."""
        if self.rot_size > 1:
            x_sorted = rotate(x_sorted, self._rot(self.hidden))
        h = grouped_swiglu(x_sorted, *self._bank(tensors, "gate_up"), offs)
        if self.rot_size > 1:
            h = rotate(h, self._rot(self.inter))
        return grouped_linear(h, *self._bank(tensors, "down"), offs)

    def forward(self, x_sorted: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        """`x_sorted` (rows, hidden) grouped by expert, `offs` (E + 1,) int32
        prefix sums of the group sizes -> (rows, hidden)."""
        return self.forward_with(x_sorted, offs, self.bank_tensors())


class FusedExpertsMemoryManager:
    """Layer offloading for an expert bank, in the shape of the toolkit's
    per-layer memory managers: the banks stay pinned on cpu and are staged to
    the compute device for each forward (0.75 GB int8 per layer, the bulk of
    the model). `MemoryManager.detach` restores `_original_forward`."""

    def __init__(self, module: FusedExperts, process_device: torch.device):
        self.module = module
        self.process_device = torch.device(process_device)
        with torch.no_grad():
            for name in module.bank_names():
                tensor = getattr(module, name)
                data = tensor.data if isinstance(tensor, nn.Parameter) else tensor
                data = data.to("cpu")
                if torch.cuda.is_available() and not data.is_pinned():
                    try:
                        data = data.pin_memory()
                    except RuntimeError:
                        pass
                if isinstance(tensor, nn.Parameter):
                    tensor.data = data
                    tensor._is_memory_managed = True
                else:
                    module._buffers[name] = data
        self._original_forward = module.forward

        def _mm_forward(x_sorted, offs):
            device = self.process_device
            if device.type != "cuda":
                return self._original_forward(x_sorted, offs)
            staged = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in module.bank_tensors().items()
            }
            return module.forward_with(x_sorted.to(device), offs.to(device), staged)

        module.forward = _mm_forward

    @classmethod
    def attach(cls, module: FusedExperts, process_device: torch.device):
        if hasattr(module, "_layer_memory_manager"):
            return
        module._layer_memory_manager = cls(module, process_device)


class SparseMoeBlock(nn.Module):
    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.experts = FusedExperts(config.num_experts, config.hidden_size, config.moe_intermediate_size)
        self.gate = Gate(config)
        # MultiRouter: vision-path tokens have their own router
        self.image_gate = Gate(config)
        self.shared_experts = MLP(
            config, config.moe_intermediate_size * config.num_shared_experts
        )

    def forward(self, hidden_states: torch.Tensor, image_mask: Optional[torch.Tensor]):
        bsz, seq_len, dim = hidden_states.shape
        x = hidden_states.reshape(-1, dim)
        topk_idx, topk_weight = self.gate(x)
        if image_mask is not None:
            # both routers always run (the extra gate is one small matmul); no
            # host sync on the mask
            image_idx, image_weight = self.image_gate(x)
            mask = image_mask.reshape(-1, 1)
            topk_idx = torch.where(mask, image_idx, topk_idx)
            topk_weight = torch.where(mask, image_weight, topk_weight)
        y = self.moe_infer(x, topk_idx, topk_weight).view(bsz, seq_len, dim)
        return y + self.shared_experts(hidden_states)

    def moe_infer(self, x, topk_ids, topk_weight):
        num_tokens, top_k = topk_ids.shape
        flat_ids = topk_ids.reshape(-1)
        order = flat_ids.argsort()
        token_idx = order // top_k
        # scatter_add rather than bincount: fixed output size, no host sync,
        # graph-capturable
        counts = torch.zeros(self.experts.num_experts, dtype=torch.int32, device=x.device)
        counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.int32))
        offs = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
        outs = self.experts(x[token_idx], offs)
        weights = topk_weight.reshape(-1)[order].to(outs.dtype)
        y = torch.zeros((num_tokens, x.shape[-1]), device=x.device, dtype=torch.float32)
        y.index_add_(0, token_idx, (outs * weights.unsqueeze(-1)).float())
        return y.to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, heads, n_rep, seq_len, head_dim)
        .reshape(batch, heads * n_rep, seq_len, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.query_key_value = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)

    def forward(self, x, attn_mask, cos, sin):
        bsz, seq_len, _ = x.shape
        qkv = self.query_key_value(x).view(bsz, seq_len, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
        q, k, v = qkv.split([self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=-2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_video_rope(q, k, cos, sin)
        k = repeat_kv(k, self.num_heads // self.num_kv_heads)
        v = repeat_kv(v, self.num_heads // self.num_kv_heads)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.dense(out.transpose(1, 2).reshape(bsz, seq_len, -1))


class DecoderLayer(nn.Module):
    def __init__(self, config: BailingMoeV2Config, layer_idx: int):
        super().__init__()
        self.attention = Attention(config)
        self.is_moe = layer_idx >= config.first_k_dense_replace
        self.mlp = SparseMoeBlock(config) if self.is_moe else MLP(config, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, attn_mask, cos, sin, image_mask):
        x = x + self.attention(self.input_layernorm(x), attn_mask, cos, sin)
        h = self.post_attention_layernorm(x)
        h = self.mlp(h, image_mask) if self.is_moe else self.mlp(h)
        return x + h


class BailingMoeV2Model(nn.Module):
    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = VideoRotaryEmbedding(config)
        # built once (a host->device copy would break CUDA-graph capture)
        self.register_buffer(
            "rope_layout_index",
            torch.tensor(
                video_rope_layout_index(self.rotary_emb.inv_freq.numel() * 2, config.mrope_section),
                dtype=torch.long,
            ),
            persistent=False,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # (B, L, hidden)
        attention_mask: torch.Tensor,  # (B, 1, L, L) bool, True = may attend
        position_ids: torch.Tensor,  # (3, B, L)
        image_mask: Optional[torch.Tensor] = None,  # (B, L) bool, vision-path tokens
        capture_layers: Sequence[int] = (),
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Returns the final (normalized) hidden states and, for each index in
        `capture_layers`, the residual stream entering that layer; the index
        equal to the layer count maps to the normalized output, matching the
        `output_hidden_states` numbering of the reference implementation."""
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        cos, sin = video_rope_layout(cos, sin, self.rope_layout_index)
        capture = set(capture_layers)
        captured: Dict[int, torch.Tensor] = {}
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            if i in capture:
                captured[i] = h
            h = layer(h, attention_mask, cos, sin, image_mask)
        h = self.norm(h)
        if len(self.layers) in capture:
            captured[len(self.layers)] = h
        return h, captured
