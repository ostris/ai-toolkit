"""MiniMax-H3 video VAE -- causal 3D-CNN encoder + non-causal 36-layer ViT decoder.

16x spatial / 4x temporal compression, 24 latent channels, per-channel
latents_mean/latents_std normalization. The temporal geometry is fixed by
``clip_length`` (17 pixel frames per encoder chunk) and ``token_drop`` (3
trailing latent frames dropped after encoding): ``17n + 5`` pixel frames
map to ``5n + 2`` latent frames. ``decode`` mirrors the chunking -- 5-token
chunks with a 2-token overlap, cross-faded over 5 pixel frames. A single
frame (image / i2v keyframe) skips the temporal chunking entirely and maps
to a single latent frame.

The reference pixel convention is ImageNet-normalized RGB over [0, 1];
``encode``/``decode`` here speak ai-toolkit's [-1, 1] and convert inside.
Spatial tiling (256 px tiles, >= 64 px overlap, linearly blended) is ON by
default for both encode and decode -- the model was released that way and
the released frames are the blended-tile ones. The checkpoint is float32
and the released decode recipe is fp16 autocast over the fp32 weights, with
the token embedder, output projection, and every norm kept in fp32.
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
]

LATENTS_STD = [
    1.2223774194717407,
    1.2767263650894165,
    1.68317747116088865,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.96531379222869875,
    1.05698859691619875,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049925,
    0.7197399735450745,
    0.69362932443618775,
    2.961095094680786,
    2.7694199085235595,
    3.0496184825897215,
    2.1088054180145265,
    3.276226282119751,
    3.1627357006073,
    2.28168129920959475,
    2.6127843856811525,
]


# ---------------------------------------------------------------------------
# causal 3D-CNN encoder
# ---------------------------------------------------------------------------


class CausalConv3d(nn.Conv3d):
    """Conv3d with symmetric reflect spatial padding and causal (front-only,
    zeros) temporal padding of ``kernel_t - 1`` frames."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, spatial_pad=0):
        super().__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.spatial_pad = spatial_pad

    def forward(self, x):
        if self.spatial_pad > 0:
            p = self.spatial_pad
            x = F.pad(x, (p, p, p, p, 0, 0), mode="reflect")
        t_pad = self.kernel_size[0] - 1
        if t_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, t_pad, 0))
        return F.conv3d(x, self.weight, self.bias, stride=self.stride)


class FrameGroupNorm(nn.GroupNorm):
    """GroupNorm with statistics computed per frame (``use_t_isolated_gn``):
    the temporal axis is folded into the batch axis."""

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = super().forward(x)
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class ResnetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups=32):
        super().__init__()
        self.norm1 = FrameGroupNorm(norm_num_groups, in_channels, eps=1e-6)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, spatial_pad=1)
        self.norm2 = FrameGroupNorm(norm_num_groups, out_channels, eps=1e-6)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, spatial_pad=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, 1)
        else:
            self.nin_shortcut = None

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class Downsample3d(nn.Module):
    """Strided 3x3x3 conv; a spatial stride of 2 is preceded by an asymmetric
    bottom/right reflect pad of 1 so the output is exactly ceil(size / 2)."""

    def __init__(self, channels, time_stride, space_stride):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            channels, channels, 3, stride=(time_stride, space_stride, space_stride)
        )

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class Encoder3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        block_out_channels,
        layers_per_block,
        space_down,
        time_down,
        norm_num_groups,
    ):
        super().__init__()
        block_in = (block_out_channels[0],) + tuple(block_out_channels[:-1])

        self.conv_in = CausalConv3d(in_channels, block_in[0], 3, spatial_pad=1)

        self.down = nn.ModuleList()
        for i in range(len(block_out_channels)):
            level = nn.Module()
            level.block = nn.ModuleList(
                ResnetBlock3d(
                    block_in[i] if j == 0 else block_out_channels[i],
                    block_out_channels[i],
                    norm_num_groups=norm_num_groups,
                )
                for j in range(layers_per_block)
            )
            if space_down[i] * time_down[i] > 1:
                level.downsample = Downsample3d(
                    block_out_channels[i], time_down[i], space_down[i]
                )
            self.down.append(level)

        self.norm_out = FrameGroupNorm(
            norm_num_groups, block_out_channels[-1], eps=1e-6
        )
        self.conv_out = CausalConv3d(
            block_out_channels[-1], out_channels, 3, spatial_pad=1
        )

    def forward(self, x):
        h = self.conv_in(x)
        for level in self.down:
            for block in level.block:
                h = block(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


# ---------------------------------------------------------------------------
# non-causal ViT decoder
# ---------------------------------------------------------------------------


class RMSNormFP32(nn.RMSNorm):
    """The reference normalizes in float32 regardless of the compute dtype
    (or the storage dtype — the Comfy repack ships this VAE in fp16)."""

    def forward(self, x):
        weight = None if self.weight is None else self.weight.float()
        out = F.rms_norm(x.float(), self.normalized_shape, weight, self.eps)
        return out.to(x.dtype)


class RotaryEmbedding3d(nn.Module):
    """3-axis rotary table. Coordinates are length-normalized to [-1, 1) per
    axis and scaled by 2*pi; the per-axis angles are concatenated and then
    duplicated, so the first ``rope_dim_ratio * head_dim`` channels of every
    head are rotated (rotate-half convention)."""

    def __init__(self, dim, theta=100.0, num_axes=3):
        super().__init__()
        inv_freq = 1.0 / theta ** torch.arange(
            0, 1, 2 * num_axes / dim, dtype=torch.float32
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        angles = (
            2.0
            * math.pi
            * position_ids[:, :, :, None].float()
            * self.inv_freq[None, None, None, :]
        )
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)  # (B, S, 1, rot_dim)
        return angles.cos(), angles.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x, cos, sin):
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat([x_rot, x_pass], dim=-1)


class Attention(nn.Module):
    def __init__(self, heads, head_dim, eps=1e-5):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.norm_q = RMSNormFP32(head_dim, eps=eps, elementwise_affine=False)
        self.norm_k = RMSNormFP32(head_dim, eps=eps, elementwise_affine=False)
        self.to_qkv = nn.Linear(inner, inner * 3, bias=True)
        self.to_out = nn.Linear(inner, inner, bias=True)

    def forward(self, x, rotary_emb):
        b, s, _ = x.shape
        # per-head-interleaved qkv layout: each head's 3 * head_dim slab is
        # split into q | k | v, NOT [Q_all | K_all | V_all]
        qkv = self.to_qkv(x).view(b, s, self.heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.norm_q(q)
        k = self.norm_k(k)
        cos, sin = rotary_emb
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return self.to_out(out.transpose(1, 2).reshape(b, s, -1))


class FeedForward(nn.Module):
    """Gated SiLU: w1 output chunks into gate | value, silu(gate) * value."""

    def __init__(self, dim, mult=4):
        super().__init__()
        inner = dim * mult
        self.w1 = nn.Linear(dim, inner * 2, bias=True)
        self.w2 = nn.Linear(inner, dim, bias=True)

    def forward(self, x):
        gate, x = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * x)


class TransformerBlock(nn.Module):
    def __init__(self, heads, head_dim, ffn_mult=4, eps=1e-5):
        super().__init__()
        dim = heads * head_dim
        self.norm1 = RMSNormFP32(dim, eps=eps, elementwise_affine=True)
        self.attn = Attention(heads, head_dim, eps=eps)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = RMSNormFP32(dim, eps=eps, elementwise_affine=True)
        self.ff = FeedForward(dim, mult=ffn_mult)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x, rotary_emb):
        x = x + self.attn(self.norm1(x), rotary_emb) * self.scale1
        return x + self.ff(self.norm2(x)) * self.scale2


class ViTDecoder3d(nn.Module):
    """Every latent voxel is one token; 4 learned register tokens plus one
    all-zero token are appended (all at rope coordinate 0), attended over with
    full self-attention, and dropped before the patch projection expands each
    token into a patch_size_t x patch_size x patch_size pixel block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        patch_size,
        patch_size_t,
        num_layers,
        heads,
        head_dim,
        num_register_tokens=4,
        ffn_mult=4,
        rope_theta=100.0,
        rope_dim_ratio=0.75,
        eps=1e-5,
    ):
        super().__init__()
        dim = heads * head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.rope = RotaryEmbedding3d(int(head_dim * rope_dim_ratio), theta=rope_theta)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        # unused at inference; kept so the checkpoint loads strict
        self.register_buffer("mask_token", torch.zeros(1, 1, dim))
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(heads, head_dim, ffn_mult=ffn_mult, eps=eps)
            for _ in range(num_layers)
        )
        self.norm_out = nn.LayerNorm(dim, eps=eps)
        self.proj_out = nn.Linear(
            dim, out_channels * patch_size_t * patch_size * patch_size
        )

        # on by default: only engages when gradients are enabled (pixel-space
        # losses through decode); no_grad sampling never takes the branch
        self.gradient_checkpointing = True

    def forward(self, z):
        b, c, t, h, w = z.shape
        tokens = z.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)

        # released recipe: token embedder / output projection / norms compute
        # fp32 even when the matmuls run fp16 (autocast or fp16 storage); the
        # block stack then runs at the storage dtype
        with torch.autocast(device_type=z.device.type, enabled=False):
            x = F.linear(
                tokens.float(),
                self.x_embedder.weight.float(),
                self.x_embedder.bias.float(),
            )
        x = x.to(self.x_embedder.weight.dtype)
        num_patches = x.shape[1]

        x = torch.cat(
            [
                x,
                self.register_tokens.expand(b, -1, -1).to(x.dtype),
                torch.zeros_like(x[:, :1]),
            ],
            dim=1,
        )

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=z.device) / size)
            - 1.0
            for size in (t, h, w)
        ]
        position_ids = torch.stack(
            torch.meshgrid(*grids, indexing="ij"), dim=-1
        ).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(b, -1, -1)
        suffix_ids = position_ids.new_zeros((b, self.num_register_tokens + 1, 3))
        with torch.autocast(device_type=z.device.type, enabled=False):
            rotary_emb = self.rope(torch.cat([position_ids, suffix_ids], dim=1))

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # pixel-space losses backprop through all 36 blocks; without
                # checkpointing the stored activations OOM at video sizes.
                # Inference runs under no_grad, where this branch is never
                # taken, so sampling pays nothing.
                x = checkpoint(block, x, rotary_emb, use_reentrant=False)
            else:
                x = block(x, rotary_emb)

        with torch.autocast(device_type=z.device.type, enabled=False):
            x = F.layer_norm(
                x.float(),
                self.norm_out.normalized_shape,
                self.norm_out.weight.float(),
                self.norm_out.bias.float(),
                self.norm_out.eps,
            )
            x = F.linear(x, self.proj_out.weight.float(), self.proj_out.bias.float())
        x = x[:, :num_patches]

        pt, ps = self.patch_size_t, self.patch_size
        x = x.view(b, t, h, w, self.out_channels, pt, ps, ps)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return x.reshape(b, self.out_channels, t * pt, h * ps, w * ps)


# ---------------------------------------------------------------------------
# full VAE
# ---------------------------------------------------------------------------


class MiniMaxH3VideoVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        latent_channels=24,
        block_out_channels=(128, 256, 256, 512, 512, 1024),
        layers_per_block=2,
        space_down=(2, 2, 2, 2, 1, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        norm_num_groups=32,
        decoder_num_layers=36,
        decoder_heads=32,
        decoder_head_dim=64,
        decoder_num_register_tokens=4,
        decoder_ffn_mult=4,
        rope_theta=100.0,
        rope_dim_ratio=0.75,
        clip_length=17,
        token_drop=3,
        latents_mean=None,
        latents_std=None,
        tile_size=256,
        tile_overlap_min=64,
        tiling=True,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_compression = int(math.prod(space_down))
        self.temporal_compression = int(math.prod(time_down))
        self.clip_length = clip_length
        self.token_drop = token_drop

        # clip_length is not a multiple of the temporal ratio, so each decoded
        # chunk carries an implicit leading pad; token_drop leaves an overlap
        # between consecutive chunks that gets cross-faded in pixel space
        self.frame_pre_padding = (-clip_length) % self.temporal_compression
        self.tokens_chunk_size = math.ceil(clip_length / self.temporal_compression)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(
            self.token_overlap * self.temporal_compression - self.frame_pre_padding, 0
        )

        self.use_tiling = tiling
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min

        self.encoder = Encoder3d(
            in_channels=in_channels,
            out_channels=2 * latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            space_down=space_down,
            time_down=time_down,
            norm_num_groups=norm_num_groups,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)
        self.decoder = ViTDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=self.spatial_compression,
            patch_size_t=self.temporal_compression,
            num_layers=decoder_num_layers,
            heads=decoder_heads,
            head_dim=decoder_head_dim,
            num_register_tokens=decoder_num_register_tokens,
            ffn_mult=decoder_ffn_mult,
            rope_theta=rope_theta,
            rope_dim_ratio=rope_dim_ratio,
        )

        if latents_mean is None:
            latents_mean = (
                LATENTS_MEAN
                if latent_channels == len(LATENTS_MEAN)
                else [0.0] * latent_channels
            )
        if latents_std is None:
            latents_std = (
                LATENTS_STD
                if latent_channels == len(LATENTS_STD)
                else [1.0] * latent_channels
            )
        self.register_buffer(
            "latents_mean",
            torch.tensor(latents_mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(latents_std, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.decoder.gradient_checkpointing = enable

    def disable_gradient_checkpointing(self):
        self.enable_gradient_checkpointing(False)

    @staticmethod
    def latent_frames(num_pixel_frames: int) -> int:
        """17n + 5 pixel frames -> 5n + 2 latent frames; 1 -> 1 (keyframe)."""
        if num_pixel_frames == 1:
            return 1
        return 5 * math.ceil(num_pixel_frames / 17) - 3

    @staticmethod
    def pixel_frames(num_latent_frames: int) -> int:
        """5n + 2 latent frames -> 17n + 5 pixel frames; 1 -> 1 (keyframe)."""
        if num_latent_frames == 1:
            return 1
        n, rem = divmod(num_latent_frames - 2, 5)
        if rem != 0 or n < 0:
            raise ValueError(
                f"latent frame count must be 5n + 2 or 1, got {num_latent_frames}"
            )
        return 17 * n + 5

    # -- spatial tiling ------------------------------------------------------

    def _split_tiles(self, length):
        """Lay tile_size-wide tiles over length pixels; the slack beyond the
        minimum overlap is spread round-robin in whole spatial_compression
        steps so every tile boundary stays latent-aligned."""
        tile_size = self.tile_size
        if tile_size >= length:
            return [0], [length], []

        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - self.tile_overlap_min * (num_tiles - 1) < length:
            num_tiles += 1

        overlaps = [self.tile_overlap_min] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression

        starts = [0]
        for i in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[i])
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * b.ndim
        shape[dim] = blend_extent
        weight_b = (positions / blend_extent).view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * (1 - weight_b) + b[tuple(slice_b)] * weight_b

        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(self, rows, y_overlaps, x_overlaps):
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(rows[i - 1][j], tile, y_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, x_overlaps[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., : -y_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -x_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode_clip(self, x):
        """Encode one temporal clip to moments, spatially tiled when enabled."""
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x))

        y_starts, y_lens, y_overlaps = self._split_tiles(x.shape[-2])
        x_starts, x_lens, x_overlaps = self._split_tiles(x.shape[-1])
        rows = []
        for i_pos, i_len in zip(y_starts, y_lens):
            row = []
            for j_pos, j_len in zip(x_starts, x_lens):
                tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                row.append(self.quant_conv(self.encoder(tile)))
            rows.append(row)
        ratio = self.spatial_compression
        return self._stitch_tiles(
            rows, [o // ratio for o in y_overlaps], [o // ratio for o in x_overlaps]
        )

    def _decode_clip(self, z):
        """Decode one temporal clip; tiles are laid out in pixel space and
        mapped back onto the latent grid, blended in pixel space."""
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))

        ratio = self.spatial_compression
        y_starts, y_lens, y_overlaps = self._split_tiles(z.shape[-2] * ratio)
        x_starts, x_lens, x_overlaps = self._split_tiles(z.shape[-1] * ratio)
        rows = []
        for i_pos, i_len in zip(y_starts, y_lens):
            row = []
            for j_pos, j_len in zip(x_starts, x_lens):
                tile = z[
                    ...,
                    i_pos // ratio : (i_pos + i_len) // ratio,
                    j_pos // ratio : (j_pos + j_len) // ratio,
                ]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    # -- temporal chunking ---------------------------------------------------

    def _encode_video(self, x):
        if x.shape[2] % self.clip_length != 0:
            pad = x[:, :, -1:].repeat(1, 1, (-x.shape[2]) % self.clip_length, 1, 1)
            x = torch.cat([x, pad], dim=2)
        moments = torch.cat(
            [
                self._encode_clip(
                    x[:, :, i * self.clip_length : (i + 1) * self.clip_length]
                )
                for i in range(x.shape[2] // self.clip_length)
            ],
            dim=2,
        )
        if self.token_drop > 0:
            moments = moments[:, :, : -self.token_drop]
        return moments

    def _decode_video(self, z):
        tcs = self.tokens_chunk_size
        ratio_t = self.temporal_compression
        chunk_frames = tcs * ratio_t
        split_count = 2 if self.token_drop > 0 else 1

        num_tokens = z.shape[2] + self.token_drop
        pad_tokens = (-num_tokens) % tcs
        num_chunks = (num_tokens + pad_tokens) // tcs - (split_count - 1)
        if num_chunks < 1:
            # too few tokens for one chunk (e.g. the 2-latent minimum clip)
            pad_tokens += tcs
            num_chunks += 1
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        decoded = []
        overlap = None
        for i in range(num_chunks):
            start = i * tcs
            clip = self._decode_clip(z[:, :, start : start + tcs + self.token_overlap])
            for j in range(split_count):
                part = clip[:, :, j * chunk_frames : (j + 1) * chunk_frames]
                part = part[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        part = self._blend(overlap, part, self.frame_overlap, dim=-3)
                    decoded.append(part)
                else:
                    overlap = part
        if overlap is not None:
            decoded.append(overlap)
        dec = torch.cat(decoded, dim=2)

        if pad_tokens > 0:
            # repeated latent frames produced trailing pixel frames that were
            # never requested; a chunk's last token only covers
            # clip_length % ratio_t frames, the others cover ratio_t
            intra_tail = self.clip_length % ratio_t
            before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (before_pad + k) % tcs == 0 else ratio_t
                for k in range(pad_tokens)
            )
            dec = dec[:, :, :-pad_frames]
        return dec

    # -- public interface ----------------------------------------------------

    def encode(self, pixels, sample=True, generator=None, fp16_round=False):
        """pixels (B, 3, T, H, W) in [-1, 1], T == 17n + 5, or T == 1 (single
        keyframe: spatial encode only, no temporal chunking). Returns
        normalized latents (B, latent_channels, t, h, w): posterior sampled
        (or the mean when sample=False), optionally rounded through fp16
        BEFORE the (z - latents_mean) / latents_std normalization
        (fp16_round=True is the released first-frame-conditioning recipe)."""
        x = pixels
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = (x.float() + 1.0) * 0.5
        x = (x - self.pixel_mean) / self.pixel_std
        x = x.to(self.dtype)

        if x.shape[2] == 1:
            moments = self._encode_clip(x)[:, :, -1:]
        else:
            moments = self._encode_video(x)

        mean, logvar = moments.float().chunk(2, dim=1)
        if sample:
            std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
            noise = torch.randn(
                mean.shape,
                generator=generator,
                device=generator.device if generator is not None else mean.device,
                dtype=mean.dtype,
            ).to(mean.device)
            z = mean + std * noise
        else:
            z = mean

        if fp16_round:
            z = z.to(torch.float16).float()
        shape = (1, -1, 1, 1, 1)
        return (z - self.latents_mean.view(shape)) / self.latents_std.view(shape)

    def decode(self, latents, autocast_fp16=True):
        """Normalized latents (B, latent_channels, t, h, w) -> pixels
        (B, 3, T, H, W) in [-1, 1]. Runs under fp16 autocast over the fp32
        weights on cuda (the released recipe) unless autocast_fp16=False."""
        shape = (1, -1, 1, 1, 1)
        z = latents.float() * self.latents_std.view(shape) + self.latents_mean.view(
            shape
        )

        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if autocast_fp16 and z.is_cuda
            else nullcontext()
        )
        with ctx:
            if z.shape[2] == 1:
                # a lone temporal token is OOD for the chunk-trained ViT
                # decoder (visibly patchy output, ~16 dB vs ~35 dB roundtrip).
                # Decode it as the first latent of a 2-latent clip instead:
                # the first latent of a chunk covers exactly pixel frame 0 in
                # the (1, 4, 4, 4, 4) grouping, so that frame is a clean,
                # in-distribution single-frame decode.
                dec = self._decode_video(torch.cat([z, z], dim=2))[:, :, :1]
            else:
                dec = self._decode_video(z)

        # out-of-place so gradients can flow through decode (pixel-space losses)
        dec = dec.float() * self.pixel_std + self.pixel_mean
        return dec.clamp(0.0, 1.0) * 2.0 - 1.0
