"""Local implementation of google/tipsv2-b14-dpt.

Self-contained port of the remote `trust_remote_code=True` model into ai-toolkit.
Includes vision encoder + DPT depth/normals/segmentation heads, with optional
gradient checkpointing on the vision transformer blocks. The text encoder is
intentionally not included — only the dense-prediction stack is used here.

Original remote code: https://huggingface.co/google/tipsv2-b14-dpt
                      https://huggingface.co/google/tipsv2-b14
"""

import functools
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


# ───────────────────────── Vision Transformer ──────────────────────────────


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        image_hw = _make_2tuple(img_size)
        patch_hw = _make_2tuple(patch_size)
        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = (
            image_hw[0] // patch_hw[0],
            image_hw[1] // patch_hw[1],
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph, pw = self.patch_size
        assert h % ph == 0, f"Input height {h} not divisible by patch {ph}"
        assert w % pw == 0, f"Input width {w} not divisible by patch {pw}"
        x = self.proj(x)
        h, w = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, h, w, self.embed_dim)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Use SDPA — drops the manual attention matmul + softmax and supports flash on cuda.
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self, dim: int, init_values: Union[float, torch.Tensor] = 1e-5
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        if keep > 0.0:
            mask.div_(keep)
        return x * mask


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """DINOv2-style ViT used as the TIPSv2 vision backbone."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = 1.0,
        ffn_layer: str = "mlp",
        num_register_tokens: int = 1,
        interpolate_antialias: bool = True,
        interpolate_offset: float = 0.0,
    ):
        super().__init__()
        norm_layer = functools.partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.gradient_checkpointing = False

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if ffn_layer != "mlp":
            raise NotImplementedError(
                f"ffn_layer={ffn_layer!r} not supported in local port"
            )

        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    ffn_layer=Mlp,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        # Maintain weight-key compat with the upstream non-chunked branch.
        self.chunked_blocks = False

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    # ---- gradient checkpointing toggles ------------------------------------

    def gradient_checkpointing_enable(self, **_kwargs) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    enable_gradient_checkpointing = gradient_checkpointing_enable
    disable_gradient_checkpointing = gradient_checkpointing_disable

    # ---- positional embedding / token prep ---------------------------------

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        num_patches = self.pos_embed.shape[1] - 1
        if npatch == num_patches and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        side = int(math.sqrt(num_patches))
        assert num_patches == side * side
        kwargs = {}
        if self.interpolate_offset:
            kwargs["scale_factor"] = (
                float(w0 + self.interpolate_offset) / side,
                float(h0 + self.interpolate_offset) / side,
            )
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, side, side, dim).permute(0, 3, 1, 2),
            mode="bilinear",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]),
                dim=1,
            )
        return x

    # ---- block runner with optional checkpointing --------------------------

    def _run_blocks(
        self, x: torch.Tensor, collect_indices: Optional[Sequence[int]] = None
    ):
        collected = [] if collect_indices is not None else None
        use_ckpt = self.gradient_checkpointing and self.training
        for i, blk in enumerate(self.blocks):
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if collected is not None and i in collect_indices:
                collected.append(x)
        return (x, collected) if collected is not None else x

    # ---- public forwards ---------------------------------------------------

    def forward_features(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> dict:
        x = self.prepare_tokens_with_masks(x, masks)
        x = self._run_blocks(x)
        x_norm = self.norm(x)
        return {
            "x_norm_1st_clstoken": x_norm[:, :1],
            "x_norm_2nd_clstoken": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        x_in = x
        x = self.prepare_tokens_with_masks(x)
        total = len(self.blocks)
        indices = list(range(total - n, total)) if isinstance(n, int) else list(n)
        _, outputs = self._run_blocks(x, collect_indices=indices)
        # Preserve the requested ordering.
        order = {idx: pos for pos, idx in enumerate(sorted(indices))}
        outputs = [outputs[order[idx]] for idx in indices]
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            b, _, w, h = x_in.shape
            outputs = [
                out.reshape(b, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, x: torch.Tensor, is_training: bool = False):
        ret = self.forward_features(x)
        if is_training:
            return ret
        return (
            self.head(ret["x_norm_1st_clstoken"]),
            self.head(ret["x_norm_2nd_clstoken"]),
            ret["x_norm_patchtokens"],
        )


def _vit_base(patch_size: int = 14, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=1,
        **kwargs,
    )


# ───────────────────────────── DPT heads ───────────────────────────────────


class PreActResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = False, expand: bool = False):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.residual_unit = PreActResidualConvUnit(features)
        self.main_unit = PreActResidualConvUnit(features)
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, bias=True)

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.has_residual and residual is not None:
            if residual.shape != x.shape:
                residual = F.interpolate(
                    residual, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            residual = self.residual_unit(residual)
            x = x + residual
        x = self.main_unit(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.out_conv(x)
        return x


class ReassembleBlocks(nn.Module):
    def __init__(
        self,
        input_embed_dim: int = 1024,
        out_channels: Tuple[int, ...] = (128, 256, 512, 1024),
        readout_type: str = "project",
    ):
        super().__init__()
        self.readout_type = readout_type
        self.out_projections = nn.ModuleList(
            [nn.Conv2d(input_embed_dim, ch, 1) for ch in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
            ]
        )
        if readout_type == "project":
            self.readout_projects = nn.ModuleList(
                [nn.Linear(2 * input_embed_dim, input_embed_dim) for _ in out_channels]
            )

    def forward(self, features):
        out = []
        for i, (cls_token, x) in enumerate(features):
            B, D, H, W = x.shape
            if self.readout_type == "project":
                x_flat = x.flatten(2).transpose(1, 2)
                readout = cls_token.unsqueeze(1).expand(-1, x_flat.shape[1], -1)
                x_cat = torch.cat([x_flat, readout], dim=-1)
                x_proj = F.gelu(self.readout_projects[i](x_cat))
                x = x_proj.transpose(1, 2).reshape(B, D, H, W)
            x = self.out_projections[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


def _build_fusion_stack(channels: int) -> nn.ModuleList:
    return nn.ModuleList(
        [
            FeatureFusionBlock(channels, has_residual=False),
            FeatureFusionBlock(channels, has_residual=True),
            FeatureFusionBlock(channels, has_residual=True),
            FeatureFusionBlock(channels, has_residual=True),
        ]
    )


class _DPTHeadBase(nn.Module):
    """Shared reassemble + fuse + project trunk used by all three task heads."""

    def __init__(
        self,
        input_embed_dim: int,
        channels: int,
        post_process_channels: Tuple[int, ...],
        readout_type: str,
    ):
        super().__init__()
        self.reassemble = ReassembleBlocks(
            input_embed_dim=input_embed_dim,
            out_channels=post_process_channels,
            readout_type=readout_type,
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(ch, channels, 3, padding=1, bias=False)
                for ch in post_process_channels
            ]
        )
        self.fusion_blocks = _build_fusion_stack(channels)
        self.project = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def _trunk(self, intermediate_features) -> torch.Tensor:
        x = self.reassemble(intermediate_features)
        x = [self.convs[i](feat) for i, feat in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, 4):
            out = self.fusion_blocks[i](out, residual=x[-(i + 1)])
        return self.project(out)


class DPTDepthHead(_DPTHeadBase):
    def __init__(
        self,
        input_embed_dim: int = 1024,
        channels: int = 256,
        post_process_channels: Tuple[int, ...] = (128, 256, 512, 1024),
        readout_type: str = "project",
        num_depth_bins: int = 256,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
    ):
        super().__init__(input_embed_dim, channels, post_process_channels, readout_type)
        self.num_depth_bins = num_depth_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_head = nn.Linear(channels, num_depth_bins)

    def forward(self, intermediate_features, image_size=None) -> torch.Tensor:
        out = F.relu(self._trunk(intermediate_features))
        out = out.permute(0, 2, 3, 1)
        out = self.depth_head(out)
        bin_centers = torch.linspace(
            self.min_depth,
            self.max_depth,
            self.num_depth_bins,
            device=out.device,
            dtype=out.dtype,
        )
        out = F.relu(out) + self.min_depth
        out_norm = out / out.sum(dim=-1, keepdim=True)
        depth = torch.einsum("bhwn,n->bhw", out_norm, bin_centers).unsqueeze(1)
        if image_size is not None:
            depth = F.interpolate(
                depth, size=image_size, mode="bilinear", align_corners=False
            )
        return depth


class DPTNormalsHead(_DPTHeadBase):
    def __init__(
        self,
        input_embed_dim: int = 1024,
        channels: int = 256,
        post_process_channels: Tuple[int, ...] = (128, 256, 512, 1024),
        readout_type: str = "project",
    ):
        super().__init__(input_embed_dim, channels, post_process_channels, readout_type)
        self.normals_head = nn.Linear(channels, 3)

    def forward(self, intermediate_features, image_size=None) -> torch.Tensor:
        out = self._trunk(intermediate_features)
        out = out.permute(0, 2, 3, 1)
        out = self.normals_head(out)
        out = F.normalize(out, p=2, dim=-1)
        out = out.permute(0, 3, 1, 2)
        if image_size is not None:
            out = F.interpolate(
                out, size=image_size, mode="bilinear", align_corners=False
            )
        return out


class DPTSegmentationHead(_DPTHeadBase):
    def __init__(
        self,
        input_embed_dim: int = 1024,
        channels: int = 256,
        post_process_channels: Tuple[int, ...] = (128, 256, 512, 1024),
        readout_type: str = "project",
        num_classes: int = 150,
    ):
        super().__init__(input_embed_dim, channels, post_process_channels, readout_type)
        self.segmentation_head = nn.Linear(channels, num_classes)

    def forward(self, intermediate_features, image_size=None) -> torch.Tensor:
        out = self._trunk(intermediate_features)
        out = out.permute(0, 2, 3, 1)
        out = self.segmentation_head(out)
        out = out.permute(0, 3, 1, 2)
        if image_size is not None:
            out = F.interpolate(
                out, size=image_size, mode="bilinear", align_corners=False
            )
        return out


# ───────────────────────────── Top-level model ─────────────────────────────


@dataclass
class TIPSv2DPTOutput:
    depth: Optional[torch.Tensor] = None
    normals: Optional[torch.Tensor] = None
    segmentation: Optional[torch.Tensor] = None


# Hard-coded config for the b14-dpt variant — matches config.json on the hub.
_B14_DPT_CONFIG = dict(
    backbone_repo="google/tipsv2-b14",
    embed_dim=768,
    channels=256,
    post_process_channels=(96, 192, 384, 768),
    block_indices=(2, 5, 8, 11),
    readout_type="project",
    num_depth_bins=256,
    min_depth=1e-3,
    max_depth=10.0,
    num_seg_classes=150,
    # Vision encoder
    vision_fn="vit_base",
    patch_size=14,
    img_size=448,
    init_values=1.0,
    num_register_tokens=1,
    ffn_layer="mlp",
)


class TIPSv2DPTModel(nn.Module):
    """TIPSv2 DPT dense-prediction model (depth, normals, segmentation).

    Use :meth:`from_pretrained` to load weights for `google/tipsv2-b14-dpt`.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        cfg = dict(_B14_DPT_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg

        builders = {"vit_base": _vit_base}
        if cfg["vision_fn"] not in builders:
            raise NotImplementedError(f"vision_fn={cfg['vision_fn']!r} not supported")

        self.vision_encoder = builders[cfg["vision_fn"]](
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            ffn_layer=cfg["ffn_layer"],
            init_values=cfg["init_values"],
            interpolate_antialias=True,
            interpolate_offset=0.0,
        )

        ppc = tuple(cfg["post_process_channels"])
        self.depth_head = DPTDepthHead(
            input_embed_dim=cfg["embed_dim"],
            channels=cfg["channels"],
            post_process_channels=ppc,
            readout_type=cfg["readout_type"],
            num_depth_bins=cfg["num_depth_bins"],
            min_depth=cfg["min_depth"],
            max_depth=cfg["max_depth"],
        )
        self.normals_head = DPTNormalsHead(
            input_embed_dim=cfg["embed_dim"],
            channels=cfg["channels"],
            post_process_channels=ppc,
            readout_type=cfg["readout_type"],
        )
        self.segmentation_head = DPTSegmentationHead(
            input_embed_dim=cfg["embed_dim"],
            channels=cfg["channels"],
            post_process_channels=ppc,
            readout_type=cfg["readout_type"],
            num_classes=cfg["num_seg_classes"],
        )

    # ---- properties + checkpointing ---------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        """Enable gradient checkpointing on the vision transformer blocks."""
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.vision_encoder.gradient_checkpointing_disable()

    enable_gradient_checkpointing = gradient_checkpointing_enable
    disable_gradient_checkpointing = gradient_checkpointing_disable

    # ---- core inference path ----------------------------------------------

    def _extract_intermediate(self, pixel_values: torch.Tensor):
        intermediate = self.vision_encoder.get_intermediate_layers(
            pixel_values,
            n=tuple(self.config["block_indices"]),
            reshape=True,
            return_class_token=True,
            norm=True,
        )
        # Returned as (cls_token, patch_feats) tuples to match the remote API.
        return [(cls_tok, patch_feat) for patch_feat, cls_tok in intermediate]

    def predict_depth(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h, w = pixel_values.shape[2:]
        return self.depth_head(
            self._extract_intermediate(pixel_values), image_size=(h, w)
        )

    def predict_normals(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h, w = pixel_values.shape[2:]
        return self.normals_head(
            self._extract_intermediate(pixel_values), image_size=(h, w)
        )

    def predict_segmentation(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h, w = pixel_values.shape[2:]
        return self.segmentation_head(
            self._extract_intermediate(pixel_values), image_size=(h, w)
        )

    def forward(self, pixel_values: torch.Tensor) -> TIPSv2DPTOutput:
        h, w = pixel_values.shape[2:]
        feats = self._extract_intermediate(pixel_values)
        return TIPSv2DPTOutput(
            depth=self.depth_head(feats, image_size=(h, w)),
            normals=self.normals_head(feats, image_size=(h, w)),
            segmentation=self.segmentation_head(feats, image_size=(h, w)),
        )

    # ---- loader -----------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "google/tipsv2-b14-dpt",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
    ) -> "TIPSv2DPTModel":
        """Build the model and load weights from the hub.

        Pulls the DPT head weights from ``model_id`` (default
        ``google/tipsv2-b14-dpt``) and the vision-encoder weights from the
        backbone repo specified in the DPT config.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if model_id != "google/tipsv2-b14-dpt":
            raise NotImplementedError(
                f"Local TIPSv2DPTModel only supports 'google/tipsv2-b14-dpt'; got {model_id!r}"
            )

        model = cls()

        dpt_ckpt = hf_hub_download(model_id, "model.safetensors", cache_dir=cache_dir)
        dpt_state = load_file(dpt_ckpt)

        backbone_ckpt = hf_hub_download(
            model.config["backbone_repo"],
            "model.safetensors",
            cache_dir=cache_dir,
        )
        backbone_state = load_file(backbone_ckpt)
        # Backbone repo stores both vision and text encoders — keep only vision_encoder.*.
        backbone_state = {
            k: v for k, v in backbone_state.items() if k.startswith("vision_encoder.")
        }

        merged = {**dpt_state, **backbone_state}
        missing, unexpected = model.load_state_dict(merged, strict=False)
        if missing:
            print(
                f"[tipsv2] Missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}"
            )
        if unexpected:
            print(
                f"[tipsv2] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
            )

        model.to(device=device, dtype=dtype)
        return model
