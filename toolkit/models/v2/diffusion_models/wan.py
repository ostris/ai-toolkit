import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel

from .._mixin import OstrisModelMixin


def patch_wan_patch_embedding(transformer) -> bool:
    """Wan's patch_embedding is a Conv3d whose kernel == stride (non-overlapping
    patchify), i.e. a plain linear projection of each flattened patch. bf16
    Conv3d has no fast cuDNN kernel and falls back to a slow, GPU-underutilizing
    path — swap the forward for the exactly-equivalent F.linear (a GEMM),
    returning the conv's own [b, out, t', h', w'] layout so it stays a drop-in.
    The weight is read lazily so this survives later .to(device)/dtype moves."""
    conv = getattr(transformer, "patch_embedding", None)
    if not isinstance(conv, torch.nn.Conv3d):
        return False
    if (
        tuple(conv.kernel_size) != tuple(conv.stride)
        or any(p != 0 for p in conv.padding)
        or any(d != 1 for d in conv.dilation)
        or conv.groups != 1
    ):
        return False

    def fast_forward(x, _conv=conv):
        weight = _conv.weight
        out_c = weight.shape[0]
        kt, kh, kw = _conv.kernel_size
        b, c, t, h, w = x.shape
        tp, hp, wp = t // kt, h // kh, w // kw
        # patchify with (c, kt, kh, kw) fastest — matches weight.reshape below
        x = x.reshape(b, c, tp, kt, hp, kh, wp, kw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, tp * hp * wp, -1)
        w2d = weight.reshape(out_c, -1)
        x = F.linear(x.to(w2d.dtype), w2d, _conv.bias)
        return x.transpose(1, 2).reshape(b, out_c, tp, hp, wp)

    conv.forward = fast_forward
    return True


class WanTransformer3DModel(DiffusersWanTransformer3DModel, OstrisModelMixin):
    """Wan 2.1/2.2 video DiT (wan22 loads two of these into its dual wrapper)."""

    aitk_subfolder = "transformer"

    aitk_comfy_repo = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
    # comfy wan files use the original key layout; convert_state_dict_on_load
    # renames them (pure substring renames, so the legacy scaled_fp8 weight /
    # scale keys ride along with their modules). wan2.2 A14B checkpoints hold
    # two DiTs, keyed by (repo, subfolder): transformer = high noise,
    # transformer_2 = low noise. wan2.1 entries override the comfy repo.
    aitk_comfy_weight_names = {
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers": [
            "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        ],
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "transformer"): [
            "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        ],
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "transformer_2"): [
            "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "transformer"): [
            "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        ],
        ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "transformer_2"): [
            "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        # the UI defaults point at the ai-toolkit bf16 repacks; same comfy files
        ("ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16", "transformer"): [
            "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        ],
        ("ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16", "transformer_2"): [
            "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        ("ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16", "transformer"): [
            "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        ],
        ("ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16", "transformer_2"): [
            "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {
            "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "files": [
                "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
                "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
            ],
        },
        "Wan-AI/Wan2.1-T2V-14B-Diffusers": {
            "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "files": [
                "split_files/diffusion_models/wan2.1_t2v_14B_fp8_scaled.safetensors",
                "split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors",
                "split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors",
            ],
        },
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": {
            "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "files": [
                "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors",
                "split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors",
                "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors",
            ],
        },
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": {
            "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "files": [
                "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp8_scaled.safetensors",
                "split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors",
                "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors",
            ],
        },
    }

    # the comfy fp8/scaled_fp8 wan files mix fp16 conv biases with fp32
    # tables; wan's forward assumes one uniform dtype, so cast the
    # non-quantized tensors to the load dtype (the old holder did this with a
    # blanket .to(device, dtype) after load)
    aitk_cast_quantized_load = True

    @classmethod
    def get_transformer_block_names(cls):
        return ["blocks"]

    def aitk_post_load(self, **kwargs):
        # slow bf16 Conv3d patchify -> equivalent GEMM, before quantize/offload
        patch_wan_patch_embedding(self)
        return super().aitk_post_load(**kwargs)

    def get_offload_ignore_modules(self):
        # fp32 modulation tables must stay resident on the compute device
        return [self.scale_shift_table] + [
            block.scale_shift_table for block in self.blocks
        ]

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        # original/comfy wan keys -> diffusers layout via diffusers' own
        # converter (rename-only for the base/i2v/vace variants)
        is_original = any(
            ".self_attn." in k
            or ".cross_attn." in k
            or k.startswith("model.diffusion_model.")
            for k in state_dict
        )
        if not is_original:
            return state_dict
        from diffusers.loaders.single_file_utils import (
            convert_wan_transformer_to_diffusers,
        )

        return convert_wan_transformer_to_diffusers(dict(state_dict))

    # inverse of diffusers' rename table, for the base/t2v/i2v variants (no
    # vace/animate: their extra mappings collide in reverse). Order matters:
    # longest/most-specific first, and the norm2/norm3 swap uses a placeholder.
    _SAVE_RENAMES = [
        ("condition_embedder.time_embedder.linear_1", "time_embedding.0"),
        ("condition_embedder.time_embedder.linear_2", "time_embedding.2"),
        ("condition_embedder.text_embedder.linear_1", "text_embedding.0"),
        ("condition_embedder.text_embedder.linear_2", "text_embedding.2"),
        ("condition_embedder.time_proj", "time_projection.1"),
        ("condition_embedder.image_embedder.norm1", "img_emb.proj.0"),
        ("condition_embedder.image_embedder.ff.net.0.proj", "img_emb.proj.1"),
        ("condition_embedder.image_embedder.ff.net.2", "img_emb.proj.3"),
        ("condition_embedder.image_embedder.norm2", "img_emb.proj.4"),
        ("ffn.net.0.proj", "ffn.0"),
        ("ffn.net.2", "ffn.2"),
        (".norm_added_k.", ".norm_k_img."),
        (".add_k_proj.", ".k_img."),
        (".add_v_proj.", ".v_img."),
        (".to_out.0.", ".o."),
        (".to_q.", ".q."),
        (".to_k.", ".k."),
        (".to_v.", ".v."),
        ("attn2", "cross_attn"),
        ("attn1", "self_attn"),
        # norm2 <-> norm3 swap back
        ("norm3", "norm__placeholder"),
        ("norm2", "norm3"),
        ("norm__placeholder", "norm2"),
        ("proj_out", "head.head"),
    ]

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        """Diffusers layout back to the original/comfy wan key layout
        (rename-only, so quantized weight/scale/marker keys ride along)."""
        if any(".self_attn." in k or ".cross_attn." in k for k in state_dict):
            return state_dict  # already original layout
        new_sd = {}
        for key, value in state_dict.items():
            k = key
            # scale_shift_table: blocks keep the name as `modulation`, the
            # top-level one belongs to the output head
            if k == "scale_shift_table":
                k = "head.modulation"
            elif k.endswith(".scale_shift_table"):
                k = k[: -len("scale_shift_table")] + "modulation"
            for src, dst in cls._SAVE_RENAMES:
                k = k.replace(src, dst)
            new_sd[k] = value
        return new_sd
