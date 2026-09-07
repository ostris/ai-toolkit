import torch
from diffusers.models.transformers import (
    ZImageTransformer2DModel as DiffusersZImageTransformer2DModel,
)

from .._mixin import OstrisModelMixin


class ZImageTransformer2DModel(DiffusersZImageTransformer2DModel, OstrisModelMixin):
    aitk_subfolder = "transformer"
    # repo to pull the config from when loading a single-file checkpoint
    aitk_config_repo = "Tongyi-MAI/Z-Image-Turbo"

    aitk_comfy_repo = "Comfy-Org/z_image_turbo"
    # the int8_convrot file marks the FUSED attention.qkv modules; the load
    # converter splits those entries exactly into to_q/to_k/to_v (row-sliced
    # qdata + scales), so it attaches onto this split layout directly
    aitk_comfy_weight_names = {
        "Tongyi-MAI/Z-Image-Turbo": [
            "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
            "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        ],
    }

    @classmethod
    def get_transformer_block_names(cls):
        return ["layers"]

    def get_offload_ignore_modules(self):
        return [self.x_pad_token, self.cap_pad_token]

    @classmethod
    def get_quantization_exclude_modules(cls):
        # sensitive modules kept in full precision (fnmatch patterns on module
        # names within ZImageTransformer2DModel):
        #   t_embedder*      - timestep embedder; feeds every block's
        #                      adaLN_modulation and the final layers
        #   cap_embedder*    - caption feature -> model width projection
        #   all_x_embedder*  - patchified latent input projections
        #   all_final_layer* - final adaLN-modulated output projections
        #   siglip_embedder* - siglip feature projection (edit models only)
        return [
            "t_embedder*",
            "cap_embedder*",
            "all_x_embedder*",
            "all_final_layer*",
            "siglip_embedder*",
        ]

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        """Convert a single-file Z-Image checkpoint to diffusers transformer keys."""
        from toolkit.util.comfy_quant_import import split_fused_quantized_keys

        state_dict = dict(state_dict)
        # quantized fused qkv entries (comfy convrot/fp8/nvfp4 files) split
        # exactly into the three projections: row-consecutive weights/scales
        for marker_key in [
            k for k in list(state_dict) if k.endswith(".attention.qkv.comfy_quant")
        ]:
            prefix = marker_key[: -len(".comfy_quant")]
            base = prefix[: -len(".qkv")]
            split_fused_quantized_keys(
                state_dict,
                prefix,
                [f"{base}.to_q", f"{base}.to_k", f"{base}.to_v"],
            )

        new_sd = {}
        for key, value in state_dict.items():
            k = key
            if k.endswith(".attention.qkv.weight"):
                # the single file fuses q,k,v into one tensor (in that order); diffusers keeps them split
                prefix = k[: -len(".attention.qkv.weight")]
                q, k_proj, v = torch.chunk(value, 3, dim=0)
                new_sd[prefix + ".attention.to_q.weight"] = q
                new_sd[prefix + ".attention.to_k.weight"] = k_proj
                new_sd[prefix + ".attention.to_v.weight"] = v
                continue
            # module-prefix renames so weight/bias/scale/marker keys all map
            k = k.replace(".attention.out.", ".attention.to_out.0.")
            k = k.replace(".attention.q_norm.", ".attention.norm_q.")
            k = k.replace(".attention.k_norm.", ".attention.norm_k.")
            if k.startswith("x_embedder."):
                k = "all_x_embedder.2-1." + k[len("x_embedder.") :]
            elif k.startswith("final_layer."):
                k = "all_final_layer.2-1." + k[len("final_layer.") :]
            new_sd[k] = value
        return new_sd

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        """Convert a diffusers transformer state dict back to the single-file layout."""
        from toolkit.util.comfy_quant_import import fuse_split_quantized_keys

        state_dict = dict(state_dict)
        # quantized split projections fuse back into the single-file qkv entry
        for marker_key in [
            k for k in list(state_dict) if k.endswith(".attention.to_q.comfy_quant")
        ]:
            base = marker_key[: -len(".to_q.comfy_quant")]
            fuse_split_quantized_keys(
                state_dict,
                [f"{base}.to_q", f"{base}.to_k", f"{base}.to_v"],
                f"{base}.qkv",
            )

        new_sd = {}
        qkv_cache = {}
        for key, value in state_dict.items():
            k = key
            matched = False
            for suffix in (
                ".attention.to_q.weight",
                ".attention.to_k.weight",
                ".attention.to_v.weight",
            ):
                if k.endswith(suffix):
                    prefix = k[: -len(suffix)]
                    cache = qkv_cache.setdefault(prefix, {})
                    cache[suffix] = value
                    if len(cache) == 3:
                        # the single file expects q,k,v fused in that order
                        qkv = torch.cat(
                            [
                                cache[".attention.to_q.weight"],
                                cache[".attention.to_k.weight"],
                                cache[".attention.to_v.weight"],
                            ],
                            dim=0,
                        )
                        new_sd[prefix + ".attention.qkv.weight"] = qkv
                        del qkv_cache[prefix]
                    matched = True
                    break
            if matched:
                continue
            k = k.replace(".attention.to_out.0.", ".attention.out.")
            k = k.replace(".attention.norm_q.", ".attention.q_norm.")
            k = k.replace(".attention.norm_k.", ".attention.k_norm.")
            if k.startswith("all_x_embedder.2-1."):
                k = "x_embedder." + k[len("all_x_embedder.2-1.") :]
            elif k.startswith("all_final_layer.2-1."):
                k = "final_layer." + k[len("all_final_layer.2-1.") :]
            new_sd[k] = value
        return new_sd
