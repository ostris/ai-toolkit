import torch
from diffusers.models.transformers import (
    ZImageTransformer2DModel as DiffusersZImageTransformer2DModel,
)

from ._mixin import OstrisModelMixin


class ZImageTransformer2DModel(DiffusersZImageTransformer2DModel, OstrisModelMixin):
    aitk_subfolder = "transformer"
    # repo to pull the config from when loading a single-file checkpoint
    aitk_config_repo = "Tongyi-MAI/Z-Image-Turbo"

    @classmethod
    def get_quantization_block_names(cls):
        return ["layers"]

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
            k = k.replace(".attention.out.weight", ".attention.to_out.0.weight")
            k = k.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
            k = k.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
            if k.startswith("x_embedder."):
                k = "all_x_embedder.2-1." + k[len("x_embedder.") :]
            elif k.startswith("final_layer."):
                k = "all_final_layer.2-1." + k[len("final_layer.") :]
            new_sd[k] = value
        return new_sd

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        """Convert a diffusers transformer state dict back to the single-file layout."""
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
            k = k.replace(".attention.to_out.0.weight", ".attention.out.weight")
            k = k.replace(".attention.norm_q.weight", ".attention.q_norm.weight")
            k = k.replace(".attention.norm_k.weight", ".attention.k_norm.weight")
            if k.startswith("all_x_embedder.2-1."):
                k = "x_embedder." + k[len("all_x_embedder.2-1.") :]
            elif k.startswith("all_final_layer.2-1."):
                k = "final_layer." + k[len("all_final_layer.2-1.") :]
            new_sd[k] = value
        return new_sd
