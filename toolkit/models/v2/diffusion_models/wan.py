from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel

from .._mixin import OstrisModelMixin


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

    @classmethod
    def get_transformer_block_names(cls):
        return ["blocks"]

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
