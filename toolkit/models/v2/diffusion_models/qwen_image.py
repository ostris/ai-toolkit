from diffusers import (
    QwenImageTransformer2DModel as DiffusersQwenImageTransformer2DModel,
)

from .._mixin import OstrisModelMixin


class QwenImageTransformer2DModel(
    DiffusersQwenImageTransformer2DModel, OstrisModelMixin
):
    aitk_subfolder = "transformer"
    aitk_config_repo = "Qwen/Qwen-Image"

    aitk_comfy_repo = "Comfy-Org/Qwen-Image_ComfyUI"
    # the comfy files use the diffusers key layout directly (no conversion);
    # fp8mixed carries float8_e4m3fn comfy_quant markers that attach straight
    # onto this class's modules
    aitk_comfy_weight_names = {
        "Qwen/Qwen-Image": [
            "split_files/diffusion_models/qwen_image_fp8mixed.safetensors",
            # raw fp8 cast (no markers, diffusers keys) — loads via from_single_file
            "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
            "split_files/diffusion_models/qwen_image_bf16.safetensors",
        ],
    }

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks"]

    @classmethod
    def _load_single_file(
        cls, file_path, dtype, config_path=None, config=None, subfolder=None
    ):
        from safetensors import safe_open

        with safe_open(file_path, framework="pt") as f:
            has_markers = any(k.endswith(".comfy_quant") for k in f.keys())
        if has_markers:
            # comfy prequantized checkpoint (diffusers key layout): the mixin
            # path attaches the quantized layers
            return super()._load_single_file(
                file_path,
                dtype,
                config_path=config_path,
                config=config,
                subfolder=subfolder,
            )
        # other single-file checkpoints carry diffusers or original key
        # layouts; diffusers' single-file machinery owns that conversion
        model = cls.from_single_file(
            file_path,
            config=config_path if config_path is not None else cls.aitk_config_repo,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        model.to(dtype)
        return model
