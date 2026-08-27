from diffusers import (
    QwenImageTransformer2DModel as DiffusersQwenImageTransformer2DModel,
)

from .._mixin import OstrisModelMixin


class QwenImageTransformer2DModel(
    DiffusersQwenImageTransformer2DModel, OstrisModelMixin
):
    aitk_subfolder = "transformer"
    aitk_config_repo = "Qwen/Qwen-Image"

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks"]

    @classmethod
    def _load_single_file(cls, file_path, dtype, config_path=None, subfolder=None):
        # single-file checkpoints in the wild carry diffusers or original key
        # layouts; diffusers' single-file machinery owns that conversion
        model = cls.from_single_file(
            file_path,
            config=config_path if config_path is not None else cls.aitk_config_repo,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        model.to(dtype)
        return model
