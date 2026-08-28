from diffusers.models.transformers import (
    LTX2VideoTransformer3DModel as DiffusersLTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import (
    LTX2TextConnectors as DiffusersLTX2TextConnectors,
)
from diffusers.pipelines.ltx2 import LTX2Vocoder as DiffusersLTX2Vocoder
from diffusers.pipelines.ltx2 import (
    LTX2VocoderWithBWE as DiffusersLTX2VocoderWithBWE,
)

from .._mixin import OstrisModelMixin


class LTX2VideoTransformer3DModel(
    DiffusersLTX2VideoTransformer3DModel, OstrisModelMixin
):
    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks"]

    def get_offload_ignore_modules(self):
        # fp32 scale/shift tables (bare tensors) must stay resident
        mods = []
        for block in self.transformer_blocks:
            mods += [
                block.scale_shift_table,
                block.audio_scale_shift_table,
                block.video_a2v_cross_attn_scale_shift_table,
                block.audio_a2v_cross_attn_scale_shift_table,
            ]
        return mods + [self.scale_shift_table, self.audio_scale_shift_table]


class LTX2TextConnectors(DiffusersLTX2TextConnectors, OstrisModelMixin):
    aitk_subfolder = "connectors"


class LTX2Vocoder(DiffusersLTX2Vocoder, OstrisModelMixin):
    aitk_subfolder = "vocoder"


class LTX2VocoderWithBWE(DiffusersLTX2VocoderWithBWE, OstrisModelMixin):
    aitk_subfolder = "vocoder"
