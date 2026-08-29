from transformers import Qwen2_5_VLForConditionalGeneration

from .._mixin import OstrisTransformersMixin


class Qwen25VLTextEncoder(Qwen2_5_VLForConditionalGeneration, OstrisTransformersMixin):
    """Qwen2.5-VL conditioning stack (qwen_image family). Loads from a
    checkpoint's text_encoder/ subfolder; the edit variants keep the vision
    tower, plain t2i drops it."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["model.language_model.layers"]

    def drop_vision_tower(self):
        """Text-only conditioning: the vision tower is dead weight."""
        if getattr(self.model, "visual", None) is not None:
            self.model.visual = None
        return self
