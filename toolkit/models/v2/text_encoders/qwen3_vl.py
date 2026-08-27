import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration

from .._mixin import OstrisTransformersMixin


def patch_qwen_vl_patch_embed(model) -> int:
    """Qwen-VL's vision patch_embed is a Conv3d whose kernel == stride, i.e. a plain
    linear projection of each flattened patch. bf16 Conv3d has no fast cuDNN kernel and
    falls back to a slow, GPU-underutilizing path. Swap it for the equivalent F.linear
    (a GEMM). The weight is read lazily so this survives later .to(device)/dtype moves.
    Returns the number of patch_embed modules patched."""
    patched = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(
            proj.stride
        ):

            def fast_forward(hidden_states, _proj=proj):
                w = _proj.weight.reshape(_proj.weight.shape[0], -1)
                x = hidden_states.view(-1, w.shape[1]).to(w.dtype)
                return F.linear(x, w, _proj.bias)

            module.forward = fast_forward
            patched += 1
    return patched


class Qwen3VLTextEncoder(Qwen3VLForConditionalGeneration, OstrisTransformersMixin):
    """Qwen3-VL conditioning stack (krea2, mageflow, nucleus_image, ideogram4,
    minimax_h3, ...). Loads from a checkpoint's text_encoder/ subfolder or a
    raw Qwen repo (pass subfolder=\"\" for the latter)."""

    aitk_subfolder = "text_encoder"
    aitk_processor_subfolder = "processor"

    @classmethod
    def get_transformer_block_names(cls):
        return ["model.language_model.layers"]

    def drop_vision_tower(self):
        """Text-only conditioning: the vision tower is dead weight — drop it to
        free VRAM and skip loading its (bf16-slow) Conv3d patch_embed."""
        if getattr(self.model, "visual", None) is not None:
            self.model.visual = None
        return self

    def patch_vision_patch_embed(self) -> int:
        """Keep the vision tower (reference images ride into the embeddings)
        but swap its Conv3d patch_embed for an equivalent GEMM."""
        return patch_qwen_vl_patch_embed(self)
