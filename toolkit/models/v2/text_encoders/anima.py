from diffusers import AnimaTextConditioner as DiffusersAnimaTextConditioner

from .._mixin import OstrisModelMixin


class AnimaTextConditioner(DiffusersAnimaTextConditioner, OstrisModelMixin):
    """Anima's learned text conditioner (rides next to the Qwen3 encoder)."""

    aitk_subfolder = "text_conditioner"
