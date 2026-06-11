from toolkit.extension import Extension


class AceStepCaptionerExtension(Extension):
    uid = "AceStepCaptioner"
    name = "Ace Step Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .AceStepCaptioner import AceStepCaptioner

        return AceStepCaptioner


class Qwen3VLCaptionerExtension(Extension):
    uid = "Qwen3VLCaptioner"
    name = "Qwen 3VL Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Qwen3VLCaptioner import Qwen3VLCaptioner

        return Qwen3VLCaptioner


AI_TOOLKIT_EXTENSIONS = [
    AceStepCaptionerExtension,
    Qwen3VLCaptionerExtension,
]
