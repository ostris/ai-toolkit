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


class Florence2CaptionerExtension(Extension):
    uid = "Florence2Captioner"
    name = "Florence-2 Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Florence2Captioner import Florence2Captioner

        return Florence2Captioner


AI_TOOLKIT_EXTENSIONS = [
    AceStepCaptionerExtension,
    Qwen3VLCaptionerExtension,
    Florence2CaptionerExtension,
]
