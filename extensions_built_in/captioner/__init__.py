from toolkit.extension import Extension


class AceStepCaptionerExtension(Extension):
    uid = "AceStepCaptioner"
    name = "Ace Step Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .AceStepCaptioner import AceStepCaptioner

        return AceStepCaptioner


class MossMusicCaptionerExtension(Extension):
    uid = "MossMusicCaptioner"
    name = "MOSS-Music Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .MossMusicCaptioner import MossMusicCaptioner

        return MossMusicCaptioner


class Qwen3VLCaptionerExtension(Extension):
    uid = "Qwen3VLCaptioner"
    name = "Qwen 3VL Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Qwen3VLCaptioner import Qwen3VLCaptioner

        return Qwen3VLCaptioner


class Qwen3OmniCaptionerExtension(Extension):
    uid = "Qwen3OmniCaptioner"
    name = "Qwen 3 Omni Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Qwen3OmniCaptioner import Qwen3OmniCaptioner

        return Qwen3OmniCaptioner


class Qwen25OmniCaptionerExtension(Extension):
    uid = "Qwen25OmniCaptioner"
    name = "Qwen 2.5 Omni Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Qwen25OmniCaptioner import Qwen25OmniCaptioner

        return Qwen25OmniCaptioner


class Ideogram4CaptionerExtension(Extension):
    uid = "Ideogram4Captioner"
    name = "Ideogram4 Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Ideogram4Captioner import Ideogram4Captioner

        return Ideogram4Captioner


AI_TOOLKIT_EXTENSIONS = [
    AceStepCaptionerExtension,
    MossMusicCaptionerExtension,
    Qwen3VLCaptionerExtension,
    Qwen3OmniCaptionerExtension,
    Qwen25OmniCaptionerExtension,
    Ideogram4CaptionerExtension,
]
