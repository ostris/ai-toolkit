from toolkit.extension import Extension


class MusicCaptionerExtension(Extension):
    uid = "MusicCaptioner"
    name = "Music Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .MusicCaptioner import MusicCaptioner

        return MusicCaptioner


AI_TOOLKIT_EXTENSIONS = [
    MusicCaptionerExtension,
]
