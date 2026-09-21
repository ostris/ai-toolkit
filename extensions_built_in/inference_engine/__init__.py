from toolkit.extension import Extension


class InferenceEngineExtension(Extension):
    uid = "InferenceEngine"
    name = "Inference Engine"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .InferenceEngine import InferenceEngine

        return InferenceEngine


AI_TOOLKIT_EXTENSIONS = [
    InferenceEngineExtension,
]
