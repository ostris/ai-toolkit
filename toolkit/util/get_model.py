from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.config_modules import ModelConfig

def get_model_class(config: ModelConfig):
    if config.arch == "wan21":
        from toolkit.models.wan21 import Wan21
        return Wan21
    elif config.arch == "cogview4":
        from toolkit.models.cogview4 import CogView4
        return CogView4
    else:
        return StableDiffusion