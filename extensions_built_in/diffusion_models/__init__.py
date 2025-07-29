from .chroma import ChromaModel
from .hidream import HidreamModel, HidreamE1Model
from .f_light import FLiteModel
from .omnigen2 import OmniGen2Model
from .flux_kontext import FluxKontextModel
from .wan22 import Wan22Model

AI_TOOLKIT_MODELS = [
    # put a list of models here
    ChromaModel,
    HidreamModel,
    HidreamE1Model,
    FLiteModel,
    OmniGen2Model,
    FluxKontextModel,
    Wan22Model,
]
