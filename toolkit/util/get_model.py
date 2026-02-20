import os
from typing import List
from toolkit.models.base_model import BaseModel
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.config_modules import ModelConfig
from toolkit.paths import TOOLKIT_ROOT
import importlib
import pkgutil

from toolkit.models.wan21 import Wan21, Wan21I2V
from toolkit.models.cogview4 import CogView4

BUILT_IN_MODELS = [
    Wan21,
    Wan21I2V,
    CogView4,
]


def get_all_models() -> List[BaseModel]:
    extension_folders = ['extensions', 'extensions_built_in']

    # This will hold the classes from all extension modules
    all_model_classes: List[BaseModel] = BUILT_IN_MODELS

    # Iterate over all directories (i.e., packages) in the "extensions" directory
    for sub_dir in extension_folders:
        extensions_dir = os.path.join(TOOLKIT_ROOT, sub_dir)
        for (_, name, _) in pkgutil.iter_modules([extensions_dir]):
            try:
                # Import the module
                module = importlib.import_module(f"{sub_dir}.{name}")
                # Get the value of the AI_TOOLKIT_MODELS variable
                models = getattr(module, "AI_TOOLKIT_MODELS", None)
                # Check if the value is a list
                if isinstance(models, list):
                    # Iterate over the list and add the classes to the main list
                    all_model_classes.extend(models)
            except ImportError as e:
                print(f"Failed to import the {name} module. Error: {str(e)}")
    return all_model_classes


def get_model_class(config: ModelConfig):
    all_models = get_all_models()
    for ModelClass in all_models:
        if ModelClass.arch == config.arch:
            return ModelClass
    # default to the legacy model
    return StableDiffusion
