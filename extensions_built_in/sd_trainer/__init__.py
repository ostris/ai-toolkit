# This is an example extension for custom training. It is great for experimenting with new ideas.
from toolkit.extension import Extension


# This is for generic training (LoRA, Dreambooth, FineTuning)
class SDTrainerExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "sd_trainer"

    # name is the name of the extension for printing
    name = "SD Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .SDTrainer import SDTrainer

        return SDTrainer


# This is for generic training (LoRA, Dreambooth, FineTuning)
class UITrainerExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "ui_trainer"

    # name is the name of the extension for printing
    name = "UI Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .UITrainer import UITrainer

        return UITrainer


# This is a universal trainer that can be from ui or api
class DiffusionTrainerExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "diffusion_trainer"

    # name is the name of the extension for printing
    name = "Diffusion Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .DiffusionTrainer import DiffusionTrainer

        return DiffusionTrainer


# for backwards compatability
class TextualInversionTrainer(SDTrainerExtension):
    uid = "textual_inversion_trainer"


AI_TOOLKIT_EXTENSIONS = [
    # you can put a list of extensions here
    SDTrainerExtension,
    TextualInversionTrainer,
    UITrainerExtension,
    DiffusionTrainerExtension,
]
