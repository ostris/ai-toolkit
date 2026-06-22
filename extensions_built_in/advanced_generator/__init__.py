# This is an example extension for custom training. It is great for experimenting with new ideas.
from toolkit.extension import Extension


# This is for generic training (LoRA, Dreambooth, FineTuning)
class AdvancedReferenceGeneratorExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "reference_generator"

    # name is the name of the extension for printing
    name = "Reference Generator"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .ReferenceGenerator import ReferenceGenerator
        return ReferenceGenerator


# This is for generic training (LoRA, Dreambooth, FineTuning)
class PureLoraGenerator(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "pure_lora_generator"

    # name is the name of the extension for printing
    name = "Pure LoRA Generator"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .PureLoraGenerator import PureLoraGenerator
        return PureLoraGenerator


# This is for generic training (LoRA, Dreambooth, FineTuning)
class Img2ImgGeneratorExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "batch_img2img"

    # name is the name of the extension for printing
    name = "Img2ImgGeneratorExtension"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Img2ImgGenerator import Img2ImgGenerator
        return Img2ImgGenerator


AI_TOOLKIT_EXTENSIONS = [
    # you can put a list of extensions here
    AdvancedReferenceGeneratorExtension, PureLoraGenerator, Img2ImgGeneratorExtension
]
