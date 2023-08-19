# This is an example extension for custom training. It is great for experimenting with new ideas.
from toolkit.extension import Extension


# We make a subclass of Extension
class UltimateSliderTrainer(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "ultimate_slider_trainer"

    # name is the name of the extension for printing
    name = "Ultimate Slider Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .UltimateSliderTrainerProcess import UltimateSliderTrainerProcess
        return UltimateSliderTrainerProcess


AI_TOOLKIT_EXTENSIONS = [
    # you can put a list of extensions here
    UltimateSliderTrainer
]
