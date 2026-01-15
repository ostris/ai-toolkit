# Image-based Concept Slider extension for training sliders using image sequences
from toolkit.extension import Extension


class ImageConceptSliderTrainer(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "image_concept_slider"

    # name is the name of the extension for printing
    name = "Image Concept Slider Trainer"

    # This is where your process class is loaded
    @classmethod
    def get_process(cls):
        from .ImageConceptSliderProcess import ImageConceptSliderProcess
        return ImageConceptSliderProcess


AI_TOOLKIT_EXTENSIONS = [
    ImageConceptSliderTrainer
]
