# This is an example extension for custom training. It is great for experimenting with new ideas.
from toolkit.extension import Extension


# We make a subclass of Extension
class ExampleMergeExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "example_merge_extension"

    # name is the name of the extension for printing
    name = "Example Merge Extension"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .ExampleMergeModels import ExampleMergeModels
        return ExampleMergeModels


AI_TOOLKIT_EXTENSIONS = [
    # you can put a list of extensions here
    ExampleMergeExtension
]
