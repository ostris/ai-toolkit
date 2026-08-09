# This is a documentation-only TEMPLATE model. Start with README.md in this
# folder for the full guide to adding a new model architecture to ai-toolkit.
#
# It is intentionally NOT registered: the parent package
# (extensions_built_in/diffusion_models/__init__.py) does not import it, so it
# never shows up as a trainable arch. To register a real model, import its
# class there and append it to the AI_TOOLKIT_MODELS list. (Models can also
# live in their own folder under extensions/, which defines its own
# AI_TOOLKIT_MODELS list -- see extensions/z_image_pixel for a tiny example.)
from .example_model import ExampleModel

__all__ = ["ExampleModel"]
