import os

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ROOT = os.path.join(TOOLKIT_ROOT, 'config')
KEYMAPS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "keymaps")
ORIG_CONFIGS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "orig_configs")
DIFFUSERS_CONFIGS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "diffusers_configs")

# check if ENV variable is set
if 'MODELS_PATH' in os.environ:
    MODELS_PATH = os.environ['MODELS_PATH']
else:
    MODELS_PATH = os.path.join(TOOLKIT_ROOT, "models")


def get_path(path):
    # we allow absolute paths, but if it is not absolute, we assume it is relative to the toolkit root
    if not os.path.isabs(path):
        path = os.path.join(TOOLKIT_ROOT, path)
    return path
