import os
from typing import Optional

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ROOT = os.path.join(TOOLKIT_ROOT, 'config')
KEYMAPS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "keymaps")
ORIG_CONFIGS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "orig_configs")
DIFFUSERS_CONFIGS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "diffusers_configs")
COMFY_PATH = os.getenv("COMFY_PATH", None)
COMFY_MODELS_PATH = None
if COMFY_PATH:
    COMFY_MODELS_PATH = os.path.join(COMFY_PATH, "models")

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


def normalize_path(path: Optional[str]) -> Optional[str]:
    """Strip leading/trailing whitespace and trailing path separators.
    Use for any path: model dirs, LoRA/safetensors files, etc.
    """
    if not isinstance(path, str):
        return path
    path = path.strip()
    return path.rstrip(os.sep).rstrip("/")
