import os
from typing import List
from toolkit.paths import COMFY_MODELS_PATH


def get_comfy_path(comfy_files: List[str]) -> str:
    """
    Get the path to the first existing file in the COMFY_MODELS_PATH.
    """
    if COMFY_MODELS_PATH is not None and comfy_files is not None and len(comfy_files) > 0:
        for file in comfy_files:
            file_path = os.path.join(COMFY_MODELS_PATH, file)
            if os.path.exists(file_path):
                return file_path
    return None