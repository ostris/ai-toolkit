import os
import json
from toolkit.paths import TOOLKIT_ROOT

possible_extensions = ['.json', '.jsonc']


def get_cwd_abs_path(path):
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    return path


def get_config(config_file_path):
    # first check if it is in the config folder
    config_path = os.path.join(TOOLKIT_ROOT, 'config', config_file_path)
    # see if it is in the config folder with any of the possible extensions if it doesnt have one
    real_config_path = None
    if not os.path.exists(config_path):
        for ext in possible_extensions:
            if os.path.exists(config_path + ext):
                real_config_path = config_path + ext
                break

    # if we didn't find it there, check if it is a full path
    if not real_config_path:
        if os.path.exists(config_file_path):
            real_config_path = config_file_path
        elif os.path.exists(get_cwd_abs_path(config_file_path)):
            real_config_path = get_cwd_abs_path(config_file_path)

    if not real_config_path:
        raise ValueError(f"Could not find config file {config_file_path}")

    # load the config
    with open(real_config_path, 'r') as f:
        config = json.load(f)

    return config
