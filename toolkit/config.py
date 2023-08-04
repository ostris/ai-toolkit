import os
import json
import oyaml as yaml
import re
from collections import OrderedDict

from toolkit.paths import TOOLKIT_ROOT

possible_extensions = ['.json', '.jsonc', '.yaml', '.yml']


def get_cwd_abs_path(path):
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    return path


def preprocess_config(config: OrderedDict, name: str = None):
    if "job" not in config:
        raise ValueError("config file must have a job key")
    if "config" not in config:
        raise ValueError("config file must have a config section")
    if "name" not in config["config"] and name is None:
        raise ValueError("config file must have a config.name key")
    # we need to replace tags. For now just [name]
    if name is not None:
        config["config"]["name"] = name
    else:
        name = config["config"]["name"]
    config_string = json.dumps(config)
    config_string = config_string.replace("[name]", name)
    config = json.loads(config_string, object_pairs_hook=OrderedDict)
    return config


# Fixes issue where yaml doesnt load exponents correctly
fixed_loader = yaml.SafeLoader
fixed_loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def get_config(config_file_path, name=None):
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

    # if we found it, check if it is a json or yaml file
    if real_config_path.endswith('.json') or real_config_path.endswith('.jsonc'):
        with open(real_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f, object_pairs_hook=OrderedDict)
    elif real_config_path.endswith('.yaml') or real_config_path.endswith('.yml'):
        with open(real_config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=fixed_loader)
    else:
        raise ValueError(f"Config file {config_file_path} must be a json or yaml file")

    return preprocess_config(config, name)
