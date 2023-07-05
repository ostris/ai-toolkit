import json
import os
import sys

from flatten_json import flatten

sys.path.insert(0, os.getcwd())
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FOLDER = os.path.join(PROJECT_ROOT, 'config')
sys.path.append(PROJECT_ROOT)

import argparse

from toolkit.lycoris_utils import extract_diff
from toolkit.config import get_config
from toolkit.metadata import create_meta
from toolkit.kohya_model_util import load_models_from_stable_diffusion_checkpoint

import torch
from safetensors.torch import save_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        help="Name of config file (eg: person_v1 for config/person_v1.json), or full path if it is not in config folder",
        type=str
    )
    return parser.parse_args()


def main():
    args = get_args()

    config_raw = get_config(args.config_file)
    config = config_raw['config'] if 'config' in config_raw else None
    if not config:
        raise ValueError('config file is invalid. Missing "config" key')

    meta = config_raw['meta'] if 'meta' in config_raw else {}

    def get_conf(key, default=None):
        if key in config:
            return config[key]
        else:
            return default

    is_v2 = get_conf('is_v2', False)
    name = get_conf('name', None)
    base_model = get_conf('base_model')
    extract_model = get_conf('extract_model')
    output_folder = get_conf('output_folder')
    process_list = get_conf('process')
    device = get_conf('device', 'cpu')
    use_sparse_bias = get_conf('use_sparse_bias', False)
    sparsity = get_conf('sparsity', 0.98)
    disable_cp = get_conf('disable_cp', False)

    if not name:
        raise ValueError('name is required')
    if not base_model:
        raise ValueError('base_model is required')
    if not extract_model:
        raise ValueError('extract_model is required')
    if not output_folder:
        raise ValueError('output_folder is required')
    if not process_list or len(process_list) == 0:
        raise ValueError('process is required')

    # check processes
    for process in process_list:
        if process['mode'] == 'fixed':
            if not process['linear_dim']:
                raise ValueError('linear_dim is required in fixed mode')
            if not process['conv_dim']:
                raise ValueError('conv_dim is required in fixed mode')
        elif process['mode'] == 'threshold':
            if not process['linear_threshold']:
                raise ValueError('linear_threshold is required in threshold mode')
            if not process['conv_threshold']:
                raise ValueError('conv_threshold is required in threshold mode')
        elif process['mode'] == 'ratio':
            if not process['linear_ratio']:
                raise ValueError('linear_ratio is required in ratio mode')
            if not process['conv_ratio']:
                raise ValueError('conv_threshold is required in threshold mode')
        elif process['mode'] == 'quantile':
            if not process['linear_quantile']:
                raise ValueError('linear_quantile is required in quantile mode')
            if not process['conv_quantile']:
                raise ValueError('conv_quantile is required in quantile mode')
        else:
            raise ValueError('mode is invalid')

    print(f"Loading base model: {base_model}")
    base = load_models_from_stable_diffusion_checkpoint(is_v2, base_model)
    print(f"Loading extract model: {extract_model}")
    extract = load_models_from_stable_diffusion_checkpoint(is_v2, extract_model)

    print(f"Running  {len(process_list)} process{'' if len(process_list) == 1 else 'es'}")

    for process in process_list:
        item_meta = json.loads(json.dumps(meta))
        item_meta['process'] = process
        if process['mode'] == 'fixed':
            linear_mode_param = int(process['linear_dim'])
            conv_mode_param = int(process['conv_dim'])
        elif process['mode'] == 'threshold':
            linear_mode_param = float(process['linear_threshold'])
            conv_mode_param = float(process['conv_threshold'])
        elif process['mode'] == 'ratio':
            linear_mode_param = float(process['linear_ratio'])
            conv_mode_param = float(process['conv_ratio'])
        elif process['mode'] == 'quantile':
            linear_mode_param = float(process['linear_quantile'])
            conv_mode_param = float(process['conv_quantile'])
        else:
            raise ValueError(f"Unknown mode: {process['mode']}")

        print(f"Running process: {process['mode']}, lin: {linear_mode_param}, conv: {conv_mode_param}")

        state_dict, extract_diff_meta = extract_diff(
            base,
            extract,
            process['mode'],
            linear_mode_param,
            conv_mode_param,
            device,
            use_sparse_bias,
            sparsity,
            not disable_cp
        )

        save_meta = create_meta([
            item_meta, extract_diff_meta
        ])

        output_file_name = f"lyco_{name}_{process['mode']}_{linear_mode_param}_{conv_mode_param}.safetensors"
        output_path = os.path.join(output_folder, output_file_name)
        os.makedirs(output_folder, exist_ok=True)

        # having issues with meta
        save_file(state_dict, output_path)
        # save_file(state_dict, output_path, {'meta': json.dumps(save_meta, indent=4)})

        print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
