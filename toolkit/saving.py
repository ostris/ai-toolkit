import json
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch
from safetensors.torch import load_file, save_file

from toolkit.train_tools import get_torch_dtype
from toolkit.paths import KEYMAPS_ROOT

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion


def get_slices_from_string(s: str) -> tuple:
    slice_strings = s.split(',')
    slices = [eval(f"slice({component.strip()})") for component in slice_strings]
    return tuple(slices)


def convert_state_dict_to_ldm_with_mapping(
        diffusers_state_dict: 'OrderedDict',
        mapping_path: str,
        base_path: Union[str, None] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
) -> 'OrderedDict':
    converted_state_dict = OrderedDict()

    # load mapping
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f, object_pairs_hook=OrderedDict)

    # keep track of keys not matched
    ldm_matched_keys = []
    diffusers_matched_keys = []

    ldm_diffusers_keymap = mapping['ldm_diffusers_keymap']
    ldm_diffusers_shape_map = mapping['ldm_diffusers_shape_map']
    ldm_diffusers_operator_map = mapping['ldm_diffusers_operator_map']

    # load base if it exists
    # the base just has come keys like timing ids and stuff diffusers doesn't have or they don't match
    if base_path is not None:
        converted_state_dict = load_file(base_path, device)
        # convert to the right dtype
        for key in converted_state_dict:
            converted_state_dict[key] = converted_state_dict[key].to(device, dtype=dtype)

    # process operators first
    for ldm_key in ldm_diffusers_operator_map:
        # if the key cat is in the ldm key, we need to process it
        if 'cat' in ldm_diffusers_operator_map[ldm_key]:
            cat_list = []
            for diffusers_key in ldm_diffusers_operator_map[ldm_key]['cat']:
                cat_list.append(diffusers_state_dict[diffusers_key].detach())
            converted_state_dict[ldm_key] = torch.cat(cat_list, dim=0).to(device, dtype=dtype)
            diffusers_matched_keys.extend(ldm_diffusers_operator_map[ldm_key]['cat'])
            ldm_matched_keys.append(ldm_key)
        if 'slice' in ldm_diffusers_operator_map[ldm_key]:
            tensor_to_slice = diffusers_state_dict[ldm_diffusers_operator_map[ldm_key]['slice'][0]]
            slice_text = diffusers_state_dict[ldm_diffusers_operator_map[ldm_key]['slice'][1]]
            converted_state_dict[ldm_key] = tensor_to_slice[get_slices_from_string(slice_text)].detach().to(device,
                                                                                                            dtype=dtype)
            diffusers_matched_keys.extend(ldm_diffusers_operator_map[ldm_key]['slice'])
            ldm_matched_keys.append(ldm_key)

    # process the rest of the keys
    for ldm_key in ldm_diffusers_keymap:
        # if the key is in the ldm key, we need to process it
        if ldm_diffusers_keymap[ldm_key] in diffusers_state_dict:
            tensor = diffusers_state_dict[ldm_diffusers_keymap[ldm_key]].detach().to(device, dtype=dtype)
            # see if we need to reshape
            if ldm_key in ldm_diffusers_shape_map:
                tensor = tensor.view(ldm_diffusers_shape_map[ldm_key][0])
            converted_state_dict[ldm_key] = tensor
            diffusers_matched_keys.append(ldm_diffusers_keymap[ldm_key])
            ldm_matched_keys.append(ldm_key)

    # see if any are missing from know mapping
    mapped_diffusers_keys = list(ldm_diffusers_keymap.values())
    mapped_ldm_keys = list(ldm_diffusers_keymap.keys())

    missing_diffusers_keys = [x for x in mapped_diffusers_keys if x not in diffusers_matched_keys]
    missing_ldm_keys = [x for x in mapped_ldm_keys if x not in ldm_matched_keys]

    if len(missing_diffusers_keys) > 0:
        print(f"WARNING!!!! Missing {len(missing_diffusers_keys)} diffusers keys")
        print(missing_diffusers_keys)
    if len(missing_ldm_keys) > 0:
        print(f"WARNING!!!! Missing {len(missing_ldm_keys)} ldm keys")
        print(missing_ldm_keys)

    return converted_state_dict


def get_ldm_state_dict_from_diffusers(
        state_dict: 'OrderedDict',
        sd_version: Literal['1', '2', 'sdxl', 'ssd', 'vega', 'sdxl_refiner'] = '2',
        device='cpu',
        dtype=get_torch_dtype('fp32'),
):
    if sd_version == '1':
        base_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_sd1_ldm_base.safetensors')
        mapping_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_sd1.json')
    elif sd_version == '2':
        base_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_sd2_ldm_base.safetensors')
        mapping_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_sd2.json')
    elif sd_version == 'sdxl':
        # load our base
        base_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_sdxl_ldm_base.safetensors')
        mapping_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_sdxl.json')
    elif sd_version == 'ssd':
        # load our base
        base_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_ssd_ldm_base.safetensors')
        mapping_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_ssd.json')
    elif sd_version == 'vega':
        # load our base
        base_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_vega_ldm_base.safetensors')
        mapping_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_vega.json')
    elif sd_version == 'sdxl_refiner':
        # load our base
        base_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_refiner_ldm_base.safetensors')
        mapping_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_refiner.json')
    else:
        raise ValueError(f"Invalid sd_version {sd_version}")

        # convert the state dict
    return convert_state_dict_to_ldm_with_mapping(
        state_dict,
        mapping_path,
        base_path,
        device=device,
        dtype=dtype
    )


def save_ldm_model_from_diffusers(
        sd: 'StableDiffusion',
        output_file: str,
        meta: 'OrderedDict',
        save_dtype=get_torch_dtype('fp16'),
        sd_version: Literal['1', '2', 'sdxl', 'ssd', 'vega'] = '2'
):
    converted_state_dict = get_ldm_state_dict_from_diffusers(
        sd.state_dict(),
        sd_version,
        device='cpu',
        dtype=save_dtype
    )

    # make sure parent folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_file(converted_state_dict, output_file, metadata=meta)


def save_lora_from_diffusers(
        lora_state_dict: 'OrderedDict',
        output_file: str,
        meta: 'OrderedDict',
        save_dtype=get_torch_dtype('fp16'),
        sd_version: Literal['1', '2', 'sdxl', 'ssd', 'vega'] = '2'
):
    converted_state_dict = OrderedDict()
    # only handle sxdxl for now
    if sd_version != 'sdxl' and sd_version != 'ssd' and sd_version != 'vega':
        raise ValueError(f"Invalid sd_version {sd_version}")
    for key, value in lora_state_dict.items():
        # todo verify if this works with ssd
        # test encoders share keys for some reason
        if key.begins_with('lora_te'):
            converted_state_dict[key] = value.detach().to('cpu', dtype=save_dtype)
        else:
            converted_key = key

    # make sure parent folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_file(converted_state_dict, output_file, metadata=meta)


def save_t2i_from_diffusers(
        t2i_state_dict: 'OrderedDict',
        output_file: str,
        meta: 'OrderedDict',
        dtype=get_torch_dtype('fp16'),
):
    # todo: test compatibility with non diffusers
    converted_state_dict = OrderedDict()
    for key, value in t2i_state_dict.items():
        converted_state_dict[key] = value.detach().to('cpu', dtype=dtype)

    # make sure parent folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_file(converted_state_dict, output_file, metadata=meta)


def load_t2i_model(
        path_to_file,
        device: Union[str] = 'cpu',
        dtype: torch.dtype = torch.float32
):
    raw_state_dict = load_file(path_to_file, device)
    converted_state_dict = OrderedDict()
    for key, value in raw_state_dict.items():
        # todo see if we need to convert dict
        converted_state_dict[key] = value.detach().to(device, dtype=dtype)
    return converted_state_dict




def save_ip_adapter_from_diffusers(
        combined_state_dict: 'OrderedDict',
        output_file: str,
        meta: 'OrderedDict',
        dtype=get_torch_dtype('fp16'),
        direct_save: bool = False
):
    # todo: test compatibility with non diffusers

    converted_state_dict = OrderedDict()
    for module_name, state_dict in combined_state_dict.items():
        if direct_save:
            converted_state_dict[module_name] = state_dict.detach().to('cpu', dtype=dtype)
        else:
            for key, value in state_dict.items():
                converted_state_dict[f"{module_name}.{key}"] = value.detach().to('cpu', dtype=dtype)

    # make sure parent folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_file(converted_state_dict, output_file, metadata=meta)


def load_ip_adapter_model(
        path_to_file,
        device: Union[str] = 'cpu',
        dtype: torch.dtype = torch.float32,
        direct_load: bool = False
):
    # check if it is safetensors or checkpoint
    if path_to_file.endswith('.safetensors'):
        raw_state_dict = load_file(path_to_file, device)
        combined_state_dict = OrderedDict()
        if direct_load:
            return raw_state_dict
        for combo_key, value in raw_state_dict.items():
            key_split = combo_key.split('.')
            module_name = key_split.pop(0)
            if module_name not in combined_state_dict:
                combined_state_dict[module_name] = OrderedDict()
            combined_state_dict[module_name]['.'.join(key_split)] = value.detach().to(device, dtype=dtype)
        return combined_state_dict
    else:
        return torch.load(path_to_file, map_location=device)

def load_custom_adapter_model(
        path_to_file,
        device: Union[str] = 'cpu',
        dtype: torch.dtype = torch.float32
):
    # check if it is safetensors or checkpoint
    if path_to_file.endswith('.safetensors'):
        raw_state_dict = load_file(path_to_file, device)
        combined_state_dict = OrderedDict()
        device = device if isinstance(device, torch.device) else torch.device(device)
        dtype = dtype if isinstance(dtype, torch.dtype) else get_torch_dtype(dtype)
        for combo_key, value in raw_state_dict.items():
            key_split = combo_key.split('.')
            module_name = key_split.pop(0)
            if module_name not in combined_state_dict:
                combined_state_dict[module_name] = OrderedDict()
            combined_state_dict[module_name]['.'.join(key_split)] = value.detach().to(device, dtype=dtype)
        return combined_state_dict
    else:
        return torch.load(path_to_file, map_location=device)


def get_lora_keymap_from_model_keymap(model_keymap: 'OrderedDict') -> 'OrderedDict':
    lora_keymap = OrderedDict()

    # see if we have dual text encoders " a key that starts with conditioner.embedders.1
    has_dual_text_encoders = False
    for key in model_keymap:
        if key.startswith('conditioner.embedders.1'):
            has_dual_text_encoders = True
            break
    # map through the keys and values
    for key, value in model_keymap.items():
        # ignore bias weights
        if key.endswith('bias'):
            continue
        if key.endswith('.weight'):
            # remove the .weight
            key = key[:-7]
        if value.endswith(".weight"):
            # remove the .weight
            value = value[:-7]

        # unet for all
        key = key.replace('model.diffusion_model', 'lora_unet')
        if value.startswith('unet'):
            value = f"lora_{value}"

        # text encoder
        if has_dual_text_encoders:
            key = key.replace('conditioner.embedders.0', 'lora_te1')
            key = key.replace('conditioner.embedders.1', 'lora_te2')
            if value.startswith('te0') or value.startswith('te1'):
                value = f"lora_{value}"
            value.replace('lora_te1', 'lora_te2')
            value.replace('lora_te0', 'lora_te1')

        key = key.replace('cond_stage_model.transformer', 'lora_te')

        if value.startswith('te_'):
            value = f"lora_{value}"

        # replace periods with underscores
        key = key.replace('.', '_')
        value = value.replace('.', '_')

        # add all the weights
        lora_keymap[f"{key}.lora_down.weight"] = f"{value}.lora_down.weight"
        lora_keymap[f"{key}.lora_down.bias"] = f"{value}.lora_down.bias"
        lora_keymap[f"{key}.lora_up.weight"] = f"{value}.lora_up.weight"
        lora_keymap[f"{key}.lora_up.bias"] = f"{value}.lora_up.bias"
        lora_keymap[f"{key}.alpha"] = f"{value}.alpha"

    return lora_keymap
