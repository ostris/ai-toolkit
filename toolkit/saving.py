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
    with open(mapping_path, 'r') as f:
        mapping = json.load(f, object_pairs_hook=OrderedDict)

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
        if 'slice' in ldm_diffusers_operator_map[ldm_key]:
            tensor_to_slice = diffusers_state_dict[ldm_diffusers_operator_map[ldm_key]['slice'][0]]
            slice_text = diffusers_state_dict[ldm_diffusers_operator_map[ldm_key]['slice'][1]]
            converted_state_dict[ldm_key] = tensor_to_slice[get_slices_from_string(slice_text)].detach().to(device,
                                                                                                             dtype=dtype)

    # process the rest of the keys
    for ldm_key in ldm_diffusers_keymap:
        # if the key is in the ldm key, we need to process it
        if ldm_diffusers_keymap[ldm_key] in diffusers_state_dict:
            tensor = diffusers_state_dict[ldm_diffusers_keymap[ldm_key]].detach().to(device, dtype=dtype)
            # see if we need to reshape
            if ldm_key in ldm_diffusers_shape_map:
                tensor = tensor.view(ldm_diffusers_shape_map[ldm_key][0])
            converted_state_dict[ldm_key] = tensor

    return converted_state_dict


def save_ldm_model_from_diffusers(
        sd: 'StableDiffusion',
        output_file: str,
        meta: 'OrderedDict',
        save_dtype=get_torch_dtype('fp16'),
        sd_version: Literal['1', '2', 'sdxl'] = '2'
):
    if sd_version != 'sdxl':
        # not supported yet
        raise NotImplementedError("Only SDXL is supported at this time with this method")
    # load our base
    base_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_sdxl_ldm_base.safetensors')
    mapping_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_sdxl.json')

    # convert the state dict
    converted_state_dict = convert_state_dict_to_ldm_with_mapping(
        sd.state_dict(),
        mapping_path,
        base_path,
        device='cpu',
        dtype=save_dtype
    )
    # make sure parent folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_file(converted_state_dict, output_file, metadata=meta)
