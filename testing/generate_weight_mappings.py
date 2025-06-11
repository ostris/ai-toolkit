import argparse
import gc
import os
import re
import os
# add project root to sys path
import sys

from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers.loaders import LoraLoaderMixin
from safetensors.torch import load_file, save_file
from collections import OrderedDict
import json
from tqdm import tqdm

from toolkit.config_modules import ModelConfig
from toolkit.stable_diffusion_model import StableDiffusion

KEYMAPS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'toolkit', 'keymaps')

device = torch.device('cpu')
dtype = torch.float32


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def get_reduced_shape(shape_tuple):
    # iterate though shape anr remove 1s
    new_shape = []
    for dim in shape_tuple:
        if dim != 1:
            new_shape.append(dim)
    return tuple(new_shape)


parser = argparse.ArgumentParser()

# require at lease one config file
parser.add_argument(
    'file_1',
    nargs='+',
    type=str,
    help='Path to first safe tensor file'
)

parser.add_argument('--name', type=str, default='stable_diffusion', help='name for mapping to make')
parser.add_argument('--sdxl', action='store_true', help='is sdxl model')
parser.add_argument('--refiner', action='store_true', help='is refiner model')
parser.add_argument('--ssd', action='store_true', help='is ssd model')
parser.add_argument('--vega', action='store_true', help='is vega model')
parser.add_argument('--sd2', action='store_true', help='is sd 2 model')

args = parser.parse_args()

file_path = args.file_1[0]

find_matches = False

print(f'Loading diffusers model')

ignore_ldm_begins_with = []

diffusers_file_path = file_path if len(args.file_1) == 1 else args.file_1[1]
if args.ssd:
    diffusers_file_path = "segmind/SSD-1B"
if args.vega:
    diffusers_file_path = "segmind/Segmind-Vega"

# if args.refiner:
#     diffusers_file_path = "stabilityai/stable-diffusion-xl-refiner-1.0"

if not args.refiner:

    diffusers_model_config = ModelConfig(
        name_or_path=diffusers_file_path,
        is_xl=args.sdxl,
        is_v2=args.sd2,
        is_ssd=args.ssd,
        is_vega=args.vega,
        dtype=dtype,
    )
    diffusers_sd = StableDiffusion(
        model_config=diffusers_model_config,
        device=device,
        dtype=dtype,
    )
    diffusers_sd.load_model()
    # delete things we dont need
    del diffusers_sd.tokenizer
    flush()

    print(f'Loading ldm model')
    diffusers_state_dict = diffusers_sd.state_dict()
else:
    # refiner wont work directly with stable diffusion
    # so we need to load the model and then load the state dict
    diffusers_pipeline = StableDiffusionXLPipeline.from_single_file(
        diffusers_file_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    # diffusers_pipeline = StableDiffusionXLPipeline.from_single_file(
    #     file_path,
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    #     variant="fp16",
    # ).to(device)

    SD_PREFIX_VAE = "vae"
    SD_PREFIX_UNET = "unet"
    SD_PREFIX_REFINER_UNET = "refiner_unet"
    SD_PREFIX_TEXT_ENCODER = "te"

    SD_PREFIX_TEXT_ENCODER1 = "te0"
    SD_PREFIX_TEXT_ENCODER2 = "te1"

    diffusers_state_dict = OrderedDict()
    for k, v in diffusers_pipeline.vae.state_dict().items():
        new_key = k if k.startswith(f"{SD_PREFIX_VAE}") else f"{SD_PREFIX_VAE}_{k}"
        diffusers_state_dict[new_key] = v
    for k, v in diffusers_pipeline.text_encoder_2.state_dict().items():
        new_key = k if k.startswith(f"{SD_PREFIX_TEXT_ENCODER2}_") else f"{SD_PREFIX_TEXT_ENCODER2}_{k}"
        diffusers_state_dict[new_key] = v
    for k, v in diffusers_pipeline.unet.state_dict().items():
        new_key = k if k.startswith(f"{SD_PREFIX_UNET}_") else f"{SD_PREFIX_UNET}_{k}"
        diffusers_state_dict[new_key] = v

    # add ignore ones as we are only going to focus on unet and copy the rest
    # ignore_ldm_begins_with = ["conditioner.", "first_stage_model."]

diffusers_dict_keys = list(diffusers_state_dict.keys())

ldm_state_dict = load_file(file_path)
ldm_dict_keys = list(ldm_state_dict.keys())

ldm_diffusers_keymap = OrderedDict()
ldm_diffusers_shape_map = OrderedDict()
ldm_operator_map = OrderedDict()
diffusers_operator_map = OrderedDict()

total_keys = len(ldm_dict_keys)

matched_ldm_keys = []
matched_diffusers_keys = []

error_margin = 1e-8

tmp_merge_key = "TMP___MERGE"

te_suffix = ''
proj_pattern_weight = None
proj_pattern_bias = None
text_proj_layer = None
if args.sdxl or args.ssd or args.vega:
    te_suffix = '1'
    ldm_res_block_prefix = "conditioner.embedders.1.model.transformer.resblocks"
    proj_pattern_weight = r"conditioner\.embedders\.1\.model\.transformer\.resblocks\.(\d+)\.attn\.in_proj_weight"
    proj_pattern_bias = r"conditioner\.embedders\.1\.model\.transformer\.resblocks\.(\d+)\.attn\.in_proj_bias"
    text_proj_layer = "conditioner.embedders.1.model.text_projection"
if args.refiner:
    te_suffix = '1'
    ldm_res_block_prefix = "conditioner.embedders.0.model.transformer.resblocks"
    proj_pattern_weight = r"conditioner\.embedders\.0\.model\.transformer\.resblocks\.(\d+)\.attn\.in_proj_weight"
    proj_pattern_bias = r"conditioner\.embedders\.0\.model\.transformer\.resblocks\.(\d+)\.attn\.in_proj_bias"
    text_proj_layer = "conditioner.embedders.0.model.text_projection"
if args.sd2:
    te_suffix = ''
    ldm_res_block_prefix = "cond_stage_model.model.transformer.resblocks"
    proj_pattern_weight = r"cond_stage_model\.model\.transformer\.resblocks\.(\d+)\.attn\.in_proj_weight"
    proj_pattern_bias = r"cond_stage_model\.model\.transformer\.resblocks\.(\d+)\.attn\.in_proj_bias"
    text_proj_layer = "cond_stage_model.model.text_projection"

if args.sdxl or args.sd2 or args.ssd or args.refiner or args.vega:
    if "conditioner.embedders.1.model.text_projection" in ldm_dict_keys:
        # d_model = int(checkpoint[prefix + "text_projection"].shape[0]))
        d_model = int(ldm_state_dict["conditioner.embedders.1.model.text_projection"].shape[0])
    elif "conditioner.embedders.1.model.text_projection.weight" in ldm_dict_keys:
        # d_model = int(checkpoint[prefix + "text_projection"].shape[0]))
        d_model = int(ldm_state_dict["conditioner.embedders.1.model.text_projection.weight"].shape[0])
    elif "conditioner.embedders.0.model.text_projection" in ldm_dict_keys:
        # d_model = int(checkpoint[prefix + "text_projection"].shape[0]))
        d_model = int(ldm_state_dict["conditioner.embedders.0.model.text_projection"].shape[0])
    else:
        d_model = 1024

    # do pre known merging
    for ldm_key in ldm_dict_keys:
        try:
            match = re.match(proj_pattern_weight, ldm_key)
            if match:
                if ldm_key == "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight":
                    print("here")
                number = int(match.group(1))
                new_val = torch.cat([
                    diffusers_state_dict[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.q_proj.weight"],
                    diffusers_state_dict[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.k_proj.weight"],
                    diffusers_state_dict[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.v_proj.weight"],
                ], dim=0)
                # add to matched so we dont check them
                matched_diffusers_keys.append(
                    f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.q_proj.weight")
                matched_diffusers_keys.append(
                    f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.k_proj.weight")
                matched_diffusers_keys.append(
                    f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.v_proj.weight")
                # make diffusers convertable_dict
                diffusers_state_dict[
                    f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.{tmp_merge_key}.weight"] = new_val

                # add operator
                ldm_operator_map[ldm_key] = {
                    "cat": [
                        f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.q_proj.weight",
                        f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.k_proj.weight",
                        f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.v_proj.weight",
                    ],
                }

                matched_ldm_keys.append(ldm_key)

                # text_model_dict[new_key + ".q_proj.weight"] = checkpoint[key][:d_model, :]
                # text_model_dict[new_key + ".k_proj.weight"] = checkpoint[key][d_model: d_model * 2, :]
                # text_model_dict[new_key + ".v_proj.weight"] = checkpoint[key][d_model * 2:, :]

                # add diffusers operators
                diffusers_operator_map[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.q_proj.weight"] = {
                    "slice": [
                        f"{ldm_res_block_prefix}.{number}.attn.in_proj_weight",
                        f"0:{d_model}, :"
                    ]
                }
                diffusers_operator_map[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.k_proj.weight"] = {
                    "slice": [
                        f"{ldm_res_block_prefix}.{number}.attn.in_proj_weight",
                        f"{d_model}:{d_model * 2}, :"
                    ]
                }
                diffusers_operator_map[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.v_proj.weight"] = {
                    "slice": [
                        f"{ldm_res_block_prefix}.{number}.attn.in_proj_weight",
                        f"{d_model * 2}:, :"
                    ]
                }

            match = re.match(proj_pattern_bias, ldm_key)
            if match:
                number = int(match.group(1))
                new_val = torch.cat([
                    diffusers_state_dict[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.q_proj.bias"],
                    diffusers_state_dict[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.k_proj.bias"],
                    diffusers_state_dict[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.v_proj.bias"],
                ], dim=0)
                # add to matched so we dont check them
                matched_diffusers_keys.append(f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.q_proj.bias")
                matched_diffusers_keys.append(f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.k_proj.bias")
                matched_diffusers_keys.append(f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.v_proj.bias")
                # make diffusers convertable_dict
                diffusers_state_dict[
                    f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.{tmp_merge_key}.bias"] = new_val

                # add operator
                ldm_operator_map[ldm_key] = {
                    "cat": [
                        f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.q_proj.bias",
                        f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.k_proj.bias",
                        f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.v_proj.bias",
                    ],
                }

                matched_ldm_keys.append(ldm_key)

                # add diffusers operators
                diffusers_operator_map[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.q_proj.bias"] = {
                    "slice": [
                        f"{ldm_res_block_prefix}.{number}.attn.in_proj_bias",
                        f"0:{d_model}, :"
                    ]
                }
                diffusers_operator_map[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.k_proj.bias"] = {
                    "slice": [
                        f"{ldm_res_block_prefix}.{number}.attn.in_proj_bias",
                        f"{d_model}:{d_model * 2}, :"
                    ]
                }
                diffusers_operator_map[f"te{te_suffix}_text_model.encoder.layers.{number}.self_attn.v_proj.bias"] = {
                    "slice": [
                        f"{ldm_res_block_prefix}.{number}.attn.in_proj_bias",
                        f"{d_model * 2}:, :"
                    ]
                }
        except Exception as e:
            print(f"Error on key {ldm_key}")
            print(e)

    # update keys
    diffusers_dict_keys = list(diffusers_state_dict.keys())

pbar = tqdm(ldm_dict_keys, desc='Matching ldm-diffusers keys', total=total_keys)
# run through all weights and check mse between them to find matches
for ldm_key in ldm_dict_keys:
    ldm_shape_tuple = ldm_state_dict[ldm_key].shape
    ldm_reduced_shape_tuple = get_reduced_shape(ldm_shape_tuple)
    for diffusers_key in diffusers_dict_keys:
        if ldm_key == "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight" and diffusers_key == "te1_text_model.encoder.layers.0.self_attn.q_proj.weight":
            print("here")

        diffusers_shape_tuple = diffusers_state_dict[diffusers_key].shape
        diffusers_reduced_shape_tuple = get_reduced_shape(diffusers_shape_tuple)

        # That was easy. Same key
        # if ldm_key == diffusers_key:
        #     ldm_diffusers_keymap[ldm_key] = diffusers_key
        #     matched_ldm_keys.append(ldm_key)
        #     matched_diffusers_keys.append(diffusers_key)
        #     break

        # if we already have this key mapped, skip it
        if diffusers_key in matched_diffusers_keys:
            continue

        # if reduced shapes do not match skip it
        if ldm_reduced_shape_tuple != diffusers_reduced_shape_tuple:
            continue

        ldm_weight = ldm_state_dict[ldm_key]
        did_reduce_ldm = False
        diffusers_weight = diffusers_state_dict[diffusers_key]
        did_reduce_diffusers = False

        # reduce the shapes to match if they are not the same
        if ldm_shape_tuple != ldm_reduced_shape_tuple:
            ldm_weight = ldm_weight.view(ldm_reduced_shape_tuple)
            did_reduce_ldm = True

        if diffusers_shape_tuple != diffusers_reduced_shape_tuple:
            diffusers_weight = diffusers_weight.view(diffusers_reduced_shape_tuple)
            did_reduce_diffusers = True

        # check to see if they match within a margin of error
        mse = torch.nn.functional.mse_loss(ldm_weight.float(), diffusers_weight.float())
        if mse < error_margin:
            ldm_diffusers_keymap[ldm_key] = diffusers_key
            matched_ldm_keys.append(ldm_key)
            matched_diffusers_keys.append(diffusers_key)

            if did_reduce_ldm or did_reduce_diffusers:
                ldm_diffusers_shape_map[ldm_key] = (ldm_shape_tuple, diffusers_shape_tuple)
                if did_reduce_ldm:
                    del ldm_weight
                if did_reduce_diffusers:
                    del diffusers_weight
                flush()

            break

    pbar.update(1)

pbar.close()

name = args.name
if args.sdxl:
    name += '_sdxl'
elif args.ssd:
    name += '_ssd'
elif args.vega:
    name += '_vega'
elif args.refiner:
    name += '_refiner'
elif args.sd2:
    name += '_sd2'
else:
    name += '_sd1'

# if len(matched_ldm_keys) != len(matched_diffusers_keys):
unmatched_ldm_keys = [x for x in ldm_dict_keys if x not in matched_ldm_keys]
unmatched_diffusers_keys = [x for x in diffusers_dict_keys if x not in matched_diffusers_keys]
# has unmatched keys

has_unmatched_keys = len(unmatched_ldm_keys) > 0 or len(unmatched_diffusers_keys) > 0


def get_slices_from_string(s: str) -> tuple:
    slice_strings = s.split(',')
    slices = [eval(f"slice({component.strip()})") for component in slice_strings]
    return tuple(slices)


if has_unmatched_keys:

    print(
        f"Found {len(unmatched_ldm_keys)} unmatched ldm keys and {len(unmatched_diffusers_keys)} unmatched diffusers keys")

    unmatched_obj = OrderedDict()
    unmatched_obj['ldm'] = OrderedDict()
    unmatched_obj['diffusers'] = OrderedDict()

    print(f"Gathering info on unmatched keys")

    for key in tqdm(unmatched_ldm_keys, desc='Unmatched LDM keys'):
        # get min, max, mean, std
        weight = ldm_state_dict[key]
        weight_min = weight.min().item()
        weight_max = weight.max().item()
        unmatched_obj['ldm'][key] = {
            'shape': weight.shape,
            "min": weight_min,
            "max": weight_max,
        }
        del weight
        flush()

    for key in tqdm(unmatched_diffusers_keys, desc='Unmatched Diffusers keys'):
        # get min, max, mean, std
        weight = diffusers_state_dict[key]
        weight_min = weight.min().item()
        weight_max = weight.max().item()
        unmatched_obj['diffusers'][key] = {
            "shape": weight.shape,
            "min": weight_min,
            "max": weight_max,
        }
        del weight
        flush()

    unmatched_path = os.path.join(KEYMAPS_FOLDER, f'{name}_unmatched.json')
    with open(unmatched_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(unmatched_obj, indent=4, ensure_ascii=False))

    print(f'Saved unmatched keys to {unmatched_path}')

# save ldm remainders
remaining_ldm_values = OrderedDict()
for key in unmatched_ldm_keys:
    remaining_ldm_values[key] = ldm_state_dict[key].detach().to('cpu', torch.float16)

save_file(remaining_ldm_values, os.path.join(KEYMAPS_FOLDER, f'{name}_ldm_base.safetensors'))
print(f'Saved remaining ldm values to {os.path.join(KEYMAPS_FOLDER, f"{name}_ldm_base.safetensors")}')

# do cleanup of some left overs and bugs
to_remove = []
for ldm_key, diffusers_key in ldm_diffusers_keymap.items():
    # get rid of tmp merge keys used to slicing
    if tmp_merge_key in diffusers_key or tmp_merge_key in ldm_key:
        to_remove.append(ldm_key)

for key in to_remove:
    del ldm_diffusers_keymap[key]

to_remove = []
# remove identical shape mappings. Not sure why they exist but they do
for ldm_key, shape_list in ldm_diffusers_shape_map.items():
    # remove identical shape mappings. Not sure why they exist but they do
    # convert to json string to make it easier to compare
    ldm_shape = json.dumps(shape_list[0], ensure_ascii=False)
    diffusers_shape = json.dumps(shape_list[1], ensure_ascii=False)
    if ldm_shape == diffusers_shape:
        to_remove.append(ldm_key)

for key in to_remove:
    del ldm_diffusers_shape_map[key]

dest_path = os.path.join(KEYMAPS_FOLDER, f'{name}.json')
save_obj = OrderedDict()
save_obj["ldm_diffusers_keymap"] = ldm_diffusers_keymap
save_obj["ldm_diffusers_shape_map"] = ldm_diffusers_shape_map
save_obj["ldm_diffusers_operator_map"] = ldm_operator_map
save_obj["diffusers_ldm_operator_map"] = diffusers_operator_map
with open(dest_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(save_obj, indent=4, ensure_ascii=False))

print(f'Saved keymap to {dest_path}')
