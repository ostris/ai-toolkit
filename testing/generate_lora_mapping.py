from collections import OrderedDict

import torch
from safetensors.torch import load_file
import argparse
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

keymap_path = os.path.join(PROJECT_ROOT, 'toolkit', 'keymaps', 'stable_diffusion_sdxl.json')

# load keymap
with open(keymap_path, 'r', encoding='utf-8') as f:
    keymap = json.load(f)

lora_keymap = OrderedDict()

# convert keymap to lora key naming
for ldm_key, diffusers_key in keymap['ldm_diffusers_keymap'].items():
    if ldm_key.endswith('.bias') or diffusers_key.endswith('.bias'):
        # skip it
        continue
    # sdxl has same te for locon with kohya and ours
    if ldm_key.startswith('conditioner'):
        #skip it
        continue
    # ignore vae
    if ldm_key.startswith('first_stage_model'):
        continue
    ldm_key = ldm_key.replace('model.diffusion_model.', 'lora_unet_')
    ldm_key = ldm_key.replace('.weight', '')
    ldm_key = ldm_key.replace('.', '_')

    diffusers_key = diffusers_key.replace('unet_', 'lora_unet_')
    diffusers_key = diffusers_key.replace('.weight', '')
    diffusers_key = diffusers_key.replace('.', '_')

    lora_keymap[f"{ldm_key}.alpha"] = f"{diffusers_key}.alpha"
    lora_keymap[f"{ldm_key}.lora_down.weight"] = f"{diffusers_key}.lora_down.weight"
    lora_keymap[f"{ldm_key}.lora_up.weight"] = f"{diffusers_key}.lora_up.weight"


parser = argparse.ArgumentParser()
parser.add_argument("input", help="input file")
parser.add_argument("input2", help="input2 file")

args = parser.parse_args()

# name = args.name
# if args.sdxl:
#     name += '_sdxl'
# elif args.sd2:
#     name += '_sd2'
# else:
#     name += '_sd1'
name = 'stable_diffusion_locon_sdxl'

locon_save = load_file(args.input)
our_save = load_file(args.input2)

our_extra_keys = list(set(our_save.keys()) - set(locon_save.keys()))
locon_extra_keys = list(set(locon_save.keys()) - set(our_save.keys()))

print(f"we have {len(our_extra_keys)} extra keys")
print(f"locon has {len(locon_extra_keys)} extra keys")

save_dtype = torch.float16
print(f"our extra keys: {our_extra_keys}")
print(f"locon extra keys: {locon_extra_keys}")


def export_state_dict(our_save):
    converted_state_dict = OrderedDict()
    for key, value in our_save.items():
        # test encoders share keys for some reason
        if key.startswith('lora_te'):
            converted_state_dict[key] = value.detach().to('cpu', dtype=save_dtype)
        else:
            converted_key = key
            for ldm_key, diffusers_key in lora_keymap.items():
                if converted_key == diffusers_key:
                    converted_key = ldm_key

            converted_state_dict[converted_key] = value.detach().to('cpu', dtype=save_dtype)
    return converted_state_dict

def import_state_dict(loaded_state_dict):
    converted_state_dict = OrderedDict()
    for key, value in loaded_state_dict.items():
        if key.startswith('lora_te'):
            converted_state_dict[key] = value.detach().to('cpu', dtype=save_dtype)
        else:
            converted_key = key
            for ldm_key, diffusers_key in lora_keymap.items():
                if converted_key == ldm_key:
                    converted_key = diffusers_key

            converted_state_dict[converted_key] = value.detach().to('cpu', dtype=save_dtype)
    return converted_state_dict


# check it again
converted_state_dict = export_state_dict(our_save)
converted_extra_keys = list(set(converted_state_dict.keys()) - set(locon_save.keys()))
locon_extra_keys = list(set(locon_save.keys()) - set(converted_state_dict.keys()))


print(f"we have {len(converted_extra_keys)} extra keys")
print(f"locon has {len(locon_extra_keys)} extra keys")

print(f"our extra keys: {converted_extra_keys}")

# convert back
cycle_state_dict = import_state_dict(converted_state_dict)
cycle_extra_keys = list(set(cycle_state_dict.keys()) - set(our_save.keys()))
our_extra_keys = list(set(our_save.keys()) - set(cycle_state_dict.keys()))

print(f"we have {len(our_extra_keys)} extra keys")
print(f"cycle has {len(cycle_extra_keys)} extra keys")

# save keymap
to_save = OrderedDict()
to_save['ldm_diffusers_keymap'] = lora_keymap

with open(os.path.join(PROJECT_ROOT, 'toolkit', 'keymaps', f'{name}.json'), 'w', encoding='utf-8') as f:
    json.dump(to_save, f, indent=4, ensure_ascii=False)



