import argparse
import os
# add project root to sys path
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers.loaders import LoraLoaderMixin
from safetensors.torch import load_file
from collections import OrderedDict
import json

from toolkit.config_modules import ModelConfig
from toolkit.paths import KEYMAPS_ROOT
from toolkit.saving import convert_state_dict_to_ldm_with_mapping, get_ldm_state_dict_from_diffusers
from toolkit.stable_diffusion_model import StableDiffusion

# this was just used to match the vae keys to the diffusers keys
# you probably wont need this. Unless they change them.... again... again
# on second thought, you probably will

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = torch.device('cpu')
dtype = torch.float32

parser = argparse.ArgumentParser()

# require at lease one config file
parser.add_argument(
    'file_1',
    nargs='+',
    type=str,
    help='Path an LDM model'
)

parser.add_argument(
    '--is_xl',
    action='store_true',
    help='Is the model an XL model'
)

parser.add_argument(
    '--is_v2',
    action='store_true',
    help='Is the model a v2 model'
)

args = parser.parse_args()

find_matches = False

print("Loading model")
state_dict_file_1 = load_file(args.file_1[0])
state_dict_1_keys = list(state_dict_file_1.keys())

print("Loading model into diffusers format")
model_config = ModelConfig(
    name_or_path=args.file_1[0],
    is_xl=args.is_xl
)
sd = StableDiffusion(
    model_config=model_config,
    device=device,
)
sd.load_model()

# load our base
base_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_sdxl_ldm_base.safetensors')
mapping_path = os.path.join(KEYMAPS_ROOT, 'stable_diffusion_sdxl.json')

print("Converting model back to LDM")
version_string = '1'
if args.is_v2:
    version_string = '2'
if args.is_xl:
    version_string = 'sdxl'
# convert the state dict
state_dict_file_2 = get_ldm_state_dict_from_diffusers(
    sd.state_dict(),
    version_string,
    device='cpu',
    dtype=dtype
)

# state_dict_file_2 = load_file(args.file_2[0])

state_dict_2_keys = list(state_dict_file_2.keys())
keys_in_both = []

keys_not_in_state_dict_2 = []
for key in state_dict_1_keys:
    if key not in state_dict_2_keys:
        keys_not_in_state_dict_2.append(key)

keys_not_in_state_dict_1 = []
for key in state_dict_2_keys:
    if key not in state_dict_1_keys:
        keys_not_in_state_dict_1.append(key)

keys_in_both = []
for key in state_dict_1_keys:
    if key in state_dict_2_keys:
        keys_in_both.append(key)

# sort them
keys_not_in_state_dict_2.sort()
keys_not_in_state_dict_1.sort()
keys_in_both.sort()

if len(keys_not_in_state_dict_2) == 0 and len(keys_not_in_state_dict_1) == 0:
    print("All keys match!")
    print("Checking values...")
    mismatch_keys = []
    loss = torch.nn.MSELoss()
    tolerance = 1e-6
    for key in tqdm(keys_in_both):
        if loss(state_dict_file_1[key], state_dict_file_2[key]) > tolerance:
            print(f"Values for key {key} don't match!")
            print(f"Loss: {loss(state_dict_file_1[key], state_dict_file_2[key])}")
            mismatch_keys.append(key)

    if len(mismatch_keys) == 0:
        print("All values match!")
    else:
        print("Some valued font match!")
        print(mismatch_keys)
        mismatched_path = os.path.join(project_root, 'config', 'mismatch.json')
        with open(mismatched_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(mismatch_keys, indent=4, ensure_ascii=False))
        exit(0)

else:
    print("Keys don't match!, generating info...")

json_data = {
    "both": keys_in_both,
    "not_in_state_dict_2": keys_not_in_state_dict_2,
    "not_in_state_dict_1": keys_not_in_state_dict_1
}
json_data = json.dumps(json_data, indent=4, ensure_ascii=False)

remaining_diffusers_values = OrderedDict()
for key in keys_not_in_state_dict_1:
    remaining_diffusers_values[key] = state_dict_file_2[key]

# print(remaining_diffusers_values.keys())

remaining_ldm_values = OrderedDict()
for key in keys_not_in_state_dict_2:
    remaining_ldm_values[key] = state_dict_file_1[key]

# print(json_data)


json_save_path = os.path.join(project_root, 'config', 'keys.json')
json_matched_save_path = os.path.join(project_root, 'config', 'matched.json')
json_duped_save_path = os.path.join(project_root, 'config', 'duped.json')
state_dict_1_filename = os.path.basename(args.file_1[0])
# state_dict_2_filename = os.path.basename(args.file_2[0])
# save key names for each in own file
with open(os.path.join(project_root, 'config', f'{state_dict_1_filename}.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(state_dict_1_keys, indent=4, ensure_ascii=False))

with open(os.path.join(project_root, 'config', f'{state_dict_1_filename}_loop.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(state_dict_2_keys, indent=4, ensure_ascii=False))

with open(json_save_path, 'w', encoding='utf-8') as f:
    f.write(json_data)
