import argparse
import os

import torch
from diffusers.loaders import LoraLoaderMixin
from safetensors.torch import load_file
from collections import OrderedDict
import json
# this was just used to match the vae keys to the diffusers keys
# you probably wont need this. Unless they change them.... again... again
# on second thought, you probably will

device = torch.device('cpu')
dtype = torch.float32

parser = argparse.ArgumentParser()

# require at lease one config file
parser.add_argument(
    'file_1',
    nargs='+',
    type=str,
    help='Path to first safe tensor file'
)

parser.add_argument(
    'file_2',
    nargs='+',
    type=str,
    help='Path to second safe tensor file'
)

args = parser.parse_args()

find_matches = False

state_dict_file_1 = load_file(args.file_1[0])
state_dict_1_keys = list(state_dict_file_1.keys())

state_dict_file_2 = load_file(args.file_2[0])
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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_save_path = os.path.join(project_root, 'config', 'keys.json')
json_matched_save_path = os.path.join(project_root, 'config', 'matched.json')
json_duped_save_path = os.path.join(project_root, 'config', 'duped.json')
state_dict_1_filename = os.path.basename(args.file_1[0])
state_dict_2_filename = os.path.basename(args.file_2[0])
# save key names for each in own file
with open(os.path.join(project_root, 'config', f'{state_dict_1_filename}.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(state_dict_1_keys, indent=4, ensure_ascii=False))

with open(os.path.join(project_root, 'config', f'{state_dict_2_filename}.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(state_dict_2_keys, indent=4, ensure_ascii=False))


with open(json_save_path, 'w', encoding='utf-8') as f:
    f.write(json_data)