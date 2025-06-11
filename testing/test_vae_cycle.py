import os

import torch
from safetensors.torch import load_file
from collections import OrderedDict
from toolkit.kohya_model_util import load_vae, convert_diffusers_back_to_ldm, vae_keys_squished_on_diffusers
import json
# this was just used to match the vae keys to the diffusers keys
# you probably wont need this. Unless they change them.... again... again
# on second thought, you probably will

device = torch.device('cpu')
dtype = torch.float32
vae_path = '/mnt/Models/stable-diffusion/models/VAE/vae-ft-mse-840000-ema-pruned/vae-ft-mse-840000-ema-pruned.safetensors'

find_matches = False

state_dict_ldm = load_file(vae_path)
diffusers_vae = load_vae(vae_path, dtype=torch.float32).to(device)

ldm_keys = state_dict_ldm.keys()

matched_keys = {}
duplicated_keys = {

}

if find_matches:
    # find values that match with a very low mse
    for ldm_key in ldm_keys:
        ldm_value = state_dict_ldm[ldm_key]
        for diffusers_key in list(diffusers_vae.state_dict().keys()):
            diffusers_value = diffusers_vae.state_dict()[diffusers_key]
            if diffusers_key in vae_keys_squished_on_diffusers:
                diffusers_value = diffusers_value.clone().unsqueeze(-1).unsqueeze(-1)
            # if they are not same shape, skip
            if ldm_value.shape != diffusers_value.shape:
                continue
            mse = torch.nn.functional.mse_loss(ldm_value, diffusers_value)
            if mse < 1e-6:
                if ldm_key in list(matched_keys.keys()):
                    print(f'{ldm_key} already matched to {matched_keys[ldm_key]}')
                    if ldm_key in duplicated_keys:
                        duplicated_keys[ldm_key].append(diffusers_key)
                    else:
                        duplicated_keys[ldm_key] = [diffusers_key]
                    continue
                matched_keys[ldm_key] = diffusers_key
                is_matched = True
                break

    print(f'Found {len(matched_keys)} matches')

dif_to_ldm_state_dict = convert_diffusers_back_to_ldm(diffusers_vae)
dif_to_ldm_state_dict_keys = list(dif_to_ldm_state_dict.keys())
keys_in_both = []

keys_not_in_diffusers = []
for key in ldm_keys:
    if key not in dif_to_ldm_state_dict_keys:
        keys_not_in_diffusers.append(key)

keys_not_in_ldm = []
for key in dif_to_ldm_state_dict_keys:
    if key not in ldm_keys:
        keys_not_in_ldm.append(key)

keys_in_both = []
for key in ldm_keys:
    if key in dif_to_ldm_state_dict_keys:
        keys_in_both.append(key)

# sort them
keys_not_in_diffusers.sort()
keys_not_in_ldm.sort()
keys_in_both.sort()

# print(f'Keys in LDM but not in Diffusers: {len(keys_not_in_diffusers)}{keys_not_in_diffusers}')
# print(f'Keys in Diffusers but not in LDM: {len(keys_not_in_ldm)}{keys_not_in_ldm}')
# print(f'Keys in both: {len(keys_in_both)}{keys_in_both}')

json_data = {
    "both": keys_in_both,
    "ldm": keys_not_in_diffusers,
    "diffusers": keys_not_in_ldm
}
json_data = json.dumps(json_data, indent=4, ensure_ascii=False)

remaining_diffusers_values = OrderedDict()
for key in keys_not_in_ldm:
    remaining_diffusers_values[key] = dif_to_ldm_state_dict[key]

# print(remaining_diffusers_values.keys())

remaining_ldm_values = OrderedDict()
for key in keys_not_in_diffusers:
    remaining_ldm_values[key] = state_dict_ldm[key]

# print(json_data)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_save_path = os.path.join(project_root, 'config', 'keys.json')
json_matched_save_path = os.path.join(project_root, 'config', 'matched.json')
json_duped_save_path = os.path.join(project_root, 'config', 'duped.json')

with open(json_save_path, 'w', encoding='utf-8') as f:
    f.write(json_data)
if find_matches:
    with open(json_matched_save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(matched_keys, indent=4, ensure_ascii=False))
    with open(json_duped_save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(duplicated_keys, indent=4, ensure_ascii=False))
