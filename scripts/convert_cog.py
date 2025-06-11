import json
from collections import OrderedDict
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file

device = torch.device('cpu')

# [diffusers] -> kohya
embedding_mapping = {
    'text_encoders_0': 'clip_l',
    'text_encoders_1': 'clip_g'
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYMAP_ROOT = os.path.join(PROJECT_ROOT, 'toolkit', 'keymaps')
sdxl_keymap_path = os.path.join(KEYMAP_ROOT, 'stable_diffusion_locon_sdxl.json')

# load keymap
with open(sdxl_keymap_path, 'r', encoding='utf-8') as f:
    ldm_diffusers_keymap = json.load(f)['ldm_diffusers_keymap']

# invert the item / key pairs
diffusers_ldm_keymap = {v: k for k, v in ldm_diffusers_keymap.items()}


def get_ldm_key(diffuser_key):
    diffuser_key = f"lora_unet_{diffuser_key.replace('.', '_')}"
    diffuser_key = diffuser_key.replace('_lora_down_weight', '.lora_down.weight')
    diffuser_key = diffuser_key.replace('_lora_up_weight', '.lora_up.weight')
    diffuser_key = diffuser_key.replace('_alpha', '.alpha')
    diffuser_key = diffuser_key.replace('_processor_to_', '_to_')
    diffuser_key = diffuser_key.replace('_to_out.', '_to_out_0.')
    if diffuser_key in diffusers_ldm_keymap:
        return diffusers_ldm_keymap[diffuser_key]
    else:
        raise KeyError(f"Key {diffuser_key} not found in keymap")


def convert_cog(lora_path, embedding_path):
    embedding_state_dict = OrderedDict()
    lora_state_dict = OrderedDict()

    # # normal dict
    # normal_dict = OrderedDict()
    # example_path = "/mnt/Models/stable-diffusion/models/LoRA/sdxl/LogoRedmond_LogoRedAF.safetensors"
    # with safe_open(example_path, framework="pt", device='cpu') as f:
    #     keys = list(f.keys())
    #     for key in keys:
    #         normal_dict[key] = f.get_tensor(key)

    with safe_open(embedding_path, framework="pt", device='cpu') as f:
        keys = list(f.keys())
        for key in keys:
            new_key = embedding_mapping[key]
            embedding_state_dict[new_key] = f.get_tensor(key)

    with safe_open(lora_path, framework="pt", device='cpu') as f:
        keys = list(f.keys())
        lora_rank = None

        # get the lora dim first. Check first 3 linear layers just to be safe
        for key in keys:
            new_key = get_ldm_key(key)
            tensor = f.get_tensor(key)
            num_checked = 0
            if len(tensor.shape) == 2:
                this_dim = min(tensor.shape)
                if lora_rank is None:
                    lora_rank = this_dim
                elif lora_rank != this_dim:
                    raise ValueError(f"lora rank is not consistent, got {tensor.shape}")
                else:
                    num_checked += 1
            if num_checked >= 3:
                break

        for key in keys:
            new_key = get_ldm_key(key)
            tensor = f.get_tensor(key)
            if new_key.endswith('.lora_down.weight'):
                alpha_key = new_key.replace('.lora_down.weight', '.alpha')
                # diffusers does not have alpha, they usa an alpha multiplier of 1 which is a tensor weight of the dims
                # assume first smallest dim is the lora rank if shape is 2
                lora_state_dict[alpha_key] = torch.ones(1).to(tensor.device, tensor.dtype) * lora_rank

            lora_state_dict[new_key] = tensor

    return lora_state_dict, embedding_state_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'lora_path',
        type=str,
        help='Path to lora file'
    )
    parser.add_argument(
        'embedding_path',
        type=str,
        help='Path to embedding file'
    )

    parser.add_argument(
        '--lora_output',
        type=str,
        default="lora_output",
    )

    parser.add_argument(
        '--embedding_output',
        type=str,
        default="embedding_output",
    )

    args = parser.parse_args()

    lora_state_dict, embedding_state_dict = convert_cog(args.lora_path, args.embedding_path)

    # save them
    save_file(lora_state_dict, args.lora_output)
    save_file(embedding_state_dict, args.embedding_output)
    print(f"Saved lora to {args.lora_output}")
    print(f"Saved embedding to {args.embedding_output}")
