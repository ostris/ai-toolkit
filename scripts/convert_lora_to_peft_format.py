# currently only works with flux as support is not quite there yet

import argparse
import os.path
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument(
    'input_path',
    type=str,
    help='Path to original sdxl model'
)
parser.add_argument(
    'output_path',
    type=str,
    help='output path'
)
args = parser.parse_args()
args.input_path = os.path.abspath(args.input_path)
args.output_path = os.path.abspath(args.output_path)

from safetensors.torch import load_file, save_file

meta = OrderedDict()
meta['format'] = 'pt'

state_dict = load_file(args.input_path)

# peft doesnt have an alpha so we need to scale the weights
alpha_keys = [
    'lora_transformer_single_transformer_blocks_0_attn_to_q.alpha'  # flux
]

# keys where the rank is in the first dimension
rank_idx0_keys = [
    'lora_transformer_single_transformer_blocks_0_attn_to_q.lora_down.weight'
    # 'transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight'
]

alpha = None
rank = None

for key in rank_idx0_keys:
    if key in state_dict:
        rank = int(state_dict[key].shape[0])
        break

if rank is None:
    raise ValueError(f'Could not find rank in state dict')

for key in alpha_keys:
    if key in state_dict:
        alpha = int(state_dict[key])
        break

if alpha is None:
    # set to rank if not found
    alpha = rank


up_multiplier = alpha / rank

new_state_dict = {}

for key, value in state_dict.items():
    if key.endswith('.alpha'):
        continue

    orig_dtype = value.dtype

    new_val = value.float() * up_multiplier

    new_key = key
    new_key = new_key.replace('lora_transformer_', 'transformer.')
    for i in range(100):
        new_key = new_key.replace(f'transformer_blocks_{i}_', f'transformer_blocks.{i}.')
    new_key = new_key.replace('lora_down', 'lora_A')
    new_key = new_key.replace('lora_up', 'lora_B')
    new_key = new_key.replace('_lora', '.lora')
    new_key = new_key.replace('attn_', 'attn.')
    new_key = new_key.replace('ff_', 'ff.')
    new_key = new_key.replace('context_net_', 'context.net.')
    new_key = new_key.replace('0_proj', '0.proj')
    new_key = new_key.replace('norm_linear', 'norm.linear')
    new_key = new_key.replace('norm_out_linear', 'norm_out.linear')
    new_key = new_key.replace('to_out_', 'to_out.')

    new_state_dict[new_key] = new_val.to(orig_dtype)

save_file(new_state_dict, args.output_path, meta)
print(f'Saved to {args.output_path}')
