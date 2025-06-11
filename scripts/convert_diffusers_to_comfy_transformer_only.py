#######################################################
# Convert Diffusers Flux/Flex to diffusion model ComfyUI safetensors file
# This will only have the transformer weights, not the TEs and VAE
# You can save the transformer weights as bf16 or 8-bit with the --do_8_bit flag
# You can also save with scaled 8-bit using the --do_8bit_scaled flag
#
# Call like this for 8-bit transformer weights with stochastic rounding:
# python convert_diffusers_to_comfy_transformer_only.py  /path/to/diffusers/checkpoint /output/path/my_finetune.safetensors --do_8_bit
#
# Call like this for 8-bit transformer weights with scaling:
# python convert_diffusers_to_comfy_transformer_only.py  /path/to/diffusers/checkpoint /output/path/my_finetune.safetensors --do_8bit_scaled
#
# Call like this for bf16 transformer weights:
# python convert_diffusers_to_comfy_transformer_only.py  /path/to/diffusers/checkpoint /output/path/my_finetune.safetensors
#
# Output should go in ComfyUI/models/diffusion_models/
#
#######################################################


import argparse
from datetime import date
import json
import os
from pathlib import Path
import safetensors
import safetensors.torch
import torch
import tqdm
from collections import OrderedDict


parser = argparse.ArgumentParser()

parser.add_argument("diffusers_path", type=str,
                    help="Path to the original Flux diffusers folder.")
parser.add_argument("flux_path", type=str,
                    help="Output path for the Flux safetensors file.")
parser.add_argument("--do_8_bit", action="store_true",
                    help="Use 8-bit weights with stochastic rounding instead of bf16.")
parser.add_argument("--do_8bit_scaled", action="store_true",
                    help="Use scaled 8-bit weights instead of bf16.")
args = parser.parse_args()

flux_path = Path(args.flux_path)
diffusers_path = Path(args.diffusers_path)

if os.path.exists(os.path.join(diffusers_path, "transformer")):
    diffusers_path = Path(os.path.join(diffusers_path, "transformer"))

do_8_bit = args.do_8_bit
do_8bit_scaled = args.do_8bit_scaled

# Don't allow both flags to be active simultaneously
if do_8_bit and do_8bit_scaled:
    print("Error: Cannot use both --do_8_bit and --do_8bit_scaled at the same time.")
    exit()

if not os.path.exists(flux_path.parent):
    os.makedirs(flux_path.parent)

if not diffusers_path.exists():
    print(f"Error: Missing transformer folder: {diffusers_path}")
    exit()

original_json_path = Path.joinpath(
    diffusers_path, "diffusion_pytorch_model.safetensors.index.json")

if not original_json_path.exists():
    print(f"Error: Missing transformer index json: {original_json_path}")
    exit()

with open(original_json_path, "r", encoding="utf-8") as f:
    original_json = json.load(f)

diffusers_map = {
    "time_in.in_layer.weight": [
        "time_text_embed.timestep_embedder.linear_1.weight",
    ],
    "time_in.in_layer.bias": [
        "time_text_embed.timestep_embedder.linear_1.bias",
    ],
    "time_in.out_layer.weight": [
        "time_text_embed.timestep_embedder.linear_2.weight",
    ],
    "time_in.out_layer.bias": [
        "time_text_embed.timestep_embedder.linear_2.bias",
    ],
    "vector_in.in_layer.weight": [
        "time_text_embed.text_embedder.linear_1.weight",
    ],
    "vector_in.in_layer.bias": [
        "time_text_embed.text_embedder.linear_1.bias",
    ],
    "vector_in.out_layer.weight": [
        "time_text_embed.text_embedder.linear_2.weight",
    ],
    "vector_in.out_layer.bias": [
        "time_text_embed.text_embedder.linear_2.bias",
    ],
    "guidance_in.in_layer.weight": [
        "time_text_embed.guidance_embedder.linear_1.weight",
    ],
    "guidance_in.in_layer.bias": [
        "time_text_embed.guidance_embedder.linear_1.bias",
    ],
    "guidance_in.out_layer.weight": [
        "time_text_embed.guidance_embedder.linear_2.weight",
    ],
    "guidance_in.out_layer.bias": [
        "time_text_embed.guidance_embedder.linear_2.bias",
    ],
    "txt_in.weight": [
        "context_embedder.weight",
    ],
    "txt_in.bias": [
        "context_embedder.bias",
    ],
    "img_in.weight": [
        "x_embedder.weight",
    ],
    "img_in.bias": [
        "x_embedder.bias",
    ],
    "double_blocks.().img_mod.lin.weight": [
        "norm1.linear.weight",
    ],
    "double_blocks.().img_mod.lin.bias": [
        "norm1.linear.bias",
    ],
    "double_blocks.().txt_mod.lin.weight": [
        "norm1_context.linear.weight",
    ],
    "double_blocks.().txt_mod.lin.bias": [
        "norm1_context.linear.bias",
    ],
    "double_blocks.().img_attn.qkv.weight": [
        "attn.to_q.weight",
        "attn.to_k.weight",
        "attn.to_v.weight",
    ],
    "double_blocks.().img_attn.qkv.bias": [
        "attn.to_q.bias",
        "attn.to_k.bias",
        "attn.to_v.bias",
    ],
    "double_blocks.().txt_attn.qkv.weight": [
        "attn.add_q_proj.weight",
        "attn.add_k_proj.weight",
        "attn.add_v_proj.weight",
    ],
    "double_blocks.().txt_attn.qkv.bias": [
        "attn.add_q_proj.bias",
        "attn.add_k_proj.bias",
        "attn.add_v_proj.bias",
    ],
    "double_blocks.().img_attn.norm.query_norm.scale": [
        "attn.norm_q.weight",
    ],
    "double_blocks.().img_attn.norm.key_norm.scale": [
        "attn.norm_k.weight",
    ],
    "double_blocks.().txt_attn.norm.query_norm.scale": [
        "attn.norm_added_q.weight",
    ],
    "double_blocks.().txt_attn.norm.key_norm.scale": [
        "attn.norm_added_k.weight",
    ],
    "double_blocks.().img_mlp.0.weight": [
        "ff.net.0.proj.weight",
    ],
    "double_blocks.().img_mlp.0.bias": [
        "ff.net.0.proj.bias",
    ],
    "double_blocks.().img_mlp.2.weight": [
        "ff.net.2.weight",
    ],
    "double_blocks.().img_mlp.2.bias": [
        "ff.net.2.bias",
    ],
    "double_blocks.().txt_mlp.0.weight": [
        "ff_context.net.0.proj.weight",
    ],
    "double_blocks.().txt_mlp.0.bias": [
        "ff_context.net.0.proj.bias",
    ],
    "double_blocks.().txt_mlp.2.weight": [
        "ff_context.net.2.weight",
    ],
    "double_blocks.().txt_mlp.2.bias": [
        "ff_context.net.2.bias",
    ],
    "double_blocks.().img_attn.proj.weight": [
        "attn.to_out.0.weight",
    ],
    "double_blocks.().img_attn.proj.bias": [
        "attn.to_out.0.bias",
    ],
    "double_blocks.().txt_attn.proj.weight": [
        "attn.to_add_out.weight",
    ],
    "double_blocks.().txt_attn.proj.bias": [
        "attn.to_add_out.bias",
    ],
    "single_blocks.().modulation.lin.weight": [
        "norm.linear.weight",
    ],
    "single_blocks.().modulation.lin.bias": [
        "norm.linear.bias",
    ],
    "single_blocks.().linear1.weight": [
        "attn.to_q.weight",
        "attn.to_k.weight",
        "attn.to_v.weight",
        "proj_mlp.weight",
    ],
    "single_blocks.().linear1.bias": [
        "attn.to_q.bias",
        "attn.to_k.bias",
        "attn.to_v.bias",
        "proj_mlp.bias",
    ],
    "single_blocks.().linear2.weight": [
        "proj_out.weight",
    ],
    "single_blocks.().norm.query_norm.scale": [
        "attn.norm_q.weight",
    ],
    "single_blocks.().norm.key_norm.scale": [
        "attn.norm_k.weight",
    ],
    "single_blocks.().linear2.weight": [
        "proj_out.weight",
    ],
    "single_blocks.().linear2.bias": [
        "proj_out.bias",
    ],
    "final_layer.linear.weight": [
        "proj_out.weight",
    ],
    "final_layer.linear.bias": [
        "proj_out.bias",
    ],
    "final_layer.adaLN_modulation.1.weight": [
        "norm_out.linear.weight",
    ],
    "final_layer.adaLN_modulation.1.bias": [
        "norm_out.linear.bias",
    ],
}


def is_in_diffusers_map(k):
    for values in diffusers_map.values():
        for value in values:
            if k.endswith(value):
                return True
    return False


diffusers = {k: Path.joinpath(diffusers_path, v)
             for k, v in original_json["weight_map"].items() if is_in_diffusers_map(k)}

original_safetensors = set(diffusers.values())

# determine the number of transformer blocks
transformer_blocks = 0
single_transformer_blocks = 0
for key in diffusers.keys():
    print(key)
    if key.startswith("transformer_blocks."):
        print(key)
        block = int(key.split(".")[1])
        if block >= transformer_blocks:
            transformer_blocks = block + 1
    elif key.startswith("single_transformer_blocks."):
        block = int(key.split(".")[1])
        if block >= single_transformer_blocks:
            single_transformer_blocks = block + 1

print(f"Transformer blocks: {transformer_blocks}")
print(f"Single transformer blocks: {single_transformer_blocks}")

for file in original_safetensors:
    if not file.exists():
        print(f"Error: Missing transformer safetensors file: {file}")
        exit()

original_safetensors = {f: safetensors.safe_open(
    f, framework="pt", device="cpu") for f in original_safetensors}


def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


flux_values = {}

for b in range(transformer_blocks):
    for key, weights in diffusers_map.items():
        if key.startswith("double_blocks."):
            block_prefix = f"transformer_blocks.{b}."
            found = True
            for weight in weights:
                if not (f"{block_prefix}{weight}" in diffusers):
                    found = False
            if found:
                flux_values[key.replace("()", f"{b}")] = [
                    f"{block_prefix}{weight}" for weight in weights]
for b in range(single_transformer_blocks):
    for key, weights in diffusers_map.items():
        if key.startswith("single_blocks."):
            block_prefix = f"single_transformer_blocks.{b}."
            found = True
            for weight in weights:
                if not (f"{block_prefix}{weight}" in diffusers):
                    found = False
            if found:
                flux_values[key.replace("()", f"{b}")] = [
                    f"{block_prefix}{weight}" for weight in weights]

for key, weights in diffusers_map.items():
    if not (key.startswith("double_blocks.") or key.startswith("single_blocks.")):
        found = True
        for weight in weights:
            if not (f"{weight}" in diffusers):
                found = False
        if found:
            flux_values[key] = [f"{weight}" for weight in weights]

flux = {}

for key, values in tqdm.tqdm(flux_values.items()):
    if len(values) == 1:
        flux[key] = original_safetensors[diffusers[values[0]]
                                         ].get_tensor(values[0]).to("cpu")
    else:
        flux[key] = torch.cat(
            [
                original_safetensors[diffusers[value]
                                     ].get_tensor(value).to("cpu")
                for value in values
            ]
        )

if "norm_out.linear.weight" in diffusers:
    flux["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(
        original_safetensors[diffusers["norm_out.linear.weight"]].get_tensor(
            "norm_out.linear.weight").to("cpu")
    )
if "norm_out.linear.bias" in diffusers:
    flux["final_layer.adaLN_modulation.1.bias"] = swap_scale_shift(
        original_safetensors[diffusers["norm_out.linear.bias"]].get_tensor(
            "norm_out.linear.bias").to("cpu")
    )


def stochastic_round_to(tensor, dtype=torch.float8_e4m3fn):
    # Define the float8 range
    min_val = torch.finfo(dtype).min
    max_val = torch.finfo(dtype).max

    # Clip values to float8 range
    tensor = torch.clamp(tensor, min_val, max_val)

    # Convert to float32 for calculations
    tensor = tensor.float()

    # Get the nearest representable float8 values
    lower = torch.floor(tensor * 256) / 256
    upper = torch.ceil(tensor * 256) / 256

    # Calculate the probability of rounding up
    prob = (tensor - lower) / (upper - lower)

    # Generate random values for stochastic rounding
    rand = torch.rand_like(tensor)

    # Perform stochastic rounding
    rounded = torch.where(rand < prob, upper, lower)

    # Convert back to float8
    return rounded.to(dtype)


# List of keys that should not be scaled (usually embedding layers and biases)
blacklist = []
for key in flux.keys():
    if not key.endswith(".weight") or "embed" in key:
        blacklist.append(key)

# Function to scale weights for 8-bit quantization
def scale_weights_to_8bit(tensor, max_value=416.0, dtype=torch.float8_e4m3fn):
    # Get the limits of the dtype
    min_val = torch.finfo(dtype).min
    max_val = torch.finfo(dtype).max
    
    # Only process 2D tensors that are not in the blacklist
    if tensor.dim() == 2:
        # Calculate the scaling factor
        abs_max = torch.max(torch.abs(tensor))
        scale = abs_max / max_value
        
        # Scale the tensor and clip to float8 range
        scaled_tensor = (tensor / scale).clip(min=min_val, max=max_val).to(dtype)
        
        return scaled_tensor, scale
    else:
        # For tensors that shouldn't be scaled, just convert to float8
        return tensor.clip(min=min_val, max=max_val).to(dtype), None


# set all the keys to appropriate dtype
if do_8_bit:
    print("Converting to 8-bit with stochastic rounding...")
    for key in flux.keys():
        flux[key] = stochastic_round_to(
            flux[key], torch.float8_e4m3fn).to('cpu')
elif do_8bit_scaled:
    print("Converting to scaled 8-bit...")
    scales = {}
    for key in tqdm.tqdm(flux.keys()):
        if key.endswith(".weight") and key not in blacklist:
            flux[key], scale = scale_weights_to_8bit(flux[key])
            if scale is not None:
                scale_key = key[:-len(".weight")] + ".scale_weight"
                scales[scale_key] = scale
        else:
            # For non-weight tensors or blacklisted ones, just convert without scaling
            min_val = torch.finfo(torch.float8_e4m3fn).min
            max_val = torch.finfo(torch.float8_e4m3fn).max
            flux[key] = flux[key].clip(min=min_val, max=max_val).to(torch.float8_e4m3fn).to('cpu')
    
    # Add all the scales to the flux dictionary
    flux.update(scales)
    
    # Add a marker tensor to indicate this is a scaled fp8 model
    flux["scaled_fp8"] = torch.tensor([]).to(torch.float8_e4m3fn)
else:
    print("Converting to bfloat16...")
    for key in flux.keys():
        flux[key] = flux[key].clone().to('cpu', torch.bfloat16)

meta = OrderedDict()
meta['format'] = 'pt'
# date format like 2024-08-01 YYYY-MM-DD
meta['modelspec.date'] = date.today().strftime("%Y-%m-%d")

os.makedirs(os.path.dirname(flux_path), exist_ok=True)

print(f"Saving to {flux_path}")

safetensors.torch.save_file(flux, flux_path, metadata=meta)

print("Done.")