import os

import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, PixArtSigmaPipeline, Transformer2DModel, PixArtTransformer2DModel
from safetensors.torch import load_file, save_file
from collections import OrderedDict
import json

# model_path = "/home/jaret/Dev/models/hf/kl-f16-d42_sd15_v01_000527000"
# te_path = "google/flan-t5-xl"
# te_aug_path = "/mnt/Train/out/ip_adapter/t5xx_sd15_v1/t5xx_sd15_v1_000032000.safetensors"
# output_path = "/home/jaret/Dev/models/hf/kl-f16-d42_sd15_t5xl_raw"
model_path = "/home/jaret/Dev/models/hf/objective-reality-16ch"
te_path = "google/flan-t5-xl"
te_aug_path = "/mnt/Train2/out/ip_adapter/t5xl-sd15-16ch_v1/t5xl-sd15-16ch_v1_000115000.safetensors"
output_path = "/home/jaret/Dev/models/hf/t5xl-sd15-16ch_sd15_v1"


print("Loading te adapter")
te_aug_sd = load_file(te_aug_path)

print("Loading model")
is_diffusers = (not os.path.exists(model_path)) or os.path.isdir(model_path)

# if "pixart" in model_path.lower():
is_pixart = "pixart" in model_path.lower()

pipeline_class = StableDiffusionPipeline

# transformer = PixArtTransformer2DModel.from_pretrained('PixArt-alpha/PixArt-Sigma-XL-2-512-MS', subfolder='transformer', torch_dtype=torch.float16)

if is_pixart:
    pipeline_class = PixArtSigmaPipeline

if is_diffusers:
    sd = pipeline_class.from_pretrained(model_path, torch_dtype=torch.float16)
else:
    sd = pipeline_class.from_single_file(model_path, torch_dtype=torch.float16)

print("Loading Text Encoder")
# Load the text encoder
te = T5EncoderModel.from_pretrained(te_path, torch_dtype=torch.float16)

# patch it
sd.text_encoder = te
sd.tokenizer = T5Tokenizer.from_pretrained(te_path)

if is_pixart:
    unet = sd.transformer
    unet_sd = sd.transformer.state_dict()
else:
    unet = sd.unet
    unet_sd = sd.unet.state_dict()


if is_pixart:
    weight_idx = 0
else:
    weight_idx = 1

new_cross_attn_dim = None

# count the num of params in state dict
start_params = sum([v.numel() for v in unet_sd.values()])

print("Building")
attn_processor_keys = []
if is_pixart:
    transformer: Transformer2DModel = unet
    for i, module in transformer.transformer_blocks.named_children():
        attn_processor_keys.append(f"transformer_blocks.{i}.attn1")
        # cross attention
        attn_processor_keys.append(f"transformer_blocks.{i}.attn2")
else:
    attn_processor_keys = list(unet.attn_processors.keys())

for name in attn_processor_keys:
    cross_attention_dim = None if name.endswith("attn1.processor") or name.endswith("attn.1") or name.endswith(
        "attn1") else \
        unet.config['cross_attention_dim']
    if name.startswith("mid_block"):
        hidden_size = unet.config['block_out_channels'][-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config['block_out_channels']))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config['block_out_channels'][block_id]
    elif name.startswith("transformer"):
        hidden_size = unet.config['cross_attention_dim']
    else:
        # they didnt have this, but would lead to undefined below
        raise ValueError(f"unknown attn processor name: {name}")
    if cross_attention_dim is None:
        pass
    else:
        layer_name = name.split(".processor")[0]
        to_k_adapter = unet_sd[layer_name + ".to_k.weight"]
        to_v_adapter = unet_sd[layer_name + ".to_v.weight"]

        te_aug_name = None
        while True:
            if is_pixart:
                te_aug_name = f"te_adapter.adapter_modules.{weight_idx}.to_k_adapter"
            else:
                te_aug_name = f"te_adapter.adapter_modules.{weight_idx}.to_k_adapter"
            if f"{te_aug_name}.weight" in te_aug_sd:
                # increment so we dont redo it next time
                weight_idx += 1
                break
            else:
                weight_idx += 1

            if weight_idx > 1000:
                raise ValueError("Could not find the next weight")

        orig_weight_shape_k = list(unet_sd[layer_name + ".to_k.weight"].shape)
        new_weight_shape_k = list(te_aug_sd[te_aug_name + ".weight"].shape)
        orig_weight_shape_v = list(unet_sd[layer_name + ".to_v.weight"].shape)
        new_weight_shape_v = list(te_aug_sd[te_aug_name.replace('to_k', 'to_v') + ".weight"].shape)

        unet_sd[layer_name + ".to_k.weight"] = te_aug_sd[te_aug_name + ".weight"]
        unet_sd[layer_name + ".to_v.weight"] = te_aug_sd[te_aug_name.replace('to_k', 'to_v') + ".weight"]

        if new_cross_attn_dim is None:
            new_cross_attn_dim = unet_sd[layer_name + ".to_k.weight"].shape[1]



if is_pixart:
    # copy the caption_projection weight
    del unet_sd['caption_projection.linear_1.bias']
    del unet_sd['caption_projection.linear_1.weight']
    del unet_sd['caption_projection.linear_2.bias']
    del unet_sd['caption_projection.linear_2.weight']

print("Saving unmodified model")
sd = sd.to("cpu", torch.float16)
sd.save_pretrained(
    output_path,
    safe_serialization=True,
)

# overwrite the unet
if is_pixart:
    unet_folder = os.path.join(output_path, "transformer")
else:
    unet_folder = os.path.join(output_path, "unet")

# move state_dict to cpu
unet_sd = {k: v.clone().cpu().to(torch.float16) for k, v in unet_sd.items()}

meta = OrderedDict()
meta["format"] = "pt"

print("Patching")

save_file(unet_sd, os.path.join(unet_folder, "diffusion_pytorch_model.safetensors"), meta)

# load the json file
with open(os.path.join(unet_folder, "config.json"), 'r', encoding='utf-8') as f:
    config = json.load(f)

config['cross_attention_dim'] = new_cross_attn_dim

if is_pixart:
    config['caption_channels'] = None

# save it
with open(os.path.join(unet_folder, "config.json"), 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("Done")

new_params = sum([v.numel() for v in unet_sd.values()])

# print new and old params with , formatted
print(f"Old params: {start_params:,}")
print(f"New params: {new_params:,}")
