import os

import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from safetensors.torch import load_file, save_file
from collections import OrderedDict
import json

model_path = "/mnt/Models/stable-diffusion/models/stable-diffusion/Ostris/objective_reality_v2.safetensors"
te_path = "google/flan-t5-xl"
te_aug_path = "/mnt/Train/out/ip_adapter/t5xx_sd15_v1/t5xx_sd15_v1_000032000.safetensors"
output_path = "/home/jaret/Dev/models/hf/t5xl_sd15_v1"

print("Loading te adapter")
te_aug_sd = load_file(te_aug_path)

print("Loading model")
sd = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16)

print("Loading Text Encoder")
# Load the text encoder
te = T5EncoderModel.from_pretrained(te_path, torch_dtype=torch.float16)

# patch it
sd.text_encoder = te
sd.tokenizer = T5Tokenizer.from_pretrained(te_path)

unet_sd = sd.unet.state_dict()

weight_idx = 1

new_cross_attn_dim = None

print("Patching UNet")
for name in sd.unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else sd.unet.config['cross_attention_dim']
    if name.startswith("mid_block"):
        hidden_size = sd.unet.config['block_out_channels'][-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(sd.unet.config['block_out_channels']))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = sd.unet.config['block_out_channels'][block_id]
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
            te_aug_name = f"te_adapter.adapter_modules.{weight_idx}.to_k_adapter"
            if f"{te_aug_name}.weight" in te_aug_sd:
                # increment so we dont redo it next time
                weight_idx += 1
                break
            else:
                weight_idx += 1

            if weight_idx > 1000:
                raise ValueError("Could not find the next weight")

        unet_sd[layer_name + ".to_k.weight"] = te_aug_sd[te_aug_name + ".weight"]
        unet_sd[layer_name + ".to_v.weight"] = te_aug_sd[te_aug_name.replace('to_k', 'to_v') + ".weight"]

        if new_cross_attn_dim is None:
            new_cross_attn_dim = unet_sd[layer_name + ".to_k.weight"].shape[1]


print("Saving unmodified model")
sd.save_pretrained(
    output_path,
    safe_serialization=True,
)

# overwrite the unet
unet_folder = os.path.join(output_path, "unet")

# move state_dict to cpu
unet_sd = {k: v.clone().cpu().to(torch.float16) for k, v in unet_sd.items()}

meta = OrderedDict()
meta["format"] = "pt"

print("Patching new unet")

save_file(unet_sd, os.path.join(unet_folder, "diffusion_pytorch_model.safetensors"), meta)

# load the json file
with open(os.path.join(unet_folder, "config.json"), 'r') as f:
    config = json.load(f)

config['cross_attention_dim'] = new_cross_attn_dim

# save it
with open(os.path.join(unet_folder, "config.json"), 'w') as f:
    json.dump(config, f, indent=2)

print("Done")
