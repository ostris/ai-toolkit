import torch
from safetensors.torch import save_file, load_file
from collections import OrderedDict
meta = OrderedDict()
meta["format"] ="pt"

attn_dict = load_file("/mnt/Train/out/ip_adapter/sd15_bigG/sd15_bigG_000266000.safetensors")
state_dict = load_file("/home/jaret/Dev/models/hf/OstrisDiffusionV1/unet/diffusion_pytorch_model.safetensors")

attn_list = []
for key, value in state_dict.items():
    if "attn1" in key:
        attn_list.append(key)

attn_names = ['down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor', 'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor', 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor', 'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor', 'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor', 'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor', 'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor', 'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor', 'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor', 'mid_block.attentions.0.transformer_blocks.0.attn2.processor']

adapter_names = []
for i in range(100):
    if f'te_adapter.adapter_modules.{i}.to_k_adapter.weight' in attn_dict:
        adapter_names.append(f"te_adapter.adapter_modules.{i}.adapter")


for i in range(len(adapter_names)):
    adapter_name = adapter_names[i]
    attn_name = attn_names[i]
    adapter_k_name = adapter_name[:-8] + '.to_k_adapter.weight'
    adapter_v_name = adapter_name[:-8] + '.to_v_adapter.weight'
    state_k_name = attn_name.replace(".processor", ".to_k.weight")
    state_v_name = attn_name.replace(".processor", ".to_v.weight")
    if adapter_k_name in attn_dict:
        state_dict[state_k_name] = attn_dict[adapter_k_name]
        state_dict[state_v_name] = attn_dict[adapter_v_name]
    else:
        print("adapter_k_name", adapter_k_name)
        print("state_k_name", state_k_name)

for key, value in state_dict.items():
    state_dict[key] = value.cpu().to(torch.float16)

save_file(state_dict, "/home/jaret/Dev/models/hf/OstrisDiffusionV1/unet/diffusion_pytorch_model.safetensors", metadata=meta)

print("Done")
