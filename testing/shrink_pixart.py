import torch
from safetensors.torch import load_file, save_file
from collections import OrderedDict

model_path = "/home/jaret/Dev/models/hf/PixArt-Sigma-XL-2-1024_tiny/transformer/diffusion_pytorch_model_orig.safetensors"
output_path = "/home/jaret/Dev/models/hf/PixArt-Sigma-XL-2-1024_tiny/transformer/diffusion_pytorch_model.safetensors"

state_dict = load_file(model_path)

meta = OrderedDict()
meta["format"] = "pt"

new_state_dict = {}

# Move non-blocks over
for key, value in state_dict.items():
    if not key.startswith("transformer_blocks."):
        new_state_dict[key] = value

block_names = ['transformer_blocks.{idx}.attn1.to_k.bias', 'transformer_blocks.{idx}.attn1.to_k.weight',
               'transformer_blocks.{idx}.attn1.to_out.0.bias', 'transformer_blocks.{idx}.attn1.to_out.0.weight',
               'transformer_blocks.{idx}.attn1.to_q.bias', 'transformer_blocks.{idx}.attn1.to_q.weight',
               'transformer_blocks.{idx}.attn1.to_v.bias', 'transformer_blocks.{idx}.attn1.to_v.weight',
               'transformer_blocks.{idx}.attn2.to_k.bias', 'transformer_blocks.{idx}.attn2.to_k.weight',
               'transformer_blocks.{idx}.attn2.to_out.0.bias', 'transformer_blocks.{idx}.attn2.to_out.0.weight',
               'transformer_blocks.{idx}.attn2.to_q.bias', 'transformer_blocks.{idx}.attn2.to_q.weight',
               'transformer_blocks.{idx}.attn2.to_v.bias', 'transformer_blocks.{idx}.attn2.to_v.weight',
               'transformer_blocks.{idx}.ff.net.0.proj.bias', 'transformer_blocks.{idx}.ff.net.0.proj.weight',
               'transformer_blocks.{idx}.ff.net.2.bias', 'transformer_blocks.{idx}.ff.net.2.weight',
               'transformer_blocks.{idx}.scale_shift_table']

# New block idx 0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27

current_idx = 0
for i in range(28):
    if i not in [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27]:
        # todo merge in with previous block
        for name in block_names:
            try:
                new_state_dict_key = name.format(idx=current_idx - 1)
                old_state_dict_key = name.format(idx=i)
                new_state_dict[new_state_dict_key] = (new_state_dict[new_state_dict_key] * 0.5) + (state_dict[old_state_dict_key] * 0.5)
            except KeyError:
                raise KeyError(f"KeyError: {name.format(idx=current_idx)}")
    else:
        for name in block_names:
            new_state_dict[name.format(idx=current_idx)] = state_dict[name.format(idx=i)]
        current_idx += 1


# make sure they are all fp16 and on cpu
for key, value in new_state_dict.items():
    new_state_dict[key] = value.to(torch.float16).cpu()

# save the new state dict
save_file(new_state_dict, output_path, metadata=meta)

new_param_count = sum([v.numel() for v in new_state_dict.values()])
old_param_count = sum([v.numel() for v in state_dict.values()])

print(f"Old param count: {old_param_count:,}")
print(f"New param count: {new_param_count:,}")