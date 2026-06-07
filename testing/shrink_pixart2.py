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

# Blocks to keep
# keep_blocks = [0, 1, 2, 6, 10, 14, 18, 22, 26, 27]
keep_blocks = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27]


def weighted_merge(kept_block, removed_block, weight):
    return kept_block * (1 - weight) + removed_block * weight


# First, copy all kept blocks to new_state_dict
for i, old_idx in enumerate(keep_blocks):
    for name in block_names:
        old_key = name.format(idx=old_idx)
        new_key = name.format(idx=i)
        new_state_dict[new_key] = state_dict[old_key].clone()

# Then, merge information from removed blocks
for i in range(28):
    if i not in keep_blocks:
        # Find the nearest kept blocks
        prev_kept = max([b for b in keep_blocks if b < i])
        next_kept = min([b for b in keep_blocks if b > i])

        # Calculate the weight based on position
        weight = (i - prev_kept) / (next_kept - prev_kept)

        for name in block_names:
            removed_key = name.format(idx=i)
            prev_new_key = name.format(idx=keep_blocks.index(prev_kept))
            next_new_key = name.format(idx=keep_blocks.index(next_kept))

            # Weighted merge for previous kept block
            new_state_dict[prev_new_key] = weighted_merge(new_state_dict[prev_new_key], state_dict[removed_key], weight)

            # Weighted merge for next kept block
            new_state_dict[next_new_key] = weighted_merge(new_state_dict[next_new_key], state_dict[removed_key],
                                                          1 - weight)

# Convert to fp16 and move to CPU
for key, value in new_state_dict.items():
    new_state_dict[key] = value.to(torch.float16).cpu()

# Save the new state dict
save_file(new_state_dict, output_path, metadata=meta)

new_param_count = sum([v.numel() for v in new_state_dict.values()])
old_param_count = sum([v.numel() for v in state_dict.values()])

print(f"Old param count: {old_param_count:,}")
print(f"New param count: {new_param_count:,}")