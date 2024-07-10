import torch
from safetensors.torch import load_file, save_file
from collections import OrderedDict

meta = OrderedDict()
meta['format'] = "pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce_weight(weight, target_size):
    weight = weight.to(device, torch.float32)
    original_shape = weight.shape
    flattened = weight.view(-1, original_shape[-1])

    if flattened.shape[1] <= target_size:
        return weight

    U, S, V = torch.svd(flattened)
    reduced = torch.mm(U[:, :target_size], torch.diag(S[:target_size]))

    if reduced.shape[1] < target_size:
        padding = torch.zeros(reduced.shape[0], target_size - reduced.shape[1], device=device)
        reduced = torch.cat((reduced, padding), dim=1)

    return reduced.view(original_shape[:-1] + (target_size,))


def reduce_bias(bias, target_size):
    bias = bias.to(device, torch.float32)
    original_size = bias.shape[0]

    if original_size <= target_size:
        return torch.nn.functional.pad(bias, (0, target_size - original_size))
    else:
        return bias.view(-1, original_size // target_size).mean(dim=1)[:target_size]


# Load your original state dict
state_dict = load_file(
    "/home/jaret/Dev/models/hf/PixArt-Sigma-XL-2-512_MS_t5large_raw/transformer/diffusion_pytorch_model.orig.safetensors")

# Create a new state dict for the reduced model
new_state_dict = {}

source_hidden_size = 1152
target_hidden_size = 1024

for key, value in state_dict.items():
    value = value.to(device, torch.float32)
    if 'weight' in key or 'scale_shift_table' in key:
        if value.shape[0] == source_hidden_size:
            value = value[:target_hidden_size]
        elif value.shape[0] == source_hidden_size * 4:
            value = value[:target_hidden_size * 4]
        elif value.shape[0] == source_hidden_size * 6:
            value = value[:target_hidden_size * 6]

        if len(value.shape) > 1 and value.shape[
            1] == source_hidden_size and 'attn2.to_k.weight' not in key and 'attn2.to_v.weight' not in key:
            value = value[:, :target_hidden_size]
        elif len(value.shape) > 1 and value.shape[1] == source_hidden_size * 4:
            value = value[:, :target_hidden_size * 4]

    elif 'bias' in key:
        if value.shape[0] == source_hidden_size:
            value = value[:target_hidden_size]
        elif value.shape[0] == source_hidden_size * 4:
            value = value[:target_hidden_size * 4]
        elif value.shape[0] == source_hidden_size * 6:
            value = value[:target_hidden_size * 6]

    new_state_dict[key] = value

# Move all to CPU and convert to float16
for key, value in new_state_dict.items():
    new_state_dict[key] = value.cpu().to(torch.float16)

# Save the new state dict
save_file(new_state_dict,
          "/home/jaret/Dev/models/hf/PixArt-Sigma-XL-2-512_MS_t5large_raw/transformer/diffusion_pytorch_model.safetensors",
          metadata=meta)

print("Done!")
