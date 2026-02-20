import math
from functools import partial

from torch import nn
import torch


class AuraFlowPatchEmbed(nn.Module):
    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        pos_embed_max_size=None,
    ):
        super().__init__()

        self.num_patches = (height // patch_size) * (width // patch_size)
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pos_embed_max_size, embed_dim) * 0.1)

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size

    def forward(self, latent):
        batch_size, num_channels, height, width = latent.size()
        latent = latent.view(
            batch_size,
            num_channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size,
        )
        latent = latent.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        latent = self.proj(latent)
        try:
            return latent + self.pos_embed
        except RuntimeError:
            raise RuntimeError(
                f"Positional embeddings are too small for the number of patches. "
                f"Please increase `pos_embed_max_size` to at least {self.num_patches}."
            )


# comfy
#     def apply_pos_embeds(self, x, h, w):
#         h = (h + 1) // self.patch_size
#         w = (w + 1) // self.patch_size
#         max_dim = max(h, w)
#
#         cur_dim = self.h_max
#         pos_encoding = self.positional_encoding.reshape(1, cur_dim, cur_dim, -1).to(device=x.device, dtype=x.dtype)
#
#         if max_dim > cur_dim:
#             pos_encoding = F.interpolate(pos_encoding.movedim(-1, 1), (max_dim, max_dim), mode="bilinear").movedim(1,
#                                                                                                                    -1)
#             cur_dim = max_dim
#
#         from_h = (cur_dim - h) // 2
#         from_w = (cur_dim - w) // 2
#         pos_encoding = pos_encoding[:, from_h:from_h + h, from_w:from_w + w]
#         return x + pos_encoding.reshape(1, -1, self.positional_encoding.shape[-1])

    # def patchify(self, x):
    #     B, C, H, W = x.size()
    #     pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
    #     pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
    #
    #     x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    #     x = x.view(
    #         B,
    #         C,
    #         (H + 1) // self.patch_size,
    #         self.patch_size,
    #         (W + 1) // self.patch_size,
    #         self.patch_size,
    #     )
    #     x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
    #     return x

def patch_auraflow_pos_embed(pos_embed):
    # we need to hijack the forward and replace with a custom one. Self is the model
    def new_forward(self, latent):
        batch_size, num_channels, height, width = latent.size()

        # add padding to the latent to make it match pos_embed
        latent_size = height * width * num_channels / 16  # todo check where 16 comes from?
        pos_embed_size = self.pos_embed.shape[1]
        if latent_size < pos_embed_size:
            total_padding = int(pos_embed_size - math.floor(latent_size))
            total_padding = total_padding // 16
            pad_height = total_padding // 2
            pad_width = total_padding - pad_height
            # mirror padding on the right side
            padding = (0, pad_width, 0, pad_height)
            latent = torch.nn.functional.pad(latent, padding, mode='reflect')
        elif latent_size > pos_embed_size:
            amount_to_remove = latent_size - pos_embed_size
            latent = latent[:, :, :-amount_to_remove]

        batch_size, num_channels, height, width = latent.size()

        latent = latent.view(
            batch_size,
            num_channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size,
        )
        latent = latent.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        latent = self.proj(latent)
        try:
            return latent + self.pos_embed
        except RuntimeError:
            raise RuntimeError(
                f"Positional embeddings are too small for the number of patches. "
                f"Please increase `pos_embed_max_size` to at least {self.num_patches}."
            )

    pos_embed.forward = partial(new_forward, pos_embed)
