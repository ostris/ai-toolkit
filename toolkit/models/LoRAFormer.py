import math
import weakref

import torch
import torch.nn as nn
from typing import TYPE_CHECKING, List, Dict, Any
from toolkit.models.clip_fusion import ZipperBlock
from toolkit.models.zipper_resampler import ZipperModule, ZipperResampler
import sys
from collections import OrderedDict

if TYPE_CHECKING:
    from toolkit.lora_special import LoRAModule
    from toolkit.stable_diffusion_model import StableDiffusion


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, cross_attn_input):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)

        # Cross-attention
        cross_attn_output, _ = self.cross_attn(x, cross_attn_input, cross_attn_input)
        x = self.norm2(x + cross_attn_output)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)

        return x


class InstantLoRAMidModule(torch.nn.Module):
    def __init__(
            self,
            index: int,
            lora_module: 'LoRAModule',
            instant_lora_module: 'InstantLoRAModule',
            up_shape: list = None,
            down_shape: list = None,
    ):
        super(InstantLoRAMidModule, self).__init__()
        self.up_shape = up_shape
        self.down_shape = down_shape
        self.index = index
        self.lora_module_ref = weakref.ref(lora_module)
        self.instant_lora_module_ref = weakref.ref(instant_lora_module)

        self.embed = None

    def down_forward(self, x, *args, **kwargs):
        # get the embed
        self.embed = self.instant_lora_module_ref().img_embeds[self.index]
        down_size = math.prod(self.down_shape)
        down_weight = self.embed[:, :down_size]

        batch_size = x.shape[0]

        # unconditional
        if down_weight.shape[0] * 2 == batch_size:
            down_weight = torch.cat([down_weight] * 2, dim=0)

        weight_chunks = torch.chunk(down_weight, batch_size, dim=0)
        x_chunks = torch.chunk(x, batch_size, dim=0)

        x_out = []
        for i in range(batch_size):
            weight_chunk = weight_chunks[i]
            x_chunk = x_chunks[i]
            # reshape
            weight_chunk = weight_chunk.view(self.down_shape)
            # check if is conv or linear
            if len(weight_chunk.shape) == 4:
                padding = 0
                if weight_chunk.shape[-1] == 3:
                    padding = 1
                x_chunk = nn.functional.conv2d(x_chunk, weight_chunk, padding=padding)
            else:
                # run a simple linear layer with the down weight
                x_chunk = x_chunk @ weight_chunk.T
            x_out.append(x_chunk)
        x = torch.cat(x_out, dim=0)
        return x


    def up_forward(self, x, *args, **kwargs):
        self.embed = self.instant_lora_module_ref().img_embeds[self.index]
        up_size = math.prod(self.up_shape)
        up_weight = self.embed[:, -up_size:]

        batch_size = x.shape[0]

        # unconditional
        if up_weight.shape[0] * 2 == batch_size:
            up_weight = torch.cat([up_weight] * 2, dim=0)

        weight_chunks = torch.chunk(up_weight, batch_size, dim=0)
        x_chunks = torch.chunk(x, batch_size, dim=0)

        x_out = []
        for i in range(batch_size):
            weight_chunk = weight_chunks[i]
            x_chunk = x_chunks[i]
            # reshape
            weight_chunk = weight_chunk.view(self.up_shape)
            # check if is conv or linear
            if len(weight_chunk.shape) == 4:
                padding = 0
                if weight_chunk.shape[-1] == 3:
                    padding = 1
                x_chunk = nn.functional.conv2d(x_chunk, weight_chunk, padding=padding)
            else:
                # run a simple linear layer with the down weight
                x_chunk = x_chunk @ weight_chunk.T
            x_out.append(x_chunk)
        x = torch.cat(x_out, dim=0)
        return x


# Initialize the network
# num_blocks = 8
# d_model = 1024  # Adjust as needed
# nhead = 16  # Adjust as needed
# dim_feedforward = 4096  # Adjust as needed
# latent_dim = 1695744

class LoRAFormer(torch.nn.Module):
    def __init__(
            self,
            num_blocks,
            d_model=1024,
            nhead=16,
            dim_feedforward=4096,
            sd: 'StableDiffusion'=None,
    ):
        super(LoRAFormer, self).__init__()
        # self.linear = torch.nn.Linear(2, 1)
        self.sd_ref = weakref.ref(sd)
        self.dim = sd.network.lora_dim

        # stores the projection vector. Grabbed by modules
        self.img_embeds: List[torch.Tensor] = None

        # disable merging in. It is slower on inference
        self.sd_ref().network.can_merge_in = False

        self.ilora_modules = torch.nn.ModuleList()

        lora_modules = self.sd_ref().network.get_all_modules()

        output_size = 0

        self.embed_lengths = []
        self.weight_mapping = []

        for idx, lora_module in enumerate(lora_modules):
            module_dict = lora_module.state_dict()
            down_shape = list(module_dict['lora_down.weight'].shape)
            up_shape = list(module_dict['lora_up.weight'].shape)

            self.weight_mapping.append([lora_module.lora_name, [down_shape, up_shape]])

            module_size = math.prod(down_shape) + math.prod(up_shape)
            output_size += module_size
            self.embed_lengths.append(module_size)


            # add a new mid module that will take the original forward and add a vector to it
            # this will be used to add the vector to the original forward
            instant_module = InstantLoRAMidModule(
                idx,
                lora_module,
                self,
                up_shape=up_shape,
                down_shape=down_shape
            )

            self.ilora_modules.append(instant_module)

            # replace the LoRA forwards
            lora_module.lora_down.forward = instant_module.down_forward
            lora_module.lora_up.forward = instant_module.up_forward


        self.output_size = output_size

        self.latent = nn.Parameter(torch.randn(1, output_size))
        self.latent_proj = nn.Linear(output_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_blocks)
        ])
        self.final_proj = nn.Linear(d_model, output_size)

        self.migrate_weight_mapping()

    def migrate_weight_mapping(self):
        return
        # # changes the names of the modules to common ones
        # keymap = self.sd_ref().network.get_keymap()
        # save_keymap = {}
        # if keymap is not None:
        #     for ldm_key, diffusers_key in keymap.items():
        #         #  invert them
        #         save_keymap[diffusers_key] = ldm_key
        #
        #     new_keymap = {}
        #     for key, value in self.weight_mapping:
        #         if key in save_keymap:
        #             new_keymap[save_keymap[key]] = value
        #         else:
        #             print(f"Key {key} not found in keymap")
        #             new_keymap[key] = value
        #     self.weight_mapping = new_keymap
        # else:
        #     print("No keymap found. Using default names")
        #     return


    def forward(self, img_embeds):
        # expand token rank if only rank 2
        if len(img_embeds.shape) == 2:
            img_embeds = img_embeds.unsqueeze(1)

        # resample the image embeddings
        img_embeds = self.resampler(img_embeds)
        img_embeds = self.proj_module(img_embeds)
        if len(img_embeds.shape) == 3:
            # merge the heads
            img_embeds = img_embeds.mean(dim=1)

        self.img_embeds = []
        # get all the slices
        start = 0
        for length in self.embed_lengths:
            self.img_embeds.append(img_embeds[:, start:start+length])
            start += length


    def get_additional_save_metadata(self) -> Dict[str, Any]:
        # save the weight mapping
        return {
            "weight_mapping": self.weight_mapping,
            "num_heads": self.num_heads,
            "vision_hidden_size": self.vision_hidden_size,
            "head_dim": self.head_dim,
            "vision_tokens": self.vision_tokens,
            "output_size": self.output_size,
        }

