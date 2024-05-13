import weakref

import torch
import torch.nn as nn
from typing import TYPE_CHECKING
from toolkit.models.clip_fusion import ZipperBlock
from toolkit.models.zipper_resampler import ZipperModule, ZipperResampler
import sys
from toolkit.paths import REPOS_ROOT
sys.path.append(REPOS_ROOT)
from ipadapter.ip_adapter.resampler import  Resampler

if TYPE_CHECKING:
    from toolkit.lora_special import LoRAModule
    from toolkit.stable_diffusion_model import StableDiffusion


class ILoRAProjModule(torch.nn.Module):
    def __init__(self, num_modules=1, dim=4, embeddings_dim=512):
        super().__init__()

        self.num_modules = num_modules
        self.num_dim = dim
        self.norm = torch.nn.LayerNorm(embeddings_dim)

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(embeddings_dim, embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(embeddings_dim * 2, num_modules * dim),
        )
        # Initialize the last linear layer weights near zero
        torch.nn.init.uniform_(self.proj[2].weight, a=-0.01, b=0.01)
        torch.nn.init.zeros_(self.proj[2].bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = x.reshape(-1, self.num_modules, self.num_dim)
        return x


class InstantLoRAMidModule(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            index: int,
            lora_module: 'LoRAModule',
            instant_lora_module: 'InstantLoRAModule'
    ):
        super(InstantLoRAMidModule, self).__init__()
        self.dim = dim
        self.index = index
        self.lora_module_ref = weakref.ref(lora_module)
        self.instant_lora_module_ref = weakref.ref(instant_lora_module)

    def forward(self, x, *args, **kwargs):
        # get the vector
        img_embeds = self.instant_lora_module_ref().img_embeds
        # project it
        scaler = img_embeds[:, self.index, :]

        # remove the channel dim (index)
        scaler = scaler.squeeze(1)

        # double up if batch is 2x the size on x (cfg)
        if x.shape[0] // 2 == scaler.shape[0]:
            scaler = torch.cat([scaler, scaler], dim=0)

        # multiply it by the scaler
        try:
            # reshape if needed
            if len(x.shape) == 3:
                scaler = scaler.unsqueeze(1)
        except Exception as e:
            print(e)
            print(x.shape)
            print(scaler.shape)
            raise e
        # apply tanh to limit values to -1 to 1
        # scaler = torch.tanh(scaler)
        return x * scaler


class InstantLoRAModule(torch.nn.Module):
    def __init__(
            self,
            vision_hidden_size: int,
            vision_tokens: int,
            sd: 'StableDiffusion'
    ):
        super(InstantLoRAModule, self).__init__()
        # self.linear = torch.nn.Linear(2, 1)
        self.sd_ref = weakref.ref(sd)
        self.dim = sd.network.lora_dim
        self.vision_hidden_size = vision_hidden_size
        self.vision_tokens = vision_tokens

        # stores the projection vector. Grabbed by modules
        self.img_embeds: torch.Tensor = None

        # disable merging in. It is slower on inference
        self.sd_ref().network.can_merge_in = False

        self.ilora_modules = torch.nn.ModuleList()

        lora_modules = self.sd_ref().network.get_all_modules()

        # resample the output so each module gets one token with a size of its dim so we can multiply by that
        # self.resampler = ZipperResampler(
        #     in_size=self.vision_hidden_size,
        #     in_tokens=self.vision_tokens,
        #     out_size=self.dim,
        #     out_tokens=len(lora_modules),
        #     hidden_size=self.vision_hidden_size,
        #     hidden_tokens=self.vision_tokens,
        #     num_blocks=1,
        # )
        # heads = 20
        # heads = 12
        # dim = 1280
        # output_dim = self.dim
        self.proj_module = ILoRAProjModule(
            num_modules=len(lora_modules),
            dim=self.dim,
            embeddings_dim=self.vision_hidden_size,
        )
        # self.resampler = Resampler(
        #         dim=dim,
        #         depth=4,
        #         dim_head=64,
        #         heads=heads,
        #         num_queries=len(lora_modules),
        #         embedding_dim=self.vision_hidden_size,
        #         max_seq_len=self.vision_tokens,
        #         output_dim=output_dim,
        #         ff_mult=4
        #     )

        for idx, lora_module in enumerate(lora_modules):
            # add a new mid module that will take the original forward and add a vector to it
            # this will be used to add the vector to the original forward
            mid_module = InstantLoRAMidModule(
                self.dim,
                idx,
                lora_module,
                self
            )

            self.ilora_modules.append(mid_module)
            # replace the LoRA lora_mid
            lora_module.lora_mid = mid_module.forward

        # add a new mid module that will take the original forward and add a vector to it
        # this will be used to add the vector to the original forward

    def forward(self, img_embeds):
        # expand token rank if only rank 2
        if len(img_embeds.shape) == 2:
            img_embeds = img_embeds.unsqueeze(1)
        img_embeds = self.proj_module(img_embeds)
        self.img_embeds = img_embeds

