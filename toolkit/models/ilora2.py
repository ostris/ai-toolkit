import math
import weakref

from toolkit.config_modules import AdapterConfig
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, List, Dict, Any
from toolkit.resampler import Resampler

if TYPE_CHECKING:
    from toolkit.lora_special import LoRAModule
    from toolkit.stable_diffusion_model import StableDiffusion


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.1, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual
        return x


class LoRAGenerator(torch.nn.Module):
    def __init__(
            self,
            input_size: int = 768,  # projection dimension
            hidden_size: int = 768,
            head_size: int = 512,
            num_heads: int = 1,
            num_mlp_layers: int = 1,
            output_size: int = 768,
            dropout: float = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.simple = False

        self.output_size = output_size

        if self.simple:
            self.head = nn.Linear(input_size, head_size, bias=False)
        else:
            self.lin_in = nn.Linear(input_size, hidden_size)

            self.mlp_blocks = nn.Sequential(*[
                MLP(hidden_size, hidden_size, hidden_size, dropout=dropout, use_residual=True) for _ in
                range(num_mlp_layers)
            ])
            self.head = nn.Linear(hidden_size, head_size, bias=False)
        self.norm = nn.LayerNorm(head_size)

        if num_heads == 1:
            self.output = nn.Linear(head_size, self.output_size)
            # for each output block. multiply weights by 0.01
            with torch.no_grad():
                self.output.weight.data *= 0.01
        else:
            head_output_size = output_size // num_heads
            self.outputs = nn.ModuleList([nn.Linear(head_size, head_output_size) for _ in range(num_heads)])
            # for each output block. multiply weights by 0.01
            with torch.no_grad():
                for output in self.outputs:
                    output.weight.data *= 0.01

    # allow get device
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, embedding):
        if len(embedding.shape) == 2:
            embedding = embedding.unsqueeze(1)

        x = embedding

        if not self.simple:
            x = self.lin_in(embedding)
            x = self.mlp_blocks(x)
        x = self.head(x)
        x = self.norm(x)

        if self.num_heads == 1:
            x = self.output(x)
        else:
            out_chunks = torch.chunk(x, self.num_heads, dim=1)
            x = []
            for out_layer, chunk in zip(self.outputs, out_chunks):
                x.append(out_layer(chunk))
            x = torch.cat(x, dim=-1)

        return x.squeeze(1)


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

        self.do_up = instant_lora_module.config.ilora_up
        self.do_down = instant_lora_module.config.ilora_down
        self.do_mid = instant_lora_module.config.ilora_mid

        self.down_dim = self.down_shape[1] if self.do_down else 0
        self.mid_dim = self.up_shape[1] if self.do_mid else 0
        self.out_dim = self.up_shape[0] if self.do_up else 0

        self.embed = None

    def down_forward(self, x, *args, **kwargs):
        if not self.do_down:
            return self.lora_module_ref().lora_down.orig_forward(x, *args, **kwargs)
        # get the embed
        self.embed = self.instant_lora_module_ref().img_embeds[self.index]
        down_weight = self.embed[:, :self.down_dim]

        batch_size = x.shape[0]

        # unconditional
        if down_weight.shape[0] * 2 == batch_size:
            down_weight = torch.cat([down_weight] * 2, dim=0)

        try:
            if len(x.shape) == 4:
                # conv
                down_weight = down_weight.view(batch_size, -1, 1, 1)
                if x.shape[1] != down_weight.shape[1]:
                    raise ValueError(f"Down weight shape not understood: {down_weight.shape} {x.shape}")
            elif len(x.shape) == 2:
                down_weight = down_weight.view(batch_size, -1)
                if x.shape[1] != down_weight.shape[1]:
                    raise ValueError(f"Down weight shape not understood: {down_weight.shape} {x.shape}")
            else:
                down_weight = down_weight.view(batch_size, 1, -1)
                if x.shape[2] != down_weight.shape[2]:
                    raise ValueError(f"Down weight shape not understood: {down_weight.shape} {x.shape}")
            x = x * down_weight
            x = self.lora_module_ref().lora_down.orig_forward(x, *args, **kwargs)
        except Exception as e:
            print(e)
            raise ValueError(f"Down weight shape not understood: {down_weight.shape} {x.shape}")

        return x

    def up_forward(self, x, *args, **kwargs):
        # do mid here
        x = self.mid_forward(x, *args, **kwargs)
        if not self.do_up:
            return self.lora_module_ref().lora_up.orig_forward(x, *args, **kwargs)
        # get the embed
        self.embed = self.instant_lora_module_ref().img_embeds[self.index]
        up_weight = self.embed[:, -self.out_dim:]

        batch_size = x.shape[0]

        # unconditional
        if up_weight.shape[0] * 2 == batch_size:
            up_weight = torch.cat([up_weight] * 2, dim=0)

        try:
            if len(x.shape) == 4:
                # conv
                up_weight = up_weight.view(batch_size, -1, 1, 1)
            elif len(x.shape) == 2:
                up_weight = up_weight.view(batch_size, -1)
            else:
                up_weight = up_weight.view(batch_size, 1, -1)
            x = self.lora_module_ref().lora_up.orig_forward(x, *args, **kwargs)
            x = x * up_weight
        except Exception as e:
            print(e)
            raise ValueError(f"Up weight shape not understood: {up_weight.shape} {x.shape}")

        return x

    def mid_forward(self, x, *args, **kwargs):
        if not self.do_mid:
            return self.lora_module_ref().lora_down.orig_forward(x, *args, **kwargs)
        batch_size = x.shape[0]
        # get the embed
        self.embed = self.instant_lora_module_ref().img_embeds[self.index]
        mid_weight = self.embed[:, self.down_dim:self.down_dim + self.mid_dim * self.mid_dim]

        # unconditional
        if mid_weight.shape[0] * 2 == batch_size:
            mid_weight = torch.cat([mid_weight] * 2, dim=0)

        weight_chunks = torch.chunk(mid_weight, batch_size, dim=0)
        x_chunks = torch.chunk(x, batch_size, dim=0)

        x_out = []
        for i in range(batch_size):
            weight_chunk = weight_chunks[i]
            x_chunk = x_chunks[i]
            # reshape
            if len(x_chunk.shape) == 4:
                # conv
                weight_chunk = weight_chunk.view(self.mid_dim, self.mid_dim, 1, 1)
            else:
                weight_chunk = weight_chunk.view(self.mid_dim, self.mid_dim)
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


class InstantLoRAModule(torch.nn.Module):
    def __init__(
            self,
            vision_hidden_size: int,
            vision_tokens: int,
            head_dim: int,
            num_heads: int,  # number of heads in the resampler
            sd: 'StableDiffusion',
            config: AdapterConfig
    ):
        super(InstantLoRAModule, self).__init__()
        # self.linear = torch.nn.Linear(2, 1)
        self.sd_ref = weakref.ref(sd)
        self.dim = sd.network.lora_dim
        self.vision_hidden_size = vision_hidden_size
        self.vision_tokens = vision_tokens
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.config: AdapterConfig = config

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

            #
            # module_size = math.prod(down_shape) + math.prod(up_shape)

            # conv weight shape is (out_channels, in_channels, kernel_size, kernel_size)
            # linear weight shape is (out_features, in_features)

            # just doing in dim and out dim
            in_dim = down_shape[1] if self.config.ilora_down else 0
            mid_dim = down_shape[0] * down_shape[0] if self.config.ilora_mid else 0
            out_dim = up_shape[0] if self.config.ilora_up else 0
            module_size = in_dim + mid_dim + out_dim

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
            lora_module.lora_down.orig_forward = lora_module.lora_down.forward
            lora_module.lora_down.forward = instant_module.down_forward
            lora_module.lora_up.orig_forward = lora_module.lora_up.forward
            lora_module.lora_up.forward = instant_module.up_forward

        self.output_size = output_size

        number_formatted_output_size = "{:,}".format(output_size)

        print(f" ILORA output size: {number_formatted_output_size}")

        # if not evenly divisible, error
        if self.output_size % self.num_heads != 0:
            raise ValueError("Output size must be divisible by the number of heads")

        self.head_output_size = self.output_size // self.num_heads

        if vision_tokens > 1:
            self.resampler = Resampler(
                dim=vision_hidden_size,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=num_heads,  # output tokens
                embedding_dim=vision_hidden_size,
                max_seq_len=vision_tokens,
                output_dim=head_dim,
                apply_pos_emb=True,  # this is new
                ff_mult=4
            )

        self.proj_module = LoRAGenerator(
            input_size=head_dim,
            hidden_size=head_dim,
            head_size=head_dim,
            num_mlp_layers=1,
            num_heads=self.num_heads,
            output_size=self.output_size,
        )

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
            self.img_embeds.append(img_embeds[:, start:start + length])
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
            "do_up": self.config.ilora_up,
            "do_mid": self.config.ilora_mid,
            "do_down": self.config.ilora_down,
        }
