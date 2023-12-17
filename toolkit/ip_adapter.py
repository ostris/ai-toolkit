import torch
import sys

from PIL import Image
from torch.nn import Parameter
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from toolkit.paths import REPOS_ROOT
from toolkit.saving import load_ip_adapter_model
from toolkit.train_tools import get_torch_dtype

sys.path.append(REPOS_ROOT)
from typing import TYPE_CHECKING, Union, Iterator, Mapping, Any, Tuple, List
from collections import OrderedDict
from ipadapter.ip_adapter.attention_processor import AttnProcessor, IPAttnProcessor
from ipadapter.ip_adapter.ip_adapter import ImageProjModel
from ipadapter.ip_adapter.resampler import Resampler
from toolkit.config_modules import AdapterConfig
from toolkit.prompt_utils import PromptEmbeds
import weakref

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from diffusers.models.attention_processor import (
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)


# loosely based on # ref https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train.py
class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, sd: 'StableDiffusion', adapter_config: 'AdapterConfig'):
        super().__init__()
        self.config = adapter_config
        self.sd_ref: weakref.ref = weakref.ref(sd)
        self.clip_image_processor = CLIPImageProcessor()
        self.device = self.sd_ref().unet.device
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(adapter_config.image_encoder_path)
        if adapter_config.type == 'ip':
            # ip-adapter
            image_proj_model = ImageProjModel(
                cross_attention_dim=sd.unet.config['cross_attention_dim'],
                clip_embeddings_dim=self.image_encoder.config.projection_dim,
                clip_extra_context_tokens=4,
            )
        elif adapter_config.type == 'ip+':
            # ip-adapter-plus
            num_tokens = 16
            image_proj_model = Resampler(
                dim=sd.unet.config['cross_attention_dim'],
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=num_tokens,
                embedding_dim=self.image_encoder.config.hidden_size,
                output_dim=sd.unet.config['cross_attention_dim'],
                ff_mult=4
            )
        else:
            raise ValueError(f"unknown adapter type: {adapter_config.type}")

        # init adapter modules
        attn_procs = {}
        unet_sd = sd.unet.state_dict()
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
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
        sd.unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(sd.unet.attn_processors.values())

        sd.adapter = self
        self.unet_ref: weakref.ref = weakref.ref(sd.unet)
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        # load the weights if we have some
        if self.config.name_or_path:
            loaded_state_dict = load_ip_adapter_model(
                self.config.name_or_path,
                device='cpu',
                dtype=sd.torch_dtype
            )
            self.load_state_dict(loaded_state_dict)

        self.set_scale(1.0)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.image_encoder.to(*args, **kwargs)
        self.image_proj_model.to(*args, **kwargs)
        self.adapter_modules.to(*args, **kwargs)
        return self

    def load_ip_adapter(self, state_dict: Union[OrderedDict, dict]):
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    # def load_state_dict(self, state_dict: Union[OrderedDict, dict]):
    #     self.load_ip_adapter(state_dict)

    def state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        state_dict["image_proj"] = self.image_proj_model.state_dict()
        state_dict["ip_adapter"] = self.adapter_modules.state_dict()
        return state_dict

    def set_scale(self, scale):
        for attn_processor in self.sd_ref().unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    @torch.no_grad()
    def get_clip_image_embeds_from_pil(self, pil_image: Union[Image.Image, List[Image.Image]], drop=False) -> torch.Tensor:
        # todo: add support for sdxl
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        if drop:
            clip_image = clip_image * 0
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        return clip_image_embeds

    @torch.no_grad()
    def get_clip_image_embeds_from_tensors(self, tensors_0_1: torch.Tensor, drop=False) -> torch.Tensor:
        # tensors should be 0-1
        # todo: add support for sdxl
        if tensors_0_1.ndim == 3:
            tensors_0_1 = tensors_0_1.unsqueeze(0)
        # training tensors are 0 - 1
        tensors_0_1 = tensors_0_1.to(self.device, dtype=torch.float16)
        # if images are out of this range throw error
        if tensors_0_1.min() < -0.3 or tensors_0_1.max() > 1.3:
            raise ValueError("image tensor values must be between 0 and 1. Got min: {}, max: {}".format(
                tensors_0_1.min(), tensors_0_1.max()
            ))

        clip_image = self.clip_image_processor(
            images=tensors_0_1,
            return_tensors="pt",
            do_resize=True,
            do_rescale=False,
        ).pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16).detach()
        if drop:
            clip_image = clip_image * 0
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        return clip_image_embeds

    # use drop for prompt dropout, or negatives
    def forward(self, embeddings: PromptEmbeds, clip_image_embeds: torch.Tensor) -> PromptEmbeds:
        clip_image_embeds = clip_image_embeds.detach()
        clip_image_embeds = clip_image_embeds.to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        image_prompt_embeds = self.image_proj_model(clip_image_embeds.detach())
        embeddings.text_embeds = torch.cat([embeddings.text_embeds, image_prompt_embeds], dim=1)
        return embeddings

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for attn_processor in self.adapter_modules:
            yield from attn_processor.parameters(recurse)
        yield from self.image_proj_model.parameters(recurse)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=strict)

