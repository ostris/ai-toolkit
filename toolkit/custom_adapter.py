import math
import torch
import sys

from PIL import Image
from torch.nn import Parameter
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, T5EncoderModel, CLIPTextModel, \
    CLIPTokenizer, T5Tokenizer

from toolkit.models.clip_fusion import CLIPFusionModule
from toolkit.models.clip_pre_processor import CLIPImagePreProcessor
from toolkit.models.ilora import InstantLoRAModule
from toolkit.models.single_value_adapter import SingleValueAdapter
from toolkit.models.te_adapter import TEAdapter
from toolkit.models.te_aug_adapter import TEAugAdapter
from toolkit.models.vd_adapter import VisionDirectAdapter
from toolkit.paths import REPOS_ROOT
from toolkit.photomaker import PhotoMakerIDEncoder, FuseModule, PhotoMakerCLIPEncoder
from toolkit.saving import load_ip_adapter_model, load_custom_adapter_model
from toolkit.train_tools import get_torch_dtype
from toolkit.models.pixtral_vision import PixtralVisionEncoderCompatible, PixtralVisionImagePreprocessorCompatible
import random

sys.path.append(REPOS_ROOT)
from typing import TYPE_CHECKING, Union, Iterator, Mapping, Any, Tuple, List, Optional, Dict
from collections import OrderedDict
from ipadapter.ip_adapter.attention_processor import AttnProcessor, IPAttnProcessor, IPAttnProcessor2_0, \
    AttnProcessor2_0
from ipadapter.ip_adapter.ip_adapter import ImageProjModel
from ipadapter.ip_adapter.resampler import Resampler
from toolkit.config_modules import AdapterConfig, AdapterTypes
from toolkit.prompt_utils import PromptEmbeds
import weakref

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPVisionModel,
    AutoImageProcessor,
    ConvNextModel,
    ConvNextForImageClassification,
    ConvNextImageProcessor,
    UMT5EncoderModel, LlamaTokenizerFast
)
from toolkit.models.size_agnostic_feature_encoder import SAFEImageProcessor, SAFEVisionModel

from transformers import ViTHybridImageProcessor, ViTHybridForImageClassification

from transformers import ViTFeatureExtractor, ViTForImageClassification

import torch.nn.functional as F


class CustomAdapter(torch.nn.Module):
    def __init__(self, sd: 'StableDiffusion', adapter_config: 'AdapterConfig'):
        super().__init__()
        self.config = adapter_config
        self.sd_ref: weakref.ref = weakref.ref(sd)
        self.device = self.sd_ref().unet.device
        self.image_processor: CLIPImageProcessor = None
        self.input_size = 224
        self.adapter_type: AdapterTypes = self.config.type
        self.current_scale = 1.0
        self.is_active = True
        self.flag_word = "fla9wor0"
        self.is_unconditional_run = False

        self.vision_encoder: Union[PhotoMakerCLIPEncoder, CLIPVisionModelWithProjection] = None

        self.fuse_module: FuseModule = None

        self.lora: None = None

        self.position_ids: Optional[List[int]] = None

        self.num_control_images = 1
        self.token_mask: Optional[torch.Tensor] = None

        # setup clip
        self.setup_clip()
        # add for dataloader
        self.clip_image_processor = self.image_processor

        self.clip_fusion_module: CLIPFusionModule = None
        self.ilora_module: InstantLoRAModule = None

        self.te: Union[T5EncoderModel, CLIPTextModel] = None
        self.tokenizer: CLIPTokenizer = None
        self.te_adapter: TEAdapter = None
        self.te_augmenter: TEAugAdapter = None
        self.vd_adapter: VisionDirectAdapter = None
        self.single_value_adapter: SingleValueAdapter = None
        self.conditional_embeds: Optional[torch.Tensor] = None
        self.unconditional_embeds: Optional[torch.Tensor] = None

        self.setup_adapter()

        if self.adapter_type == 'photo_maker':
            # try to load from our name_or_path
            if self.config.name_or_path is not None and self.config.name_or_path.endswith('.bin'):
                self.load_state_dict(torch.load(self.config.name_or_path, map_location=self.device), strict=False)
            # add the trigger word to the tokenizer
            if isinstance(self.sd_ref().tokenizer, list):
                for tokenizer in self.sd_ref().tokenizer:
                    tokenizer.add_tokens([self.flag_word], special_tokens=True)
            else:
                self.sd_ref().tokenizer.add_tokens([self.flag_word], special_tokens=True)
        elif self.config.name_or_path is not None:
            loaded_state_dict = load_custom_adapter_model(
                self.config.name_or_path,
                self.sd_ref().device,
                dtype=self.sd_ref().dtype,
            )
            self.load_state_dict(loaded_state_dict, strict=False)

    def setup_adapter(self):
        torch_dtype = get_torch_dtype(self.sd_ref().dtype)
        if self.adapter_type == 'photo_maker':
            sd = self.sd_ref()
            embed_dim = sd.unet.config['cross_attention_dim']
            self.fuse_module = FuseModule(embed_dim)
        elif self.adapter_type == 'clip_fusion':
            sd = self.sd_ref()
            embed_dim = sd.unet.config['cross_attention_dim']

            vision_tokens = ((self.vision_encoder.config.image_size // self.vision_encoder.config.patch_size) ** 2)
            if self.config.image_encoder_arch == 'clip':
                vision_tokens = vision_tokens + 1
            self.clip_fusion_module = CLIPFusionModule(
                text_hidden_size=embed_dim,
                text_tokens=77,
                vision_hidden_size=self.vision_encoder.config.hidden_size,
                vision_tokens=vision_tokens
            )
        elif self.adapter_type == 'ilora':
            vision_tokens = ((self.vision_encoder.config.image_size // self.vision_encoder.config.patch_size) ** 2)
            if self.config.image_encoder_arch == 'clip':
                vision_tokens = vision_tokens + 1

            vision_hidden_size = self.vision_encoder.config.hidden_size

            if self.config.clip_layer == 'image_embeds':
                vision_tokens = 1
                vision_hidden_size = self.vision_encoder.config.projection_dim

            self.ilora_module = InstantLoRAModule(
                vision_tokens=vision_tokens,
                vision_hidden_size=vision_hidden_size,
                head_dim=self.config.head_dim,
                num_heads=self.config.num_heads,
                sd=self.sd_ref(),
                config=self.config
            )
        elif self.adapter_type == 'text_encoder':
            if self.config.text_encoder_arch == 't5':
                te_kwargs = {}
                # te_kwargs['load_in_4bit'] = True
                # te_kwargs['load_in_8bit'] = True
                te_kwargs['device_map'] = "auto"
                te_is_quantized = True

                self.te = T5EncoderModel.from_pretrained(
                    self.config.text_encoder_path,
                    torch_dtype=torch_dtype,
                    **te_kwargs
                )

                # self.te.to = lambda *args, **kwargs: None
                self.tokenizer = T5Tokenizer.from_pretrained(self.config.text_encoder_path)
            elif self.config.text_encoder_arch == 'pile-t5':
                te_kwargs = {}
                # te_kwargs['load_in_4bit'] = True
                # te_kwargs['load_in_8bit'] = True
                te_kwargs['device_map'] = "auto"
                te_is_quantized = True

                self.te = UMT5EncoderModel.from_pretrained(
                    self.config.text_encoder_path,
                    torch_dtype=torch_dtype,
                    **te_kwargs
                )

                # self.te.to = lambda *args, **kwargs: None
                self.tokenizer = LlamaTokenizerFast.from_pretrained(self.config.text_encoder_path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            elif self.config.text_encoder_arch == 'clip':
                self.te = CLIPTextModel.from_pretrained(self.config.text_encoder_path).to(self.sd_ref().unet.device,
                                                                                          dtype=torch_dtype)
                self.tokenizer = CLIPTokenizer.from_pretrained(self.config.text_encoder_path)
            else:
                raise ValueError(f"unknown text encoder arch: {self.config.text_encoder_arch}")

            self.te_adapter = TEAdapter(self, self.sd_ref(), self.te, self.tokenizer)
        elif self.adapter_type == 'te_augmenter':
            self.te_augmenter = TEAugAdapter(self, self.sd_ref())
        elif self.adapter_type == 'vision_direct':
            self.vd_adapter = VisionDirectAdapter(self, self.sd_ref(), self.vision_encoder)
        elif self.adapter_type == 'single_value':
            self.single_value_adapter = SingleValueAdapter(self, self.sd_ref(), num_values=self.config.num_tokens)
        else:
            raise ValueError(f"unknown adapter type: {self.adapter_type}")

    def forward(self, *args, **kwargs):
        # dont think this is used
        # if self.adapter_type == 'photo_maker':
        #     id_pixel_values = args[0]
        #     prompt_embeds: PromptEmbeds = args[1]
        #     class_tokens_mask = args[2]
        #
        #     grads_on_image_encoder = self.config.train_image_encoder and torch.is_grad_enabled()
        #
        #     with torch.set_grad_enabled(grads_on_image_encoder):
        #         id_embeds = self.vision_encoder(self, id_pixel_values, do_projection2=False)
        #
        #     if not grads_on_image_encoder:
        #         id_embeds = id_embeds.detach()
        #
        #     prompt_embeds = prompt_embeds.detach()
        #
        #     updated_prompt_embeds = self.fuse_module(
        #         prompt_embeds, id_embeds, class_tokens_mask
        #     )
        #
        #     return updated_prompt_embeds
        # else:
        raise NotImplementedError

    def setup_clip(self):
        adapter_config = self.config
        sd = self.sd_ref()
        if self.config.type == "text_encoder" or self.config.type == "single_value":
            return
        if self.config.type == 'photo_maker':
            try:
                self.image_processor = CLIPImageProcessor.from_pretrained(self.config.image_encoder_path)
            except EnvironmentError:
                self.image_processor = CLIPImageProcessor()
            if self.config.image_encoder_path is None:
                self.vision_encoder = PhotoMakerCLIPEncoder()
            else:
                self.vision_encoder = PhotoMakerCLIPEncoder.from_pretrained(self.config.image_encoder_path)
        elif self.config.image_encoder_arch == 'clip' or self.config.image_encoder_arch == 'clip+':
            try:
                self.image_processor = CLIPImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                self.image_processor = CLIPImageProcessor()
            self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
                adapter_config.image_encoder_path,
                ignore_mismatched_sizes=True).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'siglip':
            from transformers import SiglipImageProcessor, SiglipVisionModel
            try:
                self.image_processor = SiglipImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                self.image_processor = SiglipImageProcessor()
            self.vision_encoder = SiglipVisionModel.from_pretrained(
                adapter_config.image_encoder_path,
                ignore_mismatched_sizes=True).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'pixtral':
            self.image_processor = PixtralVisionImagePreprocessorCompatible(
                max_image_size=self.config.pixtral_max_image_size,
            )
            self.vision_encoder = PixtralVisionEncoderCompatible.from_pretrained(
                adapter_config.image_encoder_path,
            ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'vit':
            try:
                self.image_processor = ViTFeatureExtractor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                self.image_processor = ViTFeatureExtractor()
            self.vision_encoder = ViTForImageClassification.from_pretrained(adapter_config.image_encoder_path).to(
                self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'safe':
            try:
                self.image_processor = SAFEImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                self.image_processor = SAFEImageProcessor()
            self.vision_encoder = SAFEVisionModel(
                in_channels=3,
                num_tokens=self.config.safe_tokens,
                num_vectors=sd.unet.config['cross_attention_dim'],
                reducer_channels=self.config.safe_reducer_channels,
                channels=self.config.safe_channels,
                downscale_factor=8
            ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'convnext':
            try:
                self.image_processor = ConvNextImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                print(f"could not load image processor from {adapter_config.image_encoder_path}")
                self.image_processor = ConvNextImageProcessor(
                    size=320,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711],
                )
            self.vision_encoder = ConvNextForImageClassification.from_pretrained(
                adapter_config.image_encoder_path,
                use_safetensors=True,
            ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'vit-hybrid':
            try:
                self.image_processor = ViTHybridImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                print(f"could not load image processor from {adapter_config.image_encoder_path}")
                self.image_processor = ViTHybridImageProcessor(
                    size=320,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711],
                )
            self.vision_encoder = ViTHybridForImageClassification.from_pretrained(
                adapter_config.image_encoder_path,
                use_safetensors=True,
            ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        else:
            raise ValueError(f"unknown image encoder arch: {adapter_config.image_encoder_arch}")

        self.input_size = self.vision_encoder.config.image_size

        if self.config.quad_image:  # 4x4 image
            # self.clip_image_processor.config
            # We do a 3x downscale of the image, so we need to adjust the input size
            preprocessor_input_size = self.vision_encoder.config.image_size * 2

            # update the preprocessor so images come in at the right size
            if 'height' in self.image_processor.size:
                self.image_processor.size['height'] = preprocessor_input_size
                self.image_processor.size['width'] = preprocessor_input_size
            elif hasattr(self.image_processor, 'crop_size'):
                self.image_processor.size['shortest_edge'] = preprocessor_input_size
                self.image_processor.crop_size['height'] = preprocessor_input_size
                self.image_processor.crop_size['width'] = preprocessor_input_size

        if self.config.image_encoder_arch == 'clip+':
            # self.image_processor.config
            # We do a 3x downscale of the image, so we need to adjust the input size
            preprocessor_input_size = self.vision_encoder.config.image_size * 4

            # update the preprocessor so images come in at the right size
            self.image_processor.size['shortest_edge'] = preprocessor_input_size
            self.image_processor.crop_size['height'] = preprocessor_input_size
            self.image_processor.crop_size['width'] = preprocessor_input_size

            self.preprocessor = CLIPImagePreProcessor(
                input_size=preprocessor_input_size,
                clip_input_size=self.vision_encoder.config.image_size,
            )
        if 'height' in self.image_processor.size:
            self.input_size = self.image_processor.size['height']
        else:
            self.input_size = self.image_processor.crop_size['height']

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        strict = False
        if self.config.train_only_image_encoder and 'vd_adapter' not in state_dict and 'dvadapter' not in state_dict:
            # we are loading pure clip weights.
            self.vision_encoder.load_state_dict(state_dict, strict=strict)

        if 'lora_weights' in state_dict:
            # todo add LoRA
            # self.sd_ref().pipeline.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")
            # self.sd_ref().pipeline.fuse_lora()
            pass
        if 'clip_fusion' in state_dict:
            self.clip_fusion_module.load_state_dict(state_dict['clip_fusion'], strict=strict)
        if 'id_encoder' in state_dict and (self.adapter_type == 'photo_maker' or self.adapter_type == 'clip_fusion'):
            self.vision_encoder.load_state_dict(state_dict['id_encoder'], strict=strict)
            # check to see if the fuse weights are there
            fuse_weights = {}
            for k, v in state_dict['id_encoder'].items():
                if k.startswith('fuse_module'):
                    k = k.replace('fuse_module.', '')
                    fuse_weights[k] = v
            if len(fuse_weights) > 0:
                try:
                    self.fuse_module.load_state_dict(fuse_weights, strict=strict)
                except Exception as e:

                    print(e)
                    # force load it
                    print(f"force loading fuse module as it did not match")
                    current_state_dict = self.fuse_module.state_dict()
                    for k, v in fuse_weights.items():
                        if len(v.shape) == 1:
                            current_state_dict[k] = v[:current_state_dict[k].shape[0]]
                        elif len(v.shape) == 2:
                            current_state_dict[k] = v[:current_state_dict[k].shape[0], :current_state_dict[k].shape[1]]
                        elif len(v.shape) == 3:
                            current_state_dict[k] = v[:current_state_dict[k].shape[0], :current_state_dict[k].shape[1],
                                                    :current_state_dict[k].shape[2]]
                        elif len(v.shape) == 4:
                            current_state_dict[k] = v[:current_state_dict[k].shape[0], :current_state_dict[k].shape[1],
                                                    :current_state_dict[k].shape[2], :current_state_dict[k].shape[3]]
                        else:
                            raise ValueError(f"unknown shape: {v.shape}")
                    self.fuse_module.load_state_dict(current_state_dict, strict=strict)

        if 'te_adapter' in state_dict:
            self.te_adapter.load_state_dict(state_dict['te_adapter'], strict=strict)

        if 'te_augmenter' in state_dict:
            self.te_augmenter.load_state_dict(state_dict['te_augmenter'], strict=strict)

        if 'vd_adapter' in state_dict:
            self.vd_adapter.load_state_dict(state_dict['vd_adapter'], strict=strict)
        if 'dvadapter' in state_dict:
            self.vd_adapter.load_state_dict(state_dict['dvadapter'], strict=False)

        if 'sv_adapter' in state_dict:
            self.single_value_adapter.load_state_dict(state_dict['sv_adapter'], strict=strict)

        if 'vision_encoder' in state_dict:
            self.vision_encoder.load_state_dict(state_dict['vision_encoder'], strict=strict)

        if 'fuse_module' in state_dict:
            self.fuse_module.load_state_dict(state_dict['fuse_module'], strict=strict)

        if 'ilora' in state_dict:
            try:
                self.ilora_module.load_state_dict(state_dict['ilora'], strict=strict)
            except Exception as e:
                print(e)

        pass

    def state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        if self.config.train_only_image_encoder:
            return self.vision_encoder.state_dict()

        if self.adapter_type == 'photo_maker':
            if self.config.train_image_encoder:
                state_dict["id_encoder"] = self.vision_encoder.state_dict()

            state_dict["fuse_module"] = self.fuse_module.state_dict()

            # todo save LoRA
            return state_dict

        elif self.adapter_type == 'clip_fusion':
            if self.config.train_image_encoder:
                state_dict["vision_encoder"] = self.vision_encoder.state_dict()
            state_dict["clip_fusion"] = self.clip_fusion_module.state_dict()
            return state_dict
        elif self.adapter_type == 'text_encoder':
            state_dict["te_adapter"] = self.te_adapter.state_dict()
            return state_dict
        elif self.adapter_type == 'te_augmenter':
            if self.config.train_image_encoder:
                state_dict["vision_encoder"] = self.vision_encoder.state_dict()
            state_dict["te_augmenter"] = self.te_augmenter.state_dict()
            return state_dict
        elif self.adapter_type == 'vision_direct':
            state_dict["dvadapter"] = self.vd_adapter.state_dict()
            # if self.config.train_image_encoder: # always return vision encoder
            state_dict["vision_encoder"] = self.vision_encoder.state_dict()
            return state_dict
        elif self.adapter_type == 'single_value':
            state_dict["sv_adapter"] = self.single_value_adapter.state_dict()
            return state_dict
        elif self.adapter_type == 'ilora':
            if self.config.train_image_encoder:
                state_dict["vision_encoder"] = self.vision_encoder.state_dict()
            state_dict["ilora"] = self.ilora_module.state_dict()
            return state_dict
        else:
            raise NotImplementedError

    def add_extra_values(self, extra_values: torch.Tensor, is_unconditional=False):
        if self.adapter_type == 'single_value':
            if is_unconditional:
                self.unconditional_embeds = extra_values.to(self.device, get_torch_dtype(self.sd_ref().dtype))
            else:
                self.conditional_embeds = extra_values.to(self.device, get_torch_dtype(self.sd_ref().dtype))


    def condition_prompt(
            self,
            prompt: Union[List[str], str],
            is_unconditional: bool = False,
    ):
        if self.adapter_type == 'clip_fusion' or self.adapter_type == 'ilora' or self.adapter_type == 'vision_direct':
            return prompt
        elif self.adapter_type == 'text_encoder':
            # todo allow for training
            with torch.no_grad():
                # encode and save the embeds
                if is_unconditional:
                    self.unconditional_embeds = self.te_adapter.encode_text(prompt).detach()
                else:
                    self.conditional_embeds = self.te_adapter.encode_text(prompt).detach()
            return prompt
        elif self.adapter_type == 'photo_maker':
            if is_unconditional:
                return prompt
            else:

                with torch.no_grad():
                    was_list = isinstance(prompt, list)
                    if not was_list:
                        prompt_list = [prompt]
                    else:
                        prompt_list = prompt

                    new_prompt_list = []
                    token_mask_list = []

                    for prompt in prompt_list:

                        our_class = None
                        # find a class in the prompt
                        prompt_parts = prompt.split(' ')
                        prompt_parts = [p.strip().lower() for p in prompt_parts if len(p) > 0]

                        new_prompt_parts = []
                        tokened_prompt_parts = []
                        for idx, prompt_part in enumerate(prompt_parts):
                            new_prompt_parts.append(prompt_part)
                            tokened_prompt_parts.append(prompt_part)
                            if prompt_part in self.config.class_names:
                                our_class = prompt_part
                                # add the flag word
                                tokened_prompt_parts.append(self.flag_word)

                                if self.num_control_images > 1:
                                    # add the rest
                                    for _ in range(self.num_control_images - 1):
                                        new_prompt_parts.extend(prompt_parts[idx + 1:])

                                # add the rest
                                tokened_prompt_parts.extend(prompt_parts[idx + 1:])
                                new_prompt_parts.extend(prompt_parts[idx + 1:])

                                break

                        prompt = " ".join(new_prompt_parts)
                        tokened_prompt = " ".join(tokened_prompt_parts)

                        if our_class is None:
                            # add the first one to the front of the prompt
                            tokened_prompt = self.config.class_names[0] + ' ' + self.flag_word + ' ' + prompt
                            our_class = self.config.class_names[0]
                            prompt = " ".join(
                                [self.config.class_names[0] for _ in range(self.num_control_images)]) + ' ' + prompt

                        # add the prompt to the list
                        new_prompt_list.append(prompt)

                        # tokenize them with just the first tokenizer
                        tokenizer = self.sd_ref().tokenizer
                        if isinstance(tokenizer, list):
                            tokenizer = tokenizer[0]

                        flag_token = tokenizer.convert_tokens_to_ids(self.flag_word)

                        tokenized_prompt = tokenizer.encode(prompt)
                        tokenized_tokened_prompt = tokenizer.encode(tokened_prompt)

                        flag_idx = tokenized_tokened_prompt.index(flag_token)

                        class_token = tokenized_prompt[flag_idx - 1]

                        boolean_mask = torch.zeros(flag_idx - 1, dtype=torch.bool)
                        boolean_mask = torch.cat((boolean_mask, torch.ones(self.num_control_images, dtype=torch.bool)))
                        boolean_mask = boolean_mask.to(self.device)
                        # zero pad it to 77
                        boolean_mask = F.pad(boolean_mask, (0, 77 - boolean_mask.shape[0]), value=False)

                        token_mask_list.append(boolean_mask)

                    self.token_mask = torch.cat(token_mask_list, dim=0).to(self.device)

                    prompt_list = new_prompt_list

                    if not was_list:
                        prompt = prompt_list[0]
                    else:
                        prompt = prompt_list

                    return prompt

        else:
            return prompt

    def condition_encoded_embeds(
            self,
            tensors_0_1: torch.Tensor,
            prompt_embeds: PromptEmbeds,
            is_training=False,
            has_been_preprocessed=False,
            is_unconditional=False,
            quad_count=4,
            is_generating_samples=False,
    ) -> PromptEmbeds:
        if self.adapter_type == 'text_encoder' and is_generating_samples:
            # replace the prompt embed with ours
            if is_unconditional:
                return self.unconditional_embeds.clone()
            return self.conditional_embeds.clone()

        if self.adapter_type == 'ilora':
            return prompt_embeds

        if self.adapter_type == 'photo_maker' or self.adapter_type == 'clip_fusion':
            if is_unconditional:
                # we dont condition the negative embeds for photo maker
                return prompt_embeds.clone()
            with torch.no_grad():
                # on training the clip image is created in the dataloader
                if not has_been_preprocessed:
                    # tensors should be 0-1
                    if tensors_0_1.ndim == 3:
                        tensors_0_1 = tensors_0_1.unsqueeze(0)
                    # training tensors are 0 - 1
                    tensors_0_1 = tensors_0_1.to(self.device, dtype=torch.float16)
                    # if images are out of this range throw error
                    if tensors_0_1.min() < -0.3 or tensors_0_1.max() > 1.3:
                        raise ValueError("image tensor values must be between 0 and 1. Got min: {}, max: {}".format(
                            tensors_0_1.min(), tensors_0_1.max()
                        ))
                    clip_image = self.image_processor(
                        images=tensors_0_1,
                        return_tensors="pt",
                        do_resize=True,
                        do_rescale=False,
                    ).pixel_values
                else:
                    clip_image = tensors_0_1
                clip_image = clip_image.to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype)).detach()

            if self.config.quad_image:
                # split the 4x4 grid and stack on batch
                ci1, ci2 = clip_image.chunk(2, dim=2)
                ci1, ci3 = ci1.chunk(2, dim=3)
                ci2, ci4 = ci2.chunk(2, dim=3)
                to_cat = []
                for i, ci in enumerate([ci1, ci2, ci3, ci4]):
                    if i < quad_count:
                        to_cat.append(ci)
                    else:
                        break

                clip_image = torch.cat(to_cat, dim=0).detach()

            if self.adapter_type == 'photo_maker':
                # Embeddings need to be  (b, num_inputs, c, h, w) for now, just put 1 input image
                clip_image = clip_image.unsqueeze(1)
                with torch.set_grad_enabled(is_training):
                    if is_training and self.config.train_image_encoder:
                        self.vision_encoder.train()
                        clip_image = clip_image.requires_grad_(True)
                        id_embeds = self.vision_encoder(
                            clip_image,
                            do_projection2=isinstance(self.sd_ref().text_encoder, list),
                        )
                    else:
                        with torch.no_grad():
                            self.vision_encoder.eval()
                            id_embeds = self.vision_encoder(
                                clip_image, do_projection2=isinstance(self.sd_ref().text_encoder, list)
                            ).detach()

                    prompt_embeds.text_embeds = self.fuse_module(
                        prompt_embeds.text_embeds,
                        id_embeds,
                        self.token_mask
                    )
                    return prompt_embeds
            elif self.adapter_type == 'clip_fusion':
                with torch.set_grad_enabled(is_training):
                    if is_training and self.config.train_image_encoder:
                        self.vision_encoder.train()
                        clip_image = clip_image.requires_grad_(True)
                        id_embeds = self.vision_encoder(
                            clip_image,
                            output_hidden_states=True,
                        )
                    else:
                        with torch.no_grad():
                            self.vision_encoder.eval()
                            id_embeds = self.vision_encoder(
                                clip_image, output_hidden_states=True
                            )

                    img_embeds = id_embeds['last_hidden_state']

                    if self.config.quad_image:
                        # get the outputs of the quat
                        chunks = img_embeds.chunk(quad_count, dim=0)
                        chunk_sum = torch.zeros_like(chunks[0])
                        for chunk in chunks:
                            chunk_sum = chunk_sum + chunk
                        # get the mean of them

                        img_embeds = chunk_sum / quad_count

                    if not is_training or not self.config.train_image_encoder:
                        img_embeds = img_embeds.detach()

                    prompt_embeds.text_embeds = self.clip_fusion_module(
                        prompt_embeds.text_embeds,
                        img_embeds
                    )
                    return prompt_embeds


        else:
            return prompt_embeds

    def get_empty_clip_image(self, batch_size: int, shape=None) -> torch.Tensor:
        with torch.no_grad():
            if shape is None:
                shape = [batch_size, 3, self.input_size, self.input_size]
            tensors_0_1 = torch.rand(shape, device=self.device)
            noise_scale = torch.rand([tensors_0_1.shape[0], 1, 1, 1], device=self.device,
                                     dtype=get_torch_dtype(self.sd_ref().dtype))
            tensors_0_1 = tensors_0_1 * noise_scale
            # tensors_0_1 = tensors_0_1 * 0
            mean = torch.tensor(self.clip_image_processor.image_mean).to(
                self.device, dtype=get_torch_dtype(self.sd_ref().dtype)
            ).detach()
            std = torch.tensor(self.clip_image_processor.image_std).to(
                self.device, dtype=get_torch_dtype(self.sd_ref().dtype)
            ).detach()
            tensors_0_1 = torch.clip((255. * tensors_0_1), 0, 255).round() / 255.0
            clip_image = (tensors_0_1 - mean.view([1, 3, 1, 1])) / std.view([1, 3, 1, 1])
        return clip_image.detach()

    def train(self, mode: bool = True):
        if self.config.train_image_encoder:
            self.vision_encoder.train(mode)
        super().train(mode)

    def trigger_pre_te(
            self,
            tensors_0_1: torch.Tensor,
            is_training=False,
            has_been_preprocessed=False,
            quad_count=4,
            batch_size=1,
    ) -> PromptEmbeds:
        if self.adapter_type == 'ilora' or self.adapter_type == 'vision_direct' or self.adapter_type == 'te_augmenter':
            skip_unconditional = self.sd_ref().is_flux
            if tensors_0_1 is None:
                tensors_0_1 = self.get_empty_clip_image(batch_size)
                has_been_preprocessed = True

            with torch.no_grad():
                # on training the clip image is created in the dataloader
                if not has_been_preprocessed:
                    # tensors should be 0-1
                    if tensors_0_1.ndim == 3:
                        tensors_0_1 = tensors_0_1.unsqueeze(0)
                    # training tensors are 0 - 1
                    tensors_0_1 = tensors_0_1.to(self.device, dtype=torch.float16)
                    # if images are out of this range throw error
                    if tensors_0_1.min() < -0.3 or tensors_0_1.max() > 1.3:
                        raise ValueError("image tensor values must be between 0 and 1. Got min: {}, max: {}".format(
                            tensors_0_1.min(), tensors_0_1.max()
                        ))
                    clip_image = self.image_processor(
                        images=tensors_0_1,
                        return_tensors="pt",
                        do_resize=True,
                        do_rescale=False,
                    ).pixel_values
                else:
                    clip_image = tensors_0_1
                    
                # if is pixtral
                if self.config.image_encoder_arch == 'pixtral' and self.config.pixtral_random_image_size:
                    # get the random size
                    random_size = random.randint(256, self.config.pixtral_max_image_size)
                    # images are already sized for max size, we have to fit them to the pixtral patch size to reduce / enlarge it farther.
                    h, w = clip_image.shape[2], clip_image.shape[3]
                    current_base_size = int(math.sqrt(w * h))
                    ratio = current_base_size / random_size
                    if ratio > 1:
                        w = round(w / ratio)
                        h = round(h / ratio)

                    width_tokens = (w - 1) // self.image_processor.image_patch_size + 1
                    height_tokens = (h - 1) // self.image_processor.image_patch_size + 1
                    assert width_tokens > 0
                    assert height_tokens > 0
                    
                    new_image_size = (
                        width_tokens * self.image_processor.image_patch_size,
                        height_tokens * self.image_processor.image_patch_size,
                    )
                    
                    # resize the image
                    clip_image = F.interpolate(clip_image, size=new_image_size, mode='bicubic', align_corners=False)
                    

                batch_size = clip_image.shape[0]
                if (self.adapter_type == 'vision_direct' or self.adapter_type == 'te_augmenter') and not skip_unconditional:
                    # add an unconditional so we can save it
                    unconditional = self.get_empty_clip_image(batch_size, shape=clip_image.shape).to(
                        clip_image.device, dtype=clip_image.dtype
                    )
                    clip_image = torch.cat([unconditional, clip_image], dim=0)

                clip_image = clip_image.to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype)).detach()

            if self.config.quad_image:
                # split the 4x4 grid and stack on batch
                ci1, ci2 = clip_image.chunk(2, dim=2)
                ci1, ci3 = ci1.chunk(2, dim=3)
                ci2, ci4 = ci2.chunk(2, dim=3)
                to_cat = []
                for i, ci in enumerate([ci1, ci2, ci3, ci4]):
                    if i < quad_count:
                        to_cat.append(ci)
                    else:
                        break

                clip_image = torch.cat(to_cat, dim=0).detach()

            if self.adapter_type == 'ilora':
                with torch.set_grad_enabled(is_training):
                    if is_training and self.config.train_image_encoder:
                        self.vision_encoder.train()
                        clip_image = clip_image.requires_grad_(True)
                        id_embeds = self.vision_encoder(
                            clip_image,
                            output_hidden_states=True,
                        )
                    else:
                        with torch.no_grad():
                            self.vision_encoder.eval()
                            id_embeds = self.vision_encoder(
                                clip_image, output_hidden_states=True
                            )

                    if self.config.clip_layer == 'penultimate_hidden_states':
                        img_embeds = id_embeds.hidden_states[-2]
                    elif self.config.clip_layer == 'last_hidden_state':
                        img_embeds = id_embeds.hidden_states[-1]
                    elif self.config.clip_layer == 'image_embeds':
                        img_embeds = id_embeds.image_embeds
                    else:
                        raise ValueError(f"unknown clip layer: {self.config.clip_layer}")

                    if self.config.quad_image:
                        # get the outputs of the quat
                        chunks = img_embeds.chunk(quad_count, dim=0)
                        chunk_sum = torch.zeros_like(chunks[0])
                        for chunk in chunks:
                            chunk_sum = chunk_sum + chunk
                        # get the mean of them

                        img_embeds = chunk_sum / quad_count

                    if not is_training or not self.config.train_image_encoder:
                        img_embeds = img_embeds.detach()

                    self.ilora_module(img_embeds)
            if self.adapter_type == 'vision_direct' or self.adapter_type == 'te_augmenter':
                with torch.set_grad_enabled(is_training):
                    if is_training and self.config.train_image_encoder:
                        self.vision_encoder.train()
                        clip_image = clip_image.requires_grad_(True)
                    else:
                        with torch.no_grad():
                            self.vision_encoder.eval()
                    clip_output = self.vision_encoder(
                        clip_image,
                        output_hidden_states=True,
                    )
                    if self.config.clip_layer == 'penultimate_hidden_states':
                        # they skip last layer for ip+
                        # https://github.com/tencent-ailab/IP-Adapter/blob/f4b6742db35ea6d81c7b829a55b0a312c7f5a677/tutorial_train_plus.py#L403C26-L403C26
                        clip_image_embeds = clip_output.hidden_states[-2]
                    elif self.config.clip_layer == 'last_hidden_state':
                        clip_image_embeds = clip_output.hidden_states[-1]
                    else:
                        if hasattr(clip_output, 'image_embeds'):
                            clip_image_embeds = clip_output.image_embeds
                        elif hasattr(clip_output, 'pooler_output'):
                            clip_image_embeds = clip_output.pooler_output
                        # TODO should we always norm image embeds?
                        # get norm embeddings
                        # l2_norm = torch.norm(clip_image_embeds, p=2)
                        # clip_image_embeds = clip_image_embeds / l2_norm

                    if not is_training or not self.config.train_image_encoder:
                        clip_image_embeds = clip_image_embeds.detach()

                    if self.adapter_type == 'te_augmenter':
                        clip_image_embeds = self.te_augmenter(clip_image_embeds)

                    if self.adapter_type == 'vision_direct':
                        clip_image_embeds = self.vd_adapter(clip_image_embeds)

                    # save them to the conditional and unconditional
                    try:
                        if skip_unconditional:
                            self.unconditional_embeds, self.conditional_embeds = None, clip_image_embeds
                        else:
                            self.unconditional_embeds, self.conditional_embeds = clip_image_embeds.chunk(2, dim=0)
                    except ValueError:
                        raise ValueError(f"could not split the clip image embeds into 2. Got shape: {clip_image_embeds.shape}")

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.config.train_only_image_encoder:
            yield from self.vision_encoder.parameters(recurse)
            return
        if self.config.type == 'photo_maker':
            yield from self.fuse_module.parameters(recurse)
            if self.config.train_image_encoder:
                yield from self.vision_encoder.parameters(recurse)
        elif self.config.type == 'clip_fusion':
            yield from self.clip_fusion_module.parameters(recurse)
            if self.config.train_image_encoder:
                yield from self.vision_encoder.parameters(recurse)
        elif self.config.type == 'ilora':
            yield from self.ilora_module.parameters(recurse)
            if self.config.train_image_encoder:
                yield from self.vision_encoder.parameters(recurse)
        elif self.config.type == 'text_encoder':
            for attn_processor in self.te_adapter.adapter_modules:
                yield from attn_processor.parameters(recurse)
        elif self.config.type == 'vision_direct':
            if self.config.train_scaler:
                # only yield the self.block_scaler = torch.nn.Parameter(torch.tensor([1.0] * num_modules)
                yield self.vd_adapter.block_scaler
            else:
                for attn_processor in self.vd_adapter.adapter_modules:
                    yield from attn_processor.parameters(recurse)
                if self.config.train_image_encoder:
                    yield from self.vision_encoder.parameters(recurse)
                if self.vd_adapter.resampler is not None:
                    yield from self.vd_adapter.resampler.parameters(recurse)
                if self.vd_adapter.pool is not None:
                    yield from self.vd_adapter.pool.parameters(recurse)
                if self.vd_adapter.sparse_autoencoder is not None:
                    yield from self.vd_adapter.sparse_autoencoder.parameters(recurse)
        elif self.config.type == 'te_augmenter':
            yield from self.te_augmenter.parameters(recurse)
            if self.config.train_image_encoder:
                yield from self.vision_encoder.parameters(recurse)
        elif self.config.type == 'single_value':
            yield from self.single_value_adapter.parameters(recurse)
        else:
            raise NotImplementedError

    def enable_gradient_checkpointing(self):
        if hasattr(self.vision_encoder, "enable_gradient_checkpointing"):
            self.vision_encoder.enable_gradient_checkpointing()
        elif hasattr(self.vision_encoder, 'gradient_checkpointing'):
            self.vision_encoder.gradient_checkpointing = True

    def get_additional_save_metadata(self) -> Dict[str, Any]:
        additional = {}
        if self.config.type == 'ilora':
            extra = self.ilora_module.get_additional_save_metadata()
            for k, v in extra.items():
                additional[k] = v
            additional['clip_layer'] = self.config.clip_layer
            additional['image_encoder_arch'] = self.config.head_dim
        return additional

    def post_weight_update(self):
        # do any kind of updates after the weight update
        if self.config.type == 'vision_direct':
            self.vd_adapter.post_weight_update()
        pass