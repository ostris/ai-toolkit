import torch
import sys

from PIL import Image
from torch.nn import Parameter
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from toolkit.models.clip_pre_processor import CLIPImagePreProcessor
from toolkit.paths import REPOS_ROOT
from toolkit.photomaker import PhotoMakerIDEncoder, FuseModule, PhotoMakerCLIPEncoder
from toolkit.saving import load_ip_adapter_model
from toolkit.train_tools import get_torch_dtype

sys.path.append(REPOS_ROOT)
from typing import TYPE_CHECKING, Union, Iterator, Mapping, Any, Tuple, List, Optional
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
    ConvNextImageProcessor
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

        self.setup_adapter()

        # try to load from our name_or_path
        if self.config.name_or_path is not None and self.config.name_or_path.endswith('.bin'):
            self.load_state_dict(torch.load(self.config.name_or_path, map_location=self.device), strict=False)

        # add the trigger word to the tokenizer
        if isinstance(self.sd_ref().tokenizer, list):
            for tokenizer in self.sd_ref().tokenizer:
                tokenizer.add_tokens([self.flag_word], special_tokens=True)
        else:
            self.sd_ref().tokenizer.add_tokens([self.flag_word], special_tokens=True)

    def setup_adapter(self):
        if self.adapter_type == 'photo_maker':
            sd = self.sd_ref()
            embed_dim = sd.unet.config['cross_attention_dim']
            self.fuse_module = FuseModule(embed_dim)
        else:
            raise ValueError(f"unknown adapter type: {self.adapter_type}")

    def forward(self, *args, **kwargs):
        if self.adapter_type == 'photo_maker':
            id_pixel_values = args[0]
            prompt_embeds: PromptEmbeds = args[1]
            class_tokens_mask = args[2]

            grads_on_image_encoder = self.config.train_image_encoder and torch.is_grad_enabled()

            with torch.set_grad_enabled(grads_on_image_encoder):
                id_embeds = self.vision_encoder(self, id_pixel_values, do_projection2=False)

            if not grads_on_image_encoder:
                id_embeds = id_embeds.detach()

            prompt_embeds = prompt_embeds.detach()

            updated_prompt_embeds = self.fuse_module(
                prompt_embeds, id_embeds, class_tokens_mask
            )

            return updated_prompt_embeds
        else:
            raise NotImplementedError

    def setup_clip(self):
        adapter_config = self.config
        sd = self.sd_ref()
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

        if 'lora_weights' in state_dict:
            # todo add LoRA
            # self.sd_ref().pipeline.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")
            # self.sd_ref().pipeline.fuse_lora()
            pass
        if 'id_encoder' in state_dict and self.adapter_type == 'photo_maker':
            self.vision_encoder.load_state_dict(state_dict['id_encoder'], strict=strict)
            # check to see if the fuse weights are there
            fuse_weights = {}
            for k, v in state_dict['id_encoder'].items():
                if k.startswith('fuse_module'):
                    k = k.replace('fuse_module.', '')
                    fuse_weights[k] = v
            if len(fuse_weights) > 0:
                self.fuse_module.load_state_dict(fuse_weights, strict=strict)

        if 'fuse_module' in state_dict:
            self.fuse_module.load_state_dict(state_dict['fuse_module'], strict=strict)

        pass

    def state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        if self.adapter_type == 'photo_maker':
            if self.train_image_encoder:
                state_dict["id_encoder"] = self.vision_encoder.state_dict()

            state_dict["fuse_module"] = self.fuse_module.state_dict()

            # todo save LoRA
        else:
            raise NotImplementedError

    def condition_prompt(
            self,
            prompt: Union[List[str], str],
            is_unconditional: bool = False,
    ):
        if self.adapter_type == 'photo_maker':
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
                            prompt = " ".join([self.config.class_names[0] for _ in range(self.num_control_images)]) + ' ' + prompt

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


                        boolean_mask = torch.zeros(flag_idx-1, dtype=torch.bool)
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

    def condition_encoded_embeds(
            self,
            tensors_0_1: torch.Tensor,
            prompt_embeds: PromptEmbeds,
            is_training=False,
            has_been_preprocessed=False,
            is_unconditional=False
    ) -> PromptEmbeds:
        if self.adapter_type == 'photo_maker':
            if is_unconditional:
                # we dont condition the negative embeds for photo maker
                return prompt_embeds
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

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.config.type == 'photo_maker':
            yield from self.fuse_module.parameters(recurse)
            if self.config.train_image_encoder:
                yield from self.vision_encoder.parameters(recurse)
