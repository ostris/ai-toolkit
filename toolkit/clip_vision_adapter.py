from typing import TYPE_CHECKING, Mapping, Any

import torch
import weakref

from toolkit.config_modules import AdapterConfig
from toolkit.models.clip_fusion import ZipperBlock
from toolkit.models.zipper_resampler import ZipperModule
from toolkit.prompt_utils import PromptEmbeds
from toolkit.train_tools import get_torch_dtype

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPVisionModel
)

from toolkit.resampler import Resampler

import torch.nn as nn


class Embedder(nn.Module):
    def __init__(
            self,
            num_input_tokens: int = 1,
            input_dim: int = 1024,
            num_output_tokens: int = 8,
            output_dim: int = 768,
            mid_dim: int = 1024
    ):
        super(Embedder, self).__init__()
        self.num_output_tokens = num_output_tokens
        self.num_input_tokens = num_input_tokens
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.gelu = nn.GELU()
        # self.fc2 = nn.Linear(mid_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, mid_dim)

        self.fc2.weight.data.zero_()

        self.layer_norm2 = nn.LayerNorm(mid_dim)
        self.fc3 = nn.Linear(mid_dim, mid_dim)
        self.gelu2 = nn.GELU()
        self.fc4 = nn.Linear(mid_dim, output_dim * num_output_tokens)

        # set the weights to 0
        self.fc3.weight.data.zero_()
        self.fc4.weight.data.zero_()


        # self.static_tokens = nn.Parameter(torch.zeros(num_output_tokens, output_dim))
        # self.scaler = nn.Parameter(torch.zeros(num_output_tokens, output_dim))

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.fc3(x)
        x = self.gelu2(x)
        x = self.fc4(x)

        x = x.view(-1, self.num_output_tokens, self.output_dim)

        return x


class ClipVisionAdapter(torch.nn.Module):
    def __init__(self, sd: 'StableDiffusion', adapter_config: AdapterConfig):
        super().__init__()
        self.config = adapter_config
        self.trigger = adapter_config.trigger
        self.trigger_class_name = adapter_config.trigger_class_name
        self.sd_ref: weakref.ref = weakref.ref(sd)
        # embedding stuff
        self.text_encoder_list = sd.text_encoder if isinstance(sd.text_encoder, list) else [sd.text_encoder]
        self.tokenizer_list = sd.tokenizer if isinstance(sd.tokenizer, list) else [sd.tokenizer]
        placeholder_tokens = [self.trigger]

        # add dummy tokens for multi-vector
        additional_tokens = []
        for i in range(1, self.config.num_tokens):
            additional_tokens.append(f"{self.trigger}_{i}")
        placeholder_tokens += additional_tokens

        # handle dual tokenizer
        self.tokenizer_list = self.sd_ref().tokenizer if isinstance(self.sd_ref().tokenizer, list) else [
            self.sd_ref().tokenizer]
        self.text_encoder_list = self.sd_ref().text_encoder if isinstance(self.sd_ref().text_encoder, list) else [
            self.sd_ref().text_encoder]

        self.placeholder_token_ids = []
        self.embedding_tokens = []

        print(f"Adding {placeholder_tokens} tokens to tokenizer")
        print(f"Adding {self.config.num_tokens} tokens to tokenizer")


        for text_encoder, tokenizer in zip(self.text_encoder_list, self.tokenizer_list):
            num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
            if num_added_tokens != self.config.num_tokens:
                raise ValueError(
                    f"The tokenizer already contains the token {self.trigger}. Please pass a different"
                    f" `placeholder_token` that is not already in the tokenizer. Only added {num_added_tokens}"
                )

            # Convert the initializer_token, placeholder_token to ids
            init_token_ids = tokenizer.encode(self.config.trigger_class_name, add_special_tokens=False)
            # if length of token ids is more than number of orm embedding tokens fill with *
            if len(init_token_ids) > self.config.num_tokens:
                init_token_ids = init_token_ids[:self.config.num_tokens]
            elif len(init_token_ids) < self.config.num_tokens:
                pad_token_id = tokenizer.encode(["*"], add_special_tokens=False)
                init_token_ids += pad_token_id * (self.config.num_tokens - len(init_token_ids))

            placeholder_token_ids = tokenizer.encode(placeholder_tokens, add_special_tokens=False)
            self.placeholder_token_ids.append(placeholder_token_ids)

            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_encoder.resize_token_embeddings(len(tokenizer))

            # Initialise the newly added placeholder token with the embeddings of the initializer token
            token_embeds = text_encoder.get_input_embeddings().weight.data
            with torch.no_grad():
                for initializer_token_id, token_id in zip(init_token_ids, placeholder_token_ids):
                    token_embeds[token_id] = token_embeds[initializer_token_id].clone()

            # replace "[name] with this. on training. This is automatically generated in pipeline on inference
            self.embedding_tokens.append(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)))

        # backup text encoder embeddings
        self.orig_embeds_params = [x.get_input_embeddings().weight.data.clone() for x in self.text_encoder_list]

        try:
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.config.image_encoder_path)
        except EnvironmentError:
            self.clip_image_processor = CLIPImageProcessor()
        self.device = self.sd_ref().unet.device
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.config.image_encoder_path,
            ignore_mismatched_sizes=True
        ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        if self.config.train_image_encoder:
            self.image_encoder.train()
        else:
            self.image_encoder.eval()

        # max_seq_len = CLIP tokens + CLS token
        image_encoder_state_dict = self.image_encoder.state_dict()
        in_tokens = 257
        if "vision_model.embeddings.position_embedding.weight" in image_encoder_state_dict:
            # clip
            in_tokens = int(image_encoder_state_dict["vision_model.embeddings.position_embedding.weight"].shape[0])

        if hasattr(self.image_encoder.config, 'hidden_sizes'):
            embedding_dim = self.image_encoder.config.hidden_sizes[-1]
        else:
            embedding_dim = self.image_encoder.config.target_hidden_size

        if self.config.clip_layer == 'image_embeds':
            in_tokens = 1
            embedding_dim = self.image_encoder.config.projection_dim

        self.embedder = Embedder(
            num_output_tokens=self.config.num_tokens,
            num_input_tokens=in_tokens,
            input_dim=embedding_dim,
            output_dim=self.sd_ref().unet.config['cross_attention_dim'],
            mid_dim=embedding_dim * self.config.num_tokens,
        ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))

        self.embedder.train()

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'embedder': self.embedder.state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        }
        if self.config.train_image_encoder:
            state_dict['image_encoder'] = self.image_encoder.state_dict(
                *args, destination=destination, prefix=prefix,
                keep_vars=keep_vars)

        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.embedder.load_state_dict(state_dict["embedder"], strict=strict)
        if self.config.train_image_encoder and 'image_encoder' in state_dict:
            self.image_encoder.load_state_dict(state_dict["image_encoder"], strict=strict)

    def parameters(self, *args, **kwargs):
        yield from self.embedder.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        yield from self.embedder.named_parameters(*args, **kwargs)

    def get_clip_image_embeds_from_tensors(
            self, tensors_0_1: torch.Tensor, drop=False,
            is_training=False,
            has_been_preprocessed=False
    ) -> torch.Tensor:
        with torch.no_grad():
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
                # unconditional
                if drop:
                    if self.clip_noise_zero:
                        tensors_0_1 = torch.rand_like(tensors_0_1).detach()
                        noise_scale = torch.rand([tensors_0_1.shape[0], 1, 1, 1], device=self.device,
                                                 dtype=get_torch_dtype(self.sd_ref().dtype))
                        tensors_0_1 = tensors_0_1 * noise_scale
                    else:
                        tensors_0_1 = torch.zeros_like(tensors_0_1).detach()
                    # tensors_0_1 = tensors_0_1 * 0
                clip_image = self.clip_image_processor(
                    images=tensors_0_1,
                    return_tensors="pt",
                    do_resize=True,
                    do_rescale=False,
                ).pixel_values
            else:
                if drop:
                    # scale the noise down
                    if self.clip_noise_zero:
                        tensors_0_1 = torch.rand_like(tensors_0_1).detach()
                        noise_scale = torch.rand([tensors_0_1.shape[0], 1, 1, 1], device=self.device,
                                                 dtype=get_torch_dtype(self.sd_ref().dtype))
                        tensors_0_1 = tensors_0_1 * noise_scale
                    else:
                        tensors_0_1 = torch.zeros_like(tensors_0_1).detach()
                    # tensors_0_1 = tensors_0_1 * 0
                    mean = torch.tensor(self.clip_image_processor.image_mean).to(
                        self.device, dtype=get_torch_dtype(self.sd_ref().dtype)
                    ).detach()
                    std = torch.tensor(self.clip_image_processor.image_std).to(
                        self.device, dtype=get_torch_dtype(self.sd_ref().dtype)
                    ).detach()
                    tensors_0_1 = torch.clip((255. * tensors_0_1), 0, 255).round() / 255.0
                    clip_image = (tensors_0_1 - mean.view([1, 3, 1, 1])) / std.view([1, 3, 1, 1])

                else:
                    clip_image = tensors_0_1
            clip_image = clip_image.to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype)).detach()
        with torch.set_grad_enabled(is_training):
            if is_training:
                self.image_encoder.train()
            else:
                self.image_encoder.eval()
            clip_output = self.image_encoder(clip_image, output_hidden_states=True)

            if self.config.clip_layer == 'penultimate_hidden_states':
                # they skip last layer for ip+
                # https://github.com/tencent-ailab/IP-Adapter/blob/f4b6742db35ea6d81c7b829a55b0a312c7f5a677/tutorial_train_plus.py#L403C26-L403C26
                clip_image_embeds = clip_output.hidden_states[-2]
            elif self.config.clip_layer == 'last_hidden_state':
                clip_image_embeds = clip_output.hidden_states[-1]
            else:
                clip_image_embeds = clip_output.image_embeds
        return clip_image_embeds

    import torch

    def set_vec(self, new_vector, text_encoder_idx=0):
        # Get the embedding layer
        embedding_layer = self.text_encoder_list[text_encoder_idx].get_input_embeddings()

        # Indices to replace in the embeddings
        indices_to_replace = self.placeholder_token_ids[text_encoder_idx]

        # Replace the specified embeddings with new_vector
        for idx in indices_to_replace:
            vector_idx = idx - indices_to_replace[0]
            embedding_layer.weight[idx] = new_vector[vector_idx]

    # adds it to the tokenizer
    def forward(self, clip_image_embeds: torch.Tensor) -> PromptEmbeds:
        clip_image_embeds = clip_image_embeds.to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        if clip_image_embeds.ndim == 2:
            #  expand the token dimension
            clip_image_embeds = clip_image_embeds.unsqueeze(1)
        image_prompt_embeds = self.embedder(clip_image_embeds)
        # todo add support for multiple batch sizes
        if image_prompt_embeds.shape[0] != 1:
            raise ValueError("Batch size must be 1 for embedder for now")

        # output on sd1.5 is bs, num_tokens, 768
        if len(self.text_encoder_list) == 1:
            # add it to the text encoder
            self.set_vec(image_prompt_embeds[0], text_encoder_idx=0)
        elif len(self.text_encoder_list) == 2:
            if self.text_encoder_list[0].config.target_hidden_size + self.text_encoder_list[1].config.target_hidden_size != \
                    image_prompt_embeds.shape[2]:
                raise ValueError("Something went wrong. The embeddings do not match the text encoder sizes")
            # sdxl variants
            # image_prompt_embeds = 2048
            # te1 = 768
            # te2 = 1280
            te1_embeds = image_prompt_embeds[:, :, :self.text_encoder_list[0].config.target_hidden_size]
            te2_embeds = image_prompt_embeds[:, :, self.text_encoder_list[0].config.target_hidden_size:]
            self.set_vec(te1_embeds[0], text_encoder_idx=0)
            self.set_vec(te2_embeds[0], text_encoder_idx=1)
        else:

            raise ValueError("Unsupported number of text encoders")
        # just a place to put a breakpoint
        pass

    def restore_embeddings(self):
        # Let's make sure we don't update any embedding weights besides the newly added token
        for text_encoder, tokenizer, orig_embeds, placeholder_token_ids in zip(
                self.text_encoder_list,
                self.tokenizer_list,
                self.orig_embeds_params,
                self.placeholder_token_ids
        ):
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[
            min(placeholder_token_ids): max(placeholder_token_ids) + 1] = False
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds[index_no_updates]
                # detach it all
                text_encoder.get_input_embeddings().weight.detach_()

    def enable_gradient_checkpointing(self):
        self.image_encoder.gradient_checkpointing = True

    def inject_trigger_into_prompt(self, prompt, expand_token=False, to_replace_list=None, add_if_not_present=True):
        output_prompt = prompt
        embedding_tokens = self.embedding_tokens[0]  # shoudl be the same
        default_replacements = ["[name]", "[trigger]"]

        replace_with = embedding_tokens if expand_token else self.trigger
        if to_replace_list is None:
            to_replace_list = default_replacements
        else:
            to_replace_list += default_replacements

        # remove duplicates
        to_replace_list = list(set(to_replace_list))

        # replace them all
        for to_replace in to_replace_list:
            # replace it
            output_prompt = output_prompt.replace(to_replace, replace_with)

        # see how many times replace_with is in the prompt
        num_instances = output_prompt.count(replace_with)

        if num_instances == 0 and add_if_not_present:
            # add it to the beginning of the prompt
            output_prompt = replace_with + " " + output_prompt

        if num_instances > 1:
            print(
                f"Warning: {replace_with} token appears {num_instances} times in prompt {output_prompt}. This may cause issues.")

        return output_prompt

    # reverses injection with class name. useful for normalizations
    def inject_trigger_class_name_into_prompt(self, prompt):
        output_prompt = prompt
        embedding_tokens = self.embedding_tokens[0]  # shoudl be the same

        default_replacements = ["[name]", "[trigger]", embedding_tokens, self.trigger]

        replace_with = self.config.trigger_class_name
        to_replace_list = default_replacements

        # remove duplicates
        to_replace_list = list(set(to_replace_list))

        # replace them all
        for to_replace in to_replace_list:
            # replace it
            output_prompt = output_prompt.replace(to_replace, replace_with)

        # see how many times replace_with is in the prompt
        num_instances = output_prompt.count(replace_with)

        if num_instances > 1:
            print(
                f"Warning: {replace_with} token appears {num_instances} times in prompt {output_prompt}. This may cause issues.")

        return output_prompt
