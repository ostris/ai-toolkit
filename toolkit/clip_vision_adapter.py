from typing import TYPE_CHECKING, Mapping, Any

import torch
import weakref

from toolkit.config_modules import AdapterConfig
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
            num_input_tokens: int = 50,
            input_dim: int = 1024,
            num_output_tokens: int = 8,
            output_dim: int = 768,
            mid_dim: int = 128,
    ):
        super(Embedder, self).__init__()
        self.num_output_tokens = num_output_tokens
        self.num_input_tokens = num_input_tokens
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Convolutional layer to reduce channel dimension
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=mid_dim, kernel_size=1)

        # GELU Activation
        self.gelu = nn.GELU()

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(mid_dim)

        # Adaptive pooling to change sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_output_tokens)

        # Linear layer for final transformation
        self.final_linear = nn.Linear(mid_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Adjust for Conv1d
        x = self.conv(x)
        x = self.gelu(x)
        x = self.layer_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # Apply LayerNorm
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)  # Adjust back
        x = self.final_linear(x)
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
        # self.embedder = Embedder(
        #     num_output_tokens=self.config.num_tokens,
        #     num_input_tokens=self.image_encoder.config.top_k,  # max_position_embeddings ?
        #     input_dim=self.image_encoder.config.hidden_size,
        #     output_dim=sd.unet.config['cross_attention_dim'],
        # ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        heads = 12 if not sd.is_xl else 20
        # dim = sd.unet.config['cross_attention_dim'] if not sd.is_xl else 1280
        dim = sd.unet.config['cross_attention_dim']
        self.embedder = Resampler(
            dim=dim,
            depth=4,
            dim_head=64,
            heads=heads,
            num_queries=self.config.num_tokens,  # usually 16
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=sd.unet.config['cross_attention_dim'],
            ff_mult=4
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
            is_training=False
    ) -> torch.Tensor:
        with torch.no_grad():
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
        with torch.set_grad_enabled(is_training):
            if is_training:
                self.image_encoder.train()
            else:
                self.image_encoder.eval()
            clip_output = self.image_encoder(clip_image, output_hidden_states=True)
            clip_image_embeds = clip_output.hidden_states[-2]
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
        image_prompt_embeds = self.embedder(clip_image_embeds)
        # todo add support for multiple batch sizes
        if image_prompt_embeds.shape[0] != 1:
            raise ValueError("Batch size must be 1 for embedder for now")

        # output on sd1.5 is bs, num_tokens, 768
        if len(self.text_encoder_list) == 1:
            # add it to the text encoder
            self.set_vec(image_prompt_embeds[0], text_encoder_idx=0)
        else:
            raise ValueError("Multiple text encoders not supported yet")
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
