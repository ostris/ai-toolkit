"""
Dataloader mixins - refactored for better organization.
"""

import base64
import glob
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Dict, Union
import traceback

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file, safe_open
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor

from toolkit.basic import flush, value_map
from toolkit.buckets import get_bucket_for_image_size, get_resolution
from toolkit.config_modules import ControlTypes
from toolkit.control_generator import ControlGenerator
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.pixtral_vision import PixtralVisionImagePreprocessorCompatible
from toolkit.prompt_utils import inject_trigger_into_prompt
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from PIL.ImageOps import exif_transpose
import albumentations as A
from toolkit.print import print_acc
from toolkit.accelerator import get_accelerator
from toolkit.prompt_utils import PromptEmbeds
from torchvision.transforms import functional as TF
from toolkit.train_tools import get_torch_dtype

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.stable_diffusion_model import StableDiffusion

accelerator = get_accelerator()


class LatentCachingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self._encoded_latent: Union[torch.Tensor, None] = None
        self._latent_path: Union[str, None] = None
        self.is_latent_cached = False
        self.is_caching_to_disk = False
        self.is_caching_to_memory = False
        self.latent_load_device = 'cpu'
        # sd1 or sdxl or others
        self.latent_space_version = 'sd1'
        # todo, increment this if we change the latent format to invalidate cache
        self.latent_version = 1

    def get_latent_info_dict(self: 'FileItemDTO'):
        item = OrderedDict([
            ("filename", os.path.basename(self.path)),
            ("scale_to_width", self.scale_to_width),
            ("scale_to_height", self.scale_to_height),
            ("crop_x", self.crop_x),
            ("crop_y", self.crop_y),
            ("crop_width", self.crop_width),
            ("crop_height", self.crop_height),
            ("latent_space_version", self.latent_space_version),
            ("latent_version", self.latent_version),
        ])
        # when adding items, do it after so we dont change old latents
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        if self.dataset_config.num_frames > 1:
            item["num_frames"] = self.dataset_config.num_frames
        return item

    def get_latent_path(self: 'FileItemDTO', recalculate=False):
        if self._latent_path is not None and not recalculate:
            return self._latent_path
        else:
            # we store latents in a folder in same path as image called _latent_cache
            img_dir = os.path.dirname(self.path)
            latent_dir = os.path.join(img_dir, '_latent_cache')
            hash_dict = self.get_latent_info_dict()
            filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
            # get base64 hash of md5 checksum of hash_dict
            hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
            hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
            hash_str = hash_str.replace('=', '')
            self._latent_path = os.path.join(latent_dir, f'{filename_no_ext}_{hash_str}.safetensors')

        return self._latent_path

    def cleanup_latent(self):
        if self._encoded_latent is not None:
            if not self.is_caching_to_memory:
                # we are caching on disk, don't save in memory
                self._encoded_latent = None
            else:
                # move it back to cpu
                self._encoded_latent = self._encoded_latent.to('cpu')

    def get_latent(self, device=None):
        if not self.is_latent_cached:
            return None

        latent_path = self.get_latent_path()

        # Two caching strategies based on configuration:
        # 1. Memory caching (cache_latents: true): Load once, keep in RAM, share across workers (TODO #2)
        # 2. Disk-only caching (cache_latents_to_disk: true): Load on-demand via mmap, minimal RAM usage
        if self.is_caching_to_memory:
            # Strategy 1: Keep in RAM for sharing across workers
            if self._encoded_latent is None:
                state_dict = load_file(latent_path, device='cpu')
                self._encoded_latent = state_dict['latent']
            return self._encoded_latent
        else:
            # Strategy 2: Load on-demand via mmap (don't cache in RAM)
            # This saves RAM at the cost of re-loading each time
            # But with mmap, the OS caches the file pages, so it's still fast
            with safe_open(latent_path, framework="pt", device="cpu") as f:
                # Get mmap-backed tensor - data is paged in by OS as needed
                latent = f.get_tensor('latent')
                # Return a clone to detach from file handle
                # The clone operation reads from mmap, avoiding full RAM allocation first
                # After this returns, the latent can be used and then discarded
                return latent.clone()



class LatentCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.latent_cache = {}

    def cache_latents_all_latents(self: 'AiToolkitDataset'):
        if self.dataset_config.num_frames > 1:
            raise Exception("Error: caching latents is not supported for multi-frame datasets")
        if self.sd is None:
            raise RuntimeError("Error: self.sd is None. Cannot cache latents without model. "
                             "This should not happen - caching should only occur in main process before pickling.")
        with accelerator.main_process_first():
            print_acc(f"Caching latents for {self.dataset_path}")
            # cache all latents to disk
            to_disk = self.is_caching_latents_to_disk
            to_memory = self.is_caching_latents_to_memory

            if to_disk:
                print_acc(" - Saving latents to disk")
            if to_memory:
                print_acc(" - Keeping latents in memory")
            # move sd items to cpu except for vae
            self.sd.set_device_state_preset('cache_latents')

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc=f'Caching latents{" to disk" if to_disk else ""}'):
                # set latent space version
                if self.sd.model_config.latent_space_version is not None:
                    file_item.latent_space_version = self.sd.model_config.latent_space_version
                elif self.sd.is_xl:
                    file_item.latent_space_version = 'sdxl'
                elif self.sd.is_v3:
                    file_item.latent_space_version = 'sd3'
                elif self.sd.is_auraflow:
                    file_item.latent_space_version = 'sdxl'
                elif self.sd.is_flux:
                    file_item.latent_space_version = 'flux1'
                elif self.sd.model_config.is_pixart_sigma:
                    file_item.latent_space_version = 'sdxl'
                else:
                    file_item.latent_space_version = self.sd.model_config.arch
                file_item.is_caching_to_disk = to_disk
                file_item.is_caching_to_memory = to_memory
                file_item.latent_load_device = self.sd.device

                latent_path = file_item.get_latent_path(recalculate=True)
                # check if it is saved to disk already
                if os.path.exists(latent_path):
                    if to_memory:
                        # load it into memory
                        state_dict = load_file(latent_path, device='cpu')
                        file_item._encoded_latent = state_dict['latent'].to('cpu', dtype=self.sd.torch_dtype)
                else:
                    # not saved to disk, calculate
                    # load the image first
                    file_item.load_and_process_image(self.transform, only_load_latents=True)
                    dtype = self.sd.torch_dtype
                    device = self.sd.device_torch
                    # add batch dimension
                    try:
                        imgs = file_item.tensor.unsqueeze(0).to(device, dtype=dtype)
                        latent = self.sd.encode_images(imgs).squeeze(0)
                    except Exception as e:
                        print_acc(f"Error processing image: {file_item.path}")
                        print_acc(f"Error: {str(e)}")
                        raise e
                    # save_latent
                    if to_disk:
                        state_dict = OrderedDict([
                            ('latent', latent.clone().detach().cpu()),
                        ])
                        # metadata
                        meta = get_meta_for_safetensors(file_item.get_latent_info_dict())
                        os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                        save_file(state_dict, latent_path, metadata=meta)

                    if to_memory:
                        # keep it in memory
                        file_item._encoded_latent = latent.to('cpu', dtype=self.sd.torch_dtype)

                    del imgs
                    del latent
                    del file_item.tensor

                    # flush(garbage_collect=False)
                file_item.is_latent_cached = True
                i += 1
                # flush every 100
                # if i % 100 == 0:
                #     flush()

            # restore device state
            self.sd.restore_device_state()



class TextEmbeddingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.prompt_embeds: Union[PromptEmbeds, None] = None
        self._text_embedding_path: Union[str, None] = None
        self.is_text_embedding_cached = False
        self.text_embedding_load_device = 'cpu'
        self.text_embedding_space_version = 'sd1'
        self.text_embedding_version = 1

    def get_text_embedding_info_dict(self: 'FileItemDTO'):
        # make sure the caption is loaded here
        # TODO: we need a way to cache all the other features like trigger words, DOP, etc. For now, we need to throw an error if not compatible.
        if self.caption is None:
            self.load_caption()
        item = OrderedDict([
            ("caption", self.caption),
            ("text_embedding_space_version", self.text_embedding_space_version),
            ("text_embedding_version", self.text_embedding_version),
        ])
        # if we have a control image, cache the path
        if self.encode_control_in_text_embeddings and self.control_path is not None:
            item["control_path"] = self.control_path
        return item

    def get_text_embedding_path(self: 'FileItemDTO', recalculate=False):
        if self._text_embedding_path is not None and not recalculate:
            return self._text_embedding_path
        else:
            # we store text embeddings in a folder in same path as image called _text_embedding_cache
            img_dir = os.path.dirname(self.path)
            te_dir = os.path.join(img_dir, '_t_e_cache')
            hash_dict = self.get_text_embedding_info_dict()
            filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
            # get base64 hash of md5 checksum of hash_dict
            hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
            hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
            hash_str = hash_str.replace('=', '')
            self._text_embedding_path = os.path.join(te_dir, f'{filename_no_ext}_{hash_str}.safetensors')

        return self._text_embedding_path

    def cleanup_text_embedding(self):
        if self.prompt_embeds is not None:
            # we are caching on disk, don't save in memory
            self.prompt_embeds = None

    def load_prompt_embedding(self, device=None):
        if not self.is_text_embedding_cached:
            return
        if self.prompt_embeds is None:
            # load it from disk
            self.prompt_embeds = PromptEmbeds.load(self.get_text_embedding_path())


class TextEmbeddingCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.is_caching_text_embeddings = self.dataset_config.cache_text_embeddings

    def cache_text_embeddings(self: 'AiToolkitDataset'):
        if self.sd is None:
            raise RuntimeError("Error: self.sd is None. Cannot cache text embeddings without model. "
                             "This should not happen - caching should only occur in main process before pickling.")
        with accelerator.main_process_first():
            print_acc(f"Caching text_embeddings for {self.dataset_path}")
            print_acc(" - Saving text embeddings to disk")

            did_move = False

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc='Caching text embeddings to disk'):
                file_item.text_embedding_space_version = self.sd.model_config.arch
                file_item.latent_load_device = self.sd.device

                text_embedding_path = file_item.get_text_embedding_path(recalculate=True)
                # only process if not saved to disk
                if not os.path.exists(text_embedding_path):
                    # load if not loaded
                    if not did_move:
                        self.sd.set_device_state_preset('cache_text_encoder')
                        did_move = True
                        
                    if file_item.encode_control_in_text_embeddings:
                        if file_item.control_path is None:
                            raise Exception(f"Could not find a control image for {file_item.path} which is needed for this model")
                        ctrl_img_list = []
                        control_path_list = file_item.control_path
                        if not isinstance(file_item.control_path, list):
                            control_path_list = [control_path_list]
                        for i in range(len(control_path_list)):
                            try:
                                img = Image.open(control_path_list[i]).convert("RGB")
                                img = exif_transpose(img)
                                # convert to 0 to 1 tensor
                                img = (
                                    TF.to_tensor(img)
                                    .unsqueeze(0)
                                    .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                                )
                                ctrl_img_list.append(img)
                            except Exception as e:
                                print_acc(f"Error: {e}")
                                print_acc(f"Error loading control image: {control_path_list[i]}")
                        
                        if len(ctrl_img_list) == 0:
                            ctrl_img = None
                        elif not self.sd.has_multiple_control_images:
                            ctrl_img = ctrl_img_list[0]
                        else:
                            ctrl_img = ctrl_img_list
                        prompt_embeds: PromptEmbeds = self.sd.encode_prompt(file_item.caption, control_images=ctrl_img)
                    else:
                        prompt_embeds: PromptEmbeds = self.sd.encode_prompt(file_item.caption)
                    # save it
                    prompt_embeds.save(text_embedding_path)
                    del prompt_embeds
                file_item.is_text_embedding_cached = True
                i += 1
            # restore device state
            # if did_move:
            #     self.sd.restore_device_state()



class CLIPCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.clip_vision_num_unconditional_cache = 20
        self.clip_vision_unconditional_cache = []

    def cache_clip_vision_to_disk(self: 'AiToolkitDataset'):
        if not self.is_caching_clip_vision_to_disk:
            return
        if self.sd is None:
            raise RuntimeError("Error: self.sd is None. Cannot cache clip vision without model. "
                             "This should not happen - caching should only occur in main process before pickling.")
        with torch.no_grad():
            print_acc(f"Caching clip vision for {self.dataset_path}")

            print_acc(" - Saving clip to disk")
            # move sd items to cpu except for vae
            self.sd.set_device_state_preset('cache_clip')

            # make sure the adapter has attributes
            if self.sd.adapter is None:
                print_acc(" - WARNING: No adapter found, skipping CLIP vision caching (only works with IP-Adapter or similar)")
                return

            clip_image_processor: CLIPImageProcessor = None
            if hasattr(self.sd.adapter, 'clip_image_processor'):
                clip_image_processor = self.sd.adapter.clip_image_processor

            if clip_image_processor is None:
                raise Exception("Error: must have a clip image processor to cache clip vision to disk")

            vision_encoder: CLIPVisionModelWithProjection = None
            if hasattr(self.sd.adapter, 'image_encoder'):
                vision_encoder = self.sd.adapter.image_encoder
            if hasattr(self.sd.adapter, 'vision_encoder'):
                vision_encoder = self.sd.adapter.vision_encoder

            if vision_encoder is None:
                raise Exception("Error: must have a vision encoder to cache clip vision to disk")

            # move vision encoder to device
            vision_encoder.to(self.sd.device)

            is_quad = self.sd.adapter.config.quad_image
            image_encoder_path = self.sd.adapter.config.image_encoder_path

            dtype = self.sd.torch_dtype
            device = self.sd.device_torch
            if hasattr(self.sd.adapter, 'clip_noise_zero') and self.sd.adapter.clip_noise_zero:
                # just to do this, we did :)
                # need more samples as it is random noise
                self.clip_vision_num_unconditional_cache = self.clip_vision_num_unconditional_cache
            else:
                # only need one since it doesnt change
                self.clip_vision_num_unconditional_cache = 1

            # cache unconditionals
            print_acc(f" - Caching {self.clip_vision_num_unconditional_cache} unconditional clip vision to disk")
            clip_vision_cache_path = os.path.join(self.dataset_config.clip_image_path, '_clip_vision_cache')

            unconditional_paths = []

            is_noise_zero = hasattr(self.sd.adapter, 'clip_noise_zero') and self.sd.adapter.clip_noise_zero

            for i in range(self.clip_vision_num_unconditional_cache):
                hash_dict = OrderedDict([
                    ("image_encoder_path", image_encoder_path),
                    ("is_quad", is_quad),
                    ("is_noise_zero", is_noise_zero),
                ])
                # get base64 hash of md5 checksum of hash_dict
                hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
                hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
                hash_str = hash_str.replace('=', '')

                uncond_path = os.path.join(clip_vision_cache_path, f'uncond_{hash_str}_{i}.safetensors')
                if os.path.exists(uncond_path):
                    # skip it
                    unconditional_paths.append(uncond_path)
                    continue

                # generate a random image
                img_shape = (1, 3, self.sd.adapter.input_size, self.sd.adapter.input_size)
                if is_noise_zero:
                    tensors_0_1 = torch.rand(img_shape).to(device, dtype=torch.float32)
                else:
                    tensors_0_1 = torch.zeros(img_shape).to(device, dtype=torch.float32)
                clip_image = clip_image_processor(
                    images=tensors_0_1,
                    return_tensors="pt",
                    do_resize=True,
                    do_rescale=False,
                ).pixel_values

                if is_quad:
                    # split the 4x4 grid and stack on batch
                    ci1, ci2 = clip_image.chunk(2, dim=2)
                    ci1, ci3 = ci1.chunk(2, dim=3)
                    ci2, ci4 = ci2.chunk(2, dim=3)
                    clip_image = torch.cat([ci1, ci2, ci3, ci4], dim=0).detach()

                clip_output = vision_encoder(
                    clip_image.to(device, dtype=dtype),
                    output_hidden_states=True
                )
                # make state_dict ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                state_dict = OrderedDict([
                    ('image_embeds', clip_output.image_embeds.clone().detach().cpu()),
                    ('last_hidden_state', clip_output.hidden_states[-1].clone().detach().cpu()),
                    ('penultimate_hidden_states', clip_output.hidden_states[-2].clone().detach().cpu()),
                ])

                os.makedirs(os.path.dirname(uncond_path), exist_ok=True)
                save_file(state_dict, uncond_path)
                unconditional_paths.append(uncond_path)

            self.clip_vision_unconditional_cache = unconditional_paths

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc=f'Caching clip vision to disk'):
                file_item.is_caching_clip_vision_to_disk = True
                file_item.clip_vision_load_device = self.sd.device
                file_item.clip_vision_is_quad = is_quad
                file_item.clip_image_encoder_path = image_encoder_path
                file_item.clip_vision_unconditional_paths = unconditional_paths
                if file_item.has_clip_augmentations:
                    raise Exception("Error: clip vision caching is not supported with clip augmentations")

                embedding_path = file_item.get_clip_vision_embeddings_path(recalculate=True)
                # check if it is saved to disk already
                if not os.path.exists(embedding_path):
                    # load the image first
                    file_item.load_clip_image()
                    # add batch dimension
                    clip_image = file_item.clip_image_tensor.unsqueeze(0).to(device, dtype=dtype)

                    if is_quad:
                        # split the 4x4 grid and stack on batch
                        ci1, ci2 = clip_image.chunk(2, dim=2)
                        ci1, ci3 = ci1.chunk(2, dim=3)
                        ci2, ci4 = ci2.chunk(2, dim=3)
                        clip_image = torch.cat([ci1, ci2, ci3, ci4], dim=0).detach()

                    clip_output = vision_encoder(
                        clip_image.to(device, dtype=dtype),
                        output_hidden_states=True
                    )

                    # make state_dict ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                    state_dict = OrderedDict([
                        ('image_embeds', clip_output.image_embeds.clone().detach().cpu()),
                        ('last_hidden_state', clip_output.hidden_states[-1].clone().detach().cpu()),
                        ('penultimate_hidden_states', clip_output.hidden_states[-2].clone().detach().cpu()),
                    ])
                    # metadata
                    meta = get_meta_for_safetensors(file_item.get_clip_vision_info_dict())
                    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
                    save_file(state_dict, embedding_path, metadata=meta)

                    del clip_image
                    del clip_output
                    del file_item.clip_image_tensor

                    # flush(garbage_collect=False)
                file_item.is_vision_clip_cached = True
                i += 1
            # flush every 100
            # if i % 100 == 0:
            #     flush()

        # restore device state
        self.sd.restore_device_state()




class ControlCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
            self.control_generator: ControlGenerator = None
    
    def add_control_path_to_file_item(self: 'AiToolkitDataset', file_item: 'FileItemDTO', control_path: str, control_type: ControlTypes):
        if control_type == 'inpaint':
            file_item.inpaint_path = control_path
            file_item.has_inpaint_image = True
        elif control_type == 'mask':
            file_item.mask_path = control_path
            file_item.has_mask_image = True
        else:
            if file_item.control_path is None:
                file_item.control_path = [control_path]
            elif isinstance(file_item.control_path, str):
                file_item.control_path = [file_item.control_path, control_path]
            elif isinstance(file_item.control_path, list):
                file_item.control_path.append(control_path)
            else:
                raise Exception(f"Error: control_path is not a string or list: {file_item.control_path}")
            file_item.has_control_image = True

    def setup_controls(self: 'AiToolkitDataset'):
        if not self.is_generating_controls:
            return
        if self.sd is None:
            raise RuntimeError("Error: self.sd is None. Cannot setup controls without model. "
                             "This should not happen - control setup should only occur in main process before pickling.")
        with torch.no_grad():
            print_acc(f"Generating controls for {self.dataset_path}")
            device = self.sd.device

            self.control_generator = ControlGenerator(
                device=device,
                sd=self.sd,
            )

            # use tqdm to show progress
            for file_item in tqdm(self.file_list, desc=f'Generating Controls'):
                for control_type in self.dataset_config.controls:
                    # generates the control if it is not already there
                    control_path = self.control_generator.get_control_path(file_item.path, control_type)
                    if control_path is not None:
                        self.add_control_path_to_file_item(file_item, control_path, control_type)
                
            # remove models
            self.control_generator.cleanup()
            self.control_generator = None
            
            flush()
