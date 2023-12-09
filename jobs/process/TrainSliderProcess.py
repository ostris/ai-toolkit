import copy
import os
import random
from collections import OrderedDict
from typing import Union

from PIL import Image
from diffusers import T2IAdapter
from torchvision.transforms import transforms
from tqdm import tqdm

from toolkit.basic import value_map
from toolkit.config_modules import SliderConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.sd_device_states_presets import get_train_sd_device_state_preset
from toolkit.train_tools import get_torch_dtype, apply_snr_weight, apply_learnable_snr_gos
import gc
from toolkit import train_tools
from toolkit.prompt_utils import \
    EncodedPromptPair, ACTION_TYPES_SLIDER, \
    EncodedAnchor, concat_prompt_pairs, \
    concat_anchors, PromptEmbedsCache, encode_prompts_to_cache, build_prompt_pair_batch_from_cache, split_anchors, \
    split_prompt_pairs

import torch
from .BaseSDTrainProcess import BaseSDTrainProcess


def flush():
    torch.cuda.empty_cache()
    gc.collect()


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class TrainSliderProcess(BaseSDTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.prompt_txt_list = None
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        self.slider_config = SliderConfig(**self.get_conf('slider', {}))
        self.prompt_cache = PromptEmbedsCache()
        self.prompt_pairs: list[EncodedPromptPair] = []
        self.anchor_pairs: list[EncodedAnchor] = []
        # keep track of prompt chunk size
        self.prompt_chunk_size = 1

        # check if we have more targets than steps
        # this can happen because of permutation son shuffling
        if len(self.slider_config.targets) > self.train_config.steps:
            # trim targets
            self.slider_config.targets = self.slider_config.targets[:self.train_config.steps]

        # get presets
        self.eval_slider_device_state = get_train_sd_device_state_preset(
            self.device_torch,
            train_unet=False,
            train_text_encoder=False,
            cached_latents=self.is_latents_cached,
            train_lora=False,
            train_adapter=False,
            train_embedding=False,
        )

        self.train_slider_device_state = get_train_sd_device_state_preset(
            self.device_torch,
            train_unet=self.train_config.train_unet,
            train_text_encoder=False,
            cached_latents=self.is_latents_cached,
            train_lora=True,
            train_adapter=False,
            train_embedding=False,
        )

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):

        # read line by line from file
        if self.slider_config.prompt_file:
            self.print(f"Loading prompt file from {self.slider_config.prompt_file}")
            with open(self.slider_config.prompt_file, 'r', encoding='utf-8') as f:
                self.prompt_txt_list = f.readlines()
                # clean empty lines
                self.prompt_txt_list = [line.strip() for line in self.prompt_txt_list if len(line.strip()) > 0]

            self.print(f"Found {len(self.prompt_txt_list)} prompts.")

            if not self.slider_config.prompt_tensors:
                print(f"Prompt tensors not found. Building prompt tensors for {self.train_config.steps} steps.")
                # shuffle
                random.shuffle(self.prompt_txt_list)
                # trim to max steps
                self.prompt_txt_list = self.prompt_txt_list[:self.train_config.steps]
                # trim list to our max steps

        cache = PromptEmbedsCache()
        print(f"Building prompt cache")

        # get encoded latents for our prompts
        with torch.no_grad():
            # list of neutrals. Can come from file or be empty
            neutral_list = self.prompt_txt_list if self.prompt_txt_list is not None else [""]

            # build the prompts to cache
            prompts_to_cache = []
            for neutral in neutral_list:
                for target in self.slider_config.targets:
                    prompt_list = [
                        f"{target.target_class}",  # target_class
                        f"{target.target_class} {neutral}",  # target_class with neutral
                        f"{target.positive}",  # positive_target
                        f"{target.positive} {neutral}",  # positive_target with neutral
                        f"{target.negative}",  # negative_target
                        f"{target.negative} {neutral}",  # negative_target with neutral
                        f"{neutral}",  # neutral
                        f"{target.positive} {target.negative}",  # both targets
                        f"{target.negative} {target.positive}",  # both targets reverse
                    ]
                    prompts_to_cache += prompt_list

            # remove duplicates
            prompts_to_cache = list(dict.fromkeys(prompts_to_cache))

            # trim to max steps if max steps is lower than prompt count
            # todo, this can break if we have more targets than steps, should be fixed, by reducing permuations, but could stil happen with low steps
            # prompts_to_cache = prompts_to_cache[:self.train_config.steps]

            # encode them
            cache = encode_prompts_to_cache(
                prompt_list=prompts_to_cache,
                sd=self.sd,
                cache=cache,
                prompt_tensor_file=self.slider_config.prompt_tensors
            )

            prompt_pairs = []
            prompt_batches = []
            for neutral in tqdm(neutral_list, desc="Building Prompt Pairs", leave=False):
                for target in self.slider_config.targets:
                    prompt_pair_batch = build_prompt_pair_batch_from_cache(
                        cache=cache,
                        target=target,
                        neutral=neutral,

                    )
                    if self.slider_config.batch_full_slide:
                        # concat the prompt pairs
                        # this allows us to run the entire 4 part process in one shot (for slider)
                        self.prompt_chunk_size = 4
                        concat_prompt_pair_batch = concat_prompt_pairs(prompt_pair_batch).to('cpu')
                        prompt_pairs += [concat_prompt_pair_batch]
                    else:
                        self.prompt_chunk_size = 1
                        # do them one at a time (probably not necessary after new optimizations)
                        prompt_pairs += [x.to('cpu') for x in prompt_pair_batch]

            # setup anchors
            anchor_pairs = []
            for anchor in self.slider_config.anchors:
                # build the cache
                for prompt in [
                    anchor.prompt,
                    anchor.neg_prompt  # empty neutral
                ]:
                    if cache[prompt] == None:
                        cache[prompt] = self.sd.encode_prompt(prompt)

                anchor_batch = []
                # we get the prompt pair multiplier from first prompt pair
                # since they are all the same. We need to match their network polarity
                prompt_pair_multipliers = prompt_pairs[0].multiplier_list
                for prompt_multiplier in prompt_pair_multipliers:
                    # match the network multiplier polarity
                    anchor_scalar = 1.0 if prompt_multiplier > 0 else -1.0
                    anchor_batch += [
                        EncodedAnchor(
                            prompt=cache[anchor.prompt],
                            neg_prompt=cache[anchor.neg_prompt],
                            multiplier=anchor.multiplier * anchor_scalar
                        )
                    ]

                anchor_pairs += [
                    concat_anchors(anchor_batch).to('cpu')
                ]
            if len(anchor_pairs) > 0:
                self.anchor_pairs = anchor_pairs

        # move to cpu to save vram
        # We don't need text encoder anymore, but keep it on cpu for sampling
        # if text encoder is list
        if isinstance(self.sd.text_encoder, list):
            for encoder in self.sd.text_encoder:
                encoder.to("cpu")
        else:
            self.sd.text_encoder.to("cpu")
        self.prompt_cache = cache
        self.prompt_pairs = prompt_pairs
        # self.anchor_pairs = anchor_pairs
        flush()
        if self.data_loader is not None:
            # we will have images, prep the vae
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        # end hook_before_train_loop

    def before_dataset_load(self):
        if self.slider_config.use_adapter == 'depth':
            print(f"Loading T2I Adapter for depth")
            # called before LoRA network is loaded but after model is loaded
            # attach the adapter here so it is there before we load the network
            adapter_path = 'TencentARC/t2iadapter_depth_sd15v2'
            if self.model_config.is_xl:
                adapter_path = 'TencentARC/t2i-adapter-depth-midas-sdxl-1.0'

            print(f"Loading T2I Adapter from {adapter_path}")

            # dont name this adapter since we are not training it
            self.t2i_adapter = T2IAdapter.from_pretrained(
                adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype), varient="fp16"
            ).to(self.device_torch)
            self.t2i_adapter.eval()
            self.t2i_adapter.requires_grad_(False)
            flush()

    @torch.no_grad()
    def get_adapter_images(self, batch: Union[None, 'DataLoaderBatchDTO']):

        img_ext_list = ['.jpg', '.jpeg', '.png', '.webp']
        adapter_folder_path = self.slider_config.adapter_img_dir
        adapter_images = []
        # loop through images
        for file_item in batch.file_items:
            img_path = file_item.path
            file_name_no_ext = os.path.basename(img_path).split('.')[0]
            # find the image
            for ext in img_ext_list:
                if os.path.exists(os.path.join(adapter_folder_path, file_name_no_ext + ext)):
                    adapter_images.append(os.path.join(adapter_folder_path, file_name_no_ext + ext))
                    break
        width, height = batch.file_items[0].crop_width, batch.file_items[0].crop_height
        adapter_tensors = []
        # load images with torch transforms
        for idx, adapter_image in enumerate(adapter_images):
            # we need to centrally crop the largest dimension of the image to match the batch shape after scaling
            # to the smallest dimension
            img: Image.Image = Image.open(adapter_image)
            if img.width > img.height:
                # scale down so height is the same as batch
                new_height = height
                new_width = int(img.width * (height / img.height))
            else:
                new_width = width
                new_height = int(img.height * (width / img.width))

            img = img.resize((new_width, new_height))
            crop_fn = transforms.CenterCrop((height, width))
            # crop the center to match batch
            img = crop_fn(img)
            img = adapter_transforms(img)
            adapter_tensors.append(img)

        # stack them
        adapter_tensors = torch.stack(adapter_tensors).to(
            self.device_torch, dtype=get_torch_dtype(self.train_config.dtype)
        )
        return adapter_tensors

    def hook_train_loop(self, batch: Union['DataLoaderBatchDTO', None]):
        # set to eval mode
        self.sd.set_device_state(self.eval_slider_device_state)
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)

            # get a random pair
            prompt_pair: EncodedPromptPair = self.prompt_pairs[
                torch.randint(0, len(self.prompt_pairs), (1,)).item()
            ]
            # move to device and dtype
            prompt_pair.to(self.device_torch, dtype=dtype)

            # get a random resolution
            height, width = self.slider_config.resolutions[
                torch.randint(0, len(self.slider_config.resolutions), (1,)).item()
            ]
            if self.train_config.gradient_checkpointing:
                # may get disabled elsewhere
                self.sd.unet.enable_gradient_checkpointing()

        noise_scheduler = self.sd.noise_scheduler
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        loss_function = torch.nn.MSELoss()

        pred_kwargs = {}

        def get_noise_pred(neg, pos, gs, cts, dn):
            down_kwargs = copy.deepcopy(pred_kwargs)
            if 'down_block_additional_residuals' in down_kwargs:
                dbr_batch_size = down_kwargs['down_block_additional_residuals'][0].shape[0]
                if dbr_batch_size != dn.shape[0]:
                    amount_to_add = int(dn.shape[0] * 2 / dbr_batch_size)
                    down_kwargs['down_block_additional_residuals'] = [
                        torch.cat([sample.clone()] * amount_to_add) for sample in
                        down_kwargs['down_block_additional_residuals']
                    ]
            return self.sd.predict_noise(
                latents=dn,
                text_embeddings=train_tools.concat_prompt_embeddings(
                    neg,  # negative prompt
                    pos,  # positive prompt
                    self.train_config.batch_size,
                ),
                timestep=cts,
                guidance_scale=gs,
                **down_kwargs
            )

        with torch.no_grad():
            adapter_images = None
            self.sd.unet.eval()

            # for a complete slider, the batch size is 4 to begin with now
            true_batch_size = prompt_pair.target_class.text_embeds.shape[0] * self.train_config.batch_size
            from_batch = False
            if batch is not None:
                # traing from a batch of images, not generating ourselves
                from_batch = True
                noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
                if self.slider_config.adapter_img_dir is not None:
                    adapter_images = self.get_adapter_images(batch)
                    adapter_strength_min = 0.9
                    adapter_strength_max = 1.0

                    def rand_strength(sample):
                        adapter_conditioning_scale = torch.rand(
                            (1,), device=self.device_torch, dtype=dtype
                        )

                        adapter_conditioning_scale = value_map(
                            adapter_conditioning_scale,
                            0.0,
                            1.0,
                            adapter_strength_min,
                            adapter_strength_max
                        )
                        return sample.to(self.device_torch, dtype=dtype).detach() * adapter_conditioning_scale

                    down_block_additional_residuals = self.t2i_adapter(adapter_images)
                    down_block_additional_residuals = [
                        rand_strength(sample) for sample in down_block_additional_residuals
                    ]
                    pred_kwargs['down_block_additional_residuals'] = down_block_additional_residuals

                denoised_latents = torch.cat([noisy_latents] * self.prompt_chunk_size, dim=0)
                current_timestep = timesteps
            else:

                self.sd.noise_scheduler.set_timesteps(
                    self.train_config.max_denoising_steps, device=self.device_torch
                )

                # ger a random number of steps
                timesteps_to = torch.randint(
                    1, self.train_config.max_denoising_steps - 1, (1,)
                ).item()

                # get noise
                noise = self.sd.get_latent_noise(
                    pixel_height=height,
                    pixel_width=width,
                    batch_size=true_batch_size,
                    noise_offset=self.train_config.noise_offset,
                ).to(self.device_torch, dtype=dtype)

                # get latents
                latents = noise * self.sd.noise_scheduler.init_noise_sigma
                latents = latents.to(self.device_torch, dtype=dtype)

                assert not self.network.is_active
                self.sd.unet.eval()
                # pass the multiplier list to the network
                # double up since we are doing cfg
                self.network.multiplier = prompt_pair.multiplier_list + prompt_pair.multiplier_list
                denoised_latents = self.sd.diffuse_some_steps(
                    latents,  # pass simple noise latents
                    train_tools.concat_prompt_embeddings(
                        prompt_pair.positive_target,  # unconditional
                        prompt_pair.target_class,  # target
                        self.train_config.batch_size,
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )


                noise_scheduler.set_timesteps(1000)

                current_timestep_index = int(timesteps_to * 1000 / self.train_config.max_denoising_steps)
                current_timestep = noise_scheduler.timesteps[current_timestep_index]

            # split the latents into out prompt pair chunks
            denoised_latent_chunks = torch.chunk(denoised_latents, self.prompt_chunk_size, dim=0)
            denoised_latent_chunks = [x.detach() for x in denoised_latent_chunks]

            # flush()  # 4.2GB to 3GB on 512x512
            mask_multiplier = torch.ones((denoised_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            has_mask = False
            if batch and batch.mask_tensor is not None:
                with self.timer('get_mask_multiplier'):
                    # upsampling no supported for bfloat16
                    mask_multiplier = batch.mask_tensor.to(self.device_torch, dtype=torch.float16).detach()
                    # scale down to the size of the latents, mask multiplier shape(bs, 1, width, height), noisy_latents shape(bs, channels, width, height)
                    mask_multiplier = torch.nn.functional.interpolate(
                        mask_multiplier, size=(noisy_latents.shape[2], noisy_latents.shape[3])
                    )
                    # expand to match latents
                    mask_multiplier = mask_multiplier.expand(-1, noisy_latents.shape[1], -1, -1)
                    mask_multiplier = mask_multiplier.to(self.device_torch, dtype=dtype).detach()
                    has_mask = True

            if has_mask:
                unmasked_target = get_noise_pred(
                    prompt_pair.positive_target,  # negative prompt
                    prompt_pair.target_class,  # positive prompt
                    1,
                    current_timestep,
                    denoised_latents
                )
                unmasked_target = unmasked_target.detach()
                unmasked_target.requires_grad = False
            else:
                unmasked_target = None

            # 4.20 GB RAM for 512x512
            positive_latents = get_noise_pred(
                prompt_pair.positive_target,  # negative prompt
                prompt_pair.negative_target,  # positive prompt
                1,
                current_timestep,
                denoised_latents
            )
            positive_latents = positive_latents.detach()
            positive_latents.requires_grad = False

            neutral_latents = get_noise_pred(
                prompt_pair.positive_target,  # negative prompt
                prompt_pair.empty_prompt,  # positive prompt (normally neutral
                1,
                current_timestep,
                denoised_latents
            )
            neutral_latents = neutral_latents.detach()
            neutral_latents.requires_grad = False

            unconditional_latents = get_noise_pred(
                prompt_pair.positive_target,  # negative prompt
                prompt_pair.positive_target,  # positive prompt
                1,
                current_timestep,
                denoised_latents
            )
            unconditional_latents = unconditional_latents.detach()
            unconditional_latents.requires_grad = False

            denoised_latents = denoised_latents.detach()

        self.sd.set_device_state(self.train_slider_device_state)
        self.sd.unet.train()
        # start accumulating gradients
        self.optimizer.zero_grad(set_to_none=True)

        anchor_loss_float = None
        if len(self.anchor_pairs) > 0:
            with torch.no_grad():
                # get a random anchor pair
                anchor: EncodedAnchor = self.anchor_pairs[
                    torch.randint(0, len(self.anchor_pairs), (1,)).item()
                ]
                anchor.to(self.device_torch, dtype=dtype)

                # first we get the target prediction without network active
                anchor_target_noise = get_noise_pred(
                    anchor.neg_prompt, anchor.prompt, 1, current_timestep, denoised_latents
                    # ).to("cpu", dtype=torch.float32)
                ).requires_grad_(False)

                # to save vram, we will run these through separately while tracking grads
                # otherwise it consumes a ton of vram and this isn't our speed bottleneck
                anchor_chunks = split_anchors(anchor, self.prompt_chunk_size)
                anchor_target_noise_chunks = torch.chunk(anchor_target_noise, self.prompt_chunk_size, dim=0)
                assert len(anchor_chunks) == len(denoised_latent_chunks)

            # 4.32 GB RAM for 512x512
            with self.network:
                assert self.network.is_active
                anchor_float_losses = []
                for anchor_chunk, denoised_latent_chunk, anchor_target_noise_chunk in zip(
                        anchor_chunks, denoised_latent_chunks, anchor_target_noise_chunks
                ):
                    self.network.multiplier = anchor_chunk.multiplier_list + anchor_chunk.multiplier_list

                    anchor_pred_noise = get_noise_pred(
                        anchor_chunk.neg_prompt, anchor_chunk.prompt, 1, current_timestep, denoised_latent_chunk
                    )
                    # 9.42 GB RAM for 512x512 -> 4.20 GB RAM for 512x512 with new grad_checkpointing
                    anchor_loss = loss_function(
                        anchor_target_noise_chunk,
                        anchor_pred_noise,
                    )
                    anchor_float_losses.append(anchor_loss.item())
                    # compute anchor loss gradients
                    # we will accumulate them later
                    # this saves a ton of memory doing them separately
                    anchor_loss.backward()
                    del anchor_pred_noise
                    del anchor_target_noise_chunk
                    del anchor_loss
                    flush()

            anchor_loss_float = sum(anchor_float_losses) / len(anchor_float_losses)
            del anchor_chunks
            del anchor_target_noise_chunks
            del anchor_target_noise
            # move anchor back to cpu
            anchor.to("cpu")

        with torch.no_grad():
            if self.slider_config.low_ram:
                prompt_pair_chunks = split_prompt_pairs(prompt_pair.detach(), self.prompt_chunk_size)
                denoised_latent_chunks = denoised_latent_chunks  # just to have it in one place
                positive_latents_chunks = torch.chunk(positive_latents.detach(), self.prompt_chunk_size, dim=0)
                neutral_latents_chunks = torch.chunk(neutral_latents.detach(), self.prompt_chunk_size, dim=0)
                unconditional_latents_chunks = torch.chunk(
                    unconditional_latents.detach(),
                    self.prompt_chunk_size,
                    dim=0
                )
                mask_multiplier_chunks = torch.chunk(mask_multiplier, self.prompt_chunk_size, dim=0)
                if unmasked_target is not None:
                    unmasked_target_chunks = torch.chunk(unmasked_target, self.prompt_chunk_size, dim=0)
                else:
                    unmasked_target_chunks = [None for _ in range(self.prompt_chunk_size)]
            else:
                # run through in one instance
                prompt_pair_chunks = [prompt_pair.detach()]
                denoised_latent_chunks = [torch.cat(denoised_latent_chunks, dim=0).detach()]
                positive_latents_chunks = [positive_latents.detach()]
                neutral_latents_chunks = [neutral_latents.detach()]
                unconditional_latents_chunks = [unconditional_latents.detach()]
                mask_multiplier_chunks = [mask_multiplier]
                unmasked_target_chunks = [unmasked_target]

            # flush()
        assert len(prompt_pair_chunks) == len(denoised_latent_chunks)
        # 3.28 GB RAM for 512x512
        with self.network:
            assert self.network.is_active
            loss_list = []
            for prompt_pair_chunk, \
                    denoised_latent_chunk, \
                    positive_latents_chunk, \
                    neutral_latents_chunk, \
                    unconditional_latents_chunk, \
                    mask_multiplier_chunk, \
                    unmasked_target_chunk \
                    in zip(
                prompt_pair_chunks,
                denoised_latent_chunks,
                positive_latents_chunks,
                neutral_latents_chunks,
                unconditional_latents_chunks,
                mask_multiplier_chunks,
                unmasked_target_chunks
            ):
                self.network.multiplier = prompt_pair_chunk.multiplier_list + prompt_pair_chunk.multiplier_list
                target_latents = get_noise_pred(
                    prompt_pair_chunk.positive_target,
                    prompt_pair_chunk.target_class,
                    1,
                    current_timestep,
                    denoised_latent_chunk
                )

                guidance_scale = 1.0

                offset = guidance_scale * (positive_latents_chunk - unconditional_latents_chunk)

                # make offset multiplier based on actions
                offset_multiplier_list = []
                for action in prompt_pair_chunk.action_list:
                    if action == ACTION_TYPES_SLIDER.ERASE_NEGATIVE:
                        offset_multiplier_list += [-1.0]
                    elif action == ACTION_TYPES_SLIDER.ENHANCE_NEGATIVE:
                        offset_multiplier_list += [1.0]

                offset_multiplier = torch.tensor(offset_multiplier_list).to(offset.device, dtype=offset.dtype)
                # make offset multiplier match rank of offset
                offset_multiplier = offset_multiplier.view(offset.shape[0], 1, 1, 1)
                offset *= offset_multiplier

                offset_neutral = neutral_latents_chunk
                # offsets are already adjusted on a per-batch basis
                offset_neutral += offset
                offset_neutral = offset_neutral.detach().requires_grad_(False)

                # 16.15 GB RAM for 512x512 -> 4.20GB RAM for 512x512 with new grad_checkpointing
                loss = torch.nn.functional.mse_loss(target_latents.float(), offset_neutral.float(), reduction="none")

                # do inverted mask to preserve non masked
                if has_mask and unmasked_target_chunk is not None:
                    loss = loss * mask_multiplier_chunk
                    # match the mask unmasked_target_chunk
                    mask_target_loss = torch.nn.functional.mse_loss(
                        target_latents.float(),
                        unmasked_target_chunk.float(),
                        reduction="none"
                    )
                    mask_target_loss = mask_target_loss * (1.0 - mask_multiplier_chunk)
                    loss += mask_target_loss

                loss = loss.mean([1, 2, 3])

                if self.train_config.learnable_snr_gos:
                    if from_batch:
                        # match batch size
                        loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler,
                                                self.train_config.min_snr_gamma)
                    else:
                        # match batch size
                        timesteps_index_list = [current_timestep_index for _ in range(target_latents.shape[0])]
                        # add snr_gamma
                        loss = apply_learnable_snr_gos(loss, timesteps_index_list, self.snr_gos)
                if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
                    if from_batch:
                        # match batch size
                        loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler,
                                                self.train_config.min_snr_gamma)
                    else:
                        # match batch size
                        timesteps_index_list = [current_timestep_index for _ in range(target_latents.shape[0])]
                        # add min_snr_gamma
                        loss = apply_snr_weight(loss, timesteps_index_list, noise_scheduler,
                                                self.train_config.min_snr_gamma)


                loss = loss.mean() * prompt_pair_chunk.weight

                loss.backward()
                loss_list.append(loss.item())
                del target_latents
                del offset_neutral
                del loss
                # flush()

        optimizer.step()
        lr_scheduler.step()

        loss_float = sum(loss_list) / len(loss_list)
        if anchor_loss_float is not None:
            loss_float += anchor_loss_float

        del (
            positive_latents,
            neutral_latents,
            unconditional_latents,
            # latents
        )
        # move back to cpu
        prompt_pair.to("cpu")
        # flush()

        # reset network
        self.network.multiplier = 1.0

        loss_dict = OrderedDict(
            {'loss': loss_float},
        )
        if anchor_loss_float is not None:
            loss_dict['sl_l'] = loss_float
            loss_dict['an_l'] = anchor_loss_float

        return loss_dict
        # end hook_train_loop
