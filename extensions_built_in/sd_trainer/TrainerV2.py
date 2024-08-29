import os
import random
from collections import OrderedDict
from typing import Union, List

import numpy as np
from diffusers import T2IAdapter, ControlNetModel
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.data_loader import get_dataloader_datasets
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.stable_diffusion_model import BlankNetwork
from toolkit.train_tools import get_torch_dtype, add_all_snr_to_noise_scheduler
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms
from diffusers import EMAModel
import math
from toolkit.train_tools import precondition_model_outputs_flow_match
from toolkit.models.unified_training_model import UnifiedTrainingModel


def flush():
    torch.cuda.empty_cache()
    gc.collect()


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class TrainerV2(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
        self.do_prior_prediction = False
        self.do_long_prompts = False
        self.do_guided_loss = False

        self._clip_image_embeds_unconditional: Union[List[str], None] = None
        self.negative_prompt_pool: Union[List[str], None] = None
        self.batch_negative_prompt: Union[List[str], None] = None

        self.scaler = torch.cuda.amp.GradScaler()

        self.is_bfloat = self.train_config.dtype == "bfloat16" or self.train_config.dtype == "bf16"

        self.do_grad_scale = True
        if self.is_fine_tuning:
            self.do_grad_scale = False
        if self.adapter_config is not None:
            if self.adapter_config.train:
                self.do_grad_scale = False

        if self.train_config.dtype in ["fp16", "float16"]:
            # patch the scaler to allow fp16 training
            org_unscale_grads = self.scaler._unscale_grads_
            def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
                return org_unscale_grads(optimizer, inv_scale, found_inf, True)
            self.scaler._unscale_grads_ = _unscale_grads_replacer

        self.unified_training_model: UnifiedTrainingModel = None
        self.device_ids = list(range(torch.cuda.device_count()))

    def before_model_load(self):
        pass

    def before_dataset_load(self):
        self.assistant_adapter = None
        # get adapter assistant if one is set
        if self.train_config.adapter_assist_name_or_path is not None:
            adapter_path = self.train_config.adapter_assist_name_or_path

            if self.train_config.adapter_assist_type == "t2i":
                # dont name this adapter since we are not training it
                self.assistant_adapter = T2IAdapter.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch)
            elif self.train_config.adapter_assist_type == "control_net":
                self.assistant_adapter = ControlNetModel.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
            else:
                raise ValueError(f"Unknown adapter assist type {self.train_config.adapter_assist_type}")

            self.assistant_adapter.eval()
            self.assistant_adapter.requires_grad_(False)
            flush()
        if self.train_config.train_turbo and self.train_config.show_turbo_outputs:
            raise ValueError("Turbo outputs are not supported on MultiGPUSDTrainer")

    def hook_before_train_loop(self):
        # if self.train_config.do_prior_divergence:
        #     self.do_prior_prediction = True
        # move vae to device if we did not cache latents
        if not self.is_latents_cached:
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        else:
            # offload it. Already cached
            self.sd.vae.to('cpu')
            flush()
        add_all_snr_to_noise_scheduler(self.sd.noise_scheduler, self.device_torch)
        if self.adapter is not None:
            self.adapter.to(self.device_torch)

            # check if we have regs and using adapter and caching clip embeddings
            has_reg = self.datasets_reg is not None and len(self.datasets_reg) > 0
            is_caching_clip_embeddings = self.datasets is not None and any([self.datasets[i].cache_clip_vision_to_disk for i in range(len(self.datasets))])

            if has_reg and is_caching_clip_embeddings:
                # we need a list of unconditional clip image embeds from other datasets to handle regs
                unconditional_clip_image_embeds = []
                datasets = get_dataloader_datasets(self.data_loader)
                for i in range(len(datasets)):
                    unconditional_clip_image_embeds += datasets[i].clip_vision_unconditional_cache

                if len(unconditional_clip_image_embeds) == 0:
                    raise ValueError("No unconditional clip image embeds found. This should not happen")

                self._clip_image_embeds_unconditional = unconditional_clip_image_embeds

        if self.train_config.negative_prompt is not None:
            raise ValueError("Negative prompt is not supported on MultiGPUSDTrainer")

        # setup the unified training model
        self.unified_training_model = UnifiedTrainingModel(
            sd=self.sd,
            network=self.network,
            adapter=self.adapter,
            assistant_adapter=self.assistant_adapter,
            train_config=self.train_config,
            adapter_config=self.adapter_config,
            embedding=self.embedding,
            timer=self.timer,
            trigger_word=self.trigger_word,
            gpu_ids=self.device_ids,
        )
        self.unified_training_model = nn.DataParallel(
            self.unified_training_model,
            device_ids=self.device_ids
        )

        self.unified_training_model = self.unified_training_model.to(self.device_torch)

        # call parent hook
        super().hook_before_train_loop()

    # you can expand these in a child class to make customization easier

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return self.unified_training_model.preprocess_batch(batch)


    def before_unet_predict(self):
        pass

    def after_unet_predict(self):
        pass

    def end_of_training_loop(self):
        pass

    def hook_train_loop(self, batch: 'DataLoaderBatchDTO'):

        self.optimizer.zero_grad(set_to_none=True)

        loss = self.unified_training_model(batch)

        if torch.isnan(loss):
            print("loss is nan")
            loss = torch.zeros_like(loss).requires_grad_(True)

        if self.network is not None:
            network = self.network
        else:
            network = BlankNetwork()

        with (network):
            with self.timer('backward'):
                # todo we have multiplier seperated. works for now as res are not in same batch, but need to change
                # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                # it will destroy the gradients. This is because the network is a context manager
                # and will change the multipliers back to 0.0 when exiting. They will be
                # 0.0 for the backward pass and the gradients will be 0.0
                # I spent weeks on fighting this. DON'T DO IT
                # with fsdp_overlap_step_with_backward():
                # if self.is_bfloat:
                # loss.backward()
                # else:
                if not self.do_grad_scale:
                    loss.backward()
                else:
                    self.scaler.scale(loss).backward()

        if not self.is_grad_accumulation_step:
            # fix this for multi params
            if self.train_config.optimizer != 'adafactor':
                if self.do_grad_scale:
                    self.scaler.unscale_(self.optimizer)
                if isinstance(self.params[0], dict):
                    for i in range(len(self.params)):
                        torch.nn.utils.clip_grad_norm_(self.params[i]['params'], self.train_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
            # only step if we are not accumulating
            with self.timer('optimizer_step'):
                # self.optimizer.step()
                if not self.do_grad_scale:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                with self.timer('ema_update'):
                    self.ema.update()
        else:
            # gradient accumulation. Just a place for breakpoint
            pass

        # TODO Should we only step scheduler on grad step? If so, need to recalculate last step
        with self.timer('scheduler_step'):
            self.lr_scheduler.step()

        if self.embedding is not None:
            with self.timer('restore_embeddings'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.embedding.restore_embeddings()
        if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
            with self.timer('restore_adapter'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.adapter.restore_embeddings()

        loss_dict = OrderedDict(
            {'loss': loss.item()}
        )

        self.end_of_training_loop()

        return loss_dict
