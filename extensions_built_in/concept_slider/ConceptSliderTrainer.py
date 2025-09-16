from collections import OrderedDict
from typing import Optional

import torch

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype


class ConceptSliderTrainerConfig:
    def __init__(self, **kwargs):
        self.guidance_strength: float = kwargs.get("guidance_strength", 3.0)
        self.anchor_strength: float = kwargs.get("anchor_strength", 1.0)
        self.positive_prompt: str = kwargs.get("positive_prompt", "")
        self.negative_prompt: str = kwargs.get("negative_prompt", "")
        self.target_class: str = kwargs.get("target_class", "")
        self.anchor_class: Optional[str] = kwargs.get("anchor_class", None)


class ConceptSliderTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.do_guided_loss = True

        self.slider: ConceptSliderTrainerConfig = ConceptSliderTrainerConfig(
            **self.config.get("slider", {})
        )

        self.positive_prompt = self.slider.positive_prompt
        self.positive_prompt_embeds: Optional[PromptEmbeds] = None
        self.negative_prompt = self.slider.negative_prompt
        self.negative_prompt_embeds: Optional[PromptEmbeds] = None
        self.target_class = self.slider.target_class
        self.target_class_embeds: Optional[PromptEmbeds] = None
        self.anchor_class = self.slider.anchor_class
        self.anchor_class_embeds: Optional[PromptEmbeds] = None

    def hook_before_train_loop(self):
        # do this before calling parent as it unloads the text encoder if requested
        if self.is_caching_text_embeddings:
            # make sure model is on cpu for this part so we don't oom.
            self.sd.unet.to("cpu")

        # cache unconditional embeds (blank prompt)
        with torch.no_grad():
            self.positive_prompt_embeds = (
                self.sd.encode_prompt(
                    [self.positive_prompt],
                )
                .to(self.device_torch, dtype=self.sd.torch_dtype)
                .detach()
            )

            self.target_class_embeds = (
                self.sd.encode_prompt(
                    [self.target_class],
                )
                .to(self.device_torch, dtype=self.sd.torch_dtype)
                .detach()
            )

            self.negative_prompt_embeds = (
                self.sd.encode_prompt(
                    [self.negative_prompt],
                )
                .to(self.device_torch, dtype=self.sd.torch_dtype)
                .detach()
            )

            if self.anchor_class is not None:
                self.anchor_class_embeds = (
                    self.sd.encode_prompt(
                        [self.anchor_class],
                    )
                    .to(self.device_torch, dtype=self.sd.torch_dtype)
                    .detach()
                )

        # call parent
        super().hook_before_train_loop()

    def get_guided_loss(
        self,
        noisy_latents: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: "DataLoaderBatchDTO",
        noise: torch.Tensor,
        unconditional_embeds: Optional[PromptEmbeds] = None,
        **kwargs,
    ):
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        

        # do out prior preds first
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)
            self.sd.unet.eval()
            noisy_latents = noisy_latents.to(self.device_torch, dtype=dtype).detach()

            batch_size = noisy_latents.shape[0]

            positive_embeds = concat_prompt_embeds(
                [self.positive_prompt_embeds] * batch_size
            ).to(self.device_torch, dtype=dtype)
            target_class_embeds = concat_prompt_embeds(
                [self.target_class_embeds] * batch_size
            ).to(self.device_torch, dtype=dtype)
            negative_embeds = concat_prompt_embeds(
                [self.negative_prompt_embeds] * batch_size
            ).to(self.device_torch, dtype=dtype)

            if self.anchor_class_embeds is not None:
                anchor_embeds = concat_prompt_embeds(
                    [self.anchor_class_embeds] * batch_size
                ).to(self.device_torch, dtype=dtype)

            if self.anchor_class_embeds is not None:
                # if we have an anchor, do it
                combo_embeds = concat_prompt_embeds(
                    [
                        positive_embeds,
                        target_class_embeds,
                        negative_embeds,
                        anchor_embeds,
                    ]
                )
                num_embeds = 4
            else:
                combo_embeds = concat_prompt_embeds(
                    [positive_embeds, target_class_embeds, negative_embeds]
                )
                num_embeds = 3

            # do them in one batch, VRAM should handle it since we are no grad
            combo_pred = self.sd.predict_noise(
                latents=torch.cat([noisy_latents] * num_embeds, dim=0),
                conditional_embeddings=combo_embeds,
                timestep=torch.cat([timesteps] * num_embeds, dim=0),
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            )

            if self.anchor_class_embeds is not None:
                positive_pred, neutral_pred, negative_pred, anchor_target = (
                    combo_pred.chunk(4, dim=0)
                )
            else:
                anchor_target = None
                positive_pred, neutral_pred, negative_pred = combo_pred.chunk(3, dim=0)

            # calculate the targets
            guidance_scale = self.slider.guidance_strength

            # enhance_positive_target = neutral_pred + guidance_scale * (
            #     positive_pred - negative_pred
            # )
            # enhance_negative_target = neutral_pred + guidance_scale * (
            #     negative_pred - positive_pred
            # )
            # erase_negative_target = neutral_pred - guidance_scale * (
            #     negative_pred - positive_pred
            # )
            # erase_positive_target = neutral_pred - guidance_scale * (
            #     positive_pred - negative_pred
            # )
            
            positive = (positive_pred - neutral_pred) - (negative_pred - neutral_pred)
            negative = (negative_pred - neutral_pred) - (positive_pred - neutral_pred)

            enhance_positive_target = neutral_pred + guidance_scale * positive
            enhance_negative_target = neutral_pred + guidance_scale * negative
            erase_negative_target = neutral_pred - guidance_scale * negative
            erase_positive_target = neutral_pred - guidance_scale * positive

            if was_unet_training:
                self.sd.unet.train()

            # restore network
            if self.network is not None:
                self.network.is_active = was_network_active

            if self.anchor_class_embeds is not None:
                # do a grad inference with our target prompt
                embeds = concat_prompt_embeds([target_class_embeds, anchor_embeds]).to(
                    self.device_torch, dtype=dtype
                )

                noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0).to(
                    self.device_torch, dtype=dtype
                )
                timesteps = torch.cat([timesteps, timesteps], dim=0)
            else:
                embeds = target_class_embeds.to(self.device_torch, dtype=dtype)

        # do positive first
        self.network.set_multiplier(1.0)
        pred = self.sd.predict_noise(
            latents=noisy_latents,
            conditional_embeddings=embeds,
            timestep=timesteps,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch,
        )

        if self.anchor_class_embeds is not None:
            class_pred, anchor_pred = pred.chunk(2, dim=0)
        else:
            class_pred = pred
            anchor_pred = None

        # enhance positive loss
        enhance_loss = torch.nn.functional.mse_loss(class_pred, enhance_positive_target)

        erase_loss = torch.nn.functional.mse_loss(class_pred, erase_negative_target)

        if anchor_target is None:
            anchor_loss = torch.zeros_like(erase_loss)
        else:
            anchor_loss = torch.nn.functional.mse_loss(anchor_pred, anchor_target)
        
        anchor_loss = anchor_loss * self.slider.anchor_strength

        # send backward now because gradient checkpointing needs network polarity intact
        total_pos_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0
        total_pos_loss.backward()
        total_pos_loss = total_pos_loss.detach()

        # now do negative
        self.network.set_multiplier(-1.0)
        pred = self.sd.predict_noise(
            latents=noisy_latents,
            conditional_embeddings=embeds,
            timestep=timesteps,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch,
        )

        if self.anchor_class_embeds is not None:
            class_pred, anchor_pred = pred.chunk(2, dim=0)
        else:
            class_pred = pred
            anchor_pred = None

        # enhance negative loss
        enhance_loss = torch.nn.functional.mse_loss(class_pred, enhance_negative_target)
        erase_loss = torch.nn.functional.mse_loss(class_pred, erase_positive_target)

        if anchor_target is None:
            anchor_loss = torch.zeros_like(erase_loss)
        else:
            anchor_loss = torch.nn.functional.mse_loss(anchor_pred, anchor_target)
        anchor_loss = anchor_loss * self.slider.anchor_strength
        total_neg_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0
        total_neg_loss.backward()
        total_neg_loss = total_neg_loss.detach()

        self.network.set_multiplier(1.0)

        total_loss = (total_pos_loss + total_neg_loss) / 2.0

        # add a grad so backward works right
        total_loss.requires_grad_(True)
        return total_loss
