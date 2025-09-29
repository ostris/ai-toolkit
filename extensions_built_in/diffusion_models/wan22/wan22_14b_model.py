from functools import partial
import os
from typing import Any, Dict, Optional, Union, List
from typing_extensions import Self
import torch
import yaml
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.models.wan21.wan_utils import add_first_frame_conditioning
from toolkit.prompt_utils import PromptEmbeds
from PIL import Image
from diffusers import UniPCMultistepScheduler
import torch
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import quantize_model
from .wan22_pipeline import Wan22Pipeline
from diffusers import WanTransformer3DModel

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from torchvision.transforms import functional as TF

from toolkit.models.wan21.wan21 import Wan21
from .wan22_5b_model import (
    scheduler_config,
    time_text_monkeypatch,
)
from safetensors.torch import load_file, save_file


boundary_ratio_t2v = 0.875
boundary_ratio_i2v = 0.9

scheduler_configUniPC = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.35.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}


class DualWanTransformer3DModel(torch.nn.Module):
    def __init__(
        self,
        transformer_1: WanTransformer3DModel,
        transformer_2: WanTransformer3DModel,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        boundary_ratio: float = boundary_ratio_t2v,
        low_vram: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_1: WanTransformer3DModel = transformer_1
        self.transformer_2: WanTransformer3DModel = transformer_2
        self.torch_dtype: torch.dtype = torch_dtype
        self.device_torch: torch.device = device
        self.boundary_ratio: float = boundary_ratio
        self.boundary: float = self.boundary_ratio * 1000
        self.low_vram: bool = low_vram
        self._active_transformer_name = "transformer_1"  # default to transformer_1

    @property
    def device(self) -> torch.device:
        return self.device_torch

    @property
    def dtype(self) -> torch.dtype:
        return self.torch_dtype

    @property
    def config(self):
        return self.transformer_1.config

    @property
    def transformer(self) -> WanTransformer3DModel:
        return getattr(self, self._active_transformer_name)

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for both transformers.
        """
        self.transformer_1.enable_gradient_checkpointing()
        self.transformer_2.enable_gradient_checkpointing()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # determine if doing high noise or low noise by meaning the timestep.
        # timesteps are in the range of 0 to 1000, so we can use a threshold
        with torch.no_grad():
            if timestep.float().mean().item() > self.boundary:
                t_name = "transformer_1"
            else:
                t_name = "transformer_2"

            # check if we are changing the active transformer, if so, we need to swap the one in
            # vram if low_vram is enabled
            # todo swap the loras as well
            if t_name != self._active_transformer_name:
                if self.low_vram:
                    getattr(self, self._active_transformer_name).to("cpu")
                    getattr(self, t_name).to(self.device_torch)
                    torch.cuda.empty_cache()
                self._active_transformer_name = t_name

        if self.transformer.device != hidden_states.device:
            if self.low_vram:
                # move other transformer to cpu
                other_tname = (
                    "transformer_1" if t_name == "transformer_2" else "transformer_2"
                )
                getattr(self, other_tname).to("cpu")

            self.transformer.to(hidden_states.device)

        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
        )

    def to(self, *args, **kwargs) -> Self:
        # do not do to, this will be handled separately
        return self


class Wan2214bModel(Wan21):
    arch = "wan22_14b"
    _wan_generation_scheduler_config = scheduler_configUniPC
    _wan_expand_timesteps = False
    _wan_vae_path = "ai-toolkit/wan2.1-vae"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            model_config=model_config,
            dtype=dtype,
            custom_pipeline=custom_pipeline,
            noise_scheduler=noise_scheduler,
            **kwargs,
        )
        # target it so we can target both transformers
        self.target_lora_modules = ["DualWanTransformer3DModel"]
        self._wan_cache = None

        self.is_multistage = True
        # multistage boundaries split the models up when sampling timesteps
        # for wan 2.2 14b. the timesteps are 1000-875 for transformer 1 and 875-0 for transformer 2
        self.multistage_boundaries: List[float] = [0.875, 0.0]

        self.train_high_noise = model_config.model_kwargs.get("train_high_noise", True)
        self.train_low_noise = model_config.model_kwargs.get("train_low_noise", True)

        self.trainable_multistage_boundaries: List[int] = []
        if self.train_high_noise:
            self.trainable_multistage_boundaries.append(0)
        if self.train_low_noise:
            self.trainable_multistage_boundaries.append(1)

        if len(self.trainable_multistage_boundaries) == 0:
            raise ValueError(
                "At least one of train_high_noise or train_low_noise must be True in model.model_kwargs"
            )
        
        # if we are only training one or the other, the target LoRA modules will be the wan transformer class
        if not self.train_high_noise or not self.train_low_noise:
            self.target_lora_modules = ["WanTransformer3DModel"]

    @property
    def max_step_saves_to_keep_multiplier(self):
        # the cleanup mechanism checks this to see how many saves to keep
        # if we are training a LoRA, we need to set this to 2 so we keep both the high noise and low noise LoRAs at saves to keep
        if (
            self.network is not None
            and self.network.network_config.split_multistage_loras
        ):
            return 2
        return 1

    def load_model(self):
        # load model from patent parent. Wan21 not immediate parent
        # super().load_model()
        super().load_model()

        # we have to split up the model on the pipeline
        self.pipeline.transformer = self.model.transformer_1
        self.pipeline.transformer_2 = self.model.transformer_2

        # patch the condition embedder
        self.model.transformer_1.condition_embedder.forward = partial(
            time_text_monkeypatch, self.model.transformer_1.condition_embedder
        )
        self.model.transformer_2.condition_embedder.forward = partial(
            time_text_monkeypatch, self.model.transformer_2.condition_embedder
        )

    def get_bucket_divisibility(self):
        # 8x compression  and 2x2 patch size
        return 16

    def load_wan_transformer(self, transformer_path, subfolder=None):
        if self.model_config.split_model_over_gpus:
            raise ValueError(
                "Splitting model over gpus is not supported for Wan2.2 models"
            )

        if (
            self.model_config.assistant_lora_path is not None
            or self.model_config.inference_lora_path is not None
        ):
            raise ValueError(
                "Assistant LoRA is not supported for Wan2.2 models currently"
            )

        if self.model_config.lora_path is not None:
            raise ValueError(
                "Loading LoRA is not supported for Wan2.2 models currently"
            )

        # transformer path can be a directory that ends with /transformer or a hf path.

        transformer_path_1 = transformer_path
        subfolder_1 = subfolder

        transformer_path_2 = transformer_path
        subfolder_2 = subfolder

        if subfolder_2 is None:
            # we have a local path, replace it with transformer_2 folder
            transformer_path_2 = os.path.join(
                os.path.dirname(transformer_path_1), "transformer_2"
            )
        else:
            # we have a hf path, replace it with transformer_2 subfolder
            subfolder_2 = "transformer_2"

        self.print_and_status_update("Loading transformer 1")
        dtype = self.torch_dtype
        transformer_1 = WanTransformer3DModel.from_pretrained(
            transformer_path_1,
            subfolder=subfolder_1,
            torch_dtype=dtype,
        ).to(dtype=dtype)

        flush()

        if not self.model_config.low_vram:
            # quantize on the device
            transformer_1.to(self.quantize_device, dtype=dtype)
            flush()

        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            # todo handle two ARAs
            self.print_and_status_update("Quantizing Transformer 1")
            quantize_model(self, transformer_1)
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer 1 to CPU")
            transformer_1.to("cpu")

        self.print_and_status_update("Loading transformer 2")
        dtype = self.torch_dtype
        transformer_2 = WanTransformer3DModel.from_pretrained(
            transformer_path_2,
            subfolder=subfolder_2,
            torch_dtype=dtype,
        ).to(dtype=dtype)

        flush()

        if not self.model_config.low_vram:
            # quantize on the device
            transformer_2.to(self.quantize_device, dtype=dtype)
            flush()

        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            # todo handle two ARAs
            self.print_and_status_update("Quantizing Transformer 2")
            quantize_model(self, transformer_2)
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer 2 to CPU")
            transformer_2.to("cpu")

        # make the combined model
        self.print_and_status_update("Creating DualWanTransformer3DModel")
        transformer = DualWanTransformer3DModel(
            transformer_1=transformer_1,
            transformer_2=transformer_2,
            torch_dtype=self.torch_dtype,
            device=self.device_torch,
            boundary_ratio=boundary_ratio_t2v,
            low_vram=self.model_config.low_vram,
        )
        
        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is not None:
            # apply the accuracy recovery adapter to both transformers
            self.print_and_status_update("Applying Accuracy Recovery Adapter to Transformers")
            quantize_model(self, transformer)
            flush()
            
        return transformer

    def get_generation_pipeline(self):
        scheduler = UniPCMultistepScheduler(**self._wan_generation_scheduler_config)
        pipeline = Wan22Pipeline(
            vae=self.vae,
            transformer=self.model.transformer_1,
            transformer_2=self.model.transformer_2,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=scheduler,
            expand_timesteps=self._wan_expand_timesteps,
            device=self.device_torch,
            aggressive_offload=self.model_config.low_vram,
            # todo detect if it is i2v or t2v
            boundary_ratio=boundary_ratio_t2v,
        )

        # pipeline = pipeline.to(self.device_torch)

        return pipeline

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler():
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler

    def get_base_model_version(self):
        return "wan_2.2_14b"

    def generate_single_image(
        self,
        pipeline: Wan22Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        return super().generate_single_image(
            pipeline=pipeline,
            gen_config=gen_config,
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            generator=generator,
            extra=extra,
        )

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs,
    ):
        # todo do we need to override this? Adjust timesteps?
        return super().get_noise_prediction(
            latent_model_input=latent_model_input,
            timestep=timestep,
            text_embeddings=text_embeddings,
            batch=batch,
            **kwargs,
        )

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer_combo: DualWanTransformer3DModel = unwrap_model(self.model)
        transformer_combo.transformer_1.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        transformer_combo.transformer_2.save_pretrained(
            save_directory=os.path.join(output_path, "transformer_2"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def save_lora(
        self,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.network.network_config.split_multistage_loras:
            # just save as a combo lora
            save_file(state_dict, output_path, metadata=metadata)
            return

        # we need to build out both dictionaries for high and low noise LoRAs
        high_noise_lora = {}
        low_noise_lora = {}
        
        only_train_high_noise = self.train_high_noise and not self.train_low_noise
        only_train_low_noise = self.train_low_noise and not self.train_high_noise

        for key in state_dict:
            if ".transformer_1." in key or only_train_high_noise:
                # this is a high noise LoRA
                new_key = key.replace(".transformer_1.", ".")
                high_noise_lora[new_key] = state_dict[key]
            elif ".transformer_2." in key or only_train_low_noise:
                # this is a low noise LoRA
                new_key = key.replace(".transformer_2.", ".")
                low_noise_lora[new_key] = state_dict[key]

        # loras have either LORA_MODEL_NAME_000005000.safetensors or LORA_MODEL_NAME.safetensors
        if len(high_noise_lora.keys()) > 0:
            # save the high noise LoRA
            high_noise_lora_path = output_path.replace(
                ".safetensors", "_high_noise.safetensors"
            )
            save_file(high_noise_lora, high_noise_lora_path, metadata=metadata)

        if len(low_noise_lora.keys()) > 0:
            # save the low noise LoRA
            low_noise_lora_path = output_path.replace(
                ".safetensors", "_low_noise.safetensors"
            )
            save_file(low_noise_lora, low_noise_lora_path, metadata=metadata)

    def load_lora(self, file: str):
        # if it doesnt have high_noise or low_noise, it is a combo LoRA
        if (
            "_high_noise.safetensors" not in file
            and "_low_noise.safetensors" not in file
        ):
            # this is a combined LoRA, we dont need to split it up
            sd = load_file(file)
            return sd

        # we may have been passed the high_noise or the low_noise LoRA path, but we need to load both
        high_noise_lora_path = file.replace(
            "_low_noise.safetensors", "_high_noise.safetensors"
        )
        low_noise_lora_path = file.replace(
            "_high_noise.safetensors", "_low_noise.safetensors"
        )

        combined_dict = {}

        if os.path.exists(high_noise_lora_path) and self.train_high_noise:
            # load the high noise LoRA
            high_noise_lora = load_file(high_noise_lora_path)
            for key in high_noise_lora:
                new_key = key.replace(
                    "diffusion_model.", "diffusion_model.transformer_1."
                )
                combined_dict[new_key] = high_noise_lora[key]
        if os.path.exists(low_noise_lora_path) and self.train_low_noise:
            # load the low noise LoRA
            low_noise_lora = load_file(low_noise_lora_path)
            for key in low_noise_lora:
                new_key = key.replace(
                    "diffusion_model.", "diffusion_model.transformer_2."
                )
                combined_dict[new_key] = low_noise_lora[key]
        
        # if we are not training both stages, we wont have transformer designations in the keys
        if not self.train_high_noise or not self.train_low_noise:
            new_dict = {}
            for key in combined_dict:
                if ".transformer_1." in key:
                    new_key = key.replace(".transformer_1.", ".")
                elif ".transformer_2." in key:
                    new_key = key.replace(".transformer_2.", ".")
                else:
                    new_key = key
                new_dict[new_key] = combined_dict[key]
            combined_dict = new_dict

        return combined_dict
    
    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)
        # todo, figure out how to do video
        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra
        )[0]

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img

    def get_model_to_train(self):
        # todo, loras wont load right unless they have the transformer_1 or transformer_2 in the key.
        # called when setting up the LoRA. We only need to get the model for the stages we want to train.
        if self.train_high_noise and self.train_low_noise:
            # we are training both stages, return the unified model
            return self.model
        elif self.train_high_noise:
            # we are only training the high noise stage, return transformer_1
            return self.model.transformer_1
        elif self.train_low_noise:
            # we are only training the low noise stage, return transformer_2
            return self.model.transformer_2
        else:
            raise ValueError(
                "At least one of train_high_noise or train_low_noise must be True in model.model_kwargs"
            )
