from functools import partial
import torch
from toolkit.prompt_utils import PromptEmbeds
from PIL import Image
from diffusers import UniPCMultistepScheduler
import torch
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from .wan22_pipeline import Wan22Pipeline

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from torchvision.transforms import functional as TF

from toolkit.models.wan21.wan21 import Wan21, AggressiveWanUnloadPipeline
from toolkit.models.wan21.wan_utils import add_first_frame_conditioning_v22


# for generation only?
scheduler_configUniPC = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.35.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 5.0,
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

# for training. I think it is right
scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 5.0,
    "use_dynamic_shifting": False,
}

# TODO: this is a temporary monkeypatch to fix the time text embedding to allow for batch sizes greater than 1. Remove this when the diffusers library is fixed.
def time_text_monkeypatch(
    self,
    timestep: torch.Tensor,
    encoder_hidden_states,
    encoder_hidden_states_image = None,
    timestep_seq_len = None,
):
    timestep = self.timesteps_proj(timestep)
    if timestep_seq_len is not None:
        timestep = timestep.unflatten(0, (encoder_hidden_states.shape[0], timestep_seq_len))

    time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
    if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
        timestep = timestep.to(time_embedder_dtype)
    temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
    timestep_proj = self.time_proj(self.act_fn(temb))

    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    if encoder_hidden_states_image is not None:
        encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image

class Wan225bModel(Wan21):
    arch = "wan22_5b"
    _wan_generation_scheduler_config = scheduler_configUniPC
    _wan_expand_timesteps = True

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

        self._wan_cache = None
    
    def load_model(self):
        super().load_model()
        
        # patch the condition embedder
        self.model.condition_embedder.forward = partial(time_text_monkeypatch, self.model.condition_embedder)

    def get_bucket_divisibility(self):
        # 16x compression  and 2x2 patch size
        return 32

    def get_generation_pipeline(self):
        scheduler = UniPCMultistepScheduler(**self._wan_generation_scheduler_config)
        pipeline = Wan22Pipeline(
            vae=self.vae,
            transformer=self.model,
            transformer_2=self.model,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=scheduler,
            expand_timesteps=self._wan_expand_timesteps,
            device=self.device_torch,
            aggressive_offload=self.model_config.low_vram,
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler():
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler

    def get_base_model_version(self):
        return "wan_2.2_5b"

    def generate_single_image(
        self,
        pipeline: AggressiveWanUnloadPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)

        num_frames = (
            (gen_config.num_frames - 1) // 4
        ) * 4 + 1  # make sure it is divisible by 4 + 1
        gen_config.num_frames = num_frames

        height = gen_config.height
        width = gen_config.width
        noise_mask = None
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img).convert("RGB")

            d = self.get_bucket_divisibility()

            # make sure they are divisible by d
            height = height // d * d
            width = width // d * d

            # resize the control image
            control_img = control_img.resize((width, height), Image.LANCZOS)

            # 5. Prepare latent variables
            num_channels_latents = self.transformer.config.in_channels
            latents = pipeline.prepare_latents(
                1,
                num_channels_latents,
                height,
                width,
                gen_config.num_frames,
                torch.float32,
                self.device_torch,
                generator,
                None,
            ).to(self.torch_dtype)

            first_frame_n1p1 = (
                TF.to_tensor(control_img)
                .unsqueeze(0)
                .to(self.device_torch, dtype=self.torch_dtype)
                * 2.0
                - 1.0
            )  # normalize to [-1, 1]

            gen_config.latents, noise_mask = add_first_frame_conditioning_v22(
                latent_model_input=latents, first_frame=first_frame_n1p1, vae=self.vae
            )

        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            height=height,
            width=width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            noise_mask=noise_mask,
            **extra,
        )[0]

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs,
    ):
        # videos come in (bs, num_frames, channels, height, width)
        # images come in (bs, channels, height, width)

        # for wan, only do i2v for video for now. Images do normal t2i
        conditioned_latent = latent_model_input
        noise_mask = None
        
        if batch.dataset_config.do_i2v:
            with torch.no_grad():
                frames = batch.tensor
                if len(frames.shape) == 4:
                    first_frames = frames
                elif len(frames.shape) == 5:
                    first_frames = frames[:, 0]
                    # Add conditioning using the standalone function
                    conditioned_latent, noise_mask = add_first_frame_conditioning_v22(
                        latent_model_input=latent_model_input.to(
                            self.device_torch, self.torch_dtype
                        ),
                        first_frame=first_frames.to(self.device_torch, self.torch_dtype),
                        vae=self.vae,
                    )
                else:
                    raise ValueError(f"Unknown frame shape {frames.shape}")

                # make the noise mask
                if noise_mask is None:
                    noise_mask = torch.ones(
                        conditioned_latent.shape,
                        dtype=conditioned_latent.dtype,
                        device=conditioned_latent.device,
                    )
                # todo write this better
                t_chunks = torch.chunk(timestep, timestep.shape[0])
                out_t_chunks = []
                for t in t_chunks:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (noise_mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    temp_ts = temp_ts.unsqueeze(0)
                    out_t_chunks.append(temp_ts)
                timestep = torch.cat(out_t_chunks, dim=0)

        noise_pred = self.model(
            hidden_states=conditioned_latent,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs,
        )[0]
        return noise_pred
