import os
from typing import List, Optional

import huggingface_hub
import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager
from safetensors.torch import load_file
from optimum.quanto import QTensor
from toolkit.metadata import get_meta_for_safetensors
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3ForCausalLM
from diffusers import AutoencoderKL
from toolkit.models.FakeVAE import FakeVAE
from .zeta_chroma_transformer import ZImageDCT, ZImageDCTParams, vae_flatten, vae_unflatten, prepare_latent_image_ids, make_text_position_ids
from .zeta_chroma_pipeline import ZetaChromaPipeline



scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}

ZETA_CHROMA_TRANSFORMER_FILENAME = "zeta-chroma-base-x0-pixel-dino-distance.safetensors"


class ZetaChromaModel(BaseModel):
    arch = "zeta_chroma"

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
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["ZImageDCT"]
        self.patch_size = 32
        self.max_sequence_length = 512

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return self.patch_size

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading ZImage model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        self.print_and_status_update("Loading transformer")


        transformer_path = model_path
        if not os.path.exists(transformer_path):
            transformer_name = ZETA_CHROMA_TRANSFORMER_FILENAME
            # if path ends with .safetensors, assume last part is filename
            # this allows users to target different file names in the repo like
            # lodestones/Zeta-Chroma/zeta-chroma-base-x0-pixel-dino-distance.safetensors

            if transformer_path.endswith(".safetensors"):
                splits = transformer_path.split("/")
                transformer_name = splits[-1]
                transformer_path = "/".join(splits[:-1])
            # assume it is from the hub
            transformer_path = huggingface_hub.hf_hub_download(
                repo_id=transformer_path,
                filename=transformer_name,
            )

        transformer_state_dict = load_file(transformer_path, device="cpu")

        # cast to dtype
        for key in transformer_state_dict:
            transformer_state_dict[key] = transformer_state_dict[key].to(dtype)
        
        # Auto-detect use_x0 from checkpoint
        use_x0 = "__x0__" in transformer_state_dict

        # Build model params
        in_channels = self.patch_size * self.patch_size * 3  # RGB patches
        model_params = ZImageDCTParams(
            patch_size=1,
            in_channels=in_channels,
            use_x0=use_x0,
        )

        with torch.device("meta"):
            transformer = ZImageDCT(model_params)
            
        transformer.load_state_dict(transformer_state_dict, assign=True)
        del transformer_state_dict

        transformer.to(self.quantize_device, dtype=dtype)

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[
                    transformer.x_pad_token,
                    transformer.cap_pad_token,
                ],
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype
        )

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading VAE")
        vae = FakeVAE(scaling_factor=1.0)
        vae.to(self.device_torch, dtype=dtype)

        self.noise_scheduler = ZetaChromaModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {}

        pipe: ZetaChromaPipeline = ZetaChromaPipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        # leave it on cpu for now
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # just to make sure everything is on the right device and dtype
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = ZetaChromaModel.get_train_scheduler()

        pipeline: ZetaChromaPipeline = ZetaChromaPipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: ZetaChromaPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            prompt_embeds_mask=conditional_embeds.attention_mask,
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            negative_prompt_embeds_mask=unconditional_embeds.attention_mask,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        
        with torch.no_grad():
            
            pixel_shape = latent_model_input.shape
            # todo: do we invert like this?
            # t_vec = (1000 - timestep) / 1000
            t_vec = timestep / 1000
            
            height = latent_model_input.shape[2]
            h_patches = height // self.patch_size
            width = latent_model_input.shape[3]
            w_patches = width // self.patch_size
            batch_size = latent_model_input.shape[0]
            
            img, _ = vae_flatten(latent_model_input, patch_size=self.patch_size)
            
            num_patches = img.shape[1]
            
            # --- Build position IDs ---
            pos_lengths = text_embeddings.attention_mask.sum(1)
            offset = pos_lengths

            image_pos_ids = prepare_latent_image_ids(
                offset, h_patches, w_patches, patch_size=1
            ).to(self.device_torch)
            pos_text_ids = make_text_position_ids(pos_lengths, self.max_sequence_length).to(
                self.device_torch
            )
            img_mask = torch.ones(
                (batch_size, num_patches), device=self.device_torch, dtype=torch.bool
            )
            
            

        # model_out_list = self.transformer(
        #     latent_model_input_list,
        #     t_vec,
        #     text_embeddings.text_embeds,
        # )[0]
        pred = self.transformer(
            img=img, #(1, 1024, 3072)
            img_ids=image_pos_ids, # (1, 1024, 3)
            img_mask=img_mask, # (1, 1024)
            txt=text_embeddings.text_embeds, # (1, 512, 2560)
            txt_ids=pos_text_ids, # (1, 512, 3)
            txt_mask=text_embeddings.attention_mask, # (1, 512)
            timesteps=t_vec, # (1,)
        )
        
        pred = vae_unflatten(pred.float(), pixel_shape, patch_size=self.patch_size)

        return pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, mask = self.pipeline._encode_prompts(
            prompt,
        )
        pe = PromptEmbeds([prompt_embeds, None], attention_mask=mask)
        
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        # only save the unet
        transformer: ZImageDCT = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, QTensor):
                v = v.dequantize()
            save_dict[k] = v.clone().to("cpu", dtype=save_dtype)

        meta = get_meta_for_safetensors(meta, name="zeta_chroma")
        save_file(save_dict, output_path, metadata=meta)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "zeta_chroma"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd
