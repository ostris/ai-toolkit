import os
from typing import TYPE_CHECKING

import torch
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.text_encoders.t5 import T5TextEncoder
from toolkit.models.v2.vae.autoencoder_kl import KLVAE
from toolkit.basic import flush
# from toolkit.pixel_shuffle_encoder import AutoencoderPixelMixer
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.accelerator import unwrap_model
from optimum.quanto import QTensor
from .pipeline import ChromaPipeline, prepare_latent_image_ids
from einops import rearrange, repeat
import random
import torch.nn.functional as F
from .src.model import Chroma, chroma_params
from safetensors.torch import load_file, save_file
from toolkit.metadata import get_meta_for_safetensors
import huggingface_hub

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True
}

class FakeConfig:
    # for diffusers compatability
    def __init__(self):
        self.attention_head_dim = 128
        self.guidance_embeds = True
        self.in_channels = 64
        self.joint_attention_dim = 4096
        self.num_attention_heads = 24
        self.num_layers = 19
        self.num_single_layers = 38
        self.patch_size = 1
        
class FakeCLIP(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.dtype = torch.bfloat16
        # the pipeline derives its execution device from this attribute;
        # nn.Module.to() does not update it
        self.device = device
        self.text_model = None
        self.tokenizer = None
        self.model_max_length = 77

    def forward(self, *args, **kwargs):
        return torch.zeros(1, 1, 1).to(self.device)


class ChromaModel(BaseModel):
    arch = "chroma"

    def get_transformer_block_names(self):
        return ["double_blocks", "single_blocks"]

    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['Chroma']

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
    
    def get_bucket_divisibility(self):
        # return the bucket divisibility for the model
        return 32

    def load_model(self):
        dtype = self.torch_dtype
        
        # will be updated if we detect a existing checkpoint in training folder
        model_path = self.model_config.name_or_path
        
        if model_path == "lodestones/Chroma":
            print("Looking for latest Chroma checkpoint")
            # get the latest checkpoint
            files_list = huggingface_hub.list_repo_files(model_path)
            print(files_list)
            latest_version = 28 # current latest version at time of writing
            while True:
                if f"chroma-unlocked-v{latest_version}.safetensors" not in files_list:
                    latest_version -= 1
                    break
                else:
                    latest_version += 1
            print(f"Using latest Chroma version: v{latest_version}")
            
            # make sure we have it
            model_path = huggingface_hub.hf_hub_download(
                repo_id=model_path,
                filename=f"chroma-unlocked-v{latest_version}.safetensors",
            )
        elif model_path.startswith("lodestones/Chroma/v"):
            # get the version number
            version = model_path.split("/")[-1].split("v")[-1]
            print(f"Using Chroma version: v{version}")
            # make sure we have it
            model_path = huggingface_hub.hf_hub_download(
                repo_id='lodestones/Chroma',
                filename=f"chroma-unlocked-v{version}.safetensors",
            )
        elif model_path.startswith("lodestones/Chroma1-"):
            # will have a file in the repo that is Chroma1-whatever.safetensors
            model_path = huggingface_hub.hf_hub_download(
                repo_id=model_path,
                filename=f"{model_path.split('/')[-1]}.safetensors",
            )
        else:
            # check if the model path is a local file
            if os.path.exists(model_path):
                print(f"Using local model: {model_path}")
            else:
                raise ValueError(f"Model path {model_path} does not exist")
        
        # extras_path = 'black-forest-labs/FLUX.1-schnell'
        # schnell model is gated now, use flex instead
        extras_path = 'ostris/Flex.1-alpha'

        self.print_and_status_update("Loading transformer")
        
        if model_path.endswith(".safetensors"):
            transformer = Chroma.load_model(model_path, dtype=dtype)
        else:
            transformer = Chroma.load_from_state_dict(load_file(model_path, "cpu"), dtype)
        # add dtype, not sure why it doesnt have it
        transformer.dtype = dtype

        transformer.config = FakeConfig()
        transformer.config.num_layers = transformer.params.depth
        transformer.config.num_single_layers = transformer.params.depth_single_blocks

        # quantize + offload + placement, all driven by model_config
        transformer.aitk_post_load(**self.component_load_kwargs("transformer"))

        flush()

        self.print_and_status_update("Loading T5")
        tokenizer_2 = T5TextEncoder.load_tokenizer(extras_path)
        text_encoder_2 = T5TextEncoder.load(
            extras_path, **self.component_load_kwargs("te")
        )

        # self.print_and_status_update("Loading CLIP")
        text_encoder = FakeCLIP(device=self.device_torch)
        tokenizer = FakeCLIP(device=self.device_torch)
        text_encoder.to(self.device_torch, dtype=dtype)

        self.noise_scheduler = ChromaModel.get_train_scheduler()
        
        self.print_and_status_update("Loading VAE")
        vae = KLVAE.load_model(extras_path, dtype=dtype, device=self.device_torch)

        self.print_and_status_update("Making pipe")

        pipe: ChromaPipeline = ChromaPipeline(
            scheduler=self.noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder_2 = text_encoder_2
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder, pipe.text_encoder_2]
        tokenizer = [pipe.tokenizer, pipe.tokenizer_2]

        pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # low_vram: text encoders stay on cpu; get_prompt_embeds moves them
        # to the gpu on demand
        if not self.low_vram:
            text_encoder[0].to(self.device_torch)
            text_encoder[1].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        text_encoder[1].requires_grad_(False)
        text_encoder[1].eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = ChromaModel.get_train_scheduler()
        pipeline = ChromaPipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            text_encoder_2=unwrap_model(self.text_encoder[1]),
            tokenizer_2=self.tokenizer[1],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer)
        )

        # pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: ChromaPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):

        extra['negative_prompt_embeds'] = unconditional_embeds.text_embeds
        extra['negative_prompt_attn_mask'] = unconditional_embeds.attention_mask
        
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            prompt_attn_mask=conditional_embeds.attention_mask,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs
    ):
        with torch.no_grad():
            bs, c, h, w = latent_model_input.shape
            latent_model_input_packed = rearrange(
                latent_model_input,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=2,
                pw=2
            )
            
            img_ids = prepare_latent_image_ids(
                bs, 
                h, 
                w,
                patch_size=2
            ).to(device=self.device_torch)

            # img_ids = torch.zeros(h // 2, w // 2, 3)
            # img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
            # img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
            # img_ids = repeat(img_ids, "h w c -> b (h w) c",
            #                  b=bs).to(self.device_torch)

            txt_ids = torch.zeros(
                bs, text_embeddings.text_embeds.shape[1], 3).to(self.device_torch)

        guidance = torch.full([1], 0, device=self.device_torch, dtype=torch.float32)
        guidance = guidance.expand(latent_model_input_packed.shape[0])

        cast_dtype = self.unet.dtype

        noise_pred = self.unet(
            img=latent_model_input_packed.to(
                self.device_torch, cast_dtype
            ),
            img_ids=img_ids,
            txt=text_embeddings.text_embeds.to(
                self.device_torch, cast_dtype
            ),
            txt_ids=txt_ids,
            txt_mask=text_embeddings.attention_mask.to(
                self.device_torch, cast_dtype
            ),
            timesteps=timestep / 1000,
            guidance=guidance
        )

        if isinstance(noise_pred, QTensor):
            noise_pred = noise_pred.dequantize()

        noise_pred = rearrange(
            noise_pred,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=latent_model_input.shape[2] // 2,
            w=latent_model_input.shape[3] // 2,
            ph=2,
            pw=2,
            c=self.vae.config.latent_channels
        )
        
        return noise_pred
    
    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        max_length = 512

        device = self.text_encoder[1].device
        dtype = self.text_encoder[1].dtype

        # T5
        text_inputs = self.tokenizer[1](
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder[1](text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder[1].dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        prompt_attention_mask = text_inputs["attention_mask"]
        
        pe = PromptEmbeds(
            prompt_embeds
        )
        pe.attention_mask = prompt_attention_mask
        return pe
    
    def get_model_has_grad(self):
        # return from a weight if it has grad
        return self.model.final_layer.linear.weight.requires_grad

    def get_te_has_grad(self):
        # return from a weight if it has grad
        return self.text_encoder[1].encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad
    
    def save_model(self, output_path, meta, save_dtype):
        # comfy-format single-file save via the mixin (chroma's class keys ARE
        # the original layout); handles torchao/Ostris dequant, not just quanto
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: Chroma = unwrap_model(self.model)
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name="chroma"),
        )

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        return (noise - batch.latents).detach()
    
    lora_keys_use_comfy_prefix = True

    
    def get_base_model_version(self):
        return "chroma"
