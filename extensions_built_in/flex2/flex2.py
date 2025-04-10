import os
from typing import TYPE_CHECKING, List

import torch
import torchvision
import yaml
from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from diffusers import FluxTransformer2DModel, AutoencoderKL
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.models.flux import add_model_gpu_splitter_to_flux, bypass_flux_guidance, restore_flux_guidance
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.accelerator import get_accelerator, unwrap_model
from optimum.quanto import freeze, QTensor
from toolkit.util.mask import generate_random_mask, random_dialate_mask
from toolkit.util.quantize import quantize, get_qtype
from transformers import T5TokenizerFast, T5EncoderModel, CLIPTextModel, CLIPTokenizer
from .pipeline import Flex2Pipeline
from einops import rearrange, repeat
import random
import torch.nn.functional as F

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


def random_blur(img, min_kernel_size=3, max_kernel_size=23, p=0.5):
    if random.random() < p:
        kernel_size = random.randint(min_kernel_size, max_kernel_size)
        # make sure it is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        img = torchvision.transforms.functional.gaussian_blur(img, kernel_size=kernel_size)
    return img

class Flex2(BaseModel):
    arch = "flex2"

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
        self.target_lora_modules = ['FluxTransformer2DModel']
        
        # for training, pass these as kwargs
        self.invert_inpaint_mask_chance = model_config.model_kwargs.get('invert_inpaint_mask_chance', 0.0)
        self.inpaint_dropout = model_config.model_kwargs.get('inpaint_dropout', 0.0)
        self.control_dropout = model_config.model_kwargs.get('control_dropout', 0.0)
        self.inpaint_random_chance = model_config.model_kwargs.get('inpaint_random_chance', 0.0)
        self.random_blur_mask = model_config.model_kwargs.get('random_blur_mask', False)
        self.random_dialate_mask = model_config.model_kwargs.get('random_dialate_mask', False)
        self.do_random_inpainting = model_config.model_kwargs.get('do_random_inpainting', False)

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Flux2 model")
        # will be updated if we detect a existing checkpoint in training folder
        model_path = self.model_config.name_or_path
        # this is the original path put in the model directory
        # it is here because for finetuning we only save the transformer usually
        # so we need this for the VAE, te, etc
        base_model_path = self.model_config.name_or_path_original

        transformer_path = model_path
        transformer_subfolder = 'transformer'
        if os.path.exists(transformer_path):
            transformer_subfolder = None
            transformer_path = os.path.join(transformer_path, 'transformer')
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, 'text_encoder')
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        self.print_and_status_update("Loading transformer")
        transformer = FluxTransformer2DModel.from_pretrained(
            transformer_path,
            subfolder=transformer_subfolder,
            torch_dtype=dtype,
        )
        transformer.to(self.quantize_device, dtype=dtype)

        if self.model_config.quantize:
            # patch the state dict method
            patch_dequantization_on_save(transformer)
            quantization_type = get_qtype(self.model_config.qtype)
            self.print_and_status_update("Quantizing transformer")
            quantize(transformer, weights=quantization_type,
                     **self.model_config.quantize_kwargs)
            freeze(transformer)
            transformer.to(self.device_torch)
        else:
            transformer.to(self.device_torch, dtype=dtype)

        flush()

        self.print_and_status_update("Loading T5")
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            base_model_path, subfolder="tokenizer_2", torch_dtype=dtype
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            base_model_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        text_encoder_2.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing T5")
            quantize(text_encoder_2, weights=get_qtype(
                self.model_config.qtype))
            freeze(text_encoder_2)
            flush()

        self.print_and_status_update("Loading CLIP")
        text_encoder = CLIPTextModel.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype)
        text_encoder.to(self.device_torch, dtype=dtype)

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype)

        self.noise_scheduler = Flex2.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        pipe: Flex2Pipeline = Flex2Pipeline(
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
        # just to make sure everything is on the right device and dtype
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        text_encoder[1].to(self.device_torch)
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
        scheduler = Flex2.get_train_scheduler()

        pipeline: Flex2Pipeline = Flex2Pipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            text_encoder_2=unwrap_model(self.text_encoder[1]),
            tokenizer_2=self.tokenizer[1],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer)
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: Flex2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if gen_config.ctrl_img is None:
            control_img = None
        else:
            control_img = Image.open(gen_config.ctrl_img)
            if ".inpaint." not in gen_config.ctrl_img:
                control_img = control_img.convert("RGB")
            else:
                # make sure it has an alpha
                if control_img.mode != "RGBA":
                    raise ValueError("Inpainting images must have an alpha channel")
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            pooled_prompt_embeds=conditional_embeds.pooled_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            control_image=control_img,
            control_image_idx=gen_config.ctrl_idx,
            **extra
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        guidance_embedding_scale: float,
        bypass_guidance_embedding: bool,
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

            img_ids = torch.zeros(h // 2, w // 2, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c",
                             b=bs).to(self.device_torch)

            txt_ids = torch.zeros(
                bs, text_embeddings.text_embeds.shape[1], 3).to(self.device_torch)

            # # handle guidance
            if self.unet_unwrapped.config.guidance_embeds:
                if isinstance(guidance_embedding_scale, list):
                    guidance = torch.tensor(
                        guidance_embedding_scale, device=self.device_torch)
                else:
                    guidance = torch.tensor(
                        [guidance_embedding_scale], device=self.device_torch)
                    guidance = guidance.expand(latent_model_input.shape[0])
            else:
                guidance = None

        if bypass_guidance_embedding:
            bypass_flux_guidance(self.unet)

        cast_dtype = self.unet.dtype
        # changes from orig implementation
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        noise_pred = self.unet(
            hidden_states=latent_model_input_packed.to(
                self.device_torch, cast_dtype),
            timestep=timestep / 1000,
            encoder_hidden_states=text_embeddings.text_embeds.to(
                self.device_torch, cast_dtype),
            pooled_projections=text_embeddings.pooled_embeds.to(
                self.device_torch, cast_dtype),
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=guidance,
            return_dict=False,
            **kwargs,
        )[0]

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

        if bypass_guidance_embedding:
            restore_flux_guidance(self.unet)
        
        return noise_pred
    
    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
        prompt_embeds, pooled_prompt_embeds = train_tools.encode_prompts_flux(
            self.tokenizer,
            self.text_encoder,
            prompt,
            max_length=512,
        )
        pe = PromptEmbeds(
            prompt_embeds
        )
        pe.pooled_embeds = pooled_prompt_embeds
        return pe
    
    def get_model_has_grad(self):
        # return from a weight if it has grad
        return self.model.proj_out.weight.requires_grad

    def get_te_has_grad(self):
        # return from a weight if it has grad
        return self.text_encoder[1].encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad
    
    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: FluxTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'transformer'),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        return (noise - batch.latents).detach()
    
    def condition_noisy_latents(self, latents: torch.Tensor, batch:'DataLoaderBatchDTO'):
        with torch.no_grad():
            # inpainting input is 0-1 (bs, 4, h, w) on batch.inpaint_tensor
            # 4th channel is the mask with 1 being keep area and 0 being area to inpaint.
            # todo handle dropout on a batch item level, this frops out the entire batch
            do_dropout = random.random() < self.inpaint_dropout if self.inpaint_dropout > 0.0 else False
            # do random mask if we dont have one
            inpaint_tensor = batch.inpaint_tensor
            if inpaint_tensor is None and batch.mask_tensor is not None:
                # we have a mask tensor, use it
                inpaint_tensor = batch.mask_tensor
            
            if self.inpaint_random_chance > 0.0:
                do_random = random.random() < self.inpaint_random_chance
                if do_random:
                    # force a random tensor
                    inpaint_tensor = None
            
            if inpaint_tensor is None and not do_dropout and self.do_random_inpainting:
                # generate a random one since we dont have one
                # this will make random blobs, invert the blobs for now as we normanlly inpaint the alpha
                inpaint_tensor = 1 - generate_random_mask(
                    batch_size=latents.shape[0],
                    height=latents.shape[2],
                    width=latents.shape[3],
                    device=latents.device,
                ).to(latents.device, latents.dtype)
            if inpaint_tensor is not None and not do_dropout:
                
                if inpaint_tensor.shape[1] == 4:
                    # get just the mask 
                    inpainting_tensor_mask = inpaint_tensor[:, 3:4, :, :].to(latents.device, dtype=latents.dtype)
                elif inpaint_tensor.shape[1] == 3:
                    # rgb mask. Just get one channel
                    inpainting_tensor_mask = inpaint_tensor[:, 0:1, :, :].to(latents.device, dtype=latents.dtype)
                    # mask is 0-1 with 1 being inpaint area, we need to invert it for now, it is re inverted later
                    inpaint_tensor = 1 - inpaint_tensor
                else:
                    inpainting_tensor_mask = inpaint_tensor
                
                # # use our batch latents so we cna avoid encoding again
                inpainting_latent = batch.latents
                
                # resize the mask to match the new encoded size
                inpainting_tensor_mask = F.interpolate(inpainting_tensor_mask, size=(inpainting_latent.shape[2], inpainting_latent.shape[3]), mode='bilinear')
                inpainting_tensor_mask = inpainting_tensor_mask.to(latents.device, latents.dtype)
                
                if self.random_blur_mask:
                    # blur the mask
                    # Give it a channel dim of 1
                    if len(inpainting_tensor_mask.shape) == 3:
                        # if it is 3d, add a channel dim
                        inpainting_tensor_mask = inpainting_tensor_mask.unsqueeze(1)
                    # we are at latent size, so keep kernel smaller
                    inpainting_tensor_mask = random_blur(
                        inpainting_tensor_mask,
                        min_kernel_size=3, 
                        max_kernel_size=8,
                        p=0.5
                    )
                
                do_mask_invert = False
                if self.invert_inpaint_mask_chance > 0.0:
                    do_mask_invert = random.random() < self.invert_inpaint_mask_chance
                if do_mask_invert:
                    # invert the mask
                    inpainting_tensor_mask = 1 - inpainting_tensor_mask
                
                # mask out the inpainting area, it is currently 0 for inpaint area, and 1 for keep area
                # we are zeroing our the latents in the inpaint area not on the pixel space.
                inpainting_latent = inpainting_latent * inpainting_tensor_mask
                
                # do the random dialation after the mask is applied so it does not match perfectly. 
                # this will make the model learn to prevent weird edges
                if self.random_dialate_mask:
                    inpainting_tensor_mask = random_dialate_mask(
                        inpainting_tensor_mask,
                        max_percent=0.05 
                    )
                
                # mask needs to be 1 for inpaint area and 0 for area to leave alone. So flip it.
                inpainting_tensor_mask = 1 - inpainting_tensor_mask
                # leave the mask as 0-1 and concat on channel of latents
                inpainting_latent = torch.cat((inpainting_latent, inpainting_tensor_mask), dim=1)
            else:
                # we have iinpainting but didnt get a control. or we are doing a dropout
                # the input needs to be all zeros for the latents and all 1s for the mask
                inpainting_latent = torch.zeros_like(latents)
                # add ones for the mask since we are technically inpainting everything
                inpainting_latent = torch.cat((inpainting_latent, torch.ones_like(inpainting_latent[:, :1, :, :])), dim=1)
            
            control_tensor = batch.control_tensor
            if control_tensor is None:
                # concat random normal noise onto the latents
                # check dimension, this is before they are rearranged
                # it is latent_model_input = torch.cat([latents, control_image], dim=2) after rearranging
                ctrl = torch.zeros(
                    latents.shape[0], # bs
                    latents.shape[1],
                    latents.shape[2], 
                    latents.shape[3], 
                    device=latents.device, 
                    dtype=latents.dtype
                )
                # inpainting always comes first
                ctrl = torch.cat((inpainting_latent, ctrl), dim=1)
                latents = torch.cat((latents, ctrl), dim=1)
                return latents.detach()
            # if we have multiple control tensors, they come in like [bs, num_control_images, ch, h, w]
            # if we have 1, it comes in like [bs, ch, h, w]
            # stack out control tensors to be [bs, ch * num_control_images, h, w]
            
            control_tensor_list = []
            if len(control_tensor.shape) == 4:
                control_tensor_list.append(control_tensor)
            else:
                num_control_images = control_tensor.shape[1]
                # reshape
                control_tensor = control_tensor.view(
                    control_tensor.shape[0], 
                    control_tensor.shape[1] * control_tensor.shape[2], 
                    control_tensor.shape[3], 
                    control_tensor.shape[4]
                )
                control_tensor_list = control_tensor.chunk(num_control_images, dim=1)
            
            do_dropout = random.random() < self.control_dropout if self.control_dropout > 0.0 else False
            if do_dropout:
                # dropout with zeros
                control_latent = torch.zeros_like(batch.latents)
            else:
                # we only have one control so we randomly pick from this list
                control_tensor = random.choice(control_tensor_list)
                # it is 0-1 need to convert to -1 to 1
                control_tensor = control_tensor * 2 - 1

                control_tensor = control_tensor.to(self.vae_device_torch, dtype=self.torch_dtype)
                
                # if it is not the size of batch.tensor, (bs,ch,h,w) then we need to resize it
                if control_tensor.shape[2] != batch.tensor.shape[2] or control_tensor.shape[3] != batch.tensor.shape[3]:
                    control_tensor = F.interpolate(control_tensor, size=(batch.tensor.shape[2], batch.tensor.shape[3]), mode='bilinear')
                
                # encode it
                control_latent = self.encode_images(control_tensor).to(latents.device, latents.dtype)
                
            # inpainting always comes first
            control_latent = torch.cat((inpainting_latent, control_latent), dim=1)
            # concat it onto the latents
            latents = torch.cat((latents, control_latent), dim=1)
            return latents.detach()