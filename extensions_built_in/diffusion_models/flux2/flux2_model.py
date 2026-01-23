import math
import os
from typing import TYPE_CHECKING, List, Optional

import huggingface_hub
import torch
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze, QTensor
from toolkit.util.quantize import quantize, get_qtype, quantize_model

from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from .src.model import Flux2, Flux2Params
from .src.pipeline import Flux2Pipeline
from .src.autoencoder import AutoEncoder, AutoEncoderParams
from safetensors.torch import load_file, save_file
from PIL import Image
import torch.nn.functional as F

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

from .src.sampling import (
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    scatter_ids,
)

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}

MISTRAL_PATH = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
FLUX2_VAE_FILENAME = "ae.safetensors"
FLUX2_TRANSFORMER_FILENAME = "flux2-dev.safetensors"

HF_TOKEN = os.getenv("HF_TOKEN", None)


class Flux2Model(BaseModel):
    arch = "flux2"
    flux2_te_type: str = "mistral"  # "mistral" or "qwen"
    flux2_vae_path: str = None
    flux2_te_filename: str = FLUX2_TRANSFORMER_FILENAME
    flux2_is_guidance_distilled: bool = True

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
        self.target_lora_modules = ["Flux2"]
        # control images will come in as a list for encoding some things if true
        self.has_multiple_control_images = True
        # do not resize control images
        self.use_raw_control_images = True

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def get_flux2_params(self):
        return Flux2Params()

    def load_te(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Mistral")

        text_encoder: Mistral3ForConditionalGeneration = (
            Mistral3ForConditionalGeneration.from_pretrained(
                MISTRAL_PATH,
                torch_dtype=dtype,
            )
        )
 
        if not self.model_config.low_vram:
            text_encoder.to(self.device_torch, dtype=dtype)

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Mistral")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        if self.model_config.layer_offloading:
            text_encoder.to('cpu')

        flush()
        tokenizer = AutoProcessor.from_pretrained(MISTRAL_PATH)
        return text_encoder, tokenizer

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Flux2 model")
        # will be updated if we detect a existing checkpoint in training folder
        model_path = self.model_config.name_or_path
        transformer_path = model_path

        self.print_and_status_update("Loading transformer")
        with torch.device("meta"):
            transformer = Flux2(self.get_flux2_params())

        # use local path if provided
        if os.path.exists(os.path.join(transformer_path, self.flux2_te_filename)):
            transformer_path = os.path.join(transformer_path, self.flux2_te_filename)

        if not os.path.exists(transformer_path):
            # assume it is from the hub
            transformer_path = huggingface_hub.hf_hub_download(
                repo_id=model_path,
                filename=self.flux2_te_filename,
                token=HF_TOKEN,
            )

        transformer_state_dict = load_file(transformer_path, device="cpu")

        # cast to dtype
        for key in transformer_state_dict:
            transformer_state_dict[key] = transformer_state_dict[key].to(dtype)

        transformer.load_state_dict(transformer_state_dict, assign=True)

        if not self.model_config.low_vram:
            transformer.to(self.quantize_device, dtype=dtype)

        if self.model_config.quantize:
            # patch the state dict method
            patch_dequantization_on_save(transformer)
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

        if self.model_config.layer_offloading:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch, dtype=dtype)

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        flush()

        text_encoder, tokenizer = self.load_te()

        self.print_and_status_update("Loading VAE")
        vae_path = self.model_config.vae_path

        if os.path.exists(os.path.join(model_path, FLUX2_VAE_FILENAME)):
            vae_path = os.path.join(model_path, FLUX2_VAE_FILENAME)

        if vae_path is None:
            vae_path = self.flux2_vae_path

        if vae_path is None or not os.path.exists(vae_path):
            p = vae_path if vae_path is not None else model_path
            # assume it is from the hub
            vae_path = huggingface_hub.hf_hub_download(
                repo_id=p,
                filename=FLUX2_VAE_FILENAME,
                token=HF_TOKEN,
            )
        with torch.device("meta"):
            vae = AutoEncoder(AutoEncoderParams())

        vae_state_dict = load_file(vae_path, device="cpu")

        # cast to dtype
        for key in vae_state_dict:
            vae_state_dict[key] = vae_state_dict[key].to(dtype)

        vae.load_state_dict(vae_state_dict, assign=True)

        self.noise_scheduler = Flux2Model.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        pipe: Flux2Pipeline = Flux2Pipeline(
            scheduler=self.noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            text_encoder_type=self.flux2_te_type,
            is_guidance_distilled=self.flux2_is_guidance_distilled,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        flush()
        # just to make sure everything is on the right device and dtype
        if not self.model_config.low_vram and not self.model_config.layer_offloading:
            text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
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
        scheduler = Flux2Model.get_train_scheduler()

        pipeline: Flux2Pipeline = Flux2Pipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
            text_encoder_type=self.flux2_te_type,
            is_guidance_distilled=self.flux2_is_guidance_distilled,
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: Flux2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        gen_config.width = (
            gen_config.width // self.get_bucket_divisibility()
        ) * self.get_bucket_divisibility()
        gen_config.height = (
            gen_config.height // self.get_bucket_divisibility()
        ) * self.get_bucket_divisibility()

        control_img_list = []
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        elif gen_config.ctrl_img_1 is not None:
            control_img = Image.open(gen_config.ctrl_img_1)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        if gen_config.ctrl_img_2 is not None:
            control_img = Image.open(gen_config.ctrl_img_2)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        if gen_config.ctrl_img_3 is not None:
            control_img = Image.open(gen_config.ctrl_img_3)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)

        if not self.flux2_is_guidance_distilled:
            extra["negative_prompt_embeds"] = unconditional_embeds.text_embeds

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            control_img_list=control_img_list,
            **extra,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        guidance_embedding_scale: float,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        with torch.no_grad():
            txt, txt_ids = batched_prc_txt(text_embeddings.text_embeds)
            packed_latents, img_ids = batched_prc_img(latent_model_input)

            # prepare image conditioning if any
            img_cond_seq: torch.Tensor | None = None
            img_cond_seq_ids: torch.Tensor | None = None

            # handle control images
            batch_control_tensor_list = batch.control_tensor_list
            if batch_control_tensor_list is None and batch.control_tensor is not None:
                batch_control_tensor_list = []
                for b in range(latent_model_input.shape[0]):
                    batch_control_tensor_list.append(batch.control_tensor[b : b + 1])

            if batch_control_tensor_list is not None:
                batch_size, num_channels_latents, height, width = (
                    latent_model_input.shape
                )

                control_image_max_res = 1024 * 1024
                if self.model_config.model_kwargs.get("match_target_res", False):
                    # use the current target size to set the control image res
                    control_image_res = (
                        height
                        * self.pipeline.vae_scale_factor
                        * width
                        * self.pipeline.vae_scale_factor
                    )
                    control_image_max_res = control_image_res

                if len(batch_control_tensor_list) != batch_size:
                    raise ValueError(
                        "Control tensor list length does not match batch size"
                    )
                for control_tensor_list in batch_control_tensor_list:
                    # control tensor list is a list of tensors for this batch item
                    controls = []
                    # pack control
                    for control_img in control_tensor_list:
                        # control images are 0 - 1 scale, shape (1, ch, height, width)
                        control_img = control_img.to(
                            self.device_torch, dtype=self.torch_dtype
                        )
                        # if it is only 3 dim, add batch dim
                        if len(control_img.shape) == 3:
                            control_img = control_img.unsqueeze(0)

                        # resize to fit within max res while keeping aspect ratio
                        if self.model_config.model_kwargs.get(
                            "match_target_res", False
                        ):
                            ratio = control_img.shape[2] / control_img.shape[3]
                            c_width = math.sqrt(control_image_res * ratio)
                            c_height = c_width / ratio

                            c_width = round(c_width / 32) * 32
                            c_height = round(c_height / 32) * 32

                            control_img = F.interpolate(
                                control_img, size=(c_height, c_width), mode="bilinear"
                            )

                        # scale to -1 to 1
                        control_img = control_img * 2 - 1
                        controls.append(control_img)

                    img_cond_seq_item, img_cond_seq_ids_item = encode_image_refs(
                        self.vae, controls, limit_pixels=control_image_max_res
                    )
                    if img_cond_seq is None:
                        img_cond_seq = img_cond_seq_item
                        img_cond_seq_ids = img_cond_seq_ids_item
                    else:
                        img_cond_seq = torch.cat(
                            (img_cond_seq, img_cond_seq_item), dim=0
                        )
                        img_cond_seq_ids = torch.cat(
                            (img_cond_seq_ids, img_cond_seq_ids_item), dim=0
                        )

            img_input = packed_latents
            img_input_ids = img_ids

            if img_cond_seq is not None:
                assert img_cond_seq_ids is not None, (
                    "You need to provide either both or neither of the sequence conditioning"
                )
                img_input = torch.cat((img_input, img_cond_seq), dim=1)
                img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

            guidance_vec = torch.full(
                (img_input.shape[0],),
                guidance_embedding_scale,
                device=img_input.device,
                dtype=img_input.dtype,
            )

            cast_dtype = self.model.dtype

        packed_noise_pred = self.transformer(
            x=img_input.to(self.device_torch, cast_dtype),
            x_ids=img_input_ids.to(self.device_torch),
            timesteps=timestep.to(self.device_torch, cast_dtype) / 1000,
            ctx=txt.to(self.device_torch, cast_dtype),
            ctx_ids=txt_ids.to(self.device_torch),
            guidance=guidance_vec.to(self.device_torch, cast_dtype),
        )

        if img_cond_seq is not None:
            packed_noise_pred = packed_noise_pred[:, : packed_latents.shape[1]]

        if isinstance(packed_noise_pred, QTensor):
            packed_noise_pred = packed_noise_pred.dequantize()

        noise_pred = torch.cat(scatter_ids(packed_noise_pred, img_ids)).squeeze(2)

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt, device=self.device_torch
        )
        pe = PromptEmbeds(prompt_embeds)
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        # only save the unet
        transformer: Flux2 = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, QTensor):
                v = v.dequantize()
            save_dict[k] = v.clone().to("cpu", dtype=save_dtype)

        meta = get_meta_for_safetensors(meta, name="flux2")
        save_file(save_dict, output_path, metadata=meta)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "flux2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["double_blocks", "single_blocks"]

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

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        # Move to vae to device if on cpu
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        # move to device and dtype
        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)

        latents = self.vae.encode(images)

        return latents
