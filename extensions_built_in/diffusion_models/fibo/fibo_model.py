from typing import TYPE_CHECKING, List

import torch
from optimum.quanto import QTensor, freeze
from safetensors.torch import save_file

from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.util.quantize import get_qtype, quantize

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

# Scheduler config for FIBO (flow-matching)
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True
}


class FiboModel(BaseModel):
    arch = "fibo"

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
        self.target_lora_modules = ['BriaFiboTransformer2DModel']

        # Will be set during load_model
        self.latents_mean = None
        self.latents_std = None

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # FIBO uses patch_size=1 with VAE scale_factor=16
        return 16

    def load_model(self):
        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path

        # Import diffusers components
        from diffusers import BriaFiboPipeline

        self.print_and_status_update("Loading FIBO pipeline")

        # Load the full pipeline from HuggingFace
        pipe = BriaFiboPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )

        # Extract components
        transformer = pipe.transformer
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        vae = pipe.vae

        # Store VAE normalization parameters
        if hasattr(vae.config, 'latents_mean') and vae.config.latents_mean is not None:
            self.latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
        else:
            self.latents_mean = None

        if hasattr(vae.config, 'latents_std') and vae.config.latents_std is not None:
            self.latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
        else:
            self.latents_std = None

        self.print_and_status_update("Moving transformer to device")
        if not self.low_vram:
            # for low vram, we leave it on the cpu. Quantizes slower, but allows training on primary gpu
            transformer.to(self.quantize_device, dtype=dtype)

        if self.model_config.quantize:
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

        self.print_and_status_update("Loading text encoder")
        text_encoder.to(self.device_torch, dtype=dtype)

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing text encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)

        flush()

        self.noise_scheduler = FiboModel.get_train_scheduler()

        self.print_and_status_update("Loading VAE")
        vae.to(self.device_torch, dtype=dtype)

        flush()

        # Set eval mode and disable gradients for inference components
        text_encoder.requires_grad_(False)
        text_encoder.eval()

        # Save components to the model class
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = transformer
        self.pipeline = pipe

        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        from diffusers import BriaFiboPipeline

        scheduler = FiboModel.get_train_scheduler()

        pipeline = BriaFiboPipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer)
        )

        return pipeline

    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        has_cached_layers = (
            hasattr(conditional_embeds, 'text_encoder_layers')
            and conditional_embeds.text_encoder_layers is not None
        )

        call_kwargs = {
            'height': gen_config.height,
            'width': gen_config.width,
            'num_inference_steps': gen_config.num_inference_steps,
            'guidance_scale': gen_config.guidance_scale,
            'latents': gen_config.latents,
            'generator': generator,
        }

        if has_cached_layers:
            # Use pre-computed embeddings — the text encoder may be unloaded
            # (replaced with FakeTextEncoder) when cache_text_embeddings is enabled.
            # The FIBO pipeline's encode_prompt doesn't support passing prompt_layers
            # directly, so we monkey-patch it to return our pre-computed values.
            original_encode_prompt = pipeline.encode_prompt
            self._inject_cached_embeds(
                pipeline, call_kwargs, gen_config,
                conditional_embeds, unconditional_embeds,
            )
            try:
                call_kwargs.update(extra)
                img = pipeline(**call_kwargs).images[0]
            finally:
                pipeline.encode_prompt = original_encode_prompt
        else:
            # No cached layers — use raw prompt strings (text encoder must be loaded)
            call_kwargs['prompt'] = gen_config.prompt
            if gen_config.guidance_scale > 1.0:
                call_kwargs['negative_prompt'] = gen_config.negative_prompt or ""
            call_kwargs.update(extra)
            img = pipeline(**call_kwargs).images[0]

        return img

    def _inject_cached_embeds(
        self, pipeline, call_kwargs, gen_config,
        conditional_embeds, unconditional_embeds,
    ):
        """Monkey-patch pipeline.encode_prompt to return pre-computed embeddings.

        The FIBO pipeline requires both prompt_embeds and prompt_layers (all
        intermediate text encoder hidden states) for DimFusion. Its encode_prompt
        doesn't support passing prompt_layers as a parameter, so we temporarily
        replace encode_prompt with a closure that returns our cached values.
        The caller is responsible for restoring the original encode_prompt.
        """
        dtype = self.unet.dtype
        device = pipeline._execution_device

        # Extract positive embeddings
        pos_embeds = conditional_embeds.text_embeds.to(device=device, dtype=dtype)
        pos_layers = [layer.to(device=device, dtype=dtype) for layer in conditional_embeds.text_encoder_layers]
        pos_mask = conditional_embeds.attention_mask
        if pos_mask is not None:
            pos_mask = pos_mask.to(device=device, dtype=dtype)

        # Extract negative embeddings (for CFG)
        neg_embeds = None
        neg_layers = None
        neg_mask = None

        if gen_config.guidance_scale > 1.0:
            neg_embeds = unconditional_embeds.text_embeds.to(device=device, dtype=dtype)
            neg_layers = [layer.to(device=device, dtype=dtype) for layer in unconditional_embeds.text_encoder_layers]
            neg_mask = unconditional_embeds.attention_mask
            if neg_mask is not None:
                neg_mask = neg_mask.to(device=device, dtype=dtype)

            # Pad positive and negative to the same sequence length
            max_tokens = max(neg_embeds.shape[1], pos_embeds.shape[1])

            pos_embeds, pos_mask = pipeline.pad_embedding(pos_embeds, max_tokens, attention_mask=pos_mask)
            pos_layers = [pipeline.pad_embedding(layer, max_tokens)[0] for layer in pos_layers]

            neg_embeds, neg_mask = pipeline.pad_embedding(neg_embeds, max_tokens, attention_mask=neg_mask)
            neg_layers = [pipeline.pad_embedding(layer, max_tokens)[0] for layer in neg_layers]
        else:
            max_tokens = pos_embeds.shape[1]
            pos_embeds, pos_mask = pipeline.pad_embedding(pos_embeds, max_tokens, attention_mask=pos_mask)
            pos_layers = [pipeline.pad_embedding(layer, max_tokens)[0] for layer in pos_layers]

        text_ids = torch.zeros(pos_embeds.shape[0], max_tokens, 3, device=device, dtype=dtype)

        # Replace encode_prompt with closure returning pre-computed values
        def patched_encode_prompt(**_kwargs):
            return (
                pos_embeds, neg_embeds, text_ids,
                pos_mask, neg_mask,
                pos_layers, neg_layers,
            )

        pipeline.encode_prompt = patched_encode_prompt

        # Pass prompt_embeds so pipeline derives batch_size from its shape
        # (instead of from a prompt string, which we don't provide)
        call_kwargs['prompt_embeds'] = pos_embeds

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs
    ):
        bs, c, h, w = latent_model_input.shape

        # FIBO uses patch_size=1, NO 2x2 packing - simple flatten
        # (B, C, H, W) -> (B, H*W, C)
        with torch.no_grad():
            latent_model_input_packed = latent_model_input.permute(0, 2, 3, 1)
            latent_model_input_packed = latent_model_input_packed.reshape(bs, h * w, c)

            # Image position IDs (full resolution, not halved like Flux)
            img_ids = torch.zeros(h, w, 3, device=self.device_torch)
            img_ids[..., 1] = torch.arange(h, device=self.device_torch)[:, None]
            img_ids[..., 2] = torch.arange(w, device=self.device_torch)[None, :]
            img_ids = img_ids.reshape(h * w, 3)
            img_ids = img_ids.unsqueeze(0).expand(bs, -1, -1)

            # Text position IDs
            txt_ids = torch.zeros(bs, text_embeddings.text_embeds.shape[1], 3, device=self.device_torch)

        cast_dtype = self.unet.dtype

        # Build transformer inputs
        transformer_kwargs = {
            'hidden_states': latent_model_input_packed.to(self.device_torch, cast_dtype),
            'timestep': timestep,  # FIBO expects raw timesteps [0-1000], no normalization
            'encoder_hidden_states': text_embeddings.text_embeds.to(self.device_torch, cast_dtype),
            'txt_ids': txt_ids,
            'img_ids': img_ids,
            'return_dict': False,
        }

        # Add attention mask if available
        if text_embeddings.attention_mask is not None:
            # Build joint attention mask
            latent_attention_mask = torch.ones(bs, h * w, device=self.device_torch, dtype=cast_dtype)
            attention_mask = torch.cat([
                text_embeddings.attention_mask.to(self.device_torch, cast_dtype),
                latent_attention_mask
            ], dim=1)

            # Convert to additive mask format
            attention_matrix = torch.einsum("bi,bj->bij", attention_mask, attention_mask)
            attention_matrix = torch.where(
                attention_matrix == 1,
                torch.zeros_like(attention_matrix),
                torch.full_like(attention_matrix, float('-inf'))
            )
            attention_matrix = attention_matrix.unsqueeze(1)  # Add head dimension

            transformer_kwargs['joint_attention_kwargs'] = {"attention_mask": attention_matrix}

        # Add text_encoder_layers for DimFusion (REQUIRED by FIBO transformer)
        total_num_layers = (
            self.unet.config.num_layers +
            self.unet.config.num_single_layers
        )

        if not hasattr(text_embeddings, 'text_encoder_layers') or text_embeddings.text_encoder_layers is None:
            raise ValueError(
                "FIBO requires text_encoder_layers for DimFusion but they are missing. "
                "If using cache_text_embeddings, delete the cache and re-run to regenerate it."
            )

        te_layers = [
            layer.to(self.device_torch, cast_dtype)
            for layer in text_embeddings.text_encoder_layers
        ]

        # Pad or truncate layers to match transformer's expected count
        if len(te_layers) >= total_num_layers:
            # Remove first layers to keep the last ones
            te_layers = te_layers[len(te_layers) - total_num_layers:]
        else:
            # Duplicate last layer to fill the gap
            te_layers = te_layers + [te_layers[-1]] * (total_num_layers - len(te_layers))

        transformer_kwargs['text_encoder_layers'] = te_layers

        # Forward pass through transformer
        noise_pred = self.unet(**transformer_kwargs)[0]

        if isinstance(noise_pred, QTensor):
            noise_pred = noise_pred.dequantize()

        # Unpack output: (B, H*W, C) -> (B, C, H, W)
        noise_pred = noise_pred.view(bs, h, w, c)
        noise_pred = noise_pred.permute(0, 3, 1, 2)

        return noise_pred

    def get_prompt_embeds(self, prompt) -> PromptEmbeds:
        # Normalize prompt to a list of strings
        if prompt is None:
            prompts = [""]
        elif isinstance(prompt, str):
            prompts = [prompt]
        else:
            # Convert to list and ensure all elements are strings
            prompts = []
            for p in prompt:
                if p is None or p == "":
                    prompts.append("")
                elif isinstance(p, str):
                    prompts.append(p)
                else:
                    # Non-string, non-None: convert to empty string as fallback
                    prompts.append("")

        # Ensure we have at least one prompt
        if not prompts:
            prompts = [""]

        device = self.text_encoder.device
        dtype = self.text_encoder.dtype
        batch_size = len(prompts)

        # Special handling for empty prompts (like the diffusers pipeline)
        bot_token_id = 128000  # Special token for empty prompts in SmolLM3

        if all(p == "" for p in prompts):
            # All empty prompts - use special token
            input_ids = torch.full((batch_size, 1), bot_token_id, dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
        else:
            # Tokenize with SmolLM3 tokenizer
            text_inputs = self.tokenizer(
                prompts,
                padding="longest",
                max_length=2048,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)

            # Handle mixed empty/non-empty prompts
            if any(p == "" for p in prompts):
                empty_rows = torch.tensor([p == "" for p in prompts], dtype=torch.bool, device=device)
                input_ids[empty_rows] = bot_token_id
                attention_mask[empty_rows] = 1

        # Forward pass with all hidden states for DimFusion
        with torch.no_grad():
            encoder_outputs = self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = encoder_outputs.hidden_states

        # Concatenate last 2 layers for main embedding (DimFusion)
        # Shape: (batch, seq_len, 4096) = 2048 + 2048
        # Order matches BriaFiboPipeline.encode_prompt: [-1] (last), [-2] (second-to-last)
        prompt_embeds = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # Store all layers for DimFusion in transformer
        text_encoder_layers = [layer.to(dtype=dtype) for layer in hidden_states]

        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = attention_mask
        pe.text_encoder_layers = text_encoder_layers  # Custom attribute for DimFusion

        return pe

    @torch.no_grad()
    def encode_images(
            self,
            image_list: List[torch.Tensor],
            device=None,
            dtype=None
    ):
        """Encode images to latents with FIBO VAE normalization."""
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        # Move VAE to device if on CPU
        if self.vae.device == torch.device('cpu'):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        # Get VAE's actual dtype to ensure consistency
        vae_dtype = self.vae.dtype

        # Move images to device and VAE's dtype for encoding
        image_list = [image.to(device, dtype=vae_dtype) for image in image_list]

        # VAE scale factor for FIBO is 16
        VAE_SCALE_FACTOR = 16

        # Resize images if not divisible by scale factor
        from torchvision.transforms import Resize
        for i in range(len(image_list)):
            image = image_list[i]
            if image.shape[1] % VAE_SCALE_FACTOR != 0 or image.shape[2] % VAE_SCALE_FACTOR != 0:
                image_list[i] = Resize((
                    image.shape[1] // VAE_SCALE_FACTOR * VAE_SCALE_FACTOR,
                    image.shape[2] // VAE_SCALE_FACTOR * VAE_SCALE_FACTOR
                ))(image)

        images = torch.stack(image_list)

        # FIBO VAE expects 5D input for video: (B, C, T, H, W)
        # Add temporal dimension for image encoding
        images_5d = images.unsqueeze(2)
        latents = self.vae.encode(images_5d).latent_dist.mean
        # Remove temporal dimension
        latents = latents.squeeze(2)

        # Apply per-channel normalization if available
        if self.latents_mean is not None and self.latents_std is not None:
            latents_mean = self.latents_mean.to(latents.device, dtype=latents.dtype)
            latents_std = self.latents_std.to(latents.device, dtype=latents.dtype)
            latents = (latents - latents_mean) * latents_std

        latents = latents.to(device, dtype=dtype)
        return latents

    def decode_latents(
            self,
            latents: torch.Tensor,
            device=None,
            dtype=None
    ):
        """Decode latents to images with FIBO VAE denormalization."""
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.torch_dtype

        # Move VAE to device if on CPU
        if self.vae.device == torch.device('cpu'):
            self.vae.to(self.device)

        # Get VAE's actual dtype to ensure consistency
        vae_dtype = self.vae.dtype

        # Move latents to VAE's dtype for decoding
        latents = latents.to(device, dtype=vae_dtype)

        # Reverse normalization before decode
        if self.latents_mean is not None and self.latents_std is not None:
            latents_mean = self.latents_mean.to(latents.device, dtype=latents.dtype)
            latents_std = self.latents_std.to(latents.device, dtype=latents.dtype)
            latents = latents / latents_std + latents_mean

        # FIBO VAE expects 5D input for video: (B, C, T, H, W)
        # Add temporal dimension
        latents_5d = latents.unsqueeze(2)
        images = self.vae.decode(latents_5d).sample
        # Remove temporal dimension
        images = images.squeeze(2)

        # Convert back to requested dtype
        images = images.to(device, dtype=dtype)
        return images

    def get_model_has_grad(self):
        # Check if transformer has gradients
        # Use a representative weight from the model
        try:
            return self.model.proj_out.weight.requires_grad
        except AttributeError:
            # Fallback to checking any parameter
            for param in self.model.parameters():
                return param.requires_grad
        return False

    def get_te_has_grad(self):
        # Check if text encoder has gradients
        try:
            # SmolLM3 structure
            return self.text_encoder.model.layers[0].self_attn.q_proj.weight.requires_grad
        except AttributeError:
            # Fallback
            for param in self.text_encoder.parameters():
                return param.requires_grad
        return False

    def save_model(self, output_path, meta, save_dtype):
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"

        # Save only the transformer
        transformer = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}

        for k, v in state_dict.items():
            if isinstance(v, QTensor):
                v = v.dequantize()
            save_dict[k] = v.clone().to('cpu', dtype=save_dtype)

        meta = get_meta_for_safetensors(meta, name='fibo')
        save_file(save_dict, output_path, metadata=meta)

    def get_loss_target(self, *args, **kwargs):
        """Return loss target for flow-matching training."""
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        # Flow-matching: target is (noise - latents)
        return (noise - batch.latents).detach()

    def convert_lora_weights_before_save(self, state_dict):
        """Convert LoRA weights for ComfyUI compatibility."""
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        """Convert LoRA weights from ComfyUI format."""
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd

    def get_base_model_version(self):
        return "fibo"
