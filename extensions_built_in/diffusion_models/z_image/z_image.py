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

from transformers import AutoTokenizer, Qwen3ForCausalLM
from diffusers import AutoencoderKL

try:
    from diffusers import ZImagePipeline
    from diffusers.models.transformers import ZImageTransformer2DModel
    # Try to import config - may be in different locations depending on diffusers version
    try:
        from diffusers.models.transformers.transformer_2d import ZImageTransformer2DModelConfig
    except ImportError:
        try:
            from diffusers import ZImageTransformer2DModelConfig
        except ImportError:
            # If config class not available, we'll create config dict instead
            ZImageTransformer2DModelConfig = None
except ImportError:
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )


scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}


class ZImageModel(BaseModel):
    arch = "zimage"

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
        self.target_lora_modules = ["ZImageTransformer2DModel"]

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2  # 16 for the VAE, 2 for patch size

    def load_training_adapter(self, transformer: ZImageTransformer2DModel):
        self.print_and_status_update("Loading assistant LoRA")
        lora_path = self.model_config.assistant_lora_path
        if not os.path.exists(lora_path):
            # assume it is a hub path
            lora_splits = lora_path.split("/")
            if len(lora_splits) != 3:
                raise ValueError(
                    f"Assistant LoRA path {lora_path} is not a valid local path or hub path."
                )
            repo_id = "/".join(lora_splits[:2])
            filename = lora_splits[2]
            try:
                lora_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                )
                # upgrade path to
                self.model_config.assistant_lora_path = lora_path
            except Exception as e:
                raise ValueError(
                    f"Failed to download assistant LoRA from {lora_path}: {e}"
                )
        # load the adapter and merge it in. We will inference with a -1.0 multiplier so the adapter effects only work during training.
        lora_state_dict = load_file(lora_path)
        dim = int(
            lora_state_dict[
                "diffusion_model.layers.0.attention.to_k.lora_A.weight"
            ].shape[0]
        )

        new_sd = {}
        for key, value in lora_state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        lora_state_dict = new_sd

        network_config = {
            "type": "lora",
            "linear": dim,
            "linear_alpha": dim,
            "transformer_only": True,
        }

        network_config = NetworkConfig(**network_config)
        LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=self.target_lora_modules,
            is_assistant_adapter=True,
            is_ara=True,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        self.print_and_status_update("Merging in assistant LoRA")
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)

        network.merge_in(merge_weight=1.0)

        # mark it as not merged so inference ignores it.
        network.is_merged_in = False

        # add the assistant so sampler will activate it while sampling
        self.assistant_lora: LoRASpecialNetwork = network

        # deactivate lora during training
        self.assistant_lora.multiplier = -1.0
        self.assistant_lora.is_active = False

        # tell the model to invert assistant on inference since we want remove lora effects
        self.invert_assistant_lora = True

    @staticmethod
    def _convert_zimage_safetensors_to_diffusers(state_dict):
        """
        Convert Z-Image safetensors keys from original format to diffusers format.
        This allows loading safetensors files directly without requiring diffusers format conversion.
        """
        out = {}
        
        def rename_key(k: str) -> str:
            # strip prefix
            if k.startswith("model.diffusion_model."):
                k = k.replace("model.diffusion_model.", "")
            
            # rename embedder + final layer to match diffusers ZImage
            k = k.replace("x_embedder", "all_x_embedder.2-1")
            k = k.replace("final_layer", "all_final_layer.2-1")
            
            # attention renames
            k = k.replace("attention.out", "attention.to_out.0")
            k = k.replace("attention.q_norm", "attention.norm_q")
            k = k.replace("attention.k_norm", "attention.norm_k")
            
            return k
        
        for key, tensor in state_dict.items():
            # handle qkv split (both weight and bias if present)
            if ".attention.qkv." in key:
                base = key.replace("model.diffusion_model.", "")
                # Remove .attention.qkv.weight or .attention.qkv.bias
                if base.endswith(".attention.qkv.weight"):
                    base = base.replace(".attention.qkv.weight", "")
                    # ZImage uses [3 * hidden, hidden] layout for weights
                    q, k, v = torch.chunk(tensor, 3, dim=0)
                    out[f"{base}.attention.to_q.weight"] = q
                    out[f"{base}.attention.to_k.weight"] = k
                    out[f"{base}.attention.to_v.weight"] = v
                elif base.endswith(".attention.qkv.bias"):
                    base = base.replace(".attention.qkv.bias", "")
                    # ZImage uses [3 * hidden] layout for bias
                    q, k, v = torch.chunk(tensor, 3, dim=0)
                    out[f"{base}.attention.to_q.bias"] = q
                    out[f"{base}.attention.to_k.bias"] = k
                    out[f"{base}.attention.to_v.bias"] = v
                continue
            
            new_key = rename_key(key)
            out[new_key] = tensor
        
        return out
    
    @staticmethod
    def _infer_config_from_state_dict(state_dict):
        """
        Infer ZImageTransformer2DModel config from state dict keys and shapes.
        Similar to make_configs.py but integrated for runtime use.
        """
        # Infer hidden size from q projection
        q_keys = [k for k in state_dict if k.endswith(".attention.to_q.weight")]
        if not q_keys:
            raise RuntimeError("No attention.to_q.weight keys found in state dict")
        
        sample_q = state_dict[q_keys[0]]
        hidden_size = sample_q.shape[0]
        
        # Infer attention heads - Z-Image uses square Q projection: [hidden, hidden]
        # head_dim is typically 64; verify divisibility
        for head_dim in (64, 128, 32):
            if hidden_size % head_dim == 0:
                num_heads = hidden_size // head_dim
                break
        else:
            raise RuntimeError(f"Cannot infer head_dim from hidden_size={hidden_size}")
        
        # Infer number of layers
        layer_ids = set()
        for k in state_dict:
            if k.startswith("layers."):
                layer_ids.add(int(k.split(".")[1]))
        
        num_layers = max(layer_ids) + 1 if layer_ids else 32  # default to 32 if not found
        
        # Infer in_channels from x_embedder weight shape
        # x_embedder has shape [hidden_size, in_channels * patch_size * patch_size]
        # patch_size is 2, so in_channels = x_embedder.shape[1] / 4
        x_keys = [k for k in state_dict if "all_x_embedder" in k and k.endswith(".weight")]
        if not x_keys:
            # Fallback: use default
            in_channels = 4  # default for standard VAE
        else:
            x_w = state_dict[x_keys[0]]
            patch_size = 2  # ZImage uses patch_size=2
            in_channels = x_w.shape[1] // (patch_size * patch_size)
        
        # Create config dict or config object depending on availability
        config_dict = {
            "num_layers": num_layers,
            "num_attention_heads": num_heads,
            "attention_head_dim": head_dim,
            "hidden_size": hidden_size,
            "in_channels": in_channels,
            "norm_type": "ada_norm_single",
            "norm_eps": 1e-05,
            "use_bias": True,
        }
        
        # Note: cross_attention_dim is not typically needed in config as it's inferred 
        # from text encoder, but if needed it can be added here
        
        if ZImageTransformer2DModelConfig is not None:
            config = ZImageTransformer2DModelConfig(**config_dict)
        else:
            # Fallback: use dict and let from_config handle it
            config = config_dict
        
        return config

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading ZImage model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        self.print_and_status_update("Loading transformer")

        # Check if model_path is a safetensors file (direct checkpoint loading)
        # Note: When loading from safetensors, only the transformer is loaded from the file.
        # Text encoder and VAE are still loaded from base_model_path (extras_name_or_path),
        # which must be set to a diffusers format path (e.g., "Tongyi-MAI/Z-Image-Turbo")
        if model_path.endswith(".safetensors") and os.path.exists(model_path):
            if not base_model_path:
                raise ValueError(
                    "When loading transformer from safetensors file, extras_name_or_path must be set "
                    "to provide the text encoder and VAE (e.g., 'Tongyi-MAI/Z-Image-Turbo')"
                )
            
            self.print_and_status_update("Loading from safetensors file (converting keys to diffusers format)")
            # Load and convert the safetensors file
            state_dict = load_file(model_path, device='cpu')
            converted_state_dict = self._convert_zimage_safetensors_to_diffusers(state_dict)
            
            # Infer config from the converted state dict
            config = self._infer_config_from_state_dict(converted_state_dict)
            
            # Create model from config
            if isinstance(config, dict):
                transformer = ZImageTransformer2DModel.from_config(config)
            else:
                transformer = ZImageTransformer2DModel(config)
            # Load the converted state dict
            transformer.load_state_dict(converted_state_dict, strict=False)
            transformer = transformer.to(dtype)
            
            # Extract values for logging (handle both dict and object)
            if isinstance(config, dict):
                hidden_size = config["hidden_size"]
                num_layers = config["num_layers"]
                num_heads = config["num_attention_heads"]
            else:
                hidden_size = config.hidden_size
                num_layers = config.num_layers
                num_heads = config.num_attention_heads
            
            self.print_and_status_update(f"Inferred config: hidden_size={hidden_size}, "
                                       f"num_layers={num_layers}, "
                                       f"num_attention_heads={num_heads}")
            self.print_and_status_update(f"Text encoder and VAE will be loaded from: {base_model_path}")
            self.print_and_status_update("Note: Only text_encoder, tokenizer, and vae subfolders will be downloaded, NOT the transformer")
        else:
            # Original diffusers format loading
            transformer_path = model_path
            transformer_subfolder = "transformer"
            if os.path.exists(transformer_path):
                transformer_subfolder = None
                transformer_path = os.path.join(transformer_path, "transformer")
                # check if the path is a full checkpoint.
                te_folder_path = os.path.join(model_path, "text_encoder")
                # if we have the te, this folder is a full checkpoint, use it as the base
                if os.path.exists(te_folder_path):
                    base_model_path = model_path

            transformer = ZImageTransformer2DModel.from_pretrained(
                transformer_path, subfolder=transformer_subfolder, torch_dtype=dtype
            )

        # load assistant lora if specified
        if self.model_config.assistant_lora_path is not None:
            self.load_training_adapter(transformer)
            # set qtype to be float8 if it is qfloat8
            if self.model_config.qtype == "qfloat8":
                self.model_config.qtype = "float8"

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
                ]
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        flush()

        # Load text encoder and tokenizer from base_model_path
        # Using subfolder parameter ensures ONLY the text_encoder and tokenizer subfolders
        # are downloaded, NOT the transformer folder (which is loaded from safetensors file)
        self.print_and_status_update("Text Encoder")
        self.print_and_status_update(f"Downloading text encoder and tokenizer from: {base_model_path}")
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

        # Load VAE from base_model_path
        # Using subfolder parameter ensures ONLY the vae subfolder is downloaded,
        # NOT the transformer folder (which is loaded from safetensors file)
        self.print_and_status_update("Loading VAE")
        self.print_and_status_update(f"Downloading VAE from: {base_model_path}")
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype
        )

        self.noise_scheduler = ZImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {}

        pipe: ZImagePipeline = ZImagePipeline(
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
        scheduler = ZImageModel.get_train_scheduler()

        pipeline: ZImagePipeline = ZImagePipeline(
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
        pipeline: ZImagePipeline,
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
            negative_prompt_embeds=unconditional_embeds.text_embeds,
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
        self.model.to(self.device_torch)

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        timestep_model_input = (1000 - timestep) / 1000

        model_out_list = self.transformer(
            latent_model_input_list,
            timestep_model_input,
            text_embeddings.text_embeds,
        )[0]

        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=self.device_torch,
        )
        pe = PromptEmbeds([prompt_embeds, None])
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: ZImageTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "zimage"

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
