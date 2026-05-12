import os
from typing import List, Optional

import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from safetensors.torch import load_file, save_file
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager

from transformers import AutoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from .src.hidream_o1.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from .src.hidream_o1.pipeline import HiDreamO1Pipeline, DEFAULT_NOISE_SCALE
from toolkit.models.FakeVAE import FakeVAE
from typing import TYPE_CHECKING
from .src.hidream_o1.model_config import model_config

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False,
}

_GLOBAL_NOISE_SCALE = DEFAULT_NOISE_SCALE


class HidreamO1FlowmatchScheduler(CustomFlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        self.noise_scale = kwargs.get("noise_scale", DEFAULT_NOISE_SCALE)
        # remove noise_scale from kwargs so it doesn't get passed to the parent class
        kwargs.pop("noise_scale", None)
        super().__init__(*args, **kwargs)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t_01 = (timesteps / 1000).to(original_samples.device)
        scaled_noise = noise * self.noise_scale
        noisy_model_input = (1.0 - t_01) * original_samples + t_01 * scaled_noise
        return noisy_model_input


def add_special_tokens(tokenizer):
    """Attach the special-token shortcuts that the pipeline relies on."""
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase

    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


class FakeConfig:
    pass


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, scaling_factor=1.0):
        super().__init__()
        self._dtype = torch.float32
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = FakeConfig()
        self.config.scaling_factor = scaling_factor

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    # mimic to from torch
    def to(self, *args, **kwargs):
        # pull out dtype and device if they exist
        if "dtype" in kwargs:
            self._dtype = kwargs["dtype"]
        if "device" in kwargs:
            self._device = kwargs["device"]
        return super().to(*args, **kwargs)


class HidreamO1Model(BaseModel):
    arch = "hidream_o1"

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
        self.target_lora_modules = ["Qwen3VLForConditionalGeneration"]
        self.noise_scale = self.model_config.model_kwargs.get(
            "noise_scale", DEFAULT_NOISE_SCALE
        )
        self.noise_scale_inference = self.model_config.model_kwargs.get(
            "noise_scale_inference", self.noise_scale
        )
        print(f"Using noise scale: {self.noise_scale}")
        global _GLOBAL_NOISE_SCALE
        _GLOBAL_NOISE_SCALE = self.noise_scale
        self.is_comfy_weight = False  # save as single file if true

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return HidreamO1FlowmatchScheduler(
            **scheduler_config, noise_scale=_GLOBAL_NOISE_SCALE
        )

    def get_bucket_divisibility(self):
        return 32  # patch size

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading HidreamO1 model")
        model_path = self.model_config.name_or_path

        self.print_and_status_update("Loading transformer")

        try:
            processor = AutoProcessor.from_pretrained(model_path)
        except Exception as e:
            print(
                f"Failed to load processor from model path {model_path}, trying original path. Error: {e}"
            )
            processor_path = self.model_config.extras_name_or_path
            if processor_path.endswith(".safetensors"):
                processor_path = "HiDream-ai/HiDream-O1-Image"
            processor = AutoProcessor.from_pretrained(processor_path)

        tokenizer = get_tokenizer(processor)
        add_special_tokens(tokenizer)

        if model_path.endswith(".safetensors"):
            self.is_comfy_weight = True
            self.print_and_status_update(
                "Model is in safetensors format, loading with safetensors"
            )
            state_dict = load_file(model_path)

            for key, value in state_dict.items():
                state_dict[key] = value.to(dtype=dtype)

            # comfy ui is missing the lm head. It isnt used, but our model needs it for now
            state_dict["lm_head.weight"] = torch.zeros(
                151936, 4096, dtype=torch.bfloat16, device="cpu"
            )

            # transformer.load_state_dict(state_dict, assign=True)
            transformer = Qwen3VLForConditionalGeneration.from_pretrained(
                None,
                config=Qwen3VLConfig(**model_config),
                state_dict=state_dict,
                torch_dtype=self.torch_dtype,
            )
            del state_dict  # free memory
        else:
            transformer = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
            )
        flush()
        if not self.model_config.low_vram:
            transformer.to(self.device_torch)

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
                ignore_modules=[],
            )

        flush()

        # move over to device now if low vram
        if self.model_config.low_vram:
            transformer.to(self.device_torch)

        # fake ones so the trainer doesnt break
        vae = FakeVAE().to(self.device_torch, dtype=dtype)
        text_encoder = FakeTextEncoder().to(self.device_torch, dtype=dtype)

        self.noise_scheduler = HidreamO1Model.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {}

        pipe: HiDreamO1Pipeline = HiDreamO1Pipeline(
            scheduler=self.noise_scheduler,
            processor=processor,
            model=None,
            **kwargs,
        )
        pipe.model = transformer

        self.print_and_status_update("Preparing Model")

        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = processor
        self.model = pipe.model
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = HidreamO1Model.get_train_scheduler()

        pipe: HiDreamO1Pipeline = HiDreamO1Pipeline(
            scheduler=scheduler,
            processor=self.tokenizer,
            model=None,
        )
        pipe.model = self.transformer

        return pipe

    def encode_images(self, image_list: torch.Tensor, device=None, dtype=None):
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.device_torch)
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        # not needed since there is not a latent space
        return image_list.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.device_torch)
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        # not needed since there is not a latent space
        return latents.to(device, dtype=dtype)

    def generate_single_image(
        self,
        pipeline: HiDreamO1Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        img = pipeline(
            # prompt=gen_config.prompt,
            prompt_input_ids=conditional_embeds.text_embeds[0],
            # negative_prompt=gen_config.negative_prompt,
            negative_prompt_input_ids=unconditional_embeds.text_embeds[0],
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            generator=generator,
            noise_scale=self.noise_scale_inference,
            **extra,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: AdvancedPromptEmbeds,
        batch: "DataLoaderBatchDTO",
        **kwargs,
    ):
        import einops
        from .src.hidream_o1.pipeline import PATCH_SIZE, T_EPS

        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        device = self.device_torch
        in_dtype = latent_model_input.dtype
        bs, _, h_pix, w_pix = latent_model_input.shape
        h_patches = h_pix // PATCH_SIZE
        w_patches = w_pix // PATCH_SIZE

        # (B, C, H, W) -> (B, H/p * W/p, C * p * p)
        z = einops.rearrange(
            latent_model_input,
            "B C (H p1) (W p2) -> B (H W) (C p1 p2)",
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        ).to(device)

        model_config = self.model.config
        pad_token_id = getattr(model_config, "pad_token_id", 0) or 0

        with torch.no_grad():
            # Build per-sample conditioning, then left-pad the text portion so
            # the boi/tms + vision-token suffix stays at the end of the
            # sequence (the t2i layout assumes vision tokens are at the tail).
            per_sample = []
            for b in range(bs):
                tokens = text_embeddings.text_embeds[b]
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)
                per_sample.append(
                    self.pipeline.build_conditioning_sample(
                        tokens.to(device),
                        h_pix,
                        w_pix,
                    )
                )

            max_seq_len = max(s["input_ids"].shape[-1] for s in per_sample)
            ids_l, pos_l, tt_l, vm_l, mask_l = [], [], [], [], []
            for s in per_sample:
                ids = s["input_ids"].to(device)
                pos = s["position_ids"].to(device)
                tt = s["token_types"].to(device)
                vm = s["vinput_mask"].to(device)
                seq_len = ids.shape[-1]
                pad_len = max_seq_len - seq_len

                if pad_len > 0:
                    ids = torch.cat(
                        [
                            torch.full(
                                (1, pad_len),
                                pad_token_id,
                                dtype=ids.dtype,
                                device=device,
                            ),
                            ids,
                        ],
                        dim=-1,
                    )
                    pos = torch.cat(
                        [
                            torch.ones((3, 1, pad_len), dtype=pos.dtype, device=device),
                            pos,
                        ],
                        dim=-1,
                    )
                    tt = torch.cat(
                        [
                            torch.zeros((1, pad_len), dtype=tt.dtype, device=device),
                            tt,
                        ],
                        dim=-1,
                    )
                    vm = torch.cat(
                        [
                            torch.zeros((1, pad_len), dtype=vm.dtype, device=device),
                            vm,
                        ],
                        dim=-1,
                    )
                    mask = torch.cat(
                        [
                            torch.zeros((1, pad_len), dtype=torch.long, device=device),
                            torch.ones((1, seq_len), dtype=torch.long, device=device),
                        ],
                        dim=-1,
                    )
                else:
                    mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

                ids_l.append(ids)
                pos_l.append(pos)
                tt_l.append(tt)
                vm_l.append(vm)
                mask_l.append(mask)

            input_ids = torch.cat(ids_l, dim=0)
            position_ids = torch.cat(pos_l, dim=1)  # (3, B, S)
            token_types = torch.cat(tt_l, dim=0)
            vinput_mask = torch.cat(vm_l, dim=0)
            attention_mask = torch.cat(mask_l, dim=0)

        # Model wants timestep as denoising progress in (0, 1) where 1=clean.
        t_pixeldit = (1.0 - timestep.float() / 1000.0).to(device)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask if bs > 1 else None,
            vinputs=z,
            timestep=t_pixeldit.reshape(-1),
            token_types=token_types,
            use_flash_attn=True,
        )
        x_pred = outputs.x_pred  # (B, S, C*p*p) over the full padded sequence

        # Pull the vision-token positions only.
        vision_pred = torch.stack(
            [x_pred[b][vinput_mask[b].bool()] for b in range(bs)],
            dim=0,
        )  # (B, image_len, C*p*p)

        x0_pred = einops.rearrange(
            vision_pred,
            "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
            H=h_patches,
            W=w_patches,
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        )

        # Model emits an x0-prediction; convert to flow-matching velocity
        # (x_1 - x_0) so it matches the loss target from get_loss_target.
        sigma = (timestep.float() / 1000.0).clamp_min(T_EPS).to(device)
        while sigma.dim() < latent_model_input.dim():
            sigma = sigma.unsqueeze(-1)
        pred = (latent_model_input.float().to(device) - x0_pred.float()) / sigma
        return pred.to(in_dtype)

    def get_prompt_embeds(self, prompt: list) -> AdvancedPromptEmbeds:
        if not isinstance(prompt, list):
            prompt = [prompt]
        # empty, we cannot use them with this omni model anyway, but will break trainer if they do not exist
        token_list = [self.pipeline.encode_prompt(p) for p in prompt]
        pe = AdvancedPromptEmbeds(text_embeds=token_list)
        pe._frozen_dtype_keys = ["text_embeds"]
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: Qwen3VLForConditionalGeneration = unwrap_model(self.model)
        if self.is_comfy_weight:
            sd = transformer.state_dict()
            save_dict = {}
            for key, value in sd.items():
                if "lm_head.weight" in key:
                    continue  # comfy checkpoint doesnt have the lm head, so skip it
                save_dict[key] = value.clone().to("cpu", dtype=save_dtype)
            
            if not output_path.endswith(".safetensors"):
                output_path += ".safetensors"
            meta = get_meta_for_safetensors(meta, name=self.arch)
            save_file(save_dict, output_path, metadata=meta)
        else:
            transformer.save_pretrained(
                save_directory=output_path,
                safe_serialization=True,
            )

            # save processor
            self.tokenizer.save_pretrained(output_path)

            meta_path = os.path.join(output_path, "aitk_meta.yaml")
            with open(meta_path, "w") as f:
                yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        noise_scale = self.noise_scale
        return (noise * noise_scale - batch.latents).detach()

    def get_base_model_version(self):
        return self.arch

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_key = new_key.replace(".model.", ".")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.model.")
            # to load legacy keys
            new_key = new_key.replace("transformer.model.model.", "transformer.model.")
            new_sd[new_key] = value
        return new_sd
