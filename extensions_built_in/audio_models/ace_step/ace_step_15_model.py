import json
import os
from typing import List, Optional
import huggingface_hub
import torch
from safetensors.torch import load_file, save_file
from extensions_built_in.audio_models.base_audio_model import BaseAudioModel
from toolkit.config_modules import GenerateImageConfig
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from .src.model import (
    AceStep15,
    OobleckVAE,
    TextEncoder,
    get_silence_latent,
    load_models,
)
from transformers import AutoTokenizer
from .src.pipeline import AceStep15Pipeline

scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False,
}

def to_number(str_or_number, default):
    if isinstance(str_or_number, (int, float)):
        return str_or_number
    if str_or_number is None:
        return default
    if str_or_number == "":
        return default
    try:
        return float(str_or_number)
    except ValueError:
        try:
            return int(str_or_number)
        except ValueError as e:
            raise ValueError(f"Could not convert {str_or_number} to a number") from e


def parse_ace_step_caption(text):
    """Parse a tagged caption file back into a dict."""
    import re

    def tag(name):
        m = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    return {
        "caption": tag("CAPTION"),
        "lyrics": tag("LYRICS"),
        "bpm": to_number(tag("BPM"), 120),
        "keyscale": tag("KEYSCALE"),
        "timesignature": tag("TIMESIGNATURE"),
        "duration": to_number(tag("DURATION"), 1.0),
        "language": tag("LANGUAGE"),
    }


class AceStep15Model(BaseAudioModel):
    arch = "ace_step_15"

    def __init__(
        self,
        device,
        model_config,
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
        # self.target_lora_modules = ['AceStep15']
        self.target_lora_modules = ["DiTModel"]

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def load_model(self):
        dtype = self.torch_dtype
        device = self.device_torch

        model_path = self.model_config.name_or_path

        if not os.path.exists(model_path):
            # assume it is a hf repo like org/repo/filename.safetensors
            path_parts = model_path.split("/")
            if len(path_parts) != 3:
                raise ValueError(
                    f"Model path {model_path} does not exist and is not a valid Hugging Face repo path"
                )
            model_path = huggingface_hub.hf_hub_download(
                repo_id=f"{path_parts[0]}/{path_parts[1]}",
                filename=path_parts[2],
            )
        # load the models from the single safetensors file
        models = load_models(model_path, device=device, dtype=dtype)

        self.model = models["model"]
        self.vae = models["vae"]
        self.text_encoder = models["text_encoder"]
        self.tokenizer = models["tokenizer"]
        self.pipeline = AceStep15Pipeline(
            transformer=self.model,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.get_train_scheduler(),
        )

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # the prompt should be json as a string. Try to parse it.
        json_prompts = []
        for p in prompts:
            try:
                json_prompts.append(parse_ace_step_caption(p))
            except json.JSONDecodeError:
                raise ValueError(
                    f"Prompt {p} is not a valid JSON string. Prompts must be JSON for this model"
                )

        if self.pipeline.text_encoder.device == torch.device("cpu"):
            self.pipeline.text_encoder.to(self.device_torch)

        device = self.text_encoder.device
        dtype = self.text_encoder.dtype

        batch_pe = None
        # TODO not sure this will allow for proper batching

        for json_prompt in json_prompts:
            prompt = json_prompt.get("caption", "")
            lyrics = json_prompt.get("lyrics", "")
            bpm = json_prompt.get("bpm", 120)
            key = json_prompt.get("key", "C")
            time_sig = json_prompt.get("time_sig", "4/4")
            duration = json_prompt.get("duration", 10)
            duration = int(duration) if isinstance(duration, (int, float)) else 10
            language = json_prompt.get("language", "en")

            text_embeddings, text_mask, lyric_embeddings, lyric_mask = (
                self.pipeline.get_text_embedings(
                    prompt, lyrics, bpm, key, time_sig, duration, language
                )
            )
            latent_len = int(duration * self.pipeline.LATENT_RATE)
            # Silence as source latent [1, 64, T] -> [1, T, 64] for DiT
            sil = get_silence_latent(latent_len, device, dtype)  # [1, 64, T]
            src = sil.transpose(1, 2)  # [1, T, 64]
            chunk_masks = torch.ones_like(src)

            # Reference audio (silence)
            ref = sil[:, :, :750].transpose(1, 2)  # [1, 750, 64]
            ref_order = torch.zeros(1, device=device, dtype=torch.long)
            enc_h, enc_m, ctx = self.pipeline.transformer.prepare_condition(
                text_embeddings,
                text_mask,
                lyric_embeddings,
                lyric_mask,
                ref,
                ref_order,
                src,
                chunk_masks,
            )

            pe = PromptEmbeds([[enc_h, ctx], None], attention_mask=enc_m)
            if batch_pe is None:
                batch_pe = pe
            else:
                batch_pe = concat_prompt_embeds(batch_pe, pe)
        return batch_pe

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]
    
    def get_generation_pipeline(self):
        return self.pipeline

    def generate_single_audio(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # make sure gen config is setup for audio
        if gen_config.output_ext not in ['mp3', 'wav']:
            gen_config.output_ext = 'mp3'
        prompt = gen_config.prompt
        json_prompt = parse_ace_step_caption(prompt)
        prompt = json_prompt.get("caption", "")
        lyrics = json_prompt.get("lyrics", "")
        bpm = json_prompt.get("bpm", 120)
        key = json_prompt.get("key", "C")
        time_sig = json_prompt.get("time_sig", "4/4")
        duration = json_prompt.get("duration", 0)
        language = json_prompt.get("language", "en")

        output = self.pipeline(
            prompt=None,  # we are passing in the embeds directly, so no need for a prompt
            encoder_embeddings=conditional_embeds.text_embeds[0],
            encoder_mask=conditional_embeds.attention_mask,
            encoder_context=conditional_embeds.text_embeds[1],
            num_inference_steps=gen_config.num_inference_steps,
            duration=duration,
            generator=generator,
            bpm=bpm,
            key=key,
            time_sig=time_sig,
            language=language,
            guidance_scale=gen_config.guidance_scale,
        )
        return output

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        raise NotImplementedError(
            "get_noise_prediction is not implemented for this model. Use the pipeline directly instead."
        )


class AceStep15XLModel(AceStep15Model):
    arch = "ace_step_15_xl"
