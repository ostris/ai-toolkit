from .flux2_model import Flux2Model, HF_TOKEN
from transformers import Qwen3ForCausalLM, Qwen2Tokenizer
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush
from .src.model import Klein9BParams, Klein4BParams


class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "ai-toolkit/flux2_vae"
    flux2_is_guidance_distilled: bool = False

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
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs,
        )
        # use the new format on this new model by default
        self.use_old_lokr_format = False
        # Allow overriding the default Qwen3 checkpoint via model config (e.g. for
        # community fine-tunes or gated HF repos). te_name_or_path is the canonical
        # ModelConfig field; if set it takes precedence over the class-level default.
        if model_config.te_name_or_path:
            self.flux2_klein_te_path = model_config.te_name_or_path

    def load_te(self):
        if self.flux2_klein_te_path is None:
            raise ValueError("flux2_klein_te_path must be set for Flux2KleinModel")
        dtype = self.torch_dtype
        self.print_and_status_update(f"Loading Qwen3 ({self.flux2_klein_te_path})")

        text_encoder: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained(
            self.flux2_klein_te_path,
            torch_dtype=dtype,
            token=HF_TOKEN,
        )
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Qwen3")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()
        elif not self.model_config.low_vram:
            text_encoder.to(self.device_torch, dtype=dtype)
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

        # Some community fine-tunes (abliterated, merged, etc.) strip chat_template
        # from tokenizer_config.json. The transformers Jinja renderer then falls back
        # to a generic template that calls .startswith() on a bool and crashes at
        # inference time. Detect the missing template early and fall back to loading
        # the tokenizer from the canonical Qwen3 class default, which is always complete.
        # The model weights loaded above are unaffected.
        tokenizer = Qwen2Tokenizer.from_pretrained(
            self.flux2_klein_te_path, token=HF_TOKEN
        )
        if not getattr(tokenizer, "chat_template", None):
            fallback_path = self.__class__.flux2_klein_te_path
            self.print_and_status_update(
                f"Tokenizer at '{self.flux2_klein_te_path}' has no chat_template; "
                f"falling back to '{fallback_path}' for tokenizer"
            )
            tokenizer = Qwen2Tokenizer.from_pretrained(fallback_path, token=HF_TOKEN)
        return text_encoder, tokenizer

    def get_prompt_embeds(self, prompt, **kwargs):
        # Guard against None or non-string values (e.g. an empty negative prompt
        # passed as False) that cause the Qwen3 chat template to crash.
        if not isinstance(prompt, str):
            prompt = "" if not prompt else str(prompt)
        return super().get_prompt_embeds(prompt, **kwargs)


class Flux2Klein4BModel(Flux2KleinModel):
    arch = "flux2_klein_4b"
    flux2_klein_te_path: str = "Qwen/Qwen3-4B"
    flux2_te_filename: str = "flux-2-klein-base-4b.safetensors"

    def get_flux2_params(self):
        return Klein4BParams()

    def get_base_model_version(self):
        return "flux2_klein_4b"


class Flux2Klein9BModel(Flux2KleinModel):
    arch = "flux2_klein_9b"
    flux2_klein_te_path: str = "Qwen/Qwen3-8B"
    flux2_te_filename: str = "flux-2-klein-base-9b.safetensors"

    def get_flux2_params(self):
        return Klein9BParams()

    def get_base_model_version(self):
        return "flux2_klein_9b"
