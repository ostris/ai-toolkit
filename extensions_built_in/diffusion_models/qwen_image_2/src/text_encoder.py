"""Qwen3-VL-8B conditioning stack for Qwen-Image 2.1.

The Comfy-Org repack keeps the transformers key layout except that the language
tower sits directly under `model.` instead of `model.language_model.`, so the
single-file load only needs that prefix rewrite before the toolkit's comfy
quantization importer attaches the int8/convrot layers.
"""

from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder

# comfy prefix -> transformers prefix for the language tower
_COMFY_PREFIXES = (
    ("model.layers.", "model.language_model.layers."),
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.norm.", "model.language_model.norm."),
)


class QwenImage21TextEncoder(Qwen3VLTextEncoder):
    aitk_config_repo = "Qwen/Qwen-Image-2.1"
    aitk_tokenizer_subfolder = "processor"
    aitk_processor_repo = "Qwen/Qwen-Image-2.1"
    aitk_processor_subfolder = "processor"

    aitk_comfy_repo = "Comfy-Org/Qwen-Image-2.1"
    # convrot8 first: it is the toolkit's default qtype, so a matching
    # pre-quantized file attaches as-is. The w4a8 file is a different backend
    # and would have to be dequantized, so it is not a candidate.
    _COMFY_FILES = [
        "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "text_encoders/qwen3vl_8b_bf16.safetensors",
    ]
    aitk_comfy_weight_names = {
        ("Comfy-Org/Qwen-Image-2.1", "text_encoder"): _COMFY_FILES,
        ("Qwen/Qwen-Image-2.1", "text_encoder"): _COMFY_FILES,
    }

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        if not any(k.startswith("model.layers.") for k in state_dict):
            return state_dict
        converted = {}
        for key, value in state_dict.items():
            for comfy_prefix, hf_prefix in _COMFY_PREFIXES:
                if key.startswith(comfy_prefix):
                    key = hf_prefix + key[len(comfy_prefix) :]
                    break
            converted[key] = value
        return converted

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        converted = {}
        for key, value in state_dict.items():
            for comfy_prefix, hf_prefix in _COMFY_PREFIXES:
                if key.startswith(hf_prefix):
                    key = comfy_prefix + key[len(hf_prefix) :]
                    break
            converted[key] = value
        return converted
