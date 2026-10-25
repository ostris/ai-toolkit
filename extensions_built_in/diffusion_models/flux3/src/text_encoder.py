"""FLUX.3 text conditioning -- Qwen3-VL-4B hidden-state stack.

The DiT text context is the stack of a fixed set of Qwen3-VL hidden layers
along the channel dim: ``ctx = (B, L, len(SELECT_LAYERS) * 2560 = 20480)``.
Prompts go through the chat template and are right-padded to a multiple of
``pad_multiple`` tokens (the package's ``text_encoder.pad_multiple``); the
model was trained with the pad tokens in-context and the DiT has no text
attention mask. FLUX.3 feeds the DiT zero ``timesteps_ctx`` and no pooled
vector.
"""

import math
from typing import List, Optional

import torch
from einops import rearrange

from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder

SELECT_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
CONTEXT_DIM = 20480  # len(SELECT_LAYERS) * qwen hidden (2560)

# package defaults (text_encoder section of config.json)
DEFAULT_PAD_MULTIPLE = 80
DEFAULT_MAX_TOKENS = 8192

QWEN3_VL_4B_REPO = "Qwen/Qwen3-VL-4B-Instruct"


class Flux3TextEncoder(Qwen3VLTextEncoder):
    """Qwen3-VL-4B-Instruct. The package's text_encoder/ and tokenizer/ are
    byte-identical to the public Qwen repo, so config + tokenizer come from
    there and the weights from the Comfy-Org single-file repack under
    MODELS_PATH/text_encoders/."""

    aitk_config_repo = QWEN3_VL_4B_REPO
    aitk_tokenizer_repo = QWEN3_VL_4B_REPO
    aitk_tokenizer_subfolder = ""
    aitk_processor_repo = QWEN3_VL_4B_REPO
    aitk_processor_subfolder = ""

    aitk_comfy_repo = "Comfy-Org/Qwen3-VL"
    _COMFY_FILES = [
        "text_encoders/qwen3vl_4b_bf16.safetensors",
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
    ]
    aitk_comfy_weight_names = {QWEN3_VL_4B_REPO: _COMFY_FILES}

    # the fp8 repack keeps the language tower directly under ``model.``
    _COMFY_PREFIXES = (
        ("model.layers.", "model.language_model.layers."),
        ("model.embed_tokens.", "model.language_model.embed_tokens."),
        ("model.norm.", "model.language_model.norm."),
    )

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        if any(k.startswith("model.layers.") for k in state_dict):
            converted = {}
            for key, value in state_dict.items():
                for comfy_prefix, hf_prefix in cls._COMFY_PREFIXES:
                    if key.startswith(comfy_prefix):
                        key = hf_prefix + key[len(comfy_prefix) :]
                        break
                converted[key] = value
            state_dict = converted
        # comfy files drop the tied lm_head; strict load wants the key
        embed = state_dict.get("model.language_model.embed_tokens.weight")
        if embed is not None and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = embed
        return state_dict

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        converted = {}
        for key, value in state_dict.items():
            for comfy_prefix, hf_prefix in cls._COMFY_PREFIXES:
                if key.startswith(hf_prefix):
                    key = comfy_prefix + key[len(hf_prefix) :]
                    break
            converted[key] = value
        return converted


def padded_token_length(
    tokenizer,
    texts: List[str],
    pad_multiple: int,
    max_tokens: int,
) -> int:
    """Common padded length for a batch: the longest prompt rounded up to a
    multiple of ``pad_multiple``, capped at ``max_tokens``."""
    longest = 0
    for text in texts:
        real = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_tokens
        )["input_ids"].shape[1]
        longest = max(longest, real)
    return min(math.ceil(longest / pad_multiple) * pad_multiple, max_tokens)


@torch.no_grad()
def encode_flux3_prompt(
    text_encoder,
    processor,
    prompts: List[str],
    max_length: int = DEFAULT_MAX_TOKENS,
    dtype: torch.dtype = torch.bfloat16,
    pad_multiple: Optional[int] = DEFAULT_PAD_MULTIPLE,
) -> torch.Tensor:
    """Prompts -> text context ``(B, L, 20480)``.

    ``text_encoder`` is a ``Qwen3VLForConditionalGeneration``; ``processor`` the
    matching ``AutoProcessor``. Prompts are wrapped in the chat template with a
    generation prompt and right-padded (matching how the DiT was trained -- do
    not strip padding). ``L`` is the batch's longest prompt rounded up to a
    multiple of ``pad_multiple``, capped at ``max_length``; with
    ``pad_multiple=None`` every prompt is padded to exactly ``max_length``.
    """
    messages = [[{"role": "user", "content": p}] for p in prompts]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    tokenizer = processor.tokenizer
    if pad_multiple is not None:
        max_length = padded_token_length(tokenizer, texts, pad_multiple, max_length)
    input_ids_list, attn_list = [], []
    for text in texts:
        toks = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids_list.append(toks["input_ids"])
        attn_list.append(toks["attention_mask"])
    input_ids = torch.cat(input_ids_list, dim=0).to(text_encoder.device)
    attention_mask = torch.cat(attn_list, dim=0).to(text_encoder.device)

    out = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    stacked = torch.stack([out.hidden_states[k] for k in SELECT_LAYERS], dim=1)
    result = rearrange(stacked, "b c l d -> b l (c d)")
    return result.to(dtype)
