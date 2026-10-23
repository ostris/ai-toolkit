"""Where Fibo 2.0 weights come from.

Weights live in the ComfyUI layout under MODELS_PATH, named like the
Comfy-Org/Fibo-2.0 repack (the vendor repo briaai/Fibo-2.0-dev is gated and
diffusers-keyed; not a source here).

- ``Comfy-Org/Fibo-2.0`` (the UI default): the dev DiT, int8 convrot or bf16,
  whichever is local (ranked for the requested qtype), else the best-ranked
  one is downloaded.
- ``org/repo/path/file.safetensors`` or a local file: that file (fine-tunes
  saved by the toolkit are in the same layout).

The text encoder is stock Qwen3-VL-4B-Instruct (the repack's bf16 is byte
identical to Comfy-Org/Qwen3-VL's), so config + tokenizer come from the Qwen
repo. The VAE is the shared FLUX.2 file.
"""

import os
from typing import Callable, Optional

from toolkit.models.v2.resolver import comfy_precision_rank, resolve_comfy_candidates

COMFY_REPO = "Comfy-Org/Fibo-2.0"
QWEN3_VL_4B_REPO = "Qwen/Qwen3-VL-4B-Instruct"

COMFY_TRANSFORMER_FILES = [
    "diffusion_models/fibo2_dev_int8_convrot.safetensors",
    "diffusion_models/fibo2_dev_bf16.safetensors",
]
COMFY_TEXT_ENCODER_FILES = [
    "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
    "text_encoders/qwen3vl_4b_bf16.safetensors",
]
COMFY_VAE_FILE = "vae/flux2-vae.safetensors"


def resolve_hub_file(
    ref: str,
    hf_token: Optional[str] = None,
    status_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """A local path, or 'org/repo/rel/file.safetensors' found at / downloaded
    to its comfy spot under MODELS_PATH."""
    if os.path.isfile(ref):
        return ref
    parts = ref.split("/")
    if len(parts) < 3:
        raise FileNotFoundError(
            f"{ref} is not a local file or an 'org/repo/path/file.safetensors' reference"
        )
    return resolve_comfy_candidates(
        ["/".join(parts[2:])],
        repo_id="/".join(parts[:2]),
        hf_token=hf_token,
        status_fn=status_fn,
    )


def _best_comfy_file(candidates, qtype, hf_token, status_fn) -> str:
    found = resolve_comfy_candidates(
        candidates, repo_id=COMFY_REPO, local_only=True, qtype=qtype
    )
    if found is not None:
        return found
    best = min(candidates, key=lambda c: comfy_precision_rank(c, qtype=qtype))
    return resolve_comfy_candidates(
        [best], repo_id=COMFY_REPO, hf_token=hf_token, status_fn=status_fn
    )


def comfy_transformer_file(
    name_or_path: str,
    qtype: Optional[str] = None,
    hf_token: Optional[str] = None,
    status_fn: Optional[Callable[[str], None]] = None,
) -> str:
    if name_or_path.endswith(".safetensors"):
        return resolve_hub_file(name_or_path, hf_token=hf_token, status_fn=status_fn)
    if name_or_path != COMFY_REPO:
        raise ValueError(
            f"Fibo 2.0 loads {COMFY_REPO} or a comfy-layout .safetensors file, not {name_or_path}"
        )
    return _best_comfy_file(COMFY_TRANSFORMER_FILES, qtype, hf_token, status_fn)


def comfy_text_encoder_file(
    qtype: Optional[str] = None,
    hf_token: Optional[str] = None,
    status_fn: Optional[Callable[[str], None]] = None,
) -> str:
    return _best_comfy_file(COMFY_TEXT_ENCODER_FILES, qtype, hf_token, status_fn)


def comfy_vae_file(
    hf_token: Optional[str] = None,
    status_fn: Optional[Callable[[str], None]] = None,
) -> str:
    return resolve_comfy_candidates(
        [COMFY_VAE_FILE], repo_id=COMFY_REPO, hf_token=hf_token, status_fn=status_fn
    )
