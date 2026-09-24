"""Where Ming-Image weights come from.

`name_or_path` is authoritative. COMFY_REPO, the single-file ComfyUI repack,
is the default: its int8_convrot files are the toolkit's convrot8 storage, so a
quantized load attaches them as-is, and copies already in the local ComfyUI
models folder win over the download. Anything else in `name_or_path` (the
vendor repo, a local checkpoint, a fine-tune) loads exactly what it names,
local comfy files or not. BASE_REPO is the vendor's diffusers-layout
checkpoint; its configs, tokenizer, image processor and (until the repack
carries one) the vision tower are used with the repack. The repack started
life under Kijai's account while its ComfyUI PR was in progress; that id is
rewritten to the Comfy-Org one (same files) so older configs keep working.
"""

BASE_REPO = "inclusionAI/Ming-Image-0.1-Design"
COMFY_REPO = "Comfy-Org/Ming-Image"
# earlier home of the same files
LEGACY_COMFY_REPOS = {"Kijai/Ming-Image-ComfyUI": COMFY_REPO}

# repo-relative comfy files per component; the resolver orders them by the
# requested qtype (convrot8 -> the int8 file, anything else -> bf16)
COMFY_TRANSFORMER_FILES = [
    "diffusion_models/ming_image_0.1_design_int8_convrot.safetensors",
    "diffusion_models/ming_image_0.1_design_bf16.safetensors",
]
COMFY_TEXT_ENCODER_FILES = [
    "text_encoders/ming_image_0.1_ling_mini_2.0_int8_convrot.safetensors",
    "text_encoders/ming_image_0.1_ling_mini_2.0_bf16.safetensors",
]
COMFY_VAE_FILES = ["vae/ming_image_vae_bf16.safetensors"]


def canonical_repo(name_or_path: str) -> str:
    """The current id for a repack that has moved; anything else unchanged."""
    return LEGACY_COMFY_REPOS.get(name_or_path, name_or_path)


def comfy_weight_names(files):
    """OstrisModelMixin candidate map: only the repack id (current or legacy)
    resolves to these files; every other name_or_path loads what it names."""
    return {repo: list(files) for repo in (COMFY_REPO, *LEGACY_COMFY_REPOS)}
