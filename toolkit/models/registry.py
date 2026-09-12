"""Per-arch capabilities + realistic defaults, one table shared by the
inference engine (`GET /models`), its tests, and eventually the UI.

Every arch registered through get_all_models()/LEGACY_ARCHS is served; an
entry here adds defaults and capability flags on top. Archs without an entry
get GENERIC. Keep `model` limited to ModelConfig kwargs and `sample` to
GenerateImageConfig kwargs.
"""

from typing import Dict, List, Optional

IMG = {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 4.0}
VID = {"width": 832, "height": 480, "num_inference_steps": 20, "guidance_scale": 5.0, "num_frames": 33, "fps": 16}

GENERIC = {"modality": "image", "model": {}, "sample": dict(IMG)}

# modality: image | video | audio
# needs_control_image: the arch requires ctrl_img (edit / i2v models)
# size_locked: only the default resolution is valid
ARCH_REGISTRY: Dict[str, dict] = {
    "zimage": {"modality": "image", "model": {"name_or_path": "Tongyi-MAI/Z-Image-Turbo"}, "sample": {**IMG, "num_inference_steps": 8, "guidance_scale": 1.0}},
    "zimage_l2p": {"modality": "image", "model": {"name_or_path": "zhen-nan/L2P/model-1k-merge.safetensors", "extras_name_or_path": "Tongyi-MAI/Z-Image-Turbo", "quantize_te": True}, "sample": {**IMG, "num_inference_steps": 8, "guidance_scale": 1.0}},
    "qwen_image": {"modality": "image", "model": {"name_or_path": "Qwen/Qwen-Image", "quantize": True, "quantize_te": True}, "sample": {**IMG, "num_inference_steps": 20}},
    "qwen_image_edit": {"modality": "image", "model": {"name_or_path": "Qwen/Qwen-Image-Edit", "quantize": True, "quantize_te": True}, "sample": {**IMG, "num_inference_steps": 20}, "needs_control_image": True},
    "qwen_image_edit_plus": {"modality": "image", "model": {"name_or_path": "Qwen/Qwen-Image-Edit-2509", "quantize": True, "quantize_te": True}, "sample": {**IMG, "num_inference_steps": 20}, "needs_control_image": True},
    "krea2": {"modality": "image", "model": {"name_or_path": "krea/Krea-2-Turbo", "quantize": True, "quantize_te": True}, "sample": {**IMG, "num_inference_steps": 8, "guidance_scale": 1.0}},
    "boogu_image": {"modality": "image", "model": {"name_or_path": "Boogu/Boogu-Image-0.1-Base", "quantize": True, "quantize_te": True}, "sample": dict(IMG)},
    "boogu_image_edit": {"modality": "image", "model": {"name_or_path": "Boogu/Boogu-Image-0.1-Edit", "quantize": True, "quantize_te": True}, "sample": dict(IMG), "needs_control_image": True},
    "ernie_image": {"modality": "image", "model": {"name_or_path": "baidu/ERNIE-Image", "quantize": True, "quantize_te": True}, "sample": dict(IMG)},
    "mageflow": {"modality": "image", "model": {"name_or_path": "microsoft/Mage-Flow-Base", "quantize": True, "quantize_te": True}, "sample": dict(IMG)},
    "mageflow_edit": {"modality": "image", "model": {"name_or_path": "microsoft/Mage-Flow-Edit", "quantize": True, "quantize_te": True}, "sample": dict(IMG), "needs_control_image": True},
    "ideogram4": {"modality": "image", "model": {"name_or_path": "ideogram-ai/ideogram-4-fp8", "quantize": True, "quantize_te": True}, "sample": dict(IMG)},
    "hidream": {"modality": "image", "model": {"name_or_path": "HiDream-ai/HiDream-I1-Full", "quantize": True, "quantize_te": True}, "sample": {**IMG, "num_inference_steps": 28, "guidance_scale": 5.0}},
    "hidream_e1": {"modality": "image", "model": {"name_or_path": "HiDream-ai/HiDream-E1-1", "quantize": True, "quantize_te": True}, "sample": {"width": 768, "height": 768, "num_inference_steps": 28, "guidance_scale": 5.0}, "needs_control_image": True, "size_locked": True},
    "hidream_o1": {"modality": "image", "model": {"name_or_path": "HiDream-ai/HiDream-O1-Image", "quantize": True, "quantize_te": True}, "sample": {**IMG, "num_inference_steps": 28, "guidance_scale": 5.0}},
    "anima": {"modality": "image", "model": {"name_or_path": "circlestone-labs/Anima-Base-v1.0-Diffusers"}, "sample": {**IMG, "guidance_scale": 4.5}},
    "nucleus_image": {"modality": "image", "model": {"name_or_path": "NucleusAI/Nucleus-Image", "quantize": True, "quantize_te": True}, "sample": dict(IMG)},
    "omnigen2": {"modality": "image", "model": {"name_or_path": "OmniGen2/OmniGen2", "quantize": True, "quantize_te": True}, "sample": dict(IMG)},
    "chroma": {"modality": "image", "model": {"name_or_path": "lodestones/Chroma1-HD", "quantize": True, "quantize_te": True}, "sample": {**IMG, "num_inference_steps": 26}},
    "chroma_radiance": {"modality": "image", "model": {"name_or_path": "lodestones/Chroma1-Radiance", "quantize": True, "quantize_te": True}, "sample": {**IMG, "num_inference_steps": 26}},
    "zeta_chroma": {"modality": "image", "model": {"name_or_path": "lodestones/Zeta-Chroma/zeta-chroma-base-x0-pixel-dino-distance.safetensors", "extras_name_or_path": "Tongyi-MAI/Z-Image-Turbo", "quantize": True, "quantize_te": True}, "sample": dict(IMG)},
    "flux_kontext": {"modality": "image", "model": {"name_or_path": "black-forest-labs/FLUX.1-Kontext-dev", "quantize": True, "quantize_te": True}, "sample": {**IMG, "num_inference_steps": 20, "guidance_scale": 2.5}, "needs_control_image": True},
    "flux2": {"modality": "image", "model": {"name_or_path": "black-forest-labs/FLUX.2-dev", "quantize": True, "quantize_te": True}, "sample": dict(IMG)},
    "flux2_klein_4b": {"modality": "image", "model": {"name_or_path": "black-forest-labs/FLUX.2-klein-base-4B", "quantize_te": True}, "sample": dict(IMG)},
    "flux2_klein_9b": {"modality": "image", "model": {"name_or_path": "black-forest-labs/FLUX.2-klein-base-9B", "quantize": True, "quantize_te": True}, "sample": dict(IMG)},
    "prx_pixel": {"modality": "image", "model": {"name_or_path": "Photoroom/prxpixel-t2i", "quantize_te": True}, "sample": dict(IMG)},
    "f-lite": {"modality": "image", "model": {"name_or_path": "Freepik/F-Lite", "quantize": True, "quantize_te": True}, "sample": dict(IMG)},
    "sd1": {"modality": "image", "model": {"name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5"}, "sample": {"width": 512, "height": 512, "num_inference_steps": 20, "guidance_scale": 7.5}},
    "sdxl": {"modality": "image", "model": {"name_or_path": "stabilityai/stable-diffusion-xl-base-1.0"}, "sample": {**IMG, "guidance_scale": 6.0}},
    "wan21": {"modality": "video", "model": {"name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}, "sample": dict(VID)},
    "wan21_i2v": {"modality": "video", "model": {"name_or_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "quantize": True, "quantize_te": True}, "sample": dict(VID), "needs_control_image": True},
    "wan22_5b": {"modality": "video", "model": {"name_or_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "quantize": True, "quantize_te": True}, "sample": {**VID, "fps": 24}},
    "wan22_14b": {"modality": "video", "model": {"name_or_path": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16", "quantize": True, "quantize_te": True, "low_vram": True}, "sample": {**VID, "guidance_scale": 3.5}},
    "wan22_14b_i2v": {"modality": "video", "model": {"name_or_path": "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16", "quantize": True, "quantize_te": True, "low_vram": True}, "sample": {**VID, "guidance_scale": 3.5}, "needs_control_image": True},
    "ltx2.3": {"modality": "video", "model": {"name_or_path": "Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors", "quantize": True, "quantize_te": True}, "sample": {**VID, "num_inference_steps": 25, "guidance_scale": 3.0, "num_frames": 49, "fps": 24}},
    "ltx2.5": {"modality": "video", "model": {"name_or_path": "Lightricks/LTX-2.5", "quantize": True, "quantize_te": True}, "sample": {**VID, "num_inference_steps": 25, "guidance_scale": 3.0, "num_frames": 49, "fps": 24}},
    "ace_step_15": {"modality": "audio", "model": {"name_or_path": "ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_base_aio.safetensors", "quantize": True, "quantize_te": True}, "sample": {"width": 512, "height": 512, "num_inference_steps": 20, "guidance_scale": 4.0}},
}

OUTPUT_EXT = {"image": "png", "video": "mp4", "audio": "mp3"}


def get_arch_entry(arch: str) -> dict:
    # variant suffixes ("zimage:turbo") share the base arch's entry
    entry = ARCH_REGISTRY.get(arch) or ARCH_REGISTRY.get(arch.split(":")[0], GENERIC)
    return {
        "arch": arch,
        "modality": entry.get("modality", "image"),
        "model": dict(entry.get("model", {})),
        "sample": dict(entry.get("sample", IMG)),
        "needs_control_image": bool(entry.get("needs_control_image", False)),
        "size_locked": bool(entry.get("size_locked", False)),
    }


def list_archs() -> List[str]:
    from toolkit.util.get_model import LEGACY_ARCHS, get_all_models

    archs = {m.arch for m in get_all_models() if getattr(m, "arch", None)}
    archs |= LEGACY_ARCHS
    return sorted(archs)


def describe_archs(archs: Optional[List[str]] = None) -> List[dict]:
    return [get_arch_entry(a) for a in (archs or list_archs())]
