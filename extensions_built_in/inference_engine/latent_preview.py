from typing import Optional

from .latent_format_tables import LATENT_FORMATS

# arch -> preview table. Archs sharing a VAE family share a table; anything
# not listed falls back to a channel-count guess in `guess_format`.
ARCH_LATENT_FORMAT = {
    "sd1": "SD15",
    "sd2": "SD15",
    "sdxl": "SDXL",
    "ssd": "SDXL",
    "vega": "SDXL",
    "sd3": "SD3",
    "flux": "Flux",
    "flux_kontext": "Flux",
    "chroma": "Flux",
    "chroma_radiance": None,  # pixel space
    "zeta_chroma": "Flux",
    "zimage": "Flux",
    "zimage_l2p": "Flux",
    "krea2": "Wan21",  # Qwen-Image VAE
    "hidream": "Flux",
    "hidream_e1": "Flux",
    "hidream_o1": "Flux",
    "omnigen2": "Flux",
    "f-lite": "Flux",
    "nucleus_image": "Wan21",  # Qwen-Image VAE
    "qwen_image": "Wan21",
    "qwen_image_edit": "Wan21",
    "qwen_image_edit_plus": "Wan21",
    "boogu_image": "Wan21",
    "boogu_image_edit": "Wan21",
    "flux2": "Flux2",
    "flux2_klein_4b": "Flux2",
    "flux2_klein_9b": "Flux2",
    "ideogram4": "Flux2",
    "mageflow": "Wan21",
    "mageflow_edit": "Wan21",
    "wan21": "Wan21",
    "wan21_i2v": "Wan21",
    "wan22_5b": "Wan22",
    "wan22_14b": "Wan21",
    "wan22_14b_i2v": "Wan21",
    "ltx2": "LTXAV",
    "ltx2.3": "LTXAV",
    "ltx2.5": "LTXAV",
    "minimax_h3": "MiniMaxH3Video",
    "minimax_h3_ref2va": "MiniMaxH3Video",
    "minimax_h3_vsa": "MiniMaxH3Video",
    "anima": "Wan21",
    "cogview4": "Flux",
    "prx_pixel": None,  # pixel space
}


def guess_format(channels: int, dims: int) -> Optional[str]:
    if dims == 2:
        return {4: "SD15", 16: "Flux", 128: "Flux2"}.get(channels)
    if dims == 3:
        return {16: "Wan21", 48: "Wan22", 128: "LTXAV", 24: "MiniMaxH3Video"}.get(channels)
    return None


def preview_info(arch: str, channels: Optional[int] = None, dims: Optional[int] = None) -> Optional[dict]:
    """The `start` frame's preview block: layout + factors, or None when the
    latent cannot be previewed linearly (pixel-space models, audio)."""
    # UI arch names may carry a variant suffix ("zimage:turbo", "wan21:1b")
    arch = (arch or "").split(":")[0]
    name = ARCH_LATENT_FORMAT.get(arch, "__unset__")
    if name is None:
        return None
    if name == "__unset__" or (channels is not None and LATENT_FORMATS[name]["channels"] != channels):
        name = guess_format(channels, dims) if channels is not None else None
    if name is None:
        return None
    fmt = LATENT_FORMATS[name]
    return {
        "format": name,
        "channels": fmt["channels"],
        "dims": fmt["dims"],
        "spatial": fmt["spatial"],
        "temporal": fmt["temporal"],
        "reshape": fmt["reshape"],
        "factors": fmt["factors"],
        "bias": fmt["bias"],
    }
