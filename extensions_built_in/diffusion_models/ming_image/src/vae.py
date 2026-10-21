"""The Ming-Image VAE: the Qwen-Image (Wan-style video) VAE with four image
channels. The ComfyUI repack ships it in Comfy's Wan-native key layout, which
the load converter maps onto the diffusers module names."""

from toolkit.models.v2.vae.qwen_image import QwenImageVAE

from .checkpoints import BASE_REPO, COMFY_REPO, COMFY_VAE_FILES, comfy_weight_names

# comfy `<block>.residual.<n>` / `shortcut` -> the diffusers resnet submodule
_RESNET_PARTS = {
    "residual.0": "norm1",
    "residual.2": "conv1",
    "residual.3": "norm2",
    "residual.6": "conv2",
    "shortcut": "conv_shortcut",
}
# comfy's `middle` Sequential is resnet, attention, resnet
_MID_PARTS = {"0": "resnets.0", "1": "attentions.0", "2": "resnets.1"}


def _resnet_suffix(inner):
    consumed = 2 if inner[0] == "residual" else 1
    return [_RESNET_PARTS[".".join(inner[:consumed])]] + inner[consumed:]


def comfy_to_diffusers_key(key: str, up_block_len: int) -> str:
    """One ComfyUI (Wan-native) VAE parameter name -> its diffusers name.
    `up_block_len` is the number of decoder entries per up block (its resnets
    plus the upsampler)."""
    parts = key.split(".")
    if parts[0] == "conv1":
        return ".".join(["quant_conv"] + parts[1:])
    if parts[0] == "conv2":
        return ".".join(["post_quant_conv"] + parts[1:])

    side, rest = parts[0], parts[1:]
    if rest[0] == "conv1":
        return ".".join([side, "conv_in"] + rest[1:])
    if rest[0] == "head":
        # head.0 is the output norm, head.2 the output conv (head.1 is SiLU)
        tail = "norm_out" if rest[1] == "0" else "conv_out"
        return ".".join([side, tail] + rest[2:])
    if rest[0] == "middle":
        part, inner = _MID_PARTS[rest[1]], rest[2:]
        if part.startswith("resnets"):
            inner = _resnet_suffix(inner)
        return ".".join([side, "mid_block", part] + inner)

    index, inner = int(rest[1]), rest[2:]
    if side == "encoder":
        # encoder.downsamples.<i>: a flat list of resnets and resamplers in
        # the same order as diffusers' down_blocks
        if inner[0] in ("resample", "time_conv"):
            return ".".join([side, "down_blocks", str(index)] + inner)
        return ".".join([side, "down_blocks", str(index)] + _resnet_suffix(inner))
    # decoder.upsamples.<i> is flat too; diffusers nests each up block's
    # resnets under it with the upsampler last
    block, j = divmod(index, up_block_len)
    if inner[0] in ("resample", "time_conv"):
        return ".".join([side, "up_blocks", str(block), "upsamplers", "0"] + inner)
    return ".".join([side, "up_blocks", str(block), "resnets", str(j)] + _resnet_suffix(inner))


def convert_comfy_vae_state_dict(state_dict):
    if "encoder.conv1.weight" not in state_dict:
        return state_dict
    # the first decoder resampler sits right after a block's resnets
    first_resample = min(
        int(k.split(".")[2])
        for k in state_dict
        if k.startswith("decoder.upsamples.") and ".resample." in k
    )
    up_block_len = first_resample + 1
    return {comfy_to_diffusers_key(k, up_block_len): v for k, v in state_dict.items()}


class MingImageVAE(QwenImageVAE):
    aitk_config_repo = BASE_REPO
    aitk_comfy_repo = COMFY_REPO
    aitk_comfy_weight_names = comfy_weight_names(COMFY_VAE_FILES)

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        return convert_comfy_vae_state_dict(state_dict)
