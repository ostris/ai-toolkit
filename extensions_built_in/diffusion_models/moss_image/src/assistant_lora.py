"""Fixed MOSS training adapters, merged once and reversed for previews."""

import hashlib
import math
import os

import torch
from safetensors.torch import load_file

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork

from .checkpoints import resolve_hub_file


def _merge_seed(name):
    """Stable per-projection seed, independent of Python's randomized hash."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


@torch.no_grad()
def merge_training_adapter(network, strength):
    """Merge once; preserve small signed updates on pre-quantized ConvRot8."""
    from toolkit.util.convrot_quant import ConvRotInt8Quantizer

    network.is_merged_in = True
    stochastic_modules = 0
    for module in network.get_all_modules():
        original = module.org_module[0]
        quantizer = getattr(original, "ostris_quantizer", None)
        # Other quantizers (including packed ConvRot variants) retain their
        # existing merge path. Their storage and quantization rules differ.
        if type(quantizer) is not ConvRotInt8Quantizer:
            module.merge_in(merge_weight=strength)
            continue
        up, down = module.lora_up.weight, module.lora_down.weight
        if not up.any() or not down.any():
            continue  # Do not perturb the base when the adapter is zero.
        weight = quantizer.dequantize(original)
        delta = up.to(device=weight.device, dtype=torch.float32) @ down.to(device=weight.device, dtype=torch.float32)
        weight.add_(delta, alpha=float(strength * module.scale))
        quantizer.requantize_(original, weight, stochastic_seed=_merge_seed(module.lora_name))
        stochastic_modules += 1
    return stochastic_modules


def prepare_adapter_weights(model, transformer, state_dict):
    """Normalize standard LoRA scales before MOSS fuses independent Q/K/V.

    Use per-module ranks: an independent Q/K/V adapter becomes rank 3r on
    the fused projection, while its other projections remain rank r.
    Reject unsupported or mismatched tensors rather than silently resizing
    or partially loading a fixed adapter intended to cancel an earlier merge.
    """
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("diffusion_model."):
            key = "transformer." + key[len("diffusion_model."):]
        key = key.replace(".lora_down.weight", ".lora_A.weight")
        key = key.replace(".lora_up.weight", ".lora_B.weight")
        if key in normalized:
            raise ValueError(f"Duplicate MOSS assistant LoRA key: {key}")
        normalized[key] = value

    pairs = {}
    used = set()
    for key, down in normalized.items():
        if not key.endswith(".lora_A.weight"):
            continue
        name = key[:-len(".lora_A.weight")]
        up_key, alpha_key = name + ".lora_B.weight", name + ".alpha"
        up = normalized.get(up_key)
        if down.ndim != 2 or down.shape[0] == 0 or up is None or up.ndim != 2:
            raise ValueError(f"Expected paired linear LoRA weights for {name}")
        rank = down.shape[0]
        if up.shape[1] != rank:
            raise ValueError(f"Mismatched LoRA rank for {name}")
        alpha = normalized.get(alpha_key)
        if alpha is not None and alpha.numel() != 1:
            raise ValueError(f"Expected scalar LoRA alpha for {name}")
        alpha = float(alpha) if alpha is not None else rank
        if not math.isfinite(alpha):
            raise ValueError(f"Expected finite LoRA alpha for {name}")
        # Toolkit treats alpha=0 as rank. Fold scales into B so PEFT loading
        # and independent Q/K/V fusion preserve each projection's delta.
        scale = (alpha or rank) / rank
        pairs[key] = down
        pairs[up_key] = up.float() * scale
        used.update((key, up_key))
        if alpha_key in normalized:
            used.add(alpha_key)
    if not pairs or used != normalized.keys():
        unsupported = sorted(normalized.keys() - used)
        raise ValueError(f"MOSS assistant requires a standard linear LoRA; unsupported keys: {unsupported[:5]}")

    weights = model.convert_lora_weights_before_load(pairs)
    modules = dict(transformer.named_modules())
    dims = {}
    for key, down in weights.items():
        if not key.endswith(".lora_A.weight"):
            continue
        name = key[:-len(".lora_A.weight")]
        module = modules.get(name.removeprefix("transformer."))
        up = weights[name + ".lora_B.weight"]
        if (
            not name.startswith("transformer.")
            or module is None
            or down.shape[1] != getattr(module, "in_features", None)
            or up.shape[0] != getattr(module, "out_features", None)
        ):
            raise ValueError(f"MOSS assistant LoRA does not match transformer module {name}")
        dims[name.replace(".", "$$")] = down.shape[0]
    return weights, dims


@torch.no_grad()
def load_training_adapter(model, transformer):
    strength = model.model_config.assistant_lora_strength
    if strength == 0:
        # No adapter is needed, including for previews.
        model.model_config.assistant_lora_path = None
        return
    path = resolve_hub_file(
        model.model_config.assistant_lora_path,
        hf_token=os.getenv("HF_TOKEN"),
        status_fn=model.print_and_status_update,
    )
    state_dict, dims = prepare_adapter_weights(model, transformer, load_file(path))
    config = NetworkConfig(type="lora", linear=1, linear_alpha=1)
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=transformer,
        train_text_encoder=False,
        train_unet=True,
        is_transformer=True,
        target_lin_modules=list(model.target_lora_modules),
        modules_dim=dims,
        modules_alpha=dims,
        network_config=config,
        base_model=model,
        is_assistant_adapter=True,
        is_ara=True,
    )
    # Register and load strictly: the general resume loader can resize ranks
    # or skip keys, which would be inappropriate for a fixed inverse adapter.
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    peft_weights = {
        k.replace(".", "$$")
        .replace("$$lora_A$$weight", ".lora_down.weight")
        .replace("$$lora_B$$weight", ".lora_up.weight"): v
        for k, v in state_dict.items()
    }
    peft_weights.update({k: v for k, v in network.state_dict().items() if k.endswith(".alpha")})
    network.load_state_dict(peft_weights, strict=True)
    network.eval().requires_grad_(False)
    network._update_torch_multiplier()
    model.print_and_status_update(f"Merging MOSS training adapter at strength {strength:g}")
    stochastic_modules = merge_training_adapter(network, strength)
    if stochastic_modules:
        model.print_and_status_update(
            f" - seeded stochastic rounding: {stochastic_modules} pre-quantized ConvRot8 projections"
        )

    # Keep the inverse on CPU during training. Previews activate it through
    # BaseModel's existing assistant path; no repeated quantized merges.
    network.is_merged_in = False
    network.multiplier = -strength
    network.is_active = False
    network.force_to("cpu", model.torch_dtype)
    model.assistant_lora = network
    model.invert_assistant_lora = True
