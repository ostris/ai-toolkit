"""Packing / sampling helpers for Ideogram 4.

This module holds the glue that turns image latents + Qwen3-VL text features into
the single packed sequence the transformer consumes, plus a minimal flow-matching
sampling pipeline used to render preview images during training.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import math
from typing import Any, List, Optional

import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from transformers.masking_utils import create_causal_mask

from .transformer import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
    SEQUENCE_PADDING_INDICATOR,
    Ideogram4Transformer2DModel,
)

_LOGSNR_MIN = -15.0
_LOGSNR_MAX = 18.0


def _logit_normal_schedule(
    u: torch.Tensor,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Reference Ideogram time schedule, where 0 is noise and 1 is clean."""
    u = torch.as_tensor(u, dtype=torch.float64)
    t = 1.0 - torch.special.expit(mean + std * torch.special.ndtri(u))
    t_min = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MAX))
    t_max = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MIN))
    return t.clamp(t_min, t_max)


def get_ideogram4_sigmas(
    num_steps: int,
    width: int,
    height: int,
    mu: float = 0.0,
    std: float = 1.75,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the resolution-aware sigma schedule used by ComfyUI/Ideogram."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if std <= 0:
        raise ValueError("std must be positive")

    mean = mu + 0.5 * math.log((width * height) / (512 * 512))
    u = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float64)
    sigmas = (1.0 - _logit_normal_schedule(u, mean, std)).flip(0)
    sigmas[-1] = 0.0
    return sigmas.to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Latent (un)patchification.
#
# The VAE produces (B, ae_ch=32, H/8, W/8) latents. The transformer works on
# tokens of dim ae_ch * patch**2 = 128. We store the patchified latent in a 4-D
# (B, 128, gh, gw) layout so the rest of ai-toolkit (noise, add_noise, loss) can
# treat it like an ordinary image latent. The channel ordering here matches the
# reference Ideogram 4 decode exactly: 128 = (patch_h, patch_w, ae_ch) with ae_ch
# the fastest-varying axis.
# ---------------------------------------------------------------------------


def patchify_latents(z: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """(B, ae_ch, H8, W8) -> (B, ae_ch * patch**2, gh, gw)."""
    b, ae_ch, h8, w8 = z.shape
    ph = pw = patch_size
    gh, gw = h8 // ph, w8 // pw
    z = z.view(b, ae_ch, gh, ph, gw, pw)
    # -> (B, ph, pw, ae_ch, gh, gw) then merge (ph, pw, ae_ch) -> channels
    z = z.permute(0, 3, 5, 1, 2, 4).reshape(b, ph * pw * ae_ch, gh, gw)
    return z


def unpatchify_latents(z: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """(B, ae_ch * patch**2, gh, gw) -> (B, ae_ch, H8, W8)."""
    b, c, gh, gw = z.shape
    ph = pw = patch_size
    ae_ch = c // (ph * pw)
    z = z.view(b, ph, pw, ae_ch, gh, gw)
    # -> (B, ae_ch, gh, ph, gw, pw) then merge spatial
    z = z.permute(0, 3, 4, 1, 5, 2).reshape(b, ae_ch, gh * ph, gw * pw)
    return z


# ---------------------------------------------------------------------------
# Qwen3-VL hidden-state extraction.
# ---------------------------------------------------------------------------


def _call_activator(
    activator: Any,
    method_names: tuple[str, ...],
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Call the first supported activator hook without imposing a hard API."""
    for method_name in method_names:
        method = getattr(activator, method_name, None)
        if not callable(method):
            continue
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        leading_args = (value,)
        if method_name == "apply_tap" and "tap_layer" in kwargs:
            leading_args = (kwargs["tap_layer"], value)
        call_kwargs = dict(kwargs)
        if method_name == "apply_tap":
            call_kwargs.pop("tap_layer", None)
        if signature is not None:
            accepted = signature.parameters
            call_kwargs = {
                key: item for key, item in call_kwargs.items() if key in accepted
            }
        result = method(*leading_args, **call_kwargs)
        if result is None:
            return value
        if isinstance(result, dict):
            for key in ("hidden_states", "inputs_embeds", "embeddings", "output"):
                if key in result:
                    return result[key]
        if isinstance(result, (tuple, list)):
            return result[0]
        return result
    return value


def _activator_runtime_context(
    activator: Any,
    runtime_mode: Optional[str],
    trigger_mask: Optional[torch.Tensor],
):
    if activator is None:
        return contextlib.nullcontext()

    stack = contextlib.ExitStack()
    runtime_module = None
    try:
        runtime_module = importlib.import_module("toolkit.trigger_binding")
    except ImportError:
        pass
    activator_module = None
    try:
        activator_module = importlib.import_module(
            "toolkit.models.ideogram4_trigger_activator"
        )
    except ImportError:
        pass

    trigger_runtime = getattr(activator_module, "trigger_runtime", None)
    if callable(trigger_runtime):
        stack.enter_context(
            trigger_runtime(
                {
                    "token_mask": trigger_mask,
                    "trigger_mask": trigger_mask,
                    "runtime_mode": runtime_mode,
                }
            )
        )
    mode_context = getattr(runtime_module, "activator_runtime_mode", None)
    if callable(mode_context) and runtime_mode is not None:
        stack.enter_context(mode_context(activator, runtime_mode))
    else:
        for method_name in ("runtime", "runtime_context", "use_runtime_mode"):
            method = getattr(activator, method_name, None)
            if callable(method):
                try:
                    stack.enter_context(method(runtime_mode))
                except TypeError:
                    stack.enter_context(method(mode=runtime_mode))
                break
    return stack


def _runtime_component_enabled(runtime_mode: Optional[str], component: str) -> bool:
    if runtime_mode is None:
        return True
    try:
        module = importlib.import_module("toolkit.trigger_binding")
        state = module.get_activator_runtime_state(runtime_mode)
    except (ImportError, AttributeError):
        return runtime_mode not in ("activator_bypass", "stock_literal")
    return bool(getattr(state, f"{component}_enabled", False))


def _adapt_text_activator(activator: Any) -> Any:
    """Give optional toolkit modules a chance to adapt their evolving runtime API."""
    if activator is None:
        return None
    for module_name in (
        "toolkit.trigger_binding",
        "toolkit.models.ideogram4_trigger_activator",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for adapter_name in (
            "adapt_ideogram4_text_activator",
            "adapt_text_activator",
            "ensure_ideogram4_text_activator",
        ):
            adapter = getattr(module, adapter_name, None)
            if callable(adapter):
                adapted = adapter(activator)
                if adapted is not None:
                    activator = adapted
                break
    return activator


def get_qwen3_vl_features(
    text_encoder,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_2d: torch.Tensor,
    trigger_mask: Optional[torch.Tensor] = None,
    text_activator: Any = None,
    runtime_mode: Optional[str] = None,
    return_taps: bool = False,
):
    """Run Qwen3-VL and optionally apply trigger-selective text activation.

    Gradient ownership belongs to the caller. With no activator and
    ``return_taps=False`` this returns the original concatenated feature tensor.
    """
    language_model = text_encoder.language_model
    text_activator = _adapt_text_activator(text_activator)
    activator_kwargs = {
        "runtime_mode": runtime_mode,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
    }

    with _activator_runtime_context(text_activator, runtime_mode, trigger_mask):
        lookup_ids = token_ids
        atomic_token_id = getattr(text_activator, "atomic_token_id", None)
        lookup_token_id = getattr(text_activator, "lookup_token_id", None)
        if atomic_token_id is not None and lookup_token_id is not None:
            lookup_ids = token_ids.masked_fill(
                token_ids == int(atomic_token_id), int(lookup_token_id)
            )
        inputs_embeds = language_model.embed_tokens(lookup_ids)
        if text_activator is not None and _runtime_component_enabled(
            runtime_mode, "embedding"
        ):
            inputs_embeds = _call_activator(
                text_activator,
                (
                    "apply_embedding",
                    "override_embeddings",
                    "apply_embedding_override",
                    "apply_embeddings",
                ),
                inputs_embeds,
                token_mask=trigger_mask,
                **activator_kwargs,
            )

        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language_model.rotary_emb(
            inputs_embeds, mrope_position_ids
        )

        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured: dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if text_activator is not None and _runtime_component_enabled(
                runtime_mode, "internal"
            ):
                hidden_states = _call_activator(
                    text_activator,
                    (
                        "apply_te_adapter",
                        "apply_internal_adapter",
                        "apply_hidden_states",
                    ),
                    hidden_states,
                    token_mask=trigger_mask,
                    layer_idx=layer_idx,
                    **activator_kwargs,
                )
            if layer_idx in tap_set:
                tap = hidden_states
                if text_activator is not None and _runtime_component_enabled(
                    runtime_mode, "tap"
                ):
                    tap = _call_activator(
                        text_activator,
                        (
                            "apply_tap",
                            "adapt_tap",
                            "apply_tap_adapter",
                            "apply_pre_concat",
                        ),
                        tap,
                        tap_layer=layer_idx,
                        token_mask=trigger_mask,
                        layer_idx=layer_idx,
                        tap_index=QWEN3_VL_ACTIVATION_LAYERS.index(layer_idx),
                        **activator_kwargs,
                    )
                captured[layer_idx] = tap

    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    batch_size, seq_len = token_ids.shape
    stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))  # (B, L, H, num_taps)
    stacked = stacked.reshape(batch_size, seq_len, -1)

    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    stacked = stacked * text_mask
    if return_taps:
        return stacked, selected
    return stacked


# ---------------------------------------------------------------------------
# Packing + velocity prediction.
# ---------------------------------------------------------------------------


def pad_text_features(
    features_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of per-sample (Lt_i, D) features into a batch.

    Captions are stored at their natural length (one tensor per batch item) and
    only padded to the batch max here, right before the model call. Returns
    ``(features (B, Lt, D), attention_mask (B, Lt))``; the mask is 1 for real
    tokens and 0 for padding (which the transformer masks out anyway).
    """
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, f in enumerate(features_list):
        n = f.shape[0]
        features[i, :n] = f.to(device, dtype)
        mask[i, :n] = 1
    return features, mask


def predict_velocity(
    transformer: Ideogram4Transformer2DModel,
    latents: torch.Tensor,  # (B, 128, gh, gw)
    t: torch.Tensor,  # (B,) toolkit flow time in [0, 1] (1 = pure noise)
    llm_features: torch.Tensor,  # (B, Lt, llm_dim)
    text_mask: torch.Tensor,  # (B, Lt) 1 for real text tokens
) -> torch.Tensor:
    """Run the transformer on the packed [text | image] sequence.

    ``t`` is in the ai-toolkit flow-matching convention: ``t=1`` is pure noise,
    ``t=0`` is clean, and the returned velocity is ``noise - clean`` (matching the
    toolkit scheduler / loss target).

    Ideogram's transformer uses the opposite convention internally (``t=1`` is
    clean) and predicts ``clean - noise``, so we feed it ``1 - t`` and negate its
    output. Returns the velocity reshaped to the (B, 128, gh, gw) latent layout.
    """
    device = latents.device
    b, c, gh, gw = latents.shape
    num_image_tokens = gh * gw
    num_text_tokens = llm_features.shape[1]
    seq_len = num_text_tokens + num_image_tokens

    # image latents -> tokens (row-major: h outer, w inner)
    image_tokens = latents.permute(0, 2, 3, 1).reshape(b, num_image_tokens, c)

    # The mask may arrive as a float (PromptEmbeds.to casts it to the embed
    # dtype); work in long so cumsum positions stay exact for long prompts.
    text_mask_bool = text_mask.to(device) > 0
    text_mask_long = text_mask_bool.long()

    # noise tokens: text region is zeroed (masked out anyway)
    x = torch.cat(
        [
            torch.zeros(b, num_text_tokens, c, device=device, dtype=image_tokens.dtype),
            image_tokens,
        ],
        dim=1,
    )

    # llm features: image region is zero
    llm_full = torch.cat(
        [
            llm_features,
            torch.zeros(
                b,
                num_image_tokens,
                llm_features.shape[-1],
                device=device,
                dtype=llm_features.dtype,
            ),
        ],
        dim=1,
    )

    # indicator: real text -> 3, image -> 2, text pad -> 0
    indicator = torch.zeros(b, seq_len, dtype=torch.long, device=device)
    indicator[:, :num_text_tokens] = text_mask_long * LLM_TOKEN_INDICATOR
    indicator[:, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

    # segment ids: real text + image -> 1, text pad -> -1 (its own padding segment)
    segment_ids = torch.ones(b, seq_len, dtype=torch.long, device=device)
    segment_ids[:, :num_text_tokens] = torch.where(
        text_mask_bool,
        torch.ones_like(text_mask_long),
        torch.full_like(text_mask_long, SEQUENCE_PADDING_INDICATOR),
    )

    # position ids (t, h, w)
    # text positions: 0..num_real-1 at the real slots (relative; pad -> 0)
    text_pos = (text_mask_long.cumsum(dim=-1) - 1).clamp(min=0)  # (B, Lt)
    text_pos_3d = text_pos.unsqueeze(-1).expand(-1, -1, 3)

    h_idx = torch.arange(gh, device=device).view(-1, 1).expand(gh, gw).reshape(-1)
    w_idx = torch.arange(gw, device=device).view(1, -1).expand(gh, gw).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
    image_pos_3d = image_pos.unsqueeze(0).expand(b, -1, -1)

    position_ids = torch.cat([text_pos_3d, image_pos_3d], dim=1)

    # Flip into the model's time convention (t=1 -> clean).
    model_t = 1.0 - t

    out = transformer(
        llm_features=llm_full,
        x=x,
        t=model_t,
        position_ids=position_ids,
        segment_ids=segment_ids,
        indicator=indicator,
    )

    image_velocity = out[:, num_text_tokens:]  # (B, Li, 128)
    image_velocity = image_velocity.reshape(b, gh, gw, c).permute(0, 3, 1, 2)
    # Model predicts clean - noise; negate to return toolkit velocity (noise - clean).
    return -image_velocity


# ---------------------------------------------------------------------------
# Minimal sampling pipeline (for training previews).
# ---------------------------------------------------------------------------


class Ideogram4Pipeline:
    """Lightweight flow-matching sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # ``model`` is the Ideogram4Model so we can reuse its encode/decode and
        # latent helpers without duplicating state.
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,
        unconditional_embeds,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer
        patch = model.patch_size

        schedule_mu = float(
            model.model_config.model_kwargs.get("ideogram_schedule_mu", 0.0)
        )
        schedule_std = float(
            model.model_config.model_kwargs.get("ideogram_schedule_std", 1.75)
        )
        sigmas = get_ideogram4_sigmas(
            num_inference_steps,
            width,
            height,
            mu=schedule_mu,
            std=schedule_std,
            device=device,
        )

        ae_scale = model.vae_scale_factor  # 8
        gh = height // (ae_scale * patch)
        gw = width // (ae_scale * patch)
        latent_channels = transformer.config.in_channels

        # Ideogram uses asymmetric CFG: the unconditional branch is image-only
        # (no text tokens) with zeroed text features -- it does NOT run a negative
        # prompt through the text encoder. So we ignore unconditional_embeds and
        # build an empty (0-length) text sequence for the uncond pass below.
        do_cfg = guidance_scale > 1.0

        if latents is None:
            shape = (1, latent_channels, gh, gw)
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=torch.float32
            )
        latents = latents.to(device, dtype=torch.float32)
        latents = latents * sigmas[0]

        cond_feats, cond_mask = pad_text_features(
            conditional_embeds.text_embeds, device, dtype
        )
        if do_cfg:
            # Image-only unconditional: zero-length text sequence. predict_velocity
            # then produces an image-token-only forward pass with zeroed llm
            # features, matching the reference's asymmetric CFG.
            batch_size = latents.shape[0]
            text_dim = cond_feats.shape[-1]
            uncond_feats = torch.zeros(
                batch_size, 0, text_dim, device=device, dtype=dtype
            )
            uncond_mask = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        # The unconditional LoRA (if present) must be active *only* on the
        # unconditional pass. We force it off before each conditional pass since the
        # outer sampling context (``with network:``) may switch it on globally.
        uncond_lora = getattr(model, "unconditional_lora", None)

        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            t01 = sigma.expand(latents.shape[0])
            if uncond_lora is not None:
                uncond_lora.is_active = False
            v_cond = predict_velocity(
                transformer, latents.to(dtype), t01, cond_feats, cond_mask
            )
            if do_cfg:
                if uncond_lora is not None:
                    uncond_lora.is_active = True
                try:
                    v_uncond = predict_velocity(
                        transformer, latents.to(dtype), t01, uncond_feats, uncond_mask
                    )
                finally:
                    if uncond_lora is not None:
                        uncond_lora.is_active = False
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            latents = latents + v.to(torch.float32) * (sigma_next - sigma)

        images = model.decode_latents(latents, device=device, dtype=dtype)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
