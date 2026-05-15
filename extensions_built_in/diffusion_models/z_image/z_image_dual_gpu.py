"""Dual-GPU model-parallel path for Z-Image LoRA training.

Activates when ``Z_IMAGE_DUAL_GPU=true``. Distributes the diffusers
``ZImageTransformer2DModel`` (the Tongyi-MAI Z-Image transformer, novel
architecture — not FLUX-shaped) across two CUDA devices with a single
PCIe boundary mid-``layers``, keeps the Qwen3 text encoder on a
configurable device, and reuses the same LoRA / multiplier patches the
FLUX.2 / Wan 2.2 / LTX-2 / Qwen-Image / HiDream / Chroma / FLUX.1-Kontext
dual-GPU paths install (those hook ai-toolkit-wide classes, not
model-specific ones, so they apply unchanged here — the per-target patch
flags make calling them all safe).

Targets the ai-toolkit Z-Image trainer
(``extensions_built_in/diffusion_models/z_image/z_image.py``).
Integration:

1. Make ``ZImageModel`` inherit from ``ZImageDualGPUMixin`` first
   (MRO ordering: mixin before ``BaseModel``).
2. In ``ZImageModel.__init__``, call ``self.init_te_device()`` after the
   base ``__init__`` to resolve ``te_device_torch``.
3. In ``ZImageModel.load_model``, after quantize, call
   ``self.setup_dual_gpu_distribution(transformer, dtype)`` gated by
   ``is_dual_gpu_enabled()``. Route the Qwen3 ``.to()`` sites to
   ``self.te_device_torch``.

Z-Image is shaped differently from FLUX-family transformers. It has:

- ``all_x_embedder``: ``nn.ModuleDict`` keyed by ``"{patch_size}-{f_patch_size}"``
- ``all_final_layer``: same key shape
- ``noise_refiner``: ``nn.ModuleList`` (2 blocks, modulation=True)
- ``context_refiner``: ``nn.ModuleList`` (2 blocks, modulation=False)
- ``siglip_refiner``: optional ``nn.ModuleList`` (Omni variant)
- ``t_embedder``, ``cap_embedder``, ``siglip_embedder`` (optional)
- ``x_pad_token``, ``cap_pad_token``, ``siglip_pad_token`` (optional)
  — module-level ``nn.Parameter`` attributes (don't follow ``.to(device)``
  on the submodules; must be moved explicitly)
- ``layers``: ``nn.ModuleList`` of 30 main blocks — the split target
- ``rope_embedder``: plain Python class (NOT ``nn.Module``); its
  ``freqs_cis`` cache follows the input tensor's device on each call, so
  no special handling needed beyond keeping pre-main-loop inputs on cuda:0

Layout: every pre-main-loop module (embedders, refiners, pad-token
Parameters) + ``layers[:split_at]`` on cuda:0; ``layers[split_at:]`` +
``all_final_layer`` on cuda:1. Default ``split_at = n_layers // 2 = 15``.

Env vars:
    Z_IMAGE_DUAL_GPU=true              enable the dual-GPU path
    Z_IMAGE_TE_DEVICE=cpu              pin the Qwen3 text encoder
    Z_IMAGE_DUAL_GPU_SPLIT_AT=15       override layers split index
                                       (default: n_layers // 2)
"""
from __future__ import annotations

import os
import types
from typing import TYPE_CHECKING, Any

import torch

from diffusers.utils import logging

if TYPE_CHECKING:
    from .z_image import ZImageModel  # noqa: F401

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def is_dual_gpu_enabled() -> bool:
    return os.getenv("Z_IMAGE_DUAL_GPU", "false").lower() == "true"


def get_te_device_override() -> torch.device | None:
    val = os.getenv("Z_IMAGE_TE_DEVICE")
    return torch.device(val) if val else None


class ZImageDualGPUMixin:
    """Mixin for :class:`ZImageModel` that adds the dual-GPU training path."""

    te_device_torch: torch.device

    # ---------------------------------------------------------------- init

    def init_te_device(self) -> None:
        override = get_te_device_override()
        self.te_device_torch = override if override is not None else self.device_torch  # type: ignore[attr-defined]
        if self.te_device_torch.type == "cpu":
            self.print_and_status_update(  # type: ignore[attr-defined]
                "Z_IMAGE_TE_DEVICE=cpu — the Qwen3 text encoder will run on "
                "CPU. Set `cache_text_embeddings: true` in your config so it "
                "runs once at startup; otherwise it runs on CPU every step."
            )

    # ----------------------------------------------------------- TE override

    def text_encoder_to(self, *args: Any, **kwargs: Any) -> None:
        target = self.te_device_torch
        if isinstance(self.text_encoder, list):  # type: ignore[attr-defined]
            for encoder in self.text_encoder:  # type: ignore[attr-defined]
                encoder.to(target)
        else:
            self.text_encoder.to(target)  # type: ignore[attr-defined]

    # ------------------------------------------------------------ load hook

    def setup_dual_gpu_distribution(self, transformer: torch.nn.Module, dtype: torch.dtype) -> None:
        """Distribute the Z-Image transformer across cuda:0 and cuda:1.

        Layout:
            cuda:0  — embedders (``t_embedder``, ``cap_embedder``,
                      optional ``siglip_embedder``), ``all_x_embedder``,
                      both refiners (``noise_refiner``, ``context_refiner``,
                      optional ``siglip_refiner``), pad-token Parameters
                      (``x_pad_token``, ``cap_pad_token``, optional
                      ``siglip_pad_token``), ``layers[:split_at]``
            cuda:1  — ``layers[split_at:]``, ``all_final_layer``

        Module-level ``nn.Parameter`` attributes (``x_pad_token``,
        ``cap_pad_token``, ``siglip_pad_token``) don't follow the
        per-submodule ``.to()`` calls used elsewhere — they need explicit
        ``.data = .data.to()`` moves. Without this they stay on whatever
        device the loader landed them (typically the staging device),
        breaking ``_prepare_sequence`` which uses them on cuda:0.

        ``rope_embedder`` is a plain Python class (not ``nn.Module``); its
        ``freqs_cis`` cache auto-follows the input tensor's device on each
        call, so no explicit handling is needed.

        Also installs the LoRA / multiplier patches that align downstream
        ai-toolkit machinery with the split layout.
        """
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError(
                "Z_IMAGE_DUAL_GPU=true needs at least 2 CUDA devices "
                f"(cuda:0 + cuda:1); found "
                f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}. "
                "Unset Z_IMAGE_DUAL_GPU for single-GPU training."
            )

        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        n_layers = len(transformer.layers)
        override = os.getenv("Z_IMAGE_DUAL_GPU_SPLIT_AT")
        if override is not None:
            try:
                split_at = int(override)
            except ValueError:
                raise ValueError(
                    "Z_IMAGE_DUAL_GPU_SPLIT_AT must be an integer; "
                    f"got {override!r}."
                )
            if not (1 <= split_at < n_layers):
                raise ValueError(
                    f"Z_IMAGE_DUAL_GPU_SPLIT_AT={split_at} is out of range; "
                    f"must be in [1, {n_layers - 1}] for a model with "
                    f"{n_layers} main-block layers."
                )
        else:
            split_at = n_layers // 2

        self.print_and_status_update(  # type: ignore[attr-defined]
            f"Distributing transformer across {d0} and {d1}"
        )
        self.print_and_status_update(  # type: ignore[attr-defined]
            f"Main layers split: {split_at} on {d0}, "
            f"{n_layers - split_at} on {d1}"
        )

        # --- cuda:0: embedders, refiners, pad-token Parameters,
        #     first half of main layers
        transformer.t_embedder.to(d0, dtype=dtype)
        transformer.cap_embedder.to(d0, dtype=dtype)
        transformer.all_x_embedder.to(d0, dtype=dtype)

        for blk in transformer.noise_refiner:
            blk.to(d0, dtype=dtype)
        for blk in transformer.context_refiner:
            blk.to(d0, dtype=dtype)

        if getattr(transformer, "siglip_embedder", None) is not None:
            transformer.siglip_embedder.to(d0, dtype=dtype)
        if getattr(transformer, "siglip_refiner", None) is not None:
            for blk in transformer.siglip_refiner:
                blk.to(d0, dtype=dtype)

        # Module-level nn.Parameter attrs — explicit .data move, since
        # they're not reached by per-submodule .to() calls.
        transformer.x_pad_token.data = transformer.x_pad_token.data.to(d0, dtype=dtype)
        transformer.cap_pad_token.data = transformer.cap_pad_token.data.to(d0, dtype=dtype)
        if getattr(transformer, "siglip_pad_token", None) is not None:
            transformer.siglip_pad_token.data = transformer.siglip_pad_token.data.to(d0, dtype=dtype)

        for i, blk in enumerate(transformer.layers):
            blk.to(d0 if i < split_at else d1, dtype=dtype)

        # --- cuda:1: final output (ModuleDict — every key goes to cuda:1)
        transformer.all_final_layer.to(d1, dtype=dtype)

        transformer.forward = types.MethodType(
            _make_split_forward(split_at), transformer
        )
        _pin_transformer_to(transformer)

        _install_external_patches()


# ---------------------------------------------------------------- helpers


def _make_split_forward(split_at: int):
    """Create a ``ZImageTransformer2DModel.forward`` variant that inserts
    a cuda:1 boundary at ``layers[split_at]``.

    Recreates the diffusers ``ZImageTransformer2DModel.forward`` from the
    *box's* installed diffusers (pulled at port time — the mac install
    diverges, so we built against the box copy to track the actual
    runtime — Qwen-Image lesson applied) with one addition:

      At ``layers[split_at]`` — the single PCIe boundary — bridge to
      cuda:1 every tensor the downstream main-block iterations +
      ``all_final_layer`` + ``unpatchify`` consume: ``unified``,
      ``unified_mask``, ``unified_freqs``, ``adaln_input`` (or
      ``t_noisy``/``t_clean`` in Omni mode), and
      ``unified_noise_tensor``. After ``unpatchify`` the output list is
      moved back to cuda:0 element-wise so downstream loss ops work
      without a device-mismatch error.

    ``controlnet_block_samples`` is preserved verbatim — consumed inside
    the per-layer step on whichever device the layer lives on. Caller
    must provide samples on the right device; out of scope for the
    LoRA-training path this port targets.
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    def forward(
        self,
        x: Any,  # list[Tensor] or list[list[Tensor]] in Omni mode
        t: torch.Tensor,
        cap_feats: Any,
        return_dict: bool = True,
        controlnet_block_samples: dict[int, torch.Tensor] | None = None,
        siglip_feats: Any | None = None,
        image_noise_mask: Any | None = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        omni_mode = isinstance(x[0], list)

        # Devices on each side of the PCIe boundary.
        d0_dev = next(self.layers[0].parameters()).device
        d1_dev = next(self.layers[split_at].parameters()).device
        device = d0_dev  # pre-main-loop work all happens on cuda:0

        if omni_mode:
            t_noisy = self.t_embedder(t.to(d0_dev) * self.t_scale).type_as(x[0][-1])
            t_clean = self.t_embedder(torch.ones_like(t).to(d0_dev) * self.t_scale).type_as(x[0][-1])
            adaln_input = None
        else:
            adaln_input = self.t_embedder(t.to(d0_dev) * self.t_scale).type_as(x[0])
            t_noisy = t_clean = None

        # Patchify
        if omni_mode:
            (
                x,
                cap_feats,
                siglip_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                siglip_pos_ids,
                x_pad_mask,
                cap_pad_mask,
                siglip_pad_mask,
                x_pos_offsets,
                x_noise_mask,
                cap_noise_mask,
                siglip_noise_mask,
            ) = self.patchify_and_embed_omni(x, cap_feats, siglip_feats, patch_size, f_patch_size, image_noise_mask)
        else:
            (
                x,
                cap_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                x_pad_mask,
                cap_pad_mask,
            ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
            x_pos_offsets = x_noise_mask = cap_noise_mask = siglip_noise_mask = None

        # X embed & refine — cuda:0
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))
        x, x_freqs, x_mask, _, x_noise_tensor = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, x_noise_mask, device
        )

        for layer in self.noise_refiner:
            x = (
                self._gradient_checkpointing_func(
                    layer, x, x_mask, x_freqs, adaln_input, x_noise_tensor, t_noisy, t_clean
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(x, x_mask, x_freqs, adaln_input, x_noise_tensor, t_noisy, t_clean)
            )

        # Cap embed & refine — cuda:0
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))
        cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, None, device
        )

        for layer in self.context_refiner:
            cap_feats = (
                self._gradient_checkpointing_func(layer, cap_feats, cap_mask, cap_freqs)
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(cap_feats, cap_mask, cap_freqs)
            )

        # Siglip embed & refine — cuda:0 (Omni only)
        siglip_seqlens = siglip_freqs = None
        if omni_mode and siglip_feats[0] is not None and self.siglip_embedder is not None:
            siglip_seqlens = [len(si) for si in siglip_feats]
            siglip_feats = self.siglip_embedder(torch.cat(siglip_feats, dim=0))
            siglip_feats, siglip_freqs, siglip_mask, _, _ = self._prepare_sequence(
                list(siglip_feats.split(siglip_seqlens, dim=0)),
                siglip_pos_ids,
                siglip_pad_mask,
                self.siglip_pad_token,
                None,
                device,
            )

            for layer in self.siglip_refiner:
                siglip_feats = (
                    self._gradient_checkpointing_func(layer, siglip_feats, siglip_mask, siglip_freqs)
                    if torch.is_grad_enabled() and self.gradient_checkpointing
                    else layer(siglip_feats, siglip_mask, siglip_freqs)
                )

        # Unified sequence — cuda:0
        unified, unified_freqs, unified_mask, unified_noise_tensor = self._build_unified_sequence(
            x,
            x_freqs,
            x_seqlens,
            x_noise_mask,
            cap_feats,
            cap_freqs,
            cap_seqlens,
            cap_noise_mask,
            siglip_feats,
            siglip_freqs,
            siglip_seqlens,
            siglip_noise_mask,
            omni_mode,
            device,
        )

        # Main transformer layers — PCIe boundary at split_at
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == split_at and d1_dev != d0_dev:
                # Bridge every tensor the downstream main-blocks + final +
                # unpatchify consume. ``unified_noise_tensor`` may be None
                # in non-omni mode; guard. ``t_noisy`` / ``t_clean`` are
                # used by final_layer in omni mode; bridge if present.
                unified = unified.to(d1_dev)
                if unified_mask is not None:
                    unified_mask = unified_mask.to(d1_dev)
                if unified_freqs is not None:
                    if isinstance(unified_freqs, (list, tuple)):
                        unified_freqs = type(unified_freqs)(
                            f.to(d1_dev) if hasattr(f, "to") else f for f in unified_freqs
                        )
                    else:
                        unified_freqs = unified_freqs.to(d1_dev)
                if adaln_input is not None:
                    adaln_input = adaln_input.to(d1_dev)
                if unified_noise_tensor is not None:
                    unified_noise_tensor = unified_noise_tensor.to(d1_dev)
                if t_noisy is not None:
                    t_noisy = t_noisy.to(d1_dev)
                if t_clean is not None:
                    t_clean = t_clean.to(d1_dev)

            unified = (
                self._gradient_checkpointing_func(
                    layer, unified, unified_mask, unified_freqs, adaln_input, unified_noise_tensor, t_noisy, t_clean
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(unified, unified_mask, unified_freqs, adaln_input, unified_noise_tensor, t_noisy, t_clean)
            )
            if controlnet_block_samples is not None and layer_idx in controlnet_block_samples:
                # Caller-supplied sample; bridge to the current unified
                # device for safety. Out of scope for LoRA training, but
                # keep parity with the vendored forward.
                cb = controlnet_block_samples[layer_idx]
                if hasattr(cb, "to") and cb.device != unified.device:
                    cb = cb.to(unified.device)
                unified = unified + cb

        # Final layer on cuda:1; adaln_input + t_noisy/t_clean were
        # bridged at the boundary already.
        unified = (
            self.all_final_layer[f"{patch_size}-{f_patch_size}"](
                unified, noise_mask=unified_noise_tensor, c_noisy=t_noisy, c_clean=t_clean
            )
            if omni_mode
            else self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
        )

        # Unpatchify — operates on cuda:1 input, returns list[Tensor] on cuda:1.
        x = self.unpatchify(list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size, x_pos_offsets)

        # Return on cuda:0 so downstream loss / scatter ops (which run on
        # the model's nominal device) work without device-mismatch errors.
        # x is a list[Tensor] — move each element.
        if isinstance(x, list):
            x = [t.to(d0_dev) if hasattr(t, "to") and t.device != d0_dev else t for t in x]
        elif hasattr(x, "to") and x.device != d0_dev:
            x = x.to(d0_dev)

        return (x,) if not return_dict else Transformer2DModelOutput(sample=x)

    return forward


def _pin_transformer_to(transformer: torch.nn.Module) -> None:
    """Wrap ``transformer.to()`` so device arguments are stripped (the model
    is already distributed) while dtype changes still pass through.

    Z-Image's ``z_image.py`` has more ``transformer.to(self.device_torch)``
    call sites than most trainers (in ``load_model``, in
    ``generate_single_image`` L303-304, and in ``get_noise_prediction``
    L329). Pinning makes all of them no-ops.
    """
    _orig_to = transformer.to

    def _split_preserving_to(*args: Any, **kwargs: Any):
        filtered_args = [
            a for a in args
            if not isinstance(a, (str, torch.device, int))
        ]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        if not filtered_args and not filtered_kwargs:
            return transformer
        return _orig_to(*filtered_args, **filtered_kwargs)

    transformer.to = _split_preserving_to  # type: ignore[assignment]


# Module-level idempotency flag — shared with FLUX.2 / Wan / LTX-2 /
# Qwen-Image / HiDream / Chroma / FLUX.1-Kontext via per-target attribute
# flags on the patched classes.
_PATCHES_INSTALLED = False


def _install_external_patches() -> None:
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return
    _patch_force_to()
    _patch_apply_to_with_lazy_forward()
    _patch_broadcast_and_multiply()
    _PATCHES_INSTALLED = True


def _patch_force_to() -> None:
    from toolkit.network_mixins import ToolkitNetworkMixin

    if getattr(ToolkitNetworkMixin, "_dual_gpu_patched", False):
        return

    def _per_layer_force_to(self_n, device, dtype):
        try:
            self_n.to(device, dtype)
        except Exception:
            pass
        loras = []
        if hasattr(self_n, "unet_loras"):
            loras += self_n.unet_loras
        if hasattr(self_n, "text_encoder_loras"):
            loras += self_n.text_encoder_loras
        for lora in loras:
            tgt = device
            try:
                parent = lora.org_module[0]
                p = next(parent.parameters(), None)
                if p is not None:
                    tgt = p.device
            except Exception:
                pass
            lora.to(tgt, dtype)
            for attr in ("lora_down", "lora_up", "lora_mid"):
                m = getattr(lora, attr, None)
                if m is not None:
                    try:
                        m.to(tgt, dtype)
                    except Exception:
                        pass

    ToolkitNetworkMixin.force_to = _per_layer_force_to
    ToolkitNetworkMixin._dual_gpu_patched = True


def _patch_apply_to_with_lazy_forward() -> None:
    from toolkit.kohya_lora import LoRANetwork

    if getattr(LoRANetwork, "_dual_gpu_apply_patched", False):
        return

    _orig_apply_to = LoRANetwork.apply_to

    def _redistribute_apply_to(self_n, *args, **kwargs):
        ret = _orig_apply_to(self_n, *args, **kwargs)
        loras = []
        if hasattr(self_n, "unet_loras"):
            loras += self_n.unet_loras
        if hasattr(self_n, "text_encoder_loras"):
            loras += self_n.text_encoder_loras
        for lora in loras:
            if getattr(lora, "_dual_gpu_wrapped", False):
                continue
            lora.forward = _make_pinned_forward(lora, lora.forward)
            try:
                lora.org_module[0].forward = lora.forward
            except Exception:
                pass
            lora._dual_gpu_wrapped = True
        return ret

    LoRANetwork.apply_to = _redistribute_apply_to
    LoRANetwork._dual_gpu_apply_patched = True


def _make_pinned_forward(lora, orig_forward):
    def _pinned(*fargs, **fkwargs):
        in_dev = None
        for a in fargs:
            if hasattr(a, "device") and getattr(a, "is_cuda", False):
                in_dev = a.device
                break
        if in_dev is None:
            return orig_forward(*fargs, **fkwargs)
        p = next(lora.parameters(), None)
        if p is not None and p.device != in_dev:
            lora.to(in_dev)
        return orig_forward(*fargs, **fkwargs)
    return _pinned


def _patch_broadcast_and_multiply() -> None:
    from toolkit import network_mixins as _nm

    if getattr(_nm, "_dual_gpu_bm_patched", False):
        return

    _orig_bm = _nm.broadcast_and_multiply

    def _device_aware_bm(tensor, multiplier):
        try:
            if hasattr(multiplier, "device") and multiplier.device != tensor.device:
                multiplier = multiplier.to(tensor.device)
        except Exception:
            pass
        return _orig_bm(tensor, multiplier)

    _nm.broadcast_and_multiply = _device_aware_bm
    _nm._dual_gpu_bm_patched = True
