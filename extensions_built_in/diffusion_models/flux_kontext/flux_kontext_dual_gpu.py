"""Dual-GPU model-parallel path for FLUX.1-Kontext LoRA training.

Activates when ``FLUX_KONTEXT_DUAL_GPU=true``. Distributes the diffusers
``FluxTransformer2DModel`` (the FLUX.1 transformer that
``FluxKontextModel`` imports directly from ``diffusers``) across two CUDA
devices with a single PCIe boundary mid-``single_transformer_blocks``,
keeps the two FLUX text encoders (CLIP-L, T5-XXL) on a configurable
device, and reuses the same LoRA / multiplier patches the FLUX.2 /
Wan 2.2 / LTX-2 / Qwen-Image / HiDream / Chroma paths install (those
hook ai-toolkit-wide classes, not model-specific ones, so they apply
unchanged here — the per-target patch flags make calling them all safe).

Targets the ai-toolkit FLUX.1-Kontext trainer
(``extensions_built_in/diffusion_models/flux_kontext/flux_kontext.py``).
Integration is a small edit at the call site:

1. Make ``FluxKontextModel`` inherit from ``FluxKontextDualGPUMixin``
   first (MRO ordering: mixin before ``BaseModel``).
2. In ``FluxKontextModel.__init__``, call ``self.init_te_device()`` after
   the base ``__init__`` to resolve ``te_device_torch``.
3. In ``FluxKontextModel.load_model``, after quantize, call
   ``self.setup_dual_gpu_distribution(transformer, dtype)`` gated by
   ``is_dual_gpu_enabled()``. Route the four TE ``.to()`` sites (T5
   post-load, CLIP post-load, the two move-back loop entries) to
   ``self.te_device_torch``.

``FluxTransformer2DModel`` has the FLUX.1 shape: ``x_embedder`` +
``context_embedder`` + ``time_text_embed`` + ``pos_embed``, then a
sequence of ``transformer_blocks`` (19 — joint stream), then
``single_transformer_blocks`` (38 — also returning both
``encoder_hidden_states`` and ``hidden_states``; recent diffusers
refactor), then ``norm_out`` + ``proj_out``. Both block loops consume
the same precomputed ``temb`` + ``image_rotary_emb``, so a single
forward replacement covers the whole split.

Layout: all 19 ``transformer_blocks`` plus
``single_transformer_blocks[:split_at]`` live on cuda:0; the remaining
``single_transformer_blocks`` plus ``norm_out`` + ``proj_out`` live on
cuda:1, with one PCIe boundary inside the single-block loop. Default
``split_at = num_single // 2 = 19``.

Env vars:
    FLUX_KONTEXT_DUAL_GPU=true              enable the dual-GPU path
    FLUX_KONTEXT_TE_DEVICE=cpu              pin the two text encoders
    FLUX_KONTEXT_DUAL_GPU_SPLIT_AT=19       override split index
                                            (default: num_single // 2)
"""
from __future__ import annotations

import os
import types
from typing import TYPE_CHECKING, Any

import torch

from diffusers.utils import logging

if TYPE_CHECKING:
    from .flux_kontext import FluxKontextModel  # noqa: F401

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def is_dual_gpu_enabled() -> bool:
    return os.getenv("FLUX_KONTEXT_DUAL_GPU", "false").lower() == "true"


def get_te_device_override() -> torch.device | None:
    """Returns the text-encoder device override if FLUX_KONTEXT_TE_DEVICE is set."""
    val = os.getenv("FLUX_KONTEXT_TE_DEVICE")
    return torch.device(val) if val else None


class FluxKontextDualGPUMixin:
    """Mixin for :class:`FluxKontextModel` that adds the dual-GPU training path.

    The mixin contributes:

    - ``te_device_torch`` attribute (resolved from
      ``FLUX_KONTEXT_TE_DEVICE`` or defaulting to ``device_torch``).
      Always present — used even when dual-GPU itself is disabled.
    - ``text_encoder_to`` override that routes the two FLUX text encoders
      (CLIP-L, T5-XXL) to ``te_device_torch`` instead of ``device_torch``.
    - ``setup_dual_gpu_distribution(transformer, dtype)`` to be called
      from ``load_model`` after the transformer has been quantized (if
      applicable) and before the pipeline composes it. Distributes
      modules across cuda:0/cuda:1, replaces
      ``FluxTransformer2DModel.forward`` with a split-aware variant,
      pins ``transformer.to()`` to ignore device arguments, and installs
      the LoRA / multiplier patches.
    """

    te_device_torch: torch.device

    # ---------------------------------------------------------------- init

    def init_te_device(self) -> None:
        """Resolve ``te_device_torch`` from env var or fall back to model device.

        Must be called from :meth:`FluxKontextModel.__init__` after the
        base ``__init__`` (which establishes ``self.device_torch``).
        """
        override = get_te_device_override()
        self.te_device_torch = override if override is not None else self.device_torch  # type: ignore[attr-defined]
        if self.te_device_torch.type == "cpu":
            self.print_and_status_update(  # type: ignore[attr-defined]
                "FLUX_KONTEXT_TE_DEVICE=cpu — the two FLUX text encoders "
                "(CLIP-L + T5-XXL) will run on CPU. Set "
                "`cache_text_embeddings: true` in your config so they run "
                "once at startup; otherwise they run on CPU every step."
            )

    # ----------------------------------------------------------- TE override

    def text_encoder_to(self, *args: Any, **kwargs: Any) -> None:
        """Override of BaseModel.text_encoder_to that honors te_device_torch."""
        target = self.te_device_torch
        if isinstance(self.text_encoder, list):  # type: ignore[attr-defined]
            for encoder in self.text_encoder:  # type: ignore[attr-defined]
                encoder.to(target)
        else:
            self.text_encoder.to(target)  # type: ignore[attr-defined]

    # ------------------------------------------------------------ load hook

    def setup_dual_gpu_distribution(self, transformer: torch.nn.Module, dtype: torch.dtype) -> None:
        """Distribute the FLUX.1 transformer across cuda:0 and cuda:1.

        Layout:
            cuda:0  — ``x_embedder``, ``context_embedder``,
                      ``time_text_embed``, ``pos_embed``, ALL
                      ``transformer_blocks``,
                      ``single_transformer_blocks[:split_at]``
            cuda:1  — ``single_transformer_blocks[split_at:]``,
                      ``norm_out``, ``proj_out``

        ``FluxTransformer2DModel`` has no module-level ``nn.Parameter``
        attributes — every weight lives inside a sub-module. ``pos_embed``
        (``FluxPosEmbed``) has no parameters either; RoPE freqs are
        computed dynamically from ``params.theta`` inside its forward.

        Also installs the LoRA / multiplier patches that align downstream
        ai-toolkit machinery with the split layout.
        """
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError(
                "FLUX_KONTEXT_DUAL_GPU=true needs at least 2 CUDA devices "
                f"(cuda:0 + cuda:1); found "
                f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}. "
                "Unset FLUX_KONTEXT_DUAL_GPU for single-GPU training."
            )

        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        n_single = len(transformer.single_transformer_blocks)
        override = os.getenv("FLUX_KONTEXT_DUAL_GPU_SPLIT_AT")
        if override is not None:
            try:
                split_at = int(override)
            except ValueError:
                raise ValueError(
                    "FLUX_KONTEXT_DUAL_GPU_SPLIT_AT must be an integer; "
                    f"got {override!r}."
                )
            if not (1 <= split_at < n_single):
                raise ValueError(
                    f"FLUX_KONTEXT_DUAL_GPU_SPLIT_AT={split_at} is out of "
                    f"range; must be in [1, {n_single - 1}] for a model "
                    f"with {n_single} single_transformer_blocks."
                )
        else:
            split_at = n_single // 2

        self.print_and_status_update(  # type: ignore[attr-defined]
            f"Distributing transformer across {d0} and {d1}"
        )
        self.print_and_status_update(  # type: ignore[attr-defined]
            f"Single-block split: {split_at} on {d0}, "
            f"{n_single - split_at} on {d1}"
        )

        # --- cuda:0: input projections, time/text/guidance embedder,
        #     positional embedder, all joint blocks, first half of single
        #     blocks
        transformer.x_embedder.to(d0, dtype=dtype)
        transformer.context_embedder.to(d0, dtype=dtype)
        transformer.time_text_embed.to(d0, dtype=dtype)
        # pos_embed (FluxPosEmbed) has no parameters; the .to() is a no-op
        # but call it for consistency in case future versions add buffers.
        transformer.pos_embed.to(d0, dtype=dtype)

        for blk in transformer.transformer_blocks:
            blk.to(d0, dtype=dtype)

        for i, blk in enumerate(transformer.single_transformer_blocks):
            blk.to(d0 if i < split_at else d1, dtype=dtype)

        # --- cuda:1: final norm + projection
        transformer.norm_out.to(d1, dtype=dtype)
        transformer.proj_out.to(d1, dtype=dtype)

        transformer.forward = types.MethodType(
            _make_split_forward(split_at), transformer
        )
        _pin_transformer_to(transformer)

        _install_external_patches()


# ---------------------------------------------------------------- helpers


def _make_split_forward(split_at: int):
    """Create a ``FluxTransformer2DModel.forward`` variant that inserts a
    cuda:1 boundary at ``single_transformer_blocks[split_at]``.

    Recreates the diffusers ``FluxTransformer2DModel.forward``
    (``diffusers/models/transformers/transformer_flux.py``, verified
    identical between the mac and box installs at the time of porting)
    with one addition:

      At ``single_transformer_blocks[split_at]`` — the single PCIe
      boundary — bridge to cuda:1 every tensor the downstream single
      blocks + ``norm_out`` + ``proj_out`` consume: ``hidden_states``,
      ``encoder_hidden_states``, ``temb``, the two ``image_rotary_emb``
      tensors (``cos``, ``sin``), and any tensor entries in
      ``joint_attention_kwargs`` (e.g. ``ip_hidden_states`` for
      IP-Adapter). After ``proj_out`` the output is moved back to cuda:0
      so downstream loss ops (which run on the model's nominal device)
      work without a device-mismatch error.

    Controlnet residual branches are preserved verbatim — they're
    consumed inside the per-block step on whichever device the block
    lives on, so caller-supplied controlnet samples on the wrong device
    would fail. Out of scope for this LoRA-training path; left intact for
    parity with the vendored forward.

    Signature and return type match the vendored forward exactly.
    """
    import numpy as np
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        # Devices of the first single block on each side of the boundary.
        # transformer_blocks[*] + single_transformer_blocks[:split_at] all
        # live on d0_dev; single_transformer_blocks[split_at:] + norm_out +
        # proj_out live on d1_dev.
        d0_dev = next(self.single_transformer_blocks[0].parameters()).device
        d1_dev = next(self.single_transformer_blocks[split_at].parameters()).device

        # Ensure inputs match the cuda:0 input projections.
        if hidden_states.device != d0_dev:
            hidden_states = hidden_states.to(d0_dev)
        if encoder_hidden_states is not None and encoder_hidden_states.device != d0_dev:
            encoder_hidden_states = encoder_hidden_states.to(d0_dev)
        if pooled_projections is not None and pooled_projections.device != d0_dev:
            pooled_projections = pooled_projections.to(d0_dev)
        if timestep is not None and timestep.device != d0_dev:
            timestep = timestep.to(d0_dev)
        if guidance is not None and guidance.device != d0_dev:
            guidance = guidance.to(d0_dev)
        if img_ids is not None and img_ids.device != d0_dev:
            img_ids = img_ids.to(d0_dev)
        if txt_ids is not None and txt_ids.device != d0_dev:
            txt_ids = txt_ids.to(d0_dev)

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # --- transformer_blocks: all on cuda:0
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block // interval_control]
                    )

        # --- single_transformer_blocks: PCIe boundary at split_at
        for index_block, block in enumerate(self.single_transformer_blocks):
            if index_block == split_at and d1_dev != d0_dev:
                # PCIe boundary: bridge every tensor the downstream blocks +
                # norm_out + proj_out consume to cuda:1. image_rotary_emb is
                # a (cos, sin) tuple of tensors.
                hidden_states = hidden_states.to(d1_dev)
                encoder_hidden_states = encoder_hidden_states.to(d1_dev)
                temb = temb.to(d1_dev)
                if isinstance(image_rotary_emb, (tuple, list)):
                    image_rotary_emb = tuple(
                        t.to(d1_dev) if hasattr(t, "to") else t for t in image_rotary_emb
                    )
                elif hasattr(image_rotary_emb, "to"):
                    image_rotary_emb = image_rotary_emb.to(d1_dev)
                if joint_attention_kwargs is not None:
                    # Bridge tensor entries only; leave non-tensor kwargs
                    # (scales, flags) alone. Build a NEW dict — do not
                    # mutate the cuda:0 view (gradient-checkpoint backward
                    # in cuda:0 blocks captures the dict by reference and
                    # re-reads it; mutating would corrupt that re-read,
                    # the Qwen-Image lesson).
                    joint_attention_kwargs = {
                        k: (v.to(d1_dev) if hasattr(v, "to") and hasattr(v, "device") else v)
                        for k, v in joint_attention_kwargs.items()
                    }

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = (
                    hidden_states + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # Return on cuda:0 so downstream loss / scatter ops (which run on
        # the model's nominal device) work without a device-mismatch error.
        if output.device != d0_dev:
            output = output.to(d0_dev)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    return forward


def _pin_transformer_to(transformer: torch.nn.Module) -> None:
    """Wrap ``transformer.to()`` so device arguments are stripped (the model
    is already distributed) while dtype changes still pass through.

    Without this, ai-toolkit hooks like ``set_device_state`` (called every
    training step) and ``pipe.transformer.to(self.device_torch)`` (the
    pipeline composer at the bottom of ``FluxKontextModel.load_model``)
    would silently re-collect the split onto cuda:0.
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


# Module-level idempotency flag. The external patches hook
# ai-toolkit-wide classes (``ToolkitNetworkMixin``, ``LoRANetwork``,
# ``network_mixins.broadcast_and_multiply``) — not Kontext-specific code —
# so they're already installed by the FLUX.2 / Wan / LTX-2 / Qwen-Image /
# HiDream / Chroma paths if any ran first. Gate all three installers
# behind a single module flag plus per-target attribute flags (defense in
# depth).
_PATCHES_INSTALLED = False


def _install_external_patches() -> None:
    """Idempotently install the LoRA-placement and multiplier-device patches."""
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return
    _patch_force_to()
    _patch_apply_to_with_lazy_forward()
    _patch_broadcast_and_multiply()
    _PATCHES_INSTALLED = True


def _patch_force_to() -> None:
    """``ToolkitNetworkMixin.force_to`` moves the network's submodules to a
    single device. Override so each LoRA module follows the device of the
    layer it wraps.
    """
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
    """``LoRANetwork.apply_to`` creates wrapped LoRA modules and registers
    them as submodules. After it runs, each LoRA's forward is wrapped to
    lazily re-pin parameters to the input's device on every call.
    """
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
    """Wrap a LoRA module's forward to re-pin its parameters to the input
    tensor's device on each call. Cheap no-op when devices already match.
    """
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
    """``network_mixins.broadcast_and_multiply(tensor, multiplier)`` fails
    when ``tensor`` and ``multiplier`` live on different devices. Auto-move
    the multiplier to match.
    """
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
