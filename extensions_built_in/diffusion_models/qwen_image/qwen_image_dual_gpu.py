"""Dual-GPU model-parallel path for Qwen-Image LoRA training.

Activates when ``QWEN_IMAGE_DUAL_GPU=true``. Distributes the diffusers
``QwenImageTransformer2DModel`` across two CUDA devices with a single
PCIe boundary mid-``transformer_blocks``, keeps the Qwen2.5-VL text
encoder on a configurable device (CPU recommended for 32 GB cards), and
reuses the same LoRA / multiplier patches the FLUX.2 / Wan 2.2 / LTX-2
dual-GPU paths install (those hook ai-toolkit-wide classes, not
model-specific ones, so they apply unchanged here — the shared
``_PATCHES_INSTALLED`` flag makes calling all four safe).

Targets the ai-toolkit Qwen-Image trainer
(``extensions_built_in/diffusion_models/qwen_image/qwen_image.py``).
Integration is a small edit at the call site:

1. Make ``QwenImageModel`` inherit from ``QwenImageDualGPUMixin`` first
   (MRO ordering: mixin before ``BaseModel``).
2. In ``QwenImageModel.load_model``, after ``quantize_model(...)``, call
   ``self.setup_dual_gpu_distribution(transformer, dtype)`` gated by
   ``is_dual_gpu_enabled()`` — and keep the transformer on CPU through
   quantize so the mixin can distribute afterward.

Unlike FLUX.2's per-block hooks, Qwen-Image (like Wan 2.2 / LTX-2)
precomputes all loop-invariant tensors (``temb``, the RoPE tuple, the
joint attention-mask kwargs) once before the block loop, so a
single-boundary ``forward`` replacement is sufficient.

Env vars:
    QWEN_IMAGE_DUAL_GPU=true            enable the dual-GPU path
    QWEN_IMAGE_TE_DEVICE=cpu            pin the Qwen2.5-VL text encoder
    QWEN_IMAGE_DUAL_GPU_SPLIT_AT=30     override transformer_blocks split
                                       index (default: num_layers // 2)
"""
from __future__ import annotations

import os
import types
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .qwen_image import QwenImageModel  # noqa: F401


def is_dual_gpu_enabled() -> bool:
    return os.getenv("QWEN_IMAGE_DUAL_GPU", "false").lower() == "true"


def get_te_device_override() -> torch.device | None:
    """Returns the text-encoder device override if QWEN_IMAGE_TE_DEVICE is set."""
    val = os.getenv("QWEN_IMAGE_TE_DEVICE")
    return torch.device(val) if val else None


class QwenImageDualGPUMixin:
    """Mixin for :class:`QwenImageModel` that adds the dual-GPU training path.

    The mixin contributes:

    - ``te_device_torch`` attribute (resolved from ``QWEN_IMAGE_TE_DEVICE``
      or defaulting to ``device_torch``). Always present — used even when
      dual-GPU itself is disabled.
    - ``text_encoder_to`` override that routes Qwen2.5-VL moves to
      ``te_device_torch`` instead of ``device_torch``.
    - ``setup_dual_gpu_distribution(transformer, dtype)`` to be called
      from ``load_model`` after the transformer has been quantized (if
      applicable) and before the pipeline composes it. Distributes
      modules across cuda:0/cuda:1, replaces
      ``QwenImageTransformer2DModel.forward`` with a split-aware variant,
      pins ``transformer.to()`` to ignore device arguments, and installs
      the LoRA / multiplier patches.
    - ``preserve_dual_gpu_split_on_pipe(pipe)`` — to be checked in place
      of the single-device ``pipe.transformer.to(self.device_torch)``
      call so the distributed layout survives pipeline composition.

    Note on diffusers ``_no_split_modules = ["QwenImageTransformerBlock"]``:
    that attribute governs HuggingFace ``accelerate`` device-map sharding
    (it forbids splitting *within* a block). Our split is *between*
    blocks at a configurable boundary, so the attribute is informational
    only and does not constrain us.
    """

    te_device_torch: torch.device

    # ---------------------------------------------------------------- init

    def init_te_device(self) -> None:
        """Resolve ``te_device_torch`` from env var or fall back to model device.

        Must be called from :meth:`QwenImageModel.__init__` after the base
        ``__init__`` (which establishes ``self.device_torch``).
        """
        override = get_te_device_override()
        self.te_device_torch = override if override is not None else self.device_torch  # type: ignore[attr-defined]

    # ----------------------------------------------------------- TE override

    def text_encoder_to(self, *args: Any, **kwargs: Any) -> None:
        """Override of BaseModel.text_encoder_to that honors te_device_torch.

        ai-toolkit's :class:`SDTrainer` hook ``hook_before_train_loop``
        calls ``self.sd.text_encoder_to(self.device_torch)``
        unconditionally. With ``QWEN_IMAGE_TE_DEVICE=cpu`` we want the
        Qwen2.5-VL text encoder to stay on CPU regardless. Qwen-Image
        stores the text encoder as a list.
        """
        target = self.te_device_torch
        if isinstance(self.text_encoder, list):  # type: ignore[attr-defined]
            for encoder in self.text_encoder:  # type: ignore[attr-defined]
                encoder.to(target)
        else:
            self.text_encoder.to(target)  # type: ignore[attr-defined]

    # ------------------------------------------------------------ load hook

    def setup_dual_gpu_distribution(self, transformer: torch.nn.Module, dtype: torch.dtype) -> None:
        """Distribute the Qwen-Image transformer across cuda:0 and cuda:1.

        Layout:
            cuda:0  — ``pos_embed`` (RoPE), ``time_text_embed``,
                      ``txt_norm``, ``img_in``, ``txt_in``,
                      ``transformer_blocks[:split_at]``
            cuda:1  — ``transformer_blocks[split_at:]``, ``norm_out``,
                      ``proj_out``

        ``QwenImageTransformer2DModel`` has no module-level ``nn.Parameter``
        attributes (unlike Wan/LTX-2's ``scale_shift_table``); every weight
        lives inside a sub-module, so plain ``.to()`` reaches all of them.

        Also installs the LoRA / multiplier patches that align downstream
        ai-toolkit machinery with the split layout.
        """
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                f"QWEN_IMAGE_DUAL_GPU=true requires >=2 CUDA devices, found "
                f"{torch.cuda.device_count()}."
            )

        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        n_blocks = len(transformer.transformer_blocks)
        override = os.getenv("QWEN_IMAGE_DUAL_GPU_SPLIT_AT")
        split_at = int(override) if override else (n_blocks // 2)
        if not 0 < split_at < n_blocks:
            raise RuntimeError(
                f"QWEN_IMAGE_DUAL_GPU_SPLIT_AT={split_at} out of range "
                f"(transformer has {n_blocks} blocks)."
            )

        self.print_and_status_update(  # type: ignore[attr-defined]
            f"Distributing transformer across {d0} and {d1}"
        )
        self.print_and_status_update(  # type: ignore[attr-defined]
            f"Block split: {split_at} on cuda:0, "
            f"{n_blocks - split_at} on cuda:1"
        )

        # --- cuda:0: RoPE, embedders, input projections, first block half
        # pos_embed is an nn.Module (QwenEmbedRope / QwenEmbedLayer3DRope).
        # Its pos_freqs/neg_freqs are plain attributes (NOT registered
        # buffers — a comment in the source notes register_buffer drops the
        # imaginary part of the complex freqs), so .to() will not move them.
        # That is fine: the forward transfers them lazily per-device via
        # _get_device_freqs(device) and we feed it cuda:0 there.
        transformer.pos_embed.to(d0)
        transformer.time_text_embed.to(d0, dtype=dtype)
        transformer.txt_norm.to(d0, dtype=dtype)
        transformer.img_in.to(d0, dtype=dtype)
        transformer.txt_in.to(d0, dtype=dtype)

        for blk in transformer.transformer_blocks[:split_at]:
            blk.to(d0, dtype=dtype)

        # --- cuda:1: second block half + output layers
        for blk in transformer.transformer_blocks[split_at:]:
            blk.to(d1, dtype=dtype)

        transformer.norm_out.to(d1, dtype=dtype)
        transformer.proj_out.to(d1, dtype=dtype)

        transformer.forward = types.MethodType(
            _make_split_forward(split_at), transformer
        )
        _pin_transformer_to(transformer)

        _install_external_patches()

    # ----------------------------------------------------- pipeline composer

    def preserve_dual_gpu_split_on_pipe(self, pipe: Any) -> bool:
        """Return True if we've already distributed; caller should skip the
        single-device ``pipe.transformer.to(device)`` call."""
        return is_dual_gpu_enabled()


# ---------------------------------------------------------------- helpers


def _make_split_forward(split_at: int):
    """Create a ``QwenImageTransformer2DModel.forward`` variant that inserts
    a cuda:1 boundary at ``transformer_blocks[split_at]``.

    Recreates the original forward (see
    ``diffusers/models/transformers/transformer_qwenimage.py``,
    ``QwenImageTransformer2DModel.forward``) with three additions:
      1. At index ``split_at``, cross BOTH streams threaded through the
         block loop (``hidden_states`` image + ``encoder_hidden_states``
         text — the block returns ``(encoder_hidden_states, hidden_states)``)
         plus every precomputed loop-invariant tensor: ``temb``, the
         ``(img_freqs, txt_freqs)`` RoPE tuple, the ``attention_mask``
         entry inside ``block_attention_kwargs``, and ``modulate_index``.
      2. After ``proj_out``, move the output back to cuda:0 so downstream
         loss ops (which run on the model's nominal device) work without
         a device-mismatch error.
      3. Preserve the ``return_dict`` contract — returns ``(output,)`` or
         ``Transformer2DModelOutput``.

    The original forward is wrapped with ``@apply_lora_scale("attention_kwargs")``;
    that decorator only scales LoRA when ``attention_kwargs`` carries a
    scale, and ai-toolkit's training path always passes
    ``attention_kwargs=None`` (see ``QwenImageModel.get_noise_prediction``),
    so the decorator is a no-op here — we mirror the Wan / LTX-2 mixins
    and omit it.
    """
    from math import prod

    import numpy as np
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.transformers.transformer_qwenimage import (
        compute_text_seq_len_from_mask,
    )
    from diffusers.utils import deprecate

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ):
        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
                "Please use `encoder_hidden_states_mask` instead. "
                "The mask-based approach is more flexible and supports variable-length sequences.",
                standard_warn=False,
            )

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Use the encoder_hidden_states sequence length for RoPE computation
        # and normalize mask.
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        # pos_embed feeds the freqs onto hidden_states.device (cuda:0 here)
        # via its lazy _get_device_freqs(device) path — no plain-attribute
        # .to() gotcha. The resulting (img_freqs, txt_freqs) tuple is
        # bridged to cuda:1 at the boundary below.
        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        # Construct joint attention mask once to avoid reconstructing in
        # every block.
        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            joint_attention_mask = joint_attention_mask[:, None, None, :]
            block_attention_kwargs["attention_mask"] = joint_attention_mask

        # Devices of the first block on each side of the boundary.
        d0_dev = next(self.transformer_blocks[0].parameters()).device
        d1_dev = next(self.transformer_blocks[split_at].parameters()).device

        # block_attention_kwargs is a dict captured *by reference* in each
        # block's gradient-checkpoint closure. Mutating it in place at the
        # boundary would corrupt the cuda:0 blocks' backward recompute — they
        # would read the cuda:1 mask and hit a device mismatch. Keep the cuda:0
        # dict immutable; build a separate cuda:1 dict at the boundary and
        # hand each block the dict for its own side. (The looped *tensors*
        # below are reassigned to new objects rather than mutated, so each
        # block's closure keeps its own correct-device copy.)
        block_attention_kwargs_d1 = block_attention_kwargs

        for index_block, block in enumerate(self.transformer_blocks):
            if index_block == split_at and d1_dev != d0_dev:
                # PCIe boundary: bridge BOTH looped streams + every
                # precomputed loop-invariant tensor to cuda:1.
                hidden_states = hidden_states.to(d1_dev)
                encoder_hidden_states = encoder_hidden_states.to(d1_dev)
                temb = temb.to(d1_dev)
                # image_rotary_emb is a tuple (img_freqs, txt_freqs).
                image_rotary_emb = (
                    image_rotary_emb[0].to(d1_dev),
                    image_rotary_emb[1].to(d1_dev),
                )
                # Separate cuda:1 dict — do NOT mutate the cuda:0 one.
                block_attention_kwargs_d1 = dict(block_attention_kwargs)
                if "attention_mask" in block_attention_kwargs_d1:
                    block_attention_kwargs_d1["attention_mask"] = (
                        block_attention_kwargs_d1["attention_mask"].to(d1_dev)
                    )
                if modulate_index is not None:
                    modulate_index = modulate_index.to(d1_dev)

            cur_attention_kwargs = (
                block_attention_kwargs_d1
                if index_block >= split_at
                else block_attention_kwargs
            )

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None,  # Don't pass encoder_hidden_states_mask (using attention_mask instead)
                    temb,
                    image_rotary_emb,
                    cur_attention_kwargs,
                    modulate_index,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,  # Don't pass (using attention_mask instead)
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=cur_attention_kwargs,
                    modulate_index=modulate_index,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                residual = controlnet_block_samples[index_block // interval_control]
                if residual.device != hidden_states.device:
                    residual = residual.to(hidden_states.device)
                hidden_states = hidden_states + residual

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        # norm_out / proj_out live on cuda:1; temb came out of the
        # embedders on cuda:0 and is bridged at the boundary above, so it
        # already matches hidden_states here. Use only the image part.
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
    pipeline composer) would silently re-collect the split onto cuda:0.
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
# ``network_mixins.broadcast_and_multiply``) — not Qwen-Image-specific
# code — so they're already installed by the FLUX.2 / Wan / LTX-2 paths
# if any ran first. Gate all three installers behind a single module
# flag plus per-target attribute flags (defense in depth).
_PATCHES_INSTALLED = False


def _install_external_patches() -> None:
    """Idempotently install the LoRA-placement and multiplier-device patches.

    These hook external ai-toolkit classes — done as runtime patches so
    this file remains a self-contained addition. Safe to call alongside
    the FLUX.2 / Wan / LTX-2 helpers; all gate on the same per-target
    flags.
    """
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

    The lazy-pin is the load-bearing safety net — any later ai-toolkit
    hook that calls ``network.to(device)`` (e.g., from
    ``accelerate.prepare``) is silently corrected on the next forward.
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
