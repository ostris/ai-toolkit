"""Dual-GPU model-parallel path for Wan 2.2 5B LoRA training.

Activates when WAN_DUAL_GPU=true. Distributes the diffusers
``WanTransformer3DModel`` across two CUDA devices with a single PCIe
boundary mid-``blocks``, keeps UMT5 on a configurable device (CPU
recommended for 32 GB cards), and reuses the same LoRA / multiplier
patches the FLUX.2 dual-GPU path installs (those hook ai-toolkit-wide
classes, not FLUX-specific ones, so they apply unchanged here).

Targets the ai-toolkit Wan 2.2 5B trainer
(``extensions_built_in/diffusion_models/wan22/wan22_5b_model.py``).
Integration is a two-line edit at the call site:

1. Make ``Wan225bModel`` inherit from ``Wan22DualGPUMixin`` first
   (MRO ordering: mixin before ``Wan21``).
2. In ``Wan21.load_wan_transformer``, replace::

       if self.model_config.split_model_over_gpus:
           raise ValueError("Splitting model over gpus is not supported for Wan2.1 models")

   with a call to ``self.setup_dual_gpu_distribution(transformer, dtype)``
   (gated by ``is_dual_gpu_enabled()``).

This file is the standalone mixin; the integration edits live elsewhere.

Env vars:
    WAN_DUAL_GPU=true              enable the dual-GPU path
    WAN_TE_DEVICE=cpu              pin UMT5 to a specific device
    WAN_DUAL_GPU_SPLIT_AT=20       override blocks split index
                                   (default: n_blocks // 2)
"""
from __future__ import annotations

import os
import types
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .wan22_5b_model import Wan225bModel  # noqa: F401


def is_dual_gpu_enabled() -> bool:
    return os.getenv("WAN_DUAL_GPU", "false").lower() == "true"


def get_te_device_override() -> torch.device | None:
    """Returns the text-encoder device override if WAN_TE_DEVICE is set."""
    val = os.getenv("WAN_TE_DEVICE")
    return torch.device(val) if val else None


class Wan22DualGPUMixin:
    """Mixin for :class:`Wan225bModel` that adds the dual-GPU training path.

    The mixin contributes:

    - ``te_device_torch`` attribute (resolved from ``WAN_TE_DEVICE`` or
      defaulting to ``device_torch``). Always present — used even when
      dual-GPU itself is disabled.
    - ``text_encoder_to`` override that routes UMT5 moves to
      ``te_device_torch`` instead of ``device_torch``.
    - ``setup_dual_gpu_distribution(transformer, dtype)`` to be called
      from ``load_wan_transformer`` after the transformer has been
      quantized (if applicable) and before the pipeline composes it.
      Distributes modules across cuda:0/cuda:1, replaces
      ``WanTransformer3DModel.forward`` with a split-aware variant, pins
      ``transformer.to()`` to ignore device arguments, and installs the
      LoRA / multiplier patches.
    - ``preserve_dual_gpu_split_on_pipe(pipe)`` — to be called in place
      of ``pipe.transformer = pipe.transformer.to(self.device_torch)``
      so the distributed layout survives pipeline composition.

    Note on diffusers ``_no_split_modules = ["WanTransformerBlock"]``:
    that attribute governs HuggingFace ``accelerate`` device-map sharding
    (it forbids splitting *within* a block). Our split is *between*
    blocks at a configurable boundary, so the attribute is informational
    only and does not constrain us.
    """

    te_device_torch: torch.device

    # ---------------------------------------------------------------- init

    def init_te_device(self) -> None:
        """Resolve ``te_device_torch`` from env var or fall back to model device.

        Must be called from :meth:`Wan225bModel.__init__` after the base
        ``__init__`` (which establishes ``self.device_torch``).
        """
        override = get_te_device_override()
        self.te_device_torch = override if override is not None else self.device_torch  # type: ignore[attr-defined]

    # ----------------------------------------------------------- TE override

    def text_encoder_to(self, *args: Any, **kwargs: Any) -> None:
        """Override of BaseModel.text_encoder_to that honors te_device_torch.

        ai-toolkit's :class:`SDTrainer` hook ``hook_before_train_loop``
        calls ``self.sd.text_encoder_to(self.device_torch)``
        unconditionally. With ``WAN_TE_DEVICE=cpu`` we want UMT5 to stay
        on CPU regardless.
        """
        target = self.te_device_torch
        if isinstance(self.text_encoder, list):  # type: ignore[attr-defined]
            for encoder in self.text_encoder:  # type: ignore[attr-defined]
                encoder.to(target)
        else:
            self.text_encoder.to(target)  # type: ignore[attr-defined]

    # ------------------------------------------------------------ load hook

    def setup_dual_gpu_distribution(self, transformer: torch.nn.Module, dtype: torch.dtype) -> None:
        """Distribute the Wan transformer across cuda:0 and cuda:1.

        Layout:
            cuda:0  — ``rope``, ``patch_embedding``, ``condition_embedder``,
                      ``blocks[:split_at]``
            cuda:1  — ``blocks[split_at:]``, ``norm_out``, ``proj_out``,
                      ``scale_shift_table``

        Also installs the LoRA / multiplier patches that align downstream
        ai-toolkit machinery with the split layout.
        """
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                f"WAN_DUAL_GPU=true requires >=2 CUDA devices, found "
                f"{torch.cuda.device_count()}."
            )

        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        n_blocks = len(transformer.blocks)
        override = os.getenv("WAN_DUAL_GPU_SPLIT_AT")
        split_at = int(override) if override else (n_blocks // 2)
        if not 0 < split_at < n_blocks:
            raise RuntimeError(
                f"WAN_DUAL_GPU_SPLIT_AT={split_at} out of range "
                f"(transformer has {n_blocks} blocks)."
            )

        self.print_and_status_update(  # type: ignore[attr-defined]
            f"Distributing transformer across {d0} and {d1}"
        )
        self.print_and_status_update(  # type: ignore[attr-defined]
            f"Block split: {split_at} on cuda:0, "
            f"{n_blocks - split_at} on cuda:1"
        )

        transformer.rope.to(d0)
        transformer.patch_embedding.to(d0, dtype=dtype)
        transformer.condition_embedder.to(d0, dtype=dtype)

        for blk in transformer.blocks[:split_at]:
            blk.to(d0, dtype=dtype)
        for blk in transformer.blocks[split_at:]:
            blk.to(d1, dtype=dtype)

        transformer.norm_out.to(d1)
        transformer.proj_out.to(d1, dtype=dtype)

        # scale_shift_table is an nn.Parameter, not a sub-module; moving
        # the containing module won't reach it because it's referenced
        # directly. Move the underlying storage in-place under no_grad.
        with torch.no_grad():
            transformer.scale_shift_table.data = transformer.scale_shift_table.data.to(d1)

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
    """Create a WanTransformer3DModel.forward variant that inserts a
    cuda:1 boundary at ``blocks[split_at]``.

    Recreates the original forward (see
    ``diffusers/models/transformers/transformer_wan.py``,
    ``WanTransformer3DModel.forward``) with three additions:
      1. At index ``split_at``, cross ``hidden_states``,
         ``encoder_hidden_states``, ``timestep_proj``, and the
         ``(cos, sin)`` rotary tuple to cuda:1.
      2. After ``proj_out``, move output back to cuda:0 so downstream
         loss ops (which run on the model's nominal device) work.
      3. Preserve the ``timestep.ndim == 2`` branch used by Wan 2.2
         ti2v for per-sequence-position timesteps.
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ):
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        d0_dev = next(self.blocks[0].parameters()).device
        d1_dev = next(self.blocks[split_at].parameters()).device

        for i, block in enumerate(self.blocks):
            if i == split_at and d1_dev != d0_dev:
                hidden_states = hidden_states.to(d1_dev)
                encoder_hidden_states = encoder_hidden_states.to(d1_dev)
                timestep_proj = timestep_proj.to(d1_dev)
                # rotary_emb is a tuple (cos, sin); both must move.
                rotary_emb = (
                    rotary_emb[0].to(d1_dev),
                    rotary_emb[1].to(d1_dev),
                )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # temb came out of condition_embedder on d0; the norm_out math
        # below runs on d1 with hidden_states. Move temb across.
        out_dev = hidden_states.device
        if temb.device != out_dev:
            temb = temb.to(out_dev)

        if temb.ndim == 3:
            shift, scale = (
                self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (
                self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
            ).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        # Return on d0 so downstream loss / scatter ops (which run on
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
# ``network_mixins.broadcast_and_multiply``) — not Wan-specific code —
# so they're already installed by the FLUX.2 path if it ran first. Gate
# all three installers behind a single module flag plus per-target
# attribute flags (defense in depth).
_PATCHES_INSTALLED = False


def _install_external_patches() -> None:
    """Idempotently install the LoRA-placement and multiplier-device patches.

    These hook external ai-toolkit classes — done as runtime patches so
    this file remains a self-contained addition. Safe to call alongside
    the FLUX.2 helper; both gate on the same per-target flags.
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
