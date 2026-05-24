"""Dual-GPU model-parallel path for HiDream-I1 LoRA training.

Activates when ``HIDREAM_DUAL_GPU=true``. Distributes the
``HiDreamImageTransformer2DModel`` (ai-toolkit's vendored copy under
``hidream/src/models/transformers/`` — NOT the diffusers-package class,
which has a substantially different forward) across two CUDA devices with a single
PCIe boundary mid-``single_stream_blocks``, keeps the four HiDream text
encoders (CLIP-L, CLIP-G, T5-XXL, Llama-3.1-8B) on a configurable device
(CPU recommended for 32 GB cards — the Llama encoder alone is ~30 GB),
and reuses the same LoRA / multiplier patches the FLUX.2 / Wan 2.2 /
LTX-2 / Qwen-Image dual-GPU paths install (those hook ai-toolkit-wide
classes, not model-specific ones, so they apply unchanged here — the
shared ``_PATCHES_INSTALLED`` flag makes calling all of them safe).

Targets the ai-toolkit HiDream trainer
(``extensions_built_in/diffusion_models/hidream/hidream_model.py``).
Integration is a small edit at the call site:

1. Make ``HidreamModel`` inherit from ``HiDreamDualGPUMixin`` first
   (MRO ordering: mixin before ``BaseModel``).
2. In ``HidreamModel.load_model``, after quantize, call
   ``self.setup_dual_gpu_distribution(transformer, dtype)`` gated by
   ``is_dual_gpu_enabled()`` — and keep the transformer on CPU through
   quantize so the mixin can distribute afterward.

Unlike FLUX.2's per-block hooks, HiDream (like Qwen-Image / Wan 2.2 /
LTX-2) precomputes all loop-invariant tensors (``temb``, the RoPE
``image_rotary_emb``, the ``caption_projection`` of the encoder states)
once before the two block loops, so a single-boundary ``forward``
replacement is sufficient. HiDream has two sequential block loops —
``double_stream_blocks`` then ``single_stream_blocks`` — that both
consume the precomputed context. All ``double_stream_blocks`` plus the
first half of ``single_stream_blocks`` live on cuda:0; the remaining
``single_stream_blocks`` plus ``final_layer`` live on cuda:1, with one
PCIe boundary inside the single-stream loop.

Env vars:
    HIDREAM_DUAL_GPU=true              enable the dual-GPU path
    HIDREAM_TE_DEVICE=cpu             pin the four text encoders
    HIDREAM_DUAL_GPU_SPLIT_AT=16      override single_stream_blocks split
                                      index (default: num_single_layers // 2)
"""
from __future__ import annotations

import os
import types
from typing import TYPE_CHECKING, Any

import torch

from diffusers.utils import logging

if TYPE_CHECKING:
    from .hidream_model import HidreamModel  # noqa: F401

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def is_dual_gpu_enabled() -> bool:
    return os.getenv("HIDREAM_DUAL_GPU", "false").lower() == "true"


def get_te_device_override() -> torch.device | None:
    """Returns the text-encoder device override if HIDREAM_TE_DEVICE is set."""
    val = os.getenv("HIDREAM_TE_DEVICE")
    return torch.device(val) if val else None


class HiDreamDualGPUMixin:
    """Mixin for :class:`HidreamModel` that adds the dual-GPU training path.

    The mixin contributes:

    - ``te_device_torch`` attribute (resolved from ``HIDREAM_TE_DEVICE``
      or defaulting to ``device_torch``). Always present — used even when
      dual-GPU itself is disabled.
    - ``text_encoder_to`` override that routes the four HiDream text
      encoders to ``te_device_torch`` instead of ``device_torch``.
    - ``setup_dual_gpu_distribution(transformer, dtype)`` to be called
      from ``load_model`` after the transformer has been quantized (if
      applicable) and before the pipeline composes it. Distributes
      modules across cuda:0/cuda:1, replaces
      ``HiDreamImageTransformer2DModel.forward`` with a split-aware
      variant, pins ``transformer.to()`` to ignore device arguments, and
      installs the LoRA / multiplier patches.
    - ``preserve_dual_gpu_split_on_pipe(pipe)`` — to be checked in place
      of the single-device ``pipe.transformer.to(self.device_torch)``
      call so the distributed layout survives pipeline composition.

    Note on diffusers ``_no_split_modules`` (``HiDreamImageTransformerBlock``
    + ``HiDreamImageSingleTransformerBlock``): that attribute governs
    HuggingFace ``accelerate`` device-map sharding (it forbids splitting
    *within* a block). Our split is *between* blocks at a configurable
    boundary, so the attribute is informational only and does not
    constrain us. HiDream's MoE (``MoEGate`` / ``MOEFeedForwardSwiGLU``)
    lives inside the blocks, so it distributes with them transparently —
    no special handling.
    """

    te_device_torch: torch.device

    # ---------------------------------------------------------------- init

    def init_te_device(self) -> None:
        """Resolve ``te_device_torch`` from env var or fall back to model device.

        Must be called from :meth:`HidreamModel.__init__` after the base
        ``__init__`` (which establishes ``self.device_torch``).
        """
        override = get_te_device_override()
        self.te_device_torch = override if override is not None else self.device_torch  # type: ignore[attr-defined]
        if self.te_device_torch.type == "cpu":
            self.print_and_status_update(  # type: ignore[attr-defined]
                "HIDREAM_TE_DEVICE=cpu — the four HiDream text encoders "
                "(incl. the ~30 GB Llama-3.1-8B) will run on CPU. Set "
                "`cache_text_embeddings: true` in your config so they run "
                "once at startup; otherwise they run on CPU every step."
            )

    # ----------------------------------------------------------- TE override

    def text_encoder_to(self, *args: Any, **kwargs: Any) -> None:
        """Override of BaseModel.text_encoder_to that honors te_device_torch.

        ai-toolkit's :class:`SDTrainer` hook ``hook_before_train_loop``
        calls ``self.sd.text_encoder_to(self.device_torch)``
        unconditionally. With ``HIDREAM_TE_DEVICE=cpu`` we want all four
        HiDream text encoders to stay on CPU regardless. HiDream stores
        them as a list (``text_encoder_list`` — CLIP-L, CLIP-G, T5-XXL,
        Llama-3.1-8B), so iterate it; fall back to a single encoder for
        safety.
        """
        target = self.te_device_torch
        if isinstance(self.text_encoder, list):  # type: ignore[attr-defined]
            for encoder in self.text_encoder:  # type: ignore[attr-defined]
                encoder.to(target)
        else:
            self.text_encoder.to(target)  # type: ignore[attr-defined]

    # ------------------------------------------------------------ load hook

    def setup_dual_gpu_distribution(self, transformer: torch.nn.Module, dtype: torch.dtype) -> None:
        """Distribute the HiDream transformer across cuda:0 and cuda:1.

        Layout:
            cuda:0  — ``x_embedder``, ``t_embedder``, ``p_embedder``,
                      ``pe_embedder``, ``caption_projection``, ALL
                      ``double_stream_blocks``, ``single_stream_blocks[:split_at]``
            cuda:1  — ``single_stream_blocks[split_at:]``, ``final_layer``

        ``HiDreamImageTransformer2DModel`` has no module-level
        ``nn.Parameter`` attributes (unlike Wan/LTX-2's
        ``scale_shift_table``); every weight lives inside a sub-module —
        including the MoE gating ``weight`` inside ``MoEGate``, which is
        nested inside the blocks — so plain ``.to()`` reaches all of
        them. RoPE is computed dynamically inside ``forward`` (no
        plain-attribute freq tensor to move).

        Also installs the LoRA / multiplier patches that align downstream
        ai-toolkit machinery with the split layout.
        """
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError(
                "HIDREAM_DUAL_GPU=true needs at least 2 CUDA devices (cuda:0 + "
                f"cuda:1); found {torch.cuda.device_count() if torch.cuda.is_available() else 0}. "
                "Unset HIDREAM_DUAL_GPU for single-GPU training."
            )

        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        n_single = len(transformer.single_stream_blocks)
        override = os.getenv("HIDREAM_DUAL_GPU_SPLIT_AT")
        if override is not None:
            try:
                split_at = int(override)
            except ValueError:
                raise ValueError(
                    "HIDREAM_DUAL_GPU_SPLIT_AT must be an integer; got "
                    f"{override!r}."
                )
            if not (1 <= split_at < n_single):
                raise ValueError(
                    f"HIDREAM_DUAL_GPU_SPLIT_AT={split_at} is out of range; "
                    f"must be in [1, {n_single - 1}] for a model with "
                    f"{n_single} single_stream_blocks."
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

        # --- cuda:0: embedders, caption projection, double-stream blocks,
        #     first half of single-stream blocks
        transformer.x_embedder.to(d0, dtype=dtype)
        transformer.t_embedder.to(d0, dtype=dtype)
        transformer.p_embedder.to(d0, dtype=dtype)
        # pe_embedder (HiDreamImageEmbedND) has no parameters — RoPE is
        # computed via torch.arange inside its forward — but move it for
        # consistency so its forward runs on cuda:0.
        transformer.pe_embedder.to(d0, dtype=dtype)
        # caption_projection is consumed once, before the block loops, to
        # project every encoder state; keep it on cuda:0.
        transformer.caption_projection.to(d0, dtype=dtype)

        for blk in transformer.double_stream_blocks:
            blk.to(d0, dtype=dtype)

        for i, blk in enumerate(transformer.single_stream_blocks):
            blk.to(d0 if i < split_at else d1, dtype=dtype)

        # --- cuda:1: final output layer
        transformer.final_layer.to(d1, dtype=dtype)

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
    """Create a ``HiDreamImageTransformer2DModel.forward`` variant that
    inserts a cuda:1 boundary at ``single_stream_blocks[split_at]``.

    Recreates ai-toolkit's *vendored* ``HiDreamImageTransformer2DModel.forward``
    (``src/models/transformers/transformer_hidream_image.py`` — the copy the
    hidream extension actually imports, NOT the diffusers-package copy) with
    two additions:

      1. The ``double_stream_blocks`` loop runs entirely on cuda:0. Its
         outputs (``hidden_states`` + the threaded
         ``initial_encoder_hidden_states``) are concatenated and fed into
         the ``single_stream_blocks`` loop, also starting on cuda:0.
      2. At ``single_stream_blocks[split_at]`` — the single PCIe boundary
         — bridge to cuda:1 every tensor the downstream single blocks +
         ``final_layer`` consume: ``hidden_states``, ``image_tokens_masks``,
         ``adaln_input``, ``rope``, and every remaining
         ``encoder_hidden_states`` entry indexed by ``block_id`` in the
         loop tail. After ``final_layer`` / ``unpatchify`` the output is
         moved back to cuda:0 so downstream loss ops (which run on the
         model's nominal device) work without a device-mismatch error.

    Signature and return type match the vendored forward exactly:
    ``tuple[torch.Tensor, torch.Tensor] | Transformer2DModelOutput`` (the
    vendored forward returns ``(output, image_tokens_masks)`` /
    ``Transformer2DModelOutput(sample=output, mask=image_tokens_masks)``).
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
    from einops import repeat

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_embeds: torch.Tensor = None,
        img_sizes: Any = None,
        img_ids: Any = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # spatial forward
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        # 0. time
        timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        p_embedder = self.p_embedder(pooled_embeds)
        adaln_input = timesteps + p_embedder

        hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states, self.max_seq, img_sizes)
        if image_tokens_masks is None:
            pH, pW = img_sizes[0]
            img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        hidden_states = self.x_embedder(hidden_states)

        T5_encoder_hidden_states = encoder_hidden_states[0]
        encoder_hidden_states = encoder_hidden_states[-1]
        encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
            T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(T5_encoder_hidden_states)

        txt_ids = torch.zeros(
            batch_size,
            encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
            3,
            device=img_ids.device, dtype=img_ids.dtype
        )
        ids = torch.cat((img_ids, txt_ids), dim=1)
        rope = self.pe_embedder(ids)

        # Devices of the first single block on each side of the boundary.
        # The double-stream blocks + single_stream_blocks[:split_at] all
        # live on d0_dev; single_stream_blocks[split_at:] + final_layer
        # live on d1_dev.
        d0_dev = next(self.single_stream_blocks[0].parameters()).device
        d1_dev = next(self.single_stream_blocks[split_at].parameters()).device

        # 2. Blocks
        block_id = 0
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.double_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id].detach()
            cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, initial_encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    image_tokens_masks,
                    cur_encoder_hidden_states,
                    adaln_input.clone(),
                    rope.clone(),
                )

            else:
                hidden_states, initial_encoder_hidden_states = block(
                    image_tokens = hidden_states,
                    image_tokens_masks = image_tokens_masks,
                    text_tokens = cur_encoder_hidden_states,
                    adaln_input = adaln_input,
                    rope = rope,
                )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if image_tokens_masks is not None:
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                device=image_tokens_masks.device, dtype=image_tokens_masks.dtype
            )
            image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_stream_blocks):
            if bid == split_at and d1_dev != d0_dev:
                # PCIe boundary: bridge every tensor the downstream single
                # blocks + final_layer consume to cuda:1. The remaining
                # encoder_hidden_states entries (indexed by block_id in the
                # loop body) are bridged here in one pass; adaln_input is
                # reused by final_layer too.
                hidden_states = hidden_states.to(d1_dev)
                adaln_input = adaln_input.to(d1_dev)
                rope = rope.to(d1_dev)
                if image_tokens_masks is not None:
                    image_tokens_masks = image_tokens_masks.to(d1_dev)
                encoder_hidden_states = [
                    ehs.to(d1_dev) if idx >= block_id else ehs
                    for idx, ehs in enumerate(encoder_hidden_states)
                ]
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id].detach()
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    image_tokens_masks,
                    None,
                    adaln_input.clone(),
                    rope.clone(),
                )
            else:
                hidden_states = block(
                    image_tokens = hidden_states,
                    image_tokens_masks = image_tokens_masks,
                    text_tokens = None,
                    adaln_input = adaln_input,
                    rope = rope,
                )
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        # final_layer lives on cuda:1; adaln_input was bridged at the boundary
        # above (when split_at < n_single it always runs), so it already matches.
        output = self.final_layer(hidden_states, adaln_input)
        output = self.unpatchify(output, img_sizes, self.training)
        if image_tokens_masks is not None:
            image_tokens_masks = image_tokens_masks[:, :image_tokens_seq_len]

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        # Return on cuda:0 so downstream loss / scatter ops (which run on the
        # model's nominal device) work without a device-mismatch error.
        if output.device != d0_dev:
            output = output.to(d0_dev)
        if image_tokens_masks is not None and image_tokens_masks.device != d0_dev:
            image_tokens_masks = image_tokens_masks.to(d0_dev)

        if not return_dict:
            return (output, image_tokens_masks)
        return Transformer2DModelOutput(sample=output, mask=image_tokens_masks)

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
# ``network_mixins.broadcast_and_multiply``) — not HiDream-specific
# code — so they're already installed by the FLUX.2 / Wan / LTX-2 /
# Qwen-Image paths if any ran first. Gate all three installers behind a
# single module flag plus per-target attribute flags (defense in depth).
_PATCHES_INSTALLED = False


def _install_external_patches() -> None:
    """Idempotently install the LoRA-placement and multiplier-device patches.

    These hook external ai-toolkit classes — done as runtime patches so
    this file remains a self-contained addition. Safe to call alongside
    the FLUX.2 / Wan / LTX-2 / Qwen-Image helpers; all gate on the same
    per-target flags.
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
