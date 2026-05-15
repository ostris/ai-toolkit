"""Dual-GPU model-parallel path for Chroma1-HD LoRA training.

Activates when ``CHROMA_DUAL_GPU=true``. Distributes the ``Chroma``
transformer (ai-toolkit's vendored copy under ``chroma/src/model.py`` —
Chroma is its own implementation, not a diffusers class) across two CUDA
devices with a single PCIe boundary mid-``single_blocks``, keeps the T5
text encoder on a configurable device (``CHROMA_TE_DEVICE``, defaulting
to ``device_torch``), and reuses the same LoRA / multiplier patches the
FLUX.2 / Wan 2.2 / LTX-2 / Qwen-Image / HiDream dual-GPU paths install
(those hook ai-toolkit-wide classes, not model-specific ones, so they
apply unchanged here — the shared per-target patch flags make calling
them all safe).

Targets the ai-toolkit Chroma trainer
(``extensions_built_in/diffusion_models/chroma/chroma_model.py``).
Integration is a small edit at the call site:

1. Make ``ChromaModel`` inherit from ``ChromaDualGPUMixin`` first
   (MRO ordering: mixin before ``BaseModel``).
2. In ``ChromaModel.__init__``, call ``self.init_te_device()`` after the
   base ``__init__`` to resolve ``te_device_torch``.
3. In ``ChromaModel.load_model``, after quantize, call
   ``self.setup_dual_gpu_distribution(transformer, dtype)`` gated by
   ``is_dual_gpu_enabled()``. Route the T5 ``.to()`` sites to
   ``self.te_device_torch`` instead of ``self.device_torch`` (the
   FakeCLIP stub site is a no-op and can stay).

Chroma has the FLUX shape: a sequence of ``double_blocks`` (19), then a
``cat(txt, img)``, then ``single_blocks`` (38), then a strip + ``final_layer``.
All loop-invariant tensors (``pe``, ``txt_img_mask``, the modulation-vector
dict produced once by ``distilled_guidance_layer``) are precomputed before
the block loops, so a single-boundary forward replacement is sufficient.

The split: all 19 ``double_blocks`` plus ``single_blocks[:split_at]`` live
on cuda:0; the remaining ``single_blocks`` plus ``final_layer`` live on
cuda:1, with one PCIe boundary inside the single-block loop. Default
``split_at = num_single_blocks // 2 = 19``.

Env vars:
    CHROMA_DUAL_GPU=true              enable the dual-GPU path
    CHROMA_TE_DEVICE=cpu              pin the T5 text encoder
    CHROMA_DUAL_GPU_SPLIT_AT=19       override single_blocks split index
                                      (default: num_single_blocks // 2)
"""
from __future__ import annotations

import os
import types
from typing import TYPE_CHECKING, Any

import torch

from diffusers.utils import logging

if TYPE_CHECKING:
    from .chroma_model import ChromaModel  # noqa: F401

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def is_dual_gpu_enabled() -> bool:
    return os.getenv("CHROMA_DUAL_GPU", "false").lower() == "true"


def get_te_device_override() -> torch.device | None:
    """Returns the text-encoder device override if CHROMA_TE_DEVICE is set."""
    val = os.getenv("CHROMA_TE_DEVICE")
    return torch.device(val) if val else None


class ChromaDualGPUMixin:
    """Mixin for :class:`ChromaModel` that adds the dual-GPU training path.

    The mixin contributes:

    - ``te_device_torch`` attribute (resolved from ``CHROMA_TE_DEVICE``
      or defaulting to ``device_torch``). Always present — used even when
      dual-GPU itself is disabled.
    - ``text_encoder_to`` override that routes the T5 text encoder to
      ``te_device_torch`` instead of ``device_torch``. The CLIP slot in
      Chroma is a ``FakeCLIP`` stub with no real parameters; iterating
      the list is harmless.
    - ``setup_dual_gpu_distribution(transformer, dtype)`` to be called
      from ``load_model`` after the transformer has been quantized (if
      applicable) and before the pipeline composes it. Distributes
      modules across cuda:0/cuda:1, replaces ``Chroma.forward`` with a
      split-aware variant, pins ``transformer.to()`` to ignore device
      arguments (preserving the layout against later ai-toolkit ``.to()``
      calls), and installs the LoRA / multiplier patches.
    """

    te_device_torch: torch.device

    # ---------------------------------------------------------------- init

    def init_te_device(self) -> None:
        """Resolve ``te_device_torch`` from env var or fall back to model device.

        Must be called from :meth:`ChromaModel.__init__` after the base
        ``__init__`` (which establishes ``self.device_torch``).
        """
        override = get_te_device_override()
        self.te_device_torch = override if override is not None else self.device_torch  # type: ignore[attr-defined]
        if self.te_device_torch.type == "cpu":
            self.print_and_status_update(  # type: ignore[attr-defined]
                "CHROMA_TE_DEVICE=cpu — the T5 text encoder will run on CPU. "
                "Set `cache_text_embeddings: true` in your config so it runs "
                "once at startup; otherwise it runs on CPU every step."
            )

    # ----------------------------------------------------------- TE override

    def text_encoder_to(self, *args: Any, **kwargs: Any) -> None:
        """Override of BaseModel.text_encoder_to that honors te_device_torch.

        ai-toolkit's :class:`SDTrainer` hook ``hook_before_train_loop``
        calls ``self.sd.text_encoder_to(self.device_torch)``
        unconditionally. With ``CHROMA_TE_DEVICE=cpu`` we want the T5
        encoder to stay on CPU regardless. Chroma stores text encoders as
        a list (``[FakeCLIP, T5]``); iterate it. FakeCLIP's ``.to()`` is
        benign — it has no real parameters.
        """
        target = self.te_device_torch
        if isinstance(self.text_encoder, list):  # type: ignore[attr-defined]
            for encoder in self.text_encoder:  # type: ignore[attr-defined]
                encoder.to(target)
        else:
            self.text_encoder.to(target)  # type: ignore[attr-defined]

    # ------------------------------------------------------------ load hook

    def setup_dual_gpu_distribution(self, transformer: torch.nn.Module, dtype: torch.dtype) -> None:
        """Distribute the Chroma transformer across cuda:0 and cuda:1.

        Layout:
            cuda:0  — ``img_in``, ``txt_in``, ``pe_embedder``,
                      ``distilled_guidance_layer``, ALL ``double_blocks``,
                      ``single_blocks[:split_at]``
            cuda:1  — ``single_blocks[split_at:]``, ``final_layer``

        ``Chroma`` has one module-level buffer (``mod_index``, non-persistent)
        which is moved with the parent ``.to(d0, dtype=dtype)`` of the
        whole transformer at the bottom of this function via the explicit
        per-submodule ``.to()`` calls — except ``mod_index`` lives directly
        on the parent module, so we move it explicitly. RoPE frequencies
        are computed dynamically inside ``EmbedND.forward`` from
        ``params.theta`` (no plain-attribute freq tensor to move).

        Also installs the LoRA / multiplier patches that align downstream
        ai-toolkit machinery with the split layout.
        """
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError(
                "CHROMA_DUAL_GPU=true needs at least 2 CUDA devices (cuda:0 + "
                f"cuda:1); found {torch.cuda.device_count() if torch.cuda.is_available() else 0}. "
                "Unset CHROMA_DUAL_GPU for single-GPU training."
            )

        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        n_single = len(transformer.single_blocks)
        override = os.getenv("CHROMA_DUAL_GPU_SPLIT_AT")
        if override is not None:
            try:
                split_at = int(override)
            except ValueError:
                raise ValueError(
                    "CHROMA_DUAL_GPU_SPLIT_AT must be an integer; got "
                    f"{override!r}."
                )
            if not (1 <= split_at < n_single):
                raise ValueError(
                    f"CHROMA_DUAL_GPU_SPLIT_AT={split_at} is out of range; "
                    f"must be in [1, {n_single - 1}] for a model with "
                    f"{n_single} single_blocks."
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

        # --- cuda:0: embedders, distillation approximator, double-stream
        #     blocks, first half of single-stream blocks
        transformer.img_in.to(d0, dtype=dtype)
        transformer.txt_in.to(d0, dtype=dtype)
        # pe_embedder (EmbedND) has no parameters — RoPE is computed
        # dynamically via torch.arange / torch.outer inside its forward —
        # but call .to so child buffers (if any future addition) follow.
        transformer.pe_embedder.to(d0, dtype=dtype)
        transformer.distilled_guidance_layer.to(d0, dtype=dtype)

        for blk in transformer.double_blocks:
            blk.to(d0, dtype=dtype)

        for i, blk in enumerate(transformer.single_blocks):
            blk.to(d0 if i < split_at else d1, dtype=dtype)

        # --- cuda:1: final output layer
        transformer.final_layer.to(d1, dtype=dtype)

        # mod_index buffer (registered non-persistent on the parent) feeds
        # timestep_embedding under no_grad on cuda:0; keep it on d0.
        if hasattr(transformer, "mod_index"):
            transformer.mod_index = transformer.mod_index.to(d0)

        transformer.forward = types.MethodType(
            _make_split_forward(split_at), transformer
        )
        _pin_transformer_to(transformer)

        _install_external_patches()


# ---------------------------------------------------------------- helpers


def _make_split_forward(split_at: int):
    """Create a ``Chroma.forward`` variant that inserts a cuda:1 boundary
    at ``single_blocks[split_at]``.

    Recreates ai-toolkit's vendored ``Chroma.forward``
    (``extensions_built_in/diffusion_models/chroma/src/model.py``) with
    two additions:

      1. The ``double_blocks`` loop runs entirely on cuda:0. Its outputs
         (``img``, ``txt``) are concatenated (``cat(txt, img, dim=1)``)
         and fed into the ``single_blocks`` loop, also starting on cuda:0.
      2. At ``single_blocks[split_at]`` — the single PCIe boundary —
         bridge to cuda:1 every tensor the downstream single blocks +
         ``final_layer`` consume: ``img`` (the concatenated tensor),
         ``pe``, ``txt_img_mask``, and a freshly-built ``mod_vectors_dict``
         containing the cuda:1-resident slices for ``single_blocks[i].modulation.lin``
         (``i >= split_at``) plus ``final_layer.adaLN_modulation.1``. The
         original cuda:0 ``mod_vectors_dict`` is NOT mutated — gradient
         checkpointing in the cuda:0 single blocks captures it by closure
         and re-reads it during backward; mutating would corrupt that
         re-read (Qwen-Image lesson).

    After ``final_layer`` the output is moved back to cuda:0 so downstream
    loss ops (which run on the model's nominal device) work without a
    device-mismatch error.

    Signature and return type match the vendored forward exactly.
    """
    import torch.utils.checkpoint as ckpt

    from .src.layers import distribute_modulations, timestep_embedding

    def forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        txt_mask: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor,
        attn_padding: int = 1,
    ) -> torch.Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # Devices of the first single block on each side of the boundary.
        # double_blocks[*] + single_blocks[:split_at] all live on d0_dev;
        # single_blocks[split_at:] + final_layer live on d1_dev.
        d0_dev = next(self.single_blocks[0].parameters()).device
        d1_dev = next(self.single_blocks[split_at].parameters()).device

        # img_in / txt_in live on cuda:0; ensure inputs match.
        if img.device != d0_dev:
            img = img.to(d0_dev)
        if txt.device != d0_dev:
            txt = txt.to(d0_dev)
        if img_ids.device != d0_dev:
            img_ids = img_ids.to(d0_dev)
        if txt_ids.device != d0_dev:
            txt_ids = txt_ids.to(d0_dev)
        if txt_mask.device != d0_dev:
            txt_mask = txt_mask.to(d0_dev)

        img = self.img_in(img)
        txt = self.txt_in(txt)

        # Distillation approximator — runs under no_grad on cuda:0,
        # produces the mod_vectors tensor consumed by every block. The
        # final tensor itself requires_grad_(True) per the vendored
        # forward; the .requires_grad_(True) call inside the no_grad
        # block is what marks the *output* to receive a backward graph
        # from downstream block usage.
        with torch.no_grad():
            distill_timestep = timestep_embedding(timesteps.to(d0_dev), 16)
            distil_guidance = timestep_embedding(guidance.to(d0_dev), 16)
            modulation_index = timestep_embedding(self.mod_index, 32)
            modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
            timestep_guidance = (
                torch.cat([distill_timestep, distil_guidance], dim=1)
                .unsqueeze(1)
                .repeat(1, self.mod_index_length, 1)
            )
            input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
            mod_vectors = self.distilled_guidance_layer(input_vec.requires_grad_(True))
        mod_vectors_dict = distribute_modulations(
            mod_vectors, self.depth_single_blocks, self.depth_double_blocks
        )

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # Build txt_img_mask on cuda:0 (consumed by every block; bridged
        # at the boundary).
        max_len = txt.shape[1]
        with torch.no_grad():
            # modify_mask_to_attend_padding is a no-grad helper in src/model.py;
            # inline the same shape for the unmask side rather than importing
            # the private fn, so a vendored rename doesn't break us. Mirror
            # the vendored forward exactly.
            from .src.model import modify_mask_to_attend_padding
            txt_mask_w_padding = modify_mask_to_attend_padding(
                txt_mask, max_len, attn_padding
            )
            txt_img_mask = torch.cat(
                [
                    txt_mask_w_padding,
                    torch.ones([img.shape[0], img.shape[1]], device=txt_mask.device),
                ],
                dim=1,
            )
            txt_img_mask = txt_img_mask.float().T @ txt_img_mask.float()
            txt_img_mask = (
                txt_img_mask[None, None, ...]
                .repeat(txt.shape[0], self.num_heads, 1, 1)
                .int()
                .bool()
            )

        # --- double_blocks: all on cuda:0
        for i, block in enumerate(self.double_blocks):
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            double_mod = [img_mod, txt_mod]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img.requires_grad_(True)
                img, txt = ckpt.checkpoint(
                    block, img, txt, pe, double_mod, txt_img_mask
                )
            else:
                img, txt = block(
                    img=img, txt=txt, pe=pe, distill_vec=double_mod, mask=txt_img_mask
                )

        txt_seq_len = txt.shape[1]
        img = torch.cat((txt, img), 1)

        # --- single_blocks: PCIe boundary at split_at
        for i, block in enumerate(self.single_blocks):
            if i == split_at and d1_dev != d0_dev:
                # PCIe boundary: bridge every tensor the downstream single
                # blocks + final_layer consume to cuda:1. Build a SEPARATE
                # cuda:1 mod_vectors_dict slice — do NOT mutate the cuda:0
                # dict (gradient-checkpoint backward re-reads it).
                img = img.to(d1_dev)
                pe = pe.to(d1_dev)
                txt_img_mask = txt_img_mask.to(d1_dev)

            if i < split_at:
                single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            else:
                # Lazy per-block bridge: each single ModulationOut becomes
                # a cuda:1 copy. ModulationOut is a dataclass with
                # shift/scale/gate tensor fields — wrap the .to() through
                # the dataclass to preserve type.
                src = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
                single_mod = _modulation_to(src, d1_dev)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img.requires_grad_(True)
                img = ckpt.checkpoint(block, img, pe, single_mod, txt_img_mask)
            else:
                img = block(img, pe=pe, distill_vec=single_mod, mask=txt_img_mask)

        img = img[:, txt_seq_len:, ...]
        # final_layer lives on cuda:1; its distill_vec is a list of two
        # tensor slices from mod_vectors. Bridge both to cuda:1.
        final_mod_src = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        final_mod = [t.to(d1_dev) for t in final_mod_src]
        img = self.final_layer(img, distill_vec=final_mod)

        # Return on cuda:0 so downstream loss / scatter ops (which run on
        # the model's nominal device) work without a device-mismatch error.
        if img.device != d0_dev:
            img = img.to(d0_dev)
        return img

    return forward


def _modulation_to(mod: Any, device: torch.device) -> Any:
    """Move a ModulationOut (or list of ModulationOut, for double blocks)
    onto ``device``. Used at the PCIe boundary to construct cuda:1-resident
    modulation slices without mutating the original cuda:0 dict.

    ModulationOut is a dataclass with ``shift``, ``scale``, ``gate``
    fields — reconstruct it on ``device``. For consistency with the
    double-block branch (list of two ModulationOut), handle the list
    case too.
    """
    from .src.layers import ModulationOut

    if isinstance(mod, list):
        return [_modulation_to(m, device) for m in mod]
    if isinstance(mod, ModulationOut):
        return ModulationOut(
            shift=mod.shift.to(device),
            scale=mod.scale.to(device),
            gate=mod.gate.to(device),
        )
    # Final-layer entry: plain list of tensors. Handled separately in
    # forward, but support the fallback path.
    if hasattr(mod, "to"):
        return mod.to(device)
    return mod


def _pin_transformer_to(transformer: torch.nn.Module) -> None:
    """Wrap ``transformer.to()`` so device arguments are stripped (the model
    is already distributed) while dtype changes still pass through.

    Without this, ai-toolkit hooks like ``set_device_state`` (called every
    training step) and ``pipe.transformer.to(self.device_torch)`` (the
    pipeline composer at the bottom of ``ChromaModel.load_model``) would
    silently re-collect the split onto cuda:0.
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
# ``network_mixins.broadcast_and_multiply``) — not Chroma-specific code —
# so they're already installed by the FLUX.2 / Wan / LTX-2 / Qwen-Image /
# HiDream paths if any ran first. Gate all three installers behind a
# single module flag plus per-target attribute flags (defense in depth).
_PATCHES_INSTALLED = False


def _install_external_patches() -> None:
    """Idempotently install the LoRA-placement and multiplier-device patches.

    These hook external ai-toolkit classes — done as runtime patches so
    this file remains a self-contained addition. Safe to call alongside
    the FLUX.2 / Wan / LTX-2 / Qwen-Image / HiDream helpers; all gate on
    the same per-target flags.
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
