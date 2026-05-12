"""Dual-GPU model-parallel path for FLUX.2 LoRA training.

Activates when FLUX2_DUAL_GPU=true. Distributes the FLUX.2 transformer
across two CUDA devices with a single PCIe boundary mid-`single_blocks`,
keeps Mistral on a configurable device (CPU recommended for 32GB cards),
and patches ai-toolkit's LoRA machinery to route per-layer LoRA modules
to the device of the transformer layer they wrap.

Validated 2026-05-10 against 2× RTX 5090 (sm_120) with FLUX.2-dev,
ai-toolkit FLUX.2 branch, qfloat8 transformer weights. See accompanying
README for performance numbers and design rationale.

Env vars:
    FLUX2_DUAL_GPU=true              enable the dual-GPU path
    FLUX2_TE_DEVICE=cpu              pin text encoder to a specific device
    FLUX2_DUAL_GPU_SPLIT_AT=24       override single_blocks split index
                                     (default: n_single_blocks // 2)
"""
from __future__ import annotations

import os
import types
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .flux2_model import Flux2Model


def is_dual_gpu_enabled() -> bool:
    return os.getenv("FLUX2_DUAL_GPU", "false").lower() == "true"


def get_te_device_override() -> torch.device | None:
    """Returns the text-encoder device override if FLUX2_TE_DEVICE is set."""
    val = os.getenv("FLUX2_TE_DEVICE")
    return torch.device(val) if val else None


class Flux2DualGPUMixin:
    """Mixin for :class:`Flux2Model` that adds the dual-GPU training path.

    The mixin contributes:

    - ``te_device_torch`` attribute (resolved from ``FLUX2_TE_DEVICE`` or
      defaulting to ``device_torch``). Always present — used even when
      dual-GPU itself is disabled.
    - ``text_encoder_to`` override that routes Mistral moves to
      ``te_device_torch`` instead of ``device_torch``.
    - ``setup_dual_gpu_distribution(transformer)`` to be called from
      ``load_model`` after the transformer has been quantized (if applicable)
      and before the pipeline composes it. Distributes modules across
      cuda:0/cuda:1, replaces ``Flux2.forward`` with a split-aware variant,
      pins ``transformer.to()`` to ignore device arguments, and installs
      the LoRA / multiplier patches.
    - ``preserve_dual_gpu_split_on_pipe(pipe)`` — to be called in place of
      ``pipe.transformer = pipe.transformer.to(self.device_torch)`` so the
      distributed layout survives pipeline composition.
    """

    # Set by Flux2Model.__init__ after super().__init__()
    te_device_torch: torch.device

    # ---------------------------------------------------------------- init

    def init_te_device(self) -> None:
        """Resolve ``te_device_torch`` from env var or fall back to model device.

        Must be called from :meth:`Flux2Model.__init__` after the base
        ``__init__`` (which establishes ``self.device_torch``).
        """
        override = get_te_device_override()
        self.te_device_torch = override if override is not None else self.device_torch
        if self.te_device_torch.type == "cpu":
            self.print_and_status_update(
                "FLUX2_TE_DEVICE=cpu — Mistral will run on CPU. Set "
                "`cache_text_embeddings: true` and `unload_text_encoder: true` "
                "in your config so it runs once at startup; otherwise it runs "
                "on CPU every step (very slow)."
            )

    # ----------------------------------------------------------- TE override

    def text_encoder_to(self, *args, **kwargs):
        """Override of BaseModel.text_encoder_to that honors te_device_torch.

        ai-toolkit's :class:`SDTrainer` hook ``hook_before_train_loop`` calls
        ``self.sd.text_encoder_to(self.device_torch)`` unconditionally. With
        ``FLUX2_TE_DEVICE=cpu`` we want Mistral to stay on CPU regardless.
        """
        target = self.te_device_torch
        if isinstance(self.text_encoder, list):
            for encoder in self.text_encoder:
                encoder.to(target)
        else:
            self.text_encoder.to(target)

    # ------------------------------------------------------------ load hook

    def setup_dual_gpu_distribution(self, transformer, dtype) -> None:
        """Distribute the FLUX.2 transformer across cuda:0 and cuda:1.

        Layout:
            cuda:0  — input projections, 3 modulation modules, all 8
                      DoubleStreamBlocks, first ``split_at`` SingleStreamBlocks
            cuda:1  — remaining SingleStreamBlocks, final_layer

        Also installs the LoRA / multiplier patches that align downstream
        ai-toolkit machinery with the split layout.
        """
        # Split lives on cuda:0 + cuda:1. (Quanto QTensors don't move cleanly
        # to a bare "cuda" device, so keep these explicitly indexed.)
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError(
                "FLUX2_DUAL_GPU=true needs at least 2 CUDA devices (cuda:0 + "
                f"cuda:1); found {torch.cuda.device_count() if torch.cuda.is_available() else 0}. "
                "Unset FLUX2_DUAL_GPU for single-GPU training."
            )
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")

        n_single = len(transformer.single_blocks)
        override = os.getenv("FLUX2_DUAL_GPU_SPLIT_AT")
        if override is not None:
            try:
                split_at = int(override)
            except ValueError:
                raise ValueError(
                    "FLUX2_DUAL_GPU_SPLIT_AT must be an integer; got "
                    f"{override!r}."
                )
            if not (1 <= split_at < n_single):
                raise ValueError(
                    f"FLUX2_DUAL_GPU_SPLIT_AT={split_at} is out of range; "
                    f"must be in [1, {n_single - 1}] for a model with "
                    f"{n_single} single_blocks."
                )
        else:
            split_at = n_single // 2

        self.print_and_status_update(
            f"Distributing transformer across {d0} and {d1}"
        )
        self.print_and_status_update(
            f"Single-block split: {split_at} on {d0}, "
            f"{n_single - split_at} on {d1}"
        )

        # input projections + positional embedder + modulations on d0
        for m in (
            transformer.img_in,
            transformer.time_in,
            transformer.txt_in,
            transformer.pe_embedder,
            transformer.double_stream_modulation_img,
            transformer.double_stream_modulation_txt,
            transformer.single_stream_modulation,
        ):
            m.to(d0, dtype=dtype)
        if transformer.use_guidance_embed:
            transformer.guidance_in.to(d0, dtype=dtype)

        # all double_blocks on d0
        for blk in transformer.double_blocks:
            blk.to(d0, dtype=dtype)

        # split single_blocks at the configured index
        for i, blk in enumerate(transformer.single_blocks):
            blk.to(d0 if i < split_at else d1, dtype=dtype)

        # final_layer on d1
        transformer.final_layer.to(d1, dtype=dtype)

        # Wire up the split-aware forward and pin .to()
        transformer.forward = types.MethodType(
            _make_split_forward(split_at), transformer
        )
        _pin_transformer_to(transformer)

        # External-class patches (LoRA placement + multiplier device align)
        _install_external_patches()

    # ----------------------------------------------------- pipeline composer

    def preserve_dual_gpu_split_on_pipe(self, pipe) -> bool:
        """Return True if we've already distributed; caller should skip the
        single-device ``pipe.transformer.to(device)`` call."""
        return is_dual_gpu_enabled()


# ---------------------------------------------------------------- helpers


def _make_split_forward(split_at: int):
    """Create a Flux2.forward variant that inserts a cuda:1 boundary at
    ``single_blocks[split_at]``.

    The original Flux2.forward expects all modules on a single device; we
    replace it with one that:
      1. Runs input projections + double_blocks loop on cuda:0
      2. Concatenates txt/img and pe_ctx/pe_x
      3. Runs single_blocks[0:split_at] on cuda:0
      4. At index split_at, .to(cuda:1) on img, pe, and single_block_mod
      5. Runs single_blocks[split_at:n] + final_layer on cuda:1
      6. Returns output on cuda:0 for downstream loss/scatter ops
    """
    import torch.utils.checkpoint as _ckpt

    def forward(
        self,
        x,
        x_ids,
        timesteps,
        ctx,
        ctx_ids,
        guidance=None,
    ):
        # Resolve timestep_embedding from the module containing the original
        # forward (it's defined at module scope alongside Flux2 itself).
        from .src.model import timestep_embedding  # type: ignore

        num_txt_tokens = ctx.shape[1]
        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        if self.use_guidance_embed:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        txt = self.txt_in(ctx)
        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        # double_blocks on cuda:0
        for block in self.double_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img, txt = _ckpt.checkpoint(
                    block, img, txt, pe_x, pe_ctx,
                    double_block_mod_img, double_block_mod_txt,
                    use_reentrant=False,
                )
            else:
                img, txt = block(
                    img, txt, pe_x, pe_ctx,
                    double_block_mod_img, double_block_mod_txt,
                )

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)

        d0_dev = next(self.single_blocks[0].parameters()).device
        d1_dev = next(self.single_blocks[split_at].parameters()).device

        for i, block in enumerate(self.single_blocks):
            if i == split_at and d1_dev != d0_dev:
                img = img.to(d1_dev)
                pe = pe.to(d1_dev)
                single_block_mod = (
                    tuple(t.to(d1_dev) for t in single_block_mod)
                    if isinstance(single_block_mod, tuple)
                    else single_block_mod.to(d1_dev)
                )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img = _ckpt.checkpoint(
                    block, img, pe, single_block_mod,
                    use_reentrant=False,
                )
            else:
                img = block(img, pe, single_block_mod)

        img = img[:, num_txt_tokens:, ...]
        vec_for_final = vec.to(d1_dev) if vec.device != d1_dev else vec
        img = self.final_layer(img, vec_for_final)

        # Return on d0 so downstream loss / scatter ops on img_ids (which
        # live on d0) work without a device-mismatch error.
        if img.device != d0_dev:
            img = img.to(d0_dev)
        return img

    return forward


def _pin_transformer_to(transformer) -> None:
    """Wrap ``transformer.to()`` so device arguments are stripped (the model
    is already distributed) while dtype changes still pass through.

    Without this, ai-toolkit hooks like ``set_device_state`` (called every
    training step) and ``pipe.transformer.to(self.device_torch)`` (the
    pipeline composer) would silently re-collect the split onto cuda:0.
    """
    _orig_to = transformer.to

    def _split_preserving_to(*args, **kwargs):
        filtered_args = [
            a for a in args
            if not isinstance(a, (str, torch.device, int))
        ]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        if not filtered_args and not filtered_kwargs:
            return transformer
        return _orig_to(*filtered_args, **filtered_kwargs)

    transformer.to = _split_preserving_to


def _install_external_patches() -> None:
    """Idempotently install the LoRA-placement and multiplier-device patches.

    These hook external ai-toolkit classes (``ToolkitNetworkMixin``,
    ``LoRANetwork``, ``network_mixins.broadcast_and_multiply``) — done as
    runtime patches so this file remains a self-contained addition.

    For maintainability, each patch is a small no-op if dual-GPU isn't
    actually distributing (heuristic: if ``lora.org_module[0]``'s parameter
    device matches the requested device, nothing changes).
    """
    _patch_force_to()
    _patch_apply_to_with_lazy_forward()
    _patch_broadcast_and_multiply()


def _patch_force_to() -> None:
    """``ToolkitNetworkMixin.force_to`` moves the network's submodules to a
    single device. Override so each LoRA module follows the device of the
    layer it wraps.
    """
    from toolkit.network_mixins import ToolkitNetworkMixin

    if getattr(ToolkitNetworkMixin, "_dual_gpu_patched", False):
        return

    def _per_layer_force_to(self_n, device, dtype):
        # Move the network container itself first (usually no params).
        try:
            self_n.to(device, dtype)
        except Exception as e:
            print(f"[flux2_dual_gpu] network.to({device}, {dtype}) failed: {e}")
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
            # Also explicitly route children — defense in depth.
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

    The lazy-pin is the load-bearing safety net — any later ai-toolkit hook
    that calls ``network.to(device)`` (e.g., from accelerate.prepare) is
    silently corrected on the next forward.
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
            # Re-bind org_module.forward so it sees our pinned wrapper
            # (apply_to captured the un-wrapped version).
            try:
                lora.org_module[0].forward = lora.forward
            except Exception as e:
                print(
                    "[flux2_dual_gpu] could not rebind org_module forward for "
                    f"{getattr(lora, 'lora_name', lora)}: {e}"
                )
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
