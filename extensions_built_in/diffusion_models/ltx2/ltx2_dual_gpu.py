"""Dual-GPU model-parallel path for LTX-2 video+audio LoRA training.

Activates when ``LTX2_DUAL_GPU=true``. Distributes the diffusers
``LTX2VideoTransformer3DModel`` across two CUDA devices with a single
PCIe boundary mid-``transformer_blocks``, keeps the Gemma3 text encoder
on a configurable device (CPU recommended for 32 GB cards), and reuses
the same LoRA / multiplier patches the FLUX.2 and Wan 2.2 dual-GPU paths
install (those hook ai-toolkit-wide classes, not model-specific ones, so
they apply unchanged here — the shared ``_PATCHES_INSTALLED`` flag makes
calling all three safe).

Targets the ai-toolkit LTX-2 trainer
(``extensions_built_in/diffusion_models/ltx2/ltx2.py``). Integration is a
small edit at the call site:

1. Make ``LTX2Model`` inherit from ``LTX2DualGPUMixin`` first
   (MRO ordering: mixin before ``BaseModel``).
2. In ``LTX2Model.load_model``, after ``quantize_model(...)``, call
   ``self.setup_dual_gpu_distribution(transformer, dtype)`` gated by
   ``is_dual_gpu_enabled()`` — and keep the transformer on CPU through
   quantize so the mixin can distribute afterward.

Unlike FLUX.2's per-block hooks, LTX-2 (like Wan 2.2) precomputes all
loop-invariant modulation tensors once before the block loop, so a
single-boundary ``forward`` replacement is sufficient.

Env vars:
    LTX2_DUAL_GPU=true            enable the dual-GPU path
    LTX2_TE_DEVICE=cpu            pin the Gemma3 text encoder to a device
    LTX2_DUAL_GPU_SPLIT_AT=24     override transformer_blocks split index
                                  (default: num_layers // 2)
"""
from __future__ import annotations

import os
import types
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .ltx2 import LTX2Model  # noqa: F401


def is_dual_gpu_enabled() -> bool:
    return os.getenv("LTX2_DUAL_GPU", "false").lower() == "true"


def get_te_device_override() -> torch.device | None:
    """Returns the text-encoder device override if LTX2_TE_DEVICE is set."""
    val = os.getenv("LTX2_TE_DEVICE")
    return torch.device(val) if val else None


class LTX2DualGPUMixin:
    """Mixin for :class:`LTX2Model` that adds the dual-GPU training path.

    The mixin contributes:

    - ``te_device_torch`` attribute (resolved from ``LTX2_TE_DEVICE`` or
      defaulting to ``device_torch``). Always present — used even when
      dual-GPU itself is disabled.
    - ``text_encoder_to`` override that routes Gemma3 moves to
      ``te_device_torch`` instead of ``device_torch``.
    - ``setup_dual_gpu_distribution(transformer, dtype)`` to be called
      from ``load_model`` after the transformer has been quantized (if
      applicable) and before the pipeline composes it. Distributes
      modules across cuda:0/cuda:1, replaces
      ``LTX2VideoTransformer3DModel.forward`` with a split-aware variant,
      pins ``transformer.to()`` to ignore device arguments, and installs
      the LoRA / multiplier patches.
    - ``preserve_dual_gpu_split_on_pipe(pipe)`` — to be checked in place
      of the single-device ``pipe.transformer.to(self.device_torch)``
      call so the distributed layout survives pipeline composition.
    """

    te_device_torch: torch.device

    # ---------------------------------------------------------------- init

    def init_te_device(self) -> None:
        """Resolve ``te_device_torch`` from env var or fall back to model device.

        Must be called from :meth:`LTX2Model.__init__` after the base
        ``__init__`` (which establishes ``self.device_torch``).
        """
        override = get_te_device_override()
        self.te_device_torch = override if override is not None else self.device_torch  # type: ignore[attr-defined]

    # ----------------------------------------------------------- TE override

    def text_encoder_to(self, *args: Any, **kwargs: Any) -> None:
        """Override of BaseModel.text_encoder_to that honors te_device_torch.

        ai-toolkit's :class:`SDTrainer` hook ``hook_before_train_loop``
        calls ``self.sd.text_encoder_to(self.device_torch)``
        unconditionally. With ``LTX2_TE_DEVICE=cpu`` we want Gemma3 to
        stay on CPU regardless. LTX-2 stores the text encoder as a list.
        """
        target = self.te_device_torch
        if isinstance(self.text_encoder, list):  # type: ignore[attr-defined]
            for encoder in self.text_encoder:  # type: ignore[attr-defined]
                encoder.to(target)
        else:
            self.text_encoder.to(target)  # type: ignore[attr-defined]

    # ------------------------------------------------------------ load hook

    def setup_dual_gpu_distribution(self, transformer: torch.nn.Module, dtype: torch.dtype) -> None:
        """Distribute the LTX-2 transformer across cuda:0 and cuda:1.

        Layout:
            cuda:0  — ``proj_in``, ``audio_proj_in``, ``caption_projection``,
                      ``audio_caption_projection``, the timestep/modulation
                      embedders (``time_embed``, ``audio_time_embed``,
                      ``av_cross_attn_*``, ``prompt_adaln`` /
                      ``audio_prompt_adaln`` if present), all four RoPE
                      modules (``rope``, ``audio_rope``, ``cross_attn_rope``,
                      ``cross_attn_audio_rope``), and
                      ``transformer_blocks[:split_at]``
            cuda:1  — ``transformer_blocks[split_at:]``, the output norm/proj
                      layers (``norm_out``, ``proj_out``, ``audio_norm_out``,
                      ``audio_proj_out``), and the output-layer
                      ``scale_shift_table`` / ``audio_scale_shift_table``
                      Parameters

        Also installs the LoRA / multiplier patches that align downstream
        ai-toolkit machinery with the split layout.
        """
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                f"LTX2_DUAL_GPU=true requires >=2 CUDA devices, found "
                f"{torch.cuda.device_count()}."
            )

        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        n_blocks = len(transformer.transformer_blocks)
        override = os.getenv("LTX2_DUAL_GPU_SPLIT_AT")
        split_at = int(override) if override else (n_blocks // 2)
        if not 0 < split_at < n_blocks:
            raise RuntimeError(
                f"LTX2_DUAL_GPU_SPLIT_AT={split_at} out of range "
                f"(transformer has {n_blocks} blocks)."
            )

        self.print_and_status_update(  # type: ignore[attr-defined]
            f"Distributing transformer across {d0} and {d1}"
        )
        self.print_and_status_update(  # type: ignore[attr-defined]
            f"Block split: {split_at} on cuda:0, "
            f"{n_blocks - split_at} on cuda:1"
        )

        # --- cuda:0: input projections, embedders, RoPE, first block half
        transformer.proj_in.to(d0, dtype=dtype)
        transformer.audio_proj_in.to(d0, dtype=dtype)

        # caption projections only exist on LTX-2.0 (use_prompt_embeddings)
        for attr in ("caption_projection", "audio_caption_projection"):
            mod = getattr(transformer, attr, None)
            if mod is not None:
                mod.to(d0, dtype=dtype)

        transformer.time_embed.to(d0, dtype=dtype)
        transformer.audio_time_embed.to(d0, dtype=dtype)
        transformer.av_cross_attn_video_scale_shift.to(d0, dtype=dtype)
        transformer.av_cross_attn_audio_scale_shift.to(d0, dtype=dtype)
        transformer.av_cross_attn_video_a2v_gate.to(d0, dtype=dtype)
        transformer.av_cross_attn_audio_v2a_gate.to(d0, dtype=dtype)

        # prompt modulation adaln layers only exist on LTX-2.3
        for attr in ("prompt_adaln", "audio_prompt_adaln"):
            mod = getattr(transformer, attr, None)
            if mod is not None:
                mod.to(d0, dtype=dtype)

        # All four RoPE modules are nn.Module subclasses, so .to() traverses
        # any registered buffers/params they hold (no plain-attribute gotcha).
        transformer.rope.to(d0)
        transformer.audio_rope.to(d0)
        transformer.cross_attn_rope.to(d0)
        transformer.cross_attn_audio_rope.to(d0)

        for blk in transformer.transformer_blocks[:split_at]:
            blk.to(d0, dtype=dtype)

        # --- cuda:1: second block half + output layers
        for blk in transformer.transformer_blocks[split_at:]:
            blk.to(d1, dtype=dtype)

        transformer.norm_out.to(d1)
        transformer.proj_out.to(d1, dtype=dtype)
        transformer.audio_norm_out.to(d1)
        transformer.audio_proj_out.to(d1, dtype=dtype)

        # scale_shift_table / audio_scale_shift_table are nn.Parameters, not
        # sub-modules; moving the containing module won't reach them because
        # they're referenced directly. Move the underlying storage in-place
        # under no_grad. They are consumed only by the output layers (cuda:1).
        with torch.no_grad():
            transformer.scale_shift_table.data = transformer.scale_shift_table.data.to(d1)
            transformer.audio_scale_shift_table.data = transformer.audio_scale_shift_table.data.to(d1)

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
    """Create an ``LTX2VideoTransformer3DModel.forward`` variant that inserts
    a cuda:1 boundary at ``transformer_blocks[split_at]``.

    Recreates the original forward (see
    ``diffusers/models/transformers/transformer_ltx2.py``,
    ``LTX2VideoTransformer3DModel.forward``) with three additions:
      1. At index ``split_at``, cross BOTH modality streams
         (``hidden_states`` video + ``audio_hidden_states`` audio), both
         text-embedding streams, every precomputed modulation tensor
         (``temb``, ``temb_audio``, the cross-attn scale/shift + gate
         tensors, ``temb_prompt`` / ``temb_prompt_audio``), the encoder
         attention masks, and all four computed RoPE tuples to cuda:1.
      2. After the output layers, move both outputs back to cuda:0 so
         downstream loss ops (which run on the model's nominal device)
         work without a device-mismatch error.
      3. Preserve the ``return_dict`` contract — returns
         ``(output, audio_output)`` or ``AudioVisualModelOutput``.

    The original forward is wrapped with ``@apply_lora_scale``; that
    decorator only scales LoRA when ``attention_kwargs`` carries a scale,
    and ai-toolkit's training path always passes ``attention_kwargs=None``
    (see ``LTX2Model.get_noise_prediction``), so the decorator is a no-op
    here — we mirror the Wan mixin and omit it.
    """
    from diffusers.models.transformers.transformer_ltx2 import AudioVisualModelOutput

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        audio_num_frames: int | None = None,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        isolate_modalities: bool = False,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        use_cross_timestep: bool = False,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ):
        # Determine timestep for audio.
        audio_timestep = audio_timestep if audio_timestep is not None else timestep
        audio_sigma = audio_sigma if audio_sigma is not None else sigma

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = (1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)) * -10000.0
            audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)

        # 1. Prepare RoPE positional embeddings
        if video_coords is None:
            video_coords = self.rope.prepare_video_coords(
                batch_size, num_frames, height, width, hidden_states.device, fps=fps
            )
        if audio_coords is None:
            audio_coords = self.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames, audio_hidden_states.device
            )

        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
        audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)

        video_cross_attn_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :], device=hidden_states.device)
        audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], device=audio_hidden_states.device
        )

        # 2. Patchify input projections
        hidden_states = self.proj_in(hidden_states)
        audio_hidden_states = self.audio_proj_in(audio_hidden_states)

        # 3. Prepare timestep embeddings and modulation parameters
        timestep_cross_attn_gate_scale_factor = (
            self.config.cross_attn_timestep_scale_multiplier / self.config.timestep_scale_multiplier
        )

        # 3.1. Prepare global modality (video and audio) timestep embedding and modulation parameters
        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        temb_audio, audio_embedded_timestep = self.audio_time_embed(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
        audio_embedded_timestep = audio_embedded_timestep.view(batch_size, -1, audio_embedded_timestep.size(-1))

        if self.prompt_modulation:
            # LTX-2.3
            temb_prompt, _ = self.prompt_adaln(
                sigma.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
            temb_prompt_audio, _ = self.audio_prompt_adaln(
                audio_sigma.flatten(), batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype
            )
            temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
            temb_prompt_audio = temb_prompt_audio.view(batch_size, -1, temb_prompt_audio.size(-1))
        else:
            temb_prompt = temb_prompt_audio = None

        # 3.2. Prepare global modality cross attention modulation parameters
        video_ca_timestep = audio_sigma.flatten() if use_cross_timestep else timestep.flatten()
        video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
            video_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
            video_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
            batch_size, -1, video_cross_attn_scale_shift.shape[-1]
        )
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(batch_size, -1, video_cross_attn_a2v_gate.shape[-1])

        audio_ca_timestep = sigma.flatten() if use_cross_timestep else audio_timestep.flatten()
        audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
            audio_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
            audio_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
            batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
        )
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(batch_size, -1, audio_cross_attn_v2a_gate.shape[-1])

        # 4. Prepare prompt embeddings (LTX-2.0)
        if self.config.use_prompt_embeddings:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

            audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(
                batch_size, -1, audio_hidden_states.size(-1)
            )

        # 5. Run transformer blocks
        spatio_temporal_guidance_blocks = spatio_temporal_guidance_blocks or []
        if len(spatio_temporal_guidance_blocks) > 0 and perturbation_mask is None:
            # If STG is being used and perturbation_mask is not supplied, default to perturbing all batch elements.
            perturbation_mask = torch.zeros((batch_size,))
        if perturbation_mask is not None and perturbation_mask.ndim == 1:
            perturbation_mask = perturbation_mask[:, None, None]  # unsqueeze to 3D to broadcast with hidden_states
        all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False
        stg_blocks = set(spatio_temporal_guidance_blocks)

        # Devices of the first block on each side of the boundary.
        d0_dev = next(self.transformer_blocks[0].parameters()).device
        d1_dev = next(self.transformer_blocks[split_at].parameters()).device

        for block_idx, block in enumerate(self.transformer_blocks):
            if block_idx == split_at and d1_dev != d0_dev:
                # PCIe boundary: bridge BOTH modality streams + both text
                # streams + every precomputed modulation tensor + masks +
                # all four RoPE tuples to cuda:1.
                hidden_states = hidden_states.to(d1_dev)
                audio_hidden_states = audio_hidden_states.to(d1_dev)
                encoder_hidden_states = encoder_hidden_states.to(d1_dev)
                audio_encoder_hidden_states = audio_encoder_hidden_states.to(d1_dev)
                temb = temb.to(d1_dev)
                temb_audio = temb_audio.to(d1_dev)
                video_cross_attn_scale_shift = video_cross_attn_scale_shift.to(d1_dev)
                audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.to(d1_dev)
                video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.to(d1_dev)
                audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.to(d1_dev)
                if temb_prompt is not None:
                    temb_prompt = temb_prompt.to(d1_dev)
                if temb_prompt_audio is not None:
                    temb_prompt_audio = temb_prompt_audio.to(d1_dev)
                if encoder_attention_mask is not None:
                    encoder_attention_mask = encoder_attention_mask.to(d1_dev)
                if audio_encoder_attention_mask is not None:
                    audio_encoder_attention_mask = audio_encoder_attention_mask.to(d1_dev)
                # Each RoPE result is a tuple of tensors (cos/sin pairs);
                # move every element across.
                video_rotary_emb = _move_rope(video_rotary_emb, d1_dev)
                audio_rotary_emb = _move_rope(audio_rotary_emb, d1_dev)
                video_cross_attn_rotary_emb = _move_rope(video_cross_attn_rotary_emb, d1_dev)
                audio_cross_attn_rotary_emb = _move_rope(audio_cross_attn_rotary_emb, d1_dev)

            block_perturbation_mask = perturbation_mask if block_idx in stg_blocks else None
            block_all_perturbed = all_perturbed if block_idx in stg_blocks else False

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, audio_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    temb,
                    temb_audio,
                    video_cross_attn_scale_shift,
                    audio_cross_attn_scale_shift,
                    video_cross_attn_a2v_gate,
                    audio_cross_attn_v2a_gate,
                    temb_prompt,
                    temb_prompt_audio,
                    video_rotary_emb,
                    audio_rotary_emb,
                    video_cross_attn_rotary_emb,
                    audio_cross_attn_rotary_emb,
                    encoder_attention_mask,
                    audio_encoder_attention_mask,
                    None,  # self_attention_mask
                    None,  # audio_self_attention_mask
                    None,  # a2v_cross_attention_mask
                    None,  # v2a_cross_attention_mask
                    not isolate_modalities,  # use_a2v_cross_attention
                    not isolate_modalities,  # use_v2a_cross_attention
                    block_perturbation_mask,
                    block_all_perturbed,
                )
            else:
                hidden_states, audio_hidden_states = block(
                    hidden_states=hidden_states,
                    audio_hidden_states=audio_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    temb=temb,
                    temb_audio=temb_audio,
                    temb_ca_scale_shift=video_cross_attn_scale_shift,
                    temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                    temb_ca_gate=video_cross_attn_a2v_gate,
                    temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                    temb_prompt=temb_prompt,
                    temb_prompt_audio=temb_prompt_audio,
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=video_cross_attn_rotary_emb,
                    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                    self_attention_mask=None,
                    audio_self_attention_mask=None,
                    a2v_cross_attention_mask=None,
                    v2a_cross_attention_mask=None,
                    use_a2v_cross_attention=not isolate_modalities,
                    use_v2a_cross_attention=not isolate_modalities,
                    perturbation_mask=block_perturbation_mask,
                    all_perturbed=block_all_perturbed,
                )

        # 6. Output layers (including unpatchification). norm_out / proj_out
        # and the scale_shift_table Parameters live on cuda:1; embedded_timestep
        # came out of the embedders on cuda:0, so bridge it across.
        out_dev = hidden_states.device
        if embedded_timestep.device != out_dev:
            embedded_timestep = embedded_timestep.to(out_dev)
        if audio_embedded_timestep.device != audio_hidden_states.device:
            audio_embedded_timestep = audio_embedded_timestep.to(audio_hidden_states.device)

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        audio_scale_shift_values = self.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
        audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]

        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = self.audio_proj_out(audio_hidden_states)

        # Return on cuda:0 so downstream loss / scatter ops (which run on the
        # model's nominal device) work without a device-mismatch error.
        if output.device != d0_dev:
            output = output.to(d0_dev)
        if audio_output.device != d0_dev:
            audio_output = audio_output.to(d0_dev)

        if not return_dict:
            return (output, audio_output)
        return AudioVisualModelOutput(sample=output, audio_sample=audio_output)

    return forward


def _move_rope(rope_emb: Any, device: torch.device) -> Any:
    """Move a RoPE forward result to ``device``.

    Each ``LTX2AudioVideoRotaryPosEmbed.forward`` returns a tuple of
    tensors (cos/sin pairs); move every tensor element, leaving non-tensor
    entries (e.g. ``None``) untouched.
    """
    if isinstance(rope_emb, torch.Tensor):
        return rope_emb.to(device)
    if isinstance(rope_emb, (tuple, list)):
        moved = [
            (e.to(device) if isinstance(e, torch.Tensor) else _move_rope(e, device))
            for e in rope_emb
        ]
        return type(rope_emb)(moved)
    return rope_emb


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
# ``network_mixins.broadcast_and_multiply``) — not LTX-2-specific code —
# so they're already installed by the FLUX.2 / Wan paths if either ran
# first. Gate all three installers behind a single module flag plus
# per-target attribute flags (defense in depth).
_PATCHES_INSTALLED = False


def _install_external_patches() -> None:
    """Idempotently install the LoRA-placement and multiplier-device patches.

    These hook external ai-toolkit classes — done as runtime patches so
    this file remains a self-contained addition. Safe to call alongside
    the FLUX.2 / Wan helpers; all gate on the same per-target flags.
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
