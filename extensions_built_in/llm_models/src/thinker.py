"""Qwen2.5-Omni thinker with the toolkit's fast paths, shared by the ACE-Step
captioner and the qwen25_omni trainable model:
  - ostris_sdpa: grouped single-query decode attention (sdpa's masked kernel is
    ~1 ms/layer against a static cache)
  - OstrisQwen25OmniAudioEncoder: per-window batched attention instead of a dense
    whole-clip block-diagonal mask (fp32-identical, ~150x less attention work)
  - OstrisQwen25OmniThinker: static-cache-safe MRoPE handling for compiled decode
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AttentionInterface, AttentionMaskInterface
from transformers import Qwen2_5OmniForConditionalGeneration
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

PAD_TOKEN_ID = 151643
EOS_TOKEN_IDS = [151645, 151643]


def ostris_sdpa_attention_forward(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs
):
    """sdpa for prefill; for single-query decode against the static cache sdpa's
    masked (memory-efficient) kernel takes ~1ms per layer, 27ms/token on the 7B.
    A grouped matmul reads the cache once instead."""
    if query.shape[2] != 1:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
        )
    b, hq, _, d = query.shape
    hk = key.shape[1]
    g = hq // hk
    L = key.shape[2]
    if scaling is None:
        scaling = d**-0.5
    # scores land in fp32 straight from the bf16 GEMM: Qwen2.5's sink logits reach ~1e4,
    # where a bf16 ulp is 64, so a bf16 score tensor rounds the softmax into a coin flip
    q = query.reshape(b * hk, g, d)
    kT = key.reshape(b * hk, L, d).transpose(-1, -2)
    scores = torch.bmm(q, kT, out_dtype=torch.float32).view(b, hk, g, L) * scaling
    if attention_mask is not None:
        mask = attention_mask[:, :, -1:, :L]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask.float()
    probs = torch.softmax(scores, dim=-1).to(query.dtype)
    out = torch.bmm(probs.view(b * hk, g, L), value.reshape(b * hk, L, d))  # [B*hk, g, d]
    return out.view(b, hq, 1, d).transpose(1, 2).contiguous(), None


AttentionInterface.register("ostris_sdpa", ostris_sdpa_attention_forward)
AttentionMaskInterface.register("ostris_sdpa", sdpa_mask)


class OstrisQwen25OmniAudioEncoder(Qwen2_5OmniAudioEncoder):
    """Upstream flattens every 2s window of the whole clip into one sequence
    and attends with a dense [S, S] block-diagonal mask (S = 15000 for 300s):
    O(S^2) memory and time for attention that is only ever within a window.
    Batch the windows instead: [num_windows, 100, d] with a key-padding mask
    for the short tail window. Same math, ~150x less attention work."""

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None, **kwargs):
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.full(
            (chunk_num.sum(),), self.n_window * 2, dtype=torch.long, device=feature_lens.device
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)
        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        x = F.gelu(self.conv1(padded_feature)) * padded_mask
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        x = x + self.positional_embedding.positional_embedding[: x.shape[1], :].unsqueeze(0).to(x.dtype)
        key_mask = None
        if not bool(padded_mask_after_cnn.all()):
            key_mask = padded_mask_after_cnn[:, None, None, :]
        n, seq_len, dim = x.shape
        for layer in self.layers:
            attn = layer.self_attn
            residual = x
            h = layer.self_attn_layer_norm(x)
            q = attn.q_proj(h).view(n, seq_len, attn.num_heads, -1).transpose(1, 2)
            k = attn.k_proj(h).view(n, seq_len, attn.num_heads, -1).transpose(1, 2)
            v = attn.v_proj(h).view(n, seq_len, attn.num_heads, -1).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, scale=attn.scaling)
            x = residual + attn.out_proj(o.transpose(1, 2).reshape(n, seq_len, dim))
            residual = x
            h = layer.final_layer_norm(x)
            x = residual + layer.fc2(layer.activation_fn(layer.fc1(h)))
        hidden_states = x[padded_mask_after_cnn]
        outs = []
        for each in hidden_states.split(aftercnn_lens.tolist(), dim=0):
            each = self.avg_pooler(each.transpose(0, 1)).transpose_(0, 1)
            outs.append(self.proj(self.ln_post(each)))
        return BaseModelOutputWithPooling(last_hidden_state=torch.cat(outs, dim=0))


class OstrisQwen25OmniThinker(Qwen2_5OmniThinkerForConditionalGeneration):
    """Thinker with static-cache-safe MRoPE handling (same fix as the Qwen3-Omni
    captioner): under a compileable cache generate hands forward a 4D mask, but
    upstream's rope block does `1 - attention_mask` and get_rope_index needs the
    2D padding mask. Prefill uses the true 2D mask stashed by the caller; decode
    continues from the cache length with no data-dependent ops."""

    _pad_mask_2d = None

    # per-file media shapes must not reach the compiled decode graph or it
    # recompiles for every file
    _PREFILL_ONLY_KEYS = (
        "input_features",
        "feature_attention_mask",
        "audio_feature_lengths",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "video_second_per_grid",
    )

    def enable_gradient_checkpointing(self):
        """Trainer hook. transformers gates its own layer checkpointing on .training,
        but the toolkit trains the frozen base in eval mode with adapters on top, so
        wrap the text-stack decoder layers to checkpoint whenever autograd is recording."""
        for layer in self.model.layers:
            if getattr(layer, "_aitk_checkpointed", False):
                continue
            orig_forward = layer.forward

            def forward(*args, _orig=orig_forward, **kwargs):
                if torch.is_grad_enabled():
                    return torch.utils.checkpoint.checkpoint(_orig, *args, use_reentrant=False, **kwargs)
                return _orig(*args, **kwargs)

            layer.forward = forward
            layer._aitk_checkpointed = True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        ids = model_inputs.get("input_ids", None)
        if ids is not None and ids.shape[1] == 1:
            for key in self._PREFILL_ONLY_KEYS:
                model_inputs.pop(key, None)
        return model_inputs

    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        use_audio_in_video=None,
        video_second_per_grid=None,
        **kwargs,
    ):
        if position_ids is None and input_ids is not None:
            if input_ids.shape[1] > 1 or self.rope_deltas is None:
                mask2d = (
                    attention_mask
                    if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2
                    else self._pad_mask_2d
                )
                if mask2d is None:
                    mask2d = torch.ones_like(input_ids)
                mask2d = mask2d.long()
                if mask2d.shape[1] != input_ids.shape[1]:
                    mask2d = mask2d[:, : input_ids.shape[1]]
                audio_lens = (
                    feature_attention_mask.sum(1)
                    if feature_attention_mask is not None
                    else audio_feature_lengths
                )
                delta0 = (1 - mask2d).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    mask2d,
                    use_audio_in_video or False,
                    audio_lens,
                    video_second_per_grid,
                )
                self.rope_deltas = rope_deltas - delta0
            else:
                batch_size, seq_length = input_ids.shape
                # static cache length is a device tensor: stays sync-free
                past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
                pos = (
                    torch.arange(seq_length, device=input_ids.device).view(1, -1)
                    + past_len
                    + self.rope_deltas.to(input_ids.device)
                )
                position_ids = pos.unsqueeze(0).expand(3, batch_size, seq_length)
        return super().forward(
            input_ids=input_ids,
            input_features=input_features,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )


def prepare_thinker(full: Qwen2_5OmniForConditionalGeneration):
    """Strip a loaded Qwen2.5-Omni to its thinker with the fast paths attached."""
    # drop the talker before anything touches the GPU
    full.disable_talker()
    model = full.thinker
    model.__class__ = OstrisQwen25OmniThinker
    model.audio_tower.__class__ = OstrisQwen25OmniAudioEncoder
    model.model.config._attn_implementation = "ostris_sdpa"
    model.generation_config.eos_token_id = full.generation_config.eos_token_id
    model.generation_config.pad_token_id = (
        full.generation_config.pad_token_id
        if full.generation_config.pad_token_id is not None
        else 151643
    )
    return model


def attach_fast_paths(thinker):
    """Swap in the fast subclasses on an already-built thinker (any load path)."""
    thinker.__class__ = OstrisQwen25OmniThinker
    thinker.audio_tower.__class__ = OstrisQwen25OmniAudioEncoder
    thinker.model.config._attn_implementation = "ostris_sdpa"
    if thinker.generation_config.eos_token_id is None:
        thinker.generation_config.eos_token_id = list(EOS_TOKEN_IDS)
    if thinker.generation_config.pad_token_id is None:
        thinker.generation_config.pad_token_id = PAD_TOKEN_ID
    return thinker


def load_thinker_single_file(ckpt_path: str, thinker_config, dtype: torch.dtype):
    """Build a thinker from a single-file checkpoint in the comfy layout written
    by scripts/convert_vllm_to_comfy.py (bf16 or convrot8-quantized). Returns
    (thinker, is_prequantized)."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    from toolkit.util.comfy_quant_import import import_comfy_quantized_layers

    with init_empty_weights(include_buffers=False):
        thinker = OstrisQwen25OmniThinker(thinker_config)
    thinker.eval()
    sd = load_file(ckpt_path)
    prequantized = any(k.endswith(".comfy_quant") for k in sd)
    if prequantized:
        sd, _ = import_comfy_quantized_layers(thinker, sd, orig_dtype=dtype)
    sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
    result = thinker.load_state_dict(sd, assign=True, strict=False)
    allowed = set()
    for name, module in thinker.named_modules():
        if hasattr(module, "ostris_quantizer"):
            allowed.add(f"{name}.weight")
            allowed.add(f"{name}.bias")
    missing = [k for k in result.missing_keys if k not in allowed]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Qwen2.5-Omni checkpoint mismatch. missing: {missing[:8]} unexpected: {result.unexpected_keys[:8]}"
        )
    leftover = [n for n, p in thinker.named_parameters() if p.device.type == "meta"]
    if leftover:
        raise RuntimeError(f"Params never loaded: {leftover[:8]}")
    # quantized layers own their buffers; cast only the plain float tensors
    for module in thinker.modules():
        if hasattr(module, "ostris_quantizer"):
            continue
        for pname, prm in list(module.named_parameters(recurse=False)):
            if prm.is_floating_point() and prm.dtype != dtype:
                prm.data = prm.data.to(dtype)
    attach_fast_paths(thinker)
    return thinker, prequantized
