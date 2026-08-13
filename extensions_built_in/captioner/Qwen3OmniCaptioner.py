from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from collections import OrderedDict

import os
import torch
import torch.nn.functional as F

from toolkit.basic import flush
from toolkit.util.comfy_quant_import import (
    import_comfy_quantized_layers,
    parse_comfy_quant_blob,
)
from toolkit.util.convrot_quant import regular_hadamard

from .BaseCaptioner import BaseCaptioner
from .Qwen3VLCaptioner import patch_qwen_vl_patch_embed
import logging
import traceback
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# frame sampling rate for video captioning
VIDEO_FPS = 2

# fixed generation ceiling under compiled decode: a constant max_length keeps
# the static kv cache (and so the compiled decode graph) at one shape for
# every video; the real per-caption budget is enforced by a stopping criterion
STATIC_MAX_LENGTH = 8192

# single-file comfy-format checkpoint (thinker only, convrot8 int8) produced by
# scripts/convert_vllm_to_comfy.py. This is always what we load — never the
# original bf16 shards.
CONVROT_FILENAME = "qwen3_omni_30b_a3b_instruct_thinker_convrot8.safetensors"
# config + processor (tokenizer, feature extractors) come from the original repo
BASE_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


class OstrisQwen3OmniThinker(Qwen3OmniMoeThinkerForConditionalGeneration):
    """Thinker with static-cache-safe MRoPE handling.

    Upstream breaks under ``cache_implementation="static"``: generate passes a
    prepared 4D bool attention mask, but the forward's rope-delta block does
    ``1 - attention_mask`` and ``get_rope_index`` assumes a 2D long padding
    mask. We compute position_ids ourselves — prefill from the true 2D mask
    (stashed by the caller before generate), decode from cache_position with
    no data-dependent ops — so the upstream block (which only runs when
    position_ids is None) is skipped entirely. Also required for CUDA-graph
    decode: the decode branch is sync-free and shape-static."""

    _pad_mask_2d = None

    # media inputs are consumed at prefill only; keeping them in decode-step
    # inputs makes the compiled decode graph guard on their (per-video) shapes,
    # forcing a recompile on the next video. Dropping them gives the decode
    # graph one fixed signature: it compiles once, ever.
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
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        use_audio_in_video=None,
        video_second_per_grid=None,
        **kwargs,
    ):
        if position_ids is None and input_ids is not None:
            if input_ids.shape[1] > 1 or self.rope_deltas is None:
                # prefill: replicate the upstream math with a valid 2D mask
                mask2d = (
                    attention_mask
                    if attention_mask is not None and attention_mask.dim() == 2
                    else self._pad_mask_2d
                )
                if mask2d is None:
                    mask2d = torch.ones_like(input_ids)
                mask2d = mask2d.long()
                if mask2d.shape[1] != input_ids.shape[1]:
                    # static cache pads the mask out to max_cache_len
                    mask2d = mask2d[:, : input_ids.shape[1]]
                if feature_attention_mask is not None:
                    rope_audio_lengths = torch.sum(feature_attention_mask, dim=1)
                else:
                    rope_audio_lengths = audio_feature_lengths
                delta0 = (1 - mask2d).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    mask2d,
                    use_audio_in_video or False,
                    rope_audio_lengths,
                    video_second_per_grid,
                )
                self.rope_deltas = rope_deltas - delta0
            else:
                # decode: continue from the cache position; sync-free
                batch_size, seq_length = input_ids.shape
                deltas = self.rope_deltas.to(input_ids.device)
                if cache_position is not None:
                    pos = cache_position.view(1, -1) + deltas
                else:
                    # get_seq_length may be a tensor (static cache); keep it on-device
                    past_len = (
                        past_key_values.get_seq_length()
                        if past_key_values is not None
                        else 0
                    )
                    pos = (
                        torch.arange(seq_length, device=input_ids.device).view(1, -1)
                        + past_len
                        + deltas
                    )
                position_ids = pos.unsqueeze(0).expand(3, batch_size, seq_length)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            input_features=input_features,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )


class ConvRot8Experts(torch.nn.Module):
    """Drop-in replacement for Qwen3OmniMoeThinkerTextExperts that keeps the
    fused expert banks in comfy convrot8 storage (regular-Hadamard rotated,
    per-output-row symmetric int8). Experts are dequantized one at a time at
    forward, so the full-precision banks (the bulk of the 30B) never
    materialize."""

    def __init__(
        self, gate_up_q, gate_up_s, gate_up_rot, down_q, down_s, down_rot, dtype
    ):
        super().__init__()
        self.num_experts = gate_up_q.shape[0]
        self.gate_up_rot = gate_up_rot
        self.down_rot = down_rot
        self.out_dtype = dtype
        self.register_buffer("gate_up_q", gate_up_q.contiguous(), persistent=False)
        self.register_buffer("down_q", down_q.contiguous(), persistent=False)
        # fp32 scales stored as uint8 byte views so a later .to(dtype=...) on the
        # model cannot silently cast them (same convention as the cr8 backend)
        self.register_buffer(
            "gate_up_s",
            gate_up_s.detach().float().contiguous().view(torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "down_s",
            down_s.detach().float().contiguous().view(torch.uint8),
            persistent=False,
        )
        # hadamard matrices as buffers: the toolkit's cached builder is a
        # global-dict lookup that torch.compile cannot trace
        self.register_buffer(
            "gate_up_h",
            regular_hadamard(gate_up_rot, torch.device("cpu"), torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "down_h",
            regular_hadamard(down_rot, torch.device("cpu"), torch.float32),
            persistent=False,
        )

    @staticmethod
    def _rotate(w, h, rot):
        shape = w.shape
        return (w.reshape(-1, shape[-1] // rot, rot) @ h).reshape(shape)

    def _dequant(self, qdata, scales_u8, h, rot, i):
        # scales are [E, out, 1]; rotation is self-inverse along the in dim
        scales = scales_u8.view(torch.float32)
        w = qdata[i].float() * scales[i]
        return self._rotate(w, h, rot).to(self.out_dtype)

    def _dequant_batch(self, qdata, scales_u8, h, rot, hit, dtype):
        """Dequantize the hit experts in one shot: [n_hit, out, in]."""
        scales = scales_u8.view(torch.float32)
        w = qdata[hit].float() * scales[hit]
        return self._rotate(w, h, rot).to(dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Fully batched MoE: group tokens by expert (sort + bincount), pad the
        groups to a rectangle, dequantize the hit experts in one op, and run the
        whole layer as two bmms — no per-expert python loop. Decode touches only
        the routed experts' weights; prefill runs every expert in one launch."""
        hidden_dim = hidden_states.shape[1]
        top_k = top_k_index.shape[-1]

        # gate on token count, not pair count: decode (1 token per sequence)
        # must ALWAYS take this path at any batch size — the grouped path's
        # nonzero()/max() are data-dependent, and inside the compiled decode
        # graph they shatter it into per-layer fragments (endless compiles,
        # broken cudagraphs). Extra cost is only duplicate expert dequants
        # (~1.6x traffic at batch 16). Prefill (many tokens, runs eager)
        # still uses the grouped path below.
        if hidden_states.shape[0] <= 32:
            # decode-size batches: one bmm per (token, expert) pair with fixed
            # shapes and NO data-dependent ops — the grouped path below needs
            # nonzero()/max() which each force a GPU sync, and 2 syncs x 48
            # layers per token is exactly what stalls the GPU at small batch
            flat = top_k_index.reshape(-1)
            x_rep = hidden_states.repeat_interleave(top_k, dim=0).unsqueeze(1)
            w_gate_up = self._dequant_batch(
                self.gate_up_q,
                self.gate_up_s,
                self.gate_up_h,
                self.gate_up_rot,
                flat,
                hidden_states.dtype,
            )
            gate, up = torch.bmm(x_rep, w_gate_up.transpose(1, 2)).chunk(2, dim=-1)
            del w_gate_up
            h = F.silu(gate) * up
            w_down = self._dequant_batch(
                self.down_q, self.down_s, self.down_h, self.down_rot, flat, hidden_states.dtype
            )
            out = torch.bmm(h, w_down.transpose(1, 2)).squeeze(1)
            del w_down
            out = out * top_k_weights.reshape(-1, 1)
            return (
                out.view(hidden_states.shape[0], top_k, hidden_dim)
                .sum(dim=1)
                .to(hidden_states.dtype)
            )
        device = hidden_states.device
        dtype = hidden_states.dtype

        flat_expert = top_k_index.reshape(-1)  # [n_tokens * top_k]
        order = flat_expert.argsort()
        sorted_expert = flat_expert[order]
        token_of_pair = order // top_k
        counts = torch.bincount(flat_expert, minlength=self.num_experts)
        hit = counts.nonzero().flatten()
        hit_counts = counts[hit]
        group_size = int(hit_counts.max())
        # rank of each routed pair inside its expert group
        group_start = (torch.cumsum(counts, 0) - counts)[sorted_expert]
        rank = torch.arange(order.shape[0], device=device) - group_start
        slot = torch.searchsorted(hit, sorted_expert)

        padded_x = torch.zeros(
            hit.shape[0], group_size, hidden_dim, device=device, dtype=dtype
        )
        padded_x[slot, rank] = hidden_states[token_of_pair]

        w_gate_up = self._dequant_batch(
            self.gate_up_q, self.gate_up_s, self.gate_up_h, self.gate_up_rot, hit, dtype
        )
        gate, up = torch.bmm(padded_x, w_gate_up.transpose(1, 2)).chunk(2, dim=-1)
        del w_gate_up
        h = F.silu(gate) * up
        w_down = self._dequant_batch(
            self.down_q, self.down_s, self.down_h, self.down_rot, hit, dtype
        )
        out = torch.bmm(h, w_down.transpose(1, 2))
        del w_down

        pair_out = out[slot, rank] * top_k_weights.reshape(-1)[order].unsqueeze(1)
        final_hidden_states = torch.zeros_like(hidden_states)
        final_hidden_states.index_add_(0, token_of_pair, pair_out.to(dtype))
        return final_hidden_states

    def _forward_dequant(self, hidden_states, top_k_index, top_k_weights):
        # mirrors Qwen3OmniMoeThinkerTextExperts.forward with per-expert dequant
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            w_gate_up = self._dequant(
                self.gate_up_q, self.gate_up_s, self.gate_up_h, self.gate_up_rot, expert_idx
            )
            gate, up = F.linear(current_state, w_gate_up).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            w_down = self._dequant(self.down_q, self.down_s, self.down_h, self.down_rot, expert_idx)
            current_hidden_states = F.linear(current_hidden_states, w_down)
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


def swap_convrot_expert_banks(root, state_dict, dtype):
    """Replace each MoE experts module with a ConvRot8Experts holding the
    quantized banks from the checkpoint, consuming their state dict entries.
    Returns (remaining_state_dict, num_swapped)."""
    state_dict = dict(state_dict)
    bank_paths = sorted(
        {
            k[: -len(".gate_up_proj.comfy_quant")]
            for k in state_dict
            if k.endswith(".gate_up_proj.comfy_quant") and ".experts" in k
        }
    )
    for experts_path in bank_paths:
        tensors = {}
        rots = {}
        for proj in ("gate_up_proj", "down_proj"):
            prefix = f"{experts_path}.{proj}"
            conf = parse_comfy_quant_blob(state_dict.pop(f"{prefix}.comfy_quant"))
            if conf.get("format") != "int8_tensorwise" or not conf.get("convrot"):
                raise ValueError(
                    f"Expert bank {prefix} has unsupported quant config {conf}"
                )
            tensors[proj + "_q"] = state_dict.pop(f"{prefix}.weight")
            tensors[proj + "_s"] = state_dict.pop(f"{prefix}.weight_scale")
            rots[proj] = int(conf.get("convrot_groupsize", 256))

        parent_path, _, attr = experts_path.rpartition(".")
        parent = root.get_submodule(parent_path)
        setattr(
            parent,
            attr,
            ConvRot8Experts(
                tensors["gate_up_proj_q"],
                tensors["gate_up_proj_s"],
                rots["gate_up_proj"],
                tensors["down_proj_q"],
                tensors["down_proj_s"],
                rots["down_proj"],
                dtype,
            ),
        )
    return state_dict, len(bank_paths)


class Qwen3OmniCaptioner(BaseCaptioner):
    """Captions videos using their audio track via the Qwen3-Omni thinker,
    loaded from the pre-quantized convrot8 single-file checkpoint."""

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(Qwen3OmniCaptioner, self).__init__(process_id, job, config, **kwargs)

    def _resolve_checkpoint(self) -> str:
        """model_name_or_path can be the checkpoint file itself, a folder
        holding it, or a hub repo. Known local spots under MODELS_PATH
        (text_encoders/, the root, then any subfolder of text_encoders/) are
        searched before downloading; downloads land in
        MODELS_PATH/text_encoders."""
        from toolkit.paths import MODELS_PATH

        name_or_path = self.caption_config.model_name_or_path
        if os.path.isfile(name_or_path):
            return name_or_path
        if os.path.isdir(name_or_path):
            candidate = os.path.join(name_or_path, CONVROT_FILENAME)
            if os.path.exists(candidate):
                return candidate
            files = [f for f in os.listdir(name_or_path) if f.endswith(".safetensors")]
            if len(files) == 1:
                return os.path.join(name_or_path, files[0])
            raise FileNotFoundError(
                f"No {CONVROT_FILENAME} (or single .safetensors) in {name_or_path}"
            )

        te_dir = os.path.join(MODELS_PATH, "text_encoders")
        for candidate in (
            os.path.join(te_dir, CONVROT_FILENAME),
            os.path.join(MODELS_PATH, CONVROT_FILENAME),
        ):
            if os.path.exists(candidate):
                return candidate
        if os.path.isdir(te_dir):
            for dirpath, dirnames, filenames in os.walk(te_dir):
                dirnames.sort()
                if CONVROT_FILENAME in filenames:
                    return os.path.join(dirpath, CONVROT_FILENAME)

        import huggingface_hub

        self.print_and_status_update(
            f"Downloading {CONVROT_FILENAME} from {name_or_path} into {te_dir}"
        )
        return huggingface_hub.hf_hub_download(
            repo_id=name_or_path, filename=CONVROT_FILENAME, local_dir=te_dir
        )

    def load_model(self):
        from accelerate import init_empty_weights
        from safetensors.torch import load_file

        ckpt_path = self._resolve_checkpoint()
        self.print_and_status_update("Loading Qwen3-Omni thinker (convrot8)")

        config = AutoConfig.from_pretrained(BASE_REPO)
        with init_empty_weights(include_buffers=False):
            model = OstrisQwen3OmniThinker(config.thinker_config)
        model.eval()

        # NOTE: flash_attention_2 was tried here and produced degenerate
        # repetitive output on real jobs (likely its padding handling against
        # the fixed-size static cache with left-padded batches); sdpa is
        # correct and nearly as fast, so we stay on it.

        state_dict = load_file(ckpt_path)

        # MoE expert banks stay int8 in ConvRot8Experts modules
        state_dict, num_banks = swap_convrot_expert_banks(
            model, state_dict, self.torch_dtype
        )
        # everything else quantized (attention, vision, audio linears) attaches
        # to the toolkit's convrot8 backend in place — no dequantization
        state_dict, num_quantized = import_comfy_quantized_layers(
            model, state_dict, orig_dtype=self.torch_dtype
        )
        self.print_and_status_update(
            f" - attached {num_banks} expert banks and {num_quantized} ConvRot layers"
        )
        result = model.load_state_dict(state_dict, assign=True, strict=False)
        # the importer already attached weights (and popped + assigned biases)
        # of quantized layers, so load_state_dict reports them as missing
        expected_missing = set()
        for name, module in model.named_modules():
            if hasattr(module, "ostris_quantizer"):
                expected_missing.add(f"{name}.weight")
                expected_missing.add(f"{name}.bias")
        bad_missing = [k for k in result.missing_keys if k not in expected_missing]
        if bad_missing or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch. missing: {bad_missing[:8]} "
                f"unexpected: {result.unexpected_keys[:8]}"
            )
        leftover_meta = [
            n for n, p in model.named_parameters() if p.device.type == "meta"
        ]
        if leftover_meta:
            raise RuntimeError(f"Params never loaded: {leftover_meta[:8]}")

        model.generation_config.pad_token_id = 151643
        model.generation_config.eos_token_id = [151645, 151643]

        # swap the slow bf16 Conv3d patch_embed for an equivalent fast linear
        patch_qwen_vl_patch_embed(model)

        if self.caption_config.quantize:
            print(
                "[AITK] Qwen3-Omni loads pre-quantized (convrot8); the quantize "
                "setting is ignored."
            )

        self.model = model
        self.model.to(self.device_torch)
        self.processor = AutoProcessor.from_pretrained(BASE_REPO)
        flush()

    def _build_messages(self, _file_path: str):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": _file_path},
                    {"type": "text", "text": self.caption_config.caption_prompt},
                ],
            }
        ]

    def _size_kwargs(self):
        max_pixels = self.caption_config.max_res * self.caption_config.max_res
        # shortest_edge/longest_edge are total pixel counts
        # (min_pixels/max_pixels), not edge lengths
        return {
            "shortest_edge": min(131072, max_pixels),
            "longest_edge": max_pixels,
        }

    def _prep_media(self, file_path: str):
        """CPU side of one video, safe to run in a worker thread: decode +
        subsample frames, extract the audio track, render the chat text. At
        batch size 1 the full processor (tokenize, resize, mel) runs here too,
        so the main thread only moves tensors and generates."""
        from transformers.video_utils import load_video
        from transformers.audio_utils import load_audio

        frames = load_video(file_path, fps=VIDEO_FPS)
        if isinstance(frames, tuple):
            frames = frames[0]
        audio = None
        try:
            a = load_audio(file_path, sampling_rate=16000)
            if a is not None and a.size > 0:
                audio = a
        except Exception:
            pass
        text = self.processor.apply_chat_template(
            self._build_messages(file_path), tokenize=False, add_generation_prompt=True
        )
        item = {"file": file_path, "frames": frames, "audio": audio, "text": text}
        if self.caption_config.batch_size <= 1:
            item["inputs"] = self._process_items([item])
        return item

    def _process_items(self, items):
        use_audio = items[0]["audio"] is not None
        return self.processor(
            text=[it["text"] for it in items],
            audio=[it["audio"] for it in items] if use_audio else None,
            videos=[it["frames"] for it in items],
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio,
            fps=VIDEO_FPS,
            do_sample_frames=False,
            size=self._size_kwargs(),
        )

    def _caption_batch(self, items):
        """Batched generate over preprocessed items (all with audio, or all
        silent). Returns captions in item order."""
        use_audio = items[0]["audio"] is not None
        if len(items) == 1 and "inputs" in items[0]:
            inputs = items[0]["inputs"]
        else:
            inputs = self._process_items(items)
        inputs = inputs.to(self.device_torch).to(self.torch_dtype)
        # under static cache, generate hands the forward a prepared 4D mask;
        # the true 2D padding mask is needed for the prefill rope index
        self.model._pad_mask_2d = inputs.get("attention_mask", None)
        gen_kwargs = {}
        if self.model.generation_config.cache_implementation == "static":
            # constant max_length -> constant cache shape -> no recompiles;
            # the actual budget comes from the stopping criterion
            from transformers.generation import MaxLengthCriteria, StoppingCriteriaList

            input_len = inputs["input_ids"].shape[1]
            gen_kwargs["max_length"] = STATIC_MAX_LENGTH
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [MaxLengthCriteria(max_length=input_len + self.caption_config.max_new_tokens)]
            )
        else:
            gen_kwargs["max_new_tokens"] = self.caption_config.max_new_tokens
        generated_ids = self.model.generate(
            **inputs,
            use_audio_in_video=use_audio,
            **gen_kwargs,
        )
        trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        captions = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [c.strip() for c in captions]

    def run_caption_loop(self):
        """Batched pipeline: CPU worker threads decode/preprocess videos ahead
        of the GPU, videos are grouped (with-audio vs silent) into batches, and
        each batch runs one model.generate call so decode work is wide enough
        to saturate the GPU."""
        import concurrent.futures
        from collections import deque

        import tqdm as tqdm_mod

        batch_size = max(1, int(self.caption_config.batch_size))
        # smoothing near 1 weights recent files heavily, so the rate estimate
        # recovers quickly after the slow compile-warmup videos
        pbar = tqdm_mod.tqdm(
            total=len(self.file_paths),
            desc="Captioning files",
            unit="file",
            smoothing=0.9,
        )

        def finish(file_path, caption):
            if caption is not None:
                self.save_caption_for_file(file_path, caption)
            self.step_num += 1
            self.update_step()
            pbar.update(1)

        def flush(bucket):
            if len(bucket) == 0:
                return
            items = list(bucket)
            bucket.clear()
            n_real = len(items)
            # keep the batch shape constant for the compiled decode graph:
            # pad a final partial bucket by repeating the last video
            if (
                self.model.generation_config.cache_implementation == "static"
                and 1 < n_real < batch_size
            ):
                items = items + [items[-1]] * (batch_size - n_real)
            try:
                captions = self._caption_batch(items)[:n_real]
                for it, cap in zip(items[:n_real], captions):
                    finish(it["file"], cap)
            except Exception as e:
                print(f"Batch failed ({e}); retrying files individually")
                traceback.print_exc()
                for it in items[:n_real]:
                    finish(it["file"], self.get_caption_for_file(it["file"]))

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(self.caption_config.num_workers))
        )
        try:
            futures = deque()
            file_iter = iter(self.file_paths)
            # keep a couple of batches of decode work in flight ahead of the GPU
            lookahead = batch_size * 2 + 2
            for _ in range(lookahead):
                path = next(file_iter, None)
                if path is None:
                    break
                futures.append((path, executor.submit(self._prep_media, path)))

            with_audio, silent = [], []
            while futures:
                if self.is_ui_captioner:
                    self.maybe_stop()
                    if self.is_stopping:
                        break
                path, fut = futures.popleft()
                nxt = next(file_iter, None)
                if nxt is not None:
                    futures.append((nxt, executor.submit(self._prep_media, nxt)))
                try:
                    item = fut.result()
                except Exception as e:
                    print(f"Error preprocessing {path}: {e}")
                    finish(path, None)
                    continue
                bucket = with_audio if item["audio"] is not None else silent
                bucket.append(item)
                if len(bucket) >= batch_size:
                    flush(bucket)
            flush(with_audio)
            flush(silent)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            pbar.close()

    def maybe_compile_models(self):
        """CUDA-graph decode: static kv cache + reduce-overhead compile of the
        text model. Each decode step replays as one captured graph, removing
        the per-kernel python/launch gaps that cap GPU utilization at small
        batch sizes. First video per batch shape is slow (compile warmup)."""
        if not self.caption_config.compile:
            return
        import importlib.util

        if importlib.util.find_spec("triton") is None:
            print("[AITK] compile requested but triton is not installed, skipping.")
            return
        # a static (compileable) cache makes generate auto-compile its decode
        # loop into one cuda graph; prefill stays eager. Per-block graphs were
        # tried and don't compose (graph capture must own the in-place kv-cache
        # writes, and cudagraph trees can't span 48 independent graphs), and
        # fusion-only block compile doesn't touch the launch gaps that matter.
        # With prepare_inputs_for_generation stripping per-video media shapes
        # from decode steps, this compiles exactly once and caches to disk.
        self.model.generation_config.cache_implementation = "static"
        print(
            "[AITK] Compiled decode enabled (static cache + cuda graphs). "
            "The first video compiles (~2 min cold, faster once cached)."
        )

    def get_caption_for_file(self, file_path: str) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": file_path,
                        },
                        {"type": "text", "text": self.caption_config.caption_prompt},
                    ],
                }
            ]

            max_pixels = self.caption_config.max_res * self.caption_config.max_res
            # render the chat text only; the media goes to the processor
            # directly so the audio track is interleaved INTO the video block
            # (use_audio_in_video) instead of forming a separate audio segment
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # pull the audio track out of the video file; silent videos fall
            # back to frames only
            from transformers.audio_utils import load_audio

            use_audio = True
            try:
                audio = load_audio(file_path, sampling_rate=16000)
                if audio.size == 0:
                    use_audio = False
            except Exception as audio_err:
                print(
                    f"No audio track for {file_path} ({audio_err}); captioning frames only"
                )
                use_audio = False

            inputs = self.processor(
                text=text,
                audio=[audio] if use_audio else None,
                videos=[file_path],
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio,
                fps=VIDEO_FPS,
                do_sample_frames=True,
                # shortest_edge/longest_edge are total pixel counts
                # (min_pixels/max_pixels), not edge lengths
                size={
                    "shortest_edge": min(131072, max_pixels),
                    "longest_edge": max_pixels,
                },
            )
            inputs = inputs.to(self.device_torch).to(self.torch_dtype)

            generated_ids = self.model.generate(
                **inputs,
                use_audio_in_video=use_audio,
                max_new_tokens=self.caption_config.max_new_tokens,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0].strip()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
            return None
