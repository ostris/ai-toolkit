from typing import Optional

try:
    import librosa
except ImportError:
    librosa = None
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from collections import OrderedDict
import traceback

from optimum.quanto import freeze
from toolkit.basic import flush
from toolkit.util.quantize import quantize, get_qtype

from .BaseCaptioner import BaseCaptioner, CaptionConfig
import transformers
import logging
import warnings

# transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

TARGET_SAMPLE_RATE = 16000
CAPTIONER_ID = "ACE-Step/acestep-captioner"
TRANSCRIBER_ID = "ACE-Step/acestep-transcriber"

LYRICS_PROMPT = "*Task* Transcribe this audio in detail"
CAPTION_PROMPT = "*Task* Describe this music in detail. Include genre, mood, instrumentation, tempo feel, and vocal style if present."

# same budget the Qwen2.5-Omni top-level generate used (thinker_max_new_tokens)
MAX_NEW_TOKENS = 1024
# fixed cache length under compiled decode: the processor caps audio at 300s
# (7500 audio tokens) so prompt + MAX_NEW_TOKENS always fits; a constant keeps
# the static kv cache (and the compiled decode graph) at one shape for every file
STATIC_MAX_LENGTH = 8704

# Key profiles for Krumhansl-Schmuckler key detection
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ═══════════════════════════════════════════════════════════════════════════════
# Audio analysis (BPM, key, time signature) via librosa
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_audio(audio_path):
    """Extract BPM, key, and time signature from audio using librosa."""
    if librosa is None:
        raise ImportError(
            "librosa is required for the AceStep captioner but is not "
            "installed (no numba/llvmlite wheels for this platform yet)."
        )
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, "__len__"):
        tempo = tempo[0]
    bpm = int(round(float(tempo)))

    # Key detection via chroma correlation with key profiles
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = chroma.mean(axis=1)
    major_corrs = np.array(
        [np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_avg)[0, 1] for i in range(12)]
    )
    minor_corrs = np.array(
        [np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_avg)[0, 1] for i in range(12)]
    )

    best_major_idx = major_corrs.argmax()
    best_minor_idx = minor_corrs.argmax()
    if major_corrs[best_major_idx] >= minor_corrs[best_minor_idx]:
        keyscale = f"{KEY_NAMES[best_major_idx]} major"
    else:
        keyscale = f"{KEY_NAMES[best_minor_idx]} minor"

    # Time signature estimation from beat strength pattern
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_est, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    if len(beats) >= 8:
        beat_strengths = onset_env[beats]
        # Check 3/4 vs 4/4 by looking at periodicity of strong beats
        acf = np.correlate(
            beat_strengths - beat_strengths.mean(),
            beat_strengths - beat_strengths.mean(),
            mode="full",
        )
        acf = acf[len(acf) // 2 :]
        if len(acf) > 6:
            # Look at autocorrelation peaks at lag 3 vs lag 4
            score_3 = acf[3] if len(acf) > 3 else 0
            score_4 = acf[4] if len(acf) > 4 else 0
            timesig = "3" if score_3 > score_4 * 1.2 else "4"
        else:
            timesig = "4"
    else:
        timesig = "4"

    return {
        "bpm": bpm,
        "keyscale": keyscale,
        "timesignature": timesig,
        "duration": int(round(duration)),
    }



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
    if scaling is None:
        scaling = d**-0.5
    q = query.view(b, hk, hq // hk, d)
    scores = torch.matmul(q, key.transpose(-1, -2)) * scaling  # [B, hk, g, L]
    if attention_mask is not None:
        mask = attention_mask[:, :, -1:, : key.shape[2]]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    out = torch.matmul(probs, value)  # [B, hk, g, d]
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


class AceStepCaptionConfig(CaptionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fixed_caption: Optional[str] = kwargs.get("fixed_caption", None)


class AceStepCaptioner(BaseCaptioner):
    caption_config_class = AceStepCaptionConfig
    caption_config: AceStepCaptionConfig

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(AceStepCaptioner, self).__init__(process_id, job, config, **kwargs)

    def _load_thinker(self, name_or_path: str, label: str):
        """Load a Qwen2.5-Omni checkpoint and keep only its thinker."""
        self.print_and_status_update(f"Loading {label} model")
        full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            name_or_path,
            dtype=self.torch_dtype,
            device_map="cpu",
        )
        model = prepare_thinker(full)
        model.to(self.device_torch)
        if self.caption_config.quantize:
            self.print_and_status_update(f"Quantizing {label} model")
            quantize(model, weights=get_qtype(self.caption_config.qtype))
            freeze(model)
            flush()
        processor = Qwen2_5OmniProcessor.from_pretrained(name_or_path)
        if self.caption_config.low_vram:
            model.to("cpu")
        return model, processor

    def load_model(self):
        self.model, self.processor = self._load_thinker(
            self.caption_config.model_name_or_path, "transcriber"
        )
        self.model2 = None
        self.processor2 = None
        if self.caption_config.fixed_caption is None:
            self.model2, self.processor2 = self._load_thinker(
                self.caption_config.model_name_or_path2, "captioner"
            )
        flush()

    def maybe_compile_models(self):
        """CUDA-graph decode via a static kv cache (see Qwen3OmniCaptioner):
        eager HF decode on the 7B thinker is launch-bound at ~17 tok/s."""
        if not self.caption_config.compile:
            return
        if self.caption_config.low_vram:
            print("[AITK] low_vram is on; skipping compiled decode.")
            return
        import importlib.util

        if importlib.util.find_spec("triton") is None:
            print("[AITK] compile requested but triton is not installed, skipping.")
            return
        for model in (self.model, self.model2):
            if model is not None:
                model.generation_config.cache_implementation = "static"
        print(
            "[AITK] Compiled decode enabled (static cache + cuda graphs). "
            "The first file per model compiles (~2 min cold, faster once cached)."
        )

    @staticmethod
    def _make_inputs(processor, audio_data, prompt_text):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "<|audio_bos|><|AUDIO|><|audio_eos|>"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        return processor(
            text=text,
            audio=[audio_data],
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            sampling_rate=TARGET_SAMPLE_RATE,
        )

    def _prep_file(self, file_path: str) -> dict:
        """CPU side of one file (runs in a worker thread): librosa analysis,
        decode + resample, and the mel/tokenize pass for each model."""
        analysis = analyze_audio(file_path)
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SAMPLE_RATE)
        audio_data = waveform.squeeze(0).numpy()
        item = {
            "file": file_path,
            "analysis": analysis,
            "lyrics_inputs": self._make_inputs(self.processor, audio_data, LYRICS_PROMPT),
        }
        if self.model2 is not None:
            item["caption_inputs"] = self._make_inputs(
                self.processor2, audio_data, CAPTION_PROMPT
            )
        return item

    def _generate(self, model, processor, inputs) -> str:
        inputs = inputs.to(model.device).to(model.dtype)
        # a generate that dies between static-cache creation and its first
        # forward leaves a half-built cache that breaks every later call
        stale_cache = getattr(model, "_cache", None)
        if stale_cache is not None and not stale_cache.is_initialized:
            del model._cache
        model._pad_mask_2d = inputs.get("attention_mask", None)
        input_len = inputs["input_ids"].shape[1]
        gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
        if model.generation_config.cache_implementation == "static":
            if input_len + 16 < STATIC_MAX_LENGTH:
                from transformers.generation import MaxLengthCriteria, StoppingCriteriaList

                gen_kwargs = {
                    "max_length": STATIC_MAX_LENGTH,
                    "stopping_criteria": StoppingCriteriaList(
                        [MaxLengthCriteria(max_length=min(input_len + MAX_NEW_TOKENS, STATIC_MAX_LENGTH))]
                    ),
                }
            else:
                # prompt would not fit the fixed cache; eager for this file
                gen_kwargs["cache_implementation"] = "dynamic"
        generated = model.generate(**inputs, **gen_kwargs)
        output = processor.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output.strip()

    def get_audio_lyrics(self, inputs) -> str:
        if (
            self.caption_config.low_vram
            and self.model2 is not None
            and self.model2.device != torch.device("cpu")
        ):
            self.model2.to("cpu")
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        return self._generate(self.model, self.processor, inputs)

    def get_audio_caption(self, inputs) -> str:
        if self.caption_config.low_vram and self.model.device != torch.device("cpu"):
            self.model.to("cpu")
        if self.model2.device == torch.device("cpu"):
            self.model2.to(self.device_torch)
        return self._generate(self.model2, self.processor2, inputs)

    def _caption_item(self, item: dict) -> str:
        analysis = item["analysis"]
        lyrics = self.get_audio_lyrics(item["lyrics_inputs"])

        language = "en"
        if "# Languages" in lyrics and "# Lyrics" in lyrics:
            language = lyrics.split("# Languages")[1].split("# Lyrics")[0]
            # remove newlines and extra spaces from language
            language = language.replace("\n", "").strip()
            lyrics = lyrics.split("# Lyrics")[1].strip()

        if self.caption_config.fixed_caption is not None:
            caption = self.caption_config.fixed_caption
        else:
            caption = self.get_audio_caption(item["caption_inputs"])

        output = f"<CAPTION>\n{caption}\n</CAPTION>\n"
        output += f"<LYRICS>\n{lyrics}\n</LYRICS>\n"
        output += f"<BPM>{analysis['bpm']}</BPM>\n"
        output += f"<KEYSCALE>{analysis['keyscale']}</KEYSCALE>\n"
        output += f"<TIMESIGNATURE>{analysis['timesignature']}</TIMESIGNATURE>\n"
        output += f"<DURATION>{analysis['duration']}</DURATION>\n"
        output += f"<LANGUAGE>{language}</LANGUAGE>"
        return output

    def run_caption_loop(self):
        """Worker threads run librosa + decode + feature extraction ahead of
        the GPU so the main thread only moves tensors and generates."""
        import concurrent.futures
        from collections import deque

        import tqdm as tqdm_mod

        pbar = tqdm_mod.tqdm(
            total=len(self.file_paths), desc="Captioning files", unit="file", smoothing=0.9
        )

        def finish(file_path, caption):
            if caption is not None:
                self.save_caption_for_file(file_path, caption)
            self.step_num += 1
            self.update_step()
            pbar.update(1)

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(self.caption_config.num_workers))
        )
        try:
            futures = deque()
            file_iter = iter(self.file_paths)
            lookahead = max(2, int(self.caption_config.num_workers)) + 1
            for _ in range(lookahead):
                path = next(file_iter, None)
                if path is None:
                    break
                futures.append((path, executor.submit(self._prep_file, path)))
            while futures:
                if self.is_ui_captioner:
                    self.maybe_stop()
                    if self.is_stopping:
                        break
                path, fut = futures.popleft()
                nxt = next(file_iter, None)
                if nxt is not None:
                    futures.append((nxt, executor.submit(self._prep_file, nxt)))
                try:
                    item = fut.result()
                except Exception as e:
                    print(f"Error preprocessing {path}: {e}")
                    finish(path, None)
                    continue
                try:
                    caption = self._caption_item(item)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    traceback.print_exc()
                    caption = None
                finish(path, caption)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            pbar.close()

    def get_caption_for_file(self, file_path: str) -> str:
        try:
            return self._caption_item(self._prep_file(file_path))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
            return None
