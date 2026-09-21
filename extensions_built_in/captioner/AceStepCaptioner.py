from typing import Optional

try:
    import librosa
except ImportError:
    librosa = None
import threading

import numpy as np
import torch
import torchaudio
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.generation import LogitsProcessor, LogitsProcessorList
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
# lyric transcription loops on repeated musical phrases more than captioning
TRANSCRIBER_REPETITION_PENALTY = 1.05
# whisper-style temperature fallback when greedy decode degenerates into a
# token loop; the penalty alone cannot break a confident single-token loop
TRANSCRIBER_RETRIES = (
    {"do_sample": True, "temperature": 0.4, "repetition_penalty": 1.20},
    {"do_sample": True, "temperature": 0.7, "repetition_penalty": 1.30},
    {"do_sample": True, "temperature": 1.0, "repetition_penalty": 1.50},
)
# a loop is a unit of up to LOOP_MAX_PERIOD tokens repeated at least
# LOOP_REPEATS times over a span of at least LOOP_MIN_SPAN tokens; sampled
# decode jitters a token now and then, so the match is a fraction, not exact.
# real outros repeat a line up to ~12 times (~128 tokens); loops never stop
LOOP_MAX_PERIOD = 32
LOOP_REPEATS = 4
LOOP_MIN_SPAN = 160
LOOP_MATCH_FRACTION = 0.9
# on a detected loop the unit's tokens are banned for max(2*period, this) steps
LOOP_BAN_STEPS = 8
# loops broken before decode is forced to stop
LOOP_BREAKS_BEFORE_STOP = 2
# repeats of the looping unit kept when truncating
LOOP_KEEP_REPEATS = 2
CAPTIONER_REPETITION_PENALTY = 1.05
# fixed cache length under compiled decode: the processor caps audio at 300s
# (7500 audio tokens) so prompt + MAX_NEW_TOKENS always fits; a constant keeps
# the static kv cache (and the compiled decode graph) at one shape for every file
STATIC_MAX_LENGTH = 8704
# silence trim on the extracted vocal stem: frames below this fraction of the
# peak frame RMS (-40 dB) at the head and tail are cut, keeping a short margin
SILENCE_FRAME_SECONDS = 0.02
SILENCE_RMS_FRACTION = 0.01
SILENCE_MARGIN_SECONDS = 0.25

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



from extensions_built_in.llm_models.src.thinker import (  # noqa: E402  shared fast paths
    OstrisQwen25OmniAudioEncoder,
    OstrisQwen25OmniThinker,
    ostris_sdpa_attention_forward,
    prepare_thinker,
)


# output layouts:
#   ace_step - <CAPTION>/<LYRICS>/<BPM>/<KEYSCALE>/<TIMESIGNATURE>/<DURATION>/<LANGUAGE> tags
#   yue2     - description paragraph, then a [Lyrics] line and the sectioned lyrics
#              (YuE2's native prompt layout; no librosa analysis needed)
CAPTION_FORMATS = ("ace_step", "yue2")


def find_loop_period(tokens) -> int:
    """Period of a unit repeated across the tail of the token ids, 0 if none."""
    arr = np.asarray(tokens)
    for period in range(1, LOOP_MAX_PERIOD + 1):
        span = max(LOOP_REPEATS * period, LOOP_MIN_SPAN)
        if span > arr.shape[0]:
            continue
        tail = arr[-span:]
        if np.mean(tail[period:] == tail[:-period]) >= LOOP_MATCH_FRACTION:
            return period
    return 0


def find_loop_start(tokens, period: int) -> int:
    """Index where the periodic run at the tail of the token ids begins."""
    # a mismatch run longer than this is real content, shorter is jitter
    tolerance = max(period, 4)
    start = len(tokens) - 1
    mismatch_run = 0
    while start - period >= 0:
        if tokens[start] == tokens[start - period]:
            mismatch_run = 0
        else:
            mismatch_run += 1
            if mismatch_run > tolerance:
                start += mismatch_run  # back out of the real content
                break
        start -= 1
    return start + 1


class LoopBreaker(LogitsProcessor):
    """Breaks token loops inside a single decode: bans the repeating unit's
    tokens for a stretch so the model has to move on, and forces EOS once it
    has looped LOOP_BREAKS_BEFORE_STOP times. Batch size 1 only."""

    def __init__(self, prompt_len: int, eos_ids):
        self.prompt_len = prompt_len
        self.eos_ids = list(eos_ids)
        self.generated: list = []
        self.banned: list = []
        self.ban_until = 0
        self.window_start = 0  # detection only looks at tokens after the last break
        self.breaks = 0
        self.stopped = False
        # first loop: where it began, its period, and where decode resumed after the ban
        self.loop_start = 0
        self.loop_period = 0
        self.loop_resume = 0

    def __call__(self, input_ids, scores):
        n = input_ids.shape[1] - self.prompt_len
        if n > len(self.generated):
            self.generated.extend(input_ids[0, self.prompt_len + len(self.generated):].tolist())
        if self.stopped:
            return self._force_eos(scores)
        if n < self.ban_until:
            scores[:, self.banned] = float("-inf")
            return scores
        period = find_loop_period(self.generated[self.window_start:])
        if not period:
            return scores
        self.breaks += 1
        if self.breaks >= LOOP_BREAKS_BEFORE_STOP:
            self.stopped = True
            return self._force_eos(scores)
        self.banned = sorted(set(self.generated[-period:]))
        self.ban_until = n + max(2 * period, LOOP_BAN_STEPS)
        self.window_start = self.ban_until
        if self.breaks == 1:
            self.loop_start = find_loop_start(self.generated, period)
            self.loop_period = period
            self.loop_resume = self.ban_until
        scores[:, self.banned] = float("-inf")
        return scores

    def _force_eos(self, scores):
        scores[:] = float("-inf")
        scores[:, self.eos_ids] = 0.0
        return scores


def clean_lyrics(text: str) -> str:
    """Strip the transcriber's '# Languages ... # Lyrics' header, keep the sectioned lyrics."""
    if "# Lyrics" in text:
        text = text.split("# Lyrics", 1)[1]
    elif text.lstrip().startswith("# Languages"):
        text = ""
    text = text.strip()
    return text if text else "[Instrumental]"


class AceStepCaptionConfig(CaptionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fixed_caption: Optional[str] = kwargs.get("fixed_caption", None)
        # transcribe a MelBandRoformer vocal stem instead of the full mix; the
        # captioner still hears the full mix
        self.extract_vocals_before_transcribe: bool = bool(
            kwargs.get("extract_vocals_before_transcribe", False)
        )
        self.caption_format: str = kwargs.get("caption_format", "ace_step") or "ace_step"
        if self.caption_format not in CAPTION_FORMATS:
            raise ValueError(
                f"caption_format must be one of {CAPTION_FORMATS}, got {self.caption_format!r}"
            )


class AceStepCaptioner(BaseCaptioner):
    caption_config_class = AceStepCaptionConfig
    caption_config: AceStepCaptionConfig

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(AceStepCaptioner, self).__init__(process_id, job, config, **kwargs)
        self.separator = None
        # prep runs in worker threads; one separation on the GPU at a time
        self.separator_lock = threading.Lock()

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
        if self.caption_config.extract_vocals_before_transcribe:
            from toolkit.audio.melbandroformer import load_melbandroformer

            self.print_and_status_update("Loading vocal separator")
            self.separator = load_melbandroformer(device=self.device_torch)
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

    @staticmethod
    def _to_model_audio(waveform: torch.Tensor, sr: int) -> np.ndarray:
        """[C, T] at any rate -> mono float array at TARGET_SAMPLE_RATE."""
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SAMPLE_RATE)
        return waveform.squeeze(0).numpy()

    @staticmethod
    def _load_audio(file_path: str):
        waveform, sr = torchaudio.load(file_path)
        return AceStepCaptioner._to_model_audio(waveform, sr)

    @staticmethod
    def trim_silence(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
        """Cut leading and trailing silence, judged by frame RMS against the loudest frame."""
        frame = max(1, int(sample_rate * SILENCE_FRAME_SECONDS))
        n_frames = len(audio) // frame
        if n_frames < 2:
            return audio
        rms = np.sqrt(np.mean(audio[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
        loud = np.flatnonzero(rms >= rms.max() * SILENCE_RMS_FRACTION)
        if len(loud) == 0:
            return audio
        margin = int(sample_rate * SILENCE_MARGIN_SECONDS)
        start = max(0, loud[0] * frame - margin)
        end = min(len(audio), (loud[-1] + 1) * frame + margin)
        return audio[start:end]

    def _extract_vocals(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """Vocal stem of [C, T] audio as mono TARGET_SAMPLE_RATE, silence-trimmed."""
        from toolkit.audio.melbandroformer import separate

        with self.separator_lock:
            vocals, _ = separate(self.separator, waveform, sr)
        return self.trim_silence(self._to_model_audio(vocals.cpu(), sr))

    def _prep_file(self, file_path: str) -> dict:
        """CPU side of one file (runs in a worker thread): librosa analysis,
        decode + resample, and the mel/tokenize pass for each model. Vocal
        separation is the one GPU step here, serialized by separator_lock."""
        waveform, sr = torchaudio.load(file_path)
        audio_data = self._to_model_audio(waveform, sr)
        lyrics_audio = (
            self._extract_vocals(waveform, sr) if self.separator is not None else audio_data
        )
        item = {
            "file": file_path,
            # BPM/key/time signature only appear in the ace_step layout
            "analysis": analyze_audio(file_path)
            if self.caption_config.caption_format == "ace_step"
            else None,
            "lyrics_inputs": self._make_inputs(self.processor, lyrics_audio, LYRICS_PROMPT),
        }
        if self.model2 is not None:
            item["caption_inputs"] = self._make_inputs(
                self.processor2, audio_data, CAPTION_PROMPT
            )
        return item

    @staticmethod
    def _place_inputs(model, inputs):
        """Move a processor batch onto the model; the thinker also reads its
        2D pad mask from the model. Called again before every decode pass."""
        inputs = inputs.to(model.device).to(model.dtype)
        model._pad_mask_2d = inputs.get("attention_mask", None)
        return inputs

    def _generate(
        self,
        model,
        processor,
        inputs,
        repetition_penalty: float,
        retries: tuple = (),
        break_loops: bool = False,
        file_path: str = "",
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        inputs = self._place_inputs(model, inputs)
        # a generate that dies between static-cache creation and its first
        # forward leaves a half-built cache that breaks every later call
        stale_cache = getattr(model, "_cache", None)
        if stale_cache is not None and not stale_cache.is_initialized:
            del model._cache
        input_len = inputs["input_ids"].shape[1]
        # greedy decode loops on repeated phrases without a penalty
        gen_kwargs = {"max_new_tokens": max_new_tokens, "repetition_penalty": repetition_penalty}
        if model.generation_config.cache_implementation == "static":
            if input_len + 16 < STATIC_MAX_LENGTH:
                from transformers.generation import MaxLengthCriteria, StoppingCriteriaList

                gen_kwargs = {
                    "repetition_penalty": repetition_penalty,
                    "max_length": STATIC_MAX_LENGTH,
                    "stopping_criteria": StoppingCriteriaList(
                        [MaxLengthCriteria(max_length=min(input_len + max_new_tokens, STATIC_MAX_LENGTH))]
                    ),
                }
            else:
                # prompt would not fit the fixed cache; eager for this file
                gen_kwargs["cache_implementation"] = "dynamic"
        eos_ids = model.generation_config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])

        def strip_eos(tokens: torch.Tensor) -> torch.Tensor:
            end = tokens.shape[0]
            while end > 0 and int(tokens[end - 1]) in eos_ids:
                end -= 1
            return tokens[:end]

        def run_pass():
            breaker = LoopBreaker(input_len, eos_ids) if break_loops else None
            kwargs = dict(gen_kwargs)
            if breaker is not None:
                kwargs["logits_processor"] = LogitsProcessorList([breaker])
            self._place_inputs(model, inputs)
            generated = model.generate(**inputs, **kwargs)
            return strip_eos(generated[0, input_len:]), breaker

        new_tokens, breaker = run_pass()
        first_tokens, first_breaker = new_tokens, breaker
        stopped = breaker is not None and breaker.stopped
        for attempt, retry in enumerate(retries, 1):
            if not stopped:
                break
            print(
                f"\n[AITK] {file_path}: transcription looped {breaker.breaks} times and was stopped, "
                f"retry {attempt}/{len(retries)} with temperature={retry['temperature']} "
                f"repetition_penalty={retry['repetition_penalty']}"
            )
            gen_kwargs.update(retry)
            new_tokens, breaker = run_pass()
            stopped = breaker.stopped
        if stopped:
            # every pass re-looped after a break; keep the greedy pass up to its first loop
            keep = first_breaker.loop_start + LOOP_KEEP_REPEATS * first_breaker.loop_period
            new_tokens = first_tokens[:keep]
            print(
                f"\n[AITK] {file_path}: transcription looped on every pass; keeping the greedy "
                f"pass truncated at its first loop (period {first_breaker.loop_period} tokens)"
            )
        elif breaker is not None and breaker.breaks:
            # drop the excess repeats and the tokens forced out under the ban
            keep = breaker.loop_start + LOOP_KEEP_REPEATS * breaker.loop_period
            new_tokens = torch.cat([new_tokens[:keep], new_tokens[breaker.loop_resume :]])
            print(
                f"\n[AITK] {file_path}: broke a transcription loop "
                f"(period {breaker.loop_period} tokens) and continued"
            )
        output = processor.batch_decode(
            new_tokens[None],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output.strip()

    def get_audio_lyrics(self, inputs, file_path: str = "") -> str:
        if (
            self.caption_config.low_vram
            and self.model2 is not None
            and self.model2.device != torch.device("cpu")
        ):
            self.model2.to("cpu")
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        return self._generate(
            self.model,
            self.processor,
            inputs,
            TRANSCRIBER_REPETITION_PENALTY,
            retries=TRANSCRIBER_RETRIES,
            break_loops=True,
            file_path=file_path,
        )

    def get_audio_caption(self, inputs, file_path: str = "") -> str:
        if self.caption_config.low_vram and self.model.device != torch.device("cpu"):
            self.model.to("cpu")
        if self.model2.device == torch.device("cpu"):
            self.model2.to(self.device_torch)
        return self._generate(self.model2, self.processor2, inputs, CAPTIONER_REPETITION_PENALTY)

    def _transcribe_item(self, item: dict) -> str:
        """Raw transcriber text for a prepped item: '# Languages ... # Lyrics ...'."""
        return self.get_audio_lyrics(item["lyrics_inputs"], item["file"])

    def _caption_item(self, item: dict) -> str:
        if self.caption_config.caption_format == "yue2":
            lyrics = clean_lyrics(self._transcribe_item(item))
            if self.caption_config.fixed_caption is not None:
                caption = self.caption_config.fixed_caption
            else:
                caption = self.get_audio_caption(item["caption_inputs"], item["file"])
            caption = " ".join(caption.split())
            return f"{caption}\n[Lyrics]\n{lyrics}"

        analysis = item["analysis"]
        lyrics = self._transcribe_item(item)

        language = "en"
        if "# Languages" in lyrics and "# Lyrics" in lyrics:
            language = lyrics.split("# Languages")[1].split("# Lyrics")[0]
            # remove newlines and extra spaces from language
            language = language.replace("\n", "").strip()
            lyrics = lyrics.split("# Lyrics")[1].strip()

        if self.caption_config.fixed_caption is not None:
            caption = self.caption_config.fixed_caption
        else:
            caption = self.get_audio_caption(item["caption_inputs"], item["file"])

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
