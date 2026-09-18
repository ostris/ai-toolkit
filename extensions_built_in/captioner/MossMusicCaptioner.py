"""MOSS-Music captioner: one 8B audio-language model (OpenMOSS-Team/MOSS-Music-8B-Instruct)
does both the lyric transcription and the description that AceStepCaptioner
runs two Qwen2.5-Omni checkpoints for. Output layouts, loop breaking and the
worker pipeline are inherited from AceStepCaptioner."""

import re

from optimum.quanto import freeze

from toolkit.audio.moss_music import HF_REPO, load_moss_music
from toolkit.basic import flush
from toolkit.util.quantize import get_qtype, quantize

from .AceStepCaptioner import (
    CAPTIONER_REPETITION_PENALTY,
    TRANSCRIBER_REPETITION_PENALTY,
    TRANSCRIBER_RETRIES,
    AceStepCaptionConfig,
    AceStepCaptioner,
    analyze_audio,
)

# timestamps are requested on purpose: each line is anchored to the input time
# markers, so decode has to advance through the song instead of looping on a
# repeated phrase. They are stripped from the saved caption.
TRANSCRIBE_PROMPT = "Transcribe the lyrics of this song with timestamps."
# the model writes multi-section essays with timings, chords and lyric quotes
# unless framed as a generator prompt; this keeps it to one plain paragraph
CAPTION_PROMPT = (
    "Write a prompt that a text-to-music generator could use to recreate this track. One paragraph, "
    "under 80 words: genre, mood, instrumentation, tempo feel, production style, vocal style. "
    "No timestamps, section timings, chord names, or lyric quotes."
)
LANGUAGE_PROMPT = (
    "What language are the lyrics sung in? Answer with only the two-letter "
    "ISO 639-1 language code, or none if there are no lyrics."
)
LANGUAGE_MAX_NEW_TOKENS = 8

# the model writes "[00:18.00 - 00:22.00] line"; a looping decode can run the
# seconds field past 59 ("[00:100.00 - ...]"), so any bracketed run of time
# characters is treated as a timestamp, plus bare mm:ss spans at a line start
_TIME = r"(?:\d{1,2}:)?\d{1,3}:\d{1,3}(?:[.:]\d{1,3})?|\d+(?:\.\d+)?\s*s"
_SPAN = rf"(?:{_TIME})(?:\s*(?:-{{1,3}}>?|–|—|~|to)\s*(?:{_TIME}))?"
_TIME_CHARS = r"[\d:.\s\-–—>~s]*\d[\d:.\s\-–—>~s]*"
TIMESTAMP_RE = re.compile(
    rf"\[{_TIME_CHARS}\]|\({_TIME_CHARS}\)|<{_TIME_CHARS}>|^\s*{_SPAN}\s*[:\-–—|]?",
    re.MULTILINE,
)


def strip_timestamps(text: str) -> str:
    """Remove time annotations and tidy the line layout they leave behind."""
    text = TIMESTAMP_RE.sub("", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    out = []
    for line in lines:
        if line == "" and (not out or out[-1] == ""):
            continue
        out.append(line)
    return "\n".join(out).strip()


class MossMusicCaptionConfig(AceStepCaptionConfig):
    def __init__(self, **kwargs):
        kwargs.setdefault("model_name_or_path", HF_REPO)
        super().__init__(**kwargs)
        self.caption_prompt: str = kwargs.get("caption_prompt") or CAPTION_PROMPT
        self.transcribe_prompt: str = kwargs.get("transcribe_prompt") or TRANSCRIBE_PROMPT
        # MOSS transcribes the full mix better than a separated vocal stem
        self.extract_vocals_before_transcribe = False
        # leave the model's "[00:18.00 - 00:22.00]" line times in the saved lyrics
        self.keep_timestamps: bool = bool(kwargs.get("keep_timestamps", False))


class MossMusicCaptioner(AceStepCaptioner):
    caption_config_class = MossMusicCaptionConfig
    caption_config: MossMusicCaptionConfig

    def load_model(self):
        self.print_and_status_update("Loading MOSS-Music model")
        self.model, self.processor = load_moss_music(
            self.caption_config.model_name_or_path,
            dtype=self.torch_dtype,
            device=str(self.device_torch),
        )
        if self.caption_config.quantize:
            self.print_and_status_update("Quantizing MOSS-Music model")
            quantize(self.model, weights=get_qtype(self.caption_config.qtype))
            freeze(self.model)
            flush()
        # one model does both jobs; AceStepCaptioner's second slot stays empty
        self.model2 = None
        self.processor2 = None
        flush()

    @staticmethod
    def _place_inputs(model, inputs):
        inputs = inputs.to(model.device)
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
        return inputs

    def _make_inputs(self, audio_data, prompt_text: str):
        inputs = self.processor(text=prompt_text, audios=[audio_data])
        inputs["audio_input_mask"] = inputs["input_ids"] == self.processor.audio_token_id
        return inputs

    def _prep_file(self, file_path: str) -> dict:
        audio_data = self._load_audio(file_path)
        is_ace_step = self.caption_config.caption_format == "ace_step"
        item = {
            "file": file_path,
            "analysis": analyze_audio(file_path) if is_ace_step else None,
            "lyrics_inputs": self._make_inputs(audio_data, self.caption_config.transcribe_prompt),
        }
        if is_ace_step:
            item["language_inputs"] = self._make_inputs(audio_data, LANGUAGE_PROMPT)
        if self.caption_config.fixed_caption is None:
            item["caption_inputs"] = self._make_inputs(
                audio_data, self.caption_config.caption_prompt
            )
        return item

    def get_audio_lyrics(self, inputs, file_path: str = "") -> str:
        text = self._generate(
            self.model,
            self.processor,
            inputs,
            TRANSCRIBER_REPETITION_PENALTY,
            retries=TRANSCRIBER_RETRIES,
            break_loops=True,
            file_path=file_path,
        )
        return text.strip() if self.caption_config.keep_timestamps else strip_timestamps(text)

    def get_audio_language(self, inputs) -> str:
        answer = self._generate(
            self.model,
            self.processor,
            inputs,
            repetition_penalty=1.0,
            max_new_tokens=LANGUAGE_MAX_NEW_TOKENS,
        )
        match = re.search(r"[a-z]{2,3}", answer.lower())
        code = match.group(0) if match else ""
        # instrumentals answer none; ACE-Step's default language is en
        return code if len(code) == 2 and code != "no" else "en"

    def _transcribe_item(self, item: dict) -> str:
        lyrics = self.get_audio_lyrics(item["lyrics_inputs"], item["file"])
        if "language_inputs" not in item:
            return lyrics
        # the layout AceStepCaptioner._caption_item parses from the ACE transcriber
        language = self.get_audio_language(item["language_inputs"])
        return f"# Languages\n{language}\n# Lyrics\n{lyrics}"

    def get_audio_caption(self, inputs, file_path: str = "") -> str:
        # a tag-list prompt can degenerate into a repeated tag; same loop
        # breaking as transcription
        caption = self._generate(
            self.model,
            self.processor,
            inputs,
            CAPTIONER_REPETITION_PENALTY,
            retries=TRANSCRIBER_RETRIES,
            break_loops=True,
            file_path=file_path,
        )
        return strip_timestamps(caption)
