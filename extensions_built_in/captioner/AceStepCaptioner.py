import librosa
import numpy as np
import torch
import torchaudio
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from collections import OrderedDict

from optimum.quanto import freeze
from toolkit.basic import flush
from toolkit.util.quantize import quantize, get_qtype

from .BaseCaptioner import BaseCaptioner
import transformers
import logging
import warnings

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

TARGET_SAMPLE_RATE = 16000
CAPTIONER_ID = "ACE-Step/acestep-captioner"
TRANSCRIBER_ID = "ACE-Step/acestep-transcriber"

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


class AceStepCaptioner(BaseCaptioner):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(AceStepCaptioner, self).__init__(process_id, job, config, **kwargs)

    def load_model(self):
        self.print_and_status_update("Loading transcriber model")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.caption_config.model_name_or_path,
            dtype=self.torch_dtype,
            device_map="cpu",
        )
        self.model.to(self.device_torch)
        self.model.disable_talker()
        if self.caption_config.quantize:
            self.print_and_status_update("Quantizing transcriber model")
            quantize(self.model, weights=get_qtype(self.caption_config.qtype))
            freeze(self.model)
            flush()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.caption_config.model_name_or_path
        )
        if self.caption_config.low_vram:
            self.model.to("cpu")

        # load captioner model
        self.print_and_status_update("Loading captioner model")
        self.model2 = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.caption_config.model_name_or_path2,
            dtype=self.torch_dtype,
            device_map="cpu",
        )
        self.model2.to(self.device_torch)
        self.model2.disable_talker()
        if self.caption_config.quantize:
            self.print_and_status_update("Quantizing captioner model")
            quantize(self.model2, weights=get_qtype(self.caption_config.qtype))
            freeze(self.model2)
            flush()
        self.processor2 = Qwen2_5OmniProcessor.from_pretrained(
            self.caption_config.model_name_or_path2,
        )

        if self.caption_config.low_vram:
            self.model2.to("cpu")
        flush()

    def run_qwen_audio(self, model, processor, audio_data, sr, prompt_text):
        """Run a Qwen2.5-Omni model on audio with a text prompt."""
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
        inputs = processor(
            text=text,
            audio=[audio_data],
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            sampling_rate=sr,
        )
        inputs = inputs.to(model.device).to(model.dtype)
        text_ids = model.generate(**inputs, return_audio=False)
        output = processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result = output[0]
        marker = "assistant\n"
        if marker in result:
            result = result[result.rfind(marker) + len(marker) :]
        return result.strip()

    def get_audio_lyrics(self, audio_data: torch.Tensor) -> str:
        if self.caption_config.low_vram and self.model2.device != torch.device("cpu"):
            # move captioner to cpu
            self.model2.to("cpu")
        # move lyric model if needed
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        prompt_text = "*Task* Transcribe this audio in detail"
        return self.run_qwen_audio(
            self.model, self.processor, audio_data, TARGET_SAMPLE_RATE, prompt_text
        )

    def get_audio_caption(self, audio_data: torch.Tensor) -> str:
        if self.caption_config.low_vram and self.model.device != torch.device("cpu"):
            # move lyricmodel to cpu
            self.model.to("cpu")
        # move captioner model if needed
        if self.model2.device == torch.device("cpu"):
            self.model2.to(self.device_torch)
        prompt_text = "*Task* Describe this music in detail. Include genre, mood, instrumentation, tempo feel, and vocal style if present."
        return self.run_qwen_audio(
            self.model2, self.processor2, audio_data, TARGET_SAMPLE_RATE, prompt_text
        )

    def get_caption_for_file(self, file_path: str) -> str:
        try:
            # analyze audio with librosa
            analysis = analyze_audio(file_path)

            # load audio with torchaudio for transcription
            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.to(self.device_torch)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != TARGET_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform, sr, TARGET_SAMPLE_RATE
                )
            audio_data = waveform.squeeze(0).cpu().numpy()

            # get the lyrics from the audio
            lyrics = self.get_audio_lyrics(audio_data)

            language = "en"

            if "# Languages" in lyrics and "# Lyrics" in lyrics:
                language = lyrics.split("# Languages")[1].split("# Lyrics")[0]
                # remove newlines and extra spaces from language
                language = language.replace("\n", "").strip()
                lyrics = lyrics.split("# Lyrics")[1].strip()

            # get the caption from the audio
            caption = self.get_audio_caption(audio_data)

            output = f"<CAPTION>\n{caption}\n</CAPTION>\n"
            output += f"<LYRICS>\n{lyrics}\n</LYRICS>\n"
            output += f"<BPM>{analysis['bpm']}</BPM>\n"
            output += f"<KEYSCALE>{analysis['keyscale']}</KEYSCALE>\n"
            output += f"<TIMESIGNATURE>{analysis['timesignature']}</TIMESIGNATURE>\n"
            output += f"<DURATION>{analysis['duration']}</DURATION>\n"
            output += f"<LANGUAGE>{language}</LANGUAGE>"
            return output
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
