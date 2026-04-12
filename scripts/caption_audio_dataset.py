#!/usr/bin/env python3
"""
Caption audio files for ACE-Step v1.5 training.

Produces .txt files containing all training metadata:
  - caption (from acestep-captioner)
  - lyrics (from acestep-transcriber)
  - bpm, keyscale, timesignature (from librosa)
  - duration, language

Requirements:
    pip install torch torchaudio transformers librosa numpy

Usage:
    python caption_dir.py input_dir/
    python caption_dir.py input_dir/ --low_vram --skip_existing
"""

import argparse
import gc
import os
import glob
import logging
import warnings

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

TARGET_SAMPLE_RATE = 16000
CAPTIONER_ID = "ACE-Step/acestep-captioner"
TRANSCRIBER_ID = "ACE-Step/acestep-transcriber"

# Key profiles for Krumhansl-Schmuckler key detection
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def get_audio_files(input_dir):
    extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.WAV", "*.MP3", "*.FLAC"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    return sorted(set(files))


def load_audio_mono_16k(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SAMPLE_RATE)
    return waveform.squeeze(0).numpy(), TARGET_SAMPLE_RATE


# ═══════════════════════════════════════════════════════════════════════════════
# Audio analysis (BPM, key, time signature) via librosa
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_audio(audio_path):
    """Extract BPM, key, and time signature from audio using librosa."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = tempo[0]
    bpm = int(round(float(tempo)))

    # Key detection via chroma correlation with key profiles
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = chroma.mean(axis=1)
    major_corrs = np.array([np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_avg)[0, 1] for i in range(12)])
    minor_corrs = np.array([np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_avg)[0, 1] for i in range(12)])

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
        acf = np.correlate(beat_strengths - beat_strengths.mean(),
                           beat_strengths - beat_strengths.mean(), mode='full')
        acf = acf[len(acf) // 2:]
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


# ═══════════════════════════════════════════════════════════════════════════════
# Model management
# ═══════════════════════════════════════════════════════════════════════════════

def offload_to_cpu(model):
    """Move model to CPU and free GPU memory."""
    if model is not None:
        model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_qwen_model(model_id, device="cuda", dtype=torch.bfloat16):
    """Load a Qwen2.5-Omni model."""
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device,
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    return model, processor


def run_qwen_audio(model, processor, audio_data, sr, prompt_text):
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
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text, audio=[audio_data], images=None, videos=None,
        return_tensors="pt", padding=True, sampling_rate=sr,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids = model.generate(**inputs, return_audio=False)
    output = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = output[0]
    marker = "assistant\n"
    if marker in result:
        result = result[result.rfind(marker) + len(marker):]
    return result.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_output(caption, lyrics, analysis, language="en"):
    """Format all metadata into tagged format for easy parsing."""
    return (
        f"<CAPTION>\n{caption}\n</CAPTION>\n"
        f"<LYRICS>\n{lyrics}\n</LYRICS>\n"
        f"<BPM>{analysis['bpm']}</BPM>\n"
        f"<KEYSCALE>{analysis['keyscale']}</KEYSCALE>\n"
        f"<TIMESIGNATURE>{analysis['timesignature']}</TIMESIGNATURE>\n"
        f"<DURATION>{analysis['duration']}</DURATION>\n"
        f"<LANGUAGE>{language}</LANGUAGE>"
    )


def parse_caption_file(path):
    """Parse a tagged caption file back into a dict."""
    import re
    text = open(path, "r", encoding="utf-8").read()
    def tag(name):
        m = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
        return m.group(1).strip() if m else ""
    return {
        "caption": tag("CAPTION"),
        "lyrics": tag("LYRICS"),
        "bpm": tag("BPM"),
        "keyscale": tag("KEYSCALE"),
        "timesignature": tag("TIMESIGNATURE"),
        "duration": tag("DURATION"),
        "language": tag("LANGUAGE"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Caption audio files for ACE-Step training")
    parser.add_argument("input_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files that already have captions")
    parser.add_argument("--low_vram", action="store_true", help="Offload models to CPU between stages")
    parser.add_argument("--language", default="en", help="Default language code (default: en)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return

    audio_files = get_audio_files(args.input_dir)
    if not audio_files:
        print("No audio files found in the directory")
        return

    print(f"Found {len(audio_files)} audio files")

    # ── Stage 1: Audio analysis (BPM, key, time sig) — no GPU needed ─────
    print("\n[Stage 1/3] Analyzing audio (BPM, key, time signature)...")
    analyses = {}
    for audio_path in tqdm(audio_files, desc="Analyzing"):
        base_name = os.path.splitext(audio_path)[0]
        if args.skip_existing and os.path.exists(base_name + ".txt"):
            continue
        try:
            analyses[audio_path] = analyze_audio(audio_path)
        except Exception as e:
            print(f"\n  Error analyzing {os.path.basename(audio_path)}: {e}")
            analyses[audio_path] = {"bpm": 120, "keyscale": "C major", "timesignature": "4",
                                    "duration": 30}

    # Filter to only files that need processing
    files_to_process = [f for f in audio_files if f in analyses]
    if not files_to_process:
        print("All files already captioned (use without --skip_existing to overwrite)")
        return

    # ── Stage 2: Captioning ──────────────────────────────────────────────
    print(f"\n[Stage 2/3] Captioning {len(files_to_process)} files...")
    print("  Loading captioner model...")
    captioner, cap_processor = load_qwen_model(CAPTIONER_ID)

    captions = {}
    for audio_path in tqdm(files_to_process, desc="Captioning"):
        try:
            audio_data, sr = load_audio_mono_16k(audio_path)
            caption = run_qwen_audio(
                captioner, cap_processor, audio_data, sr,
                "*Task* Describe this music in detail. Include genre, mood, instrumentation, tempo feel, and vocal style if present."
            )
            captions[audio_path] = caption
        except Exception as e:
            print(f"\n  Error captioning {os.path.basename(audio_path)}: {e}")
            captions[audio_path] = ""

    if args.low_vram:
        print("  Offloading captioner to CPU...")
        offload_to_cpu(captioner)
        del captioner, cap_processor

    # ── Stage 3: Lyrics transcription ────────────────────────────────────
    print(f"\n[Stage 3/3] Transcribing lyrics for {len(files_to_process)} files...")
    print("  Loading transcriber model...")
    transcriber, trans_processor = load_qwen_model(TRANSCRIBER_ID)

    lyrics_map = {}
    for audio_path in tqdm(files_to_process, desc="Transcribing"):
        try:
            audio_data, sr = load_audio_mono_16k(audio_path)
            lyrics = run_qwen_audio(
                transcriber, trans_processor, audio_data, sr,
                "*Task* Transcribe this audio in detail"
            )
            lyrics_map[audio_path] = lyrics
        except Exception as e:
            print(f"\n  Error transcribing {os.path.basename(audio_path)}: {e}")
            lyrics_map[audio_path] = "[Instrumental]"

    if args.low_vram:
        print("  Offloading transcriber to CPU...")
        offload_to_cpu(transcriber)
        del transcriber, trans_processor

    # ── Write output files ───────────────────────────────────────────────
    print("\nWriting output files...")
    for audio_path in files_to_process:
        base_name = os.path.splitext(audio_path)[0]
        output_path = base_name + ".txt"

        caption = captions.get(audio_path, "")
        lyrics = lyrics_map.get(audio_path, "[Instrumental]")
        analysis = analyses[audio_path]

        output = format_output(caption, lyrics, analysis, args.language)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)

    print(f"Done! Processed {len(files_to_process)} files.")


if __name__ == "__main__":
    main()
