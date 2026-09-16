"""Generic Qwen2.5-Omni captioner: same prompt-driven image / audio / video
flow as Qwen3OmniCaptioner, on the Qwen2.5-Omni thinker from
extensions_built_in/llm_models (single-file convrot8 checkpoints or an HF repo).
Unlike AceStepCaptioner (fixed transcribe + describe prompts, ACE-Step/YuE2
layouts), the caption is whatever caption_prompt asks for."""

from collections import OrderedDict

import os
import torch

from transformers import AutoConfig, AutoProcessor, Qwen2_5OmniForConditionalGeneration

from toolkit.basic import flush

from .BaseCaptioner import BaseCaptioner
import logging
import traceback
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from extensions_built_in.llm_models.src.thinker import (  # noqa: E402  shared fast paths
    load_thinker_single_file,
    prepare_thinker,
)

# frame sampling rate for video captioning
VIDEO_FPS = 2

# still-image files caption through the image pipeline (no audio, no frames)
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
# audio-only files caption through the audio pipeline (no frames)
AUDIO_EXTENSIONS = {"mp3", "wav", "flac", "ogg", "m4a", "aac"}

TARGET_SAMPLE_RATE = 16000

# fixed generation ceiling under compiled decode: a constant max_length keeps
# the static kv cache (and so the compiled decode graph) at one shape for every
# file; the real per-caption budget is enforced by a stopping criterion
STATIC_MAX_LENGTH = 8704

# thinker hidden size -> config/processor repo; single-file checkpoints carry no config
BASE_REPO_BY_HIDDEN = {3584: "Qwen/Qwen2.5-Omni-7B", 2048: "Qwen/Qwen2.5-Omni-3B"}
DEFAULT_BASE_REPO = "Qwen/Qwen2.5-Omni-7B"


class Qwen25OmniCaptioner(BaseCaptioner):
    """Captions images, audio and video with the Qwen2.5-Omni thinker."""

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(Qwen25OmniCaptioner, self).__init__(process_id, job, config, **kwargs)

    def _config_source(self, ckpt: str) -> str:
        """Single-file checkpoints carry no config; pick the base repo (config +
        processor) from the thinker hidden size."""
        from safetensors import safe_open

        with safe_open(ckpt, "pt") as f:
            hidden = f.get_slice("model.embed_tokens.weight").get_shape()[1]
        try:
            return BASE_REPO_BY_HIDDEN[hidden]
        except KeyError:
            raise RuntimeError(
                f"Unknown Qwen2.5-Omni thinker (hidden size {hidden}); "
                "use a folder/repo that ships its own config instead"
            ) from None

    def load_model(self):
        name_or_path = self.caption_config.model_name_or_path
        want_quant = self.caption_config.quantize

        if name_or_path.endswith(".safetensors"):
            from toolkit.models.v2.resolver import resolve_component_file

            # thinker checkpoints live in the comfy text_encoders/ folder
            ckpt = resolve_component_file(
                name_or_path,
                folder="text_encoders",
                component="qwen2.5-omni thinker",
                status_fn=self.print_and_status_update,
            )
            self.print_and_status_update(
                f"Loading Qwen2.5-Omni thinker from {os.path.basename(ckpt)}"
            )
            cfg_src = self._config_source(ckpt)
            config = AutoConfig.from_pretrained(cfg_src)
            model, prequantized = load_thinker_single_file(
                ckpt, config.thinker_config, self.torch_dtype
            )
            if prequantized and want_quant:
                print(
                    "[AITK] Checkpoint is pre-quantized (convrot8); the quantize "
                    "setting is ignored."
                )
                want_quant = False
        else:
            self.print_and_status_update(f"Loading Qwen2.5-Omni from {name_or_path}")
            cfg_src = name_or_path
            full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                name_or_path, dtype=self.torch_dtype, device_map="cpu"
            )
            model = prepare_thinker(full)
            del full

        model.eval()
        model.to(self.device_torch)
        if want_quant:
            from optimum.quanto import freeze

            from toolkit.util.quantize import get_qtype, quantize

            self.print_and_status_update(
                f"Quantizing thinker ({self.caption_config.qtype})"
            )
            quantize(model, weights=get_qtype(self.caption_config.qtype))
            freeze(model)
            flush()

        # built from config on the single-file path, so no sampling defaults were
        # loaded; greedy decode falls into repetition loops on long captions
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7
        model.generation_config.top_p = 0.8
        model.generation_config.top_k = 20
        model.generation_config.repetition_penalty = 1.05

        self.model = model
        self.processor = AutoProcessor.from_pretrained(cfg_src)
        if self.caption_config.low_vram:
            self.model.to("cpu")
        flush()

    def maybe_compile_models(self):
        """CUDA-graph decode: static kv cache + compiled decode (see
        Qwen3OmniCaptioner). Eager HF decode on the 7B thinker is launch-bound."""
        if not self.caption_config.compile:
            return
        if self.caption_config.low_vram:
            print("[AITK] low_vram is on; skipping compiled decode.")
            return
        import importlib.util

        if importlib.util.find_spec("triton") is None:
            print("[AITK] compile requested but triton is not installed, skipping.")
            return
        self.model.generation_config.cache_implementation = "static"
        print(
            "[AITK] Compiled decode enabled (static cache + cuda graphs). "
            "The first file compiles (~2 min cold, faster once cached)."
        )

    @staticmethod
    def _is_image_file(file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower().lstrip(".") in IMAGE_EXTENSIONS

    @staticmethod
    def _is_audio_file(file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower().lstrip(".") in AUDIO_EXTENSIONS

    def _build_messages(self, _file_path: str):
        if self._is_image_file(_file_path):
            media = {"type": "image", "image": _file_path}
        elif self._is_audio_file(_file_path):
            media = {"type": "audio", "audio": _file_path}
        else:
            media = {"type": "video", "video": _file_path}
        return [
            {
                "role": "user",
                "content": [
                    media,
                    {"type": "text", "text": self.caption_config.caption_prompt},
                ],
            }
        ]

    def _size_kwargs(self):
        max_pixels = self.caption_config.max_res * self.caption_config.max_res
        # Qwen2.5-Omni's image processor takes total pixel budgets, not edges
        return {"min_pixels": min(64 * 28 * 28, max_pixels), "max_pixels": max_pixels}

    @staticmethod
    def _load_audio(file_path: str):
        import torchaudio

        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SAMPLE_RATE)
        return waveform.squeeze(0).numpy()

    def _prep_media(self, file_path: str):
        """CPU side of one file, safe to run in a worker thread: decode +
        subsample frames (or load the image), extract the audio track, render
        the chat text. At batch size 1 the full processor (tokenize, resize,
        mel) runs here too, so the main thread only moves tensors and
        generates."""
        if self._is_image_file(file_path):
            from PIL import Image

            image = Image.open(file_path).convert("RGB")
            item = {"file": file_path, "kind": "image", "image": image, "audio": None}
        elif self._is_audio_file(file_path):
            item = {
                "file": file_path,
                "kind": "audio",
                "audio": self._load_audio(file_path),
            }
        else:
            from transformers.video_utils import load_video

            frames = load_video(file_path, fps=VIDEO_FPS)
            if isinstance(frames, tuple):
                frames = frames[0]
            audio = None
            try:
                a = self._load_audio(file_path)
                if a is not None and a.size > 0:
                    audio = a
            except Exception:
                pass
            item = {
                "file": file_path,
                "kind": "video_audio" if audio is not None else "video_silent",
                "frames": frames,
                "audio": audio,
            }
        item["text"] = self.processor.apply_chat_template(
            self._build_messages(file_path),
            tokenize=False,
            add_generation_prompt=True,
        )
        if self.caption_config.batch_size <= 1:
            item["inputs"] = self._process_items([item])
        return item

    def _process_items(self, items):
        kind = items[0]["kind"]
        if kind == "image":
            return self.processor(
                text=[it["text"] for it in items],
                images=[it["image"] for it in items],
                return_tensors="pt",
                padding=True,
                **self._size_kwargs(),
            )
        if kind == "audio":
            return self.processor(
                text=[it["text"] for it in items],
                audio=[it["audio"] for it in items],
                return_tensors="pt",
                padding=True,
                sampling_rate=TARGET_SAMPLE_RATE,
            )
        use_audio = kind == "video_audio"
        return self.processor(
            text=[it["text"] for it in items],
            audio=[it["audio"] for it in items] if use_audio else None,
            videos=[it["frames"] for it in items],
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            fps=VIDEO_FPS,
            **self._size_kwargs(),
        )

    def _gen_kwargs(self, input_len: int) -> dict:
        max_new = self.caption_config.max_new_tokens
        gen_kwargs = {"max_new_tokens": max_new}
        if self.model.generation_config.cache_implementation == "static":
            if input_len + 16 < STATIC_MAX_LENGTH:
                from transformers.generation import (
                    MaxLengthCriteria,
                    StoppingCriteriaList,
                )

                gen_kwargs = {
                    "max_length": STATIC_MAX_LENGTH,
                    "stopping_criteria": StoppingCriteriaList(
                        [
                            MaxLengthCriteria(
                                max_length=min(input_len + max_new, STATIC_MAX_LENGTH)
                            )
                        ]
                    ),
                }
            else:
                # prompt would not fit the fixed cache; eager for this file
                gen_kwargs["cache_implementation"] = "dynamic"
        return gen_kwargs

    def _caption_batch(self, items):
        """Batched generate over preprocessed items (all the same kind: image,
        audio, video with audio, or silent video). Returns captions in item order."""
        use_audio = items[0]["kind"] == "video_audio"
        if len(items) == 1 and "inputs" in items[0]:
            inputs = items[0]["inputs"]
        else:
            inputs = self._process_items(items)
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        inputs = inputs.to(self.device_torch).to(self.torch_dtype)
        # a generate that dies between static-cache creation and its first
        # forward leaves model._cache with uninitialized layers; transformers
        # then raises AttributeError reading cache.max_batch_size on every
        # later call, masking the original error — drop the stale cache
        stale_cache = getattr(self.model, "_cache", None)
        if stale_cache is not None and not stale_cache.is_initialized:
            del self.model._cache
        # under static cache, generate hands the forward a prepared 4D mask;
        # the true 2D padding mask is needed for the prefill rope index
        self.model._pad_mask_2d = inputs.get("attention_mask", None)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = self.model.generate(
            **inputs,
            use_audio_in_video=use_audio,
            **self._gen_kwargs(input_len),
        )
        captions = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [c.strip() for c in captions]

    def run_caption_loop(self):
        """Batched pipeline: CPU worker threads decode/preprocess files ahead of
        the GPU, files are grouped by kind into batches, and each batch runs one
        model.generate call so decode work is wide enough to saturate the GPU."""
        import concurrent.futures
        from collections import deque

        import tqdm as tqdm_mod

        batch_size = max(1, int(self.caption_config.batch_size))
        # smoothing near 1 weights recent files heavily, so the rate estimate
        # recovers quickly after the slow compile-warmup files
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

        def flush_bucket(bucket):
            if len(bucket) == 0:
                return
            items = list(bucket)
            bucket.clear()
            n_real = len(items)
            # keep the batch shape constant for the compiled decode graph:
            # pad a final partial bucket by repeating the last item
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

            # batches must be homogeneous: the processor call differs per kind
            buckets = {"image": [], "audio": [], "video_audio": [], "video_silent": []}
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
                bucket = buckets[item["kind"]]
                bucket.append(item)
                if len(bucket) >= batch_size:
                    flush_bucket(bucket)
            for bucket in buckets.values():
                flush_bucket(bucket)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            pbar.close()

    def get_caption_for_file(self, file_path: str) -> str:
        # single-file path (and the per-file fallback when a batch fails):
        # same prep + generate flow as the batched loop, for one item
        try:
            return self._caption_batch([self._prep_media(file_path)])[0]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
            return None
