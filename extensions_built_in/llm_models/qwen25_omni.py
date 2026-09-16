"""Qwen2.5-Omni thinker as a trainable text-generation model (arch ``qwen25_omni``).

Input media (audio, image or video file + a .txt caption) is encoded by the
frozen audio / vision towers into the tower embeddings [N, H] plus the
processor's expanded placeholder token span and the MRoPE metadata. This runs
on the GPU per step (a few hundred ms per file); latent caching is optional and
stores the same tensors (~54 MB per 300 s of audio). Training runs the text stack on
``prefix + media span + instruction + assistant header + caption`` and applies
next-token cross-entropy on the caption (LoRA on the text stack; towers and
lm_head stay frozen). ``is_llm`` routes the trainer to ``train_llm_accumulation``,
which calls ``get_llm_loss`` directly: no noise, scheduler, VAE or prompt encoding.

Samples: ``ctrl_img`` is the media file (any of the three kinds), the sample
prompt is the instruction; the generated text is written as a .txt.

model_kwargs:
  instruction            user-turn text used for every training item (default below)
  system_prompt          override the chat template's default system message
  max_pixels/min_pixels  image and video frame budget for the vision tower
  video_fps              fps the dataset extracted video frames at (dataset fps)
  max_new_tokens         sample generation budget (default 512)
  sample_compile         static-cache compiled decode for samples (default false)
  loss_chunk             positions per lm_head chunk in the CE (default 512)
"""

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoProcessor, Qwen2_5OmniForConditionalGeneration

from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dto import DTO
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.resolver import resolve_component_file
from toolkit.print import print_acc

from .src.thinker import attach_fast_paths, load_thinker_single_file, prepare_thinker

BASE_REPO = "Qwen/Qwen2.5-Omni-7B"
# thinker hidden size -> config/processor repo; single-file checkpoints carry no config
BASE_REPO_BY_HIDDEN = {3584: "Qwen/Qwen2.5-Omni-7B", 2048: "Qwen/Qwen2.5-Omni-3B"}
# single-file convrot8 thinker written by scripts/convert_vllm_to_comfy.py
DEFAULT_CHECKPOINT = "ai-toolkit/Qwen2.5-Omni-7B/qwen2_5_omni_7b_convrot8.safetensors"
DEFAULT_INSTRUCTION = "Describe this in detail."
SAMPLE_RATE = 16000

KIND_AUDIO, KIND_IMAGE, KIND_VIDEO = 0, 1, 2
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".m4v", ".flv")


def _media_kind(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext in AUDIO_EXTS:
        return KIND_AUDIO
    if ext in VIDEO_EXTS:
        return KIND_VIDEO
    return KIND_IMAGE


class Qwen25OmniLLM(BaseModel):
    arch = "qwen25_omni"
    is_llm = True
    # the data loader accepts audio, image and video files in one dataset
    is_multimodal_llm = True
    sample_rate = SAMPLE_RATE

    def __init__(self, device, model_config: ModelConfig, dtype="bf16", custom_pipeline=None, noise_scheduler=None, **kwargs):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_transformer = True
        # LoRA goes on the text stack only (towers are frozen, lm_head stays base)
        self.target_lora_modules = ["Qwen2_5OmniThinkerTextModel"]
        kw = self.model_config.model_kwargs
        self.instruction = str(kw.get("instruction", DEFAULT_INSTRUCTION))
        self.system_prompt: Optional[str] = kw.get("system_prompt", None)
        self.max_pixels = int(kw.get("max_pixels", 512 * 512))
        self.min_pixels = int(kw.get("min_pixels", 64 * 28 * 28))
        self.video_fps = float(kw.get("video_fps", 24))
        self.max_new_tokens = int(kw.get("max_new_tokens", 512))
        self.sample_compile = bool(kw.get("sample_compile", False))
        self.loss_chunk = int(kw.get("loss_chunk", 512))
        self.debug = bool(kw.get("debug", False))
        self.processor = None
        self.additional_loss_logs = {}

    @staticmethod
    def get_train_scheduler():
        # no diffusion: the trainer never samples timesteps for this model
        return None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _config_source(self, ckpt: Optional[str] = None) -> str:
        extras = self.model_config.extras_name_or_path
        if extras and extras != self.model_config.name_or_path and not extras.endswith(".safetensors"):
            return extras
        if ckpt is None:
            return BASE_REPO
        from safetensors import safe_open

        with safe_open(ckpt, "pt") as f:
            hidden = f.get_slice("model.embed_tokens.weight").get_shape()[1]
        try:
            return BASE_REPO_BY_HIDDEN[hidden]
        except KeyError:
            raise RuntimeError(
                f"Unknown Qwen2.5-Omni thinker (hidden size {hidden}); set extras_name_or_path to its HF repo"
            ) from None

    def load_model(self):
        dtype = self.torch_dtype
        device = self.device_torch
        name_or_path = self.model_config.name_or_path or DEFAULT_CHECKPOINT
        cfg_src = self._config_source()
        want_quant = self.model_config.quantize
        qtype = self.model_config.qtype
        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:
            raise NotImplementedError("Layer offloading is not implemented for qwen25_omni")

        if name_or_path.endswith(".safetensors"):
            # thinker checkpoints live in the comfy text_encoders/ folder
            ckpt = resolve_component_file(
                name_or_path, folder="text_encoders", component="qwen2.5-omni thinker"
            )
            self.print_and_status_update(f"Loading Qwen2.5-Omni thinker from {os.path.basename(ckpt)}")
            cfg_src = self._config_source(ckpt)
            config = AutoConfig.from_pretrained(cfg_src)
            thinker, prequantized = load_thinker_single_file(ckpt, config.thinker_config, dtype)
            if prequantized:
                thinker.aitk_is_quantized = True
                if not want_quant or qtype != "convrot8":
                    print_acc("Checkpoint is pre-quantized (convrot8); keeping it")
                want_quant = False
        else:
            self.print_and_status_update(f"Loading Qwen2.5-Omni from {name_or_path}")
            full = Qwen2_5OmniForConditionalGeneration.from_pretrained(name_or_path, dtype=dtype, device_map="cpu")
            thinker = prepare_thinker(full)
            del full
        thinker.eval()
        thinker.requires_grad_(False)
        thinker.to(device)
        if want_quant:
            from optimum.quanto import freeze

            from toolkit.util.quantize import get_qtype, quantize

            self.print_and_status_update(f"Quantizing thinker ({qtype})")
            quantize(thinker, weights=get_qtype(qtype))
            freeze(thinker)
            thinker.aitk_is_quantized = True
            flush()
        self.model = thinker
        self.processor = AutoProcessor.from_pretrained(cfg_src)
        self.tokenizer = self.processor.tokenizer
        self.text_encoder = None
        self.vae = None
        self.pipeline = self
        self._build_template()
        flush()

    def _build_template(self):
        """Split the chat template once around the media span and the instruction so
        training sequences can be assembled from cached media spans without the processor."""
        mark = "<<AITK_INSTRUCTION>>"
        msgs = []
        if self.system_prompt is not None:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": [{"type": "audio", "audio": "x"}, {"type": "text", "text": mark}]})
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        span = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        if span not in text or mark not in text:
            raise RuntimeError("Unexpected Qwen2.5-Omni chat template; cannot locate the media span")
        before, rest = text.split(span, 1)
        mid, after = rest.split(mark, 1)
        enc = lambda t: self.tokenizer(t, add_special_tokens=False).input_ids
        self._prefix_ids = enc(before)
        self._mid_ids = enc(mid)
        self._after_ids = enc(after)
        # probe texts: the processor expands the single placeholder in these
        self._probe = {
            KIND_AUDIO: text.replace(mark, "x"),
            KIND_IMAGE: text.replace(span, "<|vision_bos|><|IMAGE|><|vision_eos|>").replace(mark, "x"),
            KIND_VIDEO: text.replace(span, "<|vision_bos|><|VIDEO|><|vision_eos|>").replace(mark, "x"),
        }
        self._span_tokens = {
            KIND_AUDIO: (self.tokenizer.convert_tokens_to_ids("<|audio_bos|>"), self.tokenizer.convert_tokens_to_ids("<|audio_eos|>")),
            KIND_IMAGE: (self.tokenizer.convert_tokens_to_ids("<|vision_bos|>"), self.tokenizer.convert_tokens_to_ids("<|vision_eos|>")),
            KIND_VIDEO: (self.tokenizer.convert_tokens_to_ids("<|vision_bos|>"), self.tokenizer.convert_tokens_to_ids("<|vision_eos|>")),
        }
        self._placeholder = {
            KIND_AUDIO: self.model.config.audio_token_index,
            KIND_IMAGE: self.model.config.image_token_index,
            KIND_VIDEO: self.model.config.video_token_index,
        }
        self._im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["model.layers"]

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def get_bucket_divisibility(self):
        return 16

    def save_model(self, output_path, meta, save_dtype):
        raise NotImplementedError("qwen25_omni: only LoRA training is supported (no full-model save)")

    # LoRA keys are thinker-relative (model.layers.N....), the layout the
    # ACE-Step captioner and scripts/extract_lora.py use
    def convert_lora_weights_before_save(self, state_dict):
        return {(k[len("transformer.") :] if k.startswith("transformer.") else k): v for k, v in state_dict.items()}

    def convert_lora_weights_before_load(self, state_dict):
        return {(k if k.startswith("transformer.") else "transformer." + k): v for k, v in state_dict.items()}

    # ------------------------------------------------------------------
    # media encoding (latent cache)
    # ------------------------------------------------------------------
    def _span(self, input_ids: torch.Tensor, kind: int) -> torch.Tensor:
        bos, eos = self._span_tokens[kind]
        ids = input_ids[0]
        s = (ids == bos).nonzero()[0].item()
        e = (ids == eos).nonzero()[-1].item()
        return ids[s : e + 1].to(torch.int32)

    def _pack(self, embeds, media_ids, kind, audio_seqlen=0, grid=(0, 0, 0), second_per_grid=0.0):
        dev = embeds.device
        return DTO(
            embeds[None].to(self.torch_dtype),
            media_ids=media_ids[None].to(dev),
            kind=torch.tensor([kind], dtype=torch.int32, device=dev),
            audio_seqlen=torch.tensor([int(audio_seqlen)], dtype=torch.int32, device=dev),
            grid_thw=torch.tensor([list(grid)], dtype=torch.int32, device=dev),
            second_per_grid=torch.tensor([float(second_per_grid)], dtype=torch.float32, device=dev),
        )

    @torch.no_grad()
    def encode_images(self, image_list: torch.Tensor, device=None, dtype=None):
        """[1, C, S] waveform (16 kHz) | [1, 3, H, W] image in [-1, 1] | [1, F, 3, H, W] frames."""
        if image_list.shape[0] != 1:
            raise ValueError("qwen25_omni encodes one item at a time (enable latent caching, batch_size 1)")
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        x = image_list[0]
        if x.ndim == 2:
            return self._encode_audio_item(x)
        if x.ndim == 3:
            return self._encode_image_item(x)
        if x.ndim == 4:
            return self._encode_video_item(x)
        raise ValueError(f"Unexpected media tensor shape {tuple(image_list.shape)}")

    def encode_audio(self, audio_data_list):
        # the data loader offers a video's own audio track here; v1 keeps video vision-only
        return torch.zeros(len(audio_data_list), 1)

    def _log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Whisper log-mel on the GPU, same math as the processor's torch path
        (n_fft 400, hop 160, 128 mels, log10, max-8 floor, (x+4)/4). [128, frames]"""
        fe = self.processor.feature_extractor
        if getattr(self, "_mel_filters", None) is None or self._mel_filters.device != wav.device:
            self._mel_filters = torch.from_numpy(fe.mel_filters).to(wav.device, torch.float32)
            self._mel_window = torch.hann_window(fe.n_fft, device=wav.device)
        stft = torch.stft(wav, fe.n_fft, fe.hop_length, window=self._mel_window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        log_spec = torch.clamp(self._mel_filters.T @ magnitudes, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0

    def _encode_audio_item(self, wav: torch.Tensor):
        """Stays on the GPU: mel here, tower on the real frames only (the processor
        pads to 300s and the tower drops the padding anyway)."""
        fe = self.processor.feature_extractor
        wav = wav.to(self.device_torch, torch.float32)
        wav = wav.mean(0) if wav.shape[0] > 1 else wav[0]
        wav = wav[: fe.n_samples]  # the processor truncates at chunk_length seconds
        # frame count as the processor's padding mask defines it: ceil(samples / hop)
        n_frames = -(-wav.shape[0] // fe.hop_length)
        # the processor zero-pads the clip to chunk_length seconds and the STFT reflects past
        # that; pad the same way (never beyond n_samples) so the tail frames match it
        wav = F.pad(wav, (0, min(n_frames * fe.hop_length + fe.n_fft, fe.n_samples) - wav.shape[0]))
        mel = self._log_mel(wav)[:, :n_frames]
        feats = self.model.get_audio_features(
            mel[None].to(self.torch_dtype),
            feature_attention_mask=torch.ones(1, n_frames, dtype=torch.long, device=self.device_torch),
        ).last_hidden_state
        # placeholder count as the processor computes it from the frame count
        n_tokens = int(((n_frames - 1) // 2 + 1 - 2) // 2 + 1)
        bos, eos = self._span_tokens[KIND_AUDIO]
        span = torch.tensor([bos] + [self._placeholder[KIND_AUDIO]] * n_tokens + [eos], dtype=torch.int32)
        return self._pack(feats, span, KIND_AUDIO, audio_seqlen=n_frames)

    @staticmethod
    def _to_uint8(img: torch.Tensor) -> np.ndarray:
        # [-1, 1] float [3, H, W] -> uint8 HWC
        return ((img.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    def _encode_image_item(self, img: torch.Tensor):
        from PIL import Image

        pil = Image.fromarray(self._to_uint8(img))
        inputs = self.processor(text=self._probe[KIND_IMAGE], images=[pil], return_tensors="pt", min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        grid = inputs["image_grid_thw"]
        feats = self.model.get_image_features(inputs["pixel_values"].to(self.device_torch, self.torch_dtype), grid.to(self.device_torch), return_dict=True).pooler_output
        return self._pack(feats, self._span(inputs["input_ids"], KIND_IMAGE), KIND_IMAGE, grid=grid[0].tolist())

    def _encode_video_item(self, frames: torch.Tensor):
        video = np.stack([self._to_uint8(f) for f in frames])  # [F, H, W, 3]
        inputs = self.processor(
            text=self._probe[KIND_VIDEO], videos=[video], return_tensors="pt", fps=self.video_fps,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels,
        )
        grid = inputs["video_grid_thw"]
        feats = self.model.get_video_features(inputs["pixel_values_videos"].to(self.device_torch, self.torch_dtype), grid.to(self.device_torch), return_dict=True).pooler_output
        spg = inputs.get("video_second_per_grid", None)
        spg = float(spg[0]) if spg is not None else 2.0 / self.video_fps
        return self._pack(feats, self._span(inputs["input_ids"], KIND_VIDEO), KIND_VIDEO, grid=grid[0].tolist(), second_per_grid=spg)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _sequence(self, media_ids: torch.Tensor, caption: str):
        """ids [L] and the index where the caption (loss span) starts."""
        instr = self.tokenizer(self.instruction, add_special_tokens=False).input_ids
        target = self.tokenizer(caption, add_special_tokens=False).input_ids + [self._im_end]
        head = self._prefix_ids + media_ids.tolist() + self._mid_ids + instr + self._after_ids
        ids = torch.tensor(head + target, dtype=torch.long, device=self.device_torch)
        return ids, len(head)

    def _rope_kwargs(self, kind: int, audio_seqlen: int, grid, second_per_grid: float):
        dev = self.device_torch
        kw = dict(image_grid_thw=None, video_grid_thw=None, audio_seqlens=None, second_per_grids=None)
        if kind == KIND_AUDIO:
            kw["audio_seqlens"] = torch.tensor([audio_seqlen], device=dev)
        elif kind == KIND_IMAGE:
            kw["image_grid_thw"] = torch.tensor([list(grid)], device=dev)
        else:
            kw["video_grid_thw"] = torch.tensor([list(grid)], device=dev)
            kw["second_per_grids"] = torch.tensor([second_per_grid], device=dev)
        return kw

    def _lm_loss(self, ids: torch.Tensor, start: int, media: torch.Tensor, kind: int, rope_kw: dict):
        model = self.model
        embeds = model.get_input_embeddings()(ids[None])
        mask = ids == self._placeholder[kind]
        n_ph = int(mask.sum())
        if n_ph != media.shape[0]:
            raise ValueError(f"media placeholder count {n_ph} != cached embeddings {media.shape[0]}")
        embeds = embeds.masked_scatter(mask[None, :, None].expand_as(embeds), media.to(embeds.dtype))
        attention_mask = torch.ones(1, ids.shape[0], dtype=torch.long, device=ids.device)
        position_ids, _ = model.get_rope_index(
            ids[None], rope_kw["image_grid_thw"], rope_kw["video_grid_thw"], attention_mask, False,
            rope_kw["audio_seqlens"], rope_kw["second_per_grids"],
        )
        out = model.model(inputs_embeds=embeds, position_ids=position_ids, attention_mask=attention_mask, use_cache=False)
        hidden = out.last_hidden_state[0, start - 1 : -1]
        targets = ids[start:]
        lm_head = model.lm_head

        def chunk_ce(h, t):
            return F.cross_entropy(lm_head(h).float(), t, reduction="sum")

        total = hidden.new_zeros((), dtype=torch.float32)
        for s0 in range(0, targets.shape[0], self.loss_chunk):
            h, t = hidden[s0 : s0 + self.loss_chunk], targets[s0 : s0 + self.loss_chunk]
            if torch.is_grad_enabled():
                total = total + torch.utils.checkpoint.checkpoint(chunk_ce, h, t, use_reentrant=False)
            else:
                total = total + chunk_ce(h, t)
        return total / targets.shape[0]

    def get_llm_loss(self, batch) -> torch.Tensor:
        """Next-token cross-entropy on the caption, from the cached media embeddings.
        Called by the trainer's train_llm_accumulation; logs ride in additional_loss_logs."""
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        lat = batch.latents
        if not isinstance(lat, DTO) or lat.get("media_ids") is None:
            # no latent cache: run the frozen towers on the raw media now
            if batch.tensor is None:
                raise ValueError("qwen25_omni got a batch with neither media tensors nor cached embeddings")
            lat = self.encode_images(batch.tensor)
        if lat.shape[0] != 1:
            raise ValueError("qwen25_omni trains with batch_size 1")
        captions = batch.get_caption_list()
        media = lat.tensor[0].to(self.device_torch, self.torch_dtype)
        kind = int(lat.get("kind")[0])
        rope_kw = self._rope_kwargs(
            kind, int(lat.get("audio_seqlen")[0]), lat.get("grid_thw")[0].tolist(), float(lat.get("second_per_grid")[0])
        )
        ids, start = self._sequence(lat.get("media_ids")[0].long(), captions[0])
        ce = self._lm_loss(ids, start, media, kind, rope_kw)
        if self.debug:
            print_acc(
                f"qwen25_omni: seq {ids.shape[0]} target {ids.shape[0] - start} ce {ce.item():.4f} "
                f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
            )
        self.additional_loss_logs = {"loss/ce": ce.detach().float().item()}
        return ce

    # ------------------------------------------------------------------
    # sampling: ctrl_img is the media file, prompt is the instruction, output is text
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return self

    def _load_sample_media(self, path: str):
        kind = _media_kind(path)
        if kind == KIND_AUDIO:
            import torchaudio

            wav, sr = torchaudio.load(path)
            wav = wav.mean(0, keepdim=True)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            return kind, {"type": "audio", "audio": "x"}, {"audio": [wav[0].numpy()]}
        if kind == KIND_IMAGE:
            from PIL import Image

            return kind, {"type": "image", "image": "x"}, {"images": [Image.open(path).convert("RGB")]}
        from transformers.video_utils import load_video

        frames = load_video(path, fps=self.video_fps)
        if isinstance(frames, tuple):
            frames = frames[0]
        return kind, {"type": "video", "video": "x"}, {"videos": [frames], "fps": self.video_fps}

    def generate_single_image(self, pipeline, gen_config: GenerateImageConfig, conditional_embeds, unconditional_embeds, generator, extra):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        instruction = gen_config.prompt.strip() if gen_config.prompt and gen_config.prompt.strip() else self.instruction
        content = []
        media_kwargs = {}
        if gen_config.ctrl_img:
            _, entry, media_kwargs = self._load_sample_media(gen_config.ctrl_img)
            content.append(entry)
        content.append({"type": "text", "text": instruction})
        msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt is not None else []) + [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=text, return_tensors="pt", padding=True, sampling_rate=SAMPLE_RATE,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels, **media_kwargs,
        ).to(self.device_torch).to(self.torch_dtype)
        input_len = inputs["input_ids"].shape[1]
        self.model._pad_mask_2d = inputs.get("attention_mask", None)
        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        if self.sample_compile:
            from transformers.generation import MaxLengthCriteria, StoppingCriteriaList

            static_len = max(8704, ((input_len + self.max_new_tokens + 255) // 256) * 256)
            gen_kwargs = {
                "max_length": static_len, "do_sample": False, "cache_implementation": "static",
                "stopping_criteria": StoppingCriteriaList([MaxLengthCriteria(max_length=input_len + self.max_new_tokens)]),
            }
        with torch.no_grad():
            ids = self.model.generate(**inputs, **gen_kwargs)
        # a str return is saved as <sample>.txt by GenerateImageConfig.save_image
        return self.processor.batch_decode(ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
