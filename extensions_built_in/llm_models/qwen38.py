"""Qwen3.8-27B as a trainable text-generation model (arch ``qwen38``).

Image or video in (plus a .txt caption), text out. The frozen vision tower encodes
the media into embeddings [N, H] plus the processor's expanded placeholder span and
the MRoPE grid metadata (cached as the "latent" when latent caching is on). Training
runs the text stack on ``prefix + media span + instruction + assistant header +
caption`` with next-token cross-entropy on the caption (LoRA on the text stack;
towers, embeddings and lm_head stay frozen). ``is_llm`` routes the trainer to
``train_llm_accumulation`` -> ``get_llm_loss``: no noise, scheduler, VAE or prompt
encoding.

Weights: the ComfyUI single file (W4A8 codebook language model, int8 convrot
embeddings / lm_head / MTP / vision) is loaded with its quantization kept as-is
through the toolkit's convrotw4a8 / convrot8 backends, or an HF layout (bf16 shards)
is loaded and quantized to the requested qtype. See src/qwen38_model.py for the
model itself (static caches, CUDA-graph decode, MTP speculative decoding).

Samples: ``ctrl_img`` is the media file, the sample prompt is the instruction; the
generated text is written as a .txt.

model_kwargs:
  instruction            user-turn text used for every training item (default below)
  system_prompt          optional system message
  enable_thinking        keep the model's thinking mode on (default false: direct answers)
  max_pixels/min_pixels  image and video frame budget for the vision tower
  video_fps              fps the dataset extracted video frames at (dataset fps)
  max_new_tokens         sample generation budget (default 512)
  mtp_depth              MTP speculative draft depth for samples (default 3, 0 = off)
  cuda_graph             CUDA-graph the decode step (default true)
  vision_bf16            run the vision tower with bf16 activations on its dequantized int8
                         weights (default true; int8 activations drift the image embeddings ~6%)
  loss_chunk             positions per lm_head chunk in the CE (default 512)
"""

import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoProcessor

from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dto import DTO
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.resolver import resolve_component_file
from toolkit.print import print_acc

from .src.qwen38_model import Qwen38Generator, build_sequence_embeds

BASE_REPO = "Qwen/Qwen3.8-27B"
DEFAULT_CHECKPOINT = "Comfy-Org/Qwen3.8-27B/text_encoders/qwen3.8_27b_w4a8.safetensors"
DEFAULT_INSTRUCTION = "Describe this in detail."

KIND_IMAGE, KIND_VIDEO = 1, 2
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".m4v", ".flv")
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a")


def _media_kind(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext in AUDIO_EXTS:
        raise ValueError(f"qwen38 takes images and videos, not audio: {path}")
    return KIND_VIDEO if ext in VIDEO_EXTS else KIND_IMAGE


class Qwen38LLM(BaseModel):
    arch = "qwen38"
    is_llm = True
    # the data loader accepts image and video files in one dataset
    is_multimodal_llm = True
    sample_rate = 16000

    def __init__(self, device, model_config: ModelConfig, dtype="bf16", custom_pipeline=None, noise_scheduler=None, **kwargs):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_transformer = True
        # LoRA goes on the text stack only (vision tower, embeddings and lm_head stay frozen)
        self.target_lora_modules = ["Qwen38TextModel"]
        kw = self.model_config.model_kwargs
        self.instruction = str(kw.get("instruction", DEFAULT_INSTRUCTION))
        self.system_prompt: Optional[str] = kw.get("system_prompt", None)
        self.enable_thinking = bool(kw.get("enable_thinking", False))
        self.max_pixels = int(kw.get("max_pixels", 1024 * 1024))
        self.min_pixels = int(kw.get("min_pixels", 64 * 32 * 32))
        self.video_fps = float(kw.get("video_fps", 24))
        self.max_new_tokens = int(kw.get("max_new_tokens", 512))
        self.mtp_depth = int(kw.get("mtp_depth", 3))
        self.cuda_graph = bool(kw.get("cuda_graph", True))
        self.vision_bf16 = bool(kw.get("vision_bf16", True))
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
    def _config_source(self) -> str:
        extras = self.model_config.extras_name_or_path
        if extras and extras != self.model_config.name_or_path and not extras.endswith(".safetensors"):
            return extras
        return BASE_REPO

    def load_model(self):
        dtype = self.torch_dtype
        device = self.device_torch
        name_or_path = self.model_config.name_or_path or DEFAULT_CHECKPOINT
        cfg_src = self._config_source()
        want_quant = self.model_config.quantize
        qtype = self.model_config.qtype
        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:
            raise NotImplementedError("Layer offloading is not implemented for qwen38")

        from .src.qwen38_load import dequantize_vision_tower, load_qwen38

        if name_or_path.endswith(".safetensors"):
            ckpt = resolve_component_file(name_or_path, folder="text_encoders", component="qwen3.8")
            self.print_and_status_update(f"Loading Qwen3.8 from {os.path.basename(ckpt)}")
            source = ckpt
        else:
            self.print_and_status_update(f"Loading Qwen3.8 from {name_or_path}")
            source = name_or_path
            if os.path.isdir(name_or_path) or "/" in name_or_path:
                cfg_src = name_or_path if self.model_config.extras_name_or_path in (None, "", name_or_path) else cfg_src
        config = AutoConfig.from_pretrained(cfg_src)
        model, prequantized = load_qwen38(source, config, dtype, device, status_fn=self.print_and_status_update)
        if prequantized:
            model.aitk_is_quantized = True
            if want_quant and qtype not in ("convrotw4a8", "convrot8"):
                print_acc(f"Checkpoint is pre-quantized (W4A8 / int8 convrot); keeping it instead of {qtype}")
            want_quant = False
        if self.vision_bf16 and prequantized:
            n = dequantize_vision_tower(model, dtype)
            print_acc(f"Vision tower: {n} int8 linears dequantized to {dtype} (model_kwargs.vision_bf16)")
        model.eval()
        model.requires_grad_(False)
        if want_quant:
            from toolkit.util.quantize import get_qtype, quantize

            self.print_and_status_update(f"Quantizing Qwen3.8 ({qtype})")
            quantize(model, weights=get_qtype(qtype))
            model.aitk_is_quantized = True
            flush()
        self.model = model
        self.model.generator = Qwen38Generator(model, use_cuda_graph=self.cuda_graph)
        self.processor = AutoProcessor.from_pretrained(cfg_src)
        self.tokenizer = self.processor.tokenizer
        self.text_encoder = None
        self.vae = None
        self.pipeline = self
        self._build_template()
        flush()

    def _template_kwargs(self):
        return {"enable_thinking": self.enable_thinking}

    def _build_template(self):
        """Split the chat template once around the media span and the instruction so
        training sequences can be assembled from cached media spans without the processor."""
        mark = "<<AITK_INSTRUCTION>>"
        msgs = []
        if self.system_prompt is not None:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": [{"type": "image", "image": "x"}, {"type": "text", "text": mark}]})
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **self._template_kwargs())
        span = "<|vision_start|><|image_pad|><|vision_end|>"
        if span not in text or mark not in text:
            raise RuntimeError("Unexpected Qwen3.8 chat template; cannot locate the media span")
        before, rest = text.split(span, 1)
        mid, after = rest.split(mark, 1)
        enc = lambda t: self.tokenizer(t, add_special_tokens=False).input_ids
        self._prefix_ids = enc(before)
        self._mid_ids = enc(mid)
        self._after_ids = enc(after)
        self._probe = {
            KIND_IMAGE: text.replace(mark, "x"),
            KIND_VIDEO: text.replace(span, "<|vision_start|><|video_pad|><|vision_end|>").replace(mark, "x"),
        }
        self._tail_ids = enc(mid + "x" + after)
        cfg = self.model.config
        self._placeholder = {KIND_IMAGE: cfg.image_token_id, KIND_VIDEO: cfg.video_token_id}
        self._im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._eos_ids = [self._im_end, self.tokenizer.convert_tokens_to_ids("<|endoftext|>")]

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["model.language_model.layers"]

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def get_bucket_divisibility(self):
        return 32

    def save_model(self, output_path, meta, save_dtype):
        raise NotImplementedError("qwen38: only LoRA training is supported (no full-model save)")

    # LoRA keys are model-relative (model.language_model.layers.N....), the HF layout
    def convert_lora_weights_before_save(self, state_dict):
        return {(k[len("transformer.") :] if k.startswith("transformer.") else k): v for k, v in state_dict.items()}

    def convert_lora_weights_before_load(self, state_dict):
        return {(k if k.startswith("transformer.") else "transformer." + k): v for k, v in state_dict.items()}

    # ------------------------------------------------------------------
    # media encoding (latent cache)
    # ------------------------------------------------------------------
    def _span(self, input_ids: torch.Tensor, types: torch.Tensor):
        """The processor-expanded media span (tokens between the template prefix and
        the instruction tail), with its token-type ids."""
        ids = input_ids[0]
        n = ids.shape[0]
        p, t = len(self._prefix_ids), len(self._tail_ids)
        if ids[:p].tolist() != self._prefix_ids or ids[n - t :].tolist() != self._tail_ids:
            raise RuntimeError("Processor output does not start/end with the template prefix/tail")
        return ids[p : n - t].to(torch.int32), types[0, p : n - t].to(torch.int32)

    def _pack(self, embeds, media_ids, media_types, kind, grid):
        dev = embeds.device
        return DTO(
            embeds[None].to(self.torch_dtype),
            media_ids=media_ids[None].to(dev),
            media_types=media_types[None].to(dev),
            kind=torch.tensor([kind], dtype=torch.int32, device=dev),
            grid_thw=torch.tensor([list(grid)], dtype=torch.int32, device=dev),
        )

    @torch.no_grad()
    def encode_images(self, image_list: torch.Tensor, device=None, dtype=None):
        """[1, 3, H, W] image in [-1, 1] | [1, F, 3, H, W] frames."""
        if image_list.shape[0] != 1:
            raise ValueError("qwen38 encodes one item at a time (batch_size 1)")
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        x = image_list[0]
        if x.ndim == 2:
            raise ValueError("qwen38 does not take audio")
        if x.ndim == 3:
            return self._encode_image_item(x)
        if x.ndim == 4:
            return self._encode_video_item(x)
        raise ValueError(f"Unexpected media tensor shape {tuple(image_list.shape)}")

    def encode_audio(self, audio_data_list):
        # the data loader offers a video's own audio track here; video is vision-only
        return torch.zeros(len(audio_data_list), 1)

    @staticmethod
    def _to_uint8(img: torch.Tensor) -> np.ndarray:
        return ((img.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    def _encode_image_item(self, img: torch.Tensor):
        from PIL import Image

        pil = Image.fromarray(self._to_uint8(img))
        inputs = self.processor(text=self._probe[KIND_IMAGE], images=[pil], return_tensors="pt", min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        grid = inputs["image_grid_thw"]
        feats = self.model.get_image_features(inputs["pixel_values"].to(self.device_torch), grid.to(self.device_torch))
        ids, types = self._span(inputs["input_ids"], inputs["mm_token_type_ids"])
        return self._pack(feats, ids, types, KIND_IMAGE, grid[0].tolist())

    def _encode_video_item(self, frames: torch.Tensor):
        video = np.stack([self._to_uint8(f) for f in frames])  # [F, H, W, 3]
        inputs = self.processor(
            text=self._probe[KIND_VIDEO], videos=[video], return_tensors="pt", fps=self.video_fps,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels,
        )
        grid = inputs["video_grid_thw"]
        feats = self.model.get_image_features(inputs["pixel_values_videos"].to(self.device_torch), grid.to(self.device_torch))
        ids, types = self._span(inputs["input_ids"], inputs["mm_token_type_ids"])
        return self._pack(feats, ids, types, KIND_VIDEO, grid[0].tolist())

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _sequence(self, media_ids: torch.Tensor, media_types: torch.Tensor, caption: str):
        """ids [L], token types [L] and the index where the caption (loss span) starts."""
        instr = self.tokenizer(self.instruction, add_special_tokens=False).input_ids
        target = self.tokenizer(caption, add_special_tokens=False).input_ids + [self._im_end]
        head = self._prefix_ids + media_ids.tolist() + self._mid_ids + instr + self._after_ids
        ids = torch.tensor(head + target, dtype=torch.long, device=self.device_torch)
        types = torch.zeros_like(ids)
        p = len(self._prefix_ids)
        types[p : p + media_ids.shape[0]] = media_types.to(types.dtype)
        return ids, types, len(head)

    def _lm_loss(self, ids, types, start: int, media: torch.Tensor, kind: int, grid):
        model = self.model
        embeds = build_sequence_embeds(model, ids, media, [self._placeholder[kind]])
        grid_t = torch.tensor([list(grid)], device=ids.device)
        position_ids, _ = model.get_rope_index(
            ids, types, image_grid_thw=grid_t if kind == KIND_IMAGE else None, video_grid_thw=grid_t if kind == KIND_VIDEO else None
        )
        hidden = model.forward_hidden(embeds, position_ids)[0, start - 1 : -1]
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
            if batch.tensor is None:
                raise ValueError("qwen38 got a batch with neither media tensors nor cached embeddings")
            lat = self.encode_images(batch.tensor)
        if lat.shape[0] != 1:
            raise ValueError("qwen38 trains with batch_size 1")
        captions = batch.get_caption_list()
        media = lat.tensor[0].to(self.device_torch, self.torch_dtype)
        kind = int(lat.get("kind")[0])
        ids, types, start = self._sequence(lat.get("media_ids")[0].long(), lat.get("media_types")[0].long(), captions[0])
        ce = self._lm_loss(ids, types, start, media, kind, lat.get("grid_thw")[0].tolist())
        if self.debug:
            print_acc(
                f"qwen38: seq {ids.shape[0]} target {ids.shape[0] - start} ce {ce.item():.4f} "
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
        if kind == KIND_IMAGE:
            from PIL import Image

            return kind, {"type": "image", "image": "x"}, {"images": [Image.open(path).convert("RGB")]}
        from transformers.video_utils import load_video

        frames = load_video(path, fps=self.video_fps)
        if isinstance(frames, tuple):
            frames = frames[0]
        return kind, {"type": "video", "video": "x"}, {"videos": [frames], "fps": self.video_fps}

    def prepare_caption_inputs(self, media_path: Optional[str], instruction: Optional[str] = None) -> dict:
        """CPU half of caption(): media decode + processor. Safe in a worker thread."""
        instruction = instruction.strip() if instruction and instruction.strip() else self.instruction
        content = []
        media_kwargs = {}
        kind = None
        if media_path:
            kind, entry, media_kwargs = self._load_sample_media(media_path)
            content.append(entry)
        content.append({"type": "text", "text": instruction})
        msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt is not None else []) + [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **self._template_kwargs())
        inputs = self.processor(text=text, return_tensors="pt", min_pixels=self.min_pixels, max_pixels=self.max_pixels, **media_kwargs)
        return {"kind": kind, "inputs": inputs}

    @torch.no_grad()
    def generate_from_inputs(self, prepared: dict, max_new_tokens: Optional[int] = None, mtp_depth: Optional[int] = None) -> List[int]:
        """GPU half of caption(): vision tower, prefill and decode. Returns token ids."""
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        kind, inputs = prepared["kind"], prepared["inputs"]
        ids = inputs["input_ids"][0].to(self.device_torch)
        types = inputs["mm_token_type_ids"][0].to(self.device_torch)
        model = self.model
        media = None
        image_grid = video_grid = None
        if kind == KIND_IMAGE:
            image_grid = inputs["image_grid_thw"].to(self.device_torch)
            media = model.get_image_features(inputs["pixel_values"].to(self.device_torch), image_grid)
        elif kind == KIND_VIDEO:
            video_grid = inputs["video_grid_thw"].to(self.device_torch)
            media = model.get_image_features(inputs["pixel_values_videos"].to(self.device_torch), video_grid)
        embeds = build_sequence_embeds(model, ids, media, [self._placeholder[KIND_IMAGE], self._placeholder[KIND_VIDEO]])
        position_ids, delta = model.get_rope_index(ids, types, image_grid_thw=image_grid, video_grid_thw=video_grid)
        t0 = time.perf_counter()
        out_ids = model.generator.generate(
            embeds, ids[None], position_ids, delta,
            max_new_tokens=self.max_new_tokens if max_new_tokens is None else max_new_tokens,
            eos_ids=self._eos_ids, mtp_depth=self.mtp_depth if mtp_depth is None else mtp_depth,
        )
        self.last_generation_stats = dict(model.generator.last_stats, seconds=time.perf_counter() - t0, prompt_tokens=int(ids.shape[0]))
        return out_ids

    def decode_text(self, out_ids: List[int]) -> str:
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        # thinking mode: keep only the answer after the think block
        if "</think>" in text:
            text = text.split("</think>")[-1]
        return text.strip()

    @torch.no_grad()
    def caption(self, media_path: Optional[str], instruction: Optional[str] = None, max_new_tokens: Optional[int] = None, mtp_depth: Optional[int] = None) -> str:
        """Generate text for one media file (or none). Also the sample path."""
        return self.decode_text(self.generate_from_inputs(self.prepare_caption_inputs(media_path, instruction), max_new_tokens, mtp_depth))

    def generate_single_image(self, pipeline, gen_config: GenerateImageConfig, conditional_embeds, unconditional_embeds, generator, extra):
        # a str return is saved as <sample>.txt by GenerateImageConfig.save_image
        return self.caption(gen_config.ctrl_img, gen_config.prompt)
