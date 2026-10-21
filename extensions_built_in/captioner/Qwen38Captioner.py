"""Qwen3.8-27B image / video captioner on the qwen38 arch from
extensions_built_in/llm_models: ComfyUI W4A8 weights kept in their quantization,
custom DeltaNet + attention stack, CUDA-graph decode and optional MTP speculative
decoding. LoRAs trained with the qwen38 arch attach as sidechains (BaseCaptioner).
Text only in the output; audio files are not captioned (use Qwen2.5/3-Omni)."""

from collections import OrderedDict

import concurrent.futures
import os
import traceback
from collections import deque

import torch

from .BaseCaptioner import BaseCaptioner, CaptionConfig
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
# reasoning budget on top of max_new_tokens when thinking is on
MAX_THINKING_TOKENS = 4096


class Qwen38CaptionConfig(CaptionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # MTP speculative decoding (draft depth tokens per step, verified in one pass)
        self.mtp = bool(kwargs.get("mtp", True))
        self.mtp_depth = int(kwargs.get("mtp_depth", 3))
        # capture the decode step as a CUDA graph (off = eager, for debugging)
        self.cuda_graph = bool(kwargs.get("cuda_graph", True))
        # vision tower on dequantized bf16 weights (int8 activations drift image embeddings ~6%)
        self.vision_bf16 = bool(kwargs.get("vision_bf16", True))
        # frames per second sampled from videos
        self.video_fps = float(kwargs.get("video_fps", 2))
        self.system_prompt = kwargs.get("system_prompt", None) or None


class Qwen38Captioner(BaseCaptioner):
    caption_config_class = Qwen38CaptionConfig

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(Qwen38Captioner, self).__init__(process_id, job, config, **kwargs)
        self.llm = None

    def load_model(self):
        from toolkit.config_modules import ModelConfig
        from extensions_built_in.llm_models.qwen38 import Qwen38LLM

        cc = self.caption_config
        if cc.low_vram:
            print("[AITK] low_vram has no effect for Qwen3.8 (W4A8 weights are ~17 GB and stay resident).")
        model_config = ModelConfig(
            arch="qwen38",
            name_or_path=cc.model_name_or_path,
            quantize=cc.quantize,
            qtype=cc.qtype,
            model_kwargs={
                "instruction": cc.caption_prompt,
                "system_prompt": cc.system_prompt,
                "enable_thinking": cc.thinking,
                "max_pixels": cc.max_res * cc.max_res,
                "max_new_tokens": cc.max_new_tokens + (MAX_THINKING_TOKENS if cc.thinking else 0),
                "mtp_depth": cc.mtp_depth if cc.mtp else 0,
                "cuda_graph": cc.cuda_graph,
                "vision_bf16": cc.vision_bf16,
                "video_fps": cc.video_fps,
            },
        )
        llm = Qwen38LLM(self.device_torch, model_config, dtype=cc.dtype)
        llm.add_status_update_hook(lambda s: self.update_status("running", s))
        llm.load_model()
        self.llm = llm
        # LoRA sidechains and the stop/compile machinery key off self.model
        self.model = llm.model
        self.processor = llm.processor
        mode = f"MTP depth {cc.mtp_depth}" if cc.mtp else "plain decode"
        print(f"[AITK] Qwen3.8 ready: {mode}, cuda graphs {'on' if cc.cuda_graph else 'off'}")

    def maybe_compile_models(self):
        # the decode step is already a captured CUDA graph; torch.compile would only
        # re-trace the custom stack
        return

    @staticmethod
    def _is_image_file(file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower().lstrip(".") in IMAGE_EXTENSIONS

    def _prep_media(self, file_path: str) -> dict:
        """CPU side of one file (image decode / frame extraction, processor); runs in a
        worker thread ahead of the GPU."""
        item = self.llm.prepare_caption_inputs(file_path, self.resolve_prompt(self.caption_config.caption_prompt, file_path))
        item["file"] = file_path
        return item

    def _generate(self, item: dict) -> str:
        out_ids = self.llm.generate_from_inputs(item)
        return self.llm.decode_text(out_ids)

    def run_caption_loop(self):
        """Worker threads preprocess files ahead of the GPU; generation runs one file
        at a time on the main thread (the decode step is weight-bandwidth bound and
        CUDA-graphed, so there is no CPU slack to hide)."""
        import tqdm as tqdm_mod

        pbar = tqdm_mod.tqdm(total=len(self.file_paths), desc="Captioning files", unit="file", smoothing=0.9)

        def finish(file_path, caption):
            if caption is not None:
                self.save_caption_for_file(file_path, caption)
            self.step_num += 1
            self.update_step()
            pbar.update(1)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(self.caption_config.num_workers)))
        try:
            futures = deque()
            file_iter = iter(self.file_paths)
            lookahead = max(2, int(self.caption_config.num_workers)) + 1
            for _ in range(lookahead):
                path = next(file_iter, None)
                if path is None:
                    break
                futures.append((path, executor.submit(self._prep_media, path)))
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
                try:
                    caption = self._generate(item)
                    st = self.llm.last_generation_stats
                    if st.get("seconds"):
                        pbar.set_postfix_str(f"{st['tokens'] / st['seconds']:.1f} tok/s, prompt {st['prompt_tokens']}")
                except Exception as e:
                    print(f"Error captioning {path}: {e}")
                    traceback.print_exc()
                    caption = None
                finish(path, caption)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            pbar.close()

    def get_caption_for_file(self, file_path: str) -> str:
        try:
            return self._generate(self._prep_media(file_path))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
            return None
