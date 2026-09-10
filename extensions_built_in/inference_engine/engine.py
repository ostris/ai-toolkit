"""Engine: request queue + holder lifecycle + generation with streamed frames.

Runs on the process main thread (`run_forever`); the HTTP server hands it
GenerationJobs from its own thread and reads the frames each job emits.
"""

import gc
import glob
import json
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from typing import Callable, Dict, List, Optional

import torch

from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.registry import OUTPUT_EXT, get_arch_entry
from toolkit.models.v2.pool import ComponentPool
from toolkit.util.get_model import LEGACY_ARCHS, get_model_class

from .latent_preview import preview_info
from .protocol import encode_frame, tensor_payload

END = None  # frames-queue terminator (after the "end" frame)


class GenerationCancelled(Exception):
    pass


class GenerationJob:
    def __init__(self, model: dict, sample: dict, stream: Optional[dict] = None):
        self.request_id = uuid.uuid4().hex[:12]
        self.model = model
        self.sample = sample
        self.stream = {"latents": "raw", "every_n_steps": 1, "max_frames": 4}
        self.stream.update(stream or {})
        self.frames: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self.cancel = threading.Event()
        self.status = "queued"
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.results: List[dict] = []
        self.error: Optional[str] = None
        self.step = 0
        self.total_steps: Optional[int] = None

    def emit(self, type_: str, payload: bytes = b"", **fields):
        header = {"type": type_, "request_id": self.request_id, **fields}
        self.frames.put(encode_frame(header, payload))

    def finish(self):
        self.finished_at = time.time()
        self.emit("end", status=self.status)
        self.frames.put(END)

    def info(self) -> dict:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "arch": self.model.get("arch"),
            "prompt": self.sample.get("prompt", ""),
            "step": self.step,
            "total_steps": self.total_steps,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": self.results,
            "error": self.error,
        }


def _model_key(model: dict) -> str:
    return json.dumps({k: v for k, v in sorted(model.items()) if v is not None}, sort_keys=True)


class Engine:
    def __init__(
        self,
        device: str,
        output_folder: str,
        assets_folder: Optional[str] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        dtype: str = "bf16",
    ):
        self.device = device
        self.dtype = dtype
        self.output_folder = output_folder
        self.assets_folder = assets_folder or os.path.join(os.path.dirname(output_folder), "assets")
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.assets_folder, exist_ok=True)
        self.status_cb = status_cb or (lambda s: None)
        self.queue: "queue.Queue[GenerationJob]" = queue.Queue()
        self.jobs: Dict[str, GenerationJob] = {}
        self.current: Optional[GenerationJob] = None
        self.stop_event = threading.Event()
        self.stop_reason = "stopped"
        self.holder = None
        self.holder_key: Optional[str] = None
        self.holder_model: Optional[dict] = None
        self.holder_is_legacy = False
        self.stats = {"holder_loads": 0, "generations": 0, "errors": 0, "cancelled": 0, "oom_retries": 0}
        self._lock = threading.Lock()
        # resident components shared across holders (same TE/VAE under two archs)
        self.pool = ComponentPool()

    # ------------------------------------------------------------------ api
    def submit(self, model: dict, sample: dict, stream: Optional[dict] = None) -> GenerationJob:
        if "arch" not in model:
            raise ValueError("model.arch is required")
        job = GenerationJob(model, sample, stream)
        with self._lock:
            self.jobs[job.request_id] = job
        self.queue.put(job)
        return job

    def cancel(self, request_id: str) -> bool:
        job = self.jobs.get(request_id)
        if job is None:
            return False
        job.cancel.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.emit("error", message="cancelled", cancelled=True)
            job.finish()
        return True

    def stop(self, reason: str = "stopped"):
        self.stop_reason = reason
        self.stop_event.set()
        if self.current is not None:
            self.current.cancel.set()

    def queued(self) -> List[dict]:
        return [j.info() for j in self.jobs.values() if j.status in ("queued", "running")]

    def recent(self, limit: int = 50) -> List[dict]:
        jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.info() for j in jobs[:limit]]

    def health(self) -> dict:
        vram = None
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(torch.device(self.device))
                vram = {"used": total - free, "total": total}
            except Exception:
                vram = None
        return {
            "ok": True,
            "busy": self.current is not None,
            "queue": len([j for j in self.jobs.values() if j.status == "queued"]),
            "current": self.current.info() if self.current else None,
            "active": {"model": self.holder_model, "arch": (self.holder_model or {}).get("arch")}
            if self.holder is not None
            else None,
            "device": self.device,
            "vram": vram,
            "stats": dict(self.stats),
            "pool": self.pool.stats(),
        }

    def _drop_holder(self):
        """Drop the holder object; its components stay resident in the pool
        until release_unused()/clear() decides."""
        holder = self.holder
        self.holder = None
        self.holder_key = None
        self.holder_model = None
        if holder is not None:
            for attr in ("pipeline", "model", "vae", "text_encoder", "image_encoder", "adapter", "network"):
                try:
                    setattr(holder, attr, None)
                except Exception:
                    pass
            del holder
        gc.collect()

    def unload(self):
        if self.holder is None and not self.pool.entries:
            return
        self.status_cb("Unloading model")
        self._drop_holder()
        self.pool.clear()
        flush()

    def preload(self, model: dict):
        """Load a model's holder ahead of any request (warm start). Runs on
        the caller's thread; meant for the engine main thread before
        run_forever."""
        job = GenerationJob(dict(model), {}, {"latents": "none"})
        job.status = "running"
        try:
            self._ensure_holder(job)
            job.status = "done"
        except Exception as e:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            job.finish()
        return job

    # ------------------------------------------------------------------ loop
    def run_forever(self):
        while not self.stop_event.is_set():
            try:
                job = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                self.stop("stopped")
                break
            if job.status != "queued":
                continue
            self.current = job
            try:
                self._run(job)
            except KeyboardInterrupt:
                job.status = "cancelled"
                job.emit("error", message="engine stopping", cancelled=True)
                job.finish()
                self.stop("stopped")
                break
            finally:
                self.current = None
                self._idle_status()
        # drain anything left queued
        while True:
            try:
                job = self.queue.get_nowait()
            except queue.Empty:
                break
            self.cancel(job.request_id)

    def _idle_status(self):
        if self.holder_model is not None:
            self.status_cb(f"Idle - {self.holder_model.get('arch')} ({self.holder_model.get('name_or_path')})")
        else:
            self.status_cb("Idle - no model loaded")

    # ------------------------------------------------------------------ holder
    def _build_model_config(self, model: dict) -> ModelConfig:
        kwargs = dict(model)
        arch = kwargs.pop("arch")
        entry = get_arch_entry(arch)
        for k, v in entry["model"].items():
            kwargs.setdefault(k, v)
        kwargs.setdefault("dtype", self.dtype)
        if kwargs.get("quantize"):
            kwargs.setdefault("qtype", "convrot8")
        if kwargs.get("quantize_te"):
            kwargs.setdefault("qtype_te", "convrot8")
        return ModelConfig(arch=arch, **kwargs)

    def _ensure_holder(self, job: GenerationJob):
        model = dict(job.model)
        entry = get_arch_entry(model["arch"])
        for k, v in entry["model"].items():
            model.setdefault(k, v)
        key = _model_key(model)
        if self.holder is not None and key == self.holder_key:
            return self.holder
        if self.holder is not None:
            self._drop_holder()

        self.pool.begin_request()
        ComponentPool.current = self.pool
        try:
            try:
                holder, is_legacy = self._build_holder(job, model)
            except torch.OutOfMemoryError:
                # components the previous model needed but this one does not
                # were still resident: free them and try once more
                self.stats["oom_retries"] += 1
                job.emit("status", message="Out of memory: freeing unused components and retrying")
                freed = self.pool.release_unused()
                flush()
                job.emit("status", message=f"Freed {freed / 1e9:.1f} GB, retrying load")
                holder, is_legacy = self._build_holder(job, model)
        finally:
            ComponentPool.current = None
        freed = self.pool.release_unused()
        if freed:
            job.emit("status", message=f"Released {freed / 1e9:.1f} GB of components not used by {model['arch']}")
        flush()
        self.holder = holder
        self.holder_key = key
        self.holder_model = model
        self.holder_is_legacy = is_legacy
        return holder

    def _build_holder(self, job: GenerationJob, model: dict):
        model_config = self._build_model_config(model)
        ModelClass = get_model_class(model_config)
        is_legacy = not hasattr(ModelClass, "get_train_scheduler")
        if is_legacy:
            from toolkit.sampler import get_sampler

            legacy_arch = "sd"
            if model_config.is_pixart:
                legacy_arch = "pixart"
            elif model_config.is_flux:
                legacy_arch = "flux"
            elif model_config.is_lumina2:
                legacy_arch = "lumina2"
            sampler = get_sampler(
                "ddpm",
                {"prediction_type": "v_prediction" if model_config.is_v_pred else "epsilon"},
                arch=legacy_arch,
            )
        else:
            sampler = ModelClass.get_train_scheduler()

        job.emit("status", message=f"Loading {model['arch']}")
        self.status_cb(f"Loading {model['arch']}: {model.get('name_or_path')}")
        holder = ModelClass(
            device=self.device,
            model_config=model_config,
            dtype=self.dtype,
            noise_scheduler=sampler,
        )

        def _status_hook(msg):
            job.emit("status", message=str(msg))
            self.status_cb(f"Loading {model['arch']}: {msg}")

        if hasattr(holder, "add_status_update_hook"):
            holder.add_status_update_hook(_status_hook)
        t0 = time.perf_counter()
        holder.load_model()
        self.stats["holder_loads"] += 1
        p = self.pool.stats()
        job.emit(
            "status",
            message=f"Loaded {model['arch']} in {time.perf_counter() - t0:.1f}s (pool hits {p['hits']}, misses {p['misses']})",
        )
        return holder, is_legacy

    # ------------------------------------------------------------------ run
    def _resolve_asset(self, value):
        if value is None:
            return None
        if os.path.isabs(value) and os.path.exists(value):
            return value
        cand = os.path.join(self.assets_folder, os.path.basename(value))
        if os.path.exists(cand):
            return cand
        if os.path.exists(value):
            return os.path.abspath(value)
        raise FileNotFoundError(f"control image not found: {value}")

    def _run(self, job: GenerationJob):
        job.status = "running"
        job.started_at = time.time()
        arch = job.model.get("arch")
        entry = get_arch_entry(arch)
        out_dir = os.path.join(self.output_folder, job.request_id)
        os.makedirs(out_dir, exist_ok=True)
        try:
            holder = self._ensure_holder(job)
            if job.cancel.is_set():
                raise GenerationCancelled()

            sample = dict(entry["sample"])
            sample.update({k: v for k, v in job.sample.items() if v is not None})
            for key in ("ctrl_img", "ctrl_img_1", "ctrl_img_2", "ctrl_img_3"):
                if sample.get(key):
                    sample[key] = self._resolve_asset(sample[key])
            if entry["needs_control_image"] and not sample.get("ctrl_img"):
                raise ValueError(f"{arch} requires a control image (ctrl_img)")
            output_ext = sample.pop("output_ext", None) or OUTPUT_EXT[entry["modality"]]
            if entry["modality"] == "video" and sample.get("num_frames", 1) > 1 and output_ext == "mp4":
                # BaseModel's video saver writes animated webp; archs with their
                # own mp4 writer (ltx2) override output_ext themselves
                output_ext = "webp"
            prompt = sample.pop("prompt", "")
            gen = GenerateImageConfig(
                prompt=prompt,
                output_folder=out_dir,
                output_ext=output_ext,
                **sample,
            )
            job.total_steps = gen.num_inference_steps
            job.emit(
                "start",
                model=self.holder_model,
                sample={
                    "prompt": gen.prompt,
                    "negative_prompt": gen.negative_prompt,
                    "width": gen.width,
                    "height": gen.height,
                    "num_inference_steps": gen.num_inference_steps,
                    "guidance_scale": gen.guidance_scale,
                    "seed": gen.seed,
                    "num_frames": gen.num_frames,
                    "fps": gen.fps,
                },
                modality=entry["modality"],
                preview=preview_info(arch),
            )

            t_start = time.perf_counter()
            every_n = max(1, int(job.stream.get("every_n_steps", 1)))
            want_latents = job.stream.get("latents", "raw") == "raw"
            max_frames = int(job.stream.get("max_frames", 4))

            def step_hook(idx, num_steps, latents):
                if job.cancel.is_set():
                    raise GenerationCancelled()
                job.step = idx + 1
                if num_steps:
                    job.total_steps = num_steps
                elapsed = time.perf_counter() - t_start
                job.emit("progress", step=job.step, total=job.total_steps, elapsed=round(elapsed, 3))
                self.status_cb(f"Generating {job.step}/{job.total_steps or '?'}")
                if want_latents and (idx % every_n == 0 or (job.total_steps and job.step == job.total_steps)):
                    self._emit_latent(job, latents, gen, max_frames)

            def status_hook(msg):
                job.emit("status", message=str(msg))

            holder.sample_step_hook = step_hook
            if hasattr(holder, "add_status_update_hook"):
                holder._status_update_hooks.append(status_hook)
            gen_kwargs = {"sampler": "ddpm"} if self.holder_is_legacy else {}
            self.status_cb(f"Generating with {arch}")
            before = set(_list_outputs(out_dir))
            t_gen = time.perf_counter()
            try:
                holder.generate_images([gen], **gen_kwargs)
            finally:
                gen_seconds = time.perf_counter() - t_gen
                holder.sample_step_hook = None
                if hasattr(holder, "_status_update_hooks"):
                    try:
                        holder._status_update_hooks.remove(status_hook)
                    except ValueError:
                        pass
            produced = [p for p in _list_outputs(out_dir) if p not in before]
            if not produced:
                raise RuntimeError("generation produced no output file")
            steps = job.total_steps or gen.num_inference_steps
            per_step = gen_seconds / steps if steps else 0
            size = f"{gen.width}x{gen.height}" + (f"x{gen.num_frames}f" if gen.num_frames > 1 else "")
            timing = f"Generated {arch} {size} in {gen_seconds:.2f}s ({steps} steps, {per_step:.2f}s/step, seed {gen.seed})"
            print(f"[AITK] {timing}", flush=True)
            job.emit("status", message=timing)
            for path in sorted(produced):
                ext = os.path.splitext(path)[1].lower().lstrip(".")
                result = {
                    "path": path,
                    "relpath": os.path.relpath(path, self.output_folder),
                    "kind": _kind_for_ext(ext),
                    "ext": ext,
                    "bytes": os.path.getsize(path),
                    "seed": gen.seed,
                    "width": gen.width,
                    "height": gen.height,
                    "num_frames": gen.num_frames,
                    "fps": gen.fps,
                    "seconds": round(gen_seconds, 3),
                    "steps": steps,
                }
                job.results.append(result)
                job.emit("result", **result)
            job.status = "done"
            self.stats["generations"] += 1
        except GenerationCancelled:
            job.status = "cancelled"
            job.error = "cancelled"
            self.stats["cancelled"] += 1
            job.emit("error", message="cancelled", cancelled=True)
            flush()
        except Exception as e:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            self.stats["errors"] += 1
            traceback.print_exc()
            job.emit("error", message=job.error, cancelled=False, traceback=traceback.format_exc())
            flush()
        finally:
            if job.status != "done":
                shutil.rmtree(out_dir, ignore_errors=True)
            job.finish()

    def _emit_latent(self, job: GenerationJob, latents, gen: GenerateImageConfig, max_frames: int):
        try:
            t = latents
            if hasattr(t, "tensor"):
                t = t.tensor
            if not isinstance(t, torch.Tensor):
                return
            t = t.detach()
            layout = None
            if t.dim() == 3 and t.shape[-1] % 4 == 0:
                # packed 2x2 patch tokens (flux family): (B, h/2*w/2, C*4)
                h = gen.height // 16
                w = gen.width // 16
                if t.shape[1] == h * w:
                    c = t.shape[-1] // 4
                    t = t.view(t.shape[0], h, w, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(t.shape[0], c, h * 2, w * 2)
            if t.dim() == 4:
                layout = "BCHW"
            elif t.dim() == 5:
                layout = "BCFHW"
                if max_frames and t.shape[2] > max_frames:
                    idx = torch.linspace(0, t.shape[2] - 1, max_frames).round().long().to(t.device)
                    t = t.index_select(2, idx)
            elif t.dim() == 3:
                layout = "BLC"  # sequence latents (audio: batch, length, channels)
            else:
                layout = "raw"
            fields, payload = tensor_payload(t[:1] if t.dim() >= 3 else t)
            extra = {}
            if not getattr(job, "_preview_sent", False) and layout in ("BCHW", "BCFHW"):
                # now that the real latent shape is known, resolve the preview
                # table (arch table, else a channel-count guess) once per job
                job._preview_sent = True
                extra["preview"] = preview_info(job.model.get("arch"), int(t.shape[1]), 2 if layout == "BCHW" else 3)
            job.emit("latent", payload, step=job.step, total=job.total_steps, layout=layout, **fields, **extra)
        except Exception as e:
            job.emit("status", message=f"latent preview skipped: {e}")


def _list_outputs(folder: str) -> List[str]:
    return [
        p
        for p in glob.glob(os.path.join(folder, "*"))
        if os.path.isfile(p) and not os.path.basename(p).startswith(".") and not p.endswith(".txt")
    ]


def _kind_for_ext(ext: str) -> str:
    if ext in ("mp4", "webm", "mov", "webp"):
        return "video" if ext != "webp" else "image"
    if ext in ("wav", "mp3", "flac", "ogg"):
        return "audio"
    return "image"
