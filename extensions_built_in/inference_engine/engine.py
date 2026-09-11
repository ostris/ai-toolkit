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
from toolkit.inference_lora import InferenceLoRA, LoRAStack
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
        # every frame so far (latents: only the newest kept) so a client can
        # attach late / reattach after a reload and see the same stream
        self.history: List[bytes] = []
        self._latent_index: Optional[int] = None
        self._subscribers: List["queue.Queue[Optional[bytes]]"] = []
        self._sub_lock = threading.Lock()
        self.finished = False
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
        data = encode_frame(header, payload)
        with self._sub_lock:
            if type_ == "latent":
                if self._latent_index is not None:
                    self.history[self._latent_index] = data
                else:
                    self._latent_index = len(self.history)
                    self.history.append(data)
            else:
                self.history.append(data)
            for q in self._subscribers:
                q.put(data)

    def subscribe(self, replay: bool = True) -> "queue.Queue[Optional[bytes]]":
        """A queue that receives this job's frames (history first when
        replay) and END once the job is finished."""
        q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        with self._sub_lock:
            if replay:
                for data in self.history:
                    q.put(data)
            if self.finished:
                q.put(END)
            else:
                self._subscribers.append(q)
        return q

    def finish(self):
        self.finished_at = time.time()
        self.emit("end", status=self.status)
        with self._sub_lock:
            self.finished = True
            for q in self._subscribers:
                q.put(END)
            self._subscribers = []

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


# settings that can change on a loaded holder without reloading any weights:
# the memory manager is detached and re-attached at the new fractions
OFFLOAD_KEYS = ("layer_offloading", "layer_offloading_transformer_percent", "layer_offloading_text_encoder_percent")
# LoRAs ride the request's model block but are applied on top of the loaded
# holder (hook mode: dynamic; merge mode: into the weights, which reloads
# the touched components when the merged set changes)
LORA_KEYS = ("loras", "lora_mode")


def _without(model: dict, keys) -> dict:
    return {k: v for k, v in model.items() if k not in keys}


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
        self.loading = False
        self.holder = None
        self.holder_key: Optional[str] = None
        self.holder_model: Optional[dict] = None
        self.holder_is_legacy = False
        self.lora_stack = None
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
            # bounded memory: drop the oldest finished jobs' frame history
            finished = [j for j in self.jobs.values() if j.finished]
            if len(finished) > 100:
                finished.sort(key=lambda j: j.created_at)
                for old in finished[: len(finished) - 100]:
                    self.jobs.pop(old.request_id, None)
        self.queue.put(job)
        return job

    def get_job(self, request_id: str) -> Optional[GenerationJob]:
        return self.jobs.get(request_id)

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
            "active": {
                "model": self.holder_model,
                "arch": (self.holder_model or {}).get("arch"),
                "loras": [l.summary() for l in self.lora_stack.loras] if self.lora_stack else [],
                "lora_mode": self.lora_stack.mode if self.lora_stack else None,
            }
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
        if self.lora_stack is not None and self.lora_stack.mode == "hook":
            self.lora_stack.remove()
        self.lora_stack = None
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
        self.current = job
        try:
            self._ensure_holder(job)
            job.status = "done"
        except Exception as e:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            self.current = None
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
        lora_specs = [l for l in (model.get("loras") or []) if l.get("path")]
        lora_mode = model.get("lora_mode") or "hook"
        model = _without(model, LORA_KEYS)
        key = _model_key(model)
        if self.holder is not None and key == self.holder_key:
            self._apply_loras(job, lora_specs, lora_mode)
            return self.holder
        if (
            self.holder is not None
            and _model_key(_without(model, OFFLOAD_KEYS)) == _model_key(_without(self.holder_model, OFFLOAD_KEYS))
        ):
            # same weights, different offloading: re-stage in place
            self._apply_offload(job, model)
            self.holder_model = model
            self.holder_key = key
            self._apply_loras(job, lora_specs, lora_mode)
            return self.holder
        if self.holder is not None:
            self._drop_holder()

        # merged LoRAs live inside pooled weights: a component merged with a
        # different set than this request wants must reload from disk
        wanted = LoRAStack(None, lora_mode)
        wanted.loras = [InferenceLoRA(l["path"], l.get("strength", 1.0)) for l in lora_specs] if lora_mode == "merge" else []
        wanted_key = wanted.key() if lora_mode == "merge" else None
        freed = self.pool.evict_where(lambda m: getattr(m, "_aitk_merged_loras", None) not in (None, wanted_key))
        if freed:
            job.emit("status", message=f"Reloading {freed / 1e9:.1f} GB of components that carried other merged LoRAs")

        self.pool.begin_request()
        ComponentPool.current = self.pool
        self.loading = True
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
            self.loading = False
        freed = self.pool.release_unused()
        if freed:
            job.emit("status", message=f"Released {freed / 1e9:.1f} GB of components not used by {model['arch']}")
        flush()
        self.holder = holder
        self.holder_key = key
        self.holder_model = model
        self.holder_is_legacy = is_legacy
        self.lora_stack = None
        self._apply_loras(job, lora_specs, lora_mode)
        return holder

    def _apply_loras(self, job: GenerationJob, specs: List[dict], mode: str):
        """Bring the holder's LoRA state to `specs` in `mode`. Hook mode:
        strengths update in place, a different set re-attaches. Merge mode:
        a different set than what the weights carry evicts and reloads them."""
        wanted = LoRAStack(self.holder, mode)
        wanted.loras = [InferenceLoRA(l["path"], l.get("strength", 1.0)) for l in specs]
        wanted_key = wanted.key()
        current = self.lora_stack
        if current is not None and current.key() == wanted_key:
            return
        if current is not None and current.mode == "hook" and mode == "hook" and current.set_strengths(specs):
            self.lora_stack = current
            return

        def status(msg):
            job.emit("status", message=msg)
            self.status_cb(msg)

        if current is not None:
            if current.mode == "merge":
                # weights carry the old set: reload the merged components
                status("Reloading weights to drop merged LoRAs")
                merged_key = current.key()
                self._drop_holder()
                self.pool.evict_where(lambda m: getattr(m, "_aitk_merged_loras", None) == merged_key)
                flush()
                self.lora_stack = None
                self._ensure_holder_for_loras(job, specs, mode)
                return
            current.remove()
            self.lora_stack = None
        if not specs:
            return
        stack = LoRAStack(self.holder, mode).load(specs, status_fn=status)
        touched = stack.apply(status_fn=status)
        if mode == "merge":
            for root in [getattr(self.holder, "model", None)] + (
                self.holder.text_encoder if isinstance(getattr(self.holder, "text_encoder", None), list) else [getattr(self.holder, "text_encoder", None)]
            ):
                if isinstance(root, torch.nn.Module) and any(id(m) in touched for m in root.modules()):
                    root._aitk_merged_loras = stack.key()
        self.lora_stack = stack

    def _ensure_holder_for_loras(self, job: GenerationJob, specs: List[dict], mode: str):
        model = dict(self.holder_model or job.model)
        model["loras"] = specs
        model["lora_mode"] = mode
        job_model = job.model
        job.model = model
        try:
            self._ensure_holder(job)
        finally:
            job.model = job_model

    def _apply_offload(self, job: GenerationJob, model: dict):
        """Detach/re-attach layer offloading on the loaded holder's components
        at the requested fractions. No weights are reloaded."""
        from toolkit.memory_management import MemoryManager

        holder = self.holder
        mc = holder.model_config
        enabled = bool(model.get("layer_offloading", False))
        t_pct = float(model.get("layer_offloading_transformer_percent", 1.0))
        te_pct = float(model.get("layer_offloading_text_encoder_percent", 1.0))
        mc.layer_offloading = enabled
        mc.layer_offloading_transformer_percent = t_pct
        mc.layer_offloading_text_encoder_percent = te_pct
        device = torch.device(self.device)

        targets = []
        model_module = getattr(holder, "model", None)
        subs = [getattr(model_module, n, None) for n in ("transformer_1", "transformer_2")]
        if model_module is not None and all(m is not None for m in subs):
            targets += [(m, t_pct) for m in subs]
        elif model_module is not None:
            targets.append((model_module, t_pct))
        tes = getattr(holder, "text_encoder", None)
        for te in tes if isinstance(tes, list) else [tes]:
            targets.append((te, te_pct))

        desc = f"offload {'on' if enabled else 'off'} (transformer {t_pct:.0%}, text encoder {te_pct:.0%})"
        job.emit("status", message=f"Applying {desc}")
        self.status_cb(f"Applying {desc}")
        changed = 0
        for module, pct in targets:
            if not isinstance(module, torch.nn.Module) or type(module).__name__.startswith("Fake"):
                continue
            if next(module.parameters(), None) is None:
                continue
            if hasattr(module, "_memory_manager"):
                MemoryManager.detach(module)
            if enabled and pct > 0:
                get_ignore = getattr(module, "get_offload_ignore_modules", None)
                ignore = get_ignore() if callable(get_ignore) else None
                MemoryManager.attach(module, device, offload_percent=pct, ignore_modules=list(ignore or []))
            else:
                module.to(device)
            # keep the pool's policy record in sync so a later identical
            # request is a no-op rather than a re-attach
            policy = getattr(module, "_aitk_policy", None)
            if isinstance(policy, dict):
                policy["offload"] = pct if enabled else 0.0
            changed += 1
        flush()
        job.emit("status", message=f"Applied {desc} to {changed} components")

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
                model={**(self.holder_model or {}), "loras": job.model.get("loras") or [], "lora_mode": job.model.get("lora_mode") or "hook"},
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
        except (GenerationCancelled, KeyboardInterrupt) as e:
            job.status = "cancelled"
            job.error = "cancelled"
            self.stats["cancelled"] += 1
            job.emit("error", message="cancelled" if isinstance(e, GenerationCancelled) else "engine stopping", cancelled=True)
            flush()
            if isinstance(e, KeyboardInterrupt):
                job.finish()
                raise
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
