"""Per-stage speed / memory / CPU profiler for the model loading test harness.

A StageProfiler splits a run into named stages (the harness hooks
BaseModel.print_and_status_update so every holder status line — "Loading
transformer", "Quantizing (convrot8)", "Loading VAE", ... — opens a new
stage). For every stage it records:

  - seconds          wall time
  - vram_peak_gb     torch.cuda.max_memory_allocated during the stage
  - vram_reserved_gb torch.cuda.max_memory_reserved during the stage
  - vram_end_gb      memory_allocated at the stage boundary (resident model)
  - rss_peak_gb      peak process RSS (sampled at 50ms)
  - cpu_avg_pct /    process CPU utilization (100 = one full core), sampled
    cpu_max_pct      at 50ms — this is what shows where quantization burns CPU
  - threads_max      peak thread count

A background sampler thread does the RSS/CPU sampling so short spikes inside
a stage are caught, not just the boundary values.
"""

import threading
import time

import psutil
import torch

_GB = 1024**3


class _Sampler:
    """Background thread sampling RSS / CPU% / thread count at 50ms."""

    def __init__(self):
        self.proc = psutil.Process()
        self._lock = threading.Lock()
        self._stop = False
        self.reset()
        self.proc.cpu_percent(None)  # prime the cpu_percent window
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def reset(self):
        with self._lock:
            self.rss_peak = self.proc.memory_info().rss
            self.cpu_samples = []
            self.threads_max = self.proc.num_threads()

    def _run(self):
        while not self._stop:
            try:
                rss = self.proc.memory_info().rss
                cpu = self.proc.cpu_percent(None)
                threads = self.proc.num_threads()
                with self._lock:
                    self.rss_peak = max(self.rss_peak, rss)
                    self.cpu_samples.append(cpu)
                    self.threads_max = max(self.threads_max, threads)
            except Exception:
                pass
            time.sleep(0.05)

    def stats(self):
        with self._lock:
            samples = self.cpu_samples or [0.0]
            return {
                "rss_peak_gb": round(self.rss_peak / _GB, 3),
                "cpu_avg_pct": round(sum(samples) / len(samples), 1),
                "cpu_max_pct": round(max(samples), 1),
                "threads_max": self.threads_max,
            }

    def stop(self):
        self._stop = True


class StageProfiler:
    def __init__(self, device):
        self.device = torch.device(device) if torch.cuda.is_available() else None
        self.stages = []
        self._current = None
        self._sampler = _Sampler()

    def _cuda_sync_reset(self):
        if self.device is not None:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def stage(self, name: str):
        """Close the current stage (if any) and open a new one."""
        self._close()
        self._cuda_sync_reset()
        self._sampler.reset()
        self._current = {"name": str(name), "t0": time.perf_counter()}

    def _close(self):
        if self._current is None:
            return
        cur = self._current
        self._current = None
        seconds = time.perf_counter() - cur["t0"]
        entry = {"name": cur["name"], "seconds": round(seconds, 3)}
        if self.device is not None:
            torch.cuda.synchronize(self.device)
            entry["vram_peak_gb"] = round(
                torch.cuda.max_memory_allocated(self.device) / _GB, 3
            )
            entry["vram_reserved_gb"] = round(
                torch.cuda.max_memory_reserved(self.device) / _GB, 3
            )
            entry["vram_end_gb"] = round(
                torch.cuda.memory_allocated(self.device) / _GB, 3
            )
        entry.update(self._sampler.stats())
        self.stages.append(entry)

    def finish(self):
        self._close()
        self._sampler.stop()
        return self.stages


def profile_top(profile, limit=40):
    """Top functions from a cProfile.Profile by cumulative time, as dicts."""
    import pstats

    stats = pstats.Stats(profile)
    stats.sort_stats("cumulative")
    rows = []
    for func in stats.fcn_list[: limit * 3]:
        cc, nc, tt, ct, _ = stats.stats[func]
        filename, line, name = func
        # drop the profiler/exec wrappers and trivial rows
        if name in ("<module>",) or ct < 0.05:
            continue
        # cProfile catches this module's own 50ms sampling thread; its idle
        # sleep loop and psutil polling otherwise show up as a giant fake
        # "time.sleep" hotspot spanning the whole stage
        if "stage_profiler" in filename or "psutil" in filename or "_pslinux" in filename:
            continue
        if name == "<built-in method time.sleep>":
            continue
        # shorten site-packages / repo paths to keep the report readable
        for marker in ("site-packages/", "ai-toolkit/"):
            if marker in filename:
                filename = filename.split(marker, 1)[1]
                break
        rows.append(
            {
                "func": f"{filename}:{line}({name})",
                "ncalls": nc,
                "tottime": round(tt, 3),
                "cumtime": round(ct, 3),
            }
        )
        if len(rows) >= limit:
            break
    return rows
