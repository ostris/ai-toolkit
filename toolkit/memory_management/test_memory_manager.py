"""
Memory-manager (layer offloading) benchmark on a ~1B parameter diffusion-style
transformer. The base model is frozen and a LoRA is trained on top of it (the
realistic training setup), so only the LoRA params get grads/optimizer state.

Reports speed (ms/step) and peak VRAM for the 2x2 matrix of:

  - bfloat16 base       vs  bfloat16 + float8-quantized base (torchao weight-only)
  - no offloading       vs  100% offloading

The MemoryManager keeps the frozen base weights pinned on the CPU and streams
them onto the GPU per forward/backward (dequantizing float8 weights on the GPU).
The LoRA wraps each base linear, so its forward calls the bounced base forward
and adds the low-rank update. This trades VRAM for PCIe traffic, so the table
shows what that trade actually costs.

Run directly: `python test_memory_manager.py`
"""
import contextlib
import gc
import io
import os
import sys
import threading
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F


class RamMonitor:
    """Sample process RSS in a background thread and track the peak. Pinned
    CPU weights (from offloading) live in RSS, so this captures the host-RAM
    cost the GPU-side peak doesn't see."""

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self._proc = psutil.Process()
        self.peak = 0

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self._proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self._proc.memory_info().rss
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._thread.join()

# Allow running this file directly without setting PYTHONPATH.
# Toolkit imports happen inside main() so they pick this up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---- model ---------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.o(a)
        h = self.ln2(x)
        x = x + self.ffn_down(F.gelu(self.ffn_up(h)))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model=2048, n_heads=16, n_layers=24, d_ff=8192):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            # Gate on is_grad_enabled (not self.training): checkpointing only
            # helps and only works when a backward will actually be run.
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return self.norm(x)


# ---- benchmark -----------------------------------------------------------

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
QTYPE = "float8"
D_MODEL = 2048
N_HEADS = 16
N_LAYERS = 24
D_FF = 8192
BATCH = 1
SEQ = 1024
WARMUP = 3
ITERS = 10
LORA_RANK = 32
LR = 1e-4

# Full matrix: {bf16, float8} x {no offload, 100% offload} x {ckpt on, ckpt off}.
# Offloading parks weights in CPU RAM; turning off checkpointing keeps activations
# resident in VRAM. We report peak VRAM *and* peak system RAM so both show up.
# (label, quantize, offload_percent, grad_checkpointing)
RUNS = []
for _do_q, _qlabel in [(False, "bf16"), (True, "float8")]:
    for _off, _olabel in [(None, ""), (1.0, "+off")]:
        for _ckpt in [True, False]:
            _label = f"{_qlabel}{_olabel} ckpt={'on' if _ckpt else 'off'}"
            RUNS.append((_label, _do_q, _off, _ckpt))


def build_model():
    torch.manual_seed(0)
    # Build on CPU; the caller decides how it reaches the GPU.
    return Transformer(D_MODEL, N_HEADS, N_LAYERS, D_FF).to(dtype=DTYPE)


def build_lora(transformer):
    """Attach a trainable LoRA to the (frozen) transformer, the same way the
    trainer does it. Returns the network; its forward hijacks each base linear."""
    from toolkit.config_modules import NetworkConfig
    from toolkit.lora_special import LoRASpecialNetwork

    network_config = NetworkConfig(
        type="lora",
        linear=LORA_RANK,
        linear_alpha=LORA_RANK,
        transformer_only=True,
    )
    LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=transformer,
        lora_dim=network_config.linear,
        multiplier=1.0,
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        is_transformer=True,
        target_lin_modules=["Transformer"],
    )
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    network.force_to(DEVICE, DTYPE)
    network._update_torch_multiplier()
    network.is_active = True
    network.train()
    return network


def benchmark(results: list, label: str, do_quantize: bool, offload_percent, grad_checkpointing):
    from toolkit.memory_management import MemoryManager
    from toolkit.util.quantize import quantize, get_qtype
    from optimum.quanto import freeze

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    network = None
    model = build_model()
    model.gradient_checkpointing = grad_checkpointing
    model.to(DEVICE)

    if do_quantize:
        # Quantize the linear weights to float8 on the GPU (torchao weight-only),
        # exactly as a quantized base model is prepared before training.
        quantize(model, weights=get_qtype(QTYPE))
        freeze(model)

    # Base model is frozen; only the LoRA trains.
    model.requires_grad_(False)

    if offload_percent is None:
        # Baseline: whole base model stays on the GPU.
        model.to(DEVICE)
    else:
        # Offloading: managed linears stay pinned on CPU and bounce per step
        # (float8 weights are dequantized on the GPU); unmanaged modules (norms)
        # move to the GPU via the patched .to(). Attach BEFORE the LoRA so the
        # LoRA wraps the bouncing forward. Layer sampling is seeded for repro.
        import random
        random.seed(0)
        MemoryManager.attach(model, DEVICE, offload_percent=offload_percent)
        model.to(DEVICE)

    # build_lora prints a banner per layer; mute it so the final table is clean.
    with contextlib.redirect_stdout(io.StringIO()):
        network = build_lora(model)
    params = network.prepare_optimizer_params(LR, LR, LR)
    opt = torch.optim.AdamW(params, lr=LR)
    x = torch.randn(BATCH, SEQ, D_MODEL, device=DEVICE, dtype=DTYPE)

    try:
        for _ in range(WARMUP):
            opt.zero_grad(set_to_none=True)
            model(x).sum().backward()
            opt.step()
        torch.cuda.synchronize()

        # Measure the steady-state TRAINING peak, not the one-time setup load.
        # (Offload first parks the whole model on the GPU before bouncing it to
        # CPU; counting that transient would hide the real per-step footprint.)
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with RamMonitor() as ram:
            for _ in range(ITERS):
                opt.zero_grad(set_to_none=True)
                model(x).sum().backward()
                opt.step()
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / ITERS * 1000
        peak = torch.cuda.max_memory_allocated() / 1024**3
        ram_gb = ram.peak / 1024**3
        results.append({"label": label, "ms": dt, "peak": peak, "ram": ram_gb, "ok": True})
    except torch.cuda.OutOfMemoryError:
        results.append({"label": label, "ms": float("inf"), "peak": float("inf"), "ram": float("inf"), "ok": False, "note": "OOM"})
    except Exception as e:
        print(f"    {label} failed: {type(e).__name__}: {e}", flush=True)
        results.append({"label": label, "ms": float("inf"), "peak": float("inf"), "ram": float("inf"), "ok": False, "note": "ERR"})
    finally:
        if offload_percent is not None:
            MemoryManager.detach(model)
        del opt, network, model
        gc.collect()
        torch.cuda.empty_cache()


def print_table(results: list):
    headers = ["#", "Configuration", "Peak VRAM", "Peak RAM", "Time/step"]
    rows = []
    for i, r in enumerate(results, 1):
        if not r["ok"]:
            rows.append([str(i), r["label"], r.get("note", "OOM"), "-", "-"])
            continue
        rows.append([str(i), r["label"], f"{r['peak']:.2f} GB", f"{r['ram']:.2f} GB", f"{r['ms']:.1f} ms"])

    widths = [max(len(str(row[c])) for row in [headers] + rows) for c in range(len(headers))]

    def fmt(row, sep=" │ "):
        return sep.join(s.ljust(widths[c]) if c == 1 else s.rjust(widths[c]) for c, s in enumerate(row))

    line_top = "─" * (sum(widths) + 3 * (len(widths) - 1))
    print()
    print(line_top)
    print(fmt(headers))
    print(line_top)
    for row in rows:
        print(fmt(row))
    print(line_top)


def run_one(idx: int):
    """Run a single config and print its result as JSON. Invoked in a fresh
    subprocess so peak RAM (and VRAM) are isolated — pinned-host and CUDA host
    caches don't release between runs, so in-process RAM peaks would accumulate."""
    import json

    label, do_quantize, offload_percent, grad_checkpointing = RUNS[idx]
    results: list = []
    benchmark(results, label, do_quantize, offload_percent, grad_checkpointing)
    print("RESULT " + json.dumps(results[0]), flush=True)


def main():
    import json
    import subprocess

    n_params = sum(p.numel() for p in build_model().parameters())
    print(f"Model:  {N_LAYERS} blocks × d_model={D_MODEL} × d_ff={D_FF}")
    print(f"        {n_params/1e6:.1f}M params")
    print(f"dtype:  {str(DTYPE).replace('torch.', '')} (quant qtype: {QTYPE})")
    print(f"Train:  LoRA rank={LORA_RANK} on a frozen base")
    print(f"Step:   batch={BATCH}, seq={SEQ}")
    print(f"Timing: {WARMUP} warmup + {ITERS} timed iters")
    print(f"Configs: {len(RUNS)} (each in an isolated subprocess)")

    results: list = []
    for idx, run in enumerate(RUNS):
        print(f"  running {run[0]}...", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--run", str(idx)],
            capture_output=True, text=True,
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(f"    {run[0]} produced no result:\n{proc.stdout}\n{proc.stderr}", flush=True)
            results.append({"label": run[0], "ok": False, "note": "ERR"})
            continue
        results.append(json.loads(line[len("RESULT "):]))
    print_table(results)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--run":
        run_one(int(sys.argv[2]))
    else:
        main()
