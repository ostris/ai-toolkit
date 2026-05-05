"""
Optimizer benchmark on a ~500M parameter fp32 transformer.

Compares speed (ms/step) and peak VRAM across:
  - AdamW (torch, unfused — traditional Python loop)
  - AdamW8bit (bitsandbytes)
  - Adafactor
  - Automagic v1
  - Automagic v2 (only optimizer using fused-backward)
  - Prodigy
"""
import contextlib
import gc
import io
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running this file directly: `python test_optimizers.py` without setting PYTHONPATH.
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
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.o(a)
        h = self.ln2(x)
        x = x + self.ffn_down(F.gelu(self.ffn_up(h)))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model=1024, n_heads=16, n_layers=40, d_ff=4096):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


# ---- benchmark -----------------------------------------------------------

DEVICE = "cuda"
DTYPE = torch.float32
D_MODEL = 1024
N_HEADS = 16
N_LAYERS = 40
D_FF = 4096
BATCH = 1
SEQ = 128
WARMUP = 3
ITERS = 10


def build_model():
    torch.manual_seed(0)
    return Transformer(D_MODEL, N_HEADS, N_LAYERS, D_FF).to(DEVICE, dtype=DTYPE)


def benchmark(results: list, label: str, opt_factory):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_model()
    # Some optimizers print on construction; mute that so the final table is clean.
    with contextlib.redirect_stdout(io.StringIO()):
        opt = opt_factory(model.parameters())
    x = torch.randn(BATCH, SEQ, D_MODEL, device=DEVICE, dtype=DTYPE)

    print(f"  running {label}...", flush=True)
    try:
        for _ in range(WARMUP):
            opt.zero_grad(set_to_none=True)
            model(x).sum().backward()
            opt.step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            opt.zero_grad(set_to_none=True)
            model(x).sum().backward()
            opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / ITERS * 1000
        peak = torch.cuda.max_memory_allocated() / 1024**3
        results.append({"label": label, "ms": dt, "peak": peak, "ok": True})
    except torch.cuda.OutOfMemoryError:
        results.append({"label": label, "ms": float("inf"), "peak": float("inf"), "ok": False})
    finally:
        del opt, model
        gc.collect()
        torch.cuda.empty_cache()


def print_table(results: list):
    results = sorted(results, key=lambda r: r["peak"])

    headers = ["#", "Optimizer", "Peak VRAM", "Time/step"]
    rows = []
    for i, r in enumerate(results, 1):
        if not r["ok"]:
            rows.append([str(i), r["label"], "OOM", "-"])
            continue
        rows.append([str(i), r["label"], f"{r['peak']:.2f} GB", f"{r['ms']:.1f} ms"])

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


def main():
    n_params = sum(p.numel() for p in build_model().parameters())
    dtype_name = str(DTYPE).replace("torch.", "")
    print(f"Model:  {N_LAYERS} blocks × d_model={D_MODEL} × d_ff={D_FF}")
    print(f"        {n_params/1e6:.1f}M params ({dtype_name})")
    print(f"Step:   batch={BATCH}, seq={SEQ}")
    print(f"Timing: {WARMUP} warmup + {ITERS} timed iters")
    print()

    from toolkit.optimizers.automagic import Automagic
    from toolkit.optimizers.automagic2 import Automagic2
    from toolkit.optimizers.adafactor import Adafactor
    from prodigyopt import Prodigy
    import bitsandbytes as bnb

    results: list = []
    benchmark(results, "AdamW",
              lambda p: torch.optim.AdamW(p, lr=1e-4, eps=1e-6, foreach=False, fused=False))
    benchmark(results, "AdamW8bit",
              lambda p: bnb.optim.AdamW8bit(p, lr=1e-4, eps=1e-6))
    benchmark(results, "Adafactor",
              lambda p: Adafactor(p, lr=1e-4, scale_parameter=False, relative_step=False, warmup_init=False))
    benchmark(results, "Automagic v1",
              lambda p: Automagic(p, lr=1e-4))
    benchmark(results, "Automagic v2",
              lambda p: Automagic2(p, lr=1e-4))
    benchmark(results, "Prodigy",
              lambda p: Prodigy(p, lr=1.0, eps=1e-6))

    print_table(results)


if __name__ == "__main__":
    main()
