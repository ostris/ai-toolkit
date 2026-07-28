"""
Reproducible speed / VRAM / accuracy benchmarks for the toolkit quantization
backends.

Compares bf16 against the custom OstrisLinear backends (convrot8, convrot4 for
now; add more qtypes to QTYPES as they land).

Measures, per qtype:
  - layer inference latency across DiT-representative shapes (vs bf16),
    eager and torch.compile'd
  - layer training latency (forward + backward through the frozen layer),
    eager and torch.compile'd
  - VRAM on a transformer-ish block stack: resident weights, peak during a
    no-grad forward, peak during a train step; forward/train-ckpt peaks also
    under torch.compile
  - accuracy drift vs bf16: output relative error per layer shape and
    accumulated through the block stack
  - weight reconstruction error and one-time quantize (conversion) time

Runs on CUDA or on Apple Silicon (MPS). See DEVICE NOTES below for what MPS
can and cannot measure, and which qtypes it cannot run at all.

Usage:
    python scripts/test_quantizations.py --gpu 1
    python scripts/test_quantizations.py --gpu 1 --qtypes bf16 convrot8
    python scripts/test_quantizations.py --device mps
"""

import argparse
import math
import os
import sys
import threading
import time

# set cuda bus ordering to be pcie
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

# ---------------------------------------------------------------- DEVICE NOTES
#
# MPS differs from CUDA in three ways that matter to a benchmark, so the numbers
# it prints are not interchangeable with CUDA's:
#
# 1. No peak-memory API. torch.mps has current_allocated_memory() but no
#    max_memory_allocated()/reset_peak_memory_stats(), so the vram peaks are
#    SAMPLED by a side thread polling the allocator (see _MpsPeakSampler)
#    instead of read exactly. Steady-state values (resident weights) are exact.
# 2. No fp8 dtype. torch.float8_e4m3fn is undefined on MPS, which rules out the
#    qfloat8 qtype and convrot4 (its nvfp4 block scales are stored as e4m3).
#    MPS_UNSUPPORTED lists them; they are dropped from a default run.
# 3. No int8/fp4 tensor cores, so the convrot backends run their W8A16 fallback
#    path rather than the W8A8/W4A4 fast path. Latency here says what Apple
#    hardware does, not what the format is worth on a GPU that can run it.


def pick_device(name: str, gpu: int) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            raise SystemExit("no cuda or mps device available")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but cuda is not available")
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
        return device
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("--device mps requested but mps is not available")
        return torch.device("mps")
    raise SystemExit(f"unsupported device {name!r}")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return (f"{device} ({p.name}, sm_{p.major}{p.minor}, "
                f"{p.total_memory / 1e9:.0f} GB)")
    import platform
    return (f"{device} (Apple {platform.machine()}, "
            f"{torch.mps.recommended_max_memory() / 1e9:.0f} GB recommended max)")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def mem_allocated(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return 0


class _MpsPeakSampler:
    """Approximate max_memory_allocated for MPS by polling the allocator.

    Tensors are allocated on the calling thread as ops are enqueued, so a
    fine-grained poll from a side thread does observe the transients; on the
    probe cases it recovered exact expected sizes (a 512 MiB transient and a
    24 MiB matmul output). It can still miss a transient shorter than the poll
    interval, so treat MPS peaks as a lower bound, not a hard number.
    """

    def __init__(self, device, interval=0.0002):
        self.device, self.interval = device, interval
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.peak = mem_allocated(self.device)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, mem_allocated(self.device))
            time.sleep(self.interval)

    def __exit__(self, *exc):
        sync(self.device)
        self.peak = max(self.peak, mem_allocated(self.device))
        self._stop.set()
        self._thread.join()
        return False


def measure_peak(run, device: torch.device, base: int) -> int:
    """Peak bytes allocated during run(), over base. run() is called once first
    so lazy init and compilation are not charged to the steady-state peak."""
    run()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        run()
        sync(device)
        return torch.cuda.max_memory_allocated(device) - base
    with _MpsPeakSampler(device) as sampler:
        run()
    return sampler.peak - base

# (tokens, in_features, out_features) — FLUX/Wan-class projections
SPEED_SHAPES = [
    (4096, 3072, 3072),
    (4096, 3072, 12288),
    (4096, 12288, 3072),
    (1024, 3072, 12288),
]

# block stack used for the vram/drift tests (mimics a DiT block's linears)
VRAM_BLOCKS = 8
VRAM_BLOCK_SHAPES = [(3072, 12288), (12288, 3072), (3072, 3072), (3072, 3072)]
VRAM_TOKENS = 4096

QTYPES = [
    "bf16", "qfloat8", "float8", "convrot8", "convrot4",
    "convrotint7", "convrotint6", "convrotint5", "convrotint4", "convrotint3",
    "convrotint2", "convrotbitnet", "convrotcomfyw4a4",
]

# qtypes that cannot run on MPS at all (see DEVICE NOTES: no fp8 dtype).
# quanto's float8 path is worse than a hard failure — it catches the dtype error,
# prints "Failed to quantize", and leaves the layer in bf16, so it would benchmark
# as bf16 under a quantized label. Dropped for the same reason.
MPS_UNSUPPORTED = {"qfloat8", "float8", "convrot4"}

STACK_KEY = f"{VRAM_BLOCKS}-block stack"


def convert(module: torch.nn.Linear, qtype: str) -> torch.nn.Linear:
    """Quantize a linear with the given qtype. Returns the (possibly replaced)
    module — quanto swaps the module object, the ostris backends convert in place."""
    if qtype == "bf16":
        return module
    from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer

    q = get_ostris_quantizer(qtype)
    if q is not None:
        assert convert_linear_to_ostris(module, q), f"conversion refused for {qtype}"
        return module

    # quanto / torchao qtypes go through the shared toolkit quantize flow; use a
    # holder so quanto's module replacement has a parent to swap into
    from optimum.quanto import freeze
    from toolkit.util.quantize import get_qtype, quantize

    holder = torch.nn.Sequential(module)
    quantize(holder, weights=get_qtype(qtype))
    freeze(holder)
    return holder[0]


def fp_weight(module: torch.nn.Linear) -> torch.Tensor:
    """Dequantized weight in float32, whatever the backend."""
    if hasattr(module, "dequantize_weight"):
        return module.dequantize_weight().float()
    w = module.weight
    if hasattr(w, "dequantize"):
        return w.dequantize().float()
    return w.detach().float()


def bench(fn, iters: int, device) -> float:
    for _ in range(max(3, iters // 5)):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000  # ms


def gb(nbytes: int) -> str:
    return f"{nbytes / 1e9:6.2f} GB"


def make_layer(k: int, n: int, device) -> torch.nn.Linear:
    lin = torch.nn.Linear(k, n, bias=True, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        lin.weight.mul_(0.02)
    # the train benches model lora-style training: base frozen, grads flow to
    # the input only. without this, bf16/quanto accumulate weight grads that
    # inflate every later vram measurement
    lin.requires_grad_(False)
    return lin


def make_stack(device) -> torch.nn.ModuleList:
    # default nn.Linear init (~1/sqrt(in) std) so block branches contribute at a
    # realistic O(1) scale to the residual stream — scaling weights down further
    # makes accumulated quantization drift look artificially tiny
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList()
    for _ in range(VRAM_BLOCKS):
        blocks.append(torch.nn.ModuleList([
            torch.nn.Linear(k, n, bias=True, dtype=torch.bfloat16, device=device)
            for k, n in VRAM_BLOCK_SHAPES
        ]))
    # frozen base (see make_layer)
    blocks.requires_grad_(False)
    return blocks


def block_forward(b, h):
    # pre-norm residual block like a real transformer, so activations stay at a
    # sane scale and quantization drift accumulates realistically across depth
    r = torch.nn.functional.layer_norm(h, h.shape[-1:])
    r = b[0](r)          # 3072 -> 12288
    r = torch.nn.functional.gelu(r)
    h = h + b[1](r)      # 12288 -> 3072
    r = torch.nn.functional.layer_norm(h, h.shape[-1:])
    return h + b[3](b[2](r))  # 3072 -> 3072 -> 3072


def stack_forward(blocks, x, checkpoint=False):
    h = x
    for b in blocks:
        if checkpoint:
            h = torch.utils.checkpoint.checkpoint(
                block_forward, b, h, use_reentrant=False
            )
        else:
            h = block_forward(b, h)
    return h


def run_speed(qtype: str, device, iters: int, results: dict):
    for m, k, n in SPEED_SHAPES:
        torch.manual_seed(0)
        lin = make_layer(k, n, device)
        lin = convert(lin, qtype)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            t_inf = bench(lambda: lin(x), iters, device)

        def train_step():
            xi = x.detach().requires_grad_(True)
            lin(xi).sum().backward()

        t_train = bench(train_step, max(10, iters // 3), device)
        results[(qtype, "inf", (m, k, n))] = t_inf
        results[(qtype, "train", (m, k, n))] = t_train

        # compiled variants (compilation happens during bench warmup, so it
        # isn't charged to the timing; a backend that won't compile records
        # nothing and shows as '-')
        lin_c = torch.compile(lin, dynamic=False)
        try:
            with torch.no_grad():
                results[(qtype, "inf_comp", (m, k, n))] = bench(
                    lambda: lin_c(x), iters, device
                )
        except Exception as e:
            print(f"  [{qtype}] compiled inference failed for {m}x{k}->{n}: {e}")

        def train_step_c():
            xi = x.detach().requires_grad_(True)
            lin_c(xi).sum().backward()

        try:
            results[(qtype, "train_comp", (m, k, n))] = bench(
                train_step_c, max(10, iters // 3), device
            )
        except Exception as e:
            print(f"  [{qtype}] compiled train failed for {m}x{k}->{n}: {e}")
        empty_cache(device)


def _stack_fwd_peak(blocks, x, device, base) -> int:
    def fwd():
        with torch.no_grad():
            stack_forward(blocks, x)

    return measure_peak(fwd, device, base)


def _stack_train_peak(blocks, x, device, base, checkpoint) -> int:
    # frozen base; grads flow to the input like lora training
    def train_step():
        xi = x.detach().requires_grad_(True)
        stack_forward(blocks, xi, checkpoint).float().pow(2).mean().backward()

    return measure_peak(train_step, device, base)


def run_vram(qtype: str, device, results: dict):
    empty_cache(device)
    base = mem_allocated(device)

    blocks = make_stack(device)
    for b in blocks:
        for i in range(len(b)):
            b[i] = convert(b[i], qtype)
    empty_cache(device)
    results[(qtype, "vram_weights")] = mem_allocated(device) - base

    x = torch.randn(VRAM_TOKENS, 3072, device=device, dtype=torch.bfloat16)

    results[(qtype, "vram_fwd_peak")] = _stack_fwd_peak(blocks, x, device, base)
    # real training checkpoints, but the plain train peak still gets reported
    results[(qtype, "vram_train_peak")] = _stack_train_peak(blocks, x, device, base, False)
    results[(qtype, "vram_train_ckpt_peak")] = _stack_train_peak(blocks, x, device, base, True)

    # same peaks with every linear compiled (mirrors the trainer's block compile)
    for b in blocks:
        for i in range(len(b)):
            b[i] = torch.compile(b[i], dynamic=False)
    try:
        results[(qtype, "vram_fwd_peak_comp")] = _stack_fwd_peak(blocks, x, device, base)
        results[(qtype, "vram_train_ckpt_peak_comp")] = _stack_train_peak(
            blocks, x, device, base, True
        )
    except Exception as e:
        print(f"  [{qtype}] compiled vram measurement failed: {e}")

    blocks = x = None  # release before the allocator accounting of the next run
    empty_cache(device)


def run_drift(qtype: str, device, results: dict):
    """Output error vs the bf16 reference, per layer shape and through the stack."""
    for m, k, n in SPEED_SHAPES:
        torch.manual_seed(0)
        lin = make_layer(k, n, device)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            y_ref = lin(x).float()
            lin = convert(lin, qtype)
            y_q = lin(x).float()
        results[(qtype, "drift", (m, k, n))] = ((y_q - y_ref).norm() / y_ref.norm()).item()
        empty_cache(device)

    blocks = make_stack(device)
    x = torch.randn(VRAM_TOKENS, 3072, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        y_ref = stack_forward(blocks, x).float()
        for b in blocks:
            for i in range(len(b)):
                b[i] = convert(b[i], qtype)
        y_q = stack_forward(blocks, x).float()
    results[(qtype, "drift", STACK_KEY)] = ((y_q - y_ref).norm() / y_ref.norm()).item()
    blocks = x = None
    empty_cache(device)


def run_quality_and_quantize_time(qtype: str, device, results: dict):
    torch.manual_seed(0)
    lin = make_layer(3072, 3072, device)
    w0 = lin.weight.detach().float().clone()
    sync(device)
    t0 = time.perf_counter()
    lin = convert(lin, qtype)
    sync(device)
    results[(qtype, "quantize_ms")] = (time.perf_counter() - t0) * 1000
    if qtype == "bf16":
        results[(qtype, "weight_err")] = 0.0
    else:
        wq = fp_weight(lin)
        results[(qtype, "weight_err")] = ((wq - w0).norm() / w0.norm()).item()
    empty_cache(device)


def print_speed_table(title: str, kind: str, qts, results):
    # speedups always reference EAGER bf16, compiled kinds included, so the
    # comp columns answer "what do I gain over plain bf16"
    ref_kind = kind.removesuffix("_comp")
    print(f"\n=== {title} (ms; speedup vs eager bf16) ===")
    print(f"{'M x K -> N':<22}" + "".join(f"{qt:>18}" for qt in qts))
    for shape in SPEED_SHAPES:
        m, k, n = shape
        row = f"{f'{m} x {k} -> {n}':<22}"
        ref = results.get(("bf16", ref_kind, shape))
        for qt in qts:
            t = results.get((qt, kind, shape))
            if t is None:
                row += f"{'-':>18}"
                continue
            is_self_ref = qt == "bf16" and kind == ref_kind
            sp = f" ({ref / t:4.2f}x)" if ref and not is_self_ref else " " * 8
            row += f"{t:8.3f}ms{sp}"
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps"],
                    help="accelerator to run on (default: cuda if present, else mps)")
    ap.add_argument("--gpu", type=int, default=0, help="cuda device id to run on")
    ap.add_argument("--qtypes", nargs="+", default=None, help=f"subset of {QTYPES}")
    ap.add_argument("--iters", type=int, default=50, help="timing iterations per case")
    args = ap.parse_args()

    device = pick_device(args.device, args.gpu)
    print(f"device: {describe_device(device)}")
    print(f"torch {torch.__version__}")

    explicit_qtypes = args.qtypes is not None
    qtypes = args.qtypes if explicit_qtypes else list(QTYPES)
    if device.type == "mps":
        blocked = [qt for qt in qtypes if qt in MPS_UNSUPPORTED]
        if blocked and not explicit_qtypes:
            qtypes = [qt for qt in qtypes if qt not in MPS_UNSUPPORTED]
            print(f"note: skipping {', '.join(blocked)} — no fp8 dtype on MPS")
        elif blocked:
            # asked for by name: try anyway, but say what is expected to happen
            print(f"note: {', '.join(blocked)} need an fp8 dtype MPS does not have; "
                  "expect them to fail or to silently stay in bf16")
        print("note: MPS vram peaks are sampled, not exact; convrot backends run "
              "their W8A16 fallback (no int8/fp4 tensor cores)")
    print()

    # warm the toolkit import chain (module imports + custom-op registration) so it
    # isn't charged to the first qtype's quantize timing
    if any(qt != "bf16" for qt in qtypes):
        from toolkit.util.ostris_quant import get_ostris_quantizer
        for qt in qtypes:
            if qt != "bf16":
                get_ostris_quantizer(qt)

    # many distinct module instances share one forward code object; the default
    # per-code cache limit (8) would silently fall back to eager and corrupt the
    # compiled columns
    torch._dynamo.config.cache_size_limit = 4096

    results = {}
    ran = []
    for qt in qtypes:
        print(f"benchmarking {qt} ...")
        torch._dynamo.reset()  # drop the previous qtype's compiled artifacts
        try:
            run_quality_and_quantize_time(qt, device, results)
            run_drift(qt, device, results)
            run_speed(qt, device, args.iters, results)
            run_vram(qt, device, results)
        except Exception as e:
            # one unsupported backend should not cost the whole run
            print(f"  [{qt}] FAILED, dropped from the tables: {type(e).__name__}: {e}")
            for key in list(results):
                if key[0] == qt:
                    del results[key]
            empty_cache(device)
            continue
        if qt != "bf16" and results.get((qt, "weight_err")) == 0.0:
            print(f"  [{qt}] quantization was a no-op (weights unchanged) — "
                  "dropped so it is not reported as a quantized result")
            for key in list(results):
                if key[0] == qt:
                    del results[key]
            continue
        ran.append(qt)

    if not ran:
        raise SystemExit("no qtype completed successfully")

    qts = ran
    print_speed_table("layer latency, inference", "inf", qts, results)
    print_speed_table("layer latency, inference (compiled)", "inf_comp", qts, results)
    print_speed_table("layer latency, train fwd+bwd", "train", qts, results)
    print_speed_table("layer latency, train fwd+bwd (compiled)", "train_comp", qts, results)

    print(f"\n=== vram on the block stack ({VRAM_BLOCKS} blocks, {VRAM_TOKENS} tokens) ===")
    print(f"{'':<28}" + "".join(f"{qt:>18}" for qt in qts))
    for key, label in (("vram_weights", "weights resident"),
                       ("vram_fwd_peak", "peak, no-grad fwd"),
                       ("vram_train_peak", "peak, train step"),
                       ("vram_train_ckpt_peak", "peak, train step (ckpt)"),
                       ("vram_fwd_peak_comp", "peak, no-grad fwd (comp)"),
                       ("vram_train_ckpt_peak_comp", "peak, train ckpt (comp)")):
        row = f"{label:<28}"
        for qt in qts:
            v = results.get((qt, key))
            row += f"{gb(v):>18}" if v is not None else f"{'-':>18}"
        print(row)

    print("\n=== accuracy drift vs bf16 (output rel err, no-grad) ===")
    print(f"{'':<28}" + "".join(f"{qt:>18}" for qt in qts))
    for shape in SPEED_SHAPES + [STACK_KEY]:
        label = f"{shape[0]} x {shape[1]} -> {shape[2]}" if isinstance(shape, tuple) else shape
        row = f"{label:<28}"
        for qt in qts:
            row += f"{results[(qt, 'drift', shape)]:>18.5f}"
        print(row)

    print("\n=== quantization ===")
    print(f"{'':<28}" + "".join(f"{qt:>18}" for qt in qts))
    row = f"{'weight rel err':<28}"
    for qt in qts:
        row += f"{results[(qt, 'weight_err')]:>18.5f}"
    print(row)
    row = f"{'quantize time (ms)':<28}"
    for qt in qts:
        row += f"{results[(qt, 'quantize_ms')]:>18.1f}"
    print(row)

    # ---- clean per-qtype breakdown: speed (geomean over shapes) + accuracy ----
    def geomean_speedup(qt, kind):
        # every speedup references EAGER bf16 (compiled kinds included), so the
        # comp columns answer "what do I gain over plain bf16"
        ref_kind = kind.removesuffix("_comp")
        logs = []
        for shape in SPEED_SHAPES:
            ref = results.get(("bf16", ref_kind, shape))
            t = results.get((qt, kind, shape))
            if ref and t:
                logs.append(math.log(ref / t))
        return math.exp(sum(logs) / len(logs)) if logs else None

    def fmt_speed(v):
        return f"{v:.2f}x" if v is not None else "-"

    print("\n=== summary (speed = geomean speedup vs bf16; drift lower is better) ===")
    print(f"{'':<18}{'inference':>12}{'inference comp':>16}{'train':>12}{'train comp':>12}"
          f"{'accuracy drift':>16}{'max vram':>12}{'max vram comp':>15}")
    for qt in qts:
        # real training checkpoints, so the ckpt peak is the meaningful train
        # number; the no-grad fwd peak still matters for sampling
        max_vram = max(results[(qt, "vram_fwd_peak")], results[(qt, "vram_train_ckpt_peak")])
        fwd_c = results.get((qt, "vram_fwd_peak_comp"))
        ckpt_c = results.get((qt, "vram_train_ckpt_peak_comp"))
        max_vram_comp = max(fwd_c, ckpt_c) if fwd_c is not None and ckpt_c is not None else None
        print(f"{qt:<18}"
              f"{fmt_speed(geomean_speedup(qt, 'inf')):>12}"
              f"{fmt_speed(geomean_speedup(qt, 'inf_comp')):>16}"
              f"{fmt_speed(geomean_speedup(qt, 'train')):>12}"
              f"{fmt_speed(geomean_speedup(qt, 'train_comp')):>12}"
              f"{results[(qt, 'drift', STACK_KEY)]:>16.5f}"
              f"{gb(max_vram).strip():>12}"
              f"{(gb(max_vram_comp).strip() if max_vram_comp is not None else '-'):>15}")


if __name__ == "__main__":
    main()
