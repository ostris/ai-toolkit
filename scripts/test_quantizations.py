"""
Reproducible speed / VRAM / accuracy benchmarks for the toolkit quantization
backends.

Compares bf16 against the custom OstrisLinear backends (convrot8, convrot4 for
now; add more qtypes to QTYPES as they land).

Measures, per qtype:
  - layer inference latency across DiT-representative shapes (vs bf16)
  - layer training latency (forward + backward through the frozen layer)
  - VRAM on a transformer-ish block stack: resident weights, peak during a
    no-grad forward, peak during a train step
  - accuracy drift vs bf16: output relative error per layer shape and
    accumulated through the block stack
  - weight reconstruction error and one-time quantize (conversion) time

Usage:
    python scripts/test_quantizations.py --gpu 1
    python scripts/test_quantizations.py --gpu 1 --qtypes bf16 convrot8
"""

import argparse
import math
import os
import sys
import time

# set cuda bus ordering to be pcie
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

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
    "bf16", "qfloat8", "float8", "orbit4", "orbitvq4", "convrot8", "convrot4",
    "convrotint7", "convrotint6", "convrotint5", "convrotint4", "convrotint3",
    "convrotint2", "convrotbitnet",
]

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
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters * 1000  # ms


def gb(nbytes: int) -> str:
    return f"{nbytes / 1e9:6.2f} GB"


def make_layer(k: int, n: int, device) -> torch.nn.Linear:
    lin = torch.nn.Linear(k, n, bias=True, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        lin.weight.mul_(0.02)
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
        torch.cuda.empty_cache()


def run_vram(qtype: str, device, results: dict):
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)

    blocks = make_stack(device)
    for b in blocks:
        for i in range(len(b)):
            b[i] = convert(b[i], qtype)
    torch.cuda.empty_cache()
    results[(qtype, "vram_weights")] = torch.cuda.memory_allocated(device) - base

    x = torch.randn(VRAM_TOKENS, 3072, device=device, dtype=torch.bfloat16)

    # no-grad forward peak (sampling); warm up first so lazy-init allocations are
    # not counted as steady-state peak
    with torch.no_grad():
        stack_forward(blocks, x)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        stack_forward(blocks, x)
    torch.cuda.synchronize(device)
    results[(qtype, "vram_fwd_peak")] = torch.cuda.max_memory_allocated(device) - base

    # train step peak (frozen base; grads flow to the input like lora training),
    # with and without per-block gradient checkpointing (real training uses it)
    def train_step(checkpoint):
        xi = x.detach().requires_grad_(True)
        stack_forward(blocks, xi, checkpoint).float().pow(2).mean().backward()

    for key, ckpt in (("vram_train_peak", False), ("vram_train_ckpt_peak", True)):
        train_step(ckpt)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        train_step(ckpt)
        torch.cuda.synchronize(device)
        results[(qtype, key)] = torch.cuda.max_memory_allocated(device) - base

    blocks = x = None  # release before the allocator accounting of the next run
    torch.cuda.empty_cache()


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
        torch.cuda.empty_cache()

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
    torch.cuda.empty_cache()


def run_quality_and_quantize_time(qtype: str, device, results: dict):
    torch.manual_seed(0)
    lin = make_layer(3072, 3072, device)
    w0 = lin.weight.detach().float().clone()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    lin = convert(lin, qtype)
    torch.cuda.synchronize(device)
    results[(qtype, "quantize_ms")] = (time.perf_counter() - t0) * 1000
    if qtype == "bf16":
        results[(qtype, "weight_err")] = 0.0
    else:
        wq = fp_weight(lin)
        results[(qtype, "weight_err")] = ((wq - w0).norm() / w0.norm()).item()
    torch.cuda.empty_cache()


def print_speed_table(title: str, kind: str, qts, results):
    print(f"\n=== {title} (ms; speedup vs bf16) ===")
    print(f"{'M x K -> N':<22}" + "".join(f"{qt:>18}" for qt in qts))
    for shape in SPEED_SHAPES:
        m, k, n = shape
        row = f"{f'{m} x {k} -> {n}':<22}"
        ref = results.get(("bf16", kind, shape))
        for qt in qts:
            t = results.get((qt, kind, shape))
            if t is None:
                row += f"{'-':>18}"
                continue
            sp = f" ({ref / t:4.2f}x)" if ref and qt != "bf16" else " " * 8
            row += f"{t:8.3f}ms{sp}"
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0, help="cuda device id to run on")
    ap.add_argument("--qtypes", nargs="+", default=QTYPES, help=f"subset of {QTYPES}")
    ap.add_argument("--iters", type=int, default=50, help="timing iterations per case")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    print(f"device: cuda:{args.gpu} ({props.name}, sm_{props.major}{props.minor}, "
          f"{props.total_memory / 1e9:.0f} GB)")
    print(f"torch {torch.__version__}\n")

    # warm the toolkit import chain (module imports + custom-op registration) so it
    # isn't charged to the first qtype's quantize timing
    if any(qt != "bf16" for qt in args.qtypes):
        from toolkit.util.ostris_quant import get_ostris_quantizer
        for qt in args.qtypes:
            if qt != "bf16":
                get_ostris_quantizer(qt)

    results = {}
    for qt in args.qtypes:
        print(f"benchmarking {qt} ...")
        run_quality_and_quantize_time(qt, device, results)
        run_drift(qt, device, results)
        run_speed(qt, device, args.iters, results)
        run_vram(qt, device, results)

    qts = args.qtypes
    print_speed_table("layer latency, inference", "inf", qts, results)
    print_speed_table("layer latency, train fwd+bwd", "train", qts, results)

    print(f"\n=== vram on the block stack ({VRAM_BLOCKS} blocks, {VRAM_TOKENS} tokens) ===")
    print(f"{'':<28}" + "".join(f"{qt:>18}" for qt in qts))
    for key, label in (("vram_weights", "weights resident"),
                       ("vram_fwd_peak", "peak, no-grad fwd"),
                       ("vram_train_peak", "peak, train step"),
                       ("vram_train_ckpt_peak", "peak, train step (ckpt)")):
        row = f"{label:<28}"
        for qt in qts:
            row += f"{gb(results[(qt, key)]):>18}"
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
        logs = []
        for shape in SPEED_SHAPES:
            ref = results.get(("bf16", kind, shape))
            t = results.get((qt, kind, shape))
            if ref and t:
                logs.append(math.log(ref / t))
        return math.exp(sum(logs) / len(logs)) if logs else float("nan")

    print("\n=== summary (speed = geomean speedup vs bf16; drift lower is better) ===")
    print(f"{'':<12}{'inference':>18}{'train':>18}{'accuracy drift':>20}{'max vram':>16}")
    for qt in qts:
        # real training checkpoints, so the ckpt peak is the meaningful train
        # number; the no-grad fwd peak still matters for sampling
        max_vram = max(results[(qt, "vram_fwd_peak")], results[(qt, "vram_train_ckpt_peak")])
        print(f"{qt:<12}"
              f"{geomean_speedup(qt, 'inf'):>17.2f}x"
              f"{geomean_speedup(qt, 'train'):>17.2f}x"
              f"{results[(qt, 'drift', STACK_KEY)]:>20.5f}"
              f"{gb(max_vram):>16}")


if __name__ == "__main__":
    main()
