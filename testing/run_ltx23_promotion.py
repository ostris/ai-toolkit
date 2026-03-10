#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path):
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run LTX-2.3 benchmark + quality gate promotion flow.")
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--profile", required=True, choices=["ltx23_safe", "ltx23_max", "ltx23_ultra_vram"])
    parser.add_argument("--baseline-db", required=True)
    parser.add_argument("--candidate-db", required=True)
    parser.add_argument("--sample-drift-json", required=False, default=None)
    parser.add_argument("--expected-gpu-substring", required=False, default=None)
    parser.add_argument("--output-dir", default="output/benchmarks")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--disable-save-sample", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    benchmark_json = Path(args.output_json).resolve() if args.output_json else (Path(args.output_dir).resolve() / "benchmark_result.json")

    bench_cmd = [
        args.python_bin,
        "testing/bench_ltx23_speed.py",
        "--baseline-config",
        args.baseline_config,
        "--candidate-config",
        args.candidate_config,
        "--repeats",
        str(args.repeats),
        "--warmup-runs",
        str(args.warmup_runs),
        "--output-dir",
        args.output_dir,
        "--output-json",
        str(benchmark_json),
    ]
    if args.steps is not None:
        bench_cmd.extend(["--steps", str(args.steps)])
    if args.disable_save_sample:
        bench_cmd.append("--disable-save-sample")
    run_cmd(bench_cmd, cwd=repo_root)

    gate_cmd = [
        args.python_bin,
        "testing/quality_gate_ltx23.py",
        "--profile",
        args.profile,
        "--benchmark-json",
        str(benchmark_json),
        "--baseline-db",
        args.baseline_db,
        "--candidate-db",
        args.candidate_db,
    ]
    if args.sample_drift_json:
        gate_cmd.extend(["--sample-drift-json", args.sample_drift_json])
    if args.expected_gpu_substring:
        gate_cmd.extend(["--expected-gpu-substring", args.expected_gpu_substring])
    run_cmd(gate_cmd, cwd=repo_root)

    print("\nPromotion checks passed.")


if __name__ == "__main__":
    main()

