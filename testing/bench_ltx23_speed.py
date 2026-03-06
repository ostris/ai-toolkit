#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_steps(config: dict) -> int:
    return int(config["config"]["process"][0]["train"]["steps"])


def is_ltx_process(process_cfg: dict) -> bool:
    model = process_cfg.get("model", {}) if isinstance(process_cfg, dict) else {}
    arch = str(model.get("arch", "")).strip().lower()
    name_or_path = str(model.get("name_or_path", "")).strip().lower()
    return arch.startswith("ltx2") or ("ltx-2.3" in name_or_path)


def assert_ltx_config(config: dict, config_path: Path):
    process_list = config.get("config", {}).get("process", [])
    if not isinstance(process_list, list) or len(process_list) == 0:
        raise ValueError(f"Invalid config format: missing config.process in {config_path}")
    non_ltx_indices = []
    for idx, process_cfg in enumerate(process_list):
        process_type = str(process_cfg.get("type", "")).strip().lower()
        if process_type not in {"sd_trainer", "diffusion_trainer", "ui_trainer"}:
            continue
        if not is_ltx_process(process_cfg):
            non_ltx_indices.append(idx)
    if non_ltx_indices:
        raise ValueError(
            f"Benchmark config must target LTX-2.3 training. Non-LTX process indices in {config_path}: {non_ltx_indices}"
        )


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    idx = int(round((len(sorted_vals) - 1) * q))
    return float(sorted_vals[max(0, min(idx, len(sorted_vals) - 1))])


def extract_runtime_metadata(log_path: Path) -> Dict[str, str]:
    metadata = {
        "resolved_profile": "",
        "gpu_name": "",
        "gpu_vram_gb": "",
    }
    if not log_path.exists():
        return metadata
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return metadata

    # Example:
    # Throughput profile resolved to ltx23_ultra_vram for GPU 'NVIDIA RTX Pro 6000' (96.0 GB)
    match = re.search(
        r"Throughput profile resolved to ([A-Za-z0-9_]+)\s+for GPU '([^']+)'\s+\(([0-9.]+)\s+GB\)",
        text,
    )
    if match:
        metadata["resolved_profile"] = match.group(1)
        metadata["gpu_name"] = match.group(2)
        metadata["gpu_vram_gb"] = match.group(3)
    return metadata


def build_run_config(
    base_config: dict,
    label: str,
    run_index: int,
    run_output_root: Path,
    override_steps: Optional[int],
    disable_save_sample: bool,
) -> dict:
    config = copy.deepcopy(base_config)
    process = config["config"]["process"][0]
    train_cfg = process.setdefault("train", {})

    # Isolate each run to avoid checkpoint resumption/collision across runs.
    run_name = f"ltx23_bench_{label}_run_{run_index:02d}"
    config["config"]["name"] = run_name
    process["training_folder"] = str((run_output_root / "training").resolve())

    if override_steps is not None:
        train_cfg["steps"] = int(override_steps)

    if disable_save_sample:
        train_cfg["disable_sampling"] = True
        process.setdefault("sample", {})["sample_every"] = 0
        process.setdefault("save", {})["save_every"] = 0

    # Keep runtime stable and reduce non-training overhead.
    train_cfg.setdefault("gradient_accumulation", 1)
    process.setdefault("logging", {})["log_every"] = 0
    return config


def write_run_config(config: dict, config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_job(run_py: Path, python_bin: str, config_path: Path, log_path: Path, env: Dict[str, str]) -> float:
    cmd = [python_bin, str(run_py), str(config_path), "--log", str(log_path)]
    started = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(run_py.parent), env=env)
    return time.perf_counter() - started


def run_suite(
    label: str,
    run_py: Path,
    python_bin: str,
    config_path: Path,
    repeats: int,
    warmup_runs: int,
    output_dir: Path,
    env: Dict[str, str],
    override_steps: Optional[int],
    disable_save_sample: bool,
) -> dict:
    cfg = load_config(config_path)
    assert_ltx_config(cfg, config_path)
    steps = get_steps(cfg)
    if override_steps is not None:
        steps = int(override_steps)
    requested_profile = cfg["config"]["process"][0]["train"].get("throughput_profile", "auto")
    elapsed_all: List[float] = []
    iter_sec_all: List[float] = []
    resolved_profiles: List[str] = []
    gpu_names: List[str] = []
    gpu_vram_gb_values: List[float] = []
    run_output_root = output_dir / f"{label}_artifacts"
    run_output_root.mkdir(parents=True, exist_ok=True)

    for idx in range(repeats + warmup_runs):
        run_cfg = build_run_config(
            base_config=cfg,
            label=label,
            run_index=idx,
            run_output_root=run_output_root,
            override_steps=override_steps,
            disable_save_sample=disable_save_sample,
        )
        run_config_path = output_dir / "configs" / f"{label}_run_{idx:02d}.yaml"
        write_run_config(run_cfg, run_config_path)

        log_path = output_dir / f"{label}_run_{idx:02d}.log"
        elapsed = run_job(run_py, python_bin, run_config_path, log_path, env)
        metadata = extract_runtime_metadata(log_path)
        if metadata["resolved_profile"]:
            resolved_profiles.append(metadata["resolved_profile"])
        if metadata["gpu_name"]:
            gpu_names.append(metadata["gpu_name"])
        if metadata["gpu_vram_gb"]:
            try:
                gpu_vram_gb_values.append(float(metadata["gpu_vram_gb"]))
            except Exception:
                pass
        if idx < warmup_runs:
            continue
        elapsed_all.append(elapsed)
        iter_sec_all.append(steps / elapsed if elapsed > 0 else 0.0)

    return {
        "label": label,
        "config_path": str(config_path),
        "steps": steps,
        "requested_throughput_profile": requested_profile,
        "runs": len(iter_sec_all),
        "elapsed_seconds": elapsed_all,
        "iter_per_sec": iter_sec_all,
        "median_iter_per_sec": statistics.median(iter_sec_all) if iter_sec_all else 0.0,
        "mean_iter_per_sec": statistics.mean(iter_sec_all) if iter_sec_all else 0.0,
        "p90_iter_per_sec": quantile(iter_sec_all, 0.9),
        "resolved_profiles": sorted(set(resolved_profiles)),
        "gpu_names": sorted(set(gpu_names)),
        "median_gpu_vram_gb": statistics.median(gpu_vram_gb_values) if gpu_vram_gb_values else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic LTX-2.3 throughput A/B benchmark runner.")
    parser.add_argument("--baseline-config", required=True, help="Path to baseline config yaml/json.")
    parser.add_argument("--candidate-config", required=True, help="Path to candidate config yaml/json.")
    parser.add_argument("--repeats", type=int, default=3, help="Measured runs per config.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Unmeasured warmup runs per config.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for launching run.py")
    parser.add_argument("--output-json", default=None, help="Optional path to write JSON results.")
    parser.add_argument("--output-dir", default="output/benchmarks", help="Directory for run logs.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional override for train.steps in both baseline and candidate configs.",
    )
    parser.add_argument(
        "--disable-save-sample",
        action="store_true",
        help="Disable save/sample events during benchmark runs for cleaner throughput measurement.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_py = repo_root / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found at {run_py}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    baseline = run_suite(
        label="baseline",
        run_py=run_py,
        python_bin=args.python_bin,
        config_path=Path(args.baseline_config).resolve(),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        output_dir=output_dir,
        env=env,
        override_steps=args.steps,
        disable_save_sample=args.disable_save_sample,
    )
    candidate = run_suite(
        label="candidate",
        run_py=run_py,
        python_bin=args.python_bin,
        config_path=Path(args.candidate_config).resolve(),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        output_dir=output_dir,
        env=env,
        override_steps=args.steps,
        disable_save_sample=args.disable_save_sample,
    )

    base_median = baseline["median_iter_per_sec"]
    cand_median = candidate["median_iter_per_sec"]
    gain_pct = ((cand_median - base_median) / base_median * 100.0) if base_median > 0 else 0.0

    result = {
        "baseline": baseline,
        "candidate": candidate,
        "median_gain_percent": gain_pct,
    }

    output_json = Path(args.output_json).resolve() if args.output_json else output_dir / "benchmark_result.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nWrote benchmark report to: {output_json}")


if __name__ == "__main__":
    main()
