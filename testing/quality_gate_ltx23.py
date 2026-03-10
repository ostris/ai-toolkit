#!/usr/bin/env python3
import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional


def load_metric_series(db_path: Path, key: str):
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT m.step, m.value_real
            FROM metrics m
            WHERE m.key = ?
            ORDER BY m.step ASC
            """,
            (key,),
        )
        rows = [(int(step), float(value)) for step, value in cur.fetchall() if value is not None]
        return rows
    finally:
        con.close()


def rolling_tail_mean(series, window: int = 50) -> Optional[float]:
    if not series:
        return None
    values = [v for _, v in series[-window:]]
    if not values:
        return None
    return sum(values) / float(len(values))


def rel_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = max(abs(a), 1e-12)
    return abs(b - a) / denom


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 quality gate for throughput-profile promotion.")
    parser.add_argument("--profile", required=True, choices=["ltx23_safe", "ltx23_max", "ltx23_ultra_vram"])
    parser.add_argument("--benchmark-json", required=True, help="Path to benchmark_result.json from bench_ltx23_speed.py")
    parser.add_argument("--baseline-db", default=None, help="Optional baseline loss_log.db")
    parser.add_argument("--candidate-db", default=None, help="Optional candidate loss_log.db")
    parser.add_argument(
        "--sample-drift-json",
        default=None,
        help="Optional JSON containing sample drift metric (expects key 'mean_drift').",
    )
    parser.add_argument("--safe-loss-delta", type=float, default=1e-6)
    parser.add_argument("--max-loss-delta", type=float, default=0.025)
    parser.add_argument("--max-audio-video-ratio-delta", type=float, default=0.05)
    parser.add_argument("--max-sample-drift", type=float, default=0.05)
    parser.add_argument(
        "--expected-gpu-substring",
        default=None,
        help="Optional substring expected in candidate GPU name metadata from benchmark JSON.",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_json).resolve()
    with benchmark_path.open("r", encoding="utf-8") as f:
        bench = json.load(f)

    gain_pct = float(bench.get("median_gain_percent", 0.0))
    checks = []
    candidate_meta = bench.get("candidate", {}) if isinstance(bench, dict) else {}
    candidate_profiles = set(candidate_meta.get("resolved_profiles", []) or [])
    candidate_gpu_names = [str(x) for x in (candidate_meta.get("gpu_names", []) or [])]

    checks.append(
        (
            "resolved_profile_match",
            args.profile in candidate_profiles if candidate_profiles else False,
            f"expected_profile={args.profile}, observed_profiles={sorted(candidate_profiles)}",
        )
    )
    if args.expected_gpu_substring:
        expected_lower = str(args.expected_gpu_substring).strip().lower()
        observed = any(expected_lower in name.lower() for name in candidate_gpu_names)
        checks.append(
            (
                "gpu_identity",
                observed,
                f"expected_substring='{args.expected_gpu_substring}', observed_gpu_names={candidate_gpu_names}",
            )
        )

    if args.profile == "ltx23_max":
        checks.append(("throughput_gain", gain_pct >= 50.0, f"gain={gain_pct:.2f}% target>=50%"))
    elif args.profile == "ltx23_ultra_vram":
        checks.append(("throughput_gain", gain_pct >= 60.0, f"gain={gain_pct:.2f}% target>=60%"))
    else:
        checks.append(("throughput_non_regression", gain_pct >= 0.0, f"gain={gain_pct:.2f}% target>=0%"))

    baseline_db = Path(args.baseline_db).resolve() if args.baseline_db else None
    candidate_db = Path(args.candidate_db).resolve() if args.candidate_db else None
    if baseline_db and candidate_db:
        base_loss = rolling_tail_mean(load_metric_series(baseline_db, "loss/loss"), window=50)
        cand_loss = rolling_tail_mean(load_metric_series(candidate_db, "loss/loss"), window=50)
        loss_delta = rel_delta(base_loss, cand_loss)

        if args.profile == "ltx23_safe":
            passed = (loss_delta is not None) and (loss_delta <= args.safe_loss_delta)
            checks.append(
                (
                    "strict_loss_parity",
                    passed,
                    f"loss_delta={loss_delta if loss_delta is not None else 'n/a'} target<={args.safe_loss_delta}",
                )
            )
        else:
            passed = (loss_delta is not None) and (loss_delta <= args.max_loss_delta)
            checks.append(
                (
                    "bounded_loss_delta",
                    passed,
                    f"loss_delta={loss_delta if loss_delta is not None else 'n/a'} target<={args.max_loss_delta}",
                )
            )

            base_audio = rolling_tail_mean(load_metric_series(baseline_db, "loss/audio"), window=50)
            base_video = rolling_tail_mean(load_metric_series(baseline_db, "loss/video"), window=50)
            cand_audio = rolling_tail_mean(load_metric_series(candidate_db, "loss/audio"), window=50)
            cand_video = rolling_tail_mean(load_metric_series(candidate_db, "loss/video"), window=50)
            if None not in (base_audio, base_video, cand_audio, cand_video) and base_video not in (0.0, None) and cand_video not in (0.0, None):
                base_ratio = float(base_audio) / float(base_video)
                cand_ratio = float(cand_audio) / float(cand_video)
                ratio_delta = rel_delta(base_ratio, cand_ratio)
                checks.append(
                    (
                        "audio_video_ratio_delta",
                        (ratio_delta is not None) and (ratio_delta <= args.max_audio_video_ratio_delta),
                        f"ratio_delta={ratio_delta if ratio_delta is not None else 'n/a'} "
                        f"target<={args.max_audio_video_ratio_delta}",
                    )
                )
            else:
                checks.append(("audio_video_ratio_delta", False, "audio/video metrics missing in one or both DBs"))
    else:
        checks.append(("quality_db_present", False, "baseline/candidate DB paths were not provided"))

    if args.profile != "ltx23_safe":
        if args.sample_drift_json:
            drift_path = Path(args.sample_drift_json).resolve()
            if drift_path.exists():
                with drift_path.open("r", encoding="utf-8") as f:
                    drift_payload = json.load(f)
                drift = float(drift_payload.get("mean_drift", 999.0))
                checks.append(
                    (
                        "sample_drift",
                        drift <= args.max_sample_drift,
                        f"mean_drift={drift:.6f} target<={args.max_sample_drift}",
                    )
                )
            else:
                checks.append(("sample_drift", False, f"sample drift file not found: {drift_path}"))
        else:
            checks.append(("sample_drift", False, "sample drift input missing (--sample-drift-json)"))

    passed = all(item[1] for item in checks)
    result = {
        "profile": args.profile,
        "benchmark_json": str(benchmark_path),
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "passed": passed,
    }
    print(json.dumps(result, indent=2))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
