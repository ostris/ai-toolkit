import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def write_metrics_db(path: Path, loss: float, audio: float, video: float):
    con = sqlite3.connect(str(path))
    try:
        cur = con.cursor()
        cur.execute("CREATE TABLE metrics (step INTEGER, key TEXT, value_real REAL)")
        # Small tail window with stable values.
        for step in range(1, 61):
            cur.execute("INSERT INTO metrics (step, key, value_real) VALUES (?, ?, ?)", (step, "loss/loss", loss))
            cur.execute("INSERT INTO metrics (step, key, value_real) VALUES (?, ?, ?)", (step, "loss/audio", audio))
            cur.execute("INSERT INTO metrics (step, key, value_real) VALUES (?, ?, ?)", (step, "loss/video", video))
        con.commit()
    finally:
        con.close()


class QualityGateTests(unittest.TestCase):
    def test_quality_gate_passes_with_matching_profile_and_gpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = root / "benchmark.json"
            baseline_db = root / "baseline.db"
            candidate_db = root / "candidate.db"
            sample_drift = root / "sample_drift.json"

            benchmark.write_text(
                json.dumps(
                    {
                        "median_gain_percent": 62.0,
                        "candidate": {
                            "resolved_profiles": ["ltx23_ultra_vram"],
                            "gpu_names": ["NVIDIA RTX Pro 6000 Blackwell"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_metrics_db(baseline_db, loss=1.0, audio=0.6, video=0.4)
            write_metrics_db(candidate_db, loss=1.01, audio=0.61, video=0.41)
            sample_drift.write_text(json.dumps({"mean_drift": 0.01}), encoding="utf-8")

            cmd = [
                sys.executable,
                "testing/quality_gate_ltx23.py",
                "--profile",
                "ltx23_ultra_vram",
                "--benchmark-json",
                str(benchmark),
                "--expected-gpu-substring",
                "RTX Pro 6000",
                "--baseline-db",
                str(baseline_db),
                "--candidate-db",
                str(candidate_db),
                "--sample-drift-json",
                str(sample_drift),
            ]
            proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)

    def test_quality_gate_fails_when_profile_does_not_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = root / "benchmark.json"
            baseline_db = root / "baseline.db"
            candidate_db = root / "candidate.db"
            sample_drift = root / "sample_drift.json"

            benchmark.write_text(
                json.dumps(
                    {
                        "median_gain_percent": 70.0,
                        "candidate": {
                            "resolved_profiles": ["ltx23_max"],
                            "gpu_names": ["NVIDIA RTX Pro 6000 Blackwell"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_metrics_db(baseline_db, loss=1.0, audio=0.6, video=0.4)
            write_metrics_db(candidate_db, loss=1.0, audio=0.6, video=0.4)
            sample_drift.write_text(json.dumps({"mean_drift": 0.0}), encoding="utf-8")

            cmd = [
                sys.executable,
                "testing/quality_gate_ltx23.py",
                "--profile",
                "ltx23_ultra_vram",
                "--benchmark-json",
                str(benchmark),
                "--baseline-db",
                str(baseline_db),
                "--candidate-db",
                str(candidate_db),
                "--sample-drift-json",
                str(sample_drift),
            ]
            proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()

