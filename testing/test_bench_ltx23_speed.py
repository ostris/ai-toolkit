import tempfile
import unittest
from pathlib import Path

from testing.bench_ltx23_speed import assert_ltx_config, build_run_config, extract_runtime_metadata


class BenchHarnessTests(unittest.TestCase):
    def test_extract_runtime_metadata_parses_profile_and_gpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            log_path.write_text(
                "Throughput profile resolved to ltx23_ultra_vram for GPU 'NVIDIA RTX Pro 6000 Blackwell' (96.0 GB)\n",
                encoding="utf-8",
            )
            meta = extract_runtime_metadata(log_path)
            self.assertEqual(meta["resolved_profile"], "ltx23_ultra_vram")
            self.assertIn("RTX Pro 6000", meta["gpu_name"])
            self.assertEqual(meta["gpu_vram_gb"], "96.0")

    def test_assert_ltx_config_rejects_non_ltx_process(self):
        cfg = {
            "job": "extension",
            "config": {
                "process": [
                    {
                        "type": "diffusion_trainer",
                        "model": {"arch": "flux", "name_or_path": "black-forest-labs/FLUX.1-dev"},
                    }
                ]
            },
        }
        with self.assertRaises(ValueError):
            assert_ltx_config(cfg, Path("/tmp/non_ltx.yaml"))

    def test_build_run_config_isolates_runs_and_disables_save_sample(self):
        base = {
            "job": "extension",
            "config": {
                "name": "orig_job",
                "process": [
                    {
                        "training_folder": "output",
                        "train": {"steps": 1000},
                        "save": {"save_every": 250},
                        "sample": {"sample_every": 250},
                        "logging": {"log_every": 10},
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = build_run_config(
                base_config=base,
                label="candidate",
                run_index=2,
                run_output_root=Path(tmpdir),
                override_steps=123,
                disable_save_sample=True,
            )
            proc = cfg["config"]["process"][0]
            self.assertIn("ltx23_bench_candidate_run_02", cfg["config"]["name"])
            self.assertEqual(proc["train"]["steps"], 123)
            self.assertTrue(proc["train"]["disable_sampling"])
            self.assertEqual(proc["save"]["save_every"], 0)
            self.assertEqual(proc["sample"]["sample_every"], 0)
            self.assertEqual(proc["logging"]["log_every"], 0)
            self.assertNotEqual(proc["training_folder"], "output")


if __name__ == "__main__":
    unittest.main()
