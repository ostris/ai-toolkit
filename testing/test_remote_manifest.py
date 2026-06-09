"""Tests for scripts/remote/manifest.py and scripts/remote/contract.py (U1).

Run directly: python testing/test_remote_manifest.py
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.remote import contract, manifest


class TestManifestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="aitk-manifest-test-")

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _full_manifest(self):
        return manifest.RunManifest(
            run_name="balfua_v3",
            state=contract.RunState.RUNNING.value,
            ssh_host="1.2.3.4",
            ssh_port=10022,
            pod_id="pod-abc123",
            image_tag="ostris/aitoolkit:0.9.11",
            image_repo_commit="deadbeef",
            gpu_requested="A100 80GB",
            gpu_provisioned="A100 80GB",
            hourly_rate=1.89,
            disk_gb=90,
            provisioned_at=1000.0,
            launched_at=1600.0,
            max_runtime_deadline=87400.0,
            max_grace_seconds=1800,
            tmux_session="aitk-train-balfua_v3",
            timer_session="aitk-timer-balfua_v3",
            job_name="balfua_v3",
            config_hash="ff" * 32,
            overlay_git_sha="cafe1234",
            overlay_dirty_hash="00" * 8,
            dataset_file_count=42,
            dataset_total_bytes=123456789,
            total_steps=3000,
            save_every=250,
            sample_every=250,
            prompt_count=12,
            last_pulled_sample_step=500,
            last_pulled_checkpoint_step=250,
            optimizer_pairing_step=250,
            last_reviewed_step=250,
            last_sentinel=None,
        )

    def test_round_trip_all_fields(self):
        m = self._full_manifest()
        manifest.save(m, base_dir=self.base)
        loaded = manifest.load("balfua_v3", base_dir=self.base)
        self.assertEqual(m.to_dict(), loaded.to_dict())

    def test_unknown_fields_survive(self):
        m = self._full_manifest()
        path = manifest.save(m, base_dir=self.base)
        with open(path) as f:
            data = json.load(f)
        data["future_feature_flag"] = {"nested": True}
        with open(path, "w") as f:
            json.dump(data, f)
        loaded = manifest.load("balfua_v3", base_dir=self.base)
        self.assertEqual(loaded.extra["future_feature_flag"], {"nested": True})
        # and it survives a re-save
        manifest.save(loaded, base_dir=self.base)
        reloaded = manifest.load("balfua_v3", base_dir=self.base)
        self.assertEqual(reloaded.extra["future_feature_flag"], {"nested": True})

    def test_missing_manifest_error_names_run_and_path(self):
        with self.assertRaises(manifest.ManifestNotFoundError) as ctx:
            manifest.load("nope", base_dir=self.base)
        self.assertIn("nope", str(ctx.exception))
        self.assertIn("manifest.json", str(ctx.exception))

    def test_atomic_write_failure_preserves_original(self):
        m = self._full_manifest()
        manifest.save(m, base_dir=self.base)
        m.state = contract.RunState.CRASHED.value
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                manifest.save(m, base_dir=self.base)
        loaded = manifest.load("balfua_v3", base_dir=self.base)
        self.assertEqual(loaded.state, contract.RunState.RUNNING.value)
        # no stray temp files left behind
        leftovers = [f for f in os.listdir(os.path.dirname(manifest.manifest_path("balfua_v3", self.base)))
                     if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_cost_derivation(self):
        m = manifest.RunManifest(run_name="r", hourly_rate=1.89, provisioned_at=0.0)
        self.assertAlmostEqual(m.estimated_cost(now=7200.0), 3.78, places=2)
        m.terminated_at = 3600.0
        self.assertAlmostEqual(m.estimated_cost(now=999999.0), 1.89, places=2)

    def test_expected_checkpoint_count_uses_observed_step(self):
        m = manifest.RunManifest(run_name="r", total_steps=5000, save_every=250)
        self.assertEqual(m.expected_checkpoint_count(observed_max_step=1500), 6)
        self.assertEqual(m.expected_checkpoint_count(observed_max_step=5000), 20)


class TestContractGrammars(unittest.TestCase):
    def test_sample_filename_parses_step(self):
        self.assertEqual(contract.parse_sample_filename("1768412345678__000000250_0.jpg"), (250, 0))
        self.assertEqual(contract.parse_sample_filename("1768412345678__000001500_11.webp"), (1500, 11))
        self.assertIsNone(contract.parse_sample_filename("not_a_sample.jpg"))
        self.assertIsNone(contract.parse_sample_filename("1768412345678_000000250_0.jpg"))  # single underscore

    def test_checkpoint_filename_parses_step(self):
        self.assertEqual(contract.parse_checkpoint_filename("balfua_v3_000001500.safetensors", "balfua_v3"), 1500)
        self.assertIsNone(contract.parse_checkpoint_filename("other_000001500.safetensors", "balfua_v3"))
        self.assertIsNone(contract.parse_checkpoint_filename("balfua_v3.safetensors", "balfua_v3"))

    def test_final_checkpoint_recognized(self):
        self.assertTrue(contract.is_final_checkpoint("balfua_v3.safetensors", "balfua_v3"))
        self.assertFalse(contract.is_final_checkpoint("balfua_v3_000001500.safetensors", "balfua_v3"))

    def test_run_name_validation(self):
        self.assertEqual(contract.validate_run_name("balfua_v3"), "balfua_v3")
        for bad in ["bad name; rm -rf", "x" * 70, "", "no/slash", "dollar$"]:
            with self.assertRaises(ValueError):
                contract.validate_run_name(bad)

    def test_shell_quote_escapes_single_quotes(self):
        quoted = contract.shell_quote("it's a path with spaces")
        self.assertTrue(quoted.startswith("'") and quoted.endswith("'"))
        self.assertNotIn("it's a", quoted.replace("'\"'\"'", ""))

    def test_disk_size_formula(self):
        # 3000/250 = 12 saves -> 60 + 12*0.6 + 20 = 87.2 -> 88
        self.assertEqual(contract.disk_size_gb(3000, 250), 88)


class TestContractLogMatchers(unittest.TestCase):
    TQDM = "flux2_dev_sulk1an_style_v6:  22%|##2       | 865/4000 [3:21:26<12:10:03, 13.97s/it]"
    OOM = "# OOM during training step, skipping batch 1/3 #"
    TB = 'Traceback (most recent call last):\n  File "run.py", line 90, in <module>\n    raise ValueError("bad config")'
    WARM = "Fetching 23 files: 12%| | 3/23 [00:40<04:21, 13.0s/it]\nLoading checkpoint shards"

    def test_progress_matches_tqdm_cr_records(self):
        # \r-separated tqdm updates, as --log tees them verbatim
        text = "setup\r" + self.TQDM + "\r" + self.TQDM
        self.assertEqual(contract.classify_log_tail(text), "progress")

    def test_traceback_wins_over_progress(self):
        self.assertEqual(contract.classify_log_tail(self.TQDM + "\n" + self.TB), "traceback")

    def test_warming_detected(self):
        # download-progress lines also look tqdm-ish; a pure loading line is warming
        self.assertEqual(contract.classify_log_tail("Loading checkpoint shards"), "warming")

    def test_silent(self):
        self.assertEqual(contract.classify_log_tail("starting up\nreading config"), "silent")

    def test_oom_count(self):
        text = self.TQDM + "\n" + self.OOM + "\r" + self.TQDM + "\n" + self.OOM + "\n" + self.OOM
        self.assertEqual(contract.count_oom_skips(text), 3)
        self.assertEqual(contract.classify_log_tail(text), "progress")  # OOM-skip alone isn't terminal


class TestRunStates(unittest.TestCase):
    def test_detached_is_not_a_persistable_state(self):
        self.assertFalse(hasattr(contract.RunState, "DETACHED"))

    def test_watch_exit_codes_distinct(self):
        codes = list(contract.WATCH_EXIT_CODES.values())
        self.assertEqual(len(codes), len(set(codes)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
