import unittest
from types import SimpleNamespace
from unittest import mock

from toolkit.throughput_profiles import (
    GPUCapability,
    apply_ltx23_throughput_profile,
    detect_gpu_capability,
    get_ltx23_profile_settings,
    resolve_ltx23_throughput_profile,
)


class ThroughputProfileTests(unittest.TestCase):
    def test_5090_like_maps_to_max(self):
        capability = GPUCapability(
            available=True,
            name="NVIDIA GeForce RTX 5090",
            total_vram_gb=32.0,
            sm_count=128,
            compute_capability=(9, 0),
        )
        self.assertEqual(resolve_ltx23_throughput_profile("auto", capability), "ltx23_max")

    def test_rtx_pro_6000_maps_to_ultra(self):
        capability = GPUCapability(
            available=True,
            name="NVIDIA RTX Pro 6000 Blackwell",
            total_vram_gb=96.0,
            sm_count=142,
            compute_capability=(9, 0),
        )
        self.assertEqual(resolve_ltx23_throughput_profile("auto", capability), "ltx23_ultra_vram")

    def test_pro6000_not_lower_than_5090(self):
        cap_5090 = GPUCapability(
            available=True,
            name="NVIDIA GeForce RTX 5090",
            total_vram_gb=32.0,
            sm_count=128,
            compute_capability=(9, 0),
        )
        cap_pro = GPUCapability(
            available=True,
            name="NVIDIA RTX Pro 6000 Blackwell",
            total_vram_gb=96.0,
            sm_count=142,
            compute_capability=(9, 0),
        )
        order = {"ltx23_safe": 0, "ltx23_max": 1, "ltx23_ultra_vram": 2}
        self.assertGreaterEqual(
            order[resolve_ltx23_throughput_profile("auto", cap_pro)],
            order[resolve_ltx23_throughput_profile("auto", cap_5090)],
        )

    def test_ultra_profile_has_no_lower_concurrency_than_max(self):
        max_cfg = get_ltx23_profile_settings("ltx23_max", cpu_count=32)
        ultra_cfg = get_ltx23_profile_settings("ltx23_ultra_vram", cpu_count=32)
        self.assertGreaterEqual(ultra_cfg["num_workers"], max_cfg["num_workers"])
        self.assertGreaterEqual(ultra_cfg["prefetch_queue_depth"], max_cfg["prefetch_queue_depth"])

    def test_apply_profile_autotunes_dataset_and_compile(self):
        train_cfg = SimpleNamespace(
            throughput_profile="ltx23_ultra_vram",
            dataloader_autotune=True,
            prefetch_to_device=True,
            prefetch_queue_depth=1,
            logger_commit_interval=5,
            allow_tf32=True,
            cudnn_benchmark=True,
            _prefetch_to_device_requested=False,
            _prefetch_queue_depth_requested=False,
            _logger_commit_interval_requested=False,
            _allow_tf32_requested=False,
            _cudnn_benchmark_requested=False,
        )
        model_cfg = SimpleNamespace(
            compile=False,
            compile_requested=False,
            compile_mode="max-autotune",
            compile_mode_requested=False,
            compile_dynamic=True,
            compile_dynamic_requested=False,
            compile_fullgraph=True,
            compile_fullgraph_requested=False,
            low_vram=True,
            low_vram_requested=False,
        )
        dataset = SimpleNamespace(num_workers=0, prefetch_factor=2, pin_memory=False, persistent_workers=False)
        resolved, _ = apply_ltx23_throughput_profile(train_cfg, model_cfg, [dataset], device=None)

        self.assertEqual(resolved, "ltx23_ultra_vram")
        self.assertTrue(model_cfg.compile)
        self.assertFalse(model_cfg.low_vram)
        self.assertTrue(dataset.pin_memory)
        self.assertTrue(dataset.persistent_workers)
        self.assertGreaterEqual(dataset.num_workers, 10)

    def test_ultra_profile_does_not_lower_existing_concurrency(self):
        train_cfg = SimpleNamespace(
            throughput_profile="ltx23_ultra_vram",
            dataloader_autotune=True,
            prefetch_to_device=True,
            prefetch_queue_depth=4,
            logger_commit_interval=20,
            allow_tf32=True,
            cudnn_benchmark=True,
            _prefetch_to_device_requested=False,
            _prefetch_queue_depth_requested=False,
            _logger_commit_interval_requested=False,
            _allow_tf32_requested=False,
            _cudnn_benchmark_requested=False,
        )
        model_cfg = SimpleNamespace(
            compile=False,
            compile_requested=False,
            compile_mode="max-autotune",
            compile_mode_requested=False,
            compile_dynamic=True,
            compile_dynamic_requested=False,
            compile_fullgraph=True,
            compile_fullgraph_requested=False,
            low_vram=False,
            low_vram_requested=False,
        )
        dataset = SimpleNamespace(num_workers=24, prefetch_factor=5, pin_memory=False, persistent_workers=False)
        resolved, _ = apply_ltx23_throughput_profile(train_cfg, model_cfg, [dataset], device=None)

        self.assertEqual(resolved, "ltx23_ultra_vram")
        self.assertEqual(dataset.num_workers, 24)
        self.assertEqual(dataset.prefetch_factor, 5)
        self.assertEqual(train_cfg.prefetch_queue_depth, 4)
        self.assertEqual(train_cfg.logger_commit_interval, 20)

    def test_ultra_profile_respects_explicit_low_vram_request(self):
        train_cfg = SimpleNamespace(
            throughput_profile="ltx23_ultra_vram",
            dataloader_autotune=True,
            prefetch_to_device=True,
            prefetch_queue_depth=1,
            logger_commit_interval=5,
            allow_tf32=True,
            cudnn_benchmark=True,
            _prefetch_to_device_requested=False,
            _prefetch_queue_depth_requested=False,
            _logger_commit_interval_requested=False,
            _allow_tf32_requested=False,
            _cudnn_benchmark_requested=False,
        )
        model_cfg = SimpleNamespace(
            compile=False,
            compile_requested=False,
            compile_mode="max-autotune",
            compile_mode_requested=False,
            compile_dynamic=True,
            compile_dynamic_requested=False,
            compile_fullgraph=True,
            compile_fullgraph_requested=False,
            low_vram=True,
            low_vram_requested=True,
        )
        dataset = SimpleNamespace(num_workers=0, prefetch_factor=2, pin_memory=False, persistent_workers=False)
        resolved, _ = apply_ltx23_throughput_profile(train_cfg, model_cfg, [dataset], device=None)

        self.assertEqual(resolved, "ltx23_ultra_vram")
        self.assertTrue(model_cfg.low_vram)

    def test_detect_gpu_capability_falls_back_to_nvidia_smi(self):
        mock_smi = "NVIDIA RTX Pro 6000 Blackwell, 98304, 512, 2500, 9.0"
        with mock.patch("toolkit.throughput_profiles.torch", None):
            with mock.patch("toolkit.throughput_profiles.subprocess.check_output", return_value=mock_smi):
                capability = detect_gpu_capability(device="cuda:0")
        self.assertTrue(capability.available)
        self.assertIn("RTX Pro 6000", capability.name)
        self.assertGreaterEqual(capability.total_vram_gb, 90.0)
        self.assertEqual(capability.compute_capability, (9, 0))
        self.assertGreaterEqual(capability.bandwidth_hint_gbps or 0.0, 300.0)


if __name__ == "__main__":
    unittest.main()
