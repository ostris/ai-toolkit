from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_BEAM_PATH = PROJECT_ROOT / "run_beam.py"
BEAM_DOCKERFILE_PATH = PROJECT_ROOT / "docker" / "Dockerfile.beam"
BEAMIGNORE_PATH = PROJECT_ROOT / ".beamignore"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"


class FakeImage:
    def __init__(self, dockerfile: str, context_dir: str):
        self.dockerfile = dockerfile
        self.context_dir = context_dir
        self.build_gpu = None

    @classmethod
    def from_dockerfile(cls, dockerfile: str, context_dir: str):
        return cls(dockerfile, context_dir)

    def build_with_gpu(self, gpu: str):
        self.build_gpu = gpu
        return self


class FakeVolume:
    def __init__(self, name: str, mount_path: str):
        self.name = name
        self.mount_path = mount_path


class FakePod:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def load_run_beam(
    gpu: str | None,
    pool: str | None = None,
    keep_warm_seconds: str | None = None,
):
    captured: dict[str, object] = {}
    fake_beam = types.ModuleType("beam")
    fake_beam.Image = FakeImage
    fake_beam.Volume = FakeVolume
    fake_beam.Pod = lambda **kwargs: FakePod(**kwargs)

    def fake_function(**kwargs):
        captured.update(kwargs)

        def decorator(func):
            func.remote = lambda *args, **call_kwargs: None
            return func

        return decorator

    fake_beam.function = fake_function

    environ = {}
    if gpu is not None:
        environ["BEAM_GPU"] = gpu
    if pool is not None:
        environ["BEAM_POOL"] = pool
    if keep_warm_seconds is not None:
        environ["BEAM_KEEP_WARM_SECONDS"] = keep_warm_seconds

    module_name = f"run_beam_test_{gpu}_{pool}_{id(captured)}"
    spec = importlib.util.spec_from_file_location(module_name, RUN_BEAM_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load run_beam.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with patch.dict(os.environ, environ, clear=True), patch.dict(
            sys.modules, {"beam": fake_beam}
        ):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    return module, captured


class BeamGpuConfigTests(unittest.TestCase):
    def test_all_serverless_gpus_use_no_pool(self):
        for gpu in ("A10G", "RTX4090", "RTX5090"):
            with self.subTest(gpu=gpu):
                module, captured = load_run_beam(gpu)
                self.assertEqual(module.beam_gpu_config.mode, "serverless")
                self.assertEqual(captured["gpu"], gpu)
                self.assertIsNone(captured["pool"])
                self.assertIsInstance(captured["gpu"], str)
                gui = module.ai_toolkit_gui.kwargs
                self.assertEqual(gui["gpu"], gpu)
                self.assertIsNone(gui["pool"])

    def test_all_on_demand_gpus_use_the_selected_pool(self):
        on_demand_gpus = (
            "RTX4090",
            "RTX5090",
            "RTXPro6000",
            "A6000",
            "L40S",
            "A100-80",
            "H100",
            "H200",
            "B200",
        )
        for gpu in on_demand_gpus:
            with self.subTest(gpu=gpu):
                module, captured = load_run_beam(gpu, "training-pool")
                self.assertEqual(module.beam_gpu_config.mode, "on-demand")
                self.assertEqual(captured["gpu"], gpu)
                self.assertEqual(captured["pool"], "training-pool")
                gui = module.ai_toolkit_gui.kwargs
                self.assertEqual(gui["gpu"], gpu)
                self.assertEqual(gui["pool"], "training-pool")

    def test_gui_pod_exposes_ui_and_persists_state(self):
        module, _ = load_run_beam("RTX4090")
        gui = module.ai_toolkit_gui.kwargs
        self.assertEqual(gui["name"], "ai-toolkit-ui")
        self.assertEqual(gui["ports"], [8675])
        self.assertEqual(gui["keep_warm_seconds"], 1800)
        self.assertFalse(gui["authorized"])
        self.assertEqual(gui["secrets"], ["HF_TOKEN", "AI_TOOLKIT_AUTH"])
        self.assertEqual(
            gui["env"]["DATABASE_URL"],
            "file:/volumes/ai-toolkit-ui-state/aitk_db.db",
        )
        self.assertEqual(
            gui["env"]["DATASETS_FOLDER"],
            "/volumes/ai-toolkit-ui-state/datasets",
        )
        self.assertEqual(gui["env"]["TRAINING_FOLDER"], "/volumes/ai-toolkit-output")
        self.assertEqual(gui["env"]["BEAM_KEEP_WARM_SECONDS"], "1800")
        self.assertIn("AI_TOOLKIT_AUTH", gui["entrypoint"][-1])
        self.assertIn("npm run start", gui["entrypoint"][-1])
        self.assertEqual(
            [volume.name for volume in gui["volumes"]],
            ["ai-toolkit-output", "ai-toolkit-cache", "ai-toolkit-ui-state"],
        )

    def test_image_build_uses_the_explicit_gpu(self):
        for gpu, pool in (("A10G", None), ("RTX4090", None), ("H100", "h100-pool")):
            with self.subTest(gpu=gpu):
                module, _ = load_run_beam(gpu, pool)
                self.assertEqual(module.image.build_gpu, gpu)
                self.assertNotEqual(module.image.context_dir, ".")
        self.assertNotIn(str(PROJECT_ROOT), module.image.context_dir)

    def test_keep_warm_timeout_is_configurable(self):
        for raw_value, expected in (("3600", 3600), ("-1", -1), ("0", 0)):
            with self.subTest(raw_value=raw_value):
                module, _ = load_run_beam("RTX4090", keep_warm_seconds=raw_value)
                self.assertEqual(module.beam_keep_warm_seconds, expected)
                self.assertEqual(
                    module.ai_toolkit_gui.kwargs["keep_warm_seconds"], expected
                )

    def test_invalid_keep_warm_timeout_is_rejected(self):
        for raw_value in ("30m", "-2", "1.5"):
            with self.subTest(raw_value=raw_value):
                with self.assertRaises(ValueError):
                    load_run_beam("RTX4090", keep_warm_seconds=raw_value)

    def test_missing_gpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "BEAM_GPU is required"):
            load_run_beam(None)

    def test_unknown_gpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported BEAM_GPU"):
            load_run_beam("T4")

    def test_on_demand_gpu_without_pool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "H100 is on-demand only"):
            load_run_beam("H100")

    def test_serverless_only_gpu_with_pool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "A10G is not supported"):
            load_run_beam("A10G", "training-pool")

    def test_empty_pool_uses_serverless_mode(self):
        module, captured = load_run_beam("RTX4090", "   ")
        self.assertEqual(module.beam_gpu_config.mode, "serverless")
        self.assertIsNone(captured["pool"])

    def test_remote_path_translation_preserves_relative_paths(self):
        module, _ = load_run_beam("RTX4090")
        self.assertEqual(
            module._to_remote_path("config/job.yaml", PROJECT_ROOT),
            "config/job.yaml",
        )

    def test_remote_path_translation_maps_repository_absolute_paths(self):
        module, _ = load_run_beam("RTX4090")
        config_path = PROJECT_ROOT / "config" / "job.yaml"
        self.assertEqual(
            module._to_remote_path(str(config_path), PROJECT_ROOT),
            "/mnt/code/config/job.yaml",
        )

    def test_remote_path_translation_rejects_external_absolute_paths(self):
        module, _ = load_run_beam("RTX4090")
        with self.assertRaisesRegex(ValueError, "must be inside the repository"):
            module._to_remote_path("/tmp/job.yaml", PROJECT_ROOT)


class BeamTrainingTests(unittest.TestCase):
    @staticmethod
    def _fake_torch():
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "Fake RTX 4090",
        )
        fake_torch.version = types.SimpleNamespace(cuda="12.9")
        fake_torch.autograd = types.SimpleNamespace(set_detect_anomaly=lambda enabled: None)
        return fake_torch

    def _run_training(self, failures: set[str], recover: bool):
        module, _ = load_run_beam("RTX4090")
        events: list[tuple[str, str]] = []
        stdout = io.StringIO()

        class FakeJob:
            def __init__(self, config_file: str):
                self.config_file = config_file
                self.config = {"process": [{}]}

            def run(self):
                events.append(("run", self.config_file))
                if self.config_file in failures:
                    raise RuntimeError(f"failed {self.config_file}")

            def cleanup(self):
                events.append(("cleanup", self.config_file))

        fake_toolkit = types.ModuleType("toolkit")
        fake_toolkit.__path__ = []
        fake_job_module = types.ModuleType("toolkit.job")
        fake_job_module.get_job = lambda config_file, name: FakeJob(config_file)

        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            module.APP_DIR = temp_dir
            module.OUTPUT_DIR = str(Path(temp_dir) / "output")
            module.CACHE_DIR = str(Path(temp_dir) / "cache")
            try:
                with patch.dict(
                    sys.modules,
                    {
                        "torch": self._fake_torch(),
                        "toolkit": fake_toolkit,
                        "toolkit.job": fake_job_module,
                    },
                ), patch.dict(
                    os.environ,
                    {"BEAM_GPU": "RTX4090", "BEAM_POOL": ""},
                ), contextlib.redirect_stdout(stdout):
                    result = module.train(["first.yaml", "second.yaml"], recover=recover)
            finally:
                os.chdir(previous_cwd)

        return result, events, stdout.getvalue()

    def test_multiple_configs_run_sequentially(self):
        result, events, output = self._run_training(failures=set(), recover=False)
        self.assertEqual(result, {"completed": 2, "failed": 0})
        self.assertIn("Requested Beam GPU: RTX4090", output)
        self.assertIn("Actual CUDA device: Fake RTX 4090", output)
        self.assertIn("PyTorch CUDA version: 12.9", output)
        self.assertIn("Training elapsed time (first.yaml):", output)
        self.assertEqual(
            events,
            [
                ("run", "first.yaml"),
                ("cleanup", "first.yaml"),
                ("run", "second.yaml"),
                ("cleanup", "second.yaml"),
            ],
        )

    def test_recover_continues_after_a_failed_config(self):
        result, events, output = self._run_training(
            failures={"first.yaml"}, recover=True
        )
        self.assertEqual(result, {"completed": 1, "failed": 1})
        self.assertIn("Error running job: failed first.yaml", output)
        self.assertEqual(
            events,
            [
                ("run", "first.yaml"),
                ("cleanup", "first.yaml"),
                ("run", "second.yaml"),
                ("cleanup", "second.yaml"),
            ],
        )


class BeamDockerfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dockerfile = BEAM_DOCKERFILE_PATH.read_text()

    def test_uses_cuda_12_9_for_all_beam_gpu_generations(self):
        self.assertIn("FROM nvidia/cuda:12.9.1-devel-ubuntu24.04", self.dockerfile)
        self.assertIn("https://download.pytorch.org/whl/cu129", self.dockerfile)
        self.assertNotIn("cu130", self.dockerfile)

    def test_builds_and_import_checks_cuda_extensions(self):
        self.assertIn("FLASH_ATTENTION_FORCE_BUILD=TRUE", self.dockerfile)
        self.assertIn('FLASH_ATTN_CUDA_ARCHS="80;90;100;120"', self.dockerfile)
        self.assertIn(
            'NATTEN_CUDA_ARCH="8.0;8.6;8.9;9.0;10.0;12.0"',
            self.dockerfile,
        )
        self.assertIn("natten==0.21.7", self.dockerfile)
        self.assertIn("flash-attn==2.8.3", self.dockerfile)
        self.assertIn("import flash_attn, natten, torchcodec", self.dockerfile)

    def test_builds_the_existing_next_ui_in_the_cuda_image(self):
        self.assertIn("nodejs", self.dockerfile)
        self.assertIn("npm", self.dockerfile)
        self.assertIn("COPY ui/package.json ui/package-lock.json", self.dockerfile)
        self.assertIn("npm ci --no-audit --no-fund", self.dockerfile)
        self.assertIn("npx prisma generate", self.dockerfile)
        self.assertIn("npm run build", self.dockerfile)
        self.assertIn("ENV DATABASE_URL=file:/mnt/code/aitk_db.db", self.dockerfile)

    def test_packaging_keeps_ui_source_but_excludes_local_build_artifacts(self):
        beamignore = BEAMIGNORE_PATH.read_text()
        dockerignore = DOCKERIGNORE_PATH.read_text()
        self.assertNotIn("\nui/\n", beamignore)
        self.assertIn("ui/node_modules/", beamignore)
        self.assertIn("ui/.next/", beamignore)
        self.assertIn("ui/node_modules/", dockerignore)
        self.assertIn("ui/.next/", dockerignore)


if __name__ == "__main__":
    unittest.main()
