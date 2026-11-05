import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

ROOT = Path(__file__).resolve().parents[1]


def _method(path, class_name, name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    scope = {"torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            path,
            "exec",
        ),
        scope,
    )
    return scope[name]


BASE_GET_SIGMAS = _method(
    "jobs/process/BaseSDTrainProcess.py", "BaseSDTrainProcess", "get_sigmas"
)
KREA_TARGET = _method(
    "extensions_built_in/diffusion_models/krea2/krea2.py",
    "Krea2Model",
    "get_loss_target",
)
KREA_TREE = ast.parse(
    (ROOT / "extensions_built_in/diffusion_models/krea2/krea2.py").read_text()
)
KREA_CONFIG = ast.literal_eval(
    next(
        node.value
        for node in KREA_TREE.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "scheduler_config"
            for target in node.targets
        )
    )
)


def _scheduler():
    return CustomFlowMatchEulerDiscreteScheduler(**KREA_CONFIG)


def _get_sigmas(scheduler, timesteps, path, n_dim=4, dtype=torch.float64):
    if path == "custom":
        return scheduler.get_sigmas(timesteps, n_dim, dtype, "cpu")
    process = SimpleNamespace(
        sd=SimpleNamespace(noise_scheduler=scheduler), device=torch.device("cpu")
    )
    return BASE_GET_SIGMAS(process, timesteps, n_dim, dtype)


class SigmoidSigmasTest(unittest.TestCase):
    def test_sigmas_match_physical_noise_for_both_paths(self):
        scheduler = _scheduler()
        torch.manual_seed(7419)
        schedule = scheduler.set_train_timesteps(1000, "cpu", "sigmoid")
        selected = schedule[torch.tensor([0, 100, 500, 900, 998])]
        expected = (selected / 1000).to(torch.float64).view(-1, 1, 1, 1)
        for path in ("custom", "base"):
            with self.subTest(path=path):
                torch.testing.assert_close(
                    _get_sigmas(scheduler, selected, path), expected, atol=0, rtol=0
                )

    def test_perfect_krea_velocity_reconstructs_source_with_zero_loss(self):
        tree = ast.parse(
            (ROOT / "extensions_built_in/sd_trainer/SDTrainer.py").read_text()
        )
        source_branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "denoised_latents"
                    for target in child.targets
                )
                for child in node.body
            )
        )
        code = compile(
            ast.fix_missing_locations(
                ast.Module(body=source_branch.body, type_ignores=[])
            ),
            "actual-source-loss-branch",
            "exec",
        )
        scheduler = _scheduler()
        torch.manual_seed(7419)
        schedule = scheduler.set_train_timesteps(1000, "cpu", "sigmoid")
        selected = schedule[torch.tensor([0, 100, 500, 900, 998])]
        clean = torch.zeros((len(selected), 1, 1, 1), dtype=torch.float64)
        noise = torch.ones_like(clean)
        velocity = KREA_TARGET(None, noise=noise, batch=SimpleNamespace(latents=clean))
        noisy = scheduler.add_noise(clean, noise, selected.view(-1, 1, 1, 1))
        self.assertEqual(
            torch.nn.functional.mse_loss(velocity, noise - clean).item(), 0
        )
        for path in ("custom", "base"):
            with self.subTest(path=path):
                scope = {
                    "self": SimpleNamespace(
                        train_config=SimpleNamespace(train_turbo=False)
                    ),
                    "batch": SimpleNamespace(
                        latents=clean, sigmas=_get_sigmas(scheduler, selected, path)
                    ),
                    "loss_target": "source",
                    "noise_pred": velocity,
                    "noisy_latents": noisy,
                }
                exec(code, scope)
                self.assertEqual(scope["loss"].mean().item(), 0)

    def test_previous_schedule_does_not_change_sigmoid_sigmas(self):
        fresh, shifted = _scheduler(), _scheduler()
        shifted.set_train_timesteps(
            1000, "cpu", "shift", latents=torch.zeros(1, 16, 128, 128), patch_size=2
        )
        self.assertFalse(torch.equal(fresh.sigmas, shifted.sigmas))
        torch.manual_seed(7419)
        first = fresh.set_train_timesteps(1000, "cpu", "sigmoid")
        torch.manual_seed(7419)
        second = shifted.set_train_timesteps(1000, "cpu", "sigmoid")
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(fresh.sigmas, shifted.sigmas))
        for path in ("custom", "base"):
            with self.subTest(path=path):
                torch.testing.assert_close(
                    _get_sigmas(fresh, first[:17], path),
                    _get_sigmas(shifted, second[:17], path),
                    atol=0,
                    rtol=0,
                )

    def test_count_device_sorting_and_original_rng_sequence(self):
        scheduler = _scheduler()
        for count in (1, 7, 1000, 2048):
            with self.subTest(count=count):
                torch.manual_seed(12345)
                expected = torch.sort(
                    (1 - torch.sigmoid(torch.randn(count))) * 1000, descending=True
                ).values
                expected_rng = torch.get_rng_state()
                torch.manual_seed(12345)
                actual = scheduler.set_train_timesteps(
                    count, torch.device("cpu"), "sigmoid"
                )
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                self.assertEqual(actual.device.type, "cpu")
                self.assertEqual(scheduler.sigmas.shape, (count + 1,))
                self.assertEqual(scheduler.sigmas.dtype, actual.dtype)
                self.assertTrue(torch.equal(scheduler.sigmas[:-1], actual / 1000))
                self.assertEqual(scheduler.sigmas[-1].item(), 0)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(((actual >= 0) & (actual <= 1000)).all())

    def test_duplicate_float32_timesteps_preserve_lookup_shape_dtype(self):
        scheduler = _scheduler()
        normal_samples = torch.tensor([-100.0, -100.0, 0.0, 100.0, 100.0])
        with mock.patch("torch.randn", return_value=normal_samples):
            timesteps = scheduler.set_train_timesteps(5, "cpu", "sigmoid")
        self.assertLess(timesteps.unique().numel(), len(timesteps))
        for path in ("custom", "base"):
            for n_dim in (1, 4, 5):
                with self.subTest(path=path, n_dim=n_dim):
                    actual = _get_sigmas(scheduler, timesteps, path, n_dim)
                    expected = (
                        (timesteps / 1000)
                        .to(torch.float64)
                        .view(5, *([1] * (n_dim - 1)))
                    )
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertEqual(actual.dtype, torch.float64)

    def test_other_training_modes_keep_their_previous_sigmas_after_sigmoid(self):
        for mode in ("linear", "weighted", "lognorm_blend"):
            with self.subTest(mode=mode):
                scheduler, reference = _scheduler(), _scheduler()
                scheduler.set_train_timesteps(17, "cpu", "sigmoid")
                scheduler.set_train_timesteps(11, "cpu", "sigmoid")

                reference.set_train_timesteps(1000, "cpu", mode)
                expected = reference.sigmas[0].to(torch.float64).view(1)
                timesteps = scheduler.set_train_timesteps(1000, "cpu", mode)

                torch.testing.assert_close(scheduler.sigmas, reference.sigmas)
                for path in ("custom", "base"):
                    actual = _get_sigmas(scheduler, timesteps[:1], path, n_dim=1)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_training_after_inference_does_not_restore_pre_inference_sigmas(self):
        scheduler = _scheduler()
        scheduler.set_train_timesteps(17, "cpu", "sigmoid")
        scheduler.set_timesteps(7, device="cpu", mu=0.90625)
        expected = scheduler.sigmas.clone()
        scheduler.set_train_timesteps(11, "cpu", "sigmoid")
        scheduler.set_train_timesteps(1000, "cpu", "linear")
        torch.testing.assert_close(scheduler.sigmas, expected)

    def test_inference_after_sigmoid_uses_live_parent_schedule(self):
        for count, shift, dynamic, invert in (
            (1000, 3.0, False, False),
            (500, 3.0, False, True),
            (1000, 1.0, True, False),
        ):
            with self.subTest(count=count, shift=shift, dynamic=dynamic, invert=invert):
                kwargs = dict(
                    num_train_timesteps=count,
                    shift=shift,
                    use_dynamic_shifting=dynamic,
                    invert_sigmas=invert,
                )
                scheduler = CustomFlowMatchEulerDiscreteScheduler(**kwargs)
                scheduler.set_train_timesteps(17, "cpu", "sigmoid")
                reference = FlowMatchEulerDiscreteScheduler(**kwargs)
                inference_kwargs = {"mu": 0.90625} if dynamic else {}
                scheduler.set_timesteps(7, device="cpu", **inference_kwargs)
                reference.set_timesteps(7, device="cpu", **inference_kwargs)
                self.assertTrue(torch.equal(scheduler.timesteps, reference.timesteps))
                self.assertTrue(torch.equal(scheduler.sigmas, reference.sigmas))
                self.assertEqual(scheduler.timestep_type, "sigmoid")
                expected = reference.sigmas[:-1].to(torch.float64).view(7, 1, 1, 1)
                for path in ("custom", "base"):
                    torch.testing.assert_close(
                        _get_sigmas(scheduler, scheduler.timesteps, path),
                        expected,
                        atol=0,
                        rtol=0,
                    )


if __name__ == "__main__":
    unittest.main()
