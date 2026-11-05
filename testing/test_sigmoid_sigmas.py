import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


def _trainer_lookup():
    # Load this model-free method without importing every training/model dependency.
    path = Path(__file__).resolve().parents[1] / "jobs/process/BaseSDTrainProcess.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseSDTrainProcess")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_sigmas")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_sigmas"]


TRAINER_LOOKUP = _trainer_lookup()


def _lookup(scheduler, timesteps, trainer):
    if trainer:
        process = SimpleNamespace(sd=SimpleNamespace(noise_scheduler=scheduler), device="cpu")
        return TRAINER_LOOKUP(process, timesteps, n_dim=4, dtype=torch.float64)
    return scheduler.get_sigmas(timesteps, n_dim=4, dtype=torch.float64, device="cpu")


class SigmoidSigmasTest(unittest.TestCase):
    def test_perfect_velocity_reconstructs_clean_source(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        torch.manual_seed(7419)
        schedule = scheduler.set_train_timesteps(1000, "cpu", "sigmoid")
        timesteps = schedule[[0, 100, 500, 900, 998]]
        clean = torch.zeros(len(timesteps), 1, 1, 1, dtype=torch.float64)
        noise = torch.ones_like(clean)
        velocity = noise - clean
        noisy = scheduler.add_noise(clean, noise, timesteps.view(-1, 1, 1, 1))
        for trainer in (False, True):
            with self.subTest(trainer=trainer):
                sigma = _lookup(scheduler, timesteps, trainer)
                source_loss = ((noisy - sigma * velocity - clean) / sigma).square().mean()
                self.assertEqual(source_loss.item(), 0.0)
                torch.testing.assert_close(sigma, noisy, atol=0, rtol=0)

    def test_sigmoid_sigmas_do_not_depend_on_previous_schedule(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler(shift=3.0)
        torch.manual_seed(7419)
        first = scheduler.set_train_timesteps(1000, "cpu", "sigmoid")
        before = [_lookup(scheduler, first[::100], trainer) for trainer in (False, True)]
        scheduler.set_train_timesteps(1000, "cpu", "shift")
        torch.manual_seed(7419)
        second = scheduler.set_train_timesteps(1000, "cpu", "sigmoid")
        self.assertTrue(torch.equal(first, second))
        for trainer in (False, True):
            with self.subTest(trainer=trainer):
                torch.testing.assert_close(_lookup(scheduler, second[::100], trainer), before[trainer], atol=0, rtol=0)

    def test_duplicate_float32_timesteps_and_sigma_metadata(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        # Saturation produces repeated 1000 and 0 timesteps in float32.
        with mock.patch("torch.randn", return_value=torch.tensor([-100.0, -100.0, 0.0, 100.0, 100.0])):
            timesteps = scheduler.set_train_timesteps(5, torch.device("cpu"), "sigmoid")
        self.assertLess(timesteps.unique().numel(), len(timesteps))
        self.assertEqual(scheduler.sigmas.shape, (6,))
        self.assertEqual(scheduler.sigmas.dtype, timesteps.dtype)
        self.assertEqual(scheduler.sigmas.device, timesteps.device)
        self.assertEqual(scheduler.sigmas[-1].item(), 0)
        expected = (timesteps / 1000).to(torch.float64).view(5, 1, 1, 1)
        for trainer in (False, True):
            with self.subTest(trainer=trainer):
                torch.testing.assert_close(_lookup(scheduler, timesteps, trainer), expected, atol=0, rtol=0)

    def test_inference_after_sigmoid_uses_current_parent_sigmas(self):
        for count, dynamic, invert in ((1000, False, False), (500, False, True), (1000, True, False)):
            with self.subTest(count=count, dynamic=dynamic, invert=invert):
                options = dict(num_train_timesteps=count, shift=3.0,
                               use_dynamic_shifting=dynamic, invert_sigmas=invert)
                scheduler = CustomFlowMatchEulerDiscreteScheduler(**options)
                scheduler.set_train_timesteps(17, "cpu", "sigmoid")
                reference = FlowMatchEulerDiscreteScheduler(**options)
                extra = {"mu": 0.90625} if dynamic else {}
                scheduler.set_timesteps(7, device="cpu", **extra)
                reference.set_timesteps(7, device="cpu", **extra)
                self.assertTrue(torch.equal(scheduler.timesteps, reference.timesteps))
                self.assertTrue(torch.equal(scheduler.sigmas, reference.sigmas))
                expected = reference.sigmas[:-1].to(torch.float64).view(7, 1, 1, 1)
                for trainer in (False, True):
                    torch.testing.assert_close(_lookup(scheduler, scheduler.timesteps, trainer), expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
