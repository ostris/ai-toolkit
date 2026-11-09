import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler

from toolkit.config_modules import ModelConfig, SaveConfig, TrainConfig, validate_configs
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


ROOT = Path(__file__).resolve().parents[1]


def _training_scheduler_call(path):
    # Exercise the real call without loading the trainer and its model dependencies.
    tree = ast.parse((ROOT / path).read_text())
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_train_timesteps"
        and any(keyword.arg is None for keyword in node.keywords)
    )
    return compile(ast.fix_missing_locations(ast.Expression(call)), path, "eval")


class _RecordingScheduler(CustomFlowMatchEulerDiscreteScheduler):
    def set_train_timesteps(self, *args, **kwargs):
        self.received_kwargs = kwargs.copy()
        return super().set_train_timesteps(*args, **kwargs)


class SigmoidParametersTest(unittest.TestCase):
    def setUp(self):
        # Seeded checks should not change the RNG state of other tests.
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    def test_defaults_preserve_original_samples_and_rng(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        for count in (1, 7, 1000, 2048):
            for kwargs in ({}, {"timestep_shift": 0.0, "timestep_scale": 1.0}):
                with self.subTest(count=count, kwargs=kwargs):
                    torch.manual_seed(7419)
                    expected = torch.sort(
                        (1 - torch.sigmoid(torch.randn(count))) * 1000,
                        descending=True,
                    ).values
                    expected_rng = torch.get_rng_state()

                    torch.manual_seed(7419)
                    actual = scheduler.set_train_timesteps(count, "cpu", "sigmoid", **kwargs)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assertTrue(torch.equal(scheduler.timesteps, actual))

    def test_configured_samples_follow_affine_normal_transform(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        for shift, scale in ((-0.7, 0.9), (0.7, 0.5), (0.0, 1.3), (2, 1)):
            for count in (1, 7, 1000, 2048):
                with self.subTest(shift=shift, scale=scale, count=count):
                    torch.manual_seed(7419)
                    normal = torch.randn(count) * scale + shift
                    expected = torch.sort((1 - normal.sigmoid()) * 1000, descending=True).values
                    expected_rng = torch.get_rng_state()

                    torch.manual_seed(7419)
                    actual = scheduler.set_train_timesteps(
                        count, torch.device("cpu"), "sigmoid",
                        timestep_shift=shift, timestep_scale=scale,
                    )
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assertEqual(actual.shape, (count,))
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue(((actual >= 0) & (actual <= 1000)).all())
                    self.assertTrue((actual[:-1] >= actual[1:]).all())

    def test_shift_and_scale_can_be_set_independently(self):
        for kwargs, shift, scale in (
            ({"timestep_shift": -0.7}, -0.7, 1.0),
            ({"timestep_scale": 0.9}, 0.0, 0.9),
        ):
            with self.subTest(kwargs=kwargs):
                torch.manual_seed(74)
                expected = torch.sort(
                    (1 - (torch.randn(1000) * scale + shift).sigmoid()) * 1000,
                    descending=True,
                ).values
                torch.manual_seed(74)
                actual = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
                    1000, "cpu", "sigmoid", **kwargs,
                )
                self.assertTrue(torch.equal(actual, expected))

    def test_scheduler_config_is_used_when_arguments_are_omitted(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        scheduler.register_to_config(timestep_shift=-0.4, timestep_scale=0.3)
        torch.manual_seed(74)
        configured = scheduler.set_train_timesteps(1000, "cpu", "sigmoid")
        torch.manual_seed(74)
        explicit = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
            1000, "cpu", "sigmoid", timestep_shift=-0.4, timestep_scale=0.3,
        )
        self.assertTrue(torch.equal(configured, explicit))

    def test_explicit_defaults_override_scheduler_config(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        scheduler.register_to_config(timestep_shift=-0.4, timestep_scale=0.3)
        torch.manual_seed(74)
        actual = scheduler.set_train_timesteps(
            1000, "cpu", "sigmoid", timestep_shift=0.0, timestep_scale=1.0,
        )
        torch.manual_seed(74)
        expected = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(1000, "cpu", "sigmoid")
        self.assertTrue(torch.equal(actual, expected))

    def test_invalid_parameters_fail_before_consuming_random_samples(self):
        for shift, scale in (
            (float("nan"), 1), (float("inf"), 1), (-float("inf"), 1),
            (0, float("nan")), (0, float("inf")), (0, -float("inf")),
            (0, 0), (0, -1),
        ):
            with self.subTest(shift=shift, scale=scale):
                scheduler = CustomFlowMatchEulerDiscreteScheduler()
                rng = torch.get_rng_state()
                with mock.patch("torch.randn") as draw:
                    parameter = "timestep_shift" if not math.isfinite(shift) else "timestep_scale"
                    with self.assertRaisesRegex(ValueError, parameter):
                        scheduler.set_train_timesteps(
                            7, "cpu", "sigmoid", timestep_shift=shift, timestep_scale=scale,
                        )
                    draw.assert_not_called()
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_tail_samples_are_not_clamped(self):
        # Cover tail draws and natural sigmoid saturation without adding cutoffs.
        normal = torch.tensor([-100.0, -6.0, 0.0, 6.0, 100.0])
        with mock.patch("torch.randn", return_value=normal):
            values = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
                len(normal), "cpu", "sigmoid", timestep_shift=-0.7, timestep_scale=0.9,
            )
        self.assertEqual(values[0].item(), 1000)
        self.assertGreater(values[1].item(), 850)
        self.assertLess(values[-2].item(), 150)
        self.assertEqual(values[-1].item(), 0)

    def test_distribution_direction_and_tail_probability(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        medians = []
        for shift in (0.7, 0.0, -0.7):
            torch.manual_seed(7419)
            values = scheduler.set_train_timesteps(
                100000, "cpu", "sigmoid", timestep_shift=shift, timestep_scale=0.9,
            )
            medians.append(values.median().item())
        self.assertAlmostEqual(medians[0], 1000 / (1 + math.exp(0.7)), delta=3)
        self.assertAlmostEqual(medians[1], 500, delta=3)
        self.assertAlmostEqual(medians[2], 1000 / (1 + math.exp(-0.7)), delta=3)
        self.assertLess(medians[0], medians[1])
        self.assertLess(medians[1], medians[2])
        # T > 600 is equivalent to Z < log(0.4 / 0.6), before the sigmoid transform.
        expected = 0.5 * (1 + math.erf((math.log(0.4 / 0.6) + 0.7) / (0.9 * math.sqrt(2))))
        self.assertAlmostEqual((values > 600).float().mean().item(), expected, delta=0.005)

    def test_normal_samples_have_configured_mean_and_standard_deviation(self):
        torch.manual_seed(7419)
        values = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
            100000, "cpu", "sigmoid", timestep_shift=-0.7, timestep_scale=0.9,
        )
        # Undo the transform to check the distribution, independently of the seeded formula.
        normal = torch.logit(1 - values.double() / 1000)
        self.assertAlmostEqual(normal.mean().item(), -0.7, delta=0.01)
        self.assertAlmostEqual(normal.std().item(), 0.9, delta=0.01)

    def test_inference_shift_and_previous_schedule_do_not_change_samples(self):
        for dynamic, invert in ((False, False), (False, True), (True, False)):
            scheduler = CustomFlowMatchEulerDiscreteScheduler(
                shift=3.0, use_dynamic_shifting=dynamic, invert_sigmas=invert,
            )
            scheduler.set_train_timesteps(1000, "cpu", "shift", latents=torch.zeros(1, 16, 64, 64), patch_size=2)
            torch.manual_seed(74)
            actual = scheduler.set_train_timesteps(
                1000, "cpu", "sigmoid", timestep_shift=-0.7, timestep_scale=0.9,
            )
            torch.manual_seed(74)
            expected = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
                1000, "cpu", "sigmoid", timestep_shift=-0.7, timestep_scale=0.9,
            )
            self.assertTrue(torch.equal(actual, expected))

    def test_parent_inference_schedule_is_unchanged(self):
        for dynamic, invert in ((False, False), (False, True), (True, False)):
            kwargs = {"shift": 3.0, "use_dynamic_shifting": dynamic, "invert_sigmas": invert}
            scheduler = CustomFlowMatchEulerDiscreteScheduler(**kwargs)
            scheduler.set_train_timesteps(1000, "cpu", "sigmoid", timestep_shift=-0.7, timestep_scale=0.9)
            reference = FlowMatchEulerDiscreteScheduler(**kwargs)
            inference = {"mu": 0.9} if dynamic else {}
            scheduler.set_timesteps(7, device="cpu", **inference)
            reference.set_timesteps(7, device="cpu", **inference)
            self.assertTrue(torch.equal(scheduler.timesteps, reference.timesteps))
            self.assertTrue(torch.equal(scheduler.sigmas, reference.sigmas))

    def test_other_training_modes_ignore_sigmoid_parameters(self):
        for kind in ("linear", "weighted", "shift", "flux_shift", "lumina2_shift", "lognorm_blend"):
            with self.subTest(kind=kind):
                plain = CustomFlowMatchEulerDiscreteScheduler(shift=3.0)
                configured = CustomFlowMatchEulerDiscreteScheduler(shift=3.0)
                torch.manual_seed(74)
                expected = plain.set_train_timesteps(1000, "cpu", kind)
                rng = torch.get_rng_state()
                torch.manual_seed(74)
                actual = configured.set_train_timesteps(
                    1000, "cpu", kind, timestep_shift=float("nan"), timestep_scale=-1,
                )
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(configured.sigmas, plain.sigmas))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_yaml_round_trip_and_defaults(self):
        for document, shift, scale in (
            ("timestep_type: sigmoid", 0.0, 1.0),
            ("timestep_type: sigmoid\ntimestep_shift: -0.7", -0.7, 1.0),
            ("timestep_type: sigmoid\ntimestep_scale: 0.9", 0.0, 0.9),
            ("timestep_type: sigmoid\ntimestep_shift: -0.7\ntimestep_scale: 0.9", -0.7, 0.9),
        ):
            config = TrainConfig(**yaml.safe_load(yaml.safe_dump(yaml.safe_load(document))))
            validate_configs(config, ModelConfig(arch="flux", name_or_path="unused"), SaveConfig(), [])
            self.assertEqual(config.timestep_type, "sigmoid")
            self.assertEqual(config.timestep_shift, shift)
            self.assertEqual(config.timestep_scale, scale)

    def test_effective_timestep_type_controls_parameter_forwarding(self):
        config = TrainConfig(timestep_type="sigmoid", timestep_shift=-0.7, timestep_scale=0.9)
        self.assertEqual(
            config.get_timestep_kwargs("sigmoid"),
            {"timestep_shift": -0.7, "timestep_scale": 0.9},
        )
        # A linear override must not forward sigmoid settings from the saved config.
        for kind in ("linear", "weighted", "shift", "next_sample"):
            with self.subTest(kind=kind):
                self.assertEqual(config.get_timestep_kwargs(kind), {})

    def test_training_calls_forward_parameters_only_for_sigmoid(self):
        config = TrainConfig(timestep_shift=-0.4, timestep_scale=0.3, max_denoising_steps=128)
        for path in ("jobs/process/BaseSDTrainProcess.py", "jobs/process/TrainSliderProcess.py", "toolkit/guidance.py"):
            call = _training_scheduler_call(path)
            for kind in ("sigmoid", "linear", "shift"):
                with self.subTest(path=path, kind=kind):
                    scheduler = _RecordingScheduler()
                    sd = SimpleNamespace(noise_scheduler=scheduler)
                    process = SimpleNamespace(sd=sd, train_config=config, device_torch=torch.device("cpu"))
                    latents = torch.zeros(1, 16, 32, 32)
                    scope = {"self": process, "sd": sd, "train_config": config, "timestep_type": kind,
                             "num_train_timesteps": 128, "device": "cpu", "latents": latents,
                             "conditional_latents": latents, "l": latents, "patch_size": 2}
                    torch.manual_seed(74)
                    actual = eval(call, scope)
                    forwarded = scheduler.received_kwargs
                    if kind == "sigmoid":
                        self.assertEqual(forwarded["timestep_shift"], -0.4)
                        self.assertEqual(forwarded["timestep_scale"], 0.3)
                        torch.manual_seed(74)
                        expected = torch.sort(
                            (1 - (torch.randn(len(actual)) * 0.3 - 0.4).sigmoid()) * 1000,
                            descending=True,
                        ).values
                        self.assertTrue(torch.equal(actual, expected))
                    else:
                        self.assertNotIn("timestep_shift", forwarded)
                        self.assertNotIn("timestep_scale", forwarded)

    def test_ming_override_forwards_parameters_with_dynamic_shifting(self):
        # Load only the override; its text encoder and model weights are not needed here.
        path = "extensions_built_in/diffusion_models/ming_image/ming_image.py"
        tree = ast.parse((ROOT / path).read_text())
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MingImageFlowMatchScheduler"
        )
        shift_params = mock.Mock(return_value=(1.35, 4096))
        scope = {"CustomFlowMatchEulerDiscreteScheduler": CustomFlowMatchEulerDiscreteScheduler,
                 "ming_shift_params": shift_params}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), path, "exec"), scope)
        scheduler = scope[cls.name](use_dynamic_shifting=True)
        torch.manual_seed(74)
        actual = scheduler.set_train_timesteps(
            1000, "cpu", "sigmoid", latents=torch.zeros(1, 16, 32, 32), patch_size=2,
            timestep_shift=-0.4, timestep_scale=0.3,
        )
        shift_params.assert_called_once_with(256)
        self.assertEqual(scheduler.config.max_shift, 1.35)
        torch.manual_seed(74)
        expected = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
            1000, "cpu", "sigmoid", timestep_shift=-0.4, timestep_scale=0.3,
        )
        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_configured_samples_stay_on_cuda(self):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            values = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
                1000, "cuda", "sigmoid", timestep_shift=-0.7, timestep_scale=0.9,
            )
        self.assertEqual(values.device.type, "cuda")
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((values[:-1] >= values[1:]).all())

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
    def test_configured_samples_stay_on_mps(self):
        self.addCleanup(torch.mps.set_rng_state, torch.mps.get_rng_state())
        values = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
            1000, "mps", "sigmoid", timestep_shift=-0.7, timestep_scale=0.9,
        )
        self.assertEqual(values.device.type, "mps")
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((values[:-1] >= values[1:]).all())


if __name__ == "__main__":
    unittest.main()
