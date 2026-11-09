import ast
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler

from toolkit.config_modules import (
    ModelConfig,
    SaveConfig,
    TrainConfig,
    validate_configs,
)
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.train_tools import get_torch_dtype

ROOT = Path(__file__).resolve().parents[1]


def _training_scheduler_call(path):
    # Exercise the trainer call without loading its model dependencies.
    tree = ast.parse((ROOT / path).read_text())
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_train_timesteps"
        and any(keyword.arg is None for keyword in node.keywords)
    )
    return compile(
        ast.fix_missing_locations(ast.Expression(call)), path, "eval"
    )


class _RecordingScheduler(CustomFlowMatchEulerDiscreteScheduler):
    def set_train_timesteps(self, *args, **kwargs):
        self.received_kwargs = kwargs.copy()
        return super().set_train_timesteps(*args, **kwargs)


class SigmoidParametersTest(unittest.TestCase):
    def setUp(self):
        # Seeded checks should not change the RNG state of other tests.
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    def test_duplicate_sigmoid_timesteps_have_matching_sigmas(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        previous = scheduler.sigmas
        timesteps = scheduler.set_train_timesteps(
            1000, "cpu", "sigmoid", timestep_shift=20, timestep_scale=0.01
        )
        self.assertEqual(timesteps.unique().numel(), 1)
        expected = (timesteps[:2] / 1000).reshape(2, 1)
        actual = scheduler.get_sigmas(timesteps[:2], 2, torch.float32, "cpu")
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(scheduler.sigmas[:-1], timesteps / 1000)

        path = "jobs/process/BaseSDTrainProcess.py"
        tree = ast.parse((ROOT / path).read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "get_sigmas"
        )
        scope = {"torch": torch}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), path, "exec"),
            scope,
        )
        trainer = SimpleNamespace(
            sd=SimpleNamespace(noise_scheduler=scheduler), device="cpu"
        )
        actual = scope["get_sigmas"](trainer, timesteps[:2], n_dim=2)
        torch.testing.assert_close(actual, expected)
        scheduler.set_train_timesteps(9, "cpu", "linear")
        self.assertIs(scheduler.sigmas, previous)

    def test_sampler_rejects_finite_values_outside_its_dtype_before_draw(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        with mock.patch("torch.randn", wraps=torch.randn) as draw:
            with self.assertRaisesRegex(ValueError, "timestep_shift"):
                scheduler.set_train_timesteps(
                    1000, "cpu", "sigmoid", timestep_shift=1e100
                )
            with self.assertRaisesRegex(ValueError, "timestep_scale"):
                scheduler.set_train_timesteps(
                    1000, "cpu", "sigmoid", timestep_scale=1e100
                )
            draw.assert_not_called()

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
                    actual = scheduler.set_train_timesteps(
                        count, "cpu", "sigmoid", **kwargs
                    )
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(
                        torch.equal(torch.get_rng_state(), expected_rng)
                    )
                    self.assertTrue(torch.equal(scheduler.timesteps, actual))

    def test_configured_samples_follow_affine_normal_transform(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        for shift, scale in ((-0.7, 0.9), (0.7, 0.5), (0.0, 1.3), (2, 1)):
            for count in (1, 7, 1000, 2048):
                with self.subTest(shift=shift, scale=scale, count=count):
                    torch.manual_seed(7419)
                    normal = torch.randn(count) * scale + shift
                    expected = torch.sort(
                        (1 - normal.sigmoid()) * 1000, descending=True
                    ).values
                    expected_rng = torch.get_rng_state()

                    torch.manual_seed(7419)
                    actual = scheduler.set_train_timesteps(
                        count,
                        torch.device("cpu"),
                        "sigmoid",
                        timestep_shift=shift,
                        timestep_scale=scale,
                    )
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(
                        torch.equal(torch.get_rng_state(), expected_rng)
                    )
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
                scheduler = CustomFlowMatchEulerDiscreteScheduler()
                actual = scheduler.set_train_timesteps(
                    1000,
                    "cpu",
                    "sigmoid",
                    **kwargs,
                )
                self.assertTrue(torch.equal(actual, expected))

    def test_scheduler_config_is_used_when_arguments_are_omitted(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        scheduler.register_to_config(timestep_shift=-0.4, timestep_scale=0.3)
        torch.manual_seed(74)
        configured = scheduler.set_train_timesteps(1000, "cpu", "sigmoid")
        torch.manual_seed(74)
        explicit = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
            1000,
            "cpu",
            "sigmoid",
            timestep_shift=-0.4,
            timestep_scale=0.3,
        )
        self.assertTrue(torch.equal(configured, explicit))

    def test_explicit_defaults_override_scheduler_config(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        scheduler.register_to_config(timestep_shift=-0.4, timestep_scale=0.3)
        torch.manual_seed(74)
        actual = scheduler.set_train_timesteps(
            1000,
            "cpu",
            "sigmoid",
            timestep_shift=0.0,
            timestep_scale=1.0,
        )
        torch.manual_seed(74)
        expected = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
            1000, "cpu", "sigmoid"
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_invalid_parameters_fail_before_consuming_random_samples(self):
        for shift, scale in (
            (float("nan"), 1),
            (float("inf"), 1),
            (-float("inf"), 1),
            (0, float("nan")),
            (0, float("inf")),
            (0, -float("inf")),
            (0, 0),
            (0, -1),
        ):
            with self.subTest(shift=shift, scale=scale):
                scheduler = CustomFlowMatchEulerDiscreteScheduler()
                rng = torch.get_rng_state()
                with mock.patch("torch.randn") as draw:
                    parameter = (
                        "timestep_shift"
                        if not math.isfinite(shift)
                        else "timestep_scale"
                    )
                    with self.assertRaisesRegex(ValueError, parameter):
                        scheduler.set_train_timesteps(
                            7,
                            "cpu",
                            "sigmoid",
                            timestep_shift=shift,
                            timestep_scale=scale,
                        )
                    draw.assert_not_called()
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_tail_samples_are_not_clamped(self):
        # Cover the tails and natural sigmoid saturation.
        normal = torch.tensor([-100.0, -6.0, 0.0, 6.0, 100.0])
        with mock.patch("torch.randn", return_value=normal):
            values = (
                CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
                    len(normal),
                    "cpu",
                    "sigmoid",
                    timestep_shift=-0.7,
                    timestep_scale=0.9,
                )
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
                100000,
                "cpu",
                "sigmoid",
                timestep_shift=shift,
                timestep_scale=0.9,
            )
            medians.append(values.median().item())
        self.assertAlmostEqual(medians[0], 1000 / (1 + math.exp(0.7)), delta=3)
        self.assertAlmostEqual(medians[1], 500, delta=3)
        self.assertAlmostEqual(
            medians[2], 1000 / (1 + math.exp(-0.7)), delta=3
        )
        self.assertLess(medians[0], medians[1])
        self.assertLess(medians[1], medians[2])
        # Before the sigmoid transform, T > 600 means Z < log(0.4 / 0.6).
        expected = 0.5 * (
            1 + math.erf((math.log(0.4 / 0.6) + 0.7) / (0.9 * math.sqrt(2)))
        )
        self.assertAlmostEqual(
            (values > 600).float().mean().item(), expected, delta=0.005
        )

    def test_normal_samples_have_configured_mean_and_standard_deviation(self):
        torch.manual_seed(7419)
        values = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
            100000,
            "cpu",
            "sigmoid",
            timestep_shift=-0.7,
            timestep_scale=0.9,
        )
        # Invert the transform to check the distribution independently.
        normal = torch.logit(1 - values.double() / 1000)
        self.assertAlmostEqual(normal.mean().item(), -0.7, delta=0.01)
        self.assertAlmostEqual(normal.std().item(), 0.9, delta=0.01)

    def test_inference_shift_and_previous_schedule_do_not_change_samples(self):
        for dynamic, invert in ((False, False), (False, True), (True, False)):
            scheduler = CustomFlowMatchEulerDiscreteScheduler(
                shift=3.0,
                use_dynamic_shifting=dynamic,
                invert_sigmas=invert,
            )
            scheduler.set_train_timesteps(
                1000,
                "cpu",
                "shift",
                latents=torch.zeros(1, 16, 64, 64),
                patch_size=2,
            )
            torch.manual_seed(74)
            actual = scheduler.set_train_timesteps(
                1000,
                "cpu",
                "sigmoid",
                timestep_shift=-0.7,
                timestep_scale=0.9,
            )
            torch.manual_seed(74)
            expected = (
                CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
                    1000,
                    "cpu",
                    "sigmoid",
                    timestep_shift=-0.7,
                    timestep_scale=0.9,
                )
            )
            self.assertTrue(torch.equal(actual, expected))

    def test_parent_inference_schedule_is_unchanged(self):
        for dynamic, invert in ((False, False), (False, True), (True, False)):
            kwargs = {
                "shift": 3.0,
                "use_dynamic_shifting": dynamic,
                "invert_sigmas": invert,
            }
            scheduler = CustomFlowMatchEulerDiscreteScheduler(**kwargs)
            scheduler.set_train_timesteps(
                1000, "cpu", "sigmoid", timestep_shift=-0.7, timestep_scale=0.9
            )
            reference = FlowMatchEulerDiscreteScheduler(**kwargs)
            inference = {"mu": 0.9} if dynamic else {}
            scheduler.set_timesteps(7, device="cpu", **inference)
            reference.set_timesteps(7, device="cpu", **inference)
            self.assertTrue(
                torch.equal(scheduler.timesteps, reference.timesteps)
            )
            self.assertTrue(torch.equal(scheduler.sigmas, reference.sigmas))

    def test_other_training_modes_ignore_sigmoid_parameters(self):
        for kind in (
            "linear",
            "weighted",
            "shift",
            "flux_shift",
            "lumina2_shift",
            "lognorm_blend",
        ):
            with self.subTest(kind=kind):
                plain = CustomFlowMatchEulerDiscreteScheduler(shift=3.0)
                configured = CustomFlowMatchEulerDiscreteScheduler(shift=3.0)
                torch.manual_seed(74)
                expected = plain.set_train_timesteps(1000, "cpu", kind)
                rng = torch.get_rng_state()
                torch.manual_seed(74)
                actual = configured.set_train_timesteps(
                    1000,
                    "cpu",
                    kind,
                    timestep_shift=float("nan"),
                    timestep_scale=-1,
                )
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(configured.sigmas, plain.sigmas))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_yaml_round_trip_and_defaults(self):
        for document, shift, scale in (
            ("timestep_type: sigmoid", 0.0, 1.0),
            ("timestep_type: sigmoid\ntimestep_shift: -0.7", -0.7, 1.0),
            ("timestep_type: sigmoid\ntimestep_scale: 0.9", 0.0, 0.9),
            (
                "timestep_type: sigmoid\n"
                "timestep_shift: -0.7\ntimestep_scale: 0.9",
                -0.7,
                0.9,
            ),
        ):
            config = TrainConfig(
                **yaml.safe_load(yaml.safe_dump(yaml.safe_load(document)))
            )
            validate_configs(
                config,
                ModelConfig(arch="flux", name_or_path="unused"),
                SaveConfig(),
                [],
            )
            self.assertEqual(config.timestep_type, "sigmoid")
            self.assertEqual(config.timestep_shift, shift)
            self.assertEqual(config.timestep_scale, scale)

    def test_effective_timestep_type_controls_parameter_forwarding(self):
        config = TrainConfig(
            timestep_type="sigmoid", timestep_shift=-0.7, timestep_scale=0.9
        )
        self.assertEqual(
            config.get_timestep_kwargs("sigmoid"),
            {"timestep_shift": -0.7, "timestep_scale": 0.9},
        )
        # A linear override ignores the saved sigmoid settings.
        for kind in ("linear", "weighted", "shift", "next_sample"):
            with self.subTest(kind=kind):
                self.assertEqual(config.get_timestep_kwargs(kind), {})

    def test_schedule_resolves_absolute_steps_after_yaml_round_trip(self):
        document = """
timestep_shift: -0.4
timestep_scale: 0.3
timestep_sigmoid_schedule:
  - {start_step: 4, shift: 1, scale: 2}
  - {start_step: 12, shift: -1, scale: 1}
  - {start_step: 16, shift: 1, scale: 2}
"""
        for step, shift, scale in (
            (
                13,
                -0.5,
                1.25,
            ),  # A resumed run may first resolve an interior point.
            (3, -0.4, 0.3),
            (4, 1, 2),
            (8, 0, 1.5),
            (11, -0.75, 1.125),
            (12, -1, 1),
            (16, 1, 2),
            (1200, 1, 2),
        ):
            with self.subTest(step=step):
                config = TrainConfig(
                    **yaml.safe_load(yaml.safe_dump(yaml.safe_load(document)))
                )
                self.assertEqual(
                    config.get_timestep_kwargs("sigmoid", step),
                    {"timestep_shift": shift, "timestep_scale": scale},
                )

    def test_zero_start_overrides_scalar_defaults(self):
        config = TrainConfig(
            timestep_sigmoid_schedule=[
                {"start_step": 0, "shift": 1, "scale": 2}
            ]
        )
        self.assertEqual(
            config.get_timestep_kwargs("sigmoid", 0),
            {"timestep_shift": 1, "timestep_scale": 2},
        )

    def test_single_point_holds_constant_from_its_start(self):
        config = TrainConfig(
            timestep_sigmoid_schedule=[
                {"start_step": 4, "shift": 2, "scale": 3}
            ]
        )
        for step, shift, scale in ((3, 0, 1), (4, 2, 3), (100, 2, 3)):
            with self.subTest(step=step):
                self.assertEqual(
                    config.get_timestep_kwargs("sigmoid", step),
                    {"timestep_shift": shift, "timestep_scale": scale},
                )

    def test_interpolation_keeps_finite_shift_and_positive_scale(self):
        smallest_scale = math.ulp(0.0)
        config = TrainConfig(
            timestep_sigmoid_schedule=[
                {
                    "start_step": 0,
                    "shift": -sys.float_info.max,
                    "scale": smallest_scale,
                },
                {
                    "start_step": 2,
                    "shift": sys.float_info.max,
                    "scale": smallest_scale,
                },
            ]
        )
        self.assertEqual(
            config.get_timestep_kwargs("sigmoid", 1),
            {"timestep_shift": 0.0, "timestep_scale": smallest_scale},
        )

    def test_empty_schedule_preserves_default_and_scalar_samples_and_rng(self):
        for shift, scale in ((0, 1), (-0.4, 0.3)):
            with self.subTest(shift=shift, scale=scale):
                scalar = {"timestep_shift": shift, "timestep_scale": scale}
                empty = TrainConfig(timestep_sigmoid_schedule=[], **scalar)
                legacy = TrainConfig(**scalar)
                scheduler = CustomFlowMatchEulerDiscreteScheduler()
                torch.manual_seed(74)
                expected = scheduler.set_train_timesteps(
                    1000,
                    "cpu",
                    "sigmoid",
                    **legacy.get_timestep_kwargs("sigmoid", 12),
                )
                expected_rng = torch.get_rng_state()
                torch.manual_seed(74)
                actual = scheduler.set_train_timesteps(
                    1000,
                    "cpu",
                    "sigmoid",
                    **empty.get_timestep_kwargs("sigmoid", 12),
                )
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(
                    torch.equal(torch.get_rng_state(), expected_rng)
                )

    def test_schedule_rejects_invalid_rows_and_field_types(self):
        valid = {"start_step": 4, "shift": 0, "scale": 1}
        for schedule, field in (
            (None, "list"),
            ({}, "list"),
            (["row"], r"\[0\]"),
            ([{"shift": 0, "scale": 1}], "start_step"),
            ([{"start_step": 0, "scale": 1}], "shift"),
            ([{"start_step": 0, "shift": 0}], "scale"),
            ([{**valid, "scales": 1}], "scales"),
            ([{**valid, "start_step": True}], "start_step"),
            ([{**valid, "start_step": -1}], "start_step"),
            ([{**valid, "start_step": 1.0}], "start_step"),
            ([{**valid, "shift": False}], "shift"),
            ([{**valid, "shift": "1"}], "shift"),
            ([{**valid, "shift": float("inf")}], "shift"),
            ([{**valid, "shift": 10**400}], "shift"),
            ([{**valid, "scale": True}], "scale"),
            ([{**valid, "scale": "1"}], "scale"),
            ([{**valid, "scale": float("nan")}], "scale"),
            ([{**valid, "scale": float("inf")}], "scale"),
            ([{**valid, "scale": 0}], "scale"),
            ([{**valid, "scale": -1}], "scale"),
            ([valid, valid], r"\[1\].*strictly increasing"),
            (
                [valid, {**valid, "start_step": 3}],
                r"\[1\].*strictly increasing",
            ),
        ):
            with self.subTest(schedule=schedule):
                with self.assertRaisesRegex(
                    ValueError, rf"timestep_sigmoid_schedule.*{field}"
                ):
                    TrainConfig(timestep_sigmoid_schedule=schedule)

    def test_schedule_conflicts_with_fixed_grid_only_on_sigmoid_path(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        scheduler.fixed_train_sigmas = [1.0, 0.7, 0.2]
        config = TrainConfig(
            timestep_sigmoid_schedule=[
                {"start_step": 4, "shift": 1, "scale": 2}
            ]
        )
        with self.assertRaisesRegex(
            ValueError, "timestep_sigmoid_schedule.*fixed_train_sigmas"
        ):
            config.get_timestep_kwargs("sigmoid", 0, scheduler=scheduler)
        self.assertEqual(
            config.get_timestep_kwargs("linear", 12, scheduler=scheduler), {}
        )
        self.assertEqual(
            TrainConfig().get_timestep_kwargs(
                "sigmoid", 12, scheduler=scheduler
            ),
            {"timestep_shift": 0.0, "timestep_scale": 1.0},
        )

    def test_schedule_requires_flowmatch_sigmoid_without_linear_override(self):
        schedule = [{"start_step": 0, "shift": 1, "scale": 2}]
        model = ModelConfig(arch="flux", name_or_path="unused")
        save = SaveConfig()
        options = {"noise_scheduler": "flowmatch", "timestep_type": "sigmoid"}
        validate_configs(
            TrainConfig(**options, timestep_sigmoid_schedule=schedule),
            model,
            save,
            [],
        )
        for overrides in (
            {"noise_scheduler": "ddpm"},
            {"timestep_type": "shift"},
            {"linear_timesteps": True},
            {"linear_timesteps2": True},
        ):
            with self.subTest(overrides=overrides):
                config = {**options, **overrides}
                with self.assertRaisesRegex(
                    ValueError, "timestep_sigmoid_schedule"
                ):
                    validate_configs(
                        TrainConfig(
                            **config, timestep_sigmoid_schedule=schedule
                        ),
                        model,
                        save,
                        [],
                    )
                validate_configs(TrainConfig(**config), model, save, [])
                validate_configs(
                    TrainConfig(**config, timestep_sigmoid_schedule=[]),
                    model,
                    save,
                    [],
                )

    def test_training_calls_forward_parameters_only_for_sigmoid(self):
        config = TrainConfig(
            timestep_shift=-0.4,
            timestep_scale=0.3,
            timestep_sigmoid_schedule=[
                {"start_step": 4, "shift": 1, "scale": 2},
                {"start_step": 12, "shift": -1, "scale": 1},
            ],
            max_denoising_steps=128,
        )
        for path in (
            "jobs/process/BaseSDTrainProcess.py",
            "jobs/process/TrainSliderProcess.py",
            "toolkit/guidance.py",
        ):
            call = _training_scheduler_call(path)
            for kind, step, shift, scale in (
                ("sigmoid", 3, -0.4, 0.3),
                ("sigmoid", 4, 1, 2),
                ("sigmoid", 4, 1, 2),  # A second microbatch in the same step.
                ("sigmoid", 8, 0, 1.5),
                ("sigmoid", 12, -1, 1),
                ("linear", 12, None, None),
                ("shift", 12, None, None),
            ):
                with self.subTest(path=path, kind=kind, step=step):
                    scheduler = _RecordingScheduler()
                    sd = SimpleNamespace(noise_scheduler=scheduler)
                    process = SimpleNamespace(
                        sd=sd,
                        train_config=config,
                        device_torch=torch.device("cpu"),
                        step_num=step,
                    )
                    latents = torch.zeros(1, 16, 32, 32)
                    scope = {
                        "self": process,
                        "sd": sd,
                        "train_config": config,
                        "timestep_type": kind,
                        "step_num": step,
                        "num_train_timesteps": 128,
                        "device": "cpu",
                        "latents": latents,
                        "conditional_latents": latents,
                        "l": latents,
                        "patch_size": 2,
                    }
                    torch.manual_seed(74)
                    actual = eval(call, scope)
                    forwarded = scheduler.received_kwargs
                    if kind == "sigmoid":
                        actual_rng = torch.get_rng_state()
                        torch.manual_seed(74)
                        expected = torch.sort(
                            (
                                1
                                - (
                                    torch.randn(len(actual)) * scale + shift
                                ).sigmoid()
                            )
                            * 1000,
                            descending=True,
                        ).values
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertTrue(
                            torch.equal(torch.get_rng_state(), actual_rng)
                        )
                    else:
                        self.assertNotIn("timestep_shift", forwarded)
                        self.assertNotIn("timestep_scale", forwarded)

    def test_polarity_guidance_uses_trainer_step(self):
        # Run the guidance chain without importing the model.
        scope = {
            "torch": torch,
            "Optional": Optional,
            "PromptEmbeds": PromptEmbeds,
            "concat_prompt_embeds": concat_prompt_embeds,
            "get_torch_dtype": get_torch_dtype,
        }
        for path, names in (
            (
                "toolkit/guidance.py",
                ("get_guidance_loss", "get_guided_loss_polarity"),
            ),
            (
                "extensions_built_in/sd_trainer/SDTrainer.py",
                ("get_guided_loss",),
            ),
        ):
            tree = ast.parse((ROOT / path).read_text())
            functions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in names
            ]
            exec(
                compile(
                    ast.Module(body=functions, type_ignores=[]), path, "exec"
                ),
                scope,
            )

        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        sd = SimpleNamespace(
            torch_dtype="float32",
            device_torch=torch.device("cpu"),
            is_flow_matching=True,
            noise_scheduler=scheduler,
            add_noise=scheduler.add_noise,
            condition_noisy_latents=lambda latents, batch: latents,
            unet=torch.nn.Identity(),
            network=SimpleNamespace(),
            predict_noise=lambda latents, **kwargs: latents.requires_grad_(),
        )
        trainer = SimpleNamespace(
            sd=sd,
            step_num=8,
            train_config=TrainConfig(
                timestep_sigmoid_schedule=[
                    {"start_step": 4, "shift": 1, "scale": 2},
                    {"start_step": 12, "shift": -1, "scale": 1},
                ],
            ),
        )
        latents = torch.zeros(1, 1, 2, 2)
        batch = SimpleNamespace(
            latents=latents,
            unconditional_latents=latents,
            file_items=[
                SimpleNamespace(
                    dataset_config=SimpleNamespace(guidance_type="polarity")
                )
            ],
        )
        torch.manual_seed(74)
        scope["get_guided_loss"](
            trainer,
            noisy_latents=latents,
            conditional_embeds=PromptEmbeds(torch.zeros(1, 1, 1)),
            match_adapter_assist=False,
            network_weight_list=[1],
            timesteps=torch.tensor([500.0]),
            pred_kwargs={},
            batch=batch,
            noise=torch.ones_like(latents),
        )
        torch.manual_seed(74)
        expected = torch.sort(
            (1 - (torch.randn(1000) * 1.5).sigmoid()) * 1000,
            descending=True,
        ).values
        self.assertTrue(torch.equal(scheduler.timesteps, expected))

    def test_ming_override_forwards_parameters_with_dynamic_shifting(self):
        # Load the override without the text encoder or model weights.
        path = "extensions_built_in/diffusion_models/ming_image/ming_image.py"
        tree = ast.parse((ROOT / path).read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "MingImageFlowMatchScheduler"
        )
        shift_params = mock.Mock(return_value=(1.35, 4096))
        scheduler_class = CustomFlowMatchEulerDiscreteScheduler
        scope = {
            scheduler_class.__name__: scheduler_class,
            "ming_shift_params": shift_params,
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[cls], type_ignores=[])
                ),
                path,
                "exec",
            ),
            scope,
        )
        scheduler = scope[cls.name](use_dynamic_shifting=True)
        torch.manual_seed(74)
        actual = scheduler.set_train_timesteps(
            1000,
            "cpu",
            "sigmoid",
            latents=torch.zeros(1, 16, 32, 32),
            patch_size=2,
            timestep_shift=-0.4,
            timestep_scale=0.3,
        )
        shift_params.assert_called_once_with(256)
        self.assertEqual(scheduler.config.max_shift, 1.35)
        torch.manual_seed(74)
        expected = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
            1000,
            "cpu",
            "sigmoid",
            timestep_shift=-0.4,
            timestep_scale=0.3,
        )
        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_configured_samples_stay_on_cuda(self):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            values = (
                CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
                    1000,
                    "cuda",
                    "sigmoid",
                    timestep_shift=-0.7,
                    timestep_scale=0.9,
                )
            )
        self.assertEqual(values.device.type, "cuda")
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((values[:-1] >= values[1:]).all())

    @unittest.skipUnless(
        torch.backends.mps.is_available(), "MPS is not available"
    )
    def test_configured_samples_stay_on_mps(self):
        self.addCleanup(torch.mps.set_rng_state, torch.mps.get_rng_state())
        values = CustomFlowMatchEulerDiscreteScheduler().set_train_timesteps(
            1000,
            "mps",
            "sigmoid",
            timestep_shift=-0.7,
            timestep_scale=0.9,
        )
        self.assertEqual(values.device.type, "mps")
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((values[:-1] >= values[1:]).all())


if __name__ == "__main__":
    unittest.main()
