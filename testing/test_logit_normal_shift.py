"""Standalone unit tests for `timestep_type='logit_normal_shift'` in
`toolkit/samplers/custom_flowmatch_sampler.py`.

Runs with system `python3` (stdlib only) by providing a lightweight, faithful
mock of the subset of `torch`, `numpy`, and `diffusers` used by
`CustomFlowMatchEulerDiscreteScheduler`.
"""

import importlib.util
import math
import os
import random
import statistics
import sys
import types
import unittest


class _FakeTensor:
    def __init__(self, data, device="cpu", dtype="float32"):
        self._data = [float(x) for x in data]
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, idx):
        res = self._data[idx]
        if isinstance(res, list):
            return _FakeTensor(res, self.device, self.dtype)
        return res

    def __setitem__(self, idx, val):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self._data)))
            if isinstance(val, (int, float)):
                for i in indices:
                    self._data[i] = float(val)
            else:
                for i, v in zip(indices, val):
                    self._data[i] = float(v)
        else:
            self._data[idx] = float(val)

    @property
    def shape(self):
        return (len(self._data),)

    def to(self, *args, **kwargs):
        device = kwargs.get("device", self.device)
        dtype = kwargs.get("dtype", self.dtype)
        if args:
            if isinstance(args[0], str) and args[0] in ("float32", "float64", "int64"):
                dtype = args[0]
            else:
                device = args[0]
        return _FakeTensor(self._data, device=device, dtype=dtype)

    def clamp(self, min_val, max_val):
        return _FakeTensor(
            [min(max(x, min_val), max_val) for x in self._data],
            self.device,
            self.dtype,
        )

    def min(self):
        return min(self._data)

    def max(self):
        return max(self._data)

    def sum(self):
        return sum(self._data)

    def __add__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor([a + b for a, b in zip(self._data, other._data)], self.device, self.dtype)
        return _FakeTensor([a + other for a in self._data], self.device, self.dtype)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor([a - b for a, b in zip(self._data, other._data)], self.device, self.dtype)
        return _FakeTensor([a - other for a in self._data], self.device, self.dtype)

    def __rsub__(self, other):
        return _FakeTensor([other - a for a in self._data], self.device, self.dtype)

    def __mul__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor([a * b for a, b in zip(self._data, other._data)], self.device, self.dtype)
        return _FakeTensor([a * other for a in self._data], self.device, self.dtype)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor([a / b for a, b in zip(self._data, other._data)], self.device, self.dtype)
        return _FakeTensor([a / other for a in self._data], self.device, self.dtype)

    def __rtruediv__(self, other):
        return _FakeTensor([other / a for a in self._data], self.device, self.dtype)

    def __pow__(self, power):
        return _FakeTensor([a ** power for a in self._data], self.device, self.dtype)


class _FakeLatents:
    def __init__(self, b, c, h, w):
        self.shape = (b, c, h, w)


class _ConfigDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)


def _install_stubs():
    """Install minimal stdlib stubs for torch/numpy/diffusers if unavailable."""
    saved = {}
    for name in (
        "torch",
        "torch.distributions",
        "numpy",
        "diffusers",
        "toolkit.timestep_weighing.default_weighing_scheme",
    ):
        saved[name] = sys.modules.get(name)

    rng = random.Random(12345)
    state = {"fixed_randn": None}

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.int = "int64"
    fake_torch.Tensor = _FakeTensor
    fake_torch.no_grad = lambda: _NoGrad()
    fake_torch.arange = lambda n, dtype="float32": _FakeTensor(range(n), dtype=dtype)
    fake_torch.exp = lambda t: _FakeTensor([math.exp(x) for x in t], t.device, t.dtype)
    fake_torch.linspace = lambda start, end, steps, device="cpu": _FakeTensor(
        [start + (end - start) * i / max(1, steps - 1) for i in range(steps)],
        device=device,
    )
    fake_torch.tensor = lambda data, dtype="float32", device="cpu": _FakeTensor(
        data, device=device, dtype=dtype
    )
    fake_torch.zeros = lambda n, device="cpu", dtype="float32": _FakeTensor(
        [0.0] * n, device=device, dtype=dtype
    )
    fake_torch.ones = lambda n, device="cpu", dtype="float32": _FakeTensor(
        [1.0] * n, device=device, dtype=dtype
    )
    fake_torch.sigmoid = lambda t: _FakeTensor(
        [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))) for x in t],
        t.device,
        t.dtype,
    )

    def _randn(shape, device="cpu", dtype="float32"):
        n = shape[0] if isinstance(shape, tuple) else int(shape)
        if state["fixed_randn"] is not None:
            vals = list(state["fixed_randn"][:n])
        else:
            vals = [rng.gauss(0.0, 1.0) for _ in range(n)]
        return _FakeTensor(vals, device=device, dtype=dtype)

    fake_torch.randn = _randn
    fake_torch.sort = lambda t, descending=False: (
        _FakeTensor(sorted(t._data, reverse=descending), t.device, t.dtype),
        None,
    )
    fake_torch.cat = lambda seq: _FakeTensor(
        [x for t in seq for x in t._data], seq[0].device, seq[0].dtype
    )
    fake_torch.from_numpy = lambda arr: _FakeTensor(list(arr))
    fake_torch._test_rng = rng
    fake_torch._test_state = state

    fake_dist = types.ModuleType("torch.distributions")
    fake_dist.LogNormal = object

    class _FakeNDArray(list):
        pass

    fake_np = types.ModuleType("numpy")
    fake_np.ndarray = _FakeNDArray
    fake_np.linspace = lambda a, b, n: _FakeNDArray(
        [a + (b - a) * i / max(1, n - 1) for i in range(n)]
    )

    class _FakeFlowMatchScheduler:
        def __init__(self, **kwargs):
            defaults = {
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "use_dynamic_shifting": False,
                "base_image_seq_len": 256,
                "max_image_seq_len": 4096,
                "base_shift": 0.5,
                "max_shift": 1.16,
                "shift_terminal": None,
                "time_shift_type": "exponential",
                "invert_sigmas": False,
                "use_karras_sigmas": False,
                "use_exponential_sigmas": False,
                "use_beta_sigmas": False,
            }
            defaults.update(kwargs)
            self.config = _ConfigDict(defaults)
            self.shift = self.config.shift
            self.sigma_max = 1.0
            self.sigma_min = 1.0 / self.config.num_train_timesteps

        def time_shift(self, mu, sigma, t):
            if self.config.time_shift_type == "exponential":
                return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)
            elif self.config.time_shift_type == "linear":
                return mu / (mu + (1.0 / t - 1.0) ** sigma)
            raise ValueError(f"Unsupported time_shift_type: {self.config.time_shift_type}")

        def stretch_shift_to_terminal(self, sigmas):
            raise AssertionError("stretch_shift_to_terminal should not be called for logit_normal_shift")

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.FlowMatchEulerDiscreteScheduler = _FakeFlowMatchScheduler

    fake_weighing = types.ModuleType("toolkit.timestep_weighing.default_weighing_scheme")
    fake_weighing.default_weighing_scheme = [1.0] * 1000
    fake_weighing.x0_weighing_scheme = [1.0] * 1000

    sys.modules["torch"] = fake_torch
    sys.modules["torch.distributions"] = fake_dist
    sys.modules["numpy"] = fake_np
    sys.modules["diffusers"] = fake_diffusers
    sys.modules["toolkit.timestep_weighing.default_weighing_scheme"] = fake_weighing

    sampler_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "toolkit",
        "samplers",
        "custom_flowmatch_sampler.py",
    )
    spec = importlib.util.spec_from_file_location("_custom_flowmatch_sampler_under_test", sampler_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, fake_torch, saved


def _restore_stubs(saved):
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


QWEN_21_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "time_shift_type": "exponential",
    "use_dynamic_shifting": True,
}


class TestLogitNormalShift(unittest.TestCase):
    def setUp(self):
        self.mod, self.fake_torch, self._saved = _install_stubs()
        self.Scheduler = self.mod.CustomFlowMatchEulerDiscreteScheduler

    def tearDown(self):
        _restore_stubs(self._saved)

    def test_exact_equivalence_to_fizgig_exponential_shift(self):
        """Matches Fizgig's `exp(mu) / (exp(mu) + (1/sigmoid(z) - 1)) == sigmoid(z + mu)`."""
        sched = self.Scheduler(**QWEN_21_SCHEDULER_CONFIG)
        # 1024x1024 with 16x VAE -> 64x64 = 4096 latent tokens
        latents = _FakeLatents(1, 64, 64, 64)
        zs = [-2.5, -1.0, 0.0, 0.75, 2.0]
        self.fake_torch._test_state["fixed_randn"] = zs

        ts = sched.set_train_timesteps(
            len(zs), device="cuda", timestep_type="logit_normal_shift", latents=latents, patch_size=1
        )

        # Qwen 2.1 mu at 4096 tokens: 0.5 + (4096 - 256) * (0.9 - 0.5) / (8192 - 256)
        expected_mu = 0.5 + (4096 - 256) * (0.9 - 0.5) / (8192 - 256)
        expected = sorted([1000.0 / (1.0 + math.exp(-(z + expected_mu))) for z in zs], reverse=True)

        self.assertEqual(len(ts), len(zs))
        self.assertEqual(len(sched.sigmas), len(zs) + 1)
        self.assertEqual(sched.sigmas[-1], 0.0)
        self.assertEqual(ts.device, "cuda")
        for got, exp in zip(ts, expected):
            self.assertAlmostEqual(got, exp, places=4)

    def test_higher_resolution_shifts_distribution_higher(self):
        """Larger latent grids produce larger mu and a higher mean timestep."""
        sched = self.Scheduler(**QWEN_21_SCHEDULER_CONFIG)
        rng = random.Random(42)
        zs = [rng.gauss(0.0, 1.0) for _ in range(1000)]
        self.fake_torch._test_state["fixed_randn"] = zs

        ts_512 = sched.set_train_timesteps(
            1000, device="cpu", timestep_type="logit_normal_shift", latents=_FakeLatents(1, 64, 32, 32)
        )
        ts_1024 = sched.set_train_timesteps(
            1000, device="cpu", timestep_type="logit_normal_shift", latents=_FakeLatents(1, 64, 64, 64)
        )

        mean_512 = statistics.mean(ts_512)
        mean_1024 = statistics.mean(ts_1024)
        self.assertGreater(mean_1024, mean_512)
        # Centered well into mid-high noise (mu ~ 0.54 vs ~ 0.69), not near 500 or extremes
        self.assertGreater(mean_512, 580.0)
        self.assertGreater(mean_1024, 610.0)
        # Descending order preserved
        self.assertTrue(all(ts_1024[i] >= ts_1024[i + 1] for i in range(len(ts_1024) - 1)))

    def test_patch_size_scales_sequence_length(self):
        sched = self.Scheduler(**QWEN_21_SCHEDULER_CONFIG)
        self.fake_torch._test_state["fixed_randn"] = [0.0]

        t_p1 = sched.set_train_timesteps(
            1, device="cpu", timestep_type="logit_normal_shift", latents=_FakeLatents(1, 64, 64, 64), patch_size=1
        )[0]
        t_p2 = sched.set_train_timesteps(
            1, device="cpu", timestep_type="logit_normal_shift", latents=_FakeLatents(1, 64, 64, 64), patch_size=2
        )[0]
        # patch_size=2 reduces seq_len 4096 -> 1024, lowering mu
        self.assertLess(t_p2, t_p1)

    def test_static_shift_without_latents(self):
        sched = self.Scheduler(use_dynamic_shifting=False, shift=3.0, num_train_timesteps=1000)
        self.fake_torch._test_state["fixed_randn"] = [0.0]  # u = 0.5
        ts = sched.set_train_timesteps(
            1, device="cpu", timestep_type="logit_normal_shift", latents=None
        )
        # 3 * 0.5 / (1 + 2 * 0.5) = 1.5 / 2.0 = 0.75 -> 750.0
        self.assertAlmostEqual(ts[0], 750.0, places=4)
        self.assertAlmostEqual(sched.sigmas[0], 0.75, places=6)
        self.assertEqual(sched.sigmas[1], 0.0)

    def test_dynamic_shift_requires_latents(self):
        sched = self.Scheduler(**QWEN_21_SCHEDULER_CONFIG)
        with self.assertRaises(ValueError):
            sched.set_train_timesteps(
                10, device="cpu", timestep_type="logit_normal_shift", latents=None
            )

    def test_invert_sigmas(self):
        cfg = dict(QWEN_21_SCHEDULER_CONFIG, invert_sigmas=True)
        sched = self.Scheduler(**cfg)
        self.fake_torch._test_state["fixed_randn"] = [-1.0, 0.0, 1.0]
        ts = sched.set_train_timesteps(
            3, device="cpu", timestep_type="logit_normal_shift", latents=_FakeLatents(1, 64, 64, 64)
        )
        # Inverted sigmas ascend toward trailing 1.0, matching `shift` behavior
        self.assertTrue(ts[0] <= ts[1] <= ts[2])
        self.assertEqual(sched.sigmas[-1], 1.0)

    def test_extreme_outliers_are_finite(self):
        sched = self.Scheduler(**QWEN_21_SCHEDULER_CONFIG)
        self.fake_torch._test_state["fixed_randn"] = [-50.0, 50.0]
        ts = sched.set_train_timesteps(
            2, device="cpu", timestep_type="logit_normal_shift", latents=_FakeLatents(1, 64, 64, 64)
        )
        for v in ts:
            self.assertTrue(math.isfinite(v))
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1000.0)

    def test_fixed_train_sigmas_takes_precedence(self):
        sched = self.Scheduler(**QWEN_21_SCHEDULER_CONFIG)
        sched.fixed_train_sigmas = [1.0, 0.5, 0.25]
        ts = sched.set_train_timesteps(
            1000, device="cpu", timestep_type="logit_normal_shift", latents=None
        )
        self.assertEqual(list(ts), [1000.0, 500.0, 250.0])


if __name__ == "__main__":
    unittest.main()
