from types import SimpleNamespace

import torch

from toolkit.memory_management.arena_offload.cap_calibrator import (
    CAP_SET,
    TrainingCapCalibrator,
)
from toolkit.memory_management.arena_offload.policy import (
    ShapePeak,
    TrainingSignalWindow,
)
from toolkit.memory_management.arena_offload.runtime import (
    ArenaOffloadRuntime,
    _SamplingCapProfile,
    _sampling_cold_working_bytes,
    _sampling_config_shape_key,
)
from toolkit.memory_management.immutable_runtime import ImmutableTransformerRuntime


def config(**overrides):
    values = {
        "width": 1024,
        "height": 1024,
        "num_frames": 1,
        "guidance_scale": 4.5,
        "batch_cfg": False,
        "ctrl_idx": None,
        "ctrl_img": None,
        "ctrl_img_1": None,
        "ctrl_img_2": None,
        "ctrl_img_3": None,
        "extra_values": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sampling_config_key_is_generic_and_shape_conservative():
    base = config()
    assert _sampling_config_shape_key(base) == _sampling_config_shape_key(config())
    assert _sampling_config_shape_key(base) != _sampling_config_shape_key(
        config(guidance_scale=1.0)
    )
    assert _sampling_config_shape_key(base) != _sampling_config_shape_key(
        config(ctrl_img="reference.png", ctrl_img_1="reference.png")
    )
    assert _sampling_cold_working_bytes(
        config(ctrl_img="reference.png", ctrl_img_1="reference.png")
    ) > _sampling_cold_working_bytes(base)


def test_only_repeated_or_previously_seen_profiles_enable_calibration():
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._cap_calibration_enabled = True
    runtime._sampling_cap_profiles = {}
    key = _sampling_config_shape_key(config())

    assert runtime._sampling_profile(key, 1) is None
    repeated = runtime._sampling_profile(key, 2)
    assert repeated is not None
    assert repeated.calibrator.enabled
    assert runtime._sampling_profile(key, 1) is repeated


def test_sampling_session_preflights_configs_and_restores_cap_before_decode():
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._config = SimpleNamespace(fp8_sampling=False)
    runtime._canonical_modules = ()
    runtime._smart_plan = {}
    runtime._training_plan = object()
    runtime._executor = SimpleNamespace(
        TRAIN="train", activate=lambda *_args: None
    )
    runtime._bind_training_cap = lambda: None
    restored = []
    runtime._bind_sampling_cap = lambda target=None, **_kwargs: restored.append(target)
    class Vae(torch.nn.Module):
        def decode(self, value):
            return ("decoded", value)

    owner = SimpleNamespace(vae=Vae())
    first = config()
    second = config()

    with runtime.sampling_session(
        gen_configs=[first, second], generation_owner=owner
    ):
        key = _sampling_config_shape_key(first)
        assert runtime._sampling_session_occurrences[key] == 2
        assert owner.vae.decode("latent") == ("decoded", "latent")

    assert restored == [None]
    assert "decode" not in owner.vae.__dict__


def test_sampling_forward_begin_uses_completed_generic_fsm(monkeypatch):
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    key = _sampling_config_shape_key(config())
    profile = _SamplingCapProfile(
        calibrator=TrainingCapCalibrator(enabled=True, notch_bytes=100),
        signals=TrainingSignalWindow(),
    )
    profile.signals._shape_peaks[key] = ShapePeak(
        steps=2, working_peak_bytes=400
    )
    runtime._active_sampling_cap_profile = profile
    runtime._active_sampling_shape_key = key
    runtime._device = "cuda"
    runtime._residency = SimpleNamespace(resident_bytes=lambda: 100)
    runtime._smart_plan = {}
    runtime._active_sampling_resident_bytes = 100
    runtime._active_sampling_ring_bytes = 0
    runtime._training_ring_bytes = lambda: 0
    runtime._sampling_cliff_cap_bytes = lambda: 1000
    targets = []
    counter_snapshots = []
    runtime._bind_sampling_cap = lambda target, **kwargs: targets.append(
        (target, kwargs)
    )
    runtime._sampling_allocator_counters = lambda: {
        "num_alloc_retries": 0,
        "num_device_alloc": 12,
        "num_device_free": 5,
    }
    original_prime = profile.signals.prime_counters

    def record_prime(**kwargs):
        counter_snapshots.append(kwargs)
        original_prime(**kwargs)

    profile.signals.prime_counters = record_prime
    monkeypatch.setattr(
        "toolkit.memory_management.arena_offload.runtime.allocator_cap.applied_cap_bytes",
        lambda _device: 1000,
    )
    monkeypatch.setattr(
        "torch.cuda.reset_peak_memory_stats", lambda _device: None
    )

    runtime._sampling_forward_begin()

    assert profile.calibrator.last_action == CAP_SET
    assert targets == [(700, {"reclaim": True})]
    assert counter_snapshots == [
        {
            "allocator_counters": {
                "num_alloc_retries": 0,
                "num_device_alloc": 12,
                "num_device_free": 5,
            }
        }
    ]


def test_per_forward_resets_preserve_the_image_wide_sampling_peak():
    executor = ImmutableTransformerRuntime.__new__(ImmutableTransformerRuntime)
    executor._sampling_baseline = {
        "external_peak_allocated": 100,
        "external_peak_reserved": 120,
    }

    executor.record_sampling_peak(allocated_bytes=300, reserved_bytes=350)
    executor.record_sampling_peak(allocated_bytes=250, reserved_bytes=320)

    assert executor._sampling_baseline["external_peak_allocated"] == 300
    assert executor._sampling_baseline["external_peak_reserved"] == 350
