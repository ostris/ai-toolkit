import os
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extensions_built_in.diffusion_models.wan22.wan22_14b_i2v_model as wan22_i2v_module

from extensions_built_in.concept_slider.ConceptSliderTrainer import (
    ConceptSliderTrainer,
    ConceptSliderTrainerConfig,
)
from extensions_built_in.diffusion_models.wan22.wan22_14b_i2v_model import (
    Wan2214bI2VModel,
)
from extensions_built_in.diffusion_models.wan22.wan22_14b_model import (
    boundary_ratio_i2v,
)
from toolkit.prompt_utils import PromptEmbeds


class _FakeTransformer:
    def __init__(self, in_channels=26):
        self.hidden_states = []
        self.patch_embedding = SimpleNamespace(in_channels=in_channels)

    def __call__(self, hidden_states, **kwargs):
        self.hidden_states.append(hidden_states)
        return (hidden_states.clone().requires_grad_(True),)


class _FakeUnet:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _FakeNetwork:
    def __init__(self):
        self.is_active = True
        self.multipliers = []

    def set_multiplier(self, value):
        self.multipliers.append(value)


def _make_prompt_embeds():
    return PromptEmbeds(torch.zeros(1, 4))


def _make_i2v_model(in_channels=26):
    model = object.__new__(Wan2214bI2VModel)
    model.model = _FakeTransformer(in_channels=in_channels)
    model.vae = object()
    model.image_i2v_conditioning = False
    model.image_i2v_conditioning_prob = 0.2
    model.image_i2v_clip_training = False
    model.image_i2v_clip_training_prob = 0.25
    model.image_i2v_clip_num_frames = 5
    model.image_i2v_clip_blur_sigma = 24.0
    model.image_i2v_clip_downscale_factor = 0.0625
    return model


def test_i2v_multistage_boundaries_use_i2v_boundary():
    model = object.__new__(Wan2214bI2VModel)

    assert model._get_wan22_multistage_boundaries() == [boundary_ratio_i2v, 0.0]


def test_single_frame_batches_use_empty_conditioning_by_default(monkeypatch):
    model = _make_i2v_model()
    latent_model_input = torch.randn(1, 16, 1, 2, 2)
    batch = SimpleNamespace(
        tensor=torch.randn(1, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=1),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("single-frame conditioning should be skipped")

    monkeypatch.setattr(wan22_i2v_module, "add_first_frame_conditioning", fail_if_called)

    model.get_noise_prediction(
        latent_model_input=latent_model_input,
        timestep=torch.tensor([1]),
        text_embeddings=_make_prompt_embeds(),
        batch=batch,
    )

    hidden_states = model.model.hidden_states[-1]
    assert hidden_states.shape == (1, 26, 1, 2, 2)
    assert torch.equal(hidden_states[:, :16], latent_model_input)
    assert torch.count_nonzero(hidden_states[:, 16:]) == 0


def test_force_flag_keeps_single_frame_empty_conditioning(monkeypatch):
    model = _make_i2v_model()
    latent_model_input = torch.randn(1, 16, 1, 2, 2)
    batch = SimpleNamespace(
        tensor=torch.randn(1, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=1),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("single-frame conditioning should be skipped")

    monkeypatch.setattr(
        wan22_i2v_module,
        "add_first_frame_conditioning",
        fail_if_called,
    )

    model.get_noise_prediction(
        latent_model_input=latent_model_input,
        timestep=torch.tensor([1]),
        text_embeddings=_make_prompt_embeds(),
        batch=batch,
        force_t2i_single_frame=True,
    )

    hidden_states = model.model.hidden_states[-1]
    assert hidden_states.shape == (1, 26, 1, 2, 2)
    assert torch.equal(hidden_states[:, :16], latent_model_input)
    assert torch.count_nonzero(hidden_states[:, 16:]) == 0


def test_single_frame_opt_in_uses_degraded_first_frame_conditioning(monkeypatch):
    model = _make_i2v_model()
    model.image_i2v_conditioning = True
    model.image_i2v_conditioning_prob = 1.0
    latent_model_input = torch.randn(1, 16, 1, 2, 2)
    sentinel = torch.randn(1, 26, 1, 2, 2)
    calls = {}
    batch = SimpleNamespace(
        tensor=torch.randn(1, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=1),
    )

    def fake_degrade(first_frames):
        calls["degraded_input_shape"] = first_frames.shape
        return first_frames * 0.5

    def fake_add_first_frame_conditioning(**kwargs):
        calls["first_frame"] = kwargs["first_frame"]
        return sentinel

    monkeypatch.setattr(
        model,
        "degrade_image_i2v_conditioning",
        fake_degrade,
    )
    monkeypatch.setattr(
        wan22_i2v_module,
        "add_first_frame_conditioning",
        fake_add_first_frame_conditioning,
    )

    model.get_noise_prediction(
        latent_model_input=latent_model_input,
        timestep=torch.tensor([1]),
        text_embeddings=_make_prompt_embeds(),
        batch=batch,
    )

    assert calls["degraded_input_shape"] == (1, 3, 8, 8)
    assert torch.equal(calls["first_frame"], batch.tensor * 0.5)
    assert torch.equal(model.model.hidden_states[-1], sentinel)


def test_single_frame_opt_in_probability_zero_uses_empty_conditioning(monkeypatch):
    model = _make_i2v_model()
    model.image_i2v_conditioning = True
    model.image_i2v_conditioning_prob = 0.0
    latent_model_input = torch.randn(1, 16, 1, 2, 2)
    batch = SimpleNamespace(
        tensor=torch.randn(1, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=1),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("probability zero should skip first-frame conditioning")

    monkeypatch.setattr(wan22_i2v_module, "add_first_frame_conditioning", fail_if_called)

    model.get_noise_prediction(
        latent_model_input=latent_model_input,
        timestep=torch.tensor([1]),
        text_embeddings=_make_prompt_embeds(),
        batch=batch,
    )

    hidden_states = model.model.hidden_states[-1]
    assert hidden_states.shape == (1, 26, 1, 2, 2)
    assert torch.equal(hidden_states[:, :16], latent_model_input)
    assert torch.count_nonzero(hidden_states[:, 16:]) == 0


def test_video_batches_still_use_first_frame_conditioning(monkeypatch):
    model = _make_i2v_model()
    latent_model_input = torch.randn(1, 16, 3, 2, 2)
    sentinel = torch.randn(1, 26, 3, 2, 2)
    calls = {}
    batch = SimpleNamespace(
        tensor=torch.randn(1, 9, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=9),
    )

    def fake_add_first_frame_conditioning(**kwargs):
        calls["first_frame"] = kwargs["first_frame"]
        calls["first_frame_shape"] = kwargs["first_frame"].shape
        return sentinel

    monkeypatch.setattr(
        wan22_i2v_module,
        "add_first_frame_conditioning",
        fake_add_first_frame_conditioning,
    )

    model.get_noise_prediction(
        latent_model_input=latent_model_input,
        timestep=torch.tensor([1]),
        text_embeddings=_make_prompt_embeds(),
        batch=batch,
        force_t2i_single_frame=True,
    )

    assert calls["first_frame_shape"] == (1, 3, 8, 8)
    assert torch.equal(calls["first_frame"], batch.tensor[:, 0])
    assert torch.equal(model.model.hidden_states[-1], sentinel)


def test_auto_frame_video_batches_use_first_frame_conditioning(monkeypatch):
    model = _make_i2v_model()
    latent_model_input = torch.randn(1, 16, 3, 2, 2)
    sentinel = torch.randn(1, 26, 3, 2, 2)
    calls = {}
    batch = SimpleNamespace(
        tensor=torch.randn(1, 9, 3, 8, 8),
        num_frames=9,
        dataset_config=SimpleNamespace(num_frames=1, auto_frame_count=True),
    )

    def fake_add_first_frame_conditioning(**kwargs):
        calls["first_frame"] = kwargs["first_frame"]
        calls["first_frame_shape"] = kwargs["first_frame"].shape
        return sentinel

    monkeypatch.setattr(
        wan22_i2v_module,
        "add_first_frame_conditioning",
        fake_add_first_frame_conditioning,
    )

    model.get_noise_prediction(
        latent_model_input=latent_model_input,
        timestep=torch.tensor([1]),
        text_embeddings=_make_prompt_embeds(),
        batch=batch,
    )

    assert calls["first_frame_shape"] == (1, 3, 8, 8)
    assert torch.equal(calls["first_frame"], batch.tensor[:, 0])
    assert torch.equal(model.model.hidden_states[-1], sentinel)


def test_degrade_image_i2v_conditioning_preserves_shape_dtype_device_and_range():
    source = torch.linspace(-1.0, 1.0, steps=2 * 3 * 8 * 8, dtype=torch.float32)
    source = source.reshape(2, 3, 8, 8)

    degraded = Wan2214bI2VModel.degrade_image_i2v_conditioning(source)

    assert degraded.shape == source.shape
    assert degraded.dtype == source.dtype
    assert degraded.device == source.device
    assert degraded.min() >= -1.0
    assert degraded.max() <= 1.0


def test_image_clip_conditioning_preserves_shape_dtype_device_and_range():
    source = torch.linspace(-1.0, 1.0, steps=2 * 3 * 8 * 8, dtype=torch.float32)
    source = source.reshape(2, 3, 8, 8)

    conditioned = Wan2214bI2VModel.make_image_i2v_clip_conditioning(
        source,
        blur_sigma=24.0,
        downscale_factor=0.0625,
    )

    assert conditioned.shape == source.shape
    assert conditioned.dtype == source.dtype
    assert conditioned.device == source.device
    assert conditioned.min() >= -1.0
    assert conditioned.max() <= 1.0
    assert not torch.equal(conditioned, source)


def test_image_clip_training_preprocess_expands_single_images_to_clean_static_clip():
    model = _make_i2v_model()
    model.image_i2v_clip_training = True
    model.image_i2v_clip_training_prob = 1.0
    model.image_i2v_clip_num_frames = 9
    source = torch.linspace(-1.0, 1.0, steps=2 * 3 * 8 * 8, dtype=torch.float32)
    source = source.reshape(2, 3, 8, 8)
    batch = SimpleNamespace(
        tensor=source.clone(),
        latents=None,
        i2v_condition_tensor=None,
        num_frames=1,
        dataset_config=SimpleNamespace(num_frames=1),
    )

    processed = model.preprocess_training_batch(batch)

    assert processed.tensor.shape == (2, 9, 3, 8, 8)
    assert processed.num_frames == 9
    for frame_idx in range(9):
        assert torch.equal(processed.tensor[:, frame_idx], source)
    assert processed.i2v_condition_tensor.shape == source.shape
    assert processed.i2v_condition_tensor.dtype == source.dtype
    assert processed.i2v_condition_tensor.device == source.device
    assert processed.i2v_condition_tensor.min() >= -1.0
    assert processed.i2v_condition_tensor.max() <= 1.0
    assert not torch.equal(processed.i2v_condition_tensor, source)


def test_image_clip_training_probability_zero_keeps_single_image_batch():
    model = _make_i2v_model()
    model.image_i2v_clip_training = True
    model.image_i2v_clip_training_prob = 0.0
    source = torch.randn(1, 3, 8, 8)
    batch = SimpleNamespace(
        tensor=source,
        latents=None,
        i2v_condition_tensor=None,
        num_frames=1,
        dataset_config=SimpleNamespace(num_frames=1),
    )

    processed = model.preprocess_training_batch(batch)

    assert processed.tensor is source
    assert processed.num_frames == 1
    assert processed.i2v_condition_tensor is None


def test_image_clip_training_probability_values_clamp_to_unit_interval():
    assert Wan2214bI2VModel._clamp_probability(-1.0) == 0.0
    assert Wan2214bI2VModel._clamp_probability(0.5) == 0.5
    assert Wan2214bI2VModel._clamp_probability(2.0) == 1.0


def test_image_clip_num_frames_normalizes_to_wan_frame_count():
    assert Wan2214bI2VModel._normalize_clip_num_frames(0) == 1
    assert Wan2214bI2VModel._normalize_clip_num_frames(5) == 5
    assert Wan2214bI2VModel._normalize_clip_num_frames(80) == 81
    assert Wan2214bI2VModel._normalize_clip_num_frames(81) == 81


def test_image_clip_training_rejects_cached_single_image_latents():
    model = _make_i2v_model()
    model.image_i2v_clip_training = True
    model.image_i2v_clip_training_prob = 1.0
    batch = SimpleNamespace(
        tensor=None,
        latents=torch.randn(1, 16, 1, 2, 2),
        num_frames=1,
        dataset_config=SimpleNamespace(num_frames=1),
    )

    with pytest.raises(ValueError, match="image_i2v_clip_training cannot be used"):
        model.preprocess_training_batch(batch)


def test_image_clip_training_leaves_video_latents_alone():
    model = _make_i2v_model()
    model.image_i2v_clip_training = True
    model.image_i2v_clip_training_prob = 1.0
    batch = SimpleNamespace(
        tensor=None,
        latents=torch.randn(1, 16, 3, 2, 2),
        num_frames=9,
        dataset_config=SimpleNamespace(num_frames=9),
    )

    assert model.preprocess_training_batch(batch) is batch


def test_image_clip_condition_tensor_is_used_for_i2v_conditioning(monkeypatch):
    model = _make_i2v_model()
    latent_model_input = torch.randn(1, 16, 3, 2, 2)
    sentinel = torch.randn(1, 26, 3, 2, 2)
    condition = torch.randn(1, 3, 8, 8)
    calls = {}
    batch = SimpleNamespace(
        tensor=torch.randn(1, 9, 3, 8, 8),
        i2v_condition_tensor=condition,
        dataset_config=SimpleNamespace(num_frames=9),
    )

    def fake_add_first_frame_conditioning(**kwargs):
        calls["first_frame"] = kwargs["first_frame"]
        return sentinel

    monkeypatch.setattr(
        wan22_i2v_module,
        "add_first_frame_conditioning",
        fake_add_first_frame_conditioning,
    )

    model.get_noise_prediction(
        latent_model_input=latent_model_input,
        timestep=torch.tensor([1]),
        text_embeddings=_make_prompt_embeds(),
        batch=batch,
    )

    assert torch.equal(calls["first_frame"], condition)
    assert torch.equal(model.model.hidden_states[-1], sentinel)


def _make_concept_slider_trainer(arch):
    trainer = object.__new__(ConceptSliderTrainer)
    trainer.sd = SimpleNamespace(
        arch=arch,
        unet=_FakeUnet(),
    )
    trainer.network = _FakeNetwork()
    trainer.train_config = SimpleNamespace(dtype="fp32")
    trainer.device_torch = torch.device("cpu")
    trainer.slider = ConceptSliderTrainerConfig(guidance_strength=1.0, anchor_strength=1.0)
    trainer.anchor_class_embeds = None
    trainer.positive_prompt_embeds = _make_prompt_embeds()
    trainer.target_class_embeds = _make_prompt_embeds()
    trainer.negative_prompt_embeds = _make_prompt_embeds()
    return trainer


def test_sd_trainer_preprocess_batch_delegates_to_model_hook():
    trainer = object.__new__(ConceptSliderTrainer)
    batch = SimpleNamespace()
    expected = SimpleNamespace()

    def preprocess_training_batch(input_batch):
        assert input_batch is batch
        return expected

    trainer.sd = SimpleNamespace(preprocess_training_batch=preprocess_training_batch)

    assert trainer.preprocess_batch(batch) is expected


def test_concept_slider_sets_force_flag_for_wan22_single_frame():
    trainer = _make_concept_slider_trainer("wan22_14b_i2v")
    calls = []

    def fake_predict_noise(**kwargs):
        calls.append(kwargs.get("force_t2i_single_frame"))
        return kwargs["latents"].clone().requires_grad_(True)

    trainer.sd.predict_noise = fake_predict_noise

    batch = SimpleNamespace(
        tensor=torch.randn(1, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=1),
    )

    trainer.get_guided_loss(
        noisy_latents=torch.randn(1, 16, 1, 2, 2),
        conditional_embeds=_make_prompt_embeds(),
        match_adapter_assist=False,
        network_weight_list=[],
        timesteps=torch.tensor([1]),
        pred_kwargs={},
        batch=batch,
        noise=torch.randn(1, 16, 1, 2, 2),
    )

    assert calls == [True, True, True]


def test_concept_slider_does_not_force_flag_for_wan22_video_batches():
    trainer = _make_concept_slider_trainer("wan22_14b_i2v")
    calls = []

    def fake_predict_noise(**kwargs):
        calls.append(kwargs.get("force_t2i_single_frame"))
        return kwargs["latents"].clone().requires_grad_(True)

    trainer.sd.predict_noise = fake_predict_noise

    batch = SimpleNamespace(
        tensor=torch.randn(1, 9, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=9),
    )

    trainer.get_guided_loss(
        noisy_latents=torch.randn(1, 16, 3, 2, 2),
        conditional_embeds=_make_prompt_embeds(),
        match_adapter_assist=False,
        network_weight_list=[],
        timesteps=torch.tensor([1]),
        pred_kwargs={},
        batch=batch,
        noise=torch.randn(1, 16, 3, 2, 2),
    )

    assert calls == [False, False, False]


def test_concept_slider_does_not_force_flag_for_wan22_auto_frame_video_batches():
    trainer = _make_concept_slider_trainer("wan22_14b_i2v")
    calls = []

    def fake_predict_noise(**kwargs):
        calls.append(kwargs.get("force_t2i_single_frame"))
        return kwargs["latents"].clone().requires_grad_(True)

    trainer.sd.predict_noise = fake_predict_noise

    batch = SimpleNamespace(
        tensor=torch.randn(1, 9, 3, 8, 8),
        num_frames=9,
        dataset_config=SimpleNamespace(num_frames=1, auto_frame_count=True),
    )

    trainer.get_guided_loss(
        noisy_latents=torch.randn(1, 16, 3, 2, 2),
        conditional_embeds=_make_prompt_embeds(),
        match_adapter_assist=False,
        network_weight_list=[],
        timesteps=torch.tensor([1]),
        pred_kwargs={},
        batch=batch,
        noise=torch.randn(1, 16, 3, 2, 2),
    )

    assert calls == [False, False, False]


def test_concept_slider_does_not_force_flag_for_non_wan_arch():
    trainer = _make_concept_slider_trainer("wan22_14b")
    calls = []

    def fake_predict_noise(**kwargs):
        calls.append(kwargs.get("force_t2i_single_frame"))
        return kwargs["latents"].clone().requires_grad_(True)

    trainer.sd.predict_noise = fake_predict_noise

    batch = SimpleNamespace(
        tensor=torch.randn(1, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=1),
    )

    trainer.get_guided_loss(
        noisy_latents=torch.randn(1, 16, 1, 2, 2),
        conditional_embeds=_make_prompt_embeds(),
        match_adapter_assist=False,
        network_weight_list=[],
        timesteps=torch.tensor([1]),
        pred_kwargs={},
        batch=batch,
        noise=torch.randn(1, 16, 1, 2, 2),
    )

    assert calls == [False, False, False]
