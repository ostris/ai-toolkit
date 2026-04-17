from __future__ import annotations

import contextlib
import importlib
import importlib.machinery
import inspect
import json
import os
import random
import re
import sys
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
import torch


class SessionError(RuntimeError):
    pass


@dataclass
class OptimizerConfig:
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0


@dataclass
class GRPOConfig:
    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    beta: float = 0.0
    noise_level: float = 0.7
    sde_type: Literal["sde", "cps"] = "sde"
    timestep_fraction: float = 1.0


@dataclass
class LoRAConfigSpec:
    enabled: bool = True
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    lora_path: str | None = None


@dataclass
class SessionConfig:
    session_id: str
    model_name: str
    model_arch: str = ""
    # legacy input kept for node compatibility
    model_family: str | None = None
    model_extras_name_or_path: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    model_paths: dict[str, str] = field(default_factory=dict)
    model_config_overrides: dict[str, Any] = field(default_factory=dict)
    device: str = "cuda"
    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    seed: int = 0
    checkpoint_root: str = "./output/aitk_flow_grpo"
    checkpoint_interval_steps: int = 25
    resume: bool = True
    lora: LoRAConfigSpec = field(default_factory=LoRAConfigSpec)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    def normalized_arch(self) -> str:
        overrides = self.model_config_overrides if isinstance(self.model_config_overrides, dict) else {}
        override_arch = str(overrides.get("arch", "")).strip().lower()
        if override_arch:
            if ":" in override_arch:
                override_arch = override_arch.split(":", 1)[0]
            return override_arch
        arch = (self.model_arch or "").strip().lower()
        if arch:
            if ":" in arch:
                arch = arch.split(":", 1)[0]
            return arch
        family = (self.model_family or "").strip().lower()
        if family in {"sd3", "flux"}:
            return family
        return "sd1"


@dataclass
class CandidateRecord:
    candidate_id: str
    session_id: str
    model_arch: str
    prompt: str
    negative_prompt: str
    seed: int
    guidance_scale: float
    num_inference_steps: int
    sampler: str
    scheduler: str
    conditional_embeds: Any
    unconditional_embeds: Any | None
    timesteps: torch.Tensor
    sigma_current: torch.Tensor
    sigma_next: torch.Tensor
    latents: torch.Tensor
    next_latents: torch.Tensor
    log_probs: torch.Tensor
    reference_images: torch.Tensor | None


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = (dtype_name or "").strip().lower()
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp32":
        return torch.float32
    raise SessionError(f"Unsupported dtype '{dtype_name}'. Use fp16, bf16, or fp32.")


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().clone()


def _clone_prompt_embeds(prompt_embeds: Any | None) -> Any | None:
    if prompt_embeds is None:
        return None
    if hasattr(prompt_embeds, "detach") and hasattr(prompt_embeds, "clone"):
        return prompt_embeds.detach().clone()
    return prompt_embeds


def _expand_like(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    out = value
    while out.ndim < ref.ndim:
        out = out.unsqueeze(-1)
    return out


def _flowmatch_step_with_logprob(
    *,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    noise_level: float,
    sde_type: str,
    prev_sample: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # everything in float32 for numerical stability
    model_output = model_output.float()
    sample = sample.float()
    sigma = sigma.float().flatten()
    sigma_next = sigma_next.float().flatten()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    sigma_e = _expand_like(sigma, sample)
    sigma_next_e = _expand_like(sigma_next, sample)
    dt = sigma_next_e - sigma_e

    eps = 1e-8
    sqrt_2pi = torch.sqrt(torch.as_tensor(2.0 * np.pi, device=sample.device, dtype=torch.float32))

    if sde_type == "sde":
        sigma_safe = torch.clamp(sigma_e, min=eps)
        one_minus_sigma = torch.clamp(1.0 - sigma_e, min=eps)
        std_dev_t = torch.sqrt(sigma_safe / one_minus_sigma) * float(noise_level)
        dt_safe = torch.clamp(-dt, min=eps)
        step_scale = torch.sqrt(dt_safe)

        prev_sample_mean = (
            sample * (1.0 + ((std_dev_t**2) / (2.0 * sigma_safe)) * dt)
            + model_output * (1.0 + ((std_dev_t**2) * (1.0 - sigma_e) / (2.0 * sigma_safe))) * dt
        )
        if prev_sample is None:
            prev_sample = prev_sample_mean + std_dev_t * step_scale * torch.randn_like(model_output)

        transition_std = torch.clamp(std_dev_t * step_scale, min=eps)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * (transition_std**2))
            - torch.log(transition_std)
            - torch.log(sqrt_2pi)
        )
    elif sde_type == "cps":
        std_dev_t = torch.clamp(
            sigma_next_e * np.sin(float(noise_level) * np.pi / 2.0),
            min=eps,
        )
        pred_original_sample = sample - sigma_e * model_output
        noise_estimate = sample + model_output * (1.0 - sigma_e)
        sqrt_term = torch.sqrt(torch.clamp((sigma_next_e**2) - (std_dev_t**2), min=eps))
        prev_sample_mean = pred_original_sample * (1.0 - sigma_next_e) + noise_estimate * sqrt_term
        if prev_sample is None:
            prev_sample = prev_sample_mean + std_dev_t * torch.randn_like(model_output)
        transition_std = std_dev_t
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * (transition_std**2))
            - torch.log(transition_std)
            - torch.log(sqrt_2pi)
        )
    else:
        raise SessionError(f"Unsupported sde_type '{sde_type}'. Use 'sde' or 'cps'.")

    reduce_dims = tuple(range(1, log_prob.ndim))
    log_prob = log_prob.mean(dim=reduce_dims)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def _ensure_module(name: str, *, package: bool) -> ModuleType:
    module = ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_optional_dependency_stubs() -> None:
    # AITK imports these unconditionally in multiple modules.
    # In constrained environments they may be absent, so we install minimal runtime stubs
    # to allow model loading for non-quantized/non-k-diffusion online RLHF paths.
    need_optimum_stub = False
    try:
        importlib.import_module("optimum.quanto")
    except Exception:
        need_optimum_stub = True

    if need_optimum_stub:
        optimum = _ensure_module("optimum", package=True)
        quanto = _ensure_module("optimum.quanto", package=True)
        quanto_quantize = _ensure_module("optimum.quanto.quantize", package=False)
        quanto_tensor = _ensure_module("optimum.quanto.tensor", package=False)

        class _QTensor(torch.Tensor):
            pass

        class _QBytesTensor(torch.Tensor):
            pass

        class _Optimizer:
            pass

        class _QType(str):
            pass

        def _freeze(*args: Any, **kwargs: Any) -> None:
            return None

        def _quantize_submodule(*args: Any, **kwargs: Any) -> None:
            return None

        quanto.QTensor = _QTensor  # type: ignore[attr-defined]
        quanto.QBytesTensor = _QBytesTensor  # type: ignore[attr-defined]
        quanto.freeze = _freeze  # type: ignore[attr-defined]
        quanto.qfloat8 = "qfloat8"  # type: ignore[attr-defined]
        quanto.qint4 = "qint4"  # type: ignore[attr-defined]

        quanto_quantize._quantize_submodule = _quantize_submodule  # type: ignore[attr-defined]
        quanto_tensor.Optimizer = _Optimizer  # type: ignore[attr-defined]
        quanto_tensor.qtype = _QType  # type: ignore[attr-defined]
        quanto_tensor.qtypes = {  # type: ignore[attr-defined]
            "qfloat8": "qfloat8",
            "qint4": "qint4",
            "qint8": "qint8",
        }
        quanto_tensor.QTensor = _QTensor  # type: ignore[attr-defined]
        quanto_tensor.QBytesTensor = _QBytesTensor  # type: ignore[attr-defined]

        sys.modules["optimum"] = optimum
        sys.modules["optimum.quanto"] = quanto
        sys.modules["optimum.quanto.quantize"] = quanto_quantize
        sys.modules["optimum.quanto.tensor"] = quanto_tensor

    need_kdiff_stub = False
    try:
        importlib.import_module("k_diffusion.external")
        importlib.import_module("k_diffusion.sampling")
    except Exception:
        need_kdiff_stub = True

    if need_kdiff_stub:
        k_diffusion = _ensure_module("k_diffusion", package=True)
        k_external = _ensure_module("k_diffusion.external", package=False)
        k_sampling = _ensure_module("k_diffusion.sampling", package=False)

        class _Dummy:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return None

        def _get_sigmas_karras(*args: Any, **kwargs: Any) -> torch.Tensor:
            n = int(kwargs.get("n", 20))
            return torch.linspace(1.0, 0.0, n)

        k_external.CompVisVDenoiser = _Dummy  # type: ignore[attr-defined]
        k_external.CompVisDenoiser = _Dummy  # type: ignore[attr-defined]
        k_sampling.get_sigmas_karras = _get_sigmas_karras  # type: ignore[attr-defined]
        k_sampling.BrownianTreeNoiseSampler = _Dummy  # type: ignore[attr-defined]

        sys.modules["k_diffusion"] = k_diffusion
        sys.modules["k_diffusion.external"] = k_external
        sys.modules["k_diffusion.sampling"] = k_sampling


_EXT_MODEL_MAP: dict[str, tuple[str, str]] | None = None
_EXT_MODEL_MAP_LOCK = threading.Lock()


def _discover_extension_model_map(repo_root: Path) -> dict[str, tuple[str, str]]:
    global _EXT_MODEL_MAP
    with _EXT_MODEL_MAP_LOCK:
        if _EXT_MODEL_MAP is not None:
            return _EXT_MODEL_MAP

        model_map: dict[str, tuple[str, str]] = {}
        search_roots = [repo_root / "extensions", repo_root / "extensions_built_in"]
        class_pattern = re.compile(
            r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*:[\s\S]{0,800}?arch\s*=\s*['\"]([^'\"]+)['\"]",
            re.MULTILINE,
        )

        for base in search_roots:
            if not base.exists():
                continue
            for py_file in base.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding="utf-8")
                except Exception:
                    continue
                for class_name, arch in class_pattern.findall(content):
                    relative = py_file.relative_to(repo_root).with_suffix("")
                    module_path = ".".join(relative.parts)
                    model_map[arch.strip().lower()] = (module_path, class_name)

        _EXT_MODEL_MAP = model_map
        return model_map


class OnlineFlowGRPOSession:
    def __init__(self, config: SessionConfig):
        self.config = config
        self._lock = threading.RLock()
        self._repo_root = Path(__file__).resolve().parents[1]
        self.model_arch = self.config.normalized_arch()

        _install_optional_dependency_stubs()

        from toolkit.config_modules import ModelConfig, NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork
        from toolkit.sampler import get_sampler
        from toolkit.stable_diffusion_model import StableDiffusion
        from toolkit.util.get_model import get_model_class

        self._ModelConfig = ModelConfig
        self._NetworkConfig = NetworkConfig
        self._LoRASpecialNetwork = LoRASpecialNetwork
        self._get_sampler = get_sampler
        self._StableDiffusion = StableDiffusion
        self._get_model_class = get_model_class

        self.device_torch = torch.device(config.device)
        self.weight_dtype = _dtype_from_name(config.dtype)
        self.model = None
        self.network = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.step_count: int = 0

        self._candidate_cache: OrderedDict[str, CandidateRecord] = OrderedDict()
        self._max_candidates = 32

        self._initialize_model_lora_optimizer()

    def _autocast_context(self):
        if self.device_torch.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.weight_dtype)
        return contextlib.nullcontext()

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.config.checkpoint_root) / self.config.session_id

    @property
    def checkpoint_state_path(self) -> Path:
        return self.checkpoint_dir / "state.json"

    @property
    def checkpoint_lora_path(self) -> Path:
        return self.checkpoint_dir / "lora.safetensors"

    @property
    def checkpoint_optimizer_path(self) -> Path:
        return self.checkpoint_dir / "optimizer.pt"

    @staticmethod
    def _sampler_arch_for_model(arch: str) -> str:
        lowered = arch.strip().lower()
        if lowered.startswith("flux"):
            return "flux"
        if lowered == "lumina2":
            return "lumina2"
        return "sd"

    def _resolve_model_class(self, model_config):
        model_class = self._get_model_class(model_config)
        if model_class is None:
            raise SessionError(
                f"Unable to resolve model class for arch '{getattr(model_config, 'arch', self.model_arch)}'."
            )
        return model_class

    def _build_model_config(self):
        kwargs: dict[str, Any] = {
            "name_or_path": self.config.model_name,
            "arch": self.model_arch,
            "dtype": self.config.dtype,
            "extras_name_or_path": (
                self.config.model_extras_name_or_path or self.config.model_name
            ),
            "model_kwargs": dict(self.config.model_kwargs),
            "model_paths": dict(self.config.model_paths),
        }

        overrides = {}
        if isinstance(self.config.model_config_overrides, dict):
            overrides = dict(self.config.model_config_overrides)
        override_model_kwargs = overrides.pop("model_kwargs", None)
        override_model_paths = overrides.pop("model_paths", None)
        kwargs.update(overrides)

        if isinstance(override_model_kwargs, dict):
            merged_kwargs = dict(kwargs.get("model_kwargs", {}))
            merged_kwargs.update(override_model_kwargs)
            kwargs["model_kwargs"] = merged_kwargs

        if isinstance(override_model_paths, dict):
            merged_paths = dict(kwargs.get("model_paths", {}))
            merged_paths.update(override_model_paths)
            kwargs["model_paths"] = merged_paths

        if not kwargs.get("name_or_path"):
            raise SessionError("Model config is missing 'name_or_path'.")

        return self._ModelConfig(**kwargs)

    def _build_flowmatch_scheduler(self, model_class, model_config) -> Any:
        if hasattr(model_class, "get_train_scheduler"):
            scheduler = model_class.get_train_scheduler()
        else:
            sampler_arch = self._sampler_arch_for_model(
                str(getattr(model_config, "arch", self.model_arch) or self.model_arch)
            )
            scheduler = self._get_sampler("flowmatch", arch=sampler_arch)
        return scheduler

    def _make_lora_network(self):
        if not self.config.lora.enabled:
            raise SessionError("Flow-GRPO online mode requires LoRA enabled.")

        model = self.model
        model_to_train = model.get_model_to_train()
        network_config = self._NetworkConfig(
            type="lora",
            linear=int(self.config.lora.rank),
            linear_alpha=float(self.config.lora.alpha),
            dropout=float(self.config.lora.dropout),
            network_kwargs={},
            transformer_only=False,
        )
        network_kwargs: dict[str, Any] = {}
        if hasattr(model, "target_lora_modules"):
            network_kwargs["target_lin_modules"] = model.target_lora_modules

        network = self._LoRASpecialNetwork(
            text_encoder=model.text_encoder,
            unet=model_to_train,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            conv_lora_dim=network_config.conv,
            conv_alpha=network_config.conv_alpha,
            is_sdxl=model.model_config.is_xl or model.model_config.is_ssd,
            is_v2=model.model_config.is_v2,
            is_v3=model.model_config.is_v3,
            is_pixart=model.model_config.is_pixart,
            is_auraflow=model.model_config.is_auraflow,
            is_flux=model.model_config.is_flux,
            is_lumina2=model.model_config.is_lumina2,
            is_ssd=model.model_config.is_ssd,
            is_vega=model.model_config.is_vega,
            dropout=network_config.dropout,
            use_text_encoder_1=model.model_config.use_text_encoder_1,
            use_text_encoder_2=model.model_config.use_text_encoder_2,
            use_bias=False,
            is_lorm=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=model.is_transformer,
            base_model=model,
            **network_kwargs,
        )
        network.force_to(self.device_torch, dtype=torch.float32)
        network.apply_to(model.text_encoder, model_to_train, False, True)
        network.prepare_grad_etc(model.text_encoder, model_to_train)
        network._update_torch_multiplier()
        return network

    def _initialize_model_lora_optimizer(self) -> None:
        with self._lock:
            _set_global_seed(self.config.seed)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            model_config = self._build_model_config()
            model_class = self._resolve_model_class(model_config)
            scheduler = self._build_flowmatch_scheduler(model_class, model_config)

            model = model_class(
                device=self.device_torch,
                model_config=model_config,
                dtype=self.config.dtype,
                noise_scheduler=scheduler,
            )
            model.load_model()
            if hasattr(model, "set_device_state_preset"):
                try:
                    model.set_device_state_preset("generate")
                except Exception:
                    pass

            # safety: online image RLHF flow currently targets image models
            if getattr(model, "is_audio_model", False):
                raise SessionError(
                    f"model_arch '{self.model_arch}' is audio-only and not supported by image vote RLHF."
                )

            self.model = model
            self.network = self._make_lora_network()
            self.model.network = self.network

            if self.config.lora.lora_path:
                load_path = Path(self.config.lora.lora_path)
                if not load_path.exists():
                    raise SessionError(f"LoRA path does not exist: {load_path}")
                self.network.load_weights(str(load_path))

            if self.config.resume and self.checkpoint_lora_path.exists():
                self.network.load_weights(str(self.checkpoint_lora_path))

            if self.config.optimizer.optimizer != "adamw":
                raise SessionError(
                    f"Unsupported optimizer '{self.config.optimizer.optimizer}'. Only adamw is supported."
                )

            param_groups = self.network.prepare_optimizer_params(
                text_encoder_lr=self.config.optimizer.learning_rate,
                unet_lr=self.config.optimizer.learning_rate,
                default_lr=self.config.optimizer.learning_rate,
            )
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.optimizer.learning_rate,
                betas=(
                    self.config.optimizer.adam_beta1,
                    self.config.optimizer.adam_beta2,
                ),
                weight_decay=self.config.optimizer.adam_weight_decay,
                eps=self.config.optimizer.adam_epsilon,
            )

            if self.config.resume and self.checkpoint_optimizer_path.exists():
                state = torch.load(
                    self.checkpoint_optimizer_path,
                    map_location=self.device_torch,
                    weights_only=True,
                )
                self.optimizer.load_state_dict(state)

            if self.config.resume and self.checkpoint_state_path.exists():
                with self.checkpoint_state_path.open("r", encoding="utf-8") as handle:
                    state = json.load(handle)
                self.step_count = int(state.get("step_count", 0))

    def _default_resolution(self) -> tuple[int, int]:
        model = self.model
        h = 512
        w = 512
        try:
            sample_size = int(model.unet_unwrapped.config.sample_size)
            vae_scale = 2 ** (len(model.vae.config["block_out_channels"]) - 1)
            h = sample_size * vae_scale
            w = sample_size * vae_scale
            if getattr(model, "is_flux", False):
                h *= 2
                w *= 2
        except Exception:
            pass
        h = max(64, int(h // 16) * 16)
        w = max(64, int(w // 16) * 16)
        return h, w

    def _prepare_scheduler_run(
        self,
        *,
        steps: int,
        latents: torch.Tensor,
        sampler: str,
        scheduler_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        allowed_samplers = {"flow_grpo_sde", "flow_sde", "flow_grpo", "flowmatch"}
        allowed_schedulers = {
            "flow_match_euler_discrete",
            "flowmatch",
            "flow_match",
        }
        if sampler.lower() not in allowed_samplers:
            raise SessionError(
                f"Unsupported sampler '{sampler}'. Use one of: {sorted(allowed_samplers)}"
            )
        if scheduler_name.lower() not in allowed_schedulers:
            raise SessionError(
                f"Unsupported scheduler '{scheduler_name}'. Use one of: {sorted(allowed_schedulers)}"
            )

        scheduler = self.model.noise_scheduler

        if hasattr(scheduler, "set_train_timesteps"):
            sig = inspect.signature(scheduler.set_train_timesteps)
            kwargs: dict[str, Any] = {
                "num_timesteps": int(steps),
                "device": self.device_torch,
            }
            if "timestep_type" in sig.parameters:
                kwargs["timestep_type"] = "shift"
            if "latents" in sig.parameters:
                kwargs["latents"] = latents
            if "patch_size" in sig.parameters:
                kwargs["patch_size"] = 2 if getattr(self.model, "is_flux", False) else 1
            timesteps = scheduler.set_train_timesteps(**kwargs)
        else:
            scheduler.set_timesteps(int(steps), device=self.device_torch)
            timesteps = scheduler.timesteps

        timesteps = timesteps.to(self.device_torch, dtype=torch.float32)

        if not hasattr(scheduler, "sigmas"):
            raise SessionError("Flow-GRPO requires a flow scheduler exposing `sigmas`.")
        sigmas = scheduler.sigmas.to(self.device_torch, dtype=torch.float32)
        if sigmas.numel() < timesteps.numel() + 1:
            raise SessionError(
                "Scheduler sigmas length is insufficient for Flow-GRPO transition probabilities."
            )
        sigma_current = sigmas[: timesteps.numel()]
        sigma_next = sigmas[1 : timesteps.numel() + 1]
        return timesteps, sigma_current, sigma_next

    def _prepare_initial_latents(
        self,
        *,
        height: int,
        width: int,
        reference_images: torch.Tensor | None,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        latents = self.model.get_latent_noise(
            pixel_height=height,
            pixel_width=width,
            batch_size=1,
        ).to(self.device_torch, dtype=self.weight_dtype)

        if reference_images is None:
            return latents

        if reference_images.ndim != 4:
            raise SessionError("reference_images must be a 4D tensor [B, C, H, W].")

        ref = reference_images[:1].to(self.device_torch, dtype=self.weight_dtype)
        if ref.shape[1] == 1:
            ref = ref.repeat(1, 3, 1, 1)
        elif ref.shape[1] > 3:
            ref = ref[:, :3, :, :]
        ref = torch.nn.functional.interpolate(
            ref,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        ref = ref * 2.0 - 1.0
        encoded = self.model.encode_images([ref[0]], device=self.device_torch, dtype=self.weight_dtype)
        noise = self.model.get_latent_noise_from_latents(encoded)
        first_timestep = timesteps[0].reshape(1).to(self.device_torch)
        return self.model.add_noise(encoded, noise, first_timestep).to(self.device_torch, dtype=self.weight_dtype)

    def _decode_image_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.model.decode_latents(latents, device=self.device_torch, dtype=self.weight_dtype)
        if decoded.ndim == 5:
            # video-like output -> preview first frame
            decoded = decoded[:, :, 0, :, :]
        if decoded.shape[1] == 1:
            decoded = decoded.repeat(1, 3, 1, 1)
        if decoded.shape[1] > 3:
            decoded = decoded[:, :3, :, :]
        return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)

    def _store_candidate(self, candidate: CandidateRecord) -> None:
        self._candidate_cache[candidate.candidate_id] = candidate
        while len(self._candidate_cache) > self._max_candidates:
            self._candidate_cache.popitem(last=False)

    def get_candidate(self, candidate_id: str) -> CandidateRecord:
        if candidate_id not in self._candidate_cache:
            raise SessionError(f"Candidate '{candidate_id}' is not available in session cache.")
        return self._candidate_cache[candidate_id]

    def pop_candidate(self, candidate_id: str) -> CandidateRecord:
        if candidate_id not in self._candidate_cache:
            raise SessionError(f"Candidate '{candidate_id}' is not available in session cache.")
        return self._candidate_cache.pop(candidate_id)

    def generate_candidate(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
        cfg: float = 4.5,
        steps: int = 10,
        sampler: str = "flow_grpo_sde",
        scheduler: str = "flow_match_euler_discrete",
        reference_images: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        with self._lock:
            if self.model is None:
                raise SessionError("Session is not initialized.")

            _set_global_seed(seed)

            conditional_embeds = self.model.encode_prompt(prompt).to(
                self.device_torch, dtype=self.weight_dtype
            )
            unconditional_embeds = None
            if cfg > 1.0:
                unconditional_embeds = self.model.encode_prompt(negative_prompt or "").to(
                    self.device_torch, dtype=self.weight_dtype
                )

            height, width = self._default_resolution()
            # temporary latents for scheduler setup (dynamic shift schedulers may use shape)
            tmp_latents = self.model.get_latent_noise(
                pixel_height=height, pixel_width=width, batch_size=1
            ).to(self.device_torch, dtype=self.weight_dtype)
            timesteps, sigma_current, sigma_next = self._prepare_scheduler_run(
                steps=int(steps),
                latents=tmp_latents,
                sampler=sampler,
                scheduler_name=scheduler,
            )
            latents = self._prepare_initial_latents(
                height=height,
                width=width,
                reference_images=reference_images,
                timesteps=timesteps,
            )

            latents_before: list[torch.Tensor] = []
            latents_after: list[torch.Tensor] = []
            log_probs: list[torch.Tensor] = []

            for idx in range(timesteps.numel()):
                t = timesteps[idx]
                t_batch = t.reshape(1).repeat(latents.shape[0]).to(
                    self.device_torch, dtype=torch.float32
                )

                with torch.no_grad():
                    with self._autocast_context():
                        noise_pred = self.model.predict_noise(
                            latents=latents,
                            conditional_embeddings=conditional_embeds,
                            unconditional_embeddings=unconditional_embeds,
                            timestep=t_batch,
                            guidance_scale=float(cfg),
                        )

                prev_latents, log_prob, _, _ = _flowmatch_step_with_logprob(
                    model_output=noise_pred,
                    sample=latents,
                    sigma=sigma_current[idx].reshape(1).repeat(latents.shape[0]),
                    sigma_next=sigma_next[idx].reshape(1).repeat(latents.shape[0]),
                    noise_level=self.config.grpo.noise_level,
                    sde_type=self.config.grpo.sde_type,
                    prev_sample=None,
                )

                latents_before.append(_clone_tensor(latents))
                latents_after.append(_clone_tensor(prev_latents))
                log_probs.append(_clone_tensor(log_prob))
                latents = prev_latents

            image_tensor = self._decode_image_tensor(latents)
            candidate_id = str(uuid.uuid4())

            record = CandidateRecord(
                candidate_id=candidate_id,
                session_id=self.config.session_id,
                model_arch=self.model_arch,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                guidance_scale=float(cfg),
                num_inference_steps=int(steps),
                sampler=sampler,
                scheduler=scheduler,
                conditional_embeds=_clone_prompt_embeds(conditional_embeds),
                unconditional_embeds=_clone_prompt_embeds(unconditional_embeds),
                timesteps=_clone_tensor(timesteps),
                sigma_current=_clone_tensor(sigma_current),
                sigma_next=_clone_tensor(sigma_next),
                latents=torch.stack(latents_before, dim=1),
                next_latents=torch.stack(latents_after, dim=1),
                log_probs=torch.stack(log_probs, dim=1),
                reference_images=_clone_tensor(reference_images),
            )
            self._store_candidate(record)

            metadata = {
                "session_id": self.config.session_id,
                "candidate_id": candidate_id,
                "model_arch": self.model_arch,
                "model_name": self.config.model_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": int(seed),
                "cfg": float(cfg),
                "steps": int(steps),
                "sampler": sampler,
                "scheduler": scheduler,
                "session_step_before_vote": self.step_count,
            }
            return image_tensor.detach().cpu(), metadata

    @contextlib.contextmanager
    def _lora_disabled(self):
        if self.network is None:
            yield
            return
        was_active = bool(getattr(self.network, "is_active", True))
        self.network.is_active = False
        try:
            yield
        finally:
            self.network.is_active = was_active

    def _compute_step_logprob(
        self,
        sample: CandidateRecord,
        step_index: int,
        *,
        disable_lora: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = sample.latents[:, step_index]
        t = sample.timesteps[step_index].reshape(1).repeat(latents.shape[0]).to(
            self.device_torch, dtype=torch.float32
        )

        context = self._lora_disabled() if disable_lora else contextlib.nullcontext()
        with context:
            with self._autocast_context():
                noise_pred = self.model.predict_noise(
                    latents=latents.to(self.device_torch, dtype=self.weight_dtype),
                    conditional_embeddings=sample.conditional_embeds,
                    unconditional_embeddings=sample.unconditional_embeds,
                    timestep=t,
                    guidance_scale=float(sample.guidance_scale),
                )

            _, log_prob, prev_mean, std_dev_t = _flowmatch_step_with_logprob(
                model_output=noise_pred,
                sample=latents.to(self.device_torch, dtype=self.weight_dtype),
                sigma=sample.sigma_current[step_index].reshape(1).repeat(latents.shape[0]),
                sigma_next=sample.sigma_next[step_index].reshape(1).repeat(latents.shape[0]),
                noise_level=self.config.grpo.noise_level,
                sde_type=self.config.grpo.sde_type,
                prev_sample=sample.next_latents[:, step_index].to(
                    self.device_torch, dtype=self.weight_dtype
                ),
            )
        return log_prob, prev_mean, std_dev_t

    def vote_step(
        self,
        *,
        candidate_id: str,
        vote: Literal["upvote", "downvote", "skip"],
        consume_candidate: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            reward_map = {"upvote": 1.0, "downvote": -1.0, "skip": 0.0}
            if vote not in reward_map:
                raise SessionError(f"Unsupported vote '{vote}'.")

            if vote == "skip":
                if consume_candidate:
                    self.pop_candidate(candidate_id)
                return {
                    "session_id": self.config.session_id,
                    "candidate_id": candidate_id,
                    "vote": vote,
                    "reward": 0.0,
                    "step_count": self.step_count,
                    "trained": False,
                }

            if self.optimizer is None:
                raise SessionError("Session optimizer is not initialized.")

            sample = self.get_candidate(candidate_id)
            reward = reward_map[vote]

            self.network.train()
            self.optimizer.zero_grad(set_to_none=True)

            total_steps = sample.log_probs.shape[1]
            train_steps = max(
                1,
                int(total_steps * max(0.0, min(self.config.grpo.timestep_fraction, 1.0))),
            )
            train_indices = list(range(train_steps))

            total_loss: torch.Tensor | None = None
            policy_losses: list[float] = []
            kl_losses: list[float] = []
            approx_kls: list[float] = []
            clipfracs: list[float] = []

            for step_idx in train_indices:
                log_prob, prev_mean, std_dev_t = self._compute_step_logprob(sample, step_idx)
                old_log_prob = sample.log_probs[:, step_idx].to(
                    self.device_torch, dtype=log_prob.dtype
                )

                advantages = torch.full_like(log_prob, float(reward))
                advantages = torch.clamp(
                    advantages,
                    -self.config.grpo.adv_clip_max,
                    self.config.grpo.adv_clip_max,
                )

                ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0))
                unclipped = -advantages * ratio
                clipped = -advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.grpo.clip_range,
                    1.0 + self.config.grpo.clip_range,
                )
                policy_loss = torch.mean(torch.maximum(unclipped, clipped))
                loss = policy_loss

                if self.config.grpo.beta > 0.0:
                    with torch.no_grad():
                        _, ref_prev_mean, _ = self._compute_step_logprob(
                            sample, step_idx, disable_lora=True
                        )
                    std_safe = torch.clamp(std_dev_t, min=1e-6)
                    kl_loss = (
                        ((prev_mean - ref_prev_mean) ** 2) / (2.0 * (std_safe**2))
                    ).mean()
                    loss = loss + self.config.grpo.beta * kl_loss
                    kl_losses.append(float(kl_loss.detach().cpu().item()))

                scaled_loss = loss / float(len(train_indices))
                total_loss = scaled_loss if total_loss is None else total_loss + scaled_loss

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                approx_kls.append(
                    float((0.5 * torch.mean((log_prob - old_log_prob) ** 2)).detach().cpu().item())
                )
                clipfracs.append(
                    float(
                        torch.mean((torch.abs(ratio - 1.0) > self.config.grpo.clip_range).float())
                        .detach()
                        .cpu()
                        .item()
                    )
                )

            if total_loss is None:
                raise SessionError("No training timesteps selected for vote_step.")

            total_loss.backward()
            clip_params = []
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if isinstance(param, torch.Tensor) and param.requires_grad:
                        clip_params.append(param)
            if clip_params:
                torch.nn.utils.clip_grad_norm_(clip_params, self.config.optimizer.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.step_count += 1
            if consume_candidate:
                self.pop_candidate(candidate_id)

            auto_checkpointed = False
            if (
                self.config.checkpoint_interval_steps > 0
                and self.step_count % self.config.checkpoint_interval_steps == 0
            ):
                self.save_checkpoint()
                auto_checkpointed = True

            return {
                "session_id": self.config.session_id,
                "candidate_id": candidate_id,
                "vote": vote,
                "reward": reward,
                "trained": True,
                "step_count": self.step_count,
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
                "kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
                "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
                "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "auto_checkpointed": auto_checkpointed,
            }

    def save_checkpoint(self) -> dict[str, Any]:
        with self._lock:
            if self.network is None or self.optimizer is None:
                raise SessionError("Session is not initialized.")

            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.network.save_weights(str(self.checkpoint_lora_path), dtype=torch.float16)
            torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer_path)

            state = {
                "session_id": self.config.session_id,
                "model_arch": self.model_arch,
                "model_name": self.config.model_name,
                "step_count": self.step_count,
                "config": {
                    **asdict(self.config),
                    "lora": asdict(self.config.lora),
                    "optimizer": asdict(self.config.optimizer),
                    "grpo": asdict(self.config.grpo),
                },
            }
            with self.checkpoint_state_path.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
            return state

    def close(self) -> None:
        with self._lock:
            self._candidate_cache.clear()
            self.model = None
            self.network = None
            self.optimizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.config.session_id,
            "model_arch": self.model_arch,
            "model_name": self.config.model_name,
            "step_count": self.step_count,
            "cached_candidates": len(self._candidate_cache),
            "checkpoint_dir": str(self.checkpoint_dir),
        }


class OnlineFlowGRPOManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: dict[str, OnlineFlowGRPOSession] = {}

    def create_or_get_session(
        self, config: SessionConfig, *, force_reset: bool = False
    ) -> OnlineFlowGRPOSession:
        with self._lock:
            existing = self._sessions.get(config.session_id)
            if existing is not None and not force_reset:
                return existing

            if existing is not None and force_reset:
                existing.close()
                del self._sessions[config.session_id]

            session = OnlineFlowGRPOSession(config)
            self._sessions[config.session_id] = session
            return session

    def get_session(self, session_id: str) -> OnlineFlowGRPOSession:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionError(f"Session '{session_id}' is not active.")
            return self._sessions[session_id]

    def close_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].close()
                del self._sessions[session_id]

    def list_sessions(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {sid: session.summary() for sid, session in self._sessions.items()}


_MANAGER: OnlineFlowGRPOManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_online_flow_grpo_manager() -> OnlineFlowGRPOManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = OnlineFlowGRPOManager()
        return _MANAGER
