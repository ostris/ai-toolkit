from functools import partial
from collections import OrderedDict
import math
import os
from typing import Any, Dict, Optional, Union, List
from typing_extensions import Self
import torch
import yaml
from huggingface_hub import hf_hub_download
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.wan21.wan_utils import add_first_frame_conditioning
from toolkit.prompt_utils import PromptEmbeds
from PIL import Image
from diffusers import UniPCMultistepScheduler
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import quantize_model
from .wan22_pipeline import Wan22Pipeline
from diffusers import WanTransformer3DModel

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from torchvision.transforms import functional as TF

from toolkit.models.wan21.wan21 import Wan21
from .wan22_5b_model import (
    scheduler_config,
    time_text_monkeypatch,
)
from toolkit.memory_management import MemoryManager
from safetensors.torch import load_file, save_file


boundary_ratio_t2v = 0.875
boundary_ratio_i2v = 0.9

scheduler_configUniPC = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.35.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}


class DualWanTransformer3DModel(torch.nn.Module):
    def __init__(
        self,
        transformer_1: WanTransformer3DModel,
        transformer_2: WanTransformer3DModel,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        boundary_ratio: float = boundary_ratio_t2v,
        low_vram: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_1: WanTransformer3DModel = transformer_1
        self.transformer_2: WanTransformer3DModel = transformer_2
        self.torch_dtype: torch.dtype = torch_dtype
        self.device_torch: torch.device = device
        self.boundary_ratio: float = boundary_ratio
        self.boundary: float = self.boundary_ratio * 1000
        self.low_vram: bool = low_vram
        self._active_transformer_name = "transformer_1"  # default to transformer_1

    @property
    def device(self) -> torch.device:
        return self.device_torch

    @property
    def dtype(self) -> torch.dtype:
        return self.torch_dtype

    @property
    def config(self):
        return self.transformer_1.config

    @property
    def transformer(self) -> WanTransformer3DModel:
        return getattr(self, self._active_transformer_name)

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for both transformers.
        """
        self.transformer_1.enable_gradient_checkpointing()
        self.transformer_2.enable_gradient_checkpointing()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # determine if doing high noise or low noise by meaning the timestep.
        # timesteps are in the range of 0 to 1000, so we can use a threshold
        with torch.no_grad():
            if timestep.float().mean().item() > self.boundary:
                t_name = "transformer_1"
            else:
                t_name = "transformer_2"

            # check if we are changing the active transformer, if so, we need to swap the one in
            # vram if low_vram is enabled
            # todo swap the loras as well
            if t_name != self._active_transformer_name:
                if self.low_vram:
                    getattr(self, self._active_transformer_name).to("cpu")
                    getattr(self, t_name).to(self.device_torch)
                    torch.cuda.empty_cache()
                self._active_transformer_name = t_name

        if self.transformer.device != hidden_states.device:
            if self.low_vram:
                # move other transformer to cpu
                other_tname = (
                    "transformer_1" if t_name == "transformer_2" else "transformer_2"
                )
                getattr(self, other_tname).to("cpu")

            self.transformer.to(hidden_states.device)

        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
        )

    def to(self, *args, **kwargs) -> Self:
        # do not do to, this will be handled separately
        return self


class Wan2214bModel(Wan21):
    arch = "wan22_14b"
    _wan_generation_scheduler_config = scheduler_configUniPC
    _wan_expand_timesteps = False
    _wan_boundary_ratio = boundary_ratio_t2v
    _wan_vae_path = "ai-toolkit/wan2.1-vae"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            model_config=model_config,
            dtype=dtype,
            custom_pipeline=custom_pipeline,
            noise_scheduler=noise_scheduler,
            **kwargs,
        )
        # target it so we can target both transformers
        self.target_lora_modules = ["DualWanTransformer3DModel"]
        self._wan_cache = None

        self.is_multistage = True
        # multistage boundaries split the models up when sampling timesteps
        # for wan 2.2 14b. the timesteps are 1000-boundary for transformer 1
        # and boundary-0 for transformer 2. T2V and I2V use different boundaries.
        self.multistage_boundaries: List[float] = (
            self._get_wan22_multistage_boundaries()
        )

        self.train_high_noise = model_config.model_kwargs.get("train_high_noise", True)
        self.train_low_noise = model_config.model_kwargs.get("train_low_noise", True)

        self.trainable_multistage_boundaries: List[int] = []
        if self.train_high_noise:
            self.trainable_multistage_boundaries.append(0)
        if self.train_low_noise:
            self.trainable_multistage_boundaries.append(1)

        if len(self.trainable_multistage_boundaries) == 0:
            raise ValueError(
                "At least one of train_high_noise or train_low_noise must be True in model.model_kwargs"
            )
        
        # if we are only training one or the other, the target LoRA modules will be the wan transformer class
        if not self.train_high_noise or not self.train_low_noise:
            self.target_lora_modules = ["WanTransformer3DModel"]

    def _get_wan22_multistage_boundaries(self) -> List[float]:
        return [self._wan_boundary_ratio, 0.0]

    @property
    def max_step_saves_to_keep_multiplier(self):
        # the cleanup mechanism checks this to see how many saves to keep
        # if we are training a LoRA, we need to set this to 2 so we keep both the high noise and low noise LoRAs at saves to keep
        if (
            self.network is not None
            and self.network.network_config.split_multistage_loras
        ):
            return 2
        return 1

    def load_model(self):
        # load model from patent parent. Wan21 not immediate parent
        # super().load_model()
        super().load_model()

        # we have to split up the model on the pipeline
        self.pipeline.transformer = self.model.transformer_1
        self.pipeline.transformer_2 = self.model.transformer_2

        # patch the condition embedder
        self.model.transformer_1.condition_embedder.forward = partial(
            time_text_monkeypatch, self.model.transformer_1.condition_embedder
        )
        self.model.transformer_2.condition_embedder.forward = partial(
            time_text_monkeypatch, self.model.transformer_2.condition_embedder
        )

    def get_bucket_divisibility(self):
        # 8x compression  and 2x2 patch size
        return 16

    def _resolve_wan22_base_lora_path(self, lora_path: str) -> str:
        if os.path.exists(lora_path):
            return lora_path

        path_split = lora_path.split("/")
        if len(path_split) == 3 and path_split[-1].endswith(".safetensors"):
            repo_id = f"{path_split[0]}/{path_split[1]}"
            self.print_and_status_update(f"Downloading Wan2.2 base merge LoRA: {lora_path}")
            return hf_hub_download(repo_id, filename=path_split[-1])

        raise ValueError(
            "Wan2.2 base merge LoRA path must be a local `.safetensors` file or a Hugging Face path "
            f"in the form `user/repo/file.safetensors`. Got: {lora_path}"
        )

    def _resolve_optional_wan22_base_lora_path(self, lora_path: str) -> Optional[str]:
        try:
            return self._resolve_wan22_base_lora_path(lora_path)
        except Exception:
            return None

    @staticmethod
    def _is_wan22_base_lora_value_set(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        return True

    @staticmethod
    def _normalize_wan22_base_lora_strength(
        strength: Any, default_strength: float, field_name: str
    ) -> float:
        if strength is None:
            strength = default_strength
        if isinstance(strength, bool):
            raise ValueError(f"Wan2.2 `{field_name}` strength must be a number, got boolean")
        try:
            normalized_strength = float(strength)
        except (TypeError, ValueError):
            raise ValueError(
                f"Wan2.2 `{field_name}` strength must be numeric. Got: {strength}"
            )
        if not math.isfinite(normalized_strength):
            raise ValueError(
                f"Wan2.2 `{field_name}` strength must be finite. Got: {strength}"
            )
        return normalized_strength

    def _normalize_wan22_base_lora_entries(
        self, value: Any, default_strength: float, field_name: str
    ) -> List[Dict[str, Any]]:
        if value is None:
            return []

        raw_entries = value if isinstance(value, (list, tuple)) else [value]
        entries: List[Dict[str, Any]] = []

        for index, entry in enumerate(raw_entries):
            entry_field_name = f"{field_name}[{index}]" if isinstance(value, (list, tuple)) else field_name
            if isinstance(entry, str):
                path = entry.strip()
                if path == "":
                    raise ValueError(f"Wan2.2 `{entry_field_name}` must be a non-empty path")
                strength = default_strength
            elif isinstance(entry, dict):
                path = entry.get("path")
                if not isinstance(path, str) or path.strip() == "":
                    raise ValueError(
                        f"Wan2.2 `{entry_field_name}` must include a non-empty `path`"
                    )
                path = path.strip()
                strength = entry.get("strength", default_strength)
            else:
                raise ValueError(
                    f"Wan2.2 `{entry_field_name}` must be a path string or an object "
                    f"with `path` and optional `strength`. Got: {type(entry).__name__}"
                )

            entries.append(
                {
                    "path": path,
                    "strength": self._normalize_wan22_base_lora_strength(
                        strength, default_strength, entry_field_name
                    ),
                }
            )

        return entries

    @staticmethod
    def _is_wan22_stage_qualified_lora_key(key: str) -> bool:
        return (
            ".transformer_1." in key
            or ".transformer_2." in key
            or "_transformer_1_" in key
            or "_transformer_2_" in key
        )

    @staticmethod
    def _is_stage_qualified_base_weight_key(key: str) -> bool:
        return Wan2214bModel._is_wan22_stage_qualified_lora_key(key)

    @staticmethod
    def _is_kohya_lora_key(key: str) -> bool:
        return key.startswith("lora_unet_")

    @staticmethod
    def _is_peft_lora_key(key: str) -> bool:
        return (
            key.startswith("transformer.")
            or key.startswith("unet.")
            or ".lora_A." in key
            or ".lora_B." in key
            or ".lora_down." in key
            or ".lora_up." in key
        )

    @staticmethod
    def _is_lycoris_lora_key(key: str) -> bool:
        return key.startswith("lycoris_") or key.startswith("lycoris_transformer_")

    @staticmethod
    def _supports_wan22_base_lora_key_format(key: str) -> bool:
        return (
            Wan2214bModel._is_stage_qualified_base_weight_key(key)
            or key.startswith("diffusion_model.")
            or Wan2214bModel._is_kohya_lora_key(key)
            or Wan2214bModel._is_peft_lora_key(key)
            or Wan2214bModel._is_lycoris_lora_key(key)
            or key.startswith("lora_transformer_")
        )

    @classmethod
    def _get_wan22_unsupported_key_error(cls, key: str) -> ValueError:
        return ValueError(
            "Wan2.2 base merge supports combined stage-qualified Wan2.2 LoRAs, "
            "`_high_noise` / `_low_noise`, `_high` / `_low`, plain single-stage Wan LoRAs when the "
            "target stage can be inferred, and Kohya-style `lora_unet_*` Wan LoRAs. "
            f"Unsupported key format: {key}"
        )

    @classmethod
    def _validate_wan22_base_lora_state_dict(cls, state_dict: Dict[str, torch.Tensor]):
        for key in state_dict.keys():
            if not cls._supports_wan22_base_lora_key_format(key):
                raise cls._get_wan22_unsupported_key_error(key)

    @staticmethod
    def _get_wan22_stage_name_from_key(key: str) -> Optional[str]:
        has_high_stage = ".transformer_1." in key or "_transformer_1_" in key
        has_low_stage = ".transformer_2." in key or "_transformer_2_" in key
        if has_high_stage and has_low_stage:
            raise ValueError(f"Wan2.2 base merge key references multiple stages: {key}")
        if has_high_stage:
            return "transformer_1"
        if has_low_stage:
            return "transformer_2"
        return None

    @staticmethod
    def _insert_stage_into_dotted_lora_key(key: str, stage_name: str) -> str:
        if "." not in key:
            raise ValueError(f"Cannot stage dotted LoRA key without module path: {key}")
        root, remainder = key.split(".", 1)
        return f"{root}.{stage_name}.{remainder}"

    @classmethod
    def _add_stage_prefix_to_wan22_lora_key(cls, key: str, stage_name: str) -> str:
        if Wan2214bModel._is_wan22_stage_qualified_lora_key(key):
            return key

        if cls._is_kohya_lora_key(key):
            return key.replace("lora_unet_", f"lora_unet_{stage_name}_", 1)

        if cls._is_peft_lora_key(key):
            if key.startswith("transformer."):
                return key.replace("transformer.", f"transformer.{stage_name}.", 1)
            if key.startswith("unet."):
                return key.replace("unet.", f"unet.{stage_name}.", 1)
            return cls._insert_stage_into_dotted_lora_key(key, stage_name)

        if cls._is_lycoris_lora_key(key):
            if key.startswith("lycoris_transformer_"):
                return key.replace("lycoris_transformer_", f"lycoris_{stage_name}_", 1)
            if key.startswith("lycoris_"):
                return key.replace("lycoris_", f"lycoris_{stage_name}_", 1)

        prefix_map = [
            ("diffusion_model.", f"diffusion_model.{stage_name}."),
            ("lycoris_diffusion_model.", f"lycoris_diffusion_model.{stage_name}."),
            ("lora_transformer_", f"lora_transformer_{stage_name}_"),
        ]
        for prefix, replacement in prefix_map:
            if key.startswith(prefix):
                return key.replace(prefix, replacement, 1)

        raise cls._get_wan22_unsupported_key_error(key)

    def _stage_wan22_lora_state_dict(
        self, state_dict: Dict[str, torch.Tensor], stage_name: str
    ) -> OrderedDict:
        staged_state_dict = OrderedDict()
        for key, value in state_dict.items():
            staged_state_dict[self._add_stage_prefix_to_wan22_lora_key(key, stage_name)] = value
        return staged_state_dict

    @staticmethod
    def _is_wan22_high_stage_file(path: str) -> bool:
        lower_path = path.lower()
        return lower_path.endswith("_high_noise.safetensors") or lower_path.endswith("_high.safetensors")

    @staticmethod
    def _is_wan22_low_stage_file(path: str) -> bool:
        lower_path = path.lower()
        return lower_path.endswith("_low_noise.safetensors") or lower_path.endswith("_low.safetensors")

    @classmethod
    def _get_wan22_split_lora_specs(cls, path: str) -> tuple[bool, str, str]:
        if cls._is_wan22_high_stage_file(path):
            if path.lower().endswith("_high_noise.safetensors"):
                return True, path, path[: -len("_high_noise.safetensors")] + "_low_noise.safetensors"
            return True, path, path[: -len("_high.safetensors")] + "_low.safetensors"
        if cls._is_wan22_low_stage_file(path):
            if path.lower().endswith("_low_noise.safetensors"):
                return True, path[: -len("_low_noise.safetensors")] + "_high_noise.safetensors", path
            return True, path[: -len("_low.safetensors")] + "_high.safetensors", path
        return False, path, path

    def _infer_single_stage_name_for_wan22_base_lora(self, lora_path: str) -> Optional[str]:
        filename = os.path.basename(lora_path)
        if self._is_wan22_high_stage_file(filename):
            return "transformer_1"
        if self._is_wan22_low_stage_file(filename):
            return "transformer_2"
        if self.train_high_noise and not self.train_low_noise:
            return "transformer_1"
        if self.train_low_noise and not self.train_high_noise:
            return "transformer_2"
        return None

    def _has_explicit_wan22_base_lora_paths(self) -> bool:
        return (
            self._is_wan22_base_lora_value_set(self.model_config.high_noise_lora_path)
            or self._is_wan22_base_lora_value_set(self.model_config.low_noise_lora_path)
        )

    def _validate_wan22_explicit_stage_state_dict(
        self, state_dict: Dict[str, torch.Tensor], stage_name: str, source_path: str
    ):
        self._validate_wan22_base_lora_state_dict(state_dict)
        for key in state_dict.keys():
            key_stage_name = self._get_wan22_stage_name_from_key(key)
            if key_stage_name is None:
                continue
            if key_stage_name != stage_name:
                raise ValueError(
                    f"Wan2.2 explicit {stage_name} LoRA path {source_path} contains keys for "
                    f"{key_stage_name}: {key}"
                )

    def _load_explicit_wan22_stage_lora_state_dict(
        self, lora_path: str, stage_name: str
    ) -> OrderedDict:
        resolved_lora_path = self._resolve_wan22_base_lora_path(lora_path)
        return self._load_explicit_wan22_stage_lora_state_dict_from_resolved_path(
            resolved_lora_path, stage_name
        )

    def _load_explicit_wan22_stage_lora_state_dict_from_resolved_path(
        self, resolved_lora_path: str, stage_name: str
    ) -> OrderedDict:
        state_dict = load_file(resolved_lora_path)
        self._validate_wan22_explicit_stage_state_dict(
            state_dict, stage_name, resolved_lora_path
        )
        return self._stage_wan22_lora_state_dict(state_dict, stage_name)

    def _build_wan22_base_lora_merge_spec(
        self,
        *,
        state_dict: Dict[str, torch.Tensor],
        strength: float,
        source_path: str,
        stage_name: Optional[str] = None,
        label: str,
    ) -> Dict[str, Any]:
        return {
            "state_dict": OrderedDict(state_dict),
            "strength": strength,
            "source_path": source_path,
            "stage_name": stage_name,
            "label": label,
        }

    def _get_explicit_wan22_base_lora_merge_specs(self) -> List[Dict[str, Any]]:
        merge_specs: List[Dict[str, Any]] = []

        high_noise_entries = self._normalize_wan22_base_lora_entries(
            self.model_config.high_noise_lora_path,
            self.model_config.high_noise_lora_merge_strength,
            "high_noise_lora_path",
        )
        low_noise_entries = self._normalize_wan22_base_lora_entries(
            self.model_config.low_noise_lora_path,
            self.model_config.low_noise_lora_merge_strength,
            "low_noise_lora_path",
        )

        for entry in high_noise_entries:
            resolved_path = self._resolve_wan22_base_lora_path(entry["path"])
            merge_specs.append(
                self._build_wan22_base_lora_merge_spec(
                    state_dict=self._load_explicit_wan22_stage_lora_state_dict_from_resolved_path(
                        resolved_path, "transformer_1"
                    ),
                    strength=entry["strength"],
                    source_path=resolved_path,
                    stage_name="transformer_1",
                    label="high-noise",
                )
            )
        for entry in low_noise_entries:
            resolved_path = self._resolve_wan22_base_lora_path(entry["path"])
            merge_specs.append(
                self._build_wan22_base_lora_merge_spec(
                    state_dict=self._load_explicit_wan22_stage_lora_state_dict_from_resolved_path(
                        resolved_path, "transformer_2"
                    ),
                    strength=entry["strength"],
                    source_path=resolved_path,
                    stage_name="transformer_2",
                    label="low-noise",
                )
            )

        if len(merge_specs) == 0:
            raise ValueError(
                "Wan2.2 explicit base merge mode requires at least one of "
                "`high_noise_lora_path` or `low_noise_lora_path`."
            )

        return merge_specs

    def _get_legacy_wan22_base_lora_merge_specs(
        self,
        lora_path: str,
        merge_strength: Optional[float] = None,
        update_model_config_path: bool = True,
    ) -> List[Dict[str, Any]]:
        if merge_strength is None:
            merge_strength = self.model_config.lora_merge_strength

        is_split_lora, high_noise_spec, low_noise_spec = self._get_wan22_split_lora_specs(lora_path)
        if not is_split_lora:
            resolved_lora_path = self._resolve_wan22_base_lora_path(lora_path)
            if update_model_config_path:
                self.model_config.lora_path = resolved_lora_path
            state_dict = load_file(resolved_lora_path)
            self._validate_wan22_base_lora_state_dict(state_dict)
            label = "combined"
            stage_name = None
            if not any(self._is_wan22_stage_qualified_lora_key(key) for key in state_dict):
                stage_name = self._infer_single_stage_name_for_wan22_base_lora(resolved_lora_path)
                if stage_name is None:
                    raise ValueError(
                        "Wan2.2 base merge found a valid single-stage LoRA, but the target stage could not be "
                        "inferred. Use a filename ending in `_high_noise`, `_low_noise`, `_high`, or `_low`, "
                        "or set only one of `train_high_noise` / `train_low_noise`."
                    )
                state_dict = self._stage_wan22_lora_state_dict(state_dict, stage_name)
                label = "single-stage"
            return [
                self._build_wan22_base_lora_merge_spec(
                    state_dict=state_dict,
                    strength=merge_strength,
                    source_path=resolved_lora_path,
                    stage_name=stage_name,
                    label=label,
                )
            ]

        high_noise_path = self._resolve_optional_wan22_base_lora_path(high_noise_spec)
        low_noise_path = self._resolve_optional_wan22_base_lora_path(low_noise_spec)

        merge_specs: List[Dict[str, Any]] = []
        if high_noise_path is not None:
            high_noise_lora = load_file(high_noise_path)
            self._validate_wan22_base_lora_state_dict(high_noise_lora)
            merge_specs.append(
                self._build_wan22_base_lora_merge_spec(
                    state_dict=self._stage_wan22_lora_state_dict(high_noise_lora, "transformer_1"),
                    strength=merge_strength,
                    source_path=high_noise_path,
                    stage_name="transformer_1",
                    label="high-noise sibling",
                )
            )
        if low_noise_path is not None:
            low_noise_lora = load_file(low_noise_path)
            self._validate_wan22_base_lora_state_dict(low_noise_lora)
            merge_specs.append(
                self._build_wan22_base_lora_merge_spec(
                    state_dict=self._stage_wan22_lora_state_dict(low_noise_lora, "transformer_2"),
                    strength=merge_strength,
                    source_path=low_noise_path,
                    stage_name="transformer_2",
                    label="low-noise sibling",
                )
            )

        if len(merge_specs) == 0:
            raise ValueError(
                "Wan2.2 base merge could not resolve a valid `_high_noise` or `_low_noise` LoRA sibling. "
                f"Expected local files or Hugging Face paths derived from: {lora_path}"
            )

        if update_model_config_path and high_noise_path is not None:
            self.model_config.lora_path = high_noise_path
        elif update_model_config_path and low_noise_path is not None:
            self.model_config.lora_path = low_noise_path

        return merge_specs

    def _get_wan22_base_lora_merge_specs(self) -> List[Dict[str, Any]]:
        if self._has_explicit_wan22_base_lora_paths():
            return self._get_explicit_wan22_base_lora_merge_specs()
        if not self._is_wan22_base_lora_value_set(self.model_config.lora_path):
            return []

        lora_path_value = self.model_config.lora_path
        entries = self._normalize_wan22_base_lora_entries(
            lora_path_value, self.model_config.lora_merge_strength, "lora_path"
        )
        update_model_config_path = isinstance(lora_path_value, str) and len(entries) == 1
        merge_specs: List[Dict[str, Any]] = []
        for entry in entries:
            merge_specs.extend(
                self._get_legacy_wan22_base_lora_merge_specs(
                    entry["path"],
                    merge_strength=entry["strength"],
                    update_model_config_path=update_model_config_path,
                )
            )
        return merge_specs

    def _load_wan22_base_lora_state_dict(self, lora_path: Any) -> OrderedDict:
        if self._has_explicit_wan22_base_lora_paths():
            merge_specs = self._get_explicit_wan22_base_lora_merge_specs()
        else:
            entries = self._normalize_wan22_base_lora_entries(
                lora_path, self.model_config.lora_merge_strength, "lora_path"
            )
            merge_specs = []
            for entry in entries:
                merge_specs.extend(
                    self._get_legacy_wan22_base_lora_merge_specs(
                        entry["path"],
                        merge_strength=entry["strength"],
                        update_model_config_path=isinstance(lora_path, str) and len(entries) == 1,
                    )
                )
        combined_state_dict = OrderedDict()
        for merge_spec in merge_specs:
            combined_state_dict.update(merge_spec["state_dict"])
        return combined_state_dict

    def _infer_wan22_base_lora_network_config(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> NetworkConfig:
        converted_state_dict = self.convert_lora_weights_before_load(state_dict)

        is_lokr = any("lokr_w1" in key for key in converted_state_dict)
        is_lora = any("lora_A" in key or "lora_down" in key for key in converted_state_dict)
        if not is_lora and not is_lokr:
            raise ValueError(
                "Wan2.2 base merge only supports Wan2.2 LoRAs with recognizable LoRA or LoKr weights."
            )

        network_kwargs: Dict[str, Any] = {
            "only_if_contains": [],
            "target_lin_modules": ["DualWanTransformer3DModel"],
        }
        network_config: Dict[str, Any] = {
            "type": "lora",
            "network_kwargs": network_kwargs,
            "transformer_only": False,
            "old_lokr_format": self.use_old_lokr_format,
        }

        if is_lokr:
            largest_factor = 0
            only_if_contains = []
            for key, value in converted_state_dict.items():
                if "lokr_w1" not in key:
                    continue
                largest_factor = max(largest_factor, int(value.shape[0]))
                contains_key = key.split(".lokr_w1")[0].replace("lycoris_", "")
                if contains_key not in only_if_contains:
                    only_if_contains.append(contains_key)
            network_config["type"] = "lokr"
            network_config["lokr_full_rank"] = True
            network_config["lokr_factor"] = largest_factor
            network_kwargs["only_if_contains"] = only_if_contains
        else:
            linear_dim = None
            only_if_contains = []
            for key, value in converted_state_dict.items():
                if "lora_A" in key:
                    linear_dim = int(value.shape[0])
                    contains_key = key.split(".lora_A")[0]
                elif "lora_down" in key:
                    linear_dim = int(value.shape[0])
                    contains_key = key.split(".lora_down")[0]
                else:
                    continue
                if contains_key not in only_if_contains:
                    only_if_contains.append(contains_key)

            if linear_dim is None:
                raise ValueError("Wan2.2 base merge could not infer the LoRA rank from the provided weights.")

            network_config["linear"] = linear_dim
            network_config["linear_alpha"] = linear_dim
            network_kwargs["only_if_contains"] = only_if_contains

        return NetworkConfig(**network_config)

    def _cleanup_wan22_base_lora_network(self, network: LoRASpecialNetwork):
        if not hasattr(network, "get_all_modules"):
            return
        for module in network.get_all_modules():
            if hasattr(module, "org_forward"):
                module.org_module[0].forward = module.org_forward

    def _merge_base_lora_into_wan22_transformer(
        self, transformer: DualWanTransformer3DModel
    ):
        merge_specs = self._get_wan22_base_lora_merge_specs()
        if len(merge_specs) == 0:
            return

        self.print_and_status_update("Loading Wan2.2 base merge LoRA")
        for merge_spec in merge_specs:
            lora_state_dict = merge_spec["state_dict"]
            merge_strength = merge_spec["strength"]
            network_config = self._infer_wan22_base_lora_network_config(lora_state_dict)

            stage_label = merge_spec["label"]
            if merge_spec["stage_name"] is not None:
                stage_label = f"{stage_label} ({merge_spec['stage_name']})"
            self.print_and_status_update(
                f"Merging Wan2.2 base LoRA into transformers: {stage_label} "
                f"from {merge_spec['source_path']} (strength {merge_strength})"
            )
            network = LoRASpecialNetwork(
                text_encoder=None,
                unet=transformer,
                lora_dim=network_config.linear,
                multiplier=1.0,
                alpha=network_config.linear_alpha,
                train_unet=True,
                train_text_encoder=False,
                conv_lora_dim=network_config.conv,
                conv_alpha=network_config.conv_alpha,
                is_transformer=True,
                network_config=network_config,
                network_type=network_config.type,
                transformer_only=network_config.transformer_only,
                base_model=self,
                **network_config.network_kwargs,
            )
            network.apply_to(None, transformer, False, True)
            try:
                network.load_weights(lora_state_dict)
                network.merge_in(merge_strength)
            finally:
                self._cleanup_wan22_base_lora_network(network)
                del network
                flush()

    def load_wan_transformer(self, transformer_path, subfolder=None):
        if self.model_config.split_model_over_gpus:
            raise ValueError(
                "Splitting model over gpus is not supported for Wan2.2 models"
            )

        if (
            self.model_config.assistant_lora_path is not None
            or self.model_config.inference_lora_path is not None
        ):
            raise ValueError(
                "Assistant LoRA is not supported for Wan2.2 models currently"
            )

        # transformer path can be a directory that ends with /transformer or a hf path.

        transformer_path_1 = transformer_path
        subfolder_1 = subfolder

        transformer_path_2 = transformer_path
        subfolder_2 = subfolder

        if subfolder_2 is None:
            # we have a local path, replace it with transformer_2 folder
            transformer_path_2 = os.path.join(
                os.path.dirname(transformer_path_1), "transformer_2"
            )
        else:
            # we have a hf path, replace it with transformer_2 subfolder
            subfolder_2 = "transformer_2"

        self.print_and_status_update("Loading transformer 1")
        dtype = self.torch_dtype
        transformer_1 = WanTransformer3DModel.from_pretrained(
            transformer_path_1,
            subfolder=subfolder_1,
            torch_dtype=dtype,
        ).to(dtype=dtype)

        flush()

        if self.model_config.low_vram:
            # quantize on the device
            transformer_1.to('cpu', dtype=dtype)
            flush()
        else:
            transformer_1.to(self.device_torch, dtype=dtype)
            flush()

        self.print_and_status_update("Loading transformer 2")
        dtype = self.torch_dtype
        transformer_2 = WanTransformer3DModel.from_pretrained(
            transformer_path_2,
            subfolder=subfolder_2,
            torch_dtype=dtype,
        ).to(dtype=dtype)

        flush()

        if self.model_config.low_vram:
            # quantize on the device
            transformer_2.to('cpu', dtype=dtype)
            flush()
        else:
            transformer_2.to(self.device_torch, dtype=dtype)
            flush()

        layer_offloading_transformer = self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0
        # make the combined model
        self.print_and_status_update("Creating DualWanTransformer3DModel")
        transformer = DualWanTransformer3DModel(
            transformer_1=transformer_1,
            transformer_2=transformer_2,
            torch_dtype=self.torch_dtype,
            device=self.device_torch,
            boundary_ratio=self._wan_boundary_ratio,
            low_vram=self.model_config.low_vram,
        )

        self._merge_base_lora_into_wan22_transformer(transformer)

        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            self.print_and_status_update("Quantizing Transformer 1")
            quantize_model(self, transformer_1)
            flush()

            self.print_and_status_update("Quantizing Transformer 2")
            quantize_model(self, transformer_2)
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer 1 to CPU")
            transformer_1.to("cpu")
            self.print_and_status_update("Moving transformer 2 to CPU")
            transformer_2.to("cpu")
        else:
            transformer_1.to(self.device_torch)
            transformer_2.to(self.device_torch)
        
        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is not None:
            # apply the accuracy recovery adapter to both transformers
            self.print_and_status_update("Applying Accuracy Recovery Adapter to Transformers")
            quantize_model(self, transformer)
            flush()
            
        
        if layer_offloading_transformer:
            MemoryManager.attach(
                transformer_1,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer_1.scale_shift_table] + [block.scale_shift_table for block in transformer_1.blocks]
            )
            MemoryManager.attach(
                transformer_2,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer_2.scale_shift_table] + [block.scale_shift_table for block in transformer_2.blocks]
            )

        return transformer

    def get_generation_pipeline(self):
        # todo unipc got broken in a diffusers update. Use euler for now.
        # scheduler = UniPCMultistepScheduler(**self._wan_generation_scheduler_config)
        scheduler = self.get_train_scheduler()
        pipeline = Wan22Pipeline(
            vae=self.vae,
            transformer=self.model.transformer_1,
            transformer_2=self.model.transformer_2,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=scheduler,
            expand_timesteps=self._wan_expand_timesteps,
            device=self.device_torch,
            aggressive_offload=self.model_config.low_vram,
            boundary_ratio=self._wan_boundary_ratio,
        )

        # pipeline = pipeline.to(self.device_torch)

        return pipeline

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler():
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler

    def get_base_model_version(self):
        return "wan_2.2_14b"

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs,
    ):
        # todo do we need to override this? Adjust timesteps?
        return super().get_noise_prediction(
            latent_model_input=latent_model_input,
            timestep=timestep,
            text_embeddings=text_embeddings,
            batch=batch,
            **kwargs,
        )

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer_combo: DualWanTransformer3DModel = unwrap_model(self.model)
        transformer_combo.transformer_1.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        transformer_combo.transformer_2.save_pretrained(
            save_directory=os.path.join(output_path, "transformer_2"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def save_lora(
        self,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.network.network_config.split_multistage_loras:
            # just save as a combo lora
            save_file(state_dict, output_path, metadata=metadata)
            return

        # we need to build out both dictionaries for high and low noise LoRAs
        high_noise_lora = {}
        low_noise_lora = {}
        
        only_train_high_noise = self.train_high_noise and not self.train_low_noise
        only_train_low_noise = self.train_low_noise and not self.train_high_noise

        for key in state_dict:
            if ".transformer_1." in key or only_train_high_noise:
                # this is a high noise LoRA
                new_key = key.replace(".transformer_1.", ".")
                high_noise_lora[new_key] = state_dict[key]
            elif ".transformer_2." in key or only_train_low_noise:
                # this is a low noise LoRA
                new_key = key.replace(".transformer_2.", ".")
                low_noise_lora[new_key] = state_dict[key]

        # loras have either LORA_MODEL_NAME_000005000.safetensors or LORA_MODEL_NAME.safetensors
        if len(high_noise_lora.keys()) > 0:
            # save the high noise LoRA
            high_noise_lora_path = output_path.replace(
                ".safetensors", "_high_noise.safetensors"
            )
            save_file(high_noise_lora, high_noise_lora_path, metadata=metadata)

        if len(low_noise_lora.keys()) > 0:
            # save the low noise LoRA
            low_noise_lora_path = output_path.replace(
                ".safetensors", "_low_noise.safetensors"
            )
            save_file(low_noise_lora, low_noise_lora_path, metadata=metadata)

    def load_lora(self, file: str):
        # if it doesnt have high_noise or low_noise, it is a combo LoRA
        is_split_lora, high_noise_lora_path, low_noise_lora_path = self._get_wan22_split_lora_specs(file)
        if not is_split_lora:
            # this is a combined LoRA, we dont need to split it up
            sd = load_file(file)
            return sd

        combined_dict = {}

        if os.path.exists(high_noise_lora_path) and self.train_high_noise:
            # load the high noise LoRA
            high_noise_lora = load_file(high_noise_lora_path)
            for key in high_noise_lora:
                new_key = key.replace(
                    "diffusion_model.", "diffusion_model.transformer_1."
                )
                combined_dict[new_key] = high_noise_lora[key]
        if os.path.exists(low_noise_lora_path) and self.train_low_noise:
            # load the low noise LoRA
            low_noise_lora = load_file(low_noise_lora_path)
            for key in low_noise_lora:
                new_key = key.replace(
                    "diffusion_model.", "diffusion_model.transformer_2."
                )
                combined_dict[new_key] = low_noise_lora[key]
        
        # if we are not training both stages, we wont have transformer designations in the keys
        if not self.train_high_noise or not self.train_low_noise:
            new_dict = {}
            for key in combined_dict:
                if ".transformer_1." in key:
                    new_key = key.replace(".transformer_1.", ".")
                elif ".transformer_2." in key:
                    new_key = key.replace(".transformer_2.", ".")
                else:
                    new_key = key
                new_dict[new_key] = combined_dict[key]
            combined_dict = new_dict

        return combined_dict
    
    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)
        # todo, figure out how to do video
        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra
        )[0]

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img

    def get_model_to_train(self):
        # todo, loras wont load right unless they have the transformer_1 or transformer_2 in the key.
        # called when setting up the LoRA. We only need to get the model for the stages we want to train.
        if self.train_high_noise and self.train_low_noise:
            # we are training both stages, return the unified model
            return self.model
        elif self.train_high_noise:
            # we are only training the high noise stage, return transformer_1
            return self.model.transformer_1
        elif self.train_low_noise:
            # we are only training the low noise stage, return transformer_2
            return self.model.transformer_2
        else:
            raise ValueError(
                "At least one of train_high_noise or train_low_noise must be True in model.model_kwargs"
            )
