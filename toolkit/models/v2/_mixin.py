"""
Universal model mixin for every model class the toolkit touches (transformers,
unets, text encoders, VAEs, ...).

OstrisModelMixin gives a model class one loading entry point, `load_model`, that
digests any of:
  - a local diffusers checkpoint directory (the model folder itself or a full
    checkpoint that contains it under `aitk_subfolder`)
  - a HuggingFace repo id ("org/repo")
  - a local single .safetensors file in the model's original key layout
  - a remote single .safetensors file ("org/repo/path/file.safetensors")
plus optional automatic quantization of the loaded weights (`qtype=...`).

Model specific behavior lives in small overridable hooks (key conversion, config
source, block names, backend loading), so subclassing for a new model usually means
setting the `aitk_*` class attrs and overriding one or two hooks. The default
backend hooks (`aitk_from_pretrained` / `aitk_load_config` / `aitk_from_config`)
speak the diffusers ModelMixin API; transformers-lib models (text encoders) override
those three to speak PreTrainedModel/AutoConfig instead.
"""

import os
from typing import Dict, List, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm

from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin

from toolkit.basic import flush


class BasicModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    pass


class OstrisModelMixin:
    # ---- per-model configuration, override in subclasses ----
    # subfolder that holds this model inside a diffusers style checkpoint
    # (e.g. "transformer", "text_encoder", "vae")
    aitk_subfolder: Optional[str] = None
    # hub repo (or local path) to pull the model config from when loading a single
    # .safetensors file and no config_path is given
    aitk_config_repo: Optional[str] = None

    # ---- state set by the loader / quantizer ----
    aitk_is_quantized: bool = False
    aitk_qtype: Optional[str] = None

    # ------------------------------------------------------------------
    # extendable hooks
    # ------------------------------------------------------------------

    @classmethod
    def convert_state_dict_on_load(cls, state_dict: Dict[str, torch.Tensor]):
        """Convert a single-file state dict from the model's original key layout to
        this class's layout. Default is a passthrough."""
        return state_dict

    @classmethod
    def convert_state_dict_on_save(cls, state_dict: Dict[str, torch.Tensor]):
        """Convert this class's state dict back to the original single-file layout.
        Default is a passthrough."""
        return state_dict

    @classmethod
    def get_quantization_block_names(cls) -> Optional[List[str]]:
        """Names (dotted paths allowed) of the repeated block lists to quantize one
        block at a time so the whole model never has to sit on the gpu at once."""
        return None

    @classmethod
    def get_quantization_exclude_modules(cls) -> Optional[List[str]]:
        """fnmatch patterns of sensitive modules to keep in full precision."""
        return None

    # ---- backend hooks: default to the diffusers ModelMixin API. transformers-lib
    # models (text encoders) override these three to use PreTrainedModel/AutoConfig.

    @classmethod
    def aitk_from_pretrained(cls, path, subfolder=None, dtype=None, **kwargs):
        return cls.from_pretrained(
            path, subfolder=subfolder, torch_dtype=dtype, **kwargs
        )

    @classmethod
    def aitk_load_config(cls, path, subfolder=None):
        return cls.load_config(path, subfolder=subfolder)

    @classmethod
    def aitk_from_config(cls, config):
        with torch.device("meta"):
            return cls.from_config(config)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def load_model(
        cls,
        name_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        qtype: Optional[str] = None,
        quantize_device: Optional[torch.device] = None,
        exclude_quant_modules: Optional[List[str]] = None,
        config_path: Optional[str] = None,
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        """Load a model universally from a given name or path.

        name_or_path can be a local diffusers directory, a hub repo id, a local
        single .safetensors file, or a remote single file
        ("org/repo/file.safetensors").

        qtype: quantize the weights after loading. quantize_device: where to run the
        quantization math; blocks are moved there one at a time and returned to where
        they were.
        config_path: config source for single-file loads, overriding aitk_config_repo.
        device: move the finished model there before returning.
        """
        if subfolder is None:
            subfolder = cls.aitk_subfolder

        if name_or_path.endswith(".safetensors"):
            file_path = cls._resolve_single_file(name_or_path)
            model = cls._load_single_file(
                file_path, dtype=dtype, config_path=config_path, subfolder=subfolder
            )
        else:
            if os.path.isdir(name_or_path):
                # a local dir may be the model folder itself or a full checkpoint
                # that nests it under the subfolder
                if subfolder is not None and not os.path.isdir(
                    os.path.join(name_or_path, subfolder)
                ):
                    subfolder = None
            model = cls.aitk_from_pretrained(
                name_or_path, subfolder=subfolder, dtype=dtype, **kwargs
            )

        if qtype is not None:
            model.quantize_(
                qtype, device=quantize_device, exclude=exclude_quant_modules
            )

        if device is not None:
            model.to(device)
        return model

    @staticmethod
    def _resolve_single_file(name_or_path: str) -> str:
        """Resolve a .safetensors reference to a local file, downloading
        'org/repo/path/file.safetensors' hub references as needed."""
        if os.path.isfile(name_or_path):
            return name_or_path
        parts = name_or_path.split("/")
        if len(parts) < 3:
            raise ValueError(
                f"'{name_or_path}' is not a local file or a hub file reference "
                "('org/repo/filename.safetensors')."
            )
        return hf_hub_download(
            repo_id="/".join(parts[:2]), filename="/".join(parts[2:])
        )

    @classmethod
    def _load_single_file_config(
        cls, config_path: Optional[str], subfolder: Optional[str]
    ):
        config_source = config_path if config_path is not None else cls.aitk_config_repo
        if config_source is None:
            raise ValueError(
                f"{cls.__name__} cannot load a single-file checkpoint without a "
                "config source; pass config_path or set aitk_config_repo."
            )
        if (
            subfolder is not None
            and os.path.isdir(config_source)
            and not os.path.isdir(os.path.join(config_source, subfolder))
        ):
            subfolder = None
        return cls.aitk_load_config(config_source, subfolder=subfolder)

    @classmethod
    def _load_single_file(
        cls,
        file_path: str,
        dtype: torch.dtype,
        config_path: Optional[str] = None,
        subfolder: Optional[str] = None,
    ):
        config = cls._load_single_file_config(config_path, subfolder)

        state_dict = load_file(file_path)
        state_dict = cls.convert_state_dict_on_load(state_dict)
        for key, value in state_dict.items():
            state_dict[key] = value.to(dtype=dtype)

        model = cls.aitk_from_config(config)
        model.load_state_dict(state_dict, assign=True)
        model.to(dtype=dtype)
        del state_dict
        flush()
        return model

    # ------------------------------------------------------------------
    # quantization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def quantize_(
        self,
        qtype: str,
        device: Optional[torch.device] = None,
        exclude: Optional[List[str]] = None,
    ):
        """Quantize the model weights in place. When device is given, the repeated
        blocks (get_quantization_block_names) are moved there one at a time for the
        quantization math and returned to their original device, so the whole model
        never has to fit on the gpu in full precision."""
        from optimum.quanto import freeze
        from toolkit.dequantize import patch_dequantization_on_save
        from toolkit.util.quantize import get_qtype, quantize

        # make full-model saves emit plain full precision weights
        patch_dequantization_on_save(self)

        quantization_type = get_qtype(qtype)
        exclude = list(exclude or []) + list(
            self.get_quantization_exclude_modules() or []
        )

        blocks: List[torch.nn.Module] = []
        for name in self.get_quantization_block_names() or []:
            # name may be a dotted path for models that nest their blocks
            block_list = self
            for part in name.split("."):
                block_list = getattr(block_list, part, None)
                if block_list is None:
                    break
            if block_list is not None:
                blocks += list(block_list)

        for block in tqdm(blocks, desc=f"Quantizing blocks ({qtype})"):
            first_param = next(block.parameters(), None)
            orig_device = first_param.device if first_param is not None else None
            if device is not None and orig_device is not None:
                block.to(device, non_blocking=True)
            quantize(block, weights=quantization_type)
            freeze(block)
            if device is not None and orig_device is not None:
                # NOT non_blocking: an async D2H allocates the cpu destination in pinned
                # memory, which the caching host allocator keeps forever (with power-of-2
                # bucket rounding on top) — that silently retained a model-sized chunk of
                # host ram after the weights moved back to the gpu for training
                block.to(orig_device)

        # everything the block pass did not cover (embedders, norms, projections, ...)
        quantize(self, weights=quantization_type, exclude=exclude)
        freeze(self)

        self.aitk_is_quantized = True
        self.aitk_qtype = qtype
        flush()
        return self
