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
plus optional automatic quantization of the loaded weights (`qtype=...`). A
single-file checkpoint carrying ComfyUI `comfy_quant` markers loads with its
pre-quantized layers attached to the toolkit's quantization backends
(convrot8 / nvfp4) automatically. `save_model` writes a single-file
.safetensors back in the original (comfy) key layout.

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


def adopt_component(module: torch.nn.Module, wrapper_cls: type) -> torch.nn.Module:
    """Rebind an already-loaded component instance onto its v2 wrapper class
    (in-place class swap, like the OstrisLinear conversion). Used when a
    component arrives from machinery that builds the base class directly
    (e.g. a diffusers pipeline load), so every resident component is a mixin
    instance the inference engine can pool and manage."""
    if isinstance(module, wrapper_cls):
        return module
    base = wrapper_cls.__mro__[1]
    if not isinstance(module, base):
        raise TypeError(
            f"cannot adopt {type(module).__name__} into {wrapper_cls.__name__} "
            f"(expected a {base.__name__})"
        )
    module.__class__ = wrapper_cls
    return module


class OstrisModelMixin:
    # ---- per-model configuration, override in subclasses ----
    # subfolder that holds this model inside a diffusers style checkpoint
    # (e.g. "transformer", "text_encoder", "vae")
    aitk_subfolder: Optional[str] = None
    # hub repo (or local path) to pull the model config from when loading a single
    # .safetensors file and no config_path is given
    aitk_config_repo: Optional[str] = None

    # ---- comfy weight sources ----
    # hub repo the comfy-format weight files are published in (Comfy-Org/...)
    aitk_comfy_repo: Optional[str] = None
    # standard name_or_path (hub repo id) -> repo-relative comfy weight file
    # candidates for this module: every precision variant the class can digest.
    # Selection ranks them convrot8 > float8 mixed > float8 > bf16 > fp16
    # (resolver.comfy_precision_rank); a locally-present candidate always wins
    # over a download.
    aitk_comfy_weight_names: Dict[str, List[str]] = {}

    # ---- tokenizer/processor source, for text-encoder modules ----
    aitk_tokenizer_repo: Optional[str] = None
    aitk_tokenizer_subfolder: Optional[str] = None
    aitk_processor_repo: Optional[str] = None
    aitk_processor_subfolder: Optional[str] = None

    # single-file loads without quant markers cast tensors to the requested
    # dtype; classes whose checkpoints carry a deliberate precision mix (e.g.
    # fp32 norms next to bf16 weights) set this False to load at stored
    # precision instead
    aitk_cast_on_load: bool = True

    # pre-quantized (comfy marker) checkpoints normally load their
    # NON-quantized tensors at stored precision (the mix can be deliberate,
    # e.g. ltx2.5's fp32 tables). Classes whose forward assumes one uniform
    # dtype (wan: fp16 conv biases next to fp32 tables in the fp8 files) set
    # this True to cast those float tensors to the load dtype instead
    aitk_cast_quantized_load: bool = False

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
    def get_transformer_block_names(cls) -> Optional[List[str]]:
        """Names (dotted paths allowed) of the repeated block lists to quantize one
        block at a time so the whole model never has to sit on the gpu at once."""
        return None

    @classmethod
    def get_quantization_exclude_modules(cls) -> Optional[List[str]]:
        """fnmatch patterns of sensitive modules to keep in full precision."""
        return None

    def get_offload_ignore_modules(self) -> Optional[List]:
        """Module/parameter objects layer offloading must keep resident (pad
        tokens, modulation tables, ...). Default: none."""
        return None

    # ---- backend hooks: default to the diffusers ModelMixin API. transformers-lib
    # models (text encoders) override these three to use PreTrainedModel/AutoConfig.

    @classmethod
    def aitk_from_pretrained(cls, path, subfolder=None, dtype=None, **kwargs):
        return cls._local_first(
            cls.from_pretrained, path, subfolder=subfolder, torch_dtype=dtype, **kwargs
        )

    @classmethod
    def aitk_load_config(cls, path, subfolder=None):
        return cls._local_first(cls.load_config, path, subfolder=subfolder)

    @staticmethod
    def _local_first(loader, *args, **kwargs):
        """Serve fully-cached repos without touching the hub. Online
        from_pretrained does an etag HEAD round-trip per file (seconds of pure
        network latency on every warm load); a local_files_only attempt is
        instant when the cache can't satisfy it, so failing over to the
        online path costs nothing."""
        import os as _os

        if _os.environ.get("HF_HUB_OFFLINE") == "1":
            return loader(*args, **kwargs)
        try:
            return loader(*args, local_files_only=True, **kwargs)
        except Exception:
            return loader(*args, **kwargs)

    @classmethod
    def aitk_from_config(cls, config):
        # params on meta (materialized by the state-dict assign), buffers real:
        # non-persistent buffers (e.g. rope tables) are computed at init and
        # never appear in checkpoints
        from accelerate import init_empty_weights

        with init_empty_weights(include_buffers=False):
            if hasattr(cls, "from_config"):
                return cls.from_config(config)
            # plain nn.Module classes take their params/config object directly
            return cls(config)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        name_or_path: str,
        qtype: Optional[str] = None,
        offload: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        base_model=None,
        quantize_device: Optional[torch.device] = None,
        exclude_quant_modules: Optional[List[str]] = None,
        config_path: Optional[str] = None,
        config=None,
        subfolder: Optional[str] = None,
        use_comfy_weights: bool = True,
        **kwargs,
    ):
        """One-call component load. Resolves/downloads the source (comfy
        candidates included), builds the module, quantizes it, attaches layer
        offloading, and places it — everything a holder used to hand-roll.

        qtype: quantization type; "type|path.safetensors" also attaches an
        accuracy recovery adapter (requires base_model for the network
        wiring). Pre-quantized checkpoints skip quantization automatically.
        offload: layer-offload fraction (0 disables); the class's
        get_offload_ignore_modules hook pins sensitive pieces.
        device: final placement. With offload active the memory manager owns
        staging, so no blanket device move happens.
        base_model: the holder — supplies status prints, quantize_kwargs, and
        ARA wiring when given.
        """
        model = cls.load_model(
            name_or_path,
            dtype=dtype,
            config_path=config_path,
            config=config,
            subfolder=subfolder,
            use_comfy_weights=use_comfy_weights,
            # comfy candidate ranking prefers the file whose shipped
            # quantization matches the request (strip any ARA suffix);
            # quantization itself happens in aitk_post_load below
            qtype=(qtype or "").split("|", 1)[0] or None,
            quantize_on_load=False,
            **kwargs,
        )
        return model.aitk_post_load(
            qtype=qtype,
            offload=offload,
            dtype=dtype,
            device=device,
            base_model=base_model,
            quantize_device=quantize_device,
            exclude_quant_modules=exclude_quant_modules,
        )

    def aitk_post_load(
        self,
        qtype: Optional[str] = None,
        offload: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        base_model=None,
        quantize_device: Optional[torch.device] = None,
        exclude_quant_modules: Optional[List[str]] = None,
        **_load_only_kwargs,  # tolerate component_load_kwargs() extras
    ):
        """The post-load half of `load` — quantize (incl. "qtype|ara"), attach
        layer offloading, place on device. Callable directly by holders whose
        checkpoint sourcing is custom but want the standard policy."""
        from toolkit.print import print_acc

        status_fn = (
            base_model.print_and_status_update if base_model is not None else print_acc
        )

        ara_path = None
        if qtype is not None and "|" in qtype:
            qtype, ara_path = qtype.split("|", 1)

        # pre-quantized checkpoint: keep the shipped quantization IFF it
        # exactly matches the request. Anything else — no quantization
        # requested (full finetuning), a different backend, or an ARA —
        # restores/requantizes to what was asked for.
        if getattr(self, "aitk_is_quantized", False):
            from toolkit.util.ostris_quant import OstrisLinear
            from toolkit.util.quantize import (
                dequantize_ostris_to_linear,
                get_qtype,
                ostristype,
            )

            shipped = sorted(
                {
                    getattr(m.ostris_quantizer, "qtype", None)
                    for m in self.modules()
                    if isinstance(m, OstrisLinear)
                }
                - {None}
            )
            # a checkpoint may deliberately mix backends (minimax's TE: nvfp4
            # LM linears + int8 embeddings). The request matches when the
            # requested backend is among the shipped ones — the whole shipped
            # quantization is then kept exactly as-is.
            matches = (
                qtype is not None
                and ara_path is None
                and qtype in shipped
            )
            if matches:
                status_fn(
                    f"Checkpoint is pre-quantized ({'/'.join(shipped)}); keeping it"
                )
            else:
                target_is_ostris = qtype is not None and isinstance(
                    get_qtype(qtype), ostristype
                )
                if qtype is None or ara_path is not None or not target_is_ostris:
                    # full precision requested, an ARA (quantizes fresh from
                    # full weights), or a quanto/torchao backend that cannot
                    # re-quantize an OstrisLinear: dequantize first
                    status_fn(
                        f"Dequantizing shipped {'/'.join(shipped)} weights "
                        f"-> {qtype or 'full precision'}"
                    )
                    dequantize_ostris_to_linear(self)
                else:
                    # ostris -> ostris: quantize_module re-quantizes each
                    # layer in place (dequant -> requant, one layer at a time)
                    status_fn(f"Requantizing {'/'.join(shipped)} -> {qtype}")
                self.aitk_is_quantized = False
                self.aitk_qtype = None

        if qtype and not getattr(self, "aitk_is_quantized", False):
            from toolkit.util.quantize import (
                attach_ara_and_quantize,
                quantize_module,
            )

            exclude = list(exclude_quant_modules or []) + list(
                self.get_quantization_exclude_modules() or []
            )
            q_device = quantize_device if quantize_device is not None else device
            # final home is the quantize gpu (no offloading, no cpu
            # parking): quantized weights stay put — one bus crossing
            # instead of three, and no model-sized host-ram copy
            keep_on_device = (
                not offload
                and device is not None
                and q_device is not None
                and torch.device(device) == torch.device(q_device)
                and torch.device(device).type != "cpu"
            )
            if ara_path is not None:
                if base_model is None:
                    raise ValueError(
                        "an accuracy recovery adapter needs base_model= for the "
                        "network wiring"
                    )
                status_fn("Quantizing with accuracy recovery adapter")
                attach_ara_and_quantize(
                    base_model,
                    self,
                    ara_path,
                    exclude=exclude,
                    device=q_device,
                    keep_on_device=keep_on_device,
                )
            else:
                status_fn(f"Quantizing ({qtype})")
                quantize_module(
                    self,
                    qtype,
                    device=q_device,
                    dtype=dtype,
                    block_names=self.get_transformer_block_names(),
                    exclude=exclude,
                    quantize_kwargs=(
                        base_model.model_config.quantize_kwargs
                        if base_model is not None
                        else None
                    ),
                    status_fn=status_fn,
                    keep_on_device=keep_on_device,
                )
            self.aitk_is_quantized = True
            self.aitk_qtype = qtype

        if offload and offload > 0:
            from toolkit.memory_management import MemoryManager

            # the manager stages layers to the COMPUTE device. `device` is the
            # component's parking device — "cpu" under low_vram — which must
            # never become the manager's process device (bounces would stage
            # to cpu against cuda activations). quantize_device carries the
            # actual gpu in the holder flow.
            compute_device = quantize_device if quantize_device is not None else device
            MemoryManager.attach(
                self,
                torch.device(compute_device),
                offload_percent=offload,
                ignore_modules=list(self.get_offload_ignore_modules() or []),
            )
        elif device is not None:
            self.to(device)
        return self

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
        config=None,
        subfolder: Optional[str] = None,
        use_comfy_weights: bool = True,
        quantize_on_load: bool = True,
        **kwargs,
    ):
        """Load a model universally from a given name or path.

        name_or_path can be a local diffusers directory, a hub repo id, a local
        single .safetensors file, or a remote single file
        ("org/repo/file.safetensors").

        When name_or_path is a standard hub repo this class has comfy weight
        candidates registered for (aitk_comfy_weight_names), the comfy file is
        the preferred source: the best-ranked local candidate, or the
        top-preference candidate downloaded into the comfy layout under
        MODELS_PATH. The standard repo still supplies the config. Local dirs
        and unregistered repos load as-is; use_comfy_weights=False opts out.

        qtype: quantize the weights after loading. quantize_device: where to run the
        quantization math; blocks are moved there one at a time and returned to where
        they were.
        config_path: config source for single-file loads, overriding aitk_config_repo.
        device: move the finished model there before returning.
        subfolder: overrides the class's aitk_subfolder; pass "" to explicitly
        load from the checkpoint root (e.g. a raw hub repo).
        """
        if subfolder is None:
            subfolder = cls.aitk_subfolder
        elif subfolder == "":
            subfolder = None

        if (
            use_comfy_weights
            and not name_or_path.endswith(".safetensors")
            and not os.path.exists(name_or_path)
        ):
            comfy_path = cls.resolve_comfy_weights(
                name_or_path, subfolder=subfolder, qtype=qtype
            )
            if comfy_path is not None:
                if config_path is None:
                    # the standard repo supplies the config for the comfy file
                    config_path = name_or_path
                name_or_path = comfy_path

        if name_or_path.endswith(".safetensors"):
            file_path = cls._resolve_single_file(name_or_path)
            model = cls._load_single_file(
                file_path,
                dtype=dtype,
                config_path=config_path,
                config=config,
                subfolder=subfolder,
                **kwargs,
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

        # quantize_on_load=False: qtype was only a comfy-candidate ranking
        # hint (the .load() path quantizes in aitk_post_load instead)
        if qtype is not None and quantize_on_load:
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

    @staticmethod
    def _readahead(file_path: str):
        """Queue kernel readahead for the whole file. safetensors tensors are
        mmap-backed; without this a cold cache faults pages in per-tensor
        cudaMemcpy order at a fraction of the drive's sequential bandwidth."""
        if not hasattr(os, "posix_fadvise"):
            return  # non-POSIX: mmap readahead heuristics only
        try:
            fd = os.open(file_path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            finally:
                os.close(fd)
        except OSError:
            pass

    @classmethod
    def _load_single_file(
        cls,
        file_path: str,
        dtype: torch.dtype,
        config_path: Optional[str] = None,
        config=None,
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        cls._readahead(file_path)
        state_dict = load_file(file_path)
        return cls.load_from_state_dict(
            state_dict,
            dtype,
            config_path=config_path,
            config=config,
            subfolder=subfolder,
            **kwargs,
        )

    @classmethod
    def aitk_config_from_state_dict(cls, state_dict: Dict[str, torch.Tensor]):
        """Derive the model config from the checkpoint itself (e.g. sniffing
        block counts from key indices). Return None (the default) to load the
        config from config_path / aitk_config_repo instead."""
        return None

    @classmethod
    def load_from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        dtype: torch.dtype,
        config_path: Optional[str] = None,
        config=None,
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        """Build the model and load an already-read single-file state dict
        (the tail of the single-file path; also callable directly for
        checkpoints read from non-safetensors sources). ``config``, when
        given, is used directly (for models whose config comes from the
        holder, e.g. model_kwargs-driven archs)."""
        state_dict = cls.convert_state_dict_on_load(state_dict)
        has_quant_markers = "scaled_fp8" in state_dict or any(
            k.endswith(".comfy_quant") for k in state_dict
        )
        if config is None:
            config = cls.aitk_config_from_state_dict(state_dict)
        if config is None:
            config = cls._load_single_file_config(config_path, subfolder)
        model = cls.aitk_from_config(config)

        if has_quant_markers:
            # pre-quantized comfy checkpoint: attach the quantized linears onto
            # the toolkit's backends, load the rest at its stored precision
            # (the checkpoint's bf16/fp16/fp32 mix is deliberate)
            from toolkit.util.comfy_quant_import import import_comfy_quantized_layers

            state_dict, num_quantized = import_comfy_quantized_layers(
                model, state_dict, orig_dtype=dtype
            )
            if cls.aitk_cast_quantized_load:
                # uniform-dtype models: the leftover (non-quantized) float
                # tensors follow the load dtype, like the non-marker path
                for key, value in state_dict.items():
                    if value.is_floating_point():
                        state_dict[key] = value.to(dtype=dtype)
            cls._load_state_dict_with_quantized(model, state_dict)
            if cls.aitk_cast_quantized_load:
                # the importer assigns quantized-layer biases directly at
                # stored dtype; they must follow too (a stray fp16 bias also
                # makes ModelMixin.dtype — the pipelines' cast target — lie)
                from toolkit.util.ostris_quant import OstrisLinear

                for m in model.modules():
                    if isinstance(m, OstrisLinear) and m.bias is not None:
                        m.bias.data = m.bias.data.to(dtype=dtype)
            model.aitk_is_quantized = True
        elif cls.aitk_cast_on_load:
            for key, value in state_dict.items():
                if value.is_floating_point():
                    state_dict[key] = value.to(dtype=dtype)
            model.load_state_dict(state_dict, assign=True)
            model.to(dtype=dtype)
        else:
            # stored-precision load (the checkpoint's dtype mix is deliberate)
            model.load_state_dict(state_dict, assign=True)
        del state_dict
        flush()
        return model

    @staticmethod
    def _load_state_dict_with_quantized(model, state_dict):
        """Load a state dict onto a (meta-built) model whose quantized linears
        were already attached by import_comfy_quantized_layers: their weight
        (and importer-assigned bias) legitimately report as missing keys."""
        from toolkit.util.ostris_quant import OstrisLinear

        result = model.load_state_dict(state_dict, assign=True, strict=False)
        quantized_param_keys = set()
        for name, m in model.named_modules():
            if isinstance(m, OstrisLinear):
                quantized_param_keys.add(f"{name}.weight")
                if m.bias is not None:
                    quantized_param_keys.add(f"{name}.bias")
        bad_missing = [k for k in result.missing_keys if k not in quantized_param_keys]
        if bad_missing or result.unexpected_keys:
            raise ValueError(
                f"{type(model).__name__} load mismatch: missing {bad_missing[:8]}, "
                f"unexpected {result.unexpected_keys[:8]}"
            )
        leftover_meta = [n for n, p in model.named_parameters() if p.is_meta]
        if leftover_meta:
            raise ValueError(
                f"{type(model).__name__} load left meta parameters: "
                f"{leftover_meta[:8]}"
            )

    # ------------------------------------------------------------------
    # comfy weight sources
    # ------------------------------------------------------------------

    @classmethod
    def resolve_comfy_weights(
        cls,
        name_or_path: str,
        subfolder: Optional[str] = None,
        local_only: bool = False,
        hf_token: Optional[str] = None,
        status_fn: Optional[callable] = None,
        qtype: Optional[str] = None,
    ) -> Optional[str]:
        """The comfy-format weight file replacing a standard ``name_or_path``,
        or None when this class has none registered for it. Best-ranked local
        candidate wins; otherwise the top-preference candidate is downloaded
        to the comfy layout under MODELS_PATH (unless local_only).

        Candidate keys may be plain repo ids or ``(repo_id, subfolder)``
        tuples for checkpoints holding several of this component (e.g.
        wan2.2's transformer / transformer_2)."""
        from toolkit.models.v2.resolver import resolve_comfy_candidates

        candidates = None
        if subfolder is not None:
            candidates = cls.aitk_comfy_weight_names.get((name_or_path, subfolder))
        if candidates is None:
            candidates = cls.aitk_comfy_weight_names.get(name_or_path)
        repo_id = cls.aitk_comfy_repo
        if isinstance(candidates, dict):
            # entries may carry their own comfy repo ({"repo": ..., "files": [...]})
            repo_id = candidates.get("repo", repo_id)
            candidates = candidates.get("files")
        if not candidates or repo_id is None:
            return None
        return resolve_comfy_candidates(
            candidates,
            repo_id=repo_id,
            hf_token=hf_token,
            status_fn=status_fn,
            local_only=local_only,
            qtype=qtype,
        )

    # ------------------------------------------------------------------
    # tokenizer / processor (text-encoder modules)
    # ------------------------------------------------------------------

    @classmethod
    def load_tokenizer(
        cls,
        name_or_path: Optional[str] = None,
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        """Load the tokenizer from ``name_or_path`` (a checkpoint dir or repo
        holding it at aitk_tokenizer_subfolder), falling back to the class's
        aitk_tokenizer_repo. subfolder overrides the class default; "" loads
        from the checkpoint root."""
        from transformers import AutoTokenizer

        source = name_or_path if name_or_path is not None else cls.aitk_tokenizer_repo
        if source is None:
            raise ValueError(f"{cls.__name__} does not declare aitk_tokenizer_repo")
        if subfolder is None:
            subfolder = cls.aitk_tokenizer_subfolder
        if subfolder and os.path.isdir(source) and not os.path.isdir(
            os.path.join(source, subfolder)
        ):
            subfolder = None
        return cls._local_first(
            AutoTokenizer.from_pretrained, source, subfolder=subfolder or "", **kwargs
        )

    @classmethod
    def load_processor(
        cls,
        name_or_path: Optional[str] = None,
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        from transformers import AutoProcessor

        source = name_or_path if name_or_path is not None else cls.aitk_processor_repo
        if source is None:
            raise ValueError(f"{cls.__name__} does not declare aitk_processor_repo")
        if subfolder is None:
            subfolder = cls.aitk_processor_subfolder
        if subfolder and os.path.isdir(source) and not os.path.isdir(
            os.path.join(source, subfolder)
        ):
            subfolder = None
        return cls._local_first(
            AutoProcessor.from_pretrained, source, subfolder=subfolder or "", **kwargs
        )

    # ------------------------------------------------------------------
    # saving
    # ------------------------------------------------------------------

    @torch.no_grad()
    def save_model(
        self,
        output_path: str,
        dtype: Optional[torch.dtype] = None,
        metadata: Optional[Dict[str, str]] = None,
        quantized: Optional[bool] = None,
    ):
        """Save as a single-file .safetensors in the model's original (comfy)
        key layout, via convert_state_dict_on_save.

        quantized: None (auto) keeps quantized storage — comfy_quant markers,
        loadable by ComfyUI and by this class — when the model holds quantized
        layers whose backend has a comfy format (convrot8 / nvfp4 /
        convrotcomfyw4a4), and saves dequantized full precision otherwise.
        True forces quantized storage (raises if any quantized layer's backend
        has no comfy format); False forces a dequantized save. dtype, when
        given, casts the floating point (non-quantized) tensors."""
        from safetensors.torch import save_file
        from toolkit.util.quantize import dequantize_if_quantized

        q_entries: Dict[str, torch.Tensor] = {}
        exported: List[str] = []
        if quantized is None or quantized:
            from toolkit.util.comfy_quant_export import export_comfy_quantized_layers

            q_entries, exported, unexportable = export_comfy_quantized_layers(self)
            if unexportable:
                if quantized:
                    raise ValueError(
                        f"{type(self).__name__}: quantized layers without a comfy "
                        f"storage format: {unexportable[:8]}"
                    )
                # auto mode: a partially-quantized file would be inconsistent —
                # fall back to a fully dequantized save
                q_entries, exported = {}, []

        skip_keys = {f"{name}.weight" for name in exported}
        state_dict = {}
        for key, value in self.state_dict().items():
            if key in skip_keys:
                continue
            value = dequantize_if_quantized(value)
            if dtype is not None and value.is_floating_point():
                value = value.to(dtype=dtype)
            state_dict[key] = value.detach().to("cpu").contiguous()
        state_dict.update(q_entries)
        state_dict = self.convert_state_dict_on_save(state_dict)

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        save_file(state_dict, output_path, metadata=metadata)
        del state_dict
        flush()

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
        blocks (get_transformer_block_names) are moved there one at a time for the
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
        for name in self.get_transformer_block_names() or []:
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


class OstrisTransformersMixin(OstrisModelMixin):
    """OstrisModelMixin with the backend hooks speaking the transformers-lib
    API (PreTrainedModel / AutoConfig) instead of diffusers ModelMixin. Base
    for text-encoder and vision-encoder modules."""

    @classmethod
    def aitk_from_pretrained(cls, path, subfolder=None, dtype=None, **kwargs):
        return cls._local_first(
            cls.from_pretrained, path, subfolder=subfolder or "", torch_dtype=dtype, **kwargs
        )

    @classmethod
    def aitk_load_config(cls, path, subfolder=None):
        from transformers import AutoConfig

        return cls._local_first(
            AutoConfig.from_pretrained, path, subfolder=subfolder or ""
        )

    @classmethod
    def aitk_from_config(cls, config):
        from accelerate import init_empty_weights

        with init_empty_weights(include_buffers=False):
            return cls(config)
