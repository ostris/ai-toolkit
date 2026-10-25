"""Ming-Image conditioning stack: the multimodal LLM plus the projections that
turn its hidden states into the two caption streams the DiT reads.

The checkpoint spreads this over three folders; one module holds it all so the
toolkit can quantize, offload and unload it as a single text encoder:

  mllm/       Ling-mini-2.0 MoE decoder (`llm`) + Qwen2.5-VL vision tower
              (`vision`) + its projector (`linear_proj`)
  mlp/        the 256 learnable query tokens, `proj_in` / `proj_out` around the
              connector, and `proj_directvlm`
  connector/  a Qwen2-1.5B run bidirectionally over the query-token states

Encoding a prompt:

  system + human turn (+ reference block) + assistant prefix
    + `<image>` + 256 x `<imagePatch>` + `</image>`
    -> LLM (query slots carry the learnable tokens, routed via image_gate)
    -> final states at the 256 query slots
         -> proj_in -> connector -> proj_out          = query condition (256, 2560)
    -> residual stream entering layers 5 and 12 plus the final normalized
       output, concatenated, at every token BEFORE the query block
         -> RMSNorm -> proj_directvlm                  = direct condition (T, 3840)

A reference image (editing) is a `<image>...</image>` block ahead of the
prompt text whose slots carry vision-tower features. Its tokens are part of the
direct condition; the DiT also gets its VAE latent as a reference frame.

The ComfyUI repack folds the same pieces, minus the vision tower, into one
file (`convert_state_dict_on_load`); the tower then comes from the vendor
checkpoint's shards.
"""

import json
import os
import random
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model, Qwen2VLImageProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)

from toolkit.models.v2._mixin import OstrisTransformersMixin
from toolkit.util.comfy_quant_import import parse_comfy_quant_blob

from .bailing_moe_v2 import (
    BailingMoeV2Config,
    BailingMoeV2Model,
    FusedExperts,
    FusedExpertsMemoryManager,
    RMSNorm,
    get_video_rope_index,
)
from .checkpoints import BASE_REPO, COMFY_REPO, COMFY_TEXT_ENCODER_FILES, comfy_weight_names

IMAGE_START = "<image>"
IMAGE_PATCH = "<imagePatch>"
IMAGE_END = "</image>"

# the vendor processor's template (processing_bailingmm2.py), NOT the
# tokenizer's own chat template: that one drops the assistant persona line
SYSTEM_PROMPT = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
HUMAN_PREFIX = "<role>HUMAN</role>"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"

# subfolders of the checkpoint this module is assembled from
MLLM_DIR = "mllm"
MLP_DIR = "mlp"
CONNECTOR_DIR = "connector"

_VISION_CONFIG_KEYS = (
    "depth",
    "hidden_size",
    "hidden_act",
    "intermediate_size",
    "num_heads",
    "in_channels",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "tokens_per_second",
    "window_size",
    "out_hidden_size",
    "fullatt_block_indexes",
)


# the checkpoint's configs, tokenizer and image processor
CONFIG_PATTERNS = [f"{MLLM_DIR}/*.json", f"{MLP_DIR}/config.json", f"{CONNECTOR_DIR}/config.json"]


def resolve_checkpoint_root(name_or_path: str, weights: bool = True) -> str:
    """The directory holding mllm/, mlp/ and connector/: a local checkpoint
    root (or one of its component folders), or the hub snapshot with just
    those folders fetched. `weights=False` fetches only the configs, tokenizer
    and image processor (a ComfyUI single file supplies the weights)."""
    if os.path.isdir(name_or_path):
        root = os.path.abspath(name_or_path)
        if os.path.basename(root) in (MLLM_DIR, MLP_DIR, CONNECTOR_DIR):
            root = os.path.dirname(root)
        return root
    patterns = [f"{d}/*" for d in (MLLM_DIR, MLP_DIR, CONNECTOR_DIR)] if weights else CONFIG_PATTERNS
    return snapshot_download(name_or_path, allow_patterns=patterns)


def _safetensors_files(folder: str) -> List[str]:
    files = sorted(f for f in os.listdir(folder) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files under {folder}")
    return [os.path.join(folder, f) for f in files]


class MingImageTextEncoder(nn.Module, OstrisTransformersMixin):
    aitk_subfolder = MLLM_DIR
    aitk_config_repo = BASE_REPO
    aitk_comfy_repo = COMFY_REPO
    aitk_comfy_weight_names = comfy_weight_names(COMFY_TEXT_ENCODER_FILES)

    def __init__(
        self,
        llm_config: BailingMoeV2Config,
        vision_config: Qwen2_5_VLVisionConfig,
        connector_config: Qwen2Config,
        mlp_config: dict,
    ):
        super().__init__()
        hidden = llm_config.hidden_size
        self.llm_config = llm_config
        self.mlp_config = dict(mlp_config)
        scales = list(mlp_config.get("img_gen_scales", [16]))
        if scales != [16]:
            raise ValueError(f"only img_gen_scales=[16] is supported, got {scales}")
        self.num_query_tokens = 16 * 16
        self.selected_layers = tuple(int(i) for i in mlp_config["selected_hidden_states_layers"])
        if not mlp_config.get("use_identity_mlp", False):
            raise NotImplementedError("Ming-Image with a non-identity diffusion MLP is not supported")
        # L2-normalize the connector output / rescale like the original text encoder
        self.normalize_query = bool(mlp_config.get("connector_norm", False))
        self.text_encoder_norm = bool(mlp_config.get("text_encoder_norm", False))

        self.llm = BailingMoeV2Model(llm_config)
        self.vision = Qwen2_5_VisionTransformerPretrainedModel(vision_config)
        self.linear_proj = nn.Sequential(
            nn.Linear(vision_config.out_hidden_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.query_tokens = nn.Parameter(torch.empty(self.num_query_tokens, hidden))

        self.proj_in = nn.Linear(hidden, connector_config.hidden_size)
        self.connector = Qwen2Model(connector_config)
        # the connector only ever sees inputs_embeds (the projected query
        # states): its token embedding is 467 MB of dead weight, and the
        # ComfyUI repack omits it
        self.connector.embed_tokens = None
        self.proj_out = nn.Linear(connector_config.hidden_size, int(mlp_config["diffusion_c_input_dim"]))

        direct_dim = hidden * len(self.selected_layers)
        self.proj_directvlm = nn.Sequential(
            RMSNorm(direct_dim, eps=1e-5),
            nn.Linear(direct_dim, int(mlp_config["diffusion_inner_dim"])),
        )

    # ------------------------------------------------------------------
    # toolkit hooks
    # ------------------------------------------------------------------
    @property
    def device(self):
        return self.query_tokens.device

    @property
    def dtype(self):
        return self.query_tokens.dtype

    @classmethod
    def get_transformer_block_names(cls):
        return ["llm.layers", "connector.layers", "vision.blocks"]

    def _apply(self, fn, *args, **kwargs):
        # a device move or dtype cast invalidates captured CUDA graphs (weight
        # addresses are baked in); the prompt encoder registers its caches
        # here so they release their device memory before the move
        for callback in getattr(self, "_on_apply", ()):
            callback()
        return super()._apply(fn, *args, **kwargs)

    @classmethod
    def get_quantization_exclude_modules(cls):
        return ["query_tokens*", "proj_in*", "proj_out*", "proj_directvlm*", "linear_proj*", "llm.word_embeddings*"]

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    @classmethod
    def build_configs(cls, root: str):
        with open(os.path.join(root, MLLM_DIR, "config.json")) as f:
            mllm = json.load(f)
        with open(os.path.join(root, MLP_DIR, "config.json")) as f:
            mlp = json.load(f)
        llm_config = BailingMoeV2Config(**mllm["llm_config"])
        vision_dict = {k: v for k, v in mllm["vision_config"].items() if k in _VISION_CONFIG_KEYS}
        vision_config = Qwen2_5_VLVisionConfig(**vision_dict)
        vision_config._attn_implementation = "sdpa"
        connector_config = Qwen2Config.from_pretrained(os.path.join(root, CONNECTOR_DIR))
        connector_config._attn_implementation = "sdpa"
        connector_config.use_cache = False
        return llm_config, vision_config, connector_config, mlp

    @classmethod
    def aitk_from_pretrained(cls, path, subfolder=None, dtype=None, **kwargs):
        dtype = dtype or torch.bfloat16
        root = resolve_checkpoint_root(path)
        llm_config, vision_config, connector_config, mlp = cls.build_configs(root)
        with init_empty_weights(include_buffers=False):
            model = cls(llm_config, vision_config, connector_config, mlp)

        state = {}

        def read(folder: str, rename):
            for file in _safetensors_files(folder):
                with safe_open(file, framework="pt") as f:
                    for key in f.keys():
                        new_key = rename(key)
                        if new_key is None:
                            continue
                        value = f.get_tensor(key)
                        if value.is_floating_point():
                            value = value.to(dtype)
                        state[new_key] = value

        def rename_mllm(key: str):
            if key.startswith("model.model."):
                key = "llm." + key[len("model.model."):]
                return None if ".mlp.audio_gate." in key else key
            if key.startswith("vision.") or key.startswith("linear_proj."):
                return key
            return None  # lm_head, audio pieces

        def rename_mlp(key: str):
            if key == "query_tokens_dict.16x16":
                return "query_tokens"
            if key.startswith(("proj_in.", "proj_out.", "proj_directvlm.")):
                return key
            return None

        def rename_connector(key: str):
            if not key.startswith("model.") or key.startswith("model.embed_tokens."):
                return None
            return "connector." + key[len("model."):]

        read(os.path.join(root, MLLM_DIR), rename_mllm)
        read(os.path.join(root, MLP_DIR), rename_mlp)
        read(os.path.join(root, CONNECTOR_DIR), rename_connector)

        # the checkpoint stores one linear per expert; stack them into the
        # grouped-GEMM banks (the per-expert tensors are dropped as they go)
        for i, layer in enumerate(model.llm.layers):
            if layer.is_moe:
                FusedExperts.fuse_state_dict(state, f"llm.layers.{i}.mlp.experts.", llm_config.num_experts)

        result = model.load_state_dict(state, assign=True, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"MingImageTextEncoder load mismatch: missing {result.missing_keys[:8]}, "
                f"unexpected {result.unexpected_keys[:8]}"
            )
        leftover = [n for n, p in model.named_parameters() if p.is_meta]
        if leftover:
            raise ValueError(f"MingImageTextEncoder load left meta parameters: {leftover[:8]}")
        model.to(dtype=dtype)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # single-file (ComfyUI repack) loading
    # ------------------------------------------------------------------
    @classmethod
    def aitk_load_config(cls, path, subfolder=None):
        return cls.build_configs(resolve_checkpoint_root(path, weights=False))

    @classmethod
    def aitk_from_config(cls, config):
        with init_empty_weights(include_buffers=False):
            return cls(*config)

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        """The ComfyUI repack's keys -> this module's. The LLM sits under
        `thinker.` with transformers-style names and fused expert banks in our
        `(E, 2*inter, hidden)` / `(E, hidden, inter)` layout; the connector,
        projections and query tokens already match. Quantized expert banks
        (int8 rows, float32 scales, optionally convrot-rotated) become the
        `*_q` / `*_scale` entries FusedExperts attaches on load; the toolkit's
        importer covers the nn.Linear markers."""
        if not any(k.startswith("thinker.") for k in state_dict):
            return state_dict
        renames = (
            (".mlp.gate.proj.", ".mlp.gate."),
            (".mlp.image_gate.proj.", ".mlp.image_gate."),
        )
        out = {}
        banks = {}
        for key, value in state_dict.items():
            if key == "tokenizer_json" or key.startswith("thinker.lm_head."):
                continue  # tokenizer comes from the vendor repo; no lm head here
            if not key.startswith("thinker."):
                out[key] = value
                continue
            key = "llm." + key[len("thinker."):]
            if key.startswith("llm.embed_tokens."):
                key = "llm.word_embeddings." + key[len("llm.embed_tokens."):]
            for old, new in renames:
                key = key.replace(old, new)
            if ".mlp.experts." in key:
                prefix, _, rest = key.partition(".mlp.experts.")
                bank, _, field = rest.partition(".")  # gate_up_proj.<field>
                banks.setdefault(prefix + ".mlp.experts.", {})[(bank[: -len("_proj")], field)] = value
                continue
            out[key] = value
        for prefix, entries in banks.items():
            if not any(field == "comfy_quant" for _, field in entries):
                for (bank, field), value in entries.items():
                    if field != "weight":
                        raise ValueError(f"{prefix}{bank}: unexpected entry {field!r}")
                    out[f"{prefix}{bank}"] = value
                continue
            rot = None
            for bank in ("gate_up", "down"):
                conf = parse_comfy_quant_blob(entries[(bank, "comfy_quant")])
                if conf.get("format") != "int8_tensorwise":
                    raise ValueError(
                        f"{prefix}{bank}: unsupported expert quantization {conf.get('format')!r}"
                    )
                bank_rot = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
                if rot is not None and bank_rot != rot:
                    raise ValueError(f"{prefix}: the two expert banks use different rotations")
                rot = bank_rot
                out[f"{prefix}{bank}_q"] = entries[(bank, "weight")]
                out[f"{prefix}{bank}_scale"] = entries[(bank, "weight_scale")]
            out[f"{prefix}rot_size"] = torch.tensor(rot)
        return out

    @classmethod
    def load_from_state_dict(
        cls, state_dict, dtype, config_path=None, config=None, subfolder=None, **kwargs
    ):
        state_dict = cls.convert_state_dict_on_load(state_dict)
        if not any(k.startswith("vision.") for k in state_dict):
            # a repack without the vision tower (the first uploads); editing
            # needs one, so it comes from the vendor repo
            state_dict = dict(state_dict)
            state_dict.update(cls.read_vision_weights(config_path or cls.aitk_config_repo, dtype))
        return super().load_from_state_dict(
            state_dict, dtype, config_path=config_path, config=config, subfolder=subfolder, **kwargs
        )

    @classmethod
    def read_vision_weights(cls, name_or_path: str, dtype) -> Dict[str, torch.Tensor]:
        """`vision.*` and `linear_proj.*` from the vendor checkpoint's mllm
        shards: only the shards holding those keys are fetched, and only
        those tensors are read."""

        def wanted(key: str) -> bool:
            return key.startswith(("vision.", "linear_proj."))

        root = resolve_checkpoint_root(name_or_path, weights=False)
        folder = os.path.join(root, MLLM_DIR)
        index_path = os.path.join(folder, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shards = sorted({v for k, v in weight_map.items() if wanted(k)})
        else:
            shards = [os.path.basename(f) for f in _safetensors_files(folder)]
        state = {}
        for shard in shards:
            path = os.path.join(folder, shard)
            if not os.path.isfile(path):
                if os.path.isdir(name_or_path):
                    raise FileNotFoundError(f"{path} is missing")
                path = hf_hub_download(name_or_path, f"{MLLM_DIR}/{shard}")
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    if wanted(key):
                        state[key] = f.get_tensor(key).to(dtype)
        if not state:
            raise ValueError(f"no vision tower weights under {folder}")
        return state

    def aitk_post_load(
        self, qtype=None, offload=0.0, quantize_device=None, device=None, dtype=torch.bfloat16, **kwargs
    ):
        # the expert banks are not nn.Linear, so the toolkit's quantizer,
        # dequantizer and layer offloader never see them. They follow the
        # request in kind before the regular pass handles the attention /
        # dense / shared-expert linears with `qtype`: any quantize_te makes
        # them int8 weight-only (ready-made comfy int8 banks stay as they are),
        # none restores full precision. With layer offloading on, the banks get
        # their own stager (pinned cpu, copied in per forward) so the offloader
        # leaves them where they are instead of making them resident.
        q_device = quantize_device if quantize_device is not None else device
        banks = [layer.mlp.experts for layer in self.llm.layers if layer.is_moe]
        for bank in banks:
            if qtype:
                bank.quantize_int8_(device=q_device)
            else:
                bank.dequantize_(dtype=dtype, device=q_device)
        if offload and offload > 0 and q_device is not None:
            for bank in banks:
                if random.random() <= offload:
                    FusedExpertsMemoryManager.attach(bank, torch.device(q_device))
        return super().aitk_post_load(
            qtype=qtype, offload=offload, quantize_device=quantize_device, device=device, dtype=dtype, **kwargs
        )

    @classmethod
    def load_tokenizer_and_processor(cls, name_or_path: str):
        root = resolve_checkpoint_root(name_or_path, weights=False)
        folder = os.path.join(root, MLLM_DIR)
        tokenizer = AutoTokenizer.from_pretrained(folder)
        image_processor = Qwen2VLImageProcessor.from_pretrained(folder)
        return tokenizer, image_processor

    # ------------------------------------------------------------------
    # forward pieces
    # ------------------------------------------------------------------
    def encode_reference_images(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Vision-tower features for the reference blocks, `(N, hidden)` in
        token order across all images."""
        features = self.vision(pixel_values.to(self.dtype), grid_thw=grid_thw).pooler_output
        return F.normalize(self.linear_proj(features), dim=-1)

    def encode_query(self, query_states: torch.Tensor) -> torch.Tensor:
        """`(B, 256, hidden)` final LLM states at the query slots -> `(B, 256, c_dim)`."""
        x = self.proj_in(query_states)
        bsz, n, _ = x.shape
        # the connector is a decoder run as a bidirectional encoder
        mask = torch.ones((bsz, 1, n, n), dtype=torch.bool, device=x.device)
        hidden = self.connector(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        out = self.proj_out(hidden)
        if self.normalize_query:
            out = F.normalize(out, dim=-1)
        if self.text_encoder_norm:
            out = out * 1000.0
        return out

    def encode_direct(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`(T, hidden * n_selected)` concatenated layer states -> `(T, dim)`."""
        return self.proj_directvlm(hidden_states)


# prompts are padded up to a multiple of this many tokens so a few captured
# graphs cover every length; longer sequences run eagerly
GRAPH_LENGTH_STEP = 64
GRAPH_MAX_LENGTH = 2048
GRAPH_MAX_ENTRIES = 8


class CudaGraphCache:
    """Replays a function through CUDA graphs, one capture per static-shape key.

    Encoding a prompt is ~3k small launches on ~30 ms of GPU work, so at
    batch 1 the encoder is CPU bound; a captured graph replays the whole
    stage in one call. Inputs are copied into the captured tensors before
    each replay; outputs are the captured tensors themselves (overwritten by
    the next replay, so callers copy what they keep). Captures are dropped
    when the module's weights move (their addresses are baked into the graph).
    """

    def __init__(self, fn: Callable, signature: Callable[[], tuple], max_entries: int = GRAPH_MAX_ENTRIES):
        self.fn = fn
        self.signature = signature
        self.max_entries = max_entries
        self.entries: "OrderedDict[tuple, tuple]" = OrderedDict()
        self.captured_signature = None

    def clear(self):
        self.entries.clear()
        self.captured_signature = None

    def __call__(self, key: tuple, inputs: Dict[str, torch.Tensor], **static_kwargs):
        signature = self.signature()
        if signature != self.captured_signature:
            self.clear()
            self.captured_signature = signature
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_entries:
                self.entries.popitem(last=False)
            static = {name: value.clone() for name, value in inputs.items()}
            stream = torch.cuda.Stream(device=next(iter(static.values())).device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # warm-up on a side stream: lazy allocations / kernel compiles
                # must not land inside the capture
                for _ in range(2):
                    self.fn(**static, **static_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self.fn(**static, **static_kwargs)
            entry = (graph, static, outputs)
            self.entries[key] = entry
        else:
            self.entries.move_to_end(key)
        graph, static, outputs = entry
        for name, value in inputs.items():
            static[name].copy_(value)
        graph.replay()
        return outputs


class MingImagePromptEncoder:
    """Prompts (+ optional single reference image each) -> the DiT's two
    caption streams, one entry per prompt.

    On a resident GPU text encoder the LLM (per padded-length bucket) and the
    connector (fixed 256 tokens) replay as CUDA graphs; set `use_cuda_graphs`
    False for eager execution. Offloaded or cpu encoders always run eagerly.
    """

    use_cuda_graphs = True

    def __init__(self, text_encoder: MingImageTextEncoder, tokenizer, image_processor):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.merge_size = int(getattr(image_processor, "merge_size", 2))
        self.patch_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PATCH)
        self.start_token_id = tokenizer.convert_tokens_to_ids(IMAGE_START)
        self.end_token_id = tokenizer.convert_tokens_to_ids(IMAGE_END)
        self.pad_token_id = tokenizer.pad_token_id
        config = text_encoder.llm_config
        te = text_encoder
        self._llm_graphs = CudaGraphCache(te.llm, self._weight_signature)
        self._connector_graphs = CudaGraphCache(te.encode_query, self._weight_signature)
        te._on_apply = [self._llm_graphs.clear, self._connector_graphs.clear]
        if (self.patch_token_id, self.start_token_id, self.end_token_id) != (
            config.image_patch_token,
            config.image_start_token,
            config.image_end_token,
        ):
            raise ValueError("tokenizer image tokens do not match the LLM config")

    def _query_block(self) -> str:
        return IMAGE_START + IMAGE_PATCH * self.text_encoder.num_query_tokens + IMAGE_END

    def _chat_text(self, content: str) -> str:
        eos = self.tokenizer.eos_token
        return SYSTEM_PROMPT + eos + HUMAN_PREFIX + content + eos + ASSISTANT_PREFIX

    def _weight_signature(self) -> tuple:
        # a handful of storages stands in for "the weights moved": any device
        # move relocates all of them
        te = self.text_encoder
        probes = [te.query_tokens, te.proj_out.weight]
        for module in (te.llm.layers[0], te.llm.layers[-1], te.connector.layers[0], te.connector.layers[-1]):
            probe = next(module.parameters(), None)
            probes.append(probe if probe is not None else next(module.buffers()))
        return tuple(int(t.data_ptr()) for t in probes)

    def _graphs_enabled(self) -> bool:
        te = self.text_encoder
        if not self.use_cuda_graphs or te.device.type != "cuda":
            return False
        # streamed weights change addresses every forward
        return not hasattr(te, "_memory_manager") and not any(
            hasattr(m, "_layer_memory_manager") for m in te.modules()
        )

    @torch.no_grad()
    def encode(
        self,
        prompts: Sequence[str],
        images: Optional[Sequence[Sequence[Image.Image]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[bool]]:
        te = self.text_encoder
        device, dtype = te.device, te.dtype
        images = list(images) if images is not None else [[] for _ in prompts]

        texts, pixel_values, grids, ref_counts = [], [], [], []
        for prompt, sample_images in zip(prompts, images):
            if len(sample_images) > 1:
                raise ValueError("Ming-Image takes at most one reference image per prompt")
            blocks = ""
            for image in sample_images:
                feats = self.image_processor(images=[image], return_tensors="pt")
                grid = feats["image_grid_thw"][0]
                n_tokens = int(grid.prod()) // (self.merge_size**2)
                blocks += IMAGE_START + IMAGE_PATCH * n_tokens + IMAGE_END + "\n"
                pixel_values.append(feats["pixel_values"])
                grids.append(grid)
            ref_counts.append(len(sample_images))
            texts.append(self._chat_text(blocks + prompt) + self._query_block())

        # the template supplies every special token; right padding keeps the
        # query block at the end of each valid sequence
        enc = self.tokenizer(
            texts, add_special_tokens=False, padding=True, padding_side="right", return_tensors="pt"
        )
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask
        use_graphs = self._graphs_enabled()
        if use_graphs:
            # pad to the bucket length; the pad slots are masked keys whose
            # own outputs are never read, so the valid tokens are unaffected
            padded = -(-input_ids.shape[1] // GRAPH_LENGTH_STEP) * GRAPH_LENGTH_STEP
            if padded > GRAPH_MAX_LENGTH:
                use_graphs = False
            elif padded > input_ids.shape[1]:
                extra = padded - input_ids.shape[1]
                input_ids = F.pad(input_ids, (0, extra), value=self.pad_token_id)
                attention_mask = F.pad(attention_mask, (0, extra), value=0)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        bsz, seq_len = input_ids.shape

        embeds = te.llm.word_embeddings(input_ids)
        ref_feats = None
        if pixel_values:
            ref_feats = te.encode_reference_images(
                torch.cat(pixel_values, dim=0).to(device), torch.stack(grids).to(device)
            )

        # fill the slots: reference features first (in order), then the query tokens
        patch_mask = input_ids == self.patch_token_id
        query_tokens = te.query_tokens.to(dtype)
        ref_offset = 0
        image_grid_thw = []
        for b in range(bsz):
            n_ref = 0
            if ref_counts[b]:
                grid = grids[ref_offset]
                n_ref = int(grid.prod()) // (self.merge_size**2)
                image_grid_thw.append(grid.tolist())
            image_grid_thw.append([1, 2, self.text_encoder.num_query_tokens * 2])
            fill = query_tokens
            if n_ref:
                fill = torch.cat([ref_feats[:n_ref].to(dtype), fill], dim=0)
                ref_feats = ref_feats[n_ref:]
                ref_offset += 1
            positions = torch.nonzero(patch_mask[b], as_tuple=False).squeeze(1)
            if positions.numel() != fill.shape[0]:
                raise ValueError(
                    f"prompt {b} has {positions.numel()} image slots but {fill.shape[0]} features"
                )
            embeds[b, positions] = fill

        position_ids = get_video_rope_index(
            te.llm_config,
            input_ids,
            attention_mask,
            torch.tensor(image_grid_thw, dtype=torch.long),
        ).to(device)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        attn_mask = causal[None, None] & attention_mask.bool()[:, None, None, :]

        capture = tuple(sorted(set(te.selected_layers) | {len(te.llm.layers)}))
        llm_inputs = dict(
            inputs_embeds=embeds.to(dtype),
            attention_mask=attn_mask,
            position_ids=position_ids,
            image_mask=patch_mask,
        )
        if use_graphs:
            final, captured = self._llm_graphs((bsz, seq_len), llm_inputs, capture_layers=capture)
        else:
            final, captured = te.llm(**llm_inputs, capture_layers=capture)

        n_query = te.num_query_tokens
        query_states, direct_inputs, direct_lens = [], [], []
        for b in range(bsz):
            length = int(attention_mask[b].sum())
            tail = input_ids[b, length - n_query - 2 : length]
            if int(tail[0]) != self.start_token_id or int(tail[-1]) != self.end_token_id:
                raise ValueError("prompt does not end with the query block")
            query_states.append(final[b, length - n_query - 1 : length - 1])
            # the direct condition covers every token before the query block
            direct_end = length - n_query - 2
            direct_inputs.append(
                torch.cat([captured[i][b, :direct_end] for i in te.selected_layers], dim=-1)
            )
            direct_lens.append(direct_end)

        query_batch = torch.stack(query_states, dim=0)
        if use_graphs:
            query_cond = self._connector_graphs((bsz,), dict(query_states=query_batch))
        else:
            query_cond = te.encode_query(query_batch)
        direct_batch = torch.nn.utils.rnn.pad_sequence(direct_inputs, batch_first=True)
        direct_cond = te.encode_direct(direct_batch)

        # clones: graph outputs are overwritten by the next replay
        query_out = [query_cond[b].clone() for b in range(bsz)]
        direct_out = [direct_cond[b, : direct_lens[b]].clone() for b in range(bsz)]
        has_reference = [count > 0 for count in ref_counts]
        return query_out, direct_out, has_reference
