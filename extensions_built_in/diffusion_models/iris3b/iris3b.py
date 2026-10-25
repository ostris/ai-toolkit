"""Iris3BModel -- Sperid Labs' Iris-3B pixel-space text-to-image DiT
(https://huggingface.co/speridlabs/iris-3b, arXiv:2610.09450) in ai-toolkit.

  - **Pixel space, no VAE.** The transformer denoises raw RGB (``in_channels=3``,
    ``patch_size=16``) through a 24-block hybrid trunk (8 dual-stream MM-DiT +
    16 single-stream) and a 4-block per-pixel PiT head. A ``FakeVAE`` (identity)
    makes the toolkit's "latents" the image in [-1, 1]; buckets snap to 16 px.
    Architecture vendored in ``src/dit.py``.
  - **v-prediction rectified flow, shift 4.** Velocity = noise - clean on
    ``x_t = (1 - t) * clean + t * noise`` -- ai-toolkit's own convention, so no
    flip/negation. The model is conditioned on *shifted* model time
    ``1000 * sigma'``; the toolkit's ``timestep_type: shift`` with the scheduler
    ``shift`` reproduces the reference training schedule and the toolkit
    ``timestep`` feeds the model as-is.
  - **Qwen3-VL-4B-Instruct text encoder, 12 stacked hidden layers.** Prompts go
    through the Iris chat template and are padded to a fixed 300 tokens
    (``src/text_encoder.py``); the DiT's layerwise-attention adapter pools the
    layer stack. Weights come from the ComfyUI repack
    (``Comfy-Org/Krea-2/text_encoders/qwen3vl_4b_bf16.safetensors``, the same
    file Krea 2 uses): a local copy under ``MODELS_PATH/text_encoders`` is used
    in place, otherwise it is downloaded there. ``model.te_name_or_path`` points
    at any other file, transformers folder or hub repo instead.
  - **Previews** use the reference FlowDPM-Solver++ (order 2) sampler
    (``src/pipeline.py``); 30 steps at CFG 3 is a good default (the paper uses
    100).

``name_or_path`` is the hub repo (``speridlabs/iris-3b``), a local export dir
(``model.safetensors`` + ``config.yaml``), or a ``.safetensors`` file. The
checkpoint's ``config.yaml`` (model / text_encoder / flow sections) drives the
architecture, the text layers and the flow shift; the Iris-3B defaults apply
when it is absent. Full fine-tunes save back in that same layout, so
``scripts/sample.py`` from the reference repo loads them directly.
"""

import os
from typing import List, Optional

import torch
import yaml
import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError

from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.models.FakeVAE import FakeVAE
from toolkit.models.v2.resolver import resolve_component_file
from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.dit import IrisConfig, IrisDiT
from .src.pipeline import Iris3BPipeline
from .src.text_encoder import (
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_TEXT_LEN,
    IrisPromptEncoder,
)

HF_TOKEN = os.getenv("HF_TOKEN", None)

# Reference flow settings (config.yaml ``flow`` section of the release).
DEFAULT_FLOW_SHIFT = 4.0
DEFAULT_NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_X_PRED_SIGMA_MIN = 0.05

# Qwen3-VL-4B-Instruct: config + tokenizer (tiny files) from the vendor repo,
# weights from the ComfyUI repack shared with Krea 2.
QWEN3_VL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_TE_PATH = "Comfy-Org/Krea-2/text_encoders/qwen3vl_4b_bf16.safetensors"


def _resolve_checkpoint(name_or_path: str, filename: Optional[str] = None):
    """``(weights_path, config_yaml_path_or_None)`` for a local file, a local
    export dir, a hub repo id, or ``org/repo/sub/file.safetensors``."""
    if os.path.isfile(name_or_path):
        cfg = os.path.join(os.path.dirname(name_or_path), "config.yaml")
        return name_or_path, (cfg if os.path.isfile(cfg) else None)
    if os.path.isdir(name_or_path):
        weights = os.path.join(name_or_path, filename or "model.safetensors")
        if not os.path.isfile(weights):
            candidates = [f for f in os.listdir(name_or_path) if f.endswith(".safetensors")]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Could not pick an Iris checkpoint in {name_or_path}: found {candidates}. "
                    "Set model.model_kwargs.checkpoint_filename."
                )
            weights = os.path.join(name_or_path, candidates[0])
        cfg = os.path.join(name_or_path, "config.yaml")
        return weights, (cfg if os.path.isfile(cfg) else None)

    parts = name_or_path.split("/")
    if name_or_path.endswith(".safetensors") and len(parts) >= 3:
        repo_id, rel = "/".join(parts[:2]), "/".join(parts[2:])
    elif len(parts) == 2:
        repo_id, rel = name_or_path, (filename or "model.safetensors")
    else:
        raise FileNotFoundError(
            f"Iris checkpoint {name_or_path!r} is not a local file/dir, a hub repo id, or "
            "'org/repo/path/file.safetensors'."
        )
    weights = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=rel, token=HF_TOKEN)
    cfg_rel = os.path.join(os.path.dirname(rel), "config.yaml") if os.path.dirname(rel) else "config.yaml"
    try:
        cfg = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=cfg_rel, token=HF_TOKEN)
    except EntryNotFoundError:
        cfg = None
    return weights, cfg


class Iris3BModel(BaseModel):
    arch = "iris3b"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        # LoRA targets, matched by class name: the trunk blocks only. The
        # transformer_only filter is a substring match on "blocks", which also
        # catches pixel_blocks (per-pixel PiT head) and y_embedder.*.blocks (text
        # adapter); scoping the target classes keeps those frozen.
        self.target_lora_modules = ["MMDiTBlock", "SingleStreamBlock"]

        self.patch_size = 16
        self.vae_scale_factor = 1  # pixel space
        # flow / text settings; refreshed from the checkpoint's config.yaml at load
        self.flow_shift = float(self.model_config.model_kwargs.get("flow_shift", DEFAULT_FLOW_SHIFT))
        self.num_train_timesteps = DEFAULT_NUM_TRAIN_TIMESTEPS
        self.prediction = "v"
        self.x_pred_sigma_min = DEFAULT_X_PRED_SIGMA_MIN
        self.text_len = DEFAULT_TEXT_LEN
        self.text_hidden_layers = DEFAULT_HIDDEN_LAYERS
        self.te_config_repo = QWEN3_VL_REPO
        self.iris_raw_config: dict = {}
        self.prompt_encoder: Optional[IrisPromptEncoder] = None

    @staticmethod
    def get_train_scheduler(
        shift: float = DEFAULT_FLOW_SHIFT,
        num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
    ):
        # called on the class by the trainer / generate process (reference
        # defaults); load_model rebuilds it with the checkpoint's own shift
        return CustomFlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=False,
        )

    def get_bucket_divisibility(self):
        return self.vae_scale_factor * self.patch_size

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _read_checkpoint_config(self, cfg_path: Optional[str]) -> IrisConfig:
        raw = {}
        if cfg_path is not None:
            with open(cfg_path, "r") as f:
                raw = yaml.safe_load(f) or {}
        self.iris_raw_config = raw

        flow = raw.get("flow", {}) or {}
        if "flow_shift" not in self.model_config.model_kwargs:
            self.flow_shift = float(flow.get("shift", DEFAULT_FLOW_SHIFT))
        self.num_train_timesteps = int(flow.get("num_train_timesteps", DEFAULT_NUM_TRAIN_TIMESTEPS))
        self.prediction = str(flow.get("prediction", "v"))
        if self.prediction not in ("v", "x"):
            raise ValueError(f"unknown flow.prediction '{self.prediction}' (v | x)")
        self.x_pred_sigma_min = float(flow.get("x_pred_sigma_min", DEFAULT_X_PRED_SIGMA_MIN))

        te = raw.get("text_encoder", {}) or {}
        self.text_len = int(te.get("max_length", DEFAULT_TEXT_LEN))
        layers = te.get("hidden_layers", None)
        self.text_hidden_layers = tuple(int(i) for i in layers) if layers else DEFAULT_HIDDEN_LAYERS
        self.te_config_repo = str(te.get("pretrained", QWEN3_VL_REPO))

        model_raw = dict(raw.get("model", {}) or {})
        model_raw.update(self.model_config.model_kwargs.get("model_config", {}) or {})
        config = IrisConfig.from_dict(model_raw)
        if config.text_adapter == "lap_blocks2" and config.text_lap_num_layers != len(self.text_hidden_layers):
            raise ValueError(
                f"model.text_lap_num_layers={config.text_lap_num_layers} but text_encoder.hidden_layers "
                f"has {len(self.text_hidden_layers)} entries"
            )
        if config.text_len != self.text_len:
            raise ValueError(
                f"model.text_len={config.text_len} but text_encoder.max_length={self.text_len}"
            )
        return config

    def _load_transformer(self) -> IrisDiT:
        self.print_and_status_update("Loading Iris-3B transformer")
        weights, cfg_path = _resolve_checkpoint(
            self.model_config.name_or_path,
            self.model_config.model_kwargs.get("checkpoint_filename", None),
        )
        config = self._read_checkpoint_config(cfg_path)
        self.patch_size = config.patch_size
        self.print_and_status_update(f"  - weights: {weights}")
        # fp32 release weights -> load dtype; quantize / offload / place per model_config
        transformer = IrisDiT.load(
            weights,
            config=config,
            **self.component_load_kwargs("transformer"),
        )
        flush()
        return transformer

    def _load_text_encoder(self):
        dtype = self.torch_dtype
        te_path = self.model_config.te_name_or_path or DEFAULT_TE_PATH
        tokenizer = Qwen3VLTextEncoder.load_tokenizer(self.te_config_repo, subfolder="", token=HF_TOKEN)

        if os.path.isdir(te_path) or not te_path.endswith(".safetensors"):
            # transformers folder or hub repo: load exactly what it names
            self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_path}")
            text_encoder = Qwen3VLTextEncoder.load_model(
                te_path, dtype=dtype, subfolder="", use_comfy_weights=False, token=HF_TOKEN
            )
        else:
            # single-file ComfyUI checkpoint: local copy in MODELS_PATH/text_encoders
            # wins, otherwise downloaded there; config from the vendor repo
            te_file = resolve_component_file(
                te_path,
                "text_encoders",
                component="Qwen3-VL text encoder",
                hf_token=HF_TOKEN,
                status_fn=self.print_and_status_update,
            )
            self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_file}")
            text_encoder = Qwen3VLTextEncoder.load_model(
                te_file,
                dtype=dtype,
                config_path=self.te_config_repo,
                subfolder="",
                use_comfy_weights=False,
            )
        # text-only conditioning: the vision tower is dead weight
        text_encoder.drop_vision_tower()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))
        flush()
        return tokenizer, text_encoder

    def load_model(self):
        transformer = self._load_transformer()
        tokenizer, text_encoder = self._load_text_encoder()

        self.print_and_status_update("Preparing pixel-space VAE (identity)")
        vae = FakeVAE(scaling_factor=1.0)
        vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)

        self.noise_scheduler = Iris3BModel.get_train_scheduler(self.flow_shift, self.num_train_timesteps)
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.prompt_encoder = IrisPromptEncoder(
            tokenizer, text_len=self.text_len, hidden_layers=self.text_hidden_layers
        )
        self.model = transformer
        self.pipeline = Iris3BPipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Sampling (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return Iris3BPipeline(self)

    def generate_single_image(
        self,
        pipeline: Iris3BPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        mkw = self.model_config.model_kwargs
        cfg_interval = mkw.get("cfg_interval", (0.0, 1.0))
        img = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            order=int(mkw.get("sample_order", 2)),
            cfg_interval=(float(cfg_interval[0]), float(cfg_interval[1])),
        )[0]
        return img

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def model_velocity(
        self,
        x_t: torch.Tensor,
        t_model: torch.Tensor,
        text_feats: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the DiT and return the flow velocity ``noise - clean``. ``t_model``
        is Iris model time in [0, 1000] (= the toolkit timestep). An
        x-prediction checkpoint is converted via ``(x_t - x0) / max(sigma, s_min)``."""
        out = self.model(x_t, t_model, text_feats, y_mask=text_mask)
        if self.prediction == "x":
            sigma = (t_model.float() / self.num_train_timesteps).view(-1, 1, 1, 1).to(out.dtype)
            out = (x_t - out) / sigma.clamp(min=self.x_pred_sigma_min)
        return out

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 3, H, W) pixels in [-1, 1] + noise
        timestep: torch.Tensor,  # 0..1000 scale, 1000 = pure noise
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # toolkit timestep IS Iris model time (1000 * noise level); with
        # timestep_type=shift that is the shifted sigma the reference trained on
        t_model = timestep.to(self.device_torch, dtype=torch.float32)
        if t_model.dim() == 0:
            t_model = t_model.unsqueeze(0)
        if t_model.shape[0] != latent_model_input.shape[0]:
            t_model = t_model.expand(latent_model_input.shape[0])

        feats = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        mask = getattr(text_embeddings, "attention_mask", None)
        if mask is None:
            raise ValueError("Iris prompt embeds must carry an attention_mask")
        mask = mask.to(self.device_torch)

        return self.model_velocity(
            latent_model_input.to(self.device_torch, self.torch_dtype), t_model, feats, mask
        )

    def get_prompt_embeds(self, prompt, **kwargs) -> PromptEmbeds:
        """Fixed-length (text_len) stacked Qwen3-VL layer features.

        ``text_embeds`` is ``(B, text_len, num_layers * 2560)`` -- the layer axis
        is flattened so cached embeds batch/concat like ordinary (B, L, D)
        embeddings; the DiT restores it. ``attention_mask`` is ``(B, text_len)``
        bool. The empty prompt encodes to the reference CFG null."""
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        embeds, mask = self.prompt_encoder(self.text_encoder, list(prompt), dtype=self.torch_dtype)
        pe = PromptEmbeds(embeds)
        pe.attention_mask = mask
        return pe

    def condition_noisy_latents(self, latents: torch.Tensor, batch):
        return latents

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        """Full fine-tune save in the reference export layout:
        ``<output_path>/model.safetensors`` + ``config.yaml``."""
        os.makedirs(output_path, exist_ok=True)
        transformer: IrisDiT = unwrap_model(self.model)
        transformer.save_model(
            os.path.join(output_path, "model.safetensors"),
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name="iris3b"),
        )
        raw = dict(self.iris_raw_config)
        raw["model"] = transformer.config.to_dict()
        te = dict(raw.get("text_encoder", {}) or {})
        te.update(
            {
                "name": "qwen3_vl",
                "pretrained": self.te_config_repo,
                "dim": transformer.config.text_dim,
                "max_length": self.text_len,
                "hidden_layers": list(self.text_hidden_layers),
            }
        )
        raw["text_encoder"] = te
        flow = dict(raw.get("flow", {}) or {})
        flow.update(
            {
                "num_train_timesteps": self.num_train_timesteps,
                "shift": self.flow_shift,
                "prediction": self.prediction,
                "x_pred_sigma_min": self.x_pred_sigma_min,
            }
        )
        raw["flow"] = flow
        with open(os.path.join(output_path, "config.yaml"), "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        with open(os.path.join(output_path, "aitk_meta.yaml"), "w") as f:
            yaml.dump(meta, f)

    def get_base_model_version(self):
        return "iris3b"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["blocks"]

    lora_keys_use_comfy_prefix = True
