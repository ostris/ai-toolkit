"""MiniMax-H3 (33B joint video+audio DiT) for ai-toolkit.

Supports t2v (t2va) and first-frame i2v (fl2va) training and sampling, with
joint audio when the dataset provides it. Image datasets train as single
latent frames (keyframe-row geometry) and sampling with num_frames 1 renders
a single image. The architecture lives in ./src/:

  - transformer.py: packed-sequence DiT, weight-compatible with the original
    ``MiniMaxAI/MiniMax-H3`` checkpoint keys
  - vae.py / audio_vae.py: the video VAE (causal CNN encoder + ViT decoder,
    16x/17n+5->5n+2) and the waveform audio VAE (DAC/BigVGAN, 32 kHz, 40 Hz)
  - packing.py: packed-sequence geometry, rotary grids, sigma-shift math
  - text_encoder.py: Qwen3-VL-32B conditioning (unnormalized hidden_states[50],
    "<Picture i>: " + vision block presentation for keyframes)
  - pipeline.py: the released sampler (no CFG — the model is guidance-distilled)

Weights load from the Comfy-Org repack (``Comfy-Org/MiniMax-H3``) by default:
the pruned int8-ConvRot transformer, the nvfp4 AWQ Qwen3-VL text encoder (kept
quantized through the toolkit's Ostris quantization backends — convrot8 and
nvfp4 — with dequantized-matmul fallbacks for GPUs without the fast kernels),
and the fp16/fp32 single-file VAEs. Files are resolved under ``MODELS_PATH``
(checked first, both at the repo-relative location and flat at the root) and
downloaded from the hub into ``MODELS_PATH`` when missing. Individual files
can be overridden via ``model_kwargs``: ``dit_<partition>_path``,
``text_encoder_path``, ``video_vae_path``, ``audio_vae_path``;
``model_kwargs.partition`` picks ``fl2va``, ``fl2va_pruned`` (default),
``ref2va``, or ``ref2va_pruned``.

Conventions bridged to ai-toolkit:
  - the model consumes t = 1 - sigma in [0, 1] (t=1 clean) and predicts the
    data-ward velocity ``clean - noise``; ai-toolkit targets ``noise - clean``
    on a 0..1000 timestep scale, so timesteps are flipped and the prediction
    negated in get_noise_prediction
  - the audio stream runs on its own flow shift (3 vs video's 12): its sigma
    is derived per step from the video sigma via the closed-form remap, in
    training and sampling alike
"""

import os
from functools import partial
from typing import TYPE_CHECKING, List, Optional

import torch
import yaml
from PIL import Image
from safetensors.torch import load_file, save_file

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management import MemoryManager
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.paths import MODELS_PATH
from toolkit.util.comfy_quant_import import import_comfy_quantized_layers
from toolkit.util.ostris_quant import OstrisLinear
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import get_qtype, quantize, quantize_model
from optimum.quanto import freeze

from .src import packing

packing_video_exts = [".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".m4v", ".flv"]
from .src.audio_vae import MiniMaxH3AudioVAE, fold_audio_vae_weight_norm
from .src.packing import (
    KEYFRAME_ENCODE_SEED,
    KEYFRAME_NOISE_AUG_T,
    build_packed_sequence,
    pack_audio_latents,
    pad_layouts_to_batch,
    unpack_audio_tokens,
    patchify_video_latents,
    remap_sigma,
    unpatchify_video_tokens,
)
from .src.pipeline import MiniMaxH3Pipeline
from .src.ref_video_cache import (
    load_ref_video_latent,
    load_video_ref_for_te,
    ref_frame_indices,
    static_image_video_ref,
)
from .src.text_encoder import (
    TEXT_ENCODER_LAYER,
    VideoRef,
    encode_minimax_h3_prompt,
    load_video_ref,
    trim_caption_tokens,
)
from .src.transformer import MiniMaxH3Transformer, MiniMaxH3TransformerParams
from .src.vae import MiniMaxH3VideoVAE

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": packing.VIDEO_SIGMA_SHIFT,
    "use_dynamic_shifting": False,
}

# Comfy-Org repack of the released weights, at ComfyUI's repo-relative paths
# under MODELS_PATH (diffusion_models/, text_encoders/, vae/). Files are used
# in place when present and downloaded to exactly these locations only when
# missing.
COMFY_REPO = "Comfy-Org/MiniMax-H3"
COMFY_FILES = {
    "dit_fl2va": "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
    "dit_fl2va_pruned": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "dit_ref2va": "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
    "dit_ref2va_pruned": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "text_encoder": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "vae/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
}
# tokenizer/processor/text-encoder config come from the original repo (tiny files)
ORIGINAL_REPO = "MiniMaxAI/MiniMax-H3"


def new_save_image_function(
    self: GenerateImageConfig, image, count=0, max_count=0, **kwargs
):
    # video (+ audio) previews save as mp4
    try:
        from diffusers.utils import encode_video
    except ImportError:
        from diffusers.pipelines.ltx2.export_utils import encode_video
    image["output_path"] = self.get_image_path(count, max_count)
    os.makedirs(os.path.dirname(image["output_path"]), exist_ok=True)
    if image.get("audio", None) is None:
        image.pop("audio", None)
        image.pop("audio_sample_rate", None)
    encode_video(**image)
    flush()


def blank_log_image_function(self, *args, **kwargs):
    # todo handle wandb logging of videos with audio
    return


class MiniMaxH3VaeBundle(torch.nn.Module):
    """Holds both frozen autoencoders behind the single ``self.vae`` handle."""

    def __init__(self, video_vae: MiniMaxH3VideoVAE, audio_vae: MiniMaxH3AudioVAE):
        super().__init__()
        self.video_vae = video_vae
        self.audio_vae = audio_vae

    @property
    def device(self):
        return self.video_vae.device

    @property
    def dtype(self):
        return self.video_vae.dtype

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.video_vae.enable_gradient_checkpointing(enable)
        self.audio_vae.enable_gradient_checkpointing(enable)

    def disable_gradient_checkpointing(self):
        self.enable_gradient_checkpointing(False)


class MinimaxH3Model(BaseModel):
    arch = "minimax_h3"
    use_old_lokr_format = False

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
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["MiniMaxH3Transformer"]
        self.supports_model_paths = True
        # keyframes ride into the Qwen3-VL conditioning as vision blocks, so
        # sampling (and control_path datasets) pass control images to
        # get_prompt_embeds
        self.encode_control_in_text_embeddings = True

        self.processor = None  # Qwen3VLProcessor
        self._warned_frame_trim = False
        # video-ref presentation context: dataset config while caching training
        # embeds; the sample's frame cap while encoding sample prompts
        self._ref_video_dataset_config = None
        self._sample_ref_max_frames = None
        self.latent_space_version = "minimax_h3_v1"
        # caption token cap (vision blocks are never truncated); the released
        # stack has no limit — set 0 to disable
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", 512)
        )

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # 16x VAE spatial compression * 2x2 transformer patch
        return 32

    def get_frame_count_snapper(self):
        # auto_frame_count: snap dataset clips down to the VAE's 17n+5 grid
        return packing.align_num_frames_down

    def prepare_sample_prompt_context(self, gen_config):
        # sample prompts: video refs are treated at the sample's length, not
        # the dataset's (which only applies while caching training embeds)
        self._ref_video_dataset_config = None
        self._sample_ref_max_frames = max(int(gen_config.num_frames), 5)

    @property
    def video_vae(self) -> MiniMaxH3VideoVAE:
        return self.vae.video_vae

    @property
    def audio_vae(self) -> MiniMaxH3AudioVAE:
        return self.vae.audio_vae

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def _find_file_recursive(root_dir: str, filename: str) -> Optional[str]:
        """First (breadth-stable, sorted) match of ``filename`` anywhere under
        ``root_dir``."""
        if not os.path.isdir(root_dir):
            return None
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames.sort()
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None

    def _resolve_comfy_file(self, component: str) -> str:
        """Find a weight file at its local location, or download it there
        when (and only when) it is missing.

        Search order: model_kwargs override, the repo-relative path under
        MODELS_PATH (diffusion_models/, text_encoders/, vae/), the bare
        filename at the root, any subfolder of the component's category
        folder (recursive — e.g. diffusion_models/my_custom_sub/), the same
        spots under name_or_path when it is a local folder, then the hub —
        downloaded to the repo-relative path under MODELS_PATH.
        """
        override = self.model_config.model_kwargs.get(f"{component}_path", None)
        if override is not None:
            if not os.path.exists(override):
                raise FileNotFoundError(
                    f"model_kwargs.{component}_path does not exist: {override}"
                )
            return override

        rel_path = COMFY_FILES[component]
        filename = os.path.basename(rel_path)
        category = os.path.dirname(rel_path)
        roots = [MODELS_PATH]
        name_or_path = self.model_config.name_or_path
        if name_or_path and os.path.isdir(name_or_path):
            roots.append(name_or_path)
        for root in roots:
            for rel in (rel_path, filename):
                candidate = os.path.join(root, rel)
                if os.path.exists(candidate):
                    return candidate
        for root in roots:
            found = self._find_file_recursive(os.path.join(root, category), filename)
            if found is not None:
                return found

        import huggingface_hub

        repo_id = COMFY_REPO
        if name_or_path and not os.path.exists(name_or_path) and "/" in name_or_path:
            repo_id = name_or_path
        self.print_and_status_update(
            f"Downloading {rel_path} from {repo_id} into {MODELS_PATH}"
        )
        return huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=rel_path, local_dir=MODELS_PATH
        )

    def _dit_component(self) -> str:
        partition = str(
            self.model_config.model_kwargs.get("partition", "fl2va_pruned")
        ).lower()
        if partition not in ("fl2va", "fl2va_pruned", "ref2va", "ref2va_pruned"):
            raise ValueError(
                "model_kwargs.partition must be fl2va, fl2va_pruned, ref2va, "
                f"or ref2va_pruned, got {partition}"
            )
        return f"dit_{partition}"

    def load_training_adapter(self, transformer: MiniMaxH3Transformer):
        """Load an assistant LoRA (e.g. a de-distillation adapter) as a LIVE
        module: active during training, deactivated by the sampler. It is
        deliberately NOT merged into the base weights — the transformer is
        pre-quantized, and a merge would resample every int8 scale.

        Path resolution: a local path is used as-is; otherwise the loras
        folder under MODELS_PATH is searched recursively for the filename;
        otherwise a ``user/repo/file.safetensors`` hub path downloads into
        MODELS_PATH/loras/training_adapters/.
        """
        from toolkit.config_modules import NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork

        self.print_and_status_update("Loading assistant LoRA")
        lora_path = self.model_config.assistant_lora_path
        if not os.path.exists(lora_path):
            filename = os.path.basename(lora_path)
            found = self._find_file_recursive(
                os.path.join(MODELS_PATH, "loras"), filename
            )
            if found is not None:
                lora_path = found
            else:
                lora_splits = lora_path.split("/")
                if len(lora_splits) != 3:
                    raise ValueError(
                        f"Assistant LoRA path {lora_path} is not a local path, a "
                        f"file under {os.path.join(MODELS_PATH, 'loras')}, or a "
                        "'user/repo/file.safetensors' hub path."
                    )
                import huggingface_hub

                target_dir = os.path.join(MODELS_PATH, "loras", "training_adapters")
                os.makedirs(target_dir, exist_ok=True)
                try:
                    lora_path = huggingface_hub.hf_hub_download(
                        repo_id="/".join(lora_splits[:2]),
                        filename=lora_splits[2],
                        local_dir=target_dir,
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to download assistant LoRA from {lora_path}: {e}"
                    )
            self.model_config.assistant_lora_path = lora_path

        # load the adapter; it stays a live module (never merged) and the
        # sampler toggles it off for previews
        lora_state_dict = load_file(lora_path)
        dim_key = next(
            k
            for k in lora_state_dict
            if k.endswith("lora_A.weight") or k.endswith("lora_down.weight")
        )
        dim = int(lora_state_dict[dim_key].shape[0])
        lora_state_dict = self.convert_lora_weights_before_load(lora_state_dict)

        network_config = NetworkConfig(
            **{
                "type": "lora",
                "linear": dim,
                "linear_alpha": dim,
                "transformer_only": True,
            }
        )
        LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=self.target_lora_modules,
            is_assistant_adapter=True,
            is_ara=True,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)

        # frozen: the adapter shapes the training distribution but is never
        # itself trained, so its params must not collect gradients
        network.is_merged_in = False
        for param in network.parameters():
            param.requires_grad_(False)
        network.eval()

        self.assistant_lora: LoRASpecialNetwork = network

        # live during training; the sampler's non-inverted assistant path
        # (BaseModel.generate_images) deactivates it for previews and turns
        # it back on afterwards
        self.assistant_lora.multiplier = 1.0
        self.assistant_lora.is_active = True
        self.invert_assistant_lora = False

    def _load_transformer(self) -> MiniMaxH3Transformer:
        dtype = self.torch_dtype
        dit_path = self._resolve_comfy_file(self._dit_component())
        self.print_and_status_update(f"Loading transformer from {dit_path}")
        state_dict = load_file(dit_path)

        params = MiniMaxH3TransformerParams()
        table = state_dict.get("adaln_t_table", None)
        if table is not None:
            # pruned checkpoint: factored timestep table instead of the MLP
            params.adaln_t_table_size = table.shape[0]
            params.time_embed_dim = table.shape[1]

        with torch.device("meta"):
            transformer = MiniMaxH3Transformer(params)

        # attach the pre-quantized (int8 ConvRot) linears onto the toolkit's
        # quantization backends; the rest loads at its stored precision (the
        # checkpoint's bf16/fp16/fp32 mix is deliberate)
        state_dict, num_quantized = import_comfy_quantized_layers(
            transformer, state_dict, orig_dtype=dtype
        )
        if num_quantized:
            self.print_and_status_update(
                f" - attached {num_quantized} pre-quantized ConvRot layers"
            )
        result = transformer.load_state_dict(state_dict, assign=True, strict=False)
        quantized_weight_keys = {
            f"{name}.weight"
            for name, m in transformer.named_modules()
            if isinstance(m, OstrisLinear)
        }
        bad_missing = [k for k in result.missing_keys if k not in quantized_weight_keys]
        if bad_missing or result.unexpected_keys:
            raise ValueError(
                f"MiniMax-H3 transformer load mismatch: missing {bad_missing[:8]}, "
                f"unexpected {result.unexpected_keys[:8]}"
            )
        del state_dict
        flush()
        return transformer

    def _load_text_encoder(self):
        from accelerate import init_empty_weights
        from transformers import (
            AutoConfig,
            AutoProcessor,
            AutoTokenizer,
            Qwen3VLForConditionalGeneration,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            ORIGINAL_REPO, subfolder="FL2VA/tokenizer"
        )
        processor = AutoProcessor.from_pretrained(
            ORIGINAL_REPO, subfolder="FL2VA/processor"
        )

        te_path = self.model_config.te_name_or_path
        if te_path is not None and os.path.isdir(te_path):
            # transformers-format folder (e.g. the original repo's text_encoder)
            self.print_and_status_update(
                f"Loading Qwen3-VL text encoder from {te_path}"
            )
            config = AutoConfig.from_pretrained(te_path)
            config.text_config.num_hidden_layers = TEXT_ENCODER_LAYER
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                te_path, config=config, torch_dtype=self.te_torch_dtype
            )
        else:
            if te_path is not None:
                te_file = te_path
            else:
                te_file = self._resolve_comfy_file("text_encoder")
            self.print_and_status_update(
                f"Loading Qwen3-VL text encoder from {te_file}"
            )
            # single-file ComfyUI checkpoint: 50 decoder layers, no final norm,
            # no lm_head; LM linears nvfp4 (AWQ), embeddings int8, vision bf16
            config = AutoConfig.from_pretrained(
                ORIGINAL_REPO, subfolder="FL2VA/text_encoder"
            )
            # only hidden_states[50] is consumed: truncate the decoder stack to
            # 50 layers; the final norm is neutralized below so
            # hidden_states[-1] stays the unnormalized layer-49 output
            config.text_config.num_hidden_layers = TEXT_ENCODER_LAYER
            config.tie_word_embeddings = False
            with init_empty_weights():
                text_encoder = Qwen3VLForConditionalGeneration(config)
            text_encoder.lm_head = None

            state_dict = load_file(te_file)

            def key_map(prefix: str) -> str:
                if prefix.startswith("model."):
                    return "model.language_model." + prefix[len("model.") :]
                if prefix.startswith("visual."):
                    return "model." + prefix
                return prefix

            state_dict, num_quantized = import_comfy_quantized_layers(
                text_encoder,
                state_dict,
                orig_dtype=self.te_torch_dtype,
                key_map=key_map,
            )
            self.print_and_status_update(
                f" - attached {num_quantized} pre-quantized nvfp4/int8 layers"
            )
            state_dict = {
                key_map(k[: k.rfind(".")]) + k[k.rfind(".") :]: v
                for k, v in state_dict.items()
            }
            result = text_encoder.load_state_dict(state_dict, assign=True, strict=False)
            quantized_keys = set()
            for name, m in text_encoder.named_modules():
                if isinstance(m, OstrisLinear):
                    quantized_keys.add(f"{name}.weight")
            allowed_missing_prefixes = (
                "lm_head",
                "model.language_model.norm",
                "model.language_model.embed_tokens",
            )
            bad_missing = [
                k
                for k in result.missing_keys
                if k not in quantized_keys
                and not k.startswith(allowed_missing_prefixes)
            ]
            if bad_missing or result.unexpected_keys:
                raise ValueError(
                    f"MiniMax-H3 text encoder load mismatch: missing {bad_missing[:8]}, "
                    f"unexpected {result.unexpected_keys[:8]}"
                )
            del state_dict
        text_encoder.model.language_model.norm = torch.nn.Identity()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()
        return tokenizer, processor, text_encoder

    def _load_vaes(self) -> MiniMaxH3VaeBundle:
        self.print_and_status_update("Loading video VAE")
        video_sd = load_file(self._resolve_comfy_file("video_vae"))
        # normalization stats ride along in the comfy file; the module holds
        # them as non-persistent buffers, keep them float32
        video_stats = {
            k: video_sd.pop(k).float()
            for k in ("latents_mean", "latents_std")
            if k in video_sd
        }
        video_vae = MiniMaxH3VideoVAE()
        video_vae.load_state_dict(video_sd, strict=True, assign=True)
        for k, v in video_stats.items():
            getattr(video_vae, k).copy_(v)
        video_vae.eval().requires_grad_(False)
        del video_sd

        self.print_and_status_update("Loading audio VAE")
        audio_sd = load_file(self._resolve_comfy_file("audio_vae"))
        audio_stats = {
            k: audio_sd.pop(k).float()
            for k in ("latents_mean", "latents_std")
            if k in audio_sd
        }
        # comfy repack ships the weight norm already folded; fold only if the
        # raw parametrization is present (original-repo file)
        if any(k.endswith("weight_g") for k in audio_sd.keys()):
            audio_sd = fold_audio_vae_weight_norm(audio_sd)
        audio_vae = MiniMaxH3AudioVAE()
        audio_vae.load_state_dict(audio_sd, strict=True, assign=True)
        for k, v in audio_stats.items():
            getattr(audio_vae, k).copy_(v)
        audio_vae.to(torch.float32).eval().requires_grad_(False)
        del audio_sd
        flush()
        return MiniMaxH3VaeBundle(video_vae, audio_vae)

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading MiniMax-H3 model")

        transformer = self._load_transformer()

        # load assistant lora if specified (merged into the quantized weights)
        if self.model_config.assistant_lora_path is not None:
            self.load_training_adapter(transformer)

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Keeping transformer on CPU")
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch)
        flush()

        tokenizer, processor, text_encoder = self._load_text_encoder()
        te_prequantized = any(
            isinstance(m, OstrisLinear) for m in text_encoder.modules()
        )
        if self.model_config.quantize_te and not te_prequantized:
            self.print_and_status_update("Quantizing text encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()
        elif self.model_config.quantize_te:
            self.print_and_status_update(
                "Text encoder is already nvfp4/int8 quantized; skipping quantize_te"
            )
        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )
        if self.model_config.low_vram:
            text_encoder.to("cpu")
        else:
            text_encoder.to(self.device_torch)
        flush()

        vae_bundle = self._load_vaes()
        vae_bundle.to(self.vae_device_torch)

        self.noise_scheduler = MinimaxH3Model.get_train_scheduler()
        self.vae = vae_bundle
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = transformer
        self.pipeline = MiniMaxH3Pipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Text conditioning
    # ------------------------------------------------------------------
    def _present_image_control(self, image: Image.Image):
        """Hook: how a control IMAGE enters the Qwen3-VL presentation. The
        default is a plain ``<Picture i>`` image; ref2va can turn it into a
        static video reference (``image_refs_as_video``)."""
        return image

    def get_prompt_embeds(self, prompt, control_images=None) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        # control tensors arrive in [0, 1]; the Qwen3-VL processor wants PIL
        keyframes_per_prompt = [None] * len(prompt)
        if control_images is not None:
            if isinstance(control_images, torch.Tensor):
                images = [control_images[i] for i in range(control_images.shape[0])]
            elif isinstance(control_images, list):
                images = [
                    c[0] if isinstance(c, torch.Tensor) and c.ndim == 4 else c
                    for c in control_images
                ]
            else:
                images = [control_images]
            pil_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    if img.ndim == 4:
                        img = img[0]
                    arr = (img.float().clamp(0, 1) * 255).round().to(torch.uint8)
                    pil_images.append(
                        self._present_image_control(
                            Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())
                        )
                    )
                elif isinstance(img, str):
                    # a control VIDEO path: 2 fps timestamped presentation over
                    # the SAME frames the latent rows use (dataset treatment
                    # when caching training embeds, sample-length at sampling)
                    ds_cfg = getattr(self, "_ref_video_dataset_config", None)
                    pil_images.append(
                        load_video_ref_for_te(
                            self, img, ds_cfg, max_frames=self._sample_ref_max_frames
                        )
                    )
                else:
                    pil_images.append(img)
            if len(pil_images) == 1:
                keyframes_per_prompt = [pil_images] * len(prompt)
            elif len(pil_images) == len(prompt):
                keyframes_per_prompt = [[img] for img in pil_images]
            else:
                keyframes_per_prompt = [pil_images] * len(prompt)

        embeds_list, tags_list = [], []
        for p, keyframes in zip(prompt, keyframes_per_prompt):
            embeds, tags = encode_minimax_h3_prompt(
                self.text_encoder,
                self.tokenizer,
                self.processor,
                p.strip(),
                keyframes=keyframes,
                device=self.device_torch,
                dtype=self.torch_dtype,
                max_length=self.max_text_length,
            )
            embeds_list.append(embeds)
            tags_list.append(tags)

        pe = AdvancedPromptEmbeds(text_embeds=embeds_list, text_token_tags=tags_list)
        pe.frozen_dtype_keys = ["text_token_tags"]
        return pe

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, image_list, device=None, dtype=None):
        """Images (C, H, W) or videos (T, C, H, W) in [-1, 1] -> normalized
        video latents (B, 24, t, h, w). Video frame counts are trimmed down to
        the VAE's 17n+5 grid when needed."""
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)

        items = []
        for image in image_list:
            if image.ndim == 3:
                items.append(image.unsqueeze(1))  # (C, 1, H, W)
            elif image.ndim == 4:
                items.append(image.permute(1, 0, 2, 3))  # (C, T, H, W)
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")

        num_frames = items[0].shape[1]
        if num_frames > 1:
            aligned = packing.align_num_frames_down(num_frames)
            if aligned != num_frames and not self._warned_frame_trim:
                print(
                    f"MiniMax-H3: trimming {num_frames}-frame clips to {aligned} "
                    f"frames (the video VAE needs 17n+5: 5, 22, 39, 56, ...). Set "
                    f"the dataset num_frames accordingly to avoid wasted decode."
                )
                self._warned_frame_trim = True
            items = [it[:, :aligned] for it in items]

        batch = torch.stack(items).to(self.vae_device_torch, self.video_vae.dtype)
        latents = self.video_vae.encode(batch, sample=True)
        return latents.to(device, dtype=dtype)

    @torch.no_grad()
    def encode_keyframe_latents(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, 3, 1, H, W) in [-1, 1] -> normalized latents (B, 24, 1, h, w),
        with the released conditioning recipe: seeded posterior sample (seed
        42, independent of the request seed) rounded to fp16 before
        normalization."""
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)
        generator = torch.Generator(device="cpu").manual_seed(KEYFRAME_ENCODE_SEED)
        latents = self.video_vae.encode(
            frames.to(self.vae_device_torch, self.video_vae.dtype),
            sample=True,
            generator=generator,
            fp16_round=True,
        )
        return latents.float()

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        # differentiable: pixel-space losses backprop through the video VAE
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)
        video = self.video_vae.decode(latents.to(self.vae.device, self.video_vae.dtype))
        if device is not None:
            video = video.to(device, dtype=dtype)
        return video

    def decode_audio_latents(self, latents: torch.Tensor):
        # differentiable, like decode_latents
        """(B, 32, T) normalized -> waveform (B, 1, T*800) at 32 kHz."""
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)
        return self.audio_vae.decode(latents.to(self.audio_vae.device, torch.float32))

    @property
    def audio_sample_rate(self) -> int:
        return packing.AUDIO_SAMPLE_RATE

    def decode_packed_audio_rows(self, rows: torch.Tensor) -> torch.Tensor:
        # differentiable: audio perceptual losses backprop through the audio VAE
        """Packed channel-major audio rows (B, 2*T, 32) -> stereo waveform
        (B, 2, T*800) at 32 kHz. Each stereo channel decodes as its own batch
        item through the mono audio VAE."""
        a_lat = rows.shape[1] // packing.AUDIO_CHANNELS
        latents = unpack_audio_tokens(rows, a_lat)  # (B, 2, 32, T)
        b = latents.shape[0]
        waveform = self.decode_audio_latents(
            latents.reshape(b * packing.AUDIO_CHANNELS, latents.shape[2], a_lat)
        )  # (B*2, 1, samples)
        return waveform.reshape(b, packing.AUDIO_CHANNELS, -1)

    @torch.no_grad()
    def encode_audio(self, audio_data_list):
        """[{"waveform": (C, L), "sample_rate": int}, ...] -> packed audio
        rows (B, 2*T, 32), normalized, channel-major stereo."""
        import torchaudio

        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.device_torch)

        packed = []
        for audio_data in audio_data_list:
            waveform = audio_data["waveform"].to(self.audio_vae.device, torch.float32)
            sample_rate = int(audio_data["sample_rate"])
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)  # mono -> stereo
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]
            if sample_rate != packing.AUDIO_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, packing.AUDIO_SAMPLE_RATE
                )
            # the mono VAE sees each stereo channel as its own batch item
            z = self.audio_vae.encode(waveform.unsqueeze(1))  # (2, 32, T)
            packed.append(pack_audio_latents(z.unsqueeze(0)))  # (1, 2*T, 32)

        max_len = max(p.shape[1] for p in packed)
        packed = [
            torch.nn.functional.pad(p, (0, 0, 0, max_len - p.shape[1])) for p in packed
        ]
        return torch.cat(packed, dim=0).to(self.device_torch, self.torch_dtype)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def _build_condition(
        self, batch: "DataLoaderBatchDTO", latent_shape, device, dtype
    ):
        """Build the packed sequence's condition segment for one train step.

        ``latent_shape`` is the target's ``(t_lat, h_lat, w_lat)``. Returns
        ``(cond_rows, cond_audio_rows, keyframe_anchors, ref_blocks)`` where ``cond_rows``
        is ``(B, num_condition_rows, 96)`` or None. The base model implements
        fl2va: the clip's first frame as a keyframe when the dataset asks for
        i2v. MinimaxH3Ref2VAModel overrides this with image references from
        the control images."""
        do_i2v = (
            batch is not None
            and batch.dataset_config.do_i2v
            and getattr(batch, "num_frames", 1) > 1
        )
        if not do_i2v:
            return None, None, (), ()

        if batch.first_frame_latents is not None:
            first_latents = batch.first_frame_latents.to(device, torch.float32)
        else:
            frames = batch.tensor
            if frames is None:
                raise ValueError(
                    "do_i2v needs the first frame; no cached "
                    "first_frame_latents or raw tensors in batch"
                )
            first_frames = frames[:, 0] if frames.ndim == 5 else frames
            first_latents = self.encode_keyframe_latents(
                first_frames.unsqueeze(2).to(device)
            )
        if first_latents.ndim == 4:
            first_latents = first_latents.unsqueeze(2)
        cond_noise = torch.randn_like(first_latents)
        first_latents = (
            KEYFRAME_NOISE_AUG_T * first_latents
            + (1.0 - KEYFRAME_NOISE_AUG_T) * cond_noise
        )
        return patchify_video_latents(first_latents).to(dtype), None, ("first",), ()

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 24, t, h, w) noisy latents
        timestep: torch.Tensor,  # (B,) on the 0..1000 scale, 1000 = pure noise
        text_embeddings: AdvancedPromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        device = self.device_torch
        dtype = self.torch_dtype
        if self.model.device == torch.device("cpu"):
            self.model.to(device)

        # a grad-enabled prediction is the primary (loss carrying) one unless
        # the trainer declared a secondary slot on the batch (prior /
        # guidance-unconditional / preservation passes). Trainers that make
        # several grad predictions per step (e.g. turbo rollouts) get one
        # primary per prediction, last writer wins.
        is_primary_pred = (
            torch.is_grad_enabled()
            and batch is not None
            and batch.audio_pred_slot is None
        )

        batch_size, _, t_lat, h_lat, w_lat = latent_model_input.shape

        with torch.no_grad():
            sigma_v = (timestep.to(device, torch.float32) / 1000.0).clamp(1e-6, 1.0)
            if sigma_v.dim() == 0:
                sigma_v = sigma_v.unsqueeze(0)
            if sigma_v.shape[0] != batch_size:
                sigma_v = sigma_v.expand(batch_size)
            sigma_a = remap_sigma(sigma_v)
            t_v = 1.0 - sigma_v
            t_a = 1.0 - sigma_a

            # --- conditioning rows (fl2va keyframe / ref2va references) ----
            (
                cond_rows,
                cond_audio_rows,
                keyframe_anchors,
                ref_blocks,
            ) = self._build_condition(batch, (t_lat, h_lat, w_lat), device, dtype)

            # --- audio rows -------------------------------------------------
            if batch is not None and getattr(batch, "num_frames", None):
                num_frames = batch.num_frames
            else:
                # invert 17n+5 -> 5n+2 from the latent frame count
                num_frames = (t_lat - 2) // 5 * 17 + 5 if t_lat > 1 else 1
            a_lat = packing.audio_latent_num_frames(num_frames)
            # audio only trains for video batches from datasets that asked for
            # it. Cached latents can carry audio after do_audio was turned off,
            # and image (single frame) batches must never pick up a soundtrack
            # — either way it rides along as silence with no audio loss.
            do_audio = (
                batch is not None
                and batch.dataset_config is not None
                and batch.dataset_config.do_audio
                and num_frames > 1
            )
            raw_audio = None
            if do_audio and batch.audio_latents is not None:
                raw_audio = batch.audio_latents.to(device, torch.float32)
            elif do_audio and getattr(batch, "audio_data", None) is not None:
                raw_audio = self.encode_audio(batch.audio_data).to(
                    device, torch.float32
                )

            sa = sigma_a.view(-1, 1, 1)
            if raw_audio is not None:
                expected_rows = a_lat * packing.AUDIO_CHANNELS
                if raw_audio.shape[1] > expected_rows:
                    raw_audio = raw_audio[:, :expected_rows]
                elif raw_audio.shape[1] < expected_rows:
                    raw_audio = torch.nn.functional.pad(
                        raw_audio, (0, 0, 0, expected_rows - raw_audio.shape[1])
                    )
                # the audio noise is drawn once per step and shared by every
                # pass (prior, primary, cfg/guidance, preservation) so they all
                # see the same soundtrack and the stored target keeps matching
                if (
                    batch.audio_noise is not None
                    and batch.audio_noise.shape == raw_audio.shape
                ):
                    audio_noise = batch.audio_noise.to(device, torch.float32)
                else:
                    audio_noise = torch.randn_like(raw_audio)
                    batch.audio_noise = audio_noise
                audio_rows = (1.0 - sa) * raw_audio + sa * audio_noise
                batch.audio_latents = raw_audio
                if batch.audio_target is None:
                    # model predicts clean - noise; audio_pred is negated below
                    # so the stored target follows ai-toolkit's noise - clean.
                    # With the shared noise this is the same value on every
                    # pass, so first writer is fine (and it keeps a guidance
                    # extrapolated target from being overwritten).
                    batch.audio_target = (audio_noise - raw_audio).detach()
                if is_primary_pred:
                    # expose what audio perceptual losses need to rebuild the
                    # clean estimate (x0 = noisy - sigma_a * pred). Tied to the
                    # primary pass so they always match audio_pred, even when a
                    # trainer makes primary predictions at several sigmas.
                    batch.audio_noisy = audio_rows
                    batch.audio_sigma = sigma_a
            else:
                # no soundtrack: silence (zeros) noised at the audio sigma
                # rides along without contributing to the loss
                audio_rows = sa * torch.randn(
                    batch_size,
                    a_lat * packing.AUDIO_CHANNELS,
                    32,
                    device=device,
                    dtype=torch.float32,
                )

            # embeds cached with a longer max_text_length: cap the caption
            # tail (vision blocks are never touched)
            trimmed = [
                trim_caption_tokens(e, t, self.max_text_length)
                for e, t in zip(
                    text_embeddings.text_embeds, text_embeddings.text_token_tags
                )
            ]
            text_embed_list = [e for e, _ in trimmed]
            text_tag_list = [t for _, t in trimmed]

            # --- packed layout (per item: text lengths differ) --------------
            layouts = []
            for i in range(batch_size):
                layouts.append(
                    build_packed_sequence(
                        text_token_tags=text_tag_list[i].to("cpu"),
                        num_latent_frames=t_lat,
                        latent_height=h_lat,
                        latent_width=w_lat,
                        num_audio_latents=a_lat,
                        keyframe_anchors=keyframe_anchors,
                        ref_blocks=ref_blocks,
                    )
                )
            (
                position_ids,
                token_tags,
                video_indices,
                audio_indices,
                text_indices,
                _,
            ) = pad_layouts_to_batch(layouts)
            num_cond = layouts[0].num_condition_video_rows
            num_cond_audio = layouts[0].num_condition_audio_rows

            # per-row timesteps: text/video rows at t_v, audio rows at t_a,
            # condition rows pinned at max(t_v, 0.999); ref soundtracks clean
            row_t = t_v.view(-1, 1).expand(-1, token_tags.shape[1]).clone()
            row_t[:, audio_indices] = t_a.view(-1, 1)
            if num_cond > 0:
                cond_t = torch.maximum(t_v, torch.full_like(t_v, KEYFRAME_NOISE_AUG_T))
                row_t[:, video_indices[:num_cond]] = cond_t.view(-1, 1)
            if num_cond_audio > 0:
                row_t[:, audio_indices[:num_cond_audio]] = 1.0
                audio_rows = torch.cat(
                    [cond_audio_rows.to(audio_rows.dtype), audio_rows], dim=1
                )

            # pad text embeds to the batch max length
            max_text = int(text_indices.shape[0])
            text_batch = torch.zeros(
                batch_size,
                max_text,
                text_embed_list[0].shape[-1],
                device=device,
                dtype=dtype,
            )
            for i, emb in enumerate(text_embed_list):
                text_batch[i, : emb.shape[0]] = emb.to(device, dtype)

            video_rows = patchify_video_latents(
                latent_model_input.to(device, torch.float32)
            ).to(dtype)
            if cond_rows is not None:
                video_rows = torch.cat([cond_rows, video_rows], dim=1)

        video_pred, audio_pred = self.model(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows.to(dtype),
            encoder_hidden_states=text_batch,
            row_timesteps=row_t.to(device),
            token_tags=token_tags.to(device),
            position_ids=position_ids.to(device),
            video_indices=video_indices.to(device),
            audio_indices=audio_indices.to(device),
            text_indices=text_indices.to(device),
        )

        if num_cond_audio > 0:
            # reference soundtrack rows are conditioning, not targets
            audio_pred = audio_pred[:, num_cond_audio:]
        if batch is not None and batch.audio_target is not None:
            # flip to ai-toolkit's noise - clean convention
            if is_primary_pred:
                batch.audio_pred = -audio_pred
            else:
                batch.set_secondary_audio_pred(-audio_pred)

        video_pred = video_pred[:, num_cond:]
        noise_pred = unpatchify_video_tokens(video_pred, t_lat, h_lat, w_lat)
        return -noise_pred

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    # ------------------------------------------------------------------
    # Sampling (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return MiniMaxH3Pipeline(self)

    def generate_single_image(
        self,
        pipeline: MiniMaxH3Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = max(sc, int(gen_config.width // sc * sc))
        gen_config.height = max(sc, int(gen_config.height // sc * sc))

        is_video = gen_config.num_frames > 1
        if is_video:
            gen_config.num_frames = packing.align_num_frames_down(gen_config.num_frames)
            gen_config.fps = packing.FPS
            gen_config.save_image = partial(new_save_image_function, gen_config)
            gen_config.log_image = partial(blank_log_image_function, gen_config)
            gen_config.output_ext = "mp4"

        ctrl_img = None
        if gen_config.ctrl_img is not None:
            ctrl_img = Image.open(gen_config.ctrl_img).convert("RGB")
            ctrl_img = packing.prepare_keyframe_image(
                ctrl_img, gen_config.height, gen_config.width, stretch=True
            )

        with_audio = bool(self.model_config.model_kwargs.get("sample_audio", True))

        result = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_frames=gen_config.num_frames,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            ctrl_img=ctrl_img,
            with_audio=with_audio and is_video,
        )
        if is_video:
            return result  # dict consumed by new_save_image_function
        return result[0]

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        from toolkit.util.quantize import dequantize_if_quantized

        transformer: MiniMaxH3Transformer = unwrap_model(self.model)
        os.makedirs(os.path.join(output_path, "transformer"), exist_ok=True)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            v = dequantize_if_quantized(v)
            if v.is_floating_point() and not k.startswith(
                MiniMaxH3Transformer.FP32_KEY_PREFIXES
            ):
                v = v.to(save_dtype)
            save_dict[k] = v.clone().to("cpu")
        meta_st = get_meta_for_safetensors(meta, name="minimax_h3")
        save_file(
            save_dict,
            os.path.join(output_path, "transformer", "model.safetensors"),
            metadata=meta_st,
        )
        with open(os.path.join(output_path, "aitk_meta.yaml"), "w") as f:
            yaml.dump(meta, f)

    def get_base_model_version(self):
        return "minimax_h3"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["blocks"]

    def get_quantization_exclude_modules(self) -> Optional[List[str]]:
        # float32 islands, the conditioning projection, the token refiner and
        # the AdaLN projections — all shipped unquantized in the pre-quantized
        # checkpoints (pruned files carry tiny fp16 adaln linears fed by the
        # 8-dim time table), so excluding them makes quantize with the
        # checkpoint's own qtype an exact no-op and keeps the sensitive
        # modulation path at full precision under any other qtype.
        return [
            "video_patch_proj*",
            "audio_patch_proj*",
            "time_embedder*",
            "final_layer*",
            "condition_proj*",
            "token_refiner*",
            "*adaln_proj*",
        ]

    def convert_lora_weights_before_save(self, state_dict):
        # ComfyUI's MiniMax-H3 keys are the original checkpoint keys, so the
        # standard diffusion_model prefix maps directly
        return {
            k.replace("transformer.", "diffusion_model."): v
            for k, v in state_dict.items()
        }

    def convert_lora_weights_before_load(self, state_dict):
        return {
            k.replace("diffusion_model.", "transformer."): v
            for k, v in state_dict.items()
        }


class MinimaxH3Ref2VAModel(MinimaxH3Model):
    """Reference-to-video (ref2va): the control images ride along as reference
    blocks in the packed sequence (plus ``<Picture i>: `` vision blocks in the
    Qwen3-VL conditioning) instead of anchoring the first frame.

    Image references only for now. Every reference keeps its OWN aspect ratio
    and is resized to the TARGET's pixel area (axes snapped to /32), then sits
    on its own aspect-normalized rotary grid — like the released ref2va's
    per-reference grids, but area-matched to the target instead of the 2048px
    reference short edge. Each reference block advances the rotary media
    clock by 1.0.

    At sampling, ctrl images are ALWAYS references, never first frames.

    ``model_kwargs.image_refs_as_video`` (default off) routes still-image
    references through the VIDEO reference path instead: the image is held
    for ``image_ref_video_frames`` frames (17n+5, default 5) as a silent
    static clip — video sizing (true area match), multi-frame latent block,
    temporal-span rotary advance, and a ``<Video k>: `` timestamped Qwen
    presentation — so a LoRA trained on image references exercises the same
    pathway that video references use at inference.
    """

    arch = "minimax_h3_ref2va"

    def _image_ref_video_frames(self) -> int:
        """Frames a still reference is held for when presented as a static
        video (0 = keep native ``<Picture>`` image references)."""
        kw = self.model_config.model_kwargs
        if not bool(kw.get("image_refs_as_video", False)):
            return 0
        return packing.align_num_frames_down(int(kw.get("image_ref_video_frames", 5)))

    @property
    def text_embedding_space_version(self):
        # the presentation of image references changes the embeds -> new cache key
        n = self._image_ref_video_frames()
        if n:
            return f"{self.arch}:img_as_vid{n}"
        return self.arch

    def _present_image_control(self, image: Image.Image):
        n = self._image_ref_video_frames()
        if n:
            return static_image_video_ref(image, n, fps=packing.FPS)
        return image

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # references arrive as a list per sample (multi_control_paths), at
        # their ORIGINAL size — this model does its own area-matched resize
        self.has_multiple_control_images = True
        self.use_raw_control_images = True
        # control VIDEOS are cached like dataset items and consumed as
        # multi-frame reference blocks
        self.supports_video_control_images = True
        # D-OPSD: a no-grad teacher pass with the target as its own reference
        # becomes the training target for the reference-free student pass
        self.dopsd = bool(self.model_config.model_kwargs.get("dopsd", False))
        if self.dopsd:
            self.dopsd_self_ref = True
            self.require_pixel_tensor_cache = True

    def _dit_component(self) -> str:
        partition = str(
            self.model_config.model_kwargs.get("partition", "ref2va_pruned")
        ).lower()
        if partition not in ("ref2va", "ref2va_pruned"):
            raise ValueError(
                f"model_kwargs.partition must be ref2va or ref2va_pruned for "
                f"{self.arch}, got {partition}"
            )
        return f"dit_{partition}"

    def get_base_model_version(self):
        return "minimax_h3_ref2va"

    def _build_condition(
        self, batch: "DataLoaderBatchDTO", latent_shape, device, dtype
    ):
        # raw-size control tensors in [0, 1]; each is resized to the TARGET's
        # pixel area with its own aspect kept, then encoded on its own grid
        if batch is None:
            return None, None, (), ()
        if getattr(batch, "dopsd_teacher_pass", False):
            return self._build_dopsd_teacher_condition(
                batch, latent_shape, device, dtype
            )
        controls_per_item = None
        if batch.control_tensor_list is not None:
            controls_per_item = batch.control_tensor_list
        elif batch.control_tensor is not None:
            controls_per_item = [
                [batch.control_tensor[b]] for b in range(batch.control_tensor.shape[0])
            ]
        if not controls_per_item:
            controls_per_item = (
                [[] for _ in batch.file_items] if batch.file_items else []
            )
            if not controls_per_item:
                return None, None, (), ()
        ref_count = len(controls_per_item[0])
        if any(len(c) != ref_count for c in controls_per_item):
            raise ValueError(
                "ref2va: every item in a batch must have the same number of "
                "reference images"
            )

        _, h_lat, w_lat = latent_shape
        target_h, target_w = h_lat * 16, w_lat * 16
        # still refs as static video clips: video sizing + multi-frame block
        as_video_frames = self._image_ref_video_frames()
        size_fn = (
            packing.reference_video_pixel_size
            if as_video_frames
            else packing.reference_pixel_size
        )

        all_rows = []
        blocks = []
        for ref_idx in range(ref_count):
            resized = []
            for c in controls_per_item:
                img = c[ref_idx]
                if img.ndim == 4:
                    img = img[0]
                ph, pw = size_fn(img.shape[2], img.shape[1], target_h, target_w)
                # LANCZOS like ComfyUI / the sampling path
                resized.append(
                    torch.nn.functional.interpolate(
                        img[None].to(device, torch.float32),
                        size=(ph, pw),
                        mode="bicubic",
                        antialias=True,
                    )[0].clamp(0.0, 1.0)
                )
            shapes = {tuple(r.shape[1:]) for r in resized}
            if len(shapes) > 1:
                raise ValueError(
                    "ref2va: reference aspect ratios must match across a batch "
                    f"(got pixel shapes {sorted(shapes)}); use batch_size 1 for "
                    "mixed-aspect references"
                )
            # [0, 1] control -> [-1, 1] pixels; (B, 3, T, H, W) with T = 1
            # for a keyframe-style image ref, or the still held for
            # ``image_ref_video_frames`` frames as a static clip
            frames = (torch.stack(resized) * 2.0 - 1.0).unsqueeze(2)
            if as_video_frames:
                frames = frames.expand(-1, -1, as_video_frames, -1, -1).contiguous()
            ref_latents = self.encode_keyframe_latents(frames)
            ref_noise = torch.randn_like(ref_latents)
            ref_latents = (
                KEYFRAME_NOISE_AUG_T * ref_latents
                + (1.0 - KEYFRAME_NOISE_AUG_T) * ref_noise
            )
            blocks.append(
                (ref_latents.shape[2], ref_latents.shape[3], ref_latents.shape[4], 0)
            )
            all_rows.append(patchify_video_latents(ref_latents).to(dtype))

        audio_rows = []
        self._append_video_ref_blocks(
            batch, all_rows, audio_rows, blocks, device, dtype, target_h, target_w
        )
        if not all_rows:
            return None, None, (), ()
        cond_audio = torch.cat(audio_rows, dim=1) if audio_rows else None
        return torch.cat(all_rows, dim=1), cond_audio, (), tuple(blocks)

    def _build_dopsd_teacher_condition(self, batch, latent_shape, device, dtype):
        """D-OPSD teacher: the target is its own (only) reference, built from
        the batch's cached pixel tensors; its soundtrack rides as clean
        reference audio rows."""
        if batch.tensor is None:
            raise ValueError(
                "D-OPSD teacher pass needs pixel tensors on the batch; set "
                "cache_tensors_to_disk: true on this dataset"
            )
        px = batch.tensor.detach().to(device, torch.float32)
        if px.ndim == 4:
            # image items: single-frame ref, or a static clip for image_refs_as_video
            frames = px.unsqueeze(2)
            as_video_frames = self._image_ref_video_frames()
            if as_video_frames:
                frames = frames.expand(-1, -1, as_video_frames, -1, -1).contiguous()
        else:
            # video items: (B, T, C, H, W) -> (B, C, T, H, W)
            frames = px.permute(0, 2, 1, 3, 4).contiguous()
        ref_latents = self.encode_keyframe_latents(frames)
        ref_noise = torch.randn_like(ref_latents)
        ref_latents = (
            KEYFRAME_NOISE_AUG_T * ref_latents
            + (1.0 - KEYFRAME_NOISE_AUG_T) * ref_noise
        )
        audio_rows = None
        a_lat = 0
        if (
            batch.dataset_config is not None
            and batch.dataset_config.do_audio
            and batch.audio_latents is not None
            and getattr(batch, "num_frames", 1) > 1
        ):
            a_lat = packing.audio_latent_num_frames(batch.num_frames)
            audio_rows = torch.stack(
                [
                    self._fit_audio_rows(
                        batch.audio_latents[b].detach().to(device, torch.float32),
                        a_lat,
                    )
                    for b in range(batch.audio_latents.shape[0])
                ]
            ).to(dtype)
        blocks = (
            (ref_latents.shape[2], ref_latents.shape[3], ref_latents.shape[4], a_lat),
        )
        return patchify_video_latents(ref_latents).to(dtype), audio_rows, (), blocks

    @torch.no_grad()
    def _encode_ref_video_for_sampling(self, path: str, gen_config) -> torch.Tensor:
        """Decode a reference video with the SAME temporal treatment training
        uses (real-time pacing from frame 0 at 24 fps, tail trimmed, snapped
        down to 17n+5, capped at the sample's frame count), area-match it to
        the target with its own aspect, and encode with the released keyframe
        recipe. Returns normalized latents (C, T, h, w)."""
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open reference video {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or packing.FPS
        n = int(total / src_fps * packing.FPS)
        n = packing.align_num_frames_down(min(n, max(gen_config.num_frames, 5)))
        indices = ref_frame_indices(total, src_fps, n, packing.FPS, trim_tail=True)
        from .src.ref_video_cache import read_frames_at

        frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in read_frames_at(cap, indices)
        ]
        cap.release()
        h0, w0 = frames[0].shape[:2]
        # match the sample canvas's pixel area, own aspect kept
        ph, pw = packing.reference_video_pixel_size(
            w0, h0, gen_config.height, gen_config.width
        )
        pixels = torch.from_numpy(np.stack(frames)).float() / 255.0 * 2.0 - 1.0
        pixels = pixels.permute(3, 0, 1, 2)[None]  # (1, 3, T, H, W)
        pixels = (
            torch.nn.functional.interpolate(
                pixels.reshape(1 * 3, len(frames), h0, w0).transpose(0, 1),
                size=(ph, pw),
                mode="bilinear",
                antialias=True,
            )
            .transpose(0, 1)
            .reshape(1, 3, len(frames), ph, pw)
        )
        generator = torch.Generator(device="cpu").manual_seed(KEYFRAME_ENCODE_SEED)
        latents = self.video_vae.encode(
            pixels.to(self.vae.device, self.video_vae.dtype),
            sample=True,
            generator=generator,
            fp16_round=True,
        )
        # soundtrack rides clean when the clip has one (same test as the TE label)
        audio_rows = None
        try:
            from .src.text_encoder import video_has_audio

            if not video_has_audio(path):
                raise RuntimeError("no audio stream")
            import torchaudio

            waveform, sample_rate = torchaudio.load(path)
            # frames cover [0, n / 24) seconds; trim the soundtrack to the
            # same window (matches the training cache) before encoding
            keep = int(round(n / packing.FPS * sample_rate))
            waveform = waveform[:, :keep]
            rows = self.encode_audio(
                [{"waveform": waveform, "sample_rate": sample_rate}]
            )[0]
            a_lat = packing.audio_latent_num_frames(n)
            audio_rows = self._fit_audio_rows(rows.float(), a_lat)
        except Exception:
            pass
        return {"latent": latents[0].float(), "audio_rows": audio_rows}

    @torch.no_grad()
    def _encode_static_image_ref_for_sampling(
        self, image: Image.Image, gen_config
    ) -> dict:
        """A still ctrl image as a silent static reference clip
        (``image_refs_as_video``): held for ``image_ref_video_frames`` frames,
        video-sized to the target's pixel area with its own aspect, encoded
        with the released keyframe recipe. Same shape of entry as
        :meth:`_encode_ref_video_for_sampling`."""
        import numpy as np

        n = self._image_ref_video_frames()
        ph, pw = packing.reference_video_pixel_size(
            image.size[0], image.size[1], gen_config.height, gen_config.width
        )
        if image.size != (pw, ph):
            image = image.resize((pw, ph), Image.Resampling.LANCZOS)
        pixels = torch.from_numpy(np.asarray(image)).float() / 255.0 * 2.0 - 1.0
        pixels = pixels.permute(2, 0, 1)[None, :, None]  # (1, 3, 1, H, W)
        pixels = pixels.expand(-1, -1, n, -1, -1).contiguous()
        latents = self.encode_keyframe_latents(pixels)
        return {"latent": latents[0].float(), "audio_rows": None}

    def _append_video_ref_blocks(
        self, batch, all_rows, audio_rows, blocks, device, dtype, target_h, target_w
    ):
        # control videos get the dataset's temporal treatment (frame count,
        # fps), are area-matched to the target, and get one VAE encode
        # disk-cached next to the video; the
        # resulting latents become multi-frame reference blocks packed after
        # the image references
        paths_per_item = getattr(batch, "control_video_paths_list", None)
        if not paths_per_item:
            return
        vid_count = len(paths_per_item[0])
        if any(len(v) != vid_count for v in paths_per_item):
            raise ValueError(
                "ref2va: every item in a batch must have the same number of "
                "reference videos"
            )
        for ref_idx in range(vid_count):
            lats = []
            auds = []
            for per_item in paths_per_item:
                entry = load_ref_video_latent(
                    self, per_item[ref_idx], batch.dataset_config, target_h, target_w
                )
                lats.append(entry["latent"].to(device, torch.float32))
                auds.append(entry.get("audio_rows"))
            shapes = {tuple(l.shape) for l in lats}
            if len(shapes) > 1:
                raise ValueError(
                    "ref2va: reference video latent shapes must match across a "
                    f"batch (got {sorted(shapes)}); use batch_size 1"
                )
            ref_latents = torch.stack(lats)  # (B, C, T, h, w)
            ref_noise = torch.randn_like(ref_latents)
            ref_latents = (
                KEYFRAME_NOISE_AUG_T * ref_latents
                + (1.0 - KEYFRAME_NOISE_AUG_T) * ref_noise
            )
            # soundtrack rows ride clean when every item in the batch has them
            a_lat = 0
            if all(a is not None for a in auds):
                a_lat = packing.audio_latent_num_frames(entry["num_frames"])
                trimmed = [
                    self._fit_audio_rows(a.to(device, torch.float32), a_lat)
                    for a in auds
                ]
                if len({t.shape for t in trimmed}) == 1:
                    audio_rows.append(torch.stack(trimmed).to(dtype))
                else:
                    a_lat = 0
            blocks.append(
                (
                    ref_latents.shape[2],
                    ref_latents.shape[3],
                    ref_latents.shape[4],
                    a_lat,
                )
            )
            all_rows.append(patchify_video_latents(ref_latents).to(dtype))

    @staticmethod
    def _fit_audio_rows(rows: torch.Tensor, a_lat: int) -> torch.Tensor:
        """Trim/pad channel-major packed rows (2*T, C) to 2*a_lat rows,
        per stereo channel."""
        t = rows.shape[0] // 2
        per_ch = rows.reshape(2, t, rows.shape[-1])
        if t > a_lat:
            per_ch = per_ch[:, :a_lat]
        elif t < a_lat:
            per_ch = torch.nn.functional.pad(per_ch, (0, 0, 0, a_lat - t))
        return per_ch.reshape(2 * a_lat, rows.shape[-1])

    def generate_single_image(
        self,
        pipeline: MiniMaxH3Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = max(sc, int(gen_config.width // sc * sc))
        gen_config.height = max(sc, int(gen_config.height // sc * sc))

        is_video = gen_config.num_frames > 1
        if is_video:
            gen_config.num_frames = packing.align_num_frames_down(gen_config.num_frames)
            gen_config.fps = packing.FPS
            gen_config.save_image = partial(new_save_image_function, gen_config)
            gen_config.log_image = partial(blank_log_image_function, gen_config)
            gen_config.output_ext = "mp4"

        # every ctrl file is a REFERENCE (never a first frame): images are
        # resized to the target's pixel area with their own aspect kept;
        # videos are dataset-style encoded into multi-frame latent blocks
        ref_images = []
        for path in (
            gen_config.ctrl_img,
            gen_config.ctrl_img_1,
            gen_config.ctrl_img_2,
            gen_config.ctrl_img_3,
        ):
            if path is None:
                continue
            if os.path.splitext(str(path))[1].lower() in packing_video_exts:
                ref_images.append(
                    self._encode_ref_video_for_sampling(str(path), gen_config)
                )
            else:
                img = Image.open(path).convert("RGB")
                if self._image_ref_video_frames():
                    ref_images.append(
                        self._encode_static_image_ref_for_sampling(img, gen_config)
                    )
                else:
                    ref_images.append(
                        packing.prepare_reference_image(
                            img, gen_config.height, gen_config.width
                        )
                    )

        with_audio = bool(self.model_config.model_kwargs.get("sample_audio", True))

        result = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_frames=gen_config.num_frames,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            ref_images=ref_images or None,
            with_audio=with_audio and is_video,
        )
        if is_video:
            return result  # dict consumed by new_save_image_function
        return result[0]
