"""P-H3-Image (PrunaAI) for ai-toolkit: MiniMax-H3 fine-tuned into an 8-step,
no-CFG text-to-image and image-editing model.

What changes against the H3 video archs (everything else is inherited from
MinimaxH3Ref2VAModel -- references ride as condition blocks in the packed
sequence and as ``<Picture i>: `` vision blocks in the Qwen conditioning):

  - DiT: the pruned fl2va layout with Pruna's LoRA merged in
    (``PrunaAI/P-H3-Image-Comfy``), 4097x16 adaln time table; one latent frame
    with its two audio latents per image
  - text encoder: Qwen3-VL-4B; hidden states [13, 19, 25, 31, 36] (the last
    one normed) stacked through a text connector (standardize -> 12800x5120
    projection -> rescale, token 0 replaced by a sink embedding) -> 5120-dim
  - VAE: the H3 encoder with a single-frame 36-layer ViT decoder (src/vae.py
    PH3ImageVideoVAE); references encode as a seed-42 posterior sample without
    the fp16 rounding of the video keyframe recipe
  - editing (ComfyUI PH3ImageConditioning): image 1 is the image to edit, up
    to 7 more are references; each is encoded at its own aspect scaled DOWN
    to the target canvas area (/32) and shown to Qwen at <= 1 MP; a short
    instruction gets ``"Edit <Picture 1>. "`` in front unless it already names
    the picture to edit (or is an upsampler brief)
  - sampling: the checkpoint's own 8-sigma grid, audio at the same sigma as
    video (ModelSamplingMiniMaxH3 1.0/1.0), guidance scale 1

Weights resolve like the H3 archs: precision variant picked by the requested
qtype (int8 ConvRot files for convrot8, bf16 otherwise), used in place under
MODELS_PATH and downloaded there only when missing. Tokenizer / processor /
text-encoder config come from the diffusers release ``PrunaAI/P-H3-Image``.
"""

import os
import re
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from toolkit.basic import flush
from toolkit.models.v2.resolver import (
    comfy_precision_rank,
    repo_id_from_name_or_path,
    resolve_comfy_file,
)
from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder
from toolkit.util.comfy_quant_import import import_comfy_quantized_layers
from toolkit.util.ostris_quant import OstrisLinear

from .minimax_h3 import MiniMaxH3VaeBundle, MinimaxH3Ref2VAModel
from .src import packing
from .src.vae import PH3ImageVideoVAE

PH3_REPO = "PrunaAI/P-H3-Image-Comfy"
# tokenizer / processor / text-encoder config (tiny files)
PH3_ORIGINAL_REPO = "PrunaAI/P-H3-Image"
# precision variants per component, repo-relative (comfy layout)
PH3_FILES = {
    "dit": [
        "diffusion_models/p-h3-image_int8_convrot.safetensors",
        "diffusion_models/p-h3-image_bf16.safetensors",
    ],
    "text_encoder": [
        "text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors",
        "text_encoders/p-h3-image_text_encoder_bf16.safetensors",
    ],
    "video_vae": [
        "vae/p-h3-image_vae_int8_convrot.safetensors",
        "vae/p-h3-image_vae_bf16.safetensors",
    ],
}
# Qwen3-VL-4B hidden states read by the text connector (36 = after the final norm)
PH3_HIDDEN_STATE_LAYERS = [13, 19, 25, 31, 36]
PH3_TEXT_DIM = 5120
# the checkpoint's own sampling grid (8 steps)
PH3_SAMPLE_SIGMAS = [
    1.0,
    0.984753,
    0.967905,
    0.951688,
    0.925372,
    0.881923,
    0.810083,
    0.528718,
    0.0,
]
# reference images are shown to Qwen at <= 1 MP
PH3_VISION_PIXELS = 1_000_000
PH3_EDIT_PREFIX = "Edit <Picture 1>. "
# an instruction that already names the picture to edit, or an upsampler brief, is used as it is
PH3_EDIT_VERBATIM = re.compile(r"\s*(Edit <Picture \d+>|subject_definitions:)")


class PH3TextConnector(torch.nn.Module):
    """Stacked hidden states (B, layers, T, C) -> (B, T, 5120): per-feature
    standardize, project, rescale; token 0 is replaced by the sink embedding.
    Computes in fp32 regardless of storage dtype (ComfyUI keeps it unquantized)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.input_mean = torch.nn.Parameter(torch.zeros(in_channels))
        self.input_std = torch.nn.Parameter(torch.ones(in_channels))
        self.proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.output_mean = torch.nn.Parameter(torch.zeros(out_channels))
        self.output_std = torch.nn.Parameter(torch.ones(out_channels))
        self.sink_embedding = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.movedim(1, 2).flatten(2).float()
        x = (x - self.input_mean.float()) / self.input_std.float()
        x = F.linear(x, self.proj.weight.float())
        x = torch.addcmul(self.output_mean.float(), x, self.output_std.float())
        sink = self.sink_embedding.float().expand(x.shape[0], 1, -1)
        return torch.cat([sink, x[:, 1:]], dim=1)


class PH3ImageModel(MinimaxH3Ref2VAModel):
    arch = "p_h3_image"
    comfy_repo = PH3_REPO
    # the 8-step grid hands both streams the same sigma: no audio remap
    audio_sigma_shift = packing.VIDEO_SIGMA_SHIFT
    # P-H3 reference encode: seeded posterior sample, no fp16 rounding
    keyframe_fp16_round = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_video_control_images = False
        self.latent_space_version = "p_h3_image_v1"
        # "Edit <Picture 1>. " in front of captions that come with references
        self.edit_prompt_prefix = bool(
            self.model_config.model_kwargs.get("edit_prompt_prefix", True)
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _component_qtype(self, component: str):
        mc = self.model_config
        if component == "text_encoder":
            return mc.qtype_te if mc.quantize_te else None
        # the VAE pairs with the DiT's precision
        return mc.qtype if mc.quantize else None

    def _resolve_comfy_file(self, component: str) -> str:
        """Pick the precision variant that matches the requested qtype
        (int8 ConvRot for convrot8, bf16 otherwise): a local candidate wins in
        rank order, else the best-ranked one is downloaded."""
        override = self.model_config.model_kwargs.get(f"{component}_path", None)
        if override is not None:
            if not os.path.exists(override):
                raise FileNotFoundError(f"{component}_path does not exist: {override}")
            return override
        name_or_path = self.model_config.name_or_path
        extra_roots = (
            [name_or_path] if name_or_path and os.path.isdir(name_or_path) else []
        )
        repo_id = repo_id_from_name_or_path(name_or_path, self.comfy_repo)
        candidates = PH3_FILES[component]
        qtype = self._component_qtype(component)
        ordered = sorted(
            candidates,
            key=lambda c: (comfy_precision_rank(c, qtype=qtype), candidates.index(c)),
        )
        for rel in ordered:
            found = resolve_comfy_file(
                rel, repo_id, extra_roots=extra_roots, local_only=True
            )
            if found is not None:
                return found
        return resolve_comfy_file(
            ordered[0],
            repo_id,
            extra_roots=extra_roots,
            status_fn=self.print_and_status_update,
        )

    def _dit_component(self) -> str:
        return "dit"

    def _load_text_encoder(self):
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            PH3_ORIGINAL_REPO, subfolder="tokenizer"
        )
        processor = AutoProcessor.from_pretrained(
            PH3_ORIGINAL_REPO, subfolder="processor"
        )

        te_path = self.model_config.te_name_or_path
        connector_file = None
        if te_path is not None and os.path.isdir(te_path):
            # diffusers layout: text_encoder/ with text_connector/ beside it
            self.print_and_status_update(
                f"Loading Qwen3-VL-4B text encoder from {te_path}"
            )
            text_encoder = Qwen3VLTextEncoder.from_pretrained(
                te_path, torch_dtype=self.te_torch_dtype
            )
            text_encoder.lm_head = None
            connector_file = os.path.join(
                os.path.dirname(te_path.rstrip("/")),
                "text_connector",
                "diffusion_pytorch_model.safetensors",
            )
            if not os.path.exists(connector_file):
                import huggingface_hub

                connector_file = huggingface_hub.hf_hub_download(
                    repo_id=PH3_ORIGINAL_REPO,
                    filename="text_connector/diffusion_pytorch_model.safetensors",
                )
            state_dict = {
                f"text_connector.{k}": v for k, v in load_file(connector_file).items()
            }
        else:
            te_file = (
                te_path
                if te_path is not None
                else self._resolve_comfy_file("text_encoder")
            )
            self.print_and_status_update(
                f"Loading Qwen3-VL-4B text encoder from {te_file}"
            )
            config = AutoConfig.from_pretrained(
                PH3_ORIGINAL_REPO, subfolder="text_encoder"
            )
            config.tie_word_embeddings = False
            with init_empty_weights():
                text_encoder = Qwen3VLTextEncoder(config)
            text_encoder.lm_head = None
            # comfy single file: transformers key layout plus text_connector.*
            state_dict = load_file(te_file)
            state_dict, num_quantized = import_comfy_quantized_layers(
                text_encoder, state_dict, orig_dtype=self.te_torch_dtype
            )
            if num_quantized:
                self.print_and_status_update(
                    f" - attached {num_quantized} pre-quantized int8 layers"
                )
        hidden = text_encoder.config.text_config.hidden_size
        with init_empty_weights():
            text_encoder.text_connector = PH3TextConnector(
                len(PH3_HIDDEN_STATE_LAYERS) * hidden, PH3_TEXT_DIM
            )
        result = text_encoder.load_state_dict(state_dict, assign=True, strict=False)
        quantized_keys = {
            f"{n}.weight"
            for n, m in text_encoder.named_modules()
            if isinstance(m, OstrisLinear)
        }
        allowed_missing_prefixes = ("lm_head", "model.language_model.embed_tokens")
        if connector_file is not None:
            # from_pretrained loaded everything but the connector
            allowed_missing_prefixes += ("model.",)
        bad_missing = [
            k
            for k in result.missing_keys
            if k not in quantized_keys and not k.startswith(allowed_missing_prefixes)
        ]
        if bad_missing or result.unexpected_keys:
            raise ValueError(
                f"P-H3-Image text encoder load mismatch: missing {bad_missing[:8]}, "
                f"unexpected {result.unexpected_keys[:8]}"
            )
        del state_dict
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()
        return tokenizer, processor, text_encoder

    def _load_vaes(self) -> MiniMaxH3VaeBundle:
        self.print_and_status_update("Loading P-H3-Image VAE")
        video_vae = PH3ImageVideoVAE.load_model(self._resolve_comfy_file("video_vae"))
        flush()
        # image only: no audio VAE
        return MiniMaxH3VaeBundle(video_vae, None)

    def component_load_kwargs(self, role: str = "transformer", dtype=None):
        kwargs = super().component_load_kwargs(role, dtype)
        if role == "te":
            # the connector is a plain fp32 module: never quantized
            kwargs["exclude_quant_modules"] = ["text_connector*"]
        return kwargs

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def _prompt_encoder_kwargs(self) -> dict:
        return dict(
            hidden_state_layers=PH3_HIDDEN_STATE_LAYERS,
            connector=self.text_encoder.text_connector,
        )

    def _image_ref_video_frames(self) -> int:
        # stills only: references are always <Picture i> blocks
        return 0

    @property
    def text_embedding_space_version(self):
        return self.arch if self.edit_prompt_prefix else f"{self.arch}:raw"

    def _present_image_control(self, image: Image.Image):
        # Qwen sees every reference at <= 1 MP, own aspect, /32
        height, width = packing.reference_pixel_size(
            image.size[0], image.size[1], 1000, PH3_VISION_PIXELS // 1000
        )
        if image.size == (width, height):
            return image
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def _edit_prompt(self, prompt, has_refs: bool) -> str:
        # sample configs hand an unset negative prompt over as False
        prompt = prompt.strip() if isinstance(prompt, str) else ""
        if (
            not has_refs
            or not self.edit_prompt_prefix
            or PH3_EDIT_VERBATIM.match(prompt)
        ):
            return prompt
        return PH3_EDIT_PREFIX + prompt

    def get_prompt_embeds(self, prompt, control_images=None):
        if isinstance(prompt, str):
            prompt = [prompt]
        has_refs = control_images is not None and (
            not isinstance(control_images, (list, tuple)) or len(control_images) > 0
        )
        prompt = [self._edit_prompt(p, has_refs) for p in prompt]
        return super().get_prompt_embeds(prompt, control_images=control_images)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def build_sample_sigmas(self, num_inference_steps: int) -> torch.Tensor:
        if num_inference_steps == len(PH3_SAMPLE_SIGMAS) - 1:
            return torch.tensor(PH3_SAMPLE_SIGMAS, dtype=torch.float32)
        return super().build_sample_sigmas(num_inference_steps)

    def generate_single_image(self, pipeline, gen_config, *args, **kwargs):
        # single images only
        gen_config.num_frames = 1
        return super().generate_single_image(pipeline, gen_config, *args, **kwargs)

    def get_base_model_version(self):
        return "p_h3_image"
