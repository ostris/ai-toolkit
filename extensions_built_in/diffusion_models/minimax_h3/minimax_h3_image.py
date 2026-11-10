"""MiniMax-H3-Image for ai-toolkit: the MiniMax-H3 DiT without its audio
stream, distilled to a 6-step, no-CFG text-to-image and image-editing model.

What changes against the H3 video archs (everything else is inherited from
MinimaxH3Ref2VAModel -- references ride as condition blocks in the packed
sequence and as vision blocks in the Qwen3-VL conditioning):

  - DiT: the pruned layout with no audio projection / audio head; the packed
    sequence is ``[text | reference images | target image]``, one latent frame
  - output heads: 6 PDD heads distilled on the checkpoint's own sigma grid
    (``H3_IMAGE_SAMPLE_SIGMAS``) rather than the video models' shifted uniform
    grid; a sampling step blends the heads of the grid intervals it spans.
    Training draws only the 6 sampling sigmas (the scheduler's
    ``fixed_train_sigmas``), each through its own head
  - text encoder: the MiniMax-H3 Qwen3-VL-32B, unchanged; references are
    labelled ``Picture i: `` (no angle brackets -- the prompt still refers to
    them as ``<Picture i>``) and the prompt is used verbatim
  - VAE: the H3 encoder (bit-identical weights, so image latents match the
    video archs') with a single-frame 24-layer 2D ViT decoder (src/vae.py
    MiniMaxH3ImageVideoVAE)
  - references keep their own size instead of following the target: the VAE
    copy has its short edge inside 1024..2048, the Qwen copy at 1024
  - sampling: the checkpoint's 6-sigma grid, guidance scale 1

Weights are the Comfy-Org repack (``Comfy-Org/MiniMax-H3-Image``), resolved
like the other H3 archs: the precision variant matching the requested qtype,
used in place under MODELS_PATH and downloaded there only when missing.
"""

import os

import torch
from PIL import Image

from toolkit.basic import flush
from toolkit.models.v2.resolver import (
    comfy_precision_rank,
    repo_id_from_name_or_path,
    resolve_comfy_file,
)

from .minimax_h3 import MiniMaxH3VaeBundle, MinimaxH3Ref2VAModel
from .src import packing
from .src.transformer import MiniMaxH3Transformer
from .src.vae import MiniMaxH3ImageVideoVAE

H3_IMAGE_REPO = "Comfy-Org/MiniMax-H3-Image"
# precision variants per component, repo-relative (comfy layout)
H3_IMAGE_FILES = {
    "dit": [
        "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
        "diffusion_models/minimax_h3_image_pruned_bf16.safetensors",
        "diffusion_models/minimax_h3_image_pruned_w4a8.safetensors",
    ],
    "text_encoder": [
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    ],
    "video_vae": ["vae/minimax_h3_image_vae_fp16.safetensors"],
}
# the sigma grid the 6 PDD heads were distilled on, which is also the sampling grid
H3_IMAGE_SAMPLE_SIGMAS = [
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
]
# short edge of the copy of a reference shown to Qwen
H3_IMAGE_VISION_SHORT_EDGE = 1024
# short edge bounds of the copy of a reference the VAE encodes
H3_IMAGE_REF_MIN_SHORT_EDGE = 1024
H3_IMAGE_REF_MAX_SHORT_EDGE = 2048
# prompt token cap of the release
H3_IMAGE_MAX_TEXT_LENGTH = 4096


class MinimaxH3ImageModel(MinimaxH3Ref2VAModel):
    arch = "minimax_h3_image"
    comfy_repo = H3_IMAGE_REPO
    # the PDD grid is a symmetric step spacing run through flow shift 3
    video_sigma_shift = 3.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.model_config.model_kwargs
        self.supports_video_control_images = False
        # latent_space_version stays the H3 one: the encoder weights are identical
        self.max_text_length = int(kw.get("max_text_length", H3_IMAGE_MAX_TEXT_LENGTH))
        self.reference_min_short_edge = int(
            kw.get("reference_min_short_edge", H3_IMAGE_REF_MIN_SHORT_EDGE)
        )
        self.reference_max_short_edge = int(
            kw.get("reference_max_short_edge", H3_IMAGE_REF_MAX_SHORT_EDGE)
        )

    @classmethod
    def get_train_scheduler(cls):
        scheduler = super().get_train_scheduler()
        # train on the sampling schedule: each head only ever sees its own sigma
        scheduler.fixed_train_sigmas = H3_IMAGE_SAMPLE_SIGMAS[:-1]
        return scheduler

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _component_qtype(self, component: str):
        mc = self.model_config
        if component == "text_encoder":
            return mc.qtype_te if mc.quantize_te else None
        return mc.qtype if mc.quantize else None

    def _resolve_comfy_file(self, component: str) -> str:
        """Pick the precision variant that matches the requested qtype: a
        local candidate wins in rank order, else the best-ranked one is
        downloaded."""
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
        candidates = H3_IMAGE_FILES[component]
        qtype = self._component_qtype(component)

        def rank(candidate: str):
            if "w4a8" in os.path.basename(candidate):
                # only ever the convrotw4a8 backend's file
                wanted = qtype is not None and "w4a8" in qtype.lower()
                return (0 if wanted else 9, candidates.index(candidate))
            return (
                1 + comfy_precision_rank(candidate, qtype=qtype),
                candidates.index(candidate),
            )

        ordered = sorted(candidates, key=rank)
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

    def _load_transformer(self) -> MiniMaxH3Transformer:
        transformer = super()._load_transformer()
        p = transformer.params
        if p.audio_enabled or p.pdd_steps != len(H3_IMAGE_SAMPLE_SIGMAS) - 1:
            raise ValueError(
                "minimax_h3_image needs a MiniMax-H3-Image checkpoint (no audio "
                f"stream, {len(H3_IMAGE_SAMPLE_SIGMAS) - 1} PDD output heads); this "
                f"file has audio_enabled={p.audio_enabled}, pdd_steps={p.pdd_steps}."
            )
        return transformer

    def _load_vaes(self) -> MiniMaxH3VaeBundle:
        self.print_and_status_update("Loading MiniMax-H3-Image VAE")
        video_vae = MiniMaxH3ImageVideoVAE.load_model(
            self._resolve_comfy_file("video_vae")
        )
        flush()
        # image only: no audio VAE
        return MiniMaxH3VaeBundle(video_vae, None)

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def _num_audio_latents(self, num_frames: int) -> int:
        return 0

    def _prompt_encoder_kwargs(self) -> dict:
        return dict(picture_label="Picture {}: ")

    def _image_ref_video_frames(self) -> int:
        # stills only: references are always picture blocks
        return 0

    @property
    def text_embedding_space_version(self):
        return self.arch

    def _present_image_control(self, image: Image.Image):
        height, width = packing.bound_short_edge_size(
            image.size[0],
            image.size[1],
            H3_IMAGE_VISION_SHORT_EDGE,
            H3_IMAGE_VISION_SHORT_EDGE,
        )
        if image.size == (width, height):
            return image
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def _reference_pixel_size(
        self, ref_width: int, ref_height: int, target_height: int, target_width: int
    ):
        # own size, never the target's
        return packing.bound_short_edge_size(
            ref_width,
            ref_height,
            self.reference_min_short_edge,
            self.reference_max_short_edge,
        )

    def get_prompt_embeds(self, prompt, control_images=None):
        if isinstance(prompt, str):
            prompt = [prompt]
        # sample configs hand an unset negative prompt over as False
        prompt = [p if isinstance(p, str) else "" for p in prompt]
        return super().get_prompt_embeds(prompt, control_images=control_images)

    # ------------------------------------------------------------------
    # PDD heads / sampling
    # ------------------------------------------------------------------
    def _pdd_weights(self, sigma_v, sigma_v_next=None):
        weights = packing.build_pdd_grid_weights(
            sigma_v.flatten().tolist(),
            H3_IMAGE_SAMPLE_SIGMAS,
            None if sigma_v_next is None else sigma_v_next.flatten().tolist(),
        )
        return weights, weights

    def build_sample_sigmas(self, num_inference_steps: int) -> torch.Tensor:
        return packing.resample_sigma_grid(H3_IMAGE_SAMPLE_SIGMAS, num_inference_steps)

    def generate_single_image(self, pipeline, gen_config, *args, **kwargs):
        # single images only
        gen_config.num_frames = 1
        return super().generate_single_image(pipeline, gen_config, *args, **kwargs)

    def get_base_model_version(self):
        return "minimax_h3_image"
