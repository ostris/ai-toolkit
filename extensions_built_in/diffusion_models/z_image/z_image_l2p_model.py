from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
import os
from typing import Dict, List, Optional, Union

import huggingface_hub
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from toolkit.basic import flush
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager

from transformers import AutoTokenizer, Qwen3ForCausalLM
from toolkit.models.FakeVAE import FakeVAE
from toolkit.paths import MODELS_PATH
from safetensors.torch import load_file, save_file
from toolkit.metadata import get_meta_for_safetensors

HF_TOKEN = os.getenv("HF_TOKEN", None)

try:
    from diffusers import ZImagePipeline
    from diffusers.models.transformers.transformer_z_image import (
        ZImageTransformer2DModel as ZImageTransformer2DModelOriginal,
    )
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
except ImportError:
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )


# Default ZImage transformer config used when loading from a single safetensors
# file (no config.json available alongside).
ZIMAGE_DEFAULT_CONFIG = {
    "_class_name": "ZImageTransformer2DModel",
    "_diffusers_version": "0.37.0.dev0",
    "all_f_patch_size": [1],
    "all_patch_size": [2],
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
    "cap_feat_dim": 2560,
    "dim": 3840,
    "in_channels": 16,
    "n_heads": 30,
    "n_kv_heads": 30,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "norm_eps": 1e-05,
    "qk_norm": True,
    "rope_theta": 256.0,
    "siglip_feat_dim": None,
    "t_scale": 1000.0,
}


class MicroDiffusionModel(nn.Module):
    """L2P pixel-space decoder: a small 4-stage U-Net that fuses the transformer
    feature map at the bottleneck and outputs pixel-space prediction."""

    def __init__(self, in_channels: int, si_t_hidden_size: int):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.SiLU()
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.SiLU()
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.SiLU()
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.SiLU()
        )
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + si_t_hidden_size, 512, kernel_size=1),
            nn.SiLU(),
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1), nn.SiLU()
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1), nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1), nn.SiLU()
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.SiLU()
        )
        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        if c.shape[-2:] != p4.shape[-2:]:
            c = F.interpolate(c, size=p4.shape[-2:], mode="nearest")
        b = self.bottleneck(torch.cat([p4, c.to(p4.dtype)], dim=1))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


class ZImageTransformer2DModel(ZImageTransformer2DModelOriginal):
    """L2P-style ZImage transformer: runs the standard trunk but replaces the
    FinalLayer + unpatchify tail with a pixel-space U-Net decoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FinalLayer is unused in L2P — the pixel-space U-Net does the decoding.
        # Removing it keeps the module out of state_dict/named_parameters.
        if hasattr(self, "all_final_layer"):
            del self.all_final_layer
        self.local_decoder = MicroDiffusionModel(
            in_channels=self.in_channels,
            si_t_hidden_size=self.dim,
        )

    def forward(
        self,
        x: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        t,
        cap_feats: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        return_dict: bool = True,
        controlnet_block_samples: Optional[Dict[int, torch.Tensor]] = None,
        siglip_feats: Optional[List[List[torch.Tensor]]] = None,
        image_noise_mask: Optional[List[List[int]]] = None,
        patch_size: int = 16,
        f_patch_size: int = 1,
    ):
        assert (
            patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        )
        assert not isinstance(x[0], list), "L2P does not support omni mode"
        device = x[0].device

        # Capture original noisy pixel images for the U-Net decoder.
        noisy_images = torch.stack(x, dim=0)
        if noisy_images.dim() == 5:
            noisy_images = noisy_images.squeeze(2)
        bsz, _, H_ori, W_ori = noisy_images.shape

        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])

        (
            x_patches,
            cap_feats_proc,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_pad_mask,
            cap_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # X embed & refine
        x_seqlens = [len(xi) for xi in x_patches]
        x_embed = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](
            torch.cat(x_patches, dim=0)
        )
        x_embed, x_freqs, x_mask, _, _ = self._prepare_sequence(
            list(x_embed.split(x_seqlens, dim=0)),
            x_pos_ids,
            x_pad_mask,
            self.x_pad_token,
            None,
            device,
        )
        for layer in self.noise_refiner:
            x_embed = (
                self._gradient_checkpointing_func(
                    layer, x_embed, x_mask, x_freqs, adaln_input, None, None, None
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(x_embed, x_mask, x_freqs, adaln_input, None, None, None)
            )

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_feats_proc]
        cap_embed = self.cap_embedder(torch.cat(cap_feats_proc, dim=0))
        cap_embed, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_embed.split(cap_seqlens, dim=0)),
            cap_pos_ids,
            cap_pad_mask,
            self.cap_pad_token,
            None,
            device,
        )
        for layer in self.context_refiner:
            cap_embed = (
                self._gradient_checkpointing_func(layer, cap_embed, cap_mask, cap_freqs)
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(cap_embed, cap_mask, cap_freqs)
            )

        # Unified sequence: basic mode = [x, cap]
        unified, unified_freqs, unified_mask, _ = self._build_unified_sequence(
            x_embed,
            x_freqs,
            x_seqlens,
            None,
            cap_embed,
            cap_freqs,
            cap_seqlens,
            None,
            None,
            None,
            None,
            None,
            False,
            device,
        )

        for layer_idx, layer in enumerate(self.layers):
            unified = (
                self._gradient_checkpointing_func(
                    layer,
                    unified,
                    unified_mask,
                    unified_freqs,
                    adaln_input,
                    None,
                    None,
                    None,
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(
                    unified, unified_mask, unified_freqs, adaln_input, None, None, None
                )
            )
            if (
                controlnet_block_samples is not None
                and layer_idx in controlnet_block_samples
            ):
                unified = unified + controlnet_block_samples[layer_idx]

        # L2P tail: extract image tokens, reshape to (B, dim, H/p, W/p), decode in pixel space.
        feat_H = H_ori // patch_size
        feat_W = W_ori // patch_size
        img_token_len = feat_H * feat_W
        img_features = unified[:, :img_token_len, :]
        feat_map = (
            img_features.reshape(bsz, feat_H, feat_W, self.dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        decoded = self.local_decoder(noisy_images, feat_map)
        decoded = decoded.unsqueeze(2)  # add F=1 axis to match (C, F, H, W) downstream
        x_out = list(decoded.unbind(0))

        return (x_out,) if not return_dict else Transformer2DModelOutput(sample=x_out)


class ZImageL2PModel(ZImageModel):
    arch = "zimage_l2p"

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading ZImage model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        # If model_path looks like "<repo_org>/<repo_name>/.../file.safetensors"
        # and isn't a local file or directory, resolve it from HF Hub. Cache the
        # downloaded file under MODELS_PATH/diffusion_models and reuse it on
        # subsequent runs.
        if (
            not os.path.isfile(model_path)
            and not os.path.isdir(model_path)
            and model_path.endswith(".safetensors")
            and model_path.count("/") >= 2
        ):
            repo_id, filename = model_path.rsplit("/", 1)
            target_dir = os.path.join(MODELS_PATH, "diffusion_models")
            target_path = os.path.join(target_dir, filename)
            if os.path.isfile(target_path):
                self.print_and_status_update(f"Using cached weights at {target_path}")
                model_path = target_path
            else:
                os.makedirs(target_dir, exist_ok=True)
                self.print_and_status_update(f"Downloading {filename} from {repo_id}")
                model_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=target_dir,
                    token=HF_TOKEN,
                )

        self.print_and_status_update("Loading transformer")

        transformer_path = model_path
        transformer_subfolder = "transformer"

        if os.path.isfile(model_path):
            # Local single-file checkpoint (e.g. a .safetensors merge).
            # No sidecar config.json — build the architecture from the hardcoded
            # ZImage default config (or a pixel-space variant if the checkpoint
            # already contains L2P / pixel-space keys) and overwrite weights.
            # Pull the rest of the pipeline (tokenizer, text encoder) from the
            # official Z-Image-Turbo repo since we have no local sidecar config.
            self.print_and_status_update(
                f"Loading transformer weights from {model_path}"
            )
            sd = load_file(model_path)
            sd = {k: v.to(dtype) for k, v in sd.items()}

            # Detect a pixel-space (L2P) checkpoint by the presence of either
            # the pixel-size x_embedder or local_decoder keys. Also infer
            # in_channels from the local_decoder shape so we match whatever
            # the checkpoint actually was trained with.
            has_l2p_keys = any(k.startswith("local_decoder.") for k in sd)
            has_pixel_xemb = "all_x_embedder.16-1.weight" in sd
            is_pixel = has_l2p_keys or has_pixel_xemb

            inferred_in_channels = None
            if "local_decoder.enc1.0.weight" in sd:
                inferred_in_channels = sd["local_decoder.enc1.0.weight"].shape[1]
            elif has_pixel_xemb:
                inferred_in_channels = 3

            config = dict(ZIMAGE_DEFAULT_CONFIG)
            if is_pixel:
                config["in_channels"] = (
                    inferred_in_channels if inferred_in_channels else 3
                )
                config["all_patch_size"] = [16]
                self.print_and_status_update(
                    f"  detected pixel-space checkpoint (L2P), in_channels={config['in_channels']}"
                )

            # Strip ConfigMixin metadata before passing to the constructor.
            init_args = {k: v for k, v in config.items() if not k.startswith("_")}
            transformer = ZImageTransformer2DModel(**init_args)
            transformer = transformer.to(dtype)
            self.print_and_status_update(
                f"  built transformer: in_channels={transformer.in_channels}, "
                f"all_patch_size={transformer.all_patch_size}"
            )

            missing, unexpected = transformer.load_state_dict(sd, strict=False)
            if unexpected:
                self.print_and_status_update(
                    f"  {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})"
                )
            if missing:
                self.print_and_status_update(
                    f"  {len(missing)} missing keys kept at init (e.g. {missing[:3]})"
                )
            del sd
            flush()
        else:
            if os.path.exists(transformer_path):
                transformer_subfolder = None
                transformer_path = os.path.join(transformer_path, "transformer")
                # check if the path is a full checkpoint.
                te_folder_path = os.path.join(model_path, "text_encoder")
                # if we have the te, this folder is a full checkpoint, use it as the base
                if os.path.exists(te_folder_path):
                    base_model_path = model_path

            transformer = ZImageTransformer2DModel.from_pretrained(
                transformer_path, subfolder=transformer_subfolder, torch_dtype=dtype
            )
        # convert it to pixel space if needed
        if transformer.config.in_channels != 3:
            self.print_and_status_update("Converting transformer to pixel space")
            old_transformer = transformer
            new_config = dict(transformer.config)
            new_config["in_channels"] = 3
            new_config["all_patch_size"] = [16]
            transformer = ZImageTransformer2DModel.from_config(new_config)

            # update the state dict
            old_transformer_state = old_transformer.state_dict()
            new_transformer_state_dict = {}
            for k, v in old_transformer_state.items():
                if k == "all_x_embedder.2-1.weight":
                    new_v = torch.randn(
                        v.shape[0],
                        768,
                        dtype=v.dtype,
                        device=v.device,
                    )
                    new_transformer_state_dict["all_x_embedder.16-1.weight"] = (
                        new_v * 0.001
                    )
                elif k.startswith("all_final_layer."):
                    # FinalLayer is unused in L2P; pixel decoding is done by local_decoder.
                    continue
                else:
                    new_transformer_state_dict[k] = v

            # local_decoder.* keys are absent from the source checkpoint; they keep
            # the random init from MicroDiffusionModel.__init__ via strict=False.
            transformer.load_state_dict(new_transformer_state_dict, strict=False)
            del old_transformer
            del old_transformer_state
            del new_transformer_state_dict
            flush()

        # load assistant lora if specified
        if self.model_config.assistant_lora_path is not None:
            self.load_training_adapter(transformer)
            # set qtype to be float8 if it is qfloat8
            if self.model_config.qtype == "qfloat8":
                self.model_config.qtype = "float8"

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
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
                ignore_modules=[
                    transformer.x_pad_token,
                    transformer.cap_pad_token,
                ],
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype
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

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading VAE")
        # vae = AutoencoderKL.from_pretrained(
        #     base_model_path, subfolder="vae", torch_dtype=dtype
        # )
        vae = FakeVAE(scaling_factor=1.0)

        self.noise_scheduler = ZImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {}

        pipe: ZImagePipeline = ZImagePipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        # leave it on cpu for now
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # just to make sure everything is on the right device and dtype
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def save_model(self, output_path, meta, save_dtype):
        transformer: ZImageTransformer2DModel = unwrap_model(self.model)
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        meta = get_meta_for_safetensors(meta, name=self.arch)

        sd = transformer.state_dict()
        save_dict = {}
        for key, value in sd.items():
            # Skip the unused FinalLayer — L2P bypasses it in forward(), so
            # the weights are dead and just bloat the checkpoint.
            if key.startswith("all_final_layer."):
                continue
            save_dict[key] = value.to("cpu").to(save_dtype)
        save_file(save_dict, output_path, metadata=meta)
