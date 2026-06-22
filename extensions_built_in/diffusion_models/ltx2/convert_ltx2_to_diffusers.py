# ref https://github.com/huggingface/diffusers/blob/17b53f08661732caca6a546295950fc4b1696ad7/scripts/convert_ltx2_to_diffusers.py

from contextlib import nullcontext
from typing import Any, Dict, Tuple

import torch
from accelerate import init_empty_weights

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2TextConnectors, LTX2Vocoder, LTX2VocoderWithBWE
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available() else nullcontext


LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT = {
    # Input Patchify Projections
    "patchify_proj": "proj_in",
    "audio_patchify_proj": "audio_proj_in",
    # Modulation Parameters
    # Handle adaln_single --> time_embed, audioln_single --> audio_time_embed separately as the original keys are
    # substrings of the other modulation parameters below
    "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
    "av_ca_a2v_gate_adaln_single": "av_cross_attn_video_a2v_gate",
    "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
    "av_ca_v2a_gate_adaln_single": "av_cross_attn_audio_v2a_gate",
    # Transformer Blocks
    # Per-Block Cross Attention Modulatin Parameters
    "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
    "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",
    # Attention QK Norms
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT = {
    **LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT,
    "audio_prompt_adaln_single": "audio_prompt_adaln",
    "prompt_adaln_single": "prompt_adaln",
}

LTX_2_0_VIDEO_VAE_RENAME_DICT = {
    # Encoder
    "down_blocks.0": "down_blocks.0",
    "down_blocks.1": "down_blocks.0.downsamplers.0",
    "down_blocks.2": "down_blocks.1",
    "down_blocks.3": "down_blocks.1.downsamplers.0",
    "down_blocks.4": "down_blocks.2",
    "down_blocks.5": "down_blocks.2.downsamplers.0",
    "down_blocks.6": "down_blocks.3",
    "down_blocks.7": "down_blocks.3.downsamplers.0",
    "down_blocks.8": "mid_block",
    # Decoder
    "up_blocks.0": "mid_block",
    "up_blocks.1": "up_blocks.0.upsamplers.0",
    "up_blocks.2": "up_blocks.0",
    "up_blocks.3": "up_blocks.1.upsamplers.0",
    "up_blocks.4": "up_blocks.1",
    "up_blocks.5": "up_blocks.2.upsamplers.0",
    "up_blocks.6": "up_blocks.2",
    "last_time_embedder": "time_embedder",
    "last_scale_shift_table": "scale_shift_table",
    # Common
    # For all 3D ResNets
    "res_blocks": "resnets",
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

LTX_2_3_VIDEO_VAE_RENAME_DICT = {
    **LTX_2_0_VIDEO_VAE_RENAME_DICT,
    # Decoder extra blocks
    "up_blocks.7": "up_blocks.3.upsamplers.0",
    "up_blocks.8": "up_blocks.3",
}

LTX_2_0_AUDIO_VAE_RENAME_DICT = {
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

LTX_2_0_VOCODER_RENAME_DICT = {
    "ups": "upsamplers",
    "resblocks": "resnets",
    "conv_pre": "conv_in",
    "conv_post": "conv_out",
}

LTX_2_3_VOCODER_RENAME_DICT = {
    # Handle upsamplers ("ups" --> "upsamplers") due to name clash
    "resblocks": "resnets",
    "conv_pre": "conv_in",
    "conv_post": "conv_out",
    "act_post": "act_out",
    "downsample.lowpass": "downsample",
}

LTX_2_0_CONNECTORS_KEYS_RENAME_DICT = {
    "connectors.": "",
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
    "transformer_1d_blocks": "transformer_blocks",
    "text_embedding_projection.aggregate_embed": "text_proj_in",
    # Attention QK Norms
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

LTX_2_3_CONNECTORS_KEYS_RENAME_DICT = {
    "connectors.": "",
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
    "transformer_1d_blocks": "transformer_blocks",
    # LTX-2.3 uses per-modality embedding projections
    "text_embedding_projection.audio_aggregate_embed": "audio_text_proj_in",
    "text_embedding_projection.video_aggregate_embed": "video_text_proj_in",
    # Attention QK Norms
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}


def update_state_dict_inplace(
    state_dict: Dict[str, Any], old_key: str, new_key: str
) -> None:
    state_dict[new_key] = state_dict.pop(old_key)


def remove_keys_inplace(key: str, state_dict: Dict[str, Any]) -> None:
    state_dict.pop(key)


def convert_ltx2_transformer_adaln_single(key: str, state_dict: Dict[str, Any]) -> None:
    # Skip if not a weight, bias
    if ".weight" not in key and ".bias" not in key:
        return

    if key.startswith("adaln_single."):
        new_key = key.replace("adaln_single.", "time_embed.")
        param = state_dict.pop(key)
        state_dict[new_key] = param

    if key.startswith("audio_adaln_single."):
        new_key = key.replace("audio_adaln_single.", "audio_time_embed.")
        param = state_dict.pop(key)
        state_dict[new_key] = param

    return


def convert_ltx2_audio_vae_per_channel_statistics(
    key: str, state_dict: Dict[str, Any]
) -> None:
    if key.startswith("per_channel_statistics"):
        new_key = ".".join(["decoder", key])
        param = state_dict.pop(key)
        state_dict[new_key] = param

    return


def convert_ltx2_3_vocoder_upsamplers(key: str, state_dict: dict[str, Any]) -> None:
    # Skip if not a weight, bias
    if ".weight" not in key and ".bias" not in key:
        return

    if ".ups." in key:
        new_key = key.replace(".ups.", ".upsamplers.")
        param = state_dict.pop(key)
        state_dict[new_key] = param
    return


LTX_2_0_TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "video_embeddings_connector": remove_keys_inplace,
    "audio_embeddings_connector": remove_keys_inplace,
    "adaln_single": convert_ltx2_transformer_adaln_single,
}

LTX_2_0_VAE_SPECIAL_KEYS_REMAP = {
    "per_channel_statistics.channel": remove_keys_inplace,
    "per_channel_statistics.mean-of-stds": remove_keys_inplace,
}

LTX_2_0_AUDIO_VAE_SPECIAL_KEYS_REMAP = {}

LTX_2_0_VOCODER_SPECIAL_KEYS_REMAP = {}

LTX_2_3_VOCODER_SPECIAL_KEYS_REMAP = {
    ".ups.": convert_ltx2_3_vocoder_upsamplers,
}

LTX_2_0_CONNECTORS_SPECIAL_KEYS_REMAP = {}


def split_transformer_and_connector_state_dict(
    state_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    connector_prefixes = (
        "video_embeddings_connector",
        "audio_embeddings_connector",
        "transformer_1d_blocks",
        "text_embedding_projection",
        "connectors.",
        "video_connector",
        "audio_connector",
        "text_proj_in",
    )

    transformer_state_dict, connector_state_dict = {}, {}
    for key, value in state_dict.items():
        if key.startswith(connector_prefixes):
            connector_state_dict[key] = value
        else:
            transformer_state_dict[key] = value

    return transformer_state_dict, connector_state_dict


def get_ltx2_transformer_config(
    version: str = "2.0",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if version == "test":
        # Produces a transformer of the same size as used in test_models_transformer_ltx2.py
        config = {
            "model_id": "diffusers-internal-dev/dummy-ltx2",
            "diffusers_config": {
                "in_channels": 4,
                "out_channels": 4,
                "patch_size": 1,
                "patch_size_t": 1,
                "num_attention_heads": 2,
                "attention_head_dim": 8,
                "cross_attention_dim": 16,
                "vae_scale_factors": (8, 32, 32),
                "pos_embed_max_pos": 20,
                "base_height": 2048,
                "base_width": 2048,
                "audio_in_channels": 4,
                "audio_out_channels": 4,
                "audio_patch_size": 1,
                "audio_patch_size_t": 1,
                "audio_num_attention_heads": 2,
                "audio_attention_head_dim": 4,
                "audio_cross_attention_dim": 8,
                "audio_scale_factor": 4,
                "audio_pos_embed_max_pos": 20,
                "audio_sampling_rate": 16000,
                "audio_hop_length": 160,
                "num_layers": 2,
                "activation_fn": "gelu-approximate",
                "qk_norm": "rms_norm_across_heads",
                "norm_elementwise_affine": False,
                "norm_eps": 1e-6,
                "caption_channels": 16,
                "attention_bias": True,
                "attention_out_bias": True,
                "rope_theta": 10000.0,
                "rope_double_precision": False,
                "causal_offset": 1,
                "timestep_scale_multiplier": 1000,
                "cross_attn_timestep_scale_multiplier": 1,
            },
        }
        rename_dict = LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT
        special_keys_remap = LTX_2_0_TRANSFORMER_SPECIAL_KEYS_REMAP
    elif version == "2.0":
        config = {
            "model_id": "Lightricks/LTX-2",
            "diffusers_config": {
                "in_channels": 128,
                "out_channels": 128,
                "patch_size": 1,
                "patch_size_t": 1,
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "vae_scale_factors": (8, 32, 32),
                "pos_embed_max_pos": 20,
                "base_height": 2048,
                "base_width": 2048,
                "gated_attn": False,
                "cross_attn_mod": False,
                "audio_in_channels": 128,
                "audio_out_channels": 128,
                "audio_patch_size": 1,
                "audio_patch_size_t": 1,
                "audio_num_attention_heads": 32,
                "audio_attention_head_dim": 64,
                "audio_cross_attention_dim": 2048,
                "audio_scale_factor": 4,
                "audio_pos_embed_max_pos": 20,
                "audio_sampling_rate": 16000,
                "audio_hop_length": 160,
                "audio_gated_attn": False,
                "audio_cross_attn_mod": False,
                "num_layers": 48,
                "activation_fn": "gelu-approximate",
                "qk_norm": "rms_norm_across_heads",
                "norm_elementwise_affine": False,
                "norm_eps": 1e-6,
                "caption_channels": 3840,
                "attention_bias": True,
                "attention_out_bias": True,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_offset": 1,
                "timestep_scale_multiplier": 1000,
                "cross_attn_timestep_scale_multiplier": 1000,
                "rope_type": "split",
                "use_prompt_embeddings": True,
                "perturbed_attn": False,
            },
        }
        rename_dict = LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT
        special_keys_remap = LTX_2_0_TRANSFORMER_SPECIAL_KEYS_REMAP
    elif version == "2.3":
        config = {
            "model_id": "Lightricks/LTX-2.3",
            "diffusers_config": {
                "in_channels": 128,
                "out_channels": 128,
                "patch_size": 1,
                "patch_size_t": 1,
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "vae_scale_factors": (8, 32, 32),
                "pos_embed_max_pos": 20,
                "base_height": 2048,
                "base_width": 2048,
                "gated_attn": True,
                "cross_attn_mod": True,
                "audio_in_channels": 128,
                "audio_out_channels": 128,
                "audio_patch_size": 1,
                "audio_patch_size_t": 1,
                "audio_num_attention_heads": 32,
                "audio_attention_head_dim": 64,
                "audio_cross_attention_dim": 2048,
                "audio_scale_factor": 4,
                "audio_pos_embed_max_pos": 20,
                "audio_sampling_rate": 16000,
                "audio_hop_length": 160,
                "audio_gated_attn": True,
                "audio_cross_attn_mod": True,
                "num_layers": 48,
                "activation_fn": "gelu-approximate",
                "qk_norm": "rms_norm_across_heads",
                "norm_elementwise_affine": False,
                "norm_eps": 1e-6,
                "caption_channels": 3840,
                "attention_bias": True,
                "attention_out_bias": True,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_offset": 1,
                "timestep_scale_multiplier": 1000,
                "cross_attn_timestep_scale_multiplier": 1000,
                "rope_type": "split",
                "use_prompt_embeddings": False,
                "perturbed_attn": True,
            },
        }
        rename_dict = LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT
        special_keys_remap = LTX_2_0_TRANSFORMER_SPECIAL_KEYS_REMAP
    return config, rename_dict, special_keys_remap


def get_ltx2_connectors_config(
    version: str = "2.0",
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if version == "test":
        config = {
            "model_id": "diffusers-internal-dev/dummy-ltx2",
            "diffusers_config": {
                "caption_channels": 16,
                "text_proj_in_factor": 3,
                "video_connector_num_attention_heads": 4,
                "video_connector_attention_head_dim": 8,
                "video_connector_num_layers": 1,
                "video_connector_num_learnable_registers": None,
                "audio_connector_num_attention_heads": 4,
                "audio_connector_attention_head_dim": 8,
                "audio_connector_num_layers": 1,
                "audio_connector_num_learnable_registers": None,
                "connector_rope_base_seq_len": 32,
                "rope_theta": 10000.0,
                "rope_double_precision": False,
                "causal_temporal_positioning": False,
            },
        }
    elif version == "2.0":
        config = {
            "model_id": "Lightricks/LTX-2",
            "diffusers_config": {
                "caption_channels": 3840,
                "text_proj_in_factor": 49,
                "video_connector_num_attention_heads": 30,
                "video_connector_attention_head_dim": 128,
                "video_connector_num_layers": 2,
                "video_connector_num_learnable_registers": 128,
                "video_gated_attn": False,
                "audio_connector_num_attention_heads": 30,
                "audio_connector_attention_head_dim": 128,
                "audio_connector_num_layers": 2,
                "audio_connector_num_learnable_registers": 128,
                "audio_gated_attn": False,
                "connector_rope_base_seq_len": 4096,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_temporal_positioning": False,
                "rope_type": "split",
                "per_modality_projections": False,
                "proj_bias": False,
            },
        }
        rename_dict = LTX_2_0_CONNECTORS_KEYS_RENAME_DICT
        special_keys_remap = LTX_2_0_CONNECTORS_SPECIAL_KEYS_REMAP
    elif version == "2.3":
        config = {
            "model_id": "Lightricks/LTX-2.3",
            "diffusers_config": {
                "caption_channels": 3840,
                "text_proj_in_factor": 49,
                "video_connector_num_attention_heads": 32,
                "video_connector_attention_head_dim": 128,
                "video_connector_num_layers": 8,
                "video_connector_num_learnable_registers": 128,
                "video_gated_attn": True,
                "audio_connector_num_attention_heads": 32,
                "audio_connector_attention_head_dim": 64,
                "audio_connector_num_layers": 8,
                "audio_connector_num_learnable_registers": 128,
                "audio_gated_attn": True,
                "connector_rope_base_seq_len": 4096,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_temporal_positioning": False,
                "rope_type": "split",
                "per_modality_projections": True,
                "video_hidden_dim": 4096,
                "audio_hidden_dim": 2048,
                "proj_bias": True,
            },
        }
        rename_dict = LTX_2_3_CONNECTORS_KEYS_RENAME_DICT
        special_keys_remap = LTX_2_0_CONNECTORS_SPECIAL_KEYS_REMAP

    return config, rename_dict, special_keys_remap


def convert_ltx2_transformer(
    original_state_dict: Dict[str, Any], version: str = "2.0"
) -> Dict[str, Any]:
    config, rename_dict, special_keys_remap = get_ltx2_transformer_config(version)
    diffusers_config = config["diffusers_config"]

    transformer_state_dict, _ = split_transformer_and_connector_state_dict(
        original_state_dict
    )

    with init_empty_weights():
        transformer = LTX2VideoTransformer3DModel.from_config(diffusers_config)

    # Handle official code --> diffusers key remapping via the remap dict
    for key in list(transformer_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(transformer_state_dict, key, new_key)

    # Handle any special logic which can't be expressed by a simple 1:1 remapping with the handlers in
    # special_keys_remap
    for key in list(transformer_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, transformer_state_dict)

    transformer.load_state_dict(transformer_state_dict, strict=True, assign=True)
    return transformer


def convert_ltx2_connectors(
    original_state_dict: Dict[str, Any], version: str = "2.0"
) -> LTX2TextConnectors:
    config, rename_dict, special_keys_remap = get_ltx2_connectors_config(version)
    diffusers_config = config["diffusers_config"]

    _, connector_state_dict = split_transformer_and_connector_state_dict(
        original_state_dict
    )
    if len(connector_state_dict) == 0:
        raise ValueError("No connector weights found in the provided state dict.")

    with init_empty_weights():
        connectors = LTX2TextConnectors.from_config(diffusers_config)

    for key in list(connector_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(connector_state_dict, key, new_key)

    for key in list(connector_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, connector_state_dict)

    connectors.load_state_dict(connector_state_dict, strict=True, assign=True)
    return connectors


def get_ltx2_video_vae_config(
    version: str = "2.0", timestep_conditioning: bool = False
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if version == "test":
        config = {
            "model_id": "diffusers-internal-dev/dummy-ltx2",
            "diffusers_config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "block_out_channels": (256, 512, 1024, 2048),
                "down_block_types": (
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                ),
                "decoder_block_out_channels": (256, 512, 1024),
                "layers_per_block": (4, 6, 6, 2, 2),
                "decoder_layers_per_block": (5, 5, 5, 5),
                "spatio_temporal_scaling": (True, True, True, True),
                "decoder_spatio_temporal_scaling": (True, True, True),
                "decoder_inject_noise": (False, False, False, False),
                "downsample_type": (
                    "spatial",
                    "temporal",
                    "spatiotemporal",
                    "spatiotemporal",
                ),
                "upsample_residual": (True, True, True),
                "upsample_factor": (2, 2, 2),
                "timestep_conditioning": timestep_conditioning,
                "patch_size": 4,
                "patch_size_t": 1,
                "resnet_norm_eps": 1e-6,
                "encoder_causal": True,
                "decoder_causal": False,
                "encoder_spatial_padding_mode": "zeros",
                "decoder_spatial_padding_mode": "reflect",
                "spatial_compression_ratio": 32,
                "temporal_compression_ratio": 8,
            },
        }
        rename_dict = LTX_2_0_VIDEO_VAE_RENAME_DICT
        special_keys_remap = LTX_2_0_VAE_SPECIAL_KEYS_REMAP
    elif version == "2.0":
        config = {
            "model_id": "Lightricks/LTX-2",
            "diffusers_config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "block_out_channels": (256, 512, 1024, 2048),
                "down_block_types": (
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                ),
                "decoder_block_out_channels": (256, 512, 1024),
                "layers_per_block": (4, 6, 6, 2, 2),
                "decoder_layers_per_block": (5, 5, 5, 5),
                "spatio_temporal_scaling": (True, True, True, True),
                "decoder_spatio_temporal_scaling": (True, True, True),
                "decoder_inject_noise": (False, False, False, False),
                "downsample_type": (
                    "spatial",
                    "temporal",
                    "spatiotemporal",
                    "spatiotemporal",
                ),
                "upsample_type": ("spatiotemporal", "spatiotemporal", "spatiotemporal"),
                "upsample_residual": (True, True, True),
                "upsample_factor": (2, 2, 2),
                "timestep_conditioning": timestep_conditioning,
                "patch_size": 4,
                "patch_size_t": 1,
                "resnet_norm_eps": 1e-6,
                "encoder_causal": True,
                "decoder_causal": False,
                "encoder_spatial_padding_mode": "zeros",
                "decoder_spatial_padding_mode": "reflect",
                "spatial_compression_ratio": 32,
                "temporal_compression_ratio": 8,
            },
        }
        rename_dict = LTX_2_0_VIDEO_VAE_RENAME_DICT
        special_keys_remap = LTX_2_0_VAE_SPECIAL_KEYS_REMAP
    elif version == "2.3":
        config = {
            "model_id": "Lightricks/LTX-2.3",
            "diffusers_config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "block_out_channels": (256, 512, 1024, 1024),
                "down_block_types": (
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                ),
                "decoder_block_out_channels": (256, 512, 512, 1024),
                "layers_per_block": (4, 6, 4, 2, 2),
                "decoder_layers_per_block": (4, 6, 4, 2, 2),
                "spatio_temporal_scaling": (True, True, True, True),
                "decoder_spatio_temporal_scaling": (True, True, True, True),
                "decoder_inject_noise": (False, False, False, False, False),
                "downsample_type": (
                    "spatial",
                    "temporal",
                    "spatiotemporal",
                    "spatiotemporal",
                ),
                "upsample_type": (
                    "spatiotemporal",
                    "spatiotemporal",
                    "temporal",
                    "spatial",
                ),
                "upsample_residual": (False, False, False, False),
                "upsample_factor": (2, 2, 1, 2),
                "timestep_conditioning": timestep_conditioning,
                "patch_size": 4,
                "patch_size_t": 1,
                "resnet_norm_eps": 1e-6,
                "encoder_causal": True,
                "decoder_causal": False,
                "encoder_spatial_padding_mode": "zeros",
                "decoder_spatial_padding_mode": "zeros",
                "spatial_compression_ratio": 32,
                "temporal_compression_ratio": 8,
            },
        }
        rename_dict = LTX_2_3_VIDEO_VAE_RENAME_DICT
        special_keys_remap = LTX_2_0_VAE_SPECIAL_KEYS_REMAP
    return config, rename_dict, special_keys_remap


def convert_ltx2_video_vae(
    original_state_dict: Dict[str, Any],
    version: str = "2.0",
    timestep_conditioning: bool = False,
) -> Dict[str, Any]:
    config, rename_dict, special_keys_remap = get_ltx2_video_vae_config(
        version, timestep_conditioning
    )
    diffusers_config = config["diffusers_config"]

    with init_empty_weights():
        vae = AutoencoderKLLTX2Video.from_config(diffusers_config)

    # Handle official code --> diffusers key remapping via the remap dict
    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    # Handle any special logic which can't be expressed by a simple 1:1 remapping with the handlers in
    # special_keys_remap
    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vae.load_state_dict(original_state_dict, strict=True, assign=True)
    return vae


def get_ltx2_audio_vae_config(
    version: str = "2.0",
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if version == "2.0":
        config = {
            "model_id": "Lightricks/LTX-2",
            "diffusers_config": {
                "base_channels": 128,
                "output_channels": 2,
                "ch_mult": (1, 2, 4),
                "num_res_blocks": 2,
                "attn_resolutions": None,
                "in_channels": 2,
                "resolution": 256,
                "latent_channels": 8,
                "norm_type": "pixel",
                "causality_axis": "height",
                "dropout": 0.0,
                "mid_block_add_attention": False,
                "sample_rate": 16000,
                "mel_hop_length": 160,
                "is_causal": True,
                "mel_bins": 64,
                "double_z": True,
            },
        }
        rename_dict = LTX_2_0_AUDIO_VAE_RENAME_DICT
        special_keys_remap = LTX_2_0_AUDIO_VAE_SPECIAL_KEYS_REMAP
    elif version == "2.3":
        config = {
            "model_id": "Lightricks/LTX-2.3",
            "diffusers_config": {
                "base_channels": 128,
                "output_channels": 2,
                "ch_mult": (1, 2, 4),
                "num_res_blocks": 2,
                "attn_resolutions": None,
                "in_channels": 2,
                "resolution": 256,
                "latent_channels": 8,
                "norm_type": "pixel",
                "causality_axis": "height",
                "dropout": 0.0,
                "mid_block_add_attention": False,
                "sample_rate": 16000,
                "mel_hop_length": 160,
                "is_causal": True,
                "mel_bins": 64,
                "double_z": True,
            },  # Same config as LTX-2.0
        }
        rename_dict = LTX_2_0_AUDIO_VAE_RENAME_DICT
        special_keys_remap = LTX_2_0_AUDIO_VAE_SPECIAL_KEYS_REMAP
    return config, rename_dict, special_keys_remap


def convert_ltx2_audio_vae(
    original_state_dict: Dict[str, Any], version: str = "2.0"
) -> Dict[str, Any]:
    config, rename_dict, special_keys_remap = get_ltx2_audio_vae_config(version)
    diffusers_config = config["diffusers_config"]

    with init_empty_weights():
        vae = AutoencoderKLLTX2Audio.from_config(diffusers_config)

    # Handle official code --> diffusers key remapping via the remap dict
    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    # Handle any special logic which can't be expressed by a simple 1:1 remapping with the handlers in
    # special_keys_remap
    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vae.load_state_dict(original_state_dict, strict=True, assign=True)
    return vae


def get_ltx2_vocoder_config(
    version: str = "2.0",
) -> tuple[Dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version == "2.0":
        config = {
            "model_id": "Lightricks/LTX-2",
            "diffusers_config": {
                "in_channels": 128,
                "hidden_channels": 1024,
                "out_channels": 2,
                "upsample_kernel_sizes": [16, 15, 8, 4, 4],
                "upsample_factors": [6, 5, 2, 2, 2],
                "resnet_kernel_sizes": [3, 7, 11],
                "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "act_fn": "leaky_relu",
                "leaky_relu_negative_slope": 0.1,
                "antialias": False,
                "final_act_fn": "tanh",
                "final_bias": True,
                "output_sampling_rate": 24000,
            },
        }
        rename_dict = LTX_2_0_VOCODER_RENAME_DICT
        special_keys_remap = LTX_2_0_VOCODER_SPECIAL_KEYS_REMAP
    elif version == "2.3":
        config = {
            "model_id": "Lightricks/LTX-2.3",
            "diffusers_config": {
                "in_channels": 128,
                "hidden_channels": 1536,
                "out_channels": 2,
                "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
                "upsample_factors": [5, 2, 2, 2, 2, 2],
                "resnet_kernel_sizes": [3, 7, 11],
                "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "act_fn": "snakebeta",
                "leaky_relu_negative_slope": 0.1,
                "antialias": True,
                "antialias_ratio": 2,
                "antialias_kernel_size": 12,
                "final_act_fn": None,
                "final_bias": False,
                "bwe_in_channels": 128,
                "bwe_hidden_channels": 512,
                "bwe_out_channels": 2,
                "bwe_upsample_kernel_sizes": [12, 11, 4, 4, 4],
                "bwe_upsample_factors": [6, 5, 2, 2, 2],
                "bwe_resnet_kernel_sizes": [3, 7, 11],
                "bwe_resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "bwe_act_fn": "snakebeta",
                "bwe_leaky_relu_negative_slope": 0.1,
                "bwe_antialias": True,
                "bwe_antialias_ratio": 2,
                "bwe_antialias_kernel_size": 12,
                "bwe_final_act_fn": None,
                "bwe_final_bias": False,
                "filter_length": 512,
                "hop_length": 80,
                "window_length": 512,
                "num_mel_channels": 64,
                "input_sampling_rate": 16000,
                "output_sampling_rate": 48000,
            },
        }
        rename_dict = LTX_2_3_VOCODER_RENAME_DICT
        special_keys_remap = LTX_2_3_VOCODER_SPECIAL_KEYS_REMAP
    return config, rename_dict, special_keys_remap


def convert_ltx2_vocoder(
    original_state_dict: Dict[str, Any], version: str
) -> Dict[str, Any]:
    config, rename_dict, special_keys_remap = get_ltx2_vocoder_config(version)
    diffusers_config = config["diffusers_config"]
    if version == "2.3":
        vocoder_cls = LTX2VocoderWithBWE
    else:
        vocoder_cls = LTX2Vocoder

    with init_empty_weights():
        vocoder = vocoder_cls.from_config(diffusers_config)

    # Handle official code --> diffusers key remapping via the remap dict
    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    # Handle any special logic which can't be expressed by a simple 1:1 remapping with the handlers in
    # special_keys_remap
    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vocoder.load_state_dict(original_state_dict, strict=True, assign=True)
    return vocoder


def get_model_state_dict_from_combined_ckpt(
    combined_ckpt: Dict[str, Any], prefix: str
) -> Dict[str, Any]:
    # Ensure that the key prefix ends with a dot (.)
    if not prefix.endswith("."):
        prefix = prefix + "."

    model_state_dict = {}
    for param_name, param in combined_ckpt.items():
        if param_name.startswith(prefix):
            model_state_dict[param_name.removeprefix(prefix)] = param

    if prefix == "model.diffusion_model.":
        # Some checkpoints store the text connector projection outside the diffusion model prefix.
        connector_prefixes = ["text_embedding_projection"]
        for param_name, param in combined_ckpt.items():
            for prefix in connector_prefixes:
                if param_name.startswith(prefix):
                    # Check to make sure we're not overwriting an existing key
                    if param_name not in model_state_dict:
                        model_state_dict[param_name] = combined_ckpt[param_name]

    return model_state_dict


def dequantize_state_dict(state_dict: Dict[str, Any]):
    keys = list(state_dict.keys())
    state_out = {}
    for k in keys:
        if k.endswith(
            (".weight_scale", ".weight_scale_2", ".pre_quant_scale", ".input_scale")
        ):
            continue

        t = state_dict[k]

        if k.endswith(".weight"):
            prefix = k[: -len(".weight")]
            wscale_k = prefix + ".weight_scale"
            if wscale_k in state_dict:
                w_q = t
                w_scale = state_dict[wscale_k]
                # Comfy quant = absmax per-tensor weight quant, nothing fancy
                w_bf16 = w_q.to(torch.bfloat16) * w_scale.to(torch.bfloat16)
                state_out[k] = w_bf16
                continue

        state_out[k] = t
    return state_out


def convert_comfy_gemma3_to_transformers(sd: dict):
    out = {}

    sd = dequantize_state_dict(sd)

    for k, v in sd.items():
        nk = k

        # Vision tower weights: checkpoint has "vision_model.*"
        # model expects "model.vision_tower.vision_model.*"
        if k.startswith("vision_model."):
            nk = "model.vision_tower." + k

        # MM projector: checkpoint has "multi_modal_projector.*"
        # model expects "model.multi_modal_projector.*"
        elif k.startswith("multi_modal_projector."):
            nk = "model." + k

        # Language model: checkpoint has "model.layers.*", "model.embed_tokens.*", "model.norm.*"
        # model expects "model.language_model.layers.*", etc.
        elif k == "model.embed_tokens.weight":
            nk = "model.language_model.embed_tokens.weight"
        elif k.startswith("model.layers."):
            nk = "model.language_model.layers." + k[len("model.layers.") :]
        elif k.startswith("model.norm."):
            nk = "model.language_model.norm." + k[len("model.norm.") :]

        # (optional) common DDP prefix
        if nk.startswith("module."):
            nk = nk[len("module.") :]

        # skip spiece_model
        if nk == "spiece_model":
            continue

        out[nk] = v

    # If lm_head is missing but embeddings exist, many Gemma-family models tie these weights.
    # Add it so strict loading won't complain (or just load strict=False and call tie_weights()).
    if (
        "lm_head.weight" not in out
        and "model.language_model.embed_tokens.weight" in out
    ):
        out["lm_head.weight"] = out["model.language_model.embed_tokens.weight"]

    return out


def convert_lora_original_to_diffusers(
    lora_state_dict: Dict[str, Any],
    version: str = "2.0",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rename_dict = LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT
    if version == "2.3":
        rename_dict = LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT

    for k, v in lora_state_dict.items():
        # Keep the "diffusion_model." prefix as-is, but apply the transformer remaps to the rest
        prefix = ""
        rest = k
        if rest.startswith("diffusion_model."):
            prefix = "diffusion_model."
            rest = rest[len(prefix) :]

        nk = rest

        # Same simple 1:1 remaps as the transformer
        for replace_key, rename_key in rename_dict.items():
            nk = nk.replace(replace_key, rename_key)

        # Same special-case remap as the transformer (applies to LoRA keys too)
        if nk.startswith("adaln_single."):
            nk = nk.replace("adaln_single.", "time_embed.", 1)
        elif nk.startswith("audio_adaln_single."):
            nk = nk.replace("audio_adaln_single.", "audio_time_embed.", 1)

        out[prefix + nk] = v

    return out


def convert_lora_diffusers_to_original(
    lora_state_dict: Dict[str, Any],
    version: str = "2.0",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rename_dict = LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT
    if version == "2.3":
        rename_dict = LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT

    inv_rename = {v: k for k, v in rename_dict.items()}
    inv_items = sorted(inv_rename.items(), key=lambda kv: len(kv[0]), reverse=True)

    for k, v in lora_state_dict.items():
        # Keep the "diffusion_model." prefix as-is, but invert remaps on the rest
        prefix = ""
        rest = k
        if rest.startswith("diffusion_model."):
            prefix = "diffusion_model."
            rest = rest[len(prefix) :]

        nk = rest

        # Inverse of the adaln_single special-case
        if nk.startswith("time_embed."):
            nk = nk.replace("time_embed.", "adaln_single.", 1)
        elif nk.startswith("audio_time_embed."):
            nk = nk.replace("audio_time_embed.", "audio_adaln_single.", 1)

        # Inverse 1:1 remaps
        for diffusers_key, original_key in inv_items:
            nk = nk.replace(diffusers_key, original_key)

        out[prefix + nk] = v

    return out
