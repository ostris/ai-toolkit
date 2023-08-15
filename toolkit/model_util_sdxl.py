import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from transformers import CLIPTextModelWithProjection, CLIPTextConfig, CLIPTextModel

from library import model_util, sdxl_original_unet
from library.sdxl_model_util import convert_sdxl_text_encoder_2_checkpoint


def load_models_from_sdxl_checkpoint(model_version, ckpt_path, map_location):
    # model_version is reserved for future use

    # Load the state dict
    if model_util.is_safetensors(ckpt_path):
        checkpoint = None
        state_dict = load_file(ckpt_path, device=map_location)
        epoch = None
        global_step = None
    else:
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
        else:
            state_dict = checkpoint
            epoch = 0
            global_step = 0
        checkpoint = None

    # U-Net
    print("building U-Net")
    unet = sdxl_original_unet.SdxlUNet2DConditionModel()

    print("loading U-Net from checkpoint")
    unet_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("model.diffusion_model."):
            unet_sd[k.replace("model.diffusion_model.", "")] = state_dict.pop(k)
    info = unet.load_state_dict(unet_sd)
    print("U-Net: ", info)
    del unet_sd

    # Text Encoders
    print("building text encoders")

    # Text Encoder 1 is same to Stability AI's SDXL
    text_model1_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=768,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    text_model1 = CLIPTextModel._from_config(text_model1_cfg)

    # Text Encoder 2 is different from Stability AI's SDXL. SDXL uses open clip, but we use the model from HuggingFace.
    # Note: Tokenizer from HuggingFace is different from SDXL. We must use open clip's tokenizer.
    text_model2_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=1280,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    text_model2 = CLIPTextModelWithProjection(text_model2_cfg)

    print("loading text encoders from checkpoint")
    te1_sd = {}
    te2_sd = {}
    for k in list(state_dict.keys()):
        if k.endswith("text_model.embeddings.position_ids"):
            # skip position_ids
            state_dict.pop(k)
        elif k.startswith("conditioner.embedders.0.transformer."):
            te1_sd[k.replace("conditioner.embedders.0.transformer.", "")] = state_dict.pop(k)
        elif k.startswith("conditioner.embedders.1.model."):
            te2_sd[k] = state_dict.pop(k)



    info1 = text_model1.load_state_dict(te1_sd)
    print("text encoder 1:", info1)

    converted_sd, logit_scale = convert_sdxl_text_encoder_2_checkpoint(te2_sd, max_length=77)
    # remove text_model.embeddings.position_ids"
    converted_sd.pop("text_model.embeddings.position_ids")
    info2 = text_model2.load_state_dict(converted_sd)
    print("text encoder 2:", info2)

    # prepare vae
    print("building VAE")
    vae_config = model_util.create_vae_diffusers_config()
    vae = AutoencoderKL(**vae_config)  # .to(device)

    print("loading VAE from checkpoint")
    converted_vae_checkpoint = model_util.convert_ldm_vae_checkpoint(state_dict, vae_config)
    info = vae.load_state_dict(converted_vae_checkpoint)
    print("VAE:", info)

    ckpt_info = (epoch, global_step) if epoch is not None else None
    return text_model1, text_model2, vae, unet, logit_scale, ckpt_info
