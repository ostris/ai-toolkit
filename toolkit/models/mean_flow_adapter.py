import inspect
import weakref
import torch
from typing import TYPE_CHECKING
from toolkit.lora_special import LoRASpecialNetwork
from diffusers import FluxTransformer2DModel
from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings,
    CombinedTimestepGuidanceTextProjEmbeddings,
)
from functools import partial


if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.config_modules import AdapterConfig, TrainConfig, ModelConfig
    from toolkit.custom_adapter import CustomAdapter


def mean_flow_time_text_embed_forward(
    self: CombinedTimestepTextProjEmbeddings, timestep, pooled_projection
):
    mean_flow_adapter: "MeanFlowAdapter" = self.mean_flow_adapter_ref()
    # make zero timestep ending if none is passed
    if mean_flow_adapter.is_active and timestep.shape[0] == pooled_projection.shape[0]:
        timestep = torch.cat(
            [timestep, torch.zeros_like(timestep)], dim=0
        )  # timestep - 0 (final timestep) == same as start timestep

    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(
        timesteps_proj.to(dtype=pooled_projection.dtype)
    )  # (N, D)

    # mean flow stuff
    if mean_flow_adapter.is_active:
        # todo make sure that timesteps is batched correctly, I think diffusers expects non batched timesteps
        orig_dtype = timesteps_emb.dtype
        timesteps_emb = timesteps_emb.to(torch.float32)
        timesteps_emb_start, timesteps_emb_end = timesteps_emb.chunk(2, dim=0)
        timesteps_emb = mean_flow_adapter.mean_flow_timestep_embedder(
            torch.cat([timesteps_emb_start, timesteps_emb_end], dim=-1)
        )
        timesteps_emb = timesteps_emb.to(orig_dtype)

    pooled_projections = self.text_embedder(pooled_projection)

    conditioning = timesteps_emb + pooled_projections

    return conditioning


def mean_flow_time_text_guidance_embed_forward(
    self: CombinedTimestepGuidanceTextProjEmbeddings,
    timestep,
    guidance,
    pooled_projection,
):
    mean_flow_adapter: "MeanFlowAdapter" = self.mean_flow_adapter_ref()
    # make zero timestep ending if none is passed
    if mean_flow_adapter.is_active and timestep.shape[0] == pooled_projection.shape[0]:
        timestep = torch.cat(
            [timestep, torch.zeros_like(timestep)], dim=0
        )  # timestep - 0 (final timestep) == same as start timestep
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(
        timesteps_proj.to(dtype=pooled_projection.dtype)
    )  # (N, D)

    guidance_proj = self.time_proj(guidance)
    guidance_emb = self.guidance_embedder(
        guidance_proj.to(dtype=pooled_projection.dtype)
    )  # (N, D)

    # mean flow stuff
    if mean_flow_adapter.is_active:
        # todo make sure that timesteps is batched correctly, I think diffusers expects non batched timesteps
        orig_dtype = timesteps_emb.dtype
        timesteps_emb = timesteps_emb.to(torch.float32)
        timesteps_emb_start, timesteps_emb_end = timesteps_emb.chunk(2, dim=0)
        timesteps_emb = mean_flow_adapter.mean_flow_timestep_embedder(
            torch.cat([timesteps_emb_start, timesteps_emb_end], dim=-1)
        )
        timesteps_emb = timesteps_emb.to(orig_dtype)

    time_guidance_emb = timesteps_emb + guidance_emb

    pooled_projections = self.text_embedder(pooled_projection)
    conditioning = time_guidance_emb + pooled_projections

    return conditioning


def convert_flux_to_mean_flow(
    transformer: FluxTransformer2DModel,
):
    if isinstance(transformer.time_text_embed, CombinedTimestepTextProjEmbeddings):
        transformer.time_text_embed.forward = partial(
            mean_flow_time_text_embed_forward, transformer.time_text_embed
        )
    elif isinstance(
        transformer.time_text_embed, CombinedTimestepGuidanceTextProjEmbeddings
    ):
        transformer.time_text_embed.forward = partial(
            mean_flow_time_text_guidance_embed_forward, transformer.time_text_embed
        )
    else:
        raise ValueError(
            "Unsupported time_text_embed type: {}".format(
                type(transformer.time_text_embed)
            )
        )


class MeanFlowAdapter(torch.nn.Module):
    def __init__(
        self,
        adapter: "CustomAdapter",
        sd: "StableDiffusion",
        config: "AdapterConfig",
        train_config: "TrainConfig",
    ):
        super().__init__()
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.sd_ref = weakref.ref(sd)
        self.model_config: ModelConfig = sd.model_config
        self.network_config = config.lora_config
        self.train_config = train_config
        self.device_torch = sd.device_torch
        self.lora = None

        if self.network_config is not None:
            network_kwargs = (
                {}
                if self.network_config.network_kwargs is None
                else self.network_config.network_kwargs
            )
            if hasattr(sd, "target_lora_modules"):
                network_kwargs["target_lin_modules"] = sd.target_lora_modules

            if "ignore_if_contains" not in network_kwargs:
                network_kwargs["ignore_if_contains"] = []

            self.lora = LoRASpecialNetwork(
                text_encoder=sd.text_encoder,
                unet=sd.unet,
                lora_dim=self.network_config.linear,
                multiplier=1.0,
                alpha=self.network_config.linear_alpha,
                train_unet=self.train_config.train_unet,
                train_text_encoder=self.train_config.train_text_encoder,
                conv_lora_dim=self.network_config.conv,
                conv_alpha=self.network_config.conv_alpha,
                is_sdxl=self.model_config.is_xl or self.model_config.is_ssd,
                is_v2=self.model_config.is_v2,
                is_v3=self.model_config.is_v3,
                is_pixart=self.model_config.is_pixart,
                is_auraflow=self.model_config.is_auraflow,
                is_flux=self.model_config.is_flux,
                is_lumina2=self.model_config.is_lumina2,
                is_ssd=self.model_config.is_ssd,
                is_vega=self.model_config.is_vega,
                dropout=self.network_config.dropout,
                use_text_encoder_1=self.model_config.use_text_encoder_1,
                use_text_encoder_2=self.model_config.use_text_encoder_2,
                use_bias=False,
                is_lorm=False,
                network_config=self.network_config,
                network_type=self.network_config.type,
                transformer_only=self.network_config.transformer_only,
                is_transformer=sd.is_transformer,
                base_model=sd,
                **network_kwargs,
            )
            self.lora.force_to(self.device_torch, dtype=torch.float32)
            self.lora._update_torch_multiplier()
            self.lora.apply_to(
                sd.text_encoder,
                sd.unet,
                self.train_config.train_text_encoder,
                self.train_config.train_unet,
            )
            self.lora.can_merge_in = False
            self.lora.prepare_grad_etc(sd.text_encoder, sd.unet)
            if self.train_config.gradient_checkpointing:
                self.lora.enable_gradient_checkpointing()

        emb_dim = None
        if self.model_config.arch in ["flux", "flex2", "flex2"]:
            transformer: FluxTransformer2DModel = sd.unet
            emb_dim = (
                transformer.config.num_attention_heads
                * transformer.config.attention_head_dim
            )
            convert_flux_to_mean_flow(transformer)
        else:
            raise ValueError(f"Unsupported architecture: {self.model_config.arch}")

        self.mean_flow_timestep_embedder = torch.nn.Linear(
            emb_dim * 2,
            emb_dim,
        )
        
        # make the model function as before adding this adapter by initializing the weights
        with torch.no_grad():
            self.mean_flow_timestep_embedder.weight.zero_()
            self.mean_flow_timestep_embedder.weight[:, :emb_dim] = torch.eye(emb_dim)
            self.mean_flow_timestep_embedder.bias.zero_()

        self.mean_flow_timestep_embedder.to(self.device_torch)

        # add our adapter as a weak ref
        if self.model_config.arch in ["flux", "flex2", "flex2"]:
            sd.unet.time_text_embed.mean_flow_adapter_ref = weakref.ref(self)

    def get_params(self):
        if self.lora is not None:
            config = {
                "text_encoder_lr": self.train_config.lr,
                "unet_lr": self.train_config.lr,
            }
            sig = inspect.signature(self.lora.prepare_optimizer_params)
            if "default_lr" in sig.parameters:
                config["default_lr"] = self.train_config.lr
            if "learning_rate" in sig.parameters:
                config["learning_rate"] = self.train_config.lr
            params_net = self.lora.prepare_optimizer_params(**config)

            # we want only tensors here
            params = []
            for p in params_net:
                if isinstance(p, dict):
                    params += p["params"]
                elif isinstance(p, torch.Tensor):
                    params.append(p)
                elif isinstance(p, list):
                    params += p
        else:
            params = []

        # make sure the embedder is float32
        self.mean_flow_timestep_embedder.to(torch.float32)
        self.mean_flow_timestep_embedder.requires_grad = True
        self.mean_flow_timestep_embedder.train()

        params += list(self.mean_flow_timestep_embedder.parameters())

        # we need to be able to yield from the list like yield from params

        return params

    def load_weights(self, state_dict, strict=True):
        lora_sd = {}
        mean_flow_embedder_sd = {}
        for key, value in state_dict.items():
            if "mean_flow_timestep_embedder" in key:
                new_key = key.replace("transformer.mean_flow_timestep_embedder.", "")
                mean_flow_embedder_sd[new_key] = value
            else:
                lora_sd[key] = value

        # todo process state dict before loading for models that need it
        if self.lora is not None:
            self.lora.load_weights(lora_sd)
        self.mean_flow_timestep_embedder.load_state_dict(
            mean_flow_embedder_sd, strict=False
        )

    def get_state_dict(self):
        if self.lora is not None:
            lora_sd = self.lora.get_state_dict(dtype=torch.float32)
        else:
            lora_sd = {}
        # todo make sure we match loras elseware.
        mean_flow_embedder_sd = self.mean_flow_timestep_embedder.state_dict()
        for key, value in mean_flow_embedder_sd.items():
            lora_sd[f"transformer.mean_flow_timestep_embedder.{key}"] = value
        return lora_sd

    @property
    def is_active(self):
        return self.adapter_ref().is_active
