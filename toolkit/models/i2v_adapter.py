from functools import partial
import inspect
import weakref
import torch
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.lora_special import LoRASpecialNetwork
from diffusers import WanTransformer3DModel
from transformers import SiglipImageProcessor, SiglipVisionModel, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_wan import WanImageEmbedding, WanTimeTextImageEmbedding
from toolkit.util.shuffle import shuffle_tensor_along_axis
import torch.nn.functional as F

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel
    from toolkit.config_modules import AdapterConfig, TrainConfig, ModelConfig
    from toolkit.custom_adapter import CustomAdapter

    
class FrameEmbedder(torch.nn.Module):
    def __init__(
        self,
        adapter: 'I2VAdapter',
        orig_layer: torch.nn.Conv3d,
        in_channels=20, # wan is 16 normally, and 36 with i2v so 20 new channels
    ):
        super().__init__()
        # goes through a conv patch embedding first and is then flattened
        # hidden_states = self.patch_embedding(hidden_states)
        # hidden_states = hidden_states.flatten(2).transpose(1, 2)
        
        inner_dim = orig_layer.out_channels
        patch_size = adapter.sd_ref().model.config.patch_size
        
        self.patch_embedding = torch.nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.orig_layer_ref: weakref.ref = weakref.ref(orig_layer)

    @classmethod
    def from_model(
        cls,
        model: WanTransformer3DModel,
        adapter: 'I2VAdapter',
    ):
        if model.__class__.__name__ == 'WanTransformer3DModel':
            new_channels = 20 # wan is 16 normally, and 36 with i2v so 20 new channels

            orig_patch_embedding: torch.nn.Conv3d = model.patch_embedding
            img_embedder = cls(
                adapter,
                orig_layer=orig_patch_embedding,
                in_channels=new_channels,
            )

            # hijack the forward method
            orig_patch_embedding._orig_i2v_adapter_forward = orig_patch_embedding.forward
            orig_patch_embedding.forward = img_embedder.forward

            # update the config of the transformer, only needed when merged in
            # model.config.in_channels = model.config.in_channels + new_channels
            # model.config["in_channels"] = model.config.in_channels + new_channels

            return img_embedder
        else:
            raise ValueError("Model not supported")

    @property
    def is_active(self):
        return self.adapter_ref().is_active

    def forward(self, x):
        if not self.is_active:
            # make sure lora is not active
            if self.adapter_ref().control_lora is not None:
                self.adapter_ref().control_lora.is_active = False
            
            if x.shape[1] > self.orig_layer_ref().in_channels:
                # we have i2v, so we need to remove the extra channels
                x = x[:, :self.orig_layer_ref().in_channels, :, :, :]
            return self.orig_layer_ref()._orig_i2v_adapter_forward(x)

        # make sure lora is active
        if self.adapter_ref().control_lora is not None:
            self.adapter_ref().control_lora.is_active = True
            
        # x is arranged channels cat(orig_input = 16, temporal_conditioning_mask = 4, encoded_first_frame=16)
        # (16 + 4 + 16) = 36 channels
        # (batch_size, 36, num_frames, latent_height, latent_width)

        orig_device = x.device
        orig_dtype = x.dtype
        
        orig_in = x[:, :16, :, :, :]
        orig_out = self.orig_layer_ref()._orig_i2v_adapter_forward(orig_in)
        
        # remove original stuff
        x = x[:, 16:, :, :, :]

        x = x.to(self.patch_embedding.weight.device, dtype=self.patch_embedding.weight.dtype)

        x = self.patch_embedding(x)
        
        x = x.to(orig_device, dtype=orig_dtype)
        
        # add the original out
        x = x + orig_out
        return x


def deactivatable_forward(
    self: 'Attention',
    *args,
    **kwargs
):
    if self._attn_hog_ref() is not None and self._attn_hog_ref().is_active:
        self.added_kv_proj_dim = None
        self.add_k_proj = self._add_k_proj
        self.add_v_proj = self._add_v_proj
        self.norm_added_q = self._norm_added_q
        self.norm_added_k = self._norm_added_k
    else:
        self.added_kv_proj_dim = self._attn_hog_ref().added_kv_proj_dim
        self.add_k_proj = None
        self.add_v_proj = None
        self.norm_added_q = None
        self.norm_added_k = None
    return self._orig_forward(*args, **kwargs)


class AttentionHog(torch.nn.Module):
    def __init__(
        self,
        added_kv_proj_dim: int,
        adapter: 'I2VAdapter',
        attn_layer: Attention,
        model: 'WanTransformer3DModel',
    ):
        super().__init__()

        # To prevent circular import.
        from diffusers.models.normalization import FP32LayerNorm, LpNorm, RMSNorm

        self.added_kv_proj_dim = added_kv_proj_dim
        self.attn_layer_ref: weakref.ref = weakref.ref(attn_layer)
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.model_ref: weakref.ref = weakref.ref(model)

        qk_norm = model.config.qk_norm
        
        # layers
        self.add_k_proj = torch.nn.Linear(
            added_kv_proj_dim,
            attn_layer.inner_kv_dim,
            bias=attn_layer.added_proj_bias
        )
        self.add_k_proj.weight.data = self.add_k_proj.weight.data * 0.001
        self.add_v_proj = torch.nn.Linear(
            added_kv_proj_dim,
            attn_layer.inner_kv_dim,
            bias=attn_layer.added_proj_bias
        )
        self.add_v_proj.weight.data = self.add_v_proj.weight.data * 0.001

        # do qk norm. It isnt stored in the class, but we can infer it from the attn layer
        self.norm_added_q = None
        self.norm_added_k = None

        if attn_layer.norm_q is not None:
            eps: float = 1e-5
            if qk_norm == "layer_norm":
                self.norm_added_q = torch.nn.LayerNorm(
                    attn_layer.norm_q.normalized_shape, eps=eps, elementwise_affine=attn_layer.norm_q.elementwise_affine)
                self.norm_added_k = torch.nn.LayerNorm(
                    attn_layer.norm_k.normalized_shape, eps=eps, elementwise_affine=attn_layer.norm_k.elementwise_affine)
            elif qk_norm == "fp32_layer_norm":
                self.norm_added_q = FP32LayerNorm(
                    attn_layer.norm_q.normalized_shape, elementwise_affine=False, bias=False, eps=eps)
                self.norm_added_k = FP32LayerNorm(
                    attn_layer.norm_k.normalized_shape, elementwise_affine=False, bias=False, eps=eps)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(attn_layer.norm_q.dim, eps=eps)
                self.norm_added_k = RMSNorm(attn_layer.norm_k.dim, eps=eps)
            elif qk_norm == "rms_norm_across_heads":
                # Wanx applies qk norm across all heads
                self.norm_added_q = RMSNorm(attn_layer.norm_q.dim, eps=eps)
                self.norm_added_k = RMSNorm(attn_layer.norm_k.dim, eps=eps)
            else:
                raise ValueError(
                    f"unknown qk_norm: {qk_norm}. Should be one of `None,'layer_norm','fp32_layer_norm','rms_norm'`"
                )

        # add these to the attn later in a way they can be deactivated
        attn_layer._add_k_proj = self.add_k_proj
        attn_layer._add_v_proj = self.add_v_proj
        attn_layer._norm_added_q = self.norm_added_q
        attn_layer._norm_added_k = self.norm_added_k

        # make it deactivateable
        attn_layer._attn_hog_ref = weakref.ref(self)
        attn_layer._orig_forward = attn_layer.forward
        attn_layer.forward = partial(deactivatable_forward, attn_layer)

    def forward(self, *args, **kwargs):
        if not self.adapter_ref().is_active:
            return self.attn_module(*args, **kwargs)

        # TODO implement this
        raise NotImplementedError("Attention hog not implemented")

    def is_active(self):
        return self.adapter_ref().is_active


def new_wan_forward(
    self: WanTransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    # prevent circular import
    from toolkit.models.wan21.wan_utils import add_first_frame_conditioning
    adapter:'I2VAdapter' = self._i2v_adapter_ref()
    
    if adapter.is_active:
        # activate the condition embedder
        self.condition_embedder.image_embedder = adapter.image_embedder
        
        # for wan they are putting the image emcoder embeds on the unconditional
        # this needs to be fixed as that wont work. For now, we will will use the embeds we have in order
        # we cache an conditional and an unconditional embed. On sampling, it samples conditional first,
        # then unconditional. So we just need to keep track of which one we are using. This is a horrible hack
        # TODO find a not stupid way to do this. 
        
        if adapter.adapter_ref().is_sampling:
            if not hasattr(self, '_do_unconditional'):
                # set it to true so we alternate to false immediatly
                self._do_unconditional = True
            
            # alternate it
            self._do_unconditional = not self._do_unconditional
            if self._do_unconditional:
                # slightly reduce strength of conditional for the unconditional
                # encoder_hidden_states_image = adapter.adapter_ref().conditional_embeds * 0.5
                # shuffle the embedding tokens so we still have all the information, but it is scrambled
                # this will prevent things like color from being cfg overweights, but still sharpen content. 
                
                encoder_hidden_states_image = shuffle_tensor_along_axis(
                    adapter.adapter_ref().conditional_embeds, 
                    axis=1
                )
                # encoder_hidden_states_image = adapter.adapter_ref().unconditional_embeds
            else:
                # use the conditional
                encoder_hidden_states_image = adapter.adapter_ref().conditional_embeds
        else:
            # doing a normal training run, always use conditional embeds
            encoder_hidden_states_image = adapter.adapter_ref().conditional_embeds
        
        # add the first frame conditioning
        if adapter.frame_embedder is not None:
            with torch.no_grad():
                # add the first frame conditioning
                conditioning_frame = adapter.adapter_ref().cached_control_image_0_1
                if conditioning_frame is None:
                    raise ValueError("No conditioning frame found")

                # make it -1 to 1
                conditioning_frame = (conditioning_frame * 2) - 1
                conditioning_frame = conditioning_frame.to(
                    hidden_states.device, dtype=hidden_states.dtype
                )
                    
                # if doing a full denoise, the latent input may be full channels here, only get first 16
                if hidden_states.shape[1] > 16:
                    hidden_states = hidden_states[:, :16, :, :, :]
                
                
                hidden_states = add_first_frame_conditioning(
                    latent_model_input=hidden_states,
                    first_frame=conditioning_frame,
                    vae=adapter.adapter_ref().sd_ref().vae,
                )
    else:
        # not active deactivate the condition embedder
        self.condition_embedder.image_embedder = None
    
    return self._orig_i2v_adapter_forward(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_image=encoder_hidden_states_image,
        return_dict=return_dict,
        attention_kwargs=attention_kwargs,
    )
    

class I2VAdapter(torch.nn.Module):
    def __init__(
        self,
        adapter: 'CustomAdapter',
        sd: 'BaseModel',
        config: 'AdapterConfig',
        train_config: 'TrainConfig',
        image_processor: Union[SiglipImageProcessor, CLIPImageProcessor],
        vision_encoder: Union[SiglipVisionModel, CLIPVisionModelWithProjection],
    ):
        super().__init__()
        # avoid circular import
        from toolkit.models.wan21.wan_attn import WanAttnProcessor2_0
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.sd_ref = weakref.ref(sd)
        self.model_config: ModelConfig = sd.model_config
        self.network_config = config.lora_config
        self.train_config = train_config
        self.config = config
        self.device_torch = sd.device_torch
        self.control_lora = None
        self.image_processor_ref: weakref.ref = weakref.ref(image_processor)
        self.vision_encoder_ref: weakref.ref = weakref.ref(vision_encoder)
        
        ve_img_size = vision_encoder.config.image_size
        ve_patch_size = vision_encoder.config.patch_size
        num_patches = (ve_img_size // ve_patch_size) ** 2
        num_vision_tokens = num_patches
        
        # siglip does not have a class token
        if not vision_encoder.__class__.__name__.lower().startswith("siglip"):
            num_vision_tokens = num_patches + 1

        model_class = sd.model.__class__.__name__

        if self.network_config is not None:

            network_kwargs = {} if self.network_config.network_kwargs is None else self.network_config.network_kwargs
            if hasattr(sd, 'target_lora_modules'):
                network_kwargs['target_lin_modules'] = sd.target_lora_modules

            if 'ignore_if_contains' not in network_kwargs:
                network_kwargs['ignore_if_contains'] = []

            network_kwargs['ignore_if_contains'] += [
                'add_k_proj',
                'add_v_proj',
                'norm_added_q',
                'norm_added_k',
            ]
            if model_class == 'WanTransformer3DModel':
                # always ignore patch_embedding
                network_kwargs['ignore_if_contains'].append('patch_embedding')

            self.control_lora = LoRASpecialNetwork(
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
                **network_kwargs
            )
            self.control_lora.force_to(self.device_torch, dtype=torch.float32)
            self.control_lora._update_torch_multiplier()
            self.control_lora.apply_to(
                sd.text_encoder,
                sd.unet,
                self.train_config.train_text_encoder,
                self.train_config.train_unet
            )
            self.control_lora.can_merge_in = False
            self.control_lora.prepare_grad_etc(sd.text_encoder, sd.unet)
            if self.train_config.gradient_checkpointing:
                self.control_lora.enable_gradient_checkpointing()

        self.frame_embedder: FrameEmbedder = None
        if self.config.i2v_do_start_frame:
            self.frame_embedder = FrameEmbedder.from_model(
                sd.unet,
                self
            )
            self.frame_embedder.to(self.device_torch)

        # hijack the blocks so we can inject our vision encoder
        attn_hog_list = []
        if model_class == 'WanTransformer3DModel':
            added_kv_proj_dim = sd.model.config.num_attention_heads * sd.model.config.attention_head_dim
            # update the model so it can accept the new input
            # wan has i2v with clip-h for i2v, additional k v attn that directly takes
            # in the penultimate_hidden_states from the vision encoder
            # the kv is on blocks[0].attn2
            sd.model.config.added_kv_proj_dim = added_kv_proj_dim
            sd.model.config['added_kv_proj_dim'] = added_kv_proj_dim

            transformer: WanTransformer3DModel = sd.model
            for block in transformer.blocks:
                block.attn2.added_kv_proj_dim = added_kv_proj_dim
                attn_module = AttentionHog(
                    added_kv_proj_dim,
                    self,
                    block.attn2,
                    transformer
                )
                # set the attn function to ours that handles custom number of vision tokens
                block.attn2.set_processor(WanAttnProcessor2_0(num_vision_tokens))
                
                attn_hog_list.append(attn_module)
        else:
            raise ValueError(f"Model {model_class} not supported")

        self.attn_hog_list = torch.nn.ModuleList(attn_hog_list)
        self.attn_hog_list.to(self.device_torch)
        
        inner_dim = sd.model.config.num_attention_heads * sd.model.config.attention_head_dim
        image_embed_dim = vision_encoder.config.hidden_size
        self.image_embedder = WanImageEmbedding(image_embed_dim, inner_dim)
        
        # override the forward method
        if model_class == 'WanTransformer3DModel':
            self.sd_ref().model._orig_i2v_adapter_forward = self.sd_ref().model.forward
            self.sd_ref().model.forward = partial(
                new_wan_forward,
                self.sd_ref().model
            )
            
            # add the wan image embedder
            self.sd_ref().model.condition_embedder._image_embedder = self.image_embedder
            self.sd_ref().model.condition_embedder._image_embedder.to(self.device_torch)
        
        self.sd_ref().model._i2v_adapter_ref = weakref.ref(self)

    def get_params(self):
        if self.control_lora is not None:
            config = {
                'text_encoder_lr': self.train_config.lr,
                'unet_lr': self.train_config.lr,
            }
            sig = inspect.signature(self.control_lora.prepare_optimizer_params)
            if 'default_lr' in sig.parameters:
                config['default_lr'] = self.train_config.lr
            if 'learning_rate' in sig.parameters:
                config['learning_rate'] = self.train_config.lr
            params_net = self.control_lora.prepare_optimizer_params(
                **config
            )

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

        if self.frame_embedder is not None:
            # make sure the embedder is float32
            self.frame_embedder.to(torch.float32)
            params += list(self.frame_embedder.parameters())

        # add the attn hogs
        for attn_hog in self.attn_hog_list:
            params += list(attn_hog.parameters())
        
        # add the image embedder
        if self.image_embedder is not None:
            params += list(self.image_embedder.parameters())
        return params

    def load_weights(self, state_dict, strict=True):
        lora_sd = {}
        attn_hog_sd = {}
        frame_embedder_sd = {}
        image_embedder_sd = {}
        
        for key, value in state_dict.items():
            if "frame_embedder" in key:
                new_key = key.replace("frame_embedder.", "")
                frame_embedder_sd[new_key] = value
            elif "attn_hog" in key:
                new_key = key.replace("attn_hog.", "")
                attn_hog_sd[new_key] = value
            elif "image_embedder" in key:
                new_key = key.replace("image_embedder.", "")
                image_embedder_sd[new_key] = value
            else:
                lora_sd[key] = value

        # todo process state dict before loading
        if self.control_lora is not None:
            self.control_lora.load_weights(lora_sd)
        if self.frame_embedder is not None:
            self.frame_embedder.load_state_dict(
                frame_embedder_sd, strict=False)
        self.attn_hog_list.load_state_dict(
            attn_hog_sd, strict=False)
        self.image_embedder.load_state_dict(
            image_embedder_sd, strict=False)

    def get_state_dict(self):
        if self.control_lora is not None:
            lora_sd = self.control_lora.get_state_dict(dtype=torch.float32)
        else:
            lora_sd = {}

        if self.frame_embedder is not None:
            frame_embedder_sd = self.frame_embedder.state_dict()
            for key, value in frame_embedder_sd.items():
                lora_sd[f"frame_embedder.{key}"] = value

        # add the attn hogs
        attn_hog_sd = self.attn_hog_list.state_dict()
        for key, value in attn_hog_sd.items():
            lora_sd[f"attn_hog.{key}"] = value
        
        # add the image embedder
        image_embedder_sd = self.image_embedder.state_dict()
        for key, value in image_embedder_sd.items():
            lora_sd[f"image_embedder.{key}"] = value
            
        return lora_sd
    
    def condition_noisy_latents(self, latents: torch.Tensor, batch:DataLoaderBatchDTO):
        # todo handle start frame
        return latents
    
    def edit_batch_processed(self, batch: DataLoaderBatchDTO):
        with torch.no_grad():
            # we will alway get a clip image frame, if one is not passed, use image
            # or if video, pull from the first frame
            # edit the batch to pull the first frame out of a video if we have it
            # videos come in (bs, num_frames, channels, height, width)
            tensor = batch.tensor
            if batch.clip_image_tensor is None:
                if len(tensor.shape) == 5:
                    # we have a video
                    first_frames = tensor[:, 0, :, :, :].clone()
                else:
                    # we have a single image
                    first_frames = tensor.clone()
                    
                # it is -1 to 1, change it to 0 to 1
                first_frames = (first_frames + 1) / 2
                    
                # clip image tensors are preprocessed. 
                tensors_0_1 = first_frames.to(dtype=torch.float16)
                clip_out = self.adapter_ref().clip_image_processor(
                    images=tensors_0_1,
                    return_tensors="pt",
                    do_resize=True,
                    do_rescale=False,
                ).pixel_values
                
                batch.clip_image_tensor = clip_out.to(self.device_torch)
        return batch

    @property
    def is_active(self):
        return self.adapter_ref().is_active
