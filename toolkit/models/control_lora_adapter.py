import inspect
import weakref
import torch
from typing import TYPE_CHECKING
from toolkit.lora_special import LoRASpecialNetwork
from diffusers import FluxTransformer2DModel
# weakref


if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.config_modules import AdapterConfig, TrainConfig, ModelConfig
    from toolkit.custom_adapter import CustomAdapter
    

# after each step we concat the control image with the latents 
# latent_model_input = torch.cat([latents, control_image], dim=2)
# the x_embedder has a full rank lora to handle the additional channels
# this replaces the x_embedder with a full rank lora. on flux this is 
# x_embedder(diffusers) or img_in(bfl)

# Flux
# img_in.lora_A.weight	[128, 128]	
# img_in.lora_B.bias	[3 072]	
# img_in.lora_B.weight	[3 072, 128]	
    

class ImgEmbedder(torch.nn.Module):
    def __init__(
        self,
        adapter: 'ControlLoraAdapter',
        orig_layer: torch.nn.Module,
        in_channels=128,
        out_channels=3072,
        bias=True
    ):
        super().__init__()
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.orig_layer_ref: weakref.ref = weakref.ref(orig_layer)
        self.lora_A = torch.nn.Linear(in_channels, in_channels, bias=False) # lora down
        self.lora_B = torch.nn.Linear(in_channels, out_channels, bias=bias) # lora up
        
    @classmethod
    def from_model(
        cls, 
        model: FluxTransformer2DModel, 
        adapter: 'ControlLoraAdapter', 
        num_channel_multiplier=2
    ):
        if model.__class__.__name__ == 'FluxTransformer2DModel':
            x_embedder: torch.nn.Linear = model.x_embedder
            img_embedder = cls(
                adapter, 
                orig_layer=x_embedder,
                in_channels=x_embedder.in_features * num_channel_multiplier, # adding additional control img channels
                out_channels=x_embedder.out_features, 
                bias=x_embedder.bias is not None
            )
            
            # hijack the forward method
            x_embedder._orig_ctrl_lora_forward = x_embedder.forward
            x_embedder.forward = img_embedder.forward
            dtype = x_embedder.weight.dtype
            device = x_embedder.weight.device
            
            # since we are adding control channels, we want those channels to be zero starting out
            # so they have no effect. It will match lora_B weight and bias, and we concat 0s for the input of the new channels
            # lora_a needs to be identity so that lora_b output matches lora_a output on init
            img_embedder.lora_A.weight.data = torch.eye(x_embedder.in_features * num_channel_multiplier).to(dtype=torch.float32, device=device)
            weight_b = x_embedder.weight.data.clone().to(dtype=torch.float32, device=device)
            # concat 0s for the new channels
            weight_b = torch.cat([weight_b, torch.zeros(weight_b.shape[0], weight_b.shape[1] * (num_channel_multiplier - 1)).to(device)], dim=1)
            img_embedder.lora_B.weight.data = weight_b.clone().to(dtype=torch.float32)
            img_embedder.lora_B.bias.data = x_embedder.bias.data.clone().to(dtype=torch.float32)
            
            # update the config of the transformer
            model.config.in_channels = model.config.in_channels * num_channel_multiplier
            model.config["in_channels"] = model.config.in_channels
            
            return img_embedder
        else:
            raise ValueError("Model not supported") 
        
    @property
    def is_active(self):
        return self.adapter_ref().is_active
        
    
    def forward(self, x):
        if not self.is_active:
            # make sure lora is not active
            self.adapter_ref().control_lora.is_active = False
            return self.orig_layer_ref()._orig_ctrl_lora_forward(x)
        
        # make sure lora is active
        self.adapter_ref().control_lora.is_active = True
        
        orig_device = x.device
        orig_dtype = x.dtype
        x = x.to(self.lora_A.weight.device, dtype=self.lora_A.weight.dtype)
        
        x = self.lora_A(x)
        x = self.lora_B(x)
        x = x.to(orig_device, dtype=orig_dtype)
        return x
    


class ControlLoraAdapter(torch.nn.Module):
    def __init__(
        self,
        adapter: 'CustomAdapter',
        sd: 'StableDiffusion',
        config: 'AdapterConfig',
        train_config: 'TrainConfig'
    ):
        super().__init__()
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.sd_ref = weakref.ref(sd)
        self.model_config: ModelConfig = sd.model_config
        self.network_config = config.lora_config
        self.train_config = train_config
        if self.network_config is None:
            raise ValueError("LoRA config is missing")
        
        network_kwargs = {} if self.network_config.network_kwargs is None else self.network_config.network_kwargs
        if hasattr(sd, 'target_lora_modules'):
            network_kwargs['target_lin_modules'] = self.sd.target_lora_modules
            
        if 'ignore_if_contains' not in network_kwargs:
            network_kwargs['ignore_if_contains'] = []
        
        # always ignore x_embedder
        network_kwargs['ignore_if_contains'].append('x_embedder')
            
        self.device_torch = sd.device_torch
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
            
        self.x_embedder = ImgEmbedder.from_model(sd.unet, self)
        self.x_embedder.to(self.device_torch)

    def get_params(self):
        # LyCORIS doesnt have default_lr
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
        
        params += list(self.x_embedder.parameters())
        
        # we need to be able to yield from the list like yield from params

        return params
    
    def load_weights(self, state_dict, strict=True):
        lora_sd = {}
        img_embedder_sd = {}
        for key, value in state_dict.items():
            if "x_embedder" in key:
                new_key = key.replace("transformer.x_embedder.", "")
                img_embedder_sd[new_key] = value
            else:
                lora_sd[key] = value
        
        # todo process state dict before loading
        self.control_lora.load_weights(lora_sd)
        self.x_embedder.load_state_dict(img_embedder_sd, strict=strict)
        
    def get_state_dict(self):
        lora_sd = self.control_lora.get_state_dict(dtype=torch.float32)
        # todo make sure we match loras elseware. 
        img_embedder_sd = self.x_embedder.state_dict()
        for key, value in img_embedder_sd.items():
            lora_sd[f"transformer.x_embedder.{key}"] = value
        return lora_sd
    
    @property
    def is_active(self):
        return self.adapter_ref().is_active
