import inspect
import weakref
import torch
from typing import TYPE_CHECKING
from toolkit.lora_special import LoRASpecialNetwork
from diffusers import FluxTransformer2DModel
# weakref
from toolkit.pixel_shuffle_encoder import AutoencoderPixelMixer


if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.config_modules import AdapterConfig, TrainConfig, ModelConfig
    from toolkit.custom_adapter import CustomAdapter
    


class InOutModule(torch.nn.Module):
    def __init__(
        self,
        adapter: 'SubpixelAdapter',
        orig_layer: torch.nn.Linear,
        in_channels=64,
        out_channels=3072
    ):
        super().__init__()
        # only do the weight for the new input. We combine with the original linear layer
        self.x_embedder = torch.nn.Linear(
            in_channels,
            out_channels,
            bias=True,
        )
        
        self.proj_out = torch.nn.Linear(
            out_channels,
            in_channels,
            bias=True,
        )
        # make sure the weight is float32
        self.x_embedder.weight.data = self.x_embedder.weight.data.float()
        self.x_embedder.bias.data = self.x_embedder.bias.data.float()
        
        self.proj_out.weight.data = self.proj_out.weight.data.float()
        self.proj_out.bias.data = self.proj_out.bias.data.float()
        
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.orig_layer_ref: weakref.ref = weakref.ref(orig_layer)
        
    @classmethod
    def from_model(
        cls, 
        model: FluxTransformer2DModel, 
        adapter: 'SubpixelAdapter',
        num_channels: int = 768,
        downscale_factor: int = 8
    ):
        if model.__class__.__name__ == 'FluxTransformer2DModel':            
            
            x_embedder: torch.nn.Linear = model.x_embedder
            proj_out: torch.nn.Linear = model.proj_out
            in_out_module = cls(
                adapter, 
                orig_layer=x_embedder,
                in_channels=num_channels,
                out_channels=x_embedder.out_features,
            )
            
            # hijack the forward method
            x_embedder._orig_ctrl_lora_forward = x_embedder.forward
            x_embedder.forward = in_out_module.in_forward
            proj_out._orig_ctrl_lora_forward = proj_out.forward
            proj_out.forward = in_out_module.out_forward

            # update the config of the transformer
            model.config.in_channels = num_channels
            model.config["in_channels"] = num_channels
            model.config.out_channels = num_channels
            model.config["out_channels"] = num_channels
            
            # if the shape matches, copy the weights
            if x_embedder.weight.shape == in_out_module.x_embedder.weight.shape:
                in_out_module.x_embedder.weight.data = x_embedder.weight.data.clone().float()
                in_out_module.x_embedder.bias.data = x_embedder.bias.data.clone().float()
                in_out_module.proj_out.weight.data = proj_out.weight.data.clone().float()
                in_out_module.proj_out.bias.data = proj_out.bias.data.clone().float()
            
            # replace the vae of the model
            sd = adapter.sd_ref()
            sd.vae = AutoencoderPixelMixer(
                in_channels=3,
                downscale_factor=downscale_factor
            )
            
            sd.pipeline.vae = sd.vae
            
            return in_out_module
        else:
            raise ValueError("Model not supported") 
        
    @property
    def is_active(self):
        return self.adapter_ref().is_active
        
    
    def in_forward(self, x):
        if not self.is_active:
            # make sure lora is not active
            if self.adapter_ref().control_lora is not None:
                self.adapter_ref().control_lora.is_active = False
            return self.orig_layer_ref()._orig_ctrl_lora_forward(x)
        
        # make sure lora is active
        if self.adapter_ref().control_lora is not None:
            self.adapter_ref().control_lora.is_active = True
        
        orig_device = x.device
        orig_dtype = x.dtype
    
        x = x.to(self.x_embedder.weight.device, dtype=self.x_embedder.weight.dtype)
        
        x = self.x_embedder(x)
        
        x = x.to(orig_device, dtype=orig_dtype)
        return x
    
    def out_forward(self, x):
        if not self.is_active:
            # make sure lora is not active
            if self.adapter_ref().control_lora is not None:
                self.adapter_ref().control_lora.is_active = False
            return self.orig_layer_ref()._orig_ctrl_lora_forward(x)
        
        # make sure lora is active
        if self.adapter_ref().control_lora is not None:
            self.adapter_ref().control_lora.is_active = True
        
        orig_device = x.device
        orig_dtype = x.dtype
    
        x = x.to(self.proj_out.weight.device, dtype=self.proj_out.weight.dtype)
        
        x = self.proj_out(x)
        
        x = x.to(orig_device, dtype=orig_dtype)
        return x
    


class SubpixelAdapter(torch.nn.Module):
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
        self.device_torch = sd.device_torch
        self.control_lora = None
        
        if self.network_config is not None:
        
            network_kwargs = {} if self.network_config.network_kwargs is None else self.network_config.network_kwargs
            if hasattr(sd, 'target_lora_modules'):
                network_kwargs['target_lin_modules'] = sd.target_lora_modules
                
            if 'ignore_if_contains' not in network_kwargs:
                network_kwargs['ignore_if_contains'] = []
            
            # always ignore x_embedder
            network_kwargs['ignore_if_contains'].append('transformer.x_embedder')
            network_kwargs['ignore_if_contains'].append('transformer.proj_out')
                
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
        
        downscale_factor = config.subpixel_downscale_factor
        if downscale_factor == 8:
            num_channels = 768
        elif downscale_factor == 16:
            num_channels = 3072
        else:
            raise ValueError(
                f"downscale_factor {downscale_factor} not supported"
            )
        
        self.in_out: InOutModule = InOutModule.from_model(
            sd.unet_unwrapped, 
            self,
            num_channels=num_channels, # packed channels
            downscale_factor=downscale_factor
        )
        self.in_out.to(self.device_torch)

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
            
        # make sure the embedder is float32
        self.in_out.to(torch.float32)
        
        params += list(self.in_out.parameters())
        
        # we need to be able to yield from the list like yield from params

        return params
    
    def load_weights(self, state_dict, strict=True):
        lora_sd = {}
        img_embedder_sd = {}
        for key, value in state_dict.items():
            if "transformer.x_embedder" in key:
                new_key = key.replace("transformer.", "")
                img_embedder_sd[new_key] = value
            elif "transformer.proj_out" in key:
                new_key = key.replace("transformer.", "")
                img_embedder_sd[new_key] = value
            else:
                lora_sd[key] = value
        
        # todo process state dict before loading
        if self.control_lora is not None:
            self.control_lora.load_weights(lora_sd)
        # automatically upgrade the x imbedder if more dims are added
        self.in_out.load_state_dict(img_embedder_sd, strict=False)
        
    def get_state_dict(self):
        if self.control_lora is not None:
            lora_sd = self.control_lora.get_state_dict(dtype=torch.float32)
        else:
            lora_sd = {}
        # todo make sure we match loras elseware. 
        img_embedder_sd = self.in_out.state_dict()
        for key, value in img_embedder_sd.items():
            lora_sd[f"transformer.{key}"] = value
        return lora_sd
    
    @property
    def is_active(self):
        return self.adapter_ref().is_active
