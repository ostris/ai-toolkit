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
        orig_layer: torch.nn.Linear,
        in_channels=64,
        out_channels=3072
    ):
        super().__init__()
        # only do the weight for the new input. We combine with the original linear layer
        init = torch.randn(out_channels, in_channels, device=orig_layer.weight.device, dtype=orig_layer.weight.dtype) * 0.01
        self.weight = torch.nn.Parameter(init)
        
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.orig_layer_ref: weakref.ref = weakref.ref(orig_layer)
        
    @classmethod
    def from_model(
        cls, 
        model: FluxTransformer2DModel, 
        adapter: 'ControlLoraAdapter', 
        num_control_images=1,
        has_inpainting_input=False
    ):
        if model.__class__.__name__ == 'FluxTransformer2DModel':            
            num_adapter_in_channels = model.x_embedder.in_features * num_control_images
            
            if has_inpainting_input:
                # inpainting has the mask before packing latents. it is normally 16 ch + 1ch mask
                # packed it is 64ch + 4ch mask
                # so we need to add 4 to the input channels
                num_adapter_in_channels += 4
            
            x_embedder: torch.nn.Linear = model.x_embedder
            img_embedder = cls(
                adapter, 
                orig_layer=x_embedder,
                in_channels=num_adapter_in_channels,
                out_channels=x_embedder.out_features,
            )
            
            # hijack the forward method
            x_embedder._orig_ctrl_lora_forward = x_embedder.forward
            x_embedder.forward = img_embedder.forward

            # update the config of the transformer
            model.config.in_channels = model.config.in_channels * (num_control_images + 1)
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
            if self.adapter_ref().control_lora is not None:
                self.adapter_ref().control_lora.is_active = False
            return self.orig_layer_ref()._orig_ctrl_lora_forward(x)
        
        # make sure lora is active
        if self.adapter_ref().control_lora is not None:
            self.adapter_ref().control_lora.is_active = True
        
        orig_device = x.device
        orig_dtype = x.dtype
    
        x = x.to(self.weight.device, dtype=self.weight.dtype)
        
        orig_weight = self.orig_layer_ref().weight.data.detach()
        orig_weight = orig_weight.to(self.weight.device, dtype=self.weight.dtype)
        linear_weight = torch.cat([orig_weight, self.weight], dim=1)
        
        bias = None
        if self.orig_layer_ref().bias is not None:
            bias = self.orig_layer_ref().bias.data.detach().to(self.weight.device, dtype=self.weight.dtype)
            
        x = torch.nn.functional.linear(x, linear_weight, bias)
        
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
        self.device_torch = sd.device_torch
        self.control_lora = None
        
        if self.network_config is not None:
        
            network_kwargs = {} if self.network_config.network_kwargs is None else self.network_config.network_kwargs
            if hasattr(sd, 'target_lora_modules'):
                network_kwargs['target_lin_modules'] = self.sd.target_lora_modules
                
            if 'ignore_if_contains' not in network_kwargs:
                network_kwargs['ignore_if_contains'] = []
            
            # always ignore x_embedder
            network_kwargs['ignore_if_contains'].append('x_embedder')
                
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
            
        self.x_embedder = ImgEmbedder.from_model(
            sd.unet, 
            self,
            num_control_images=config.num_control_images,
            has_inpainting_input=config.has_inpainting_input
        )
        self.x_embedder.to(self.device_torch)

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
        self.x_embedder.to(torch.float32)
        
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
        if self.control_lora is not None:
            self.control_lora.load_weights(lora_sd)
        # automatically upgrade the x imbedder if more dims are added
        if self.x_embedder.weight.shape[1] > img_embedder_sd['weight'].shape[1]:
            print("Upgrading x_embedder from {} to {}".format(
                img_embedder_sd['weight'].shape[1], 
                self.x_embedder.weight.shape[1]
            ))
            while img_embedder_sd['weight'].shape[1] < self.x_embedder.weight.shape[1]:
                img_embedder_sd['weight'] = torch.cat([img_embedder_sd['weight'] ] * 2, dim=1)
            if img_embedder_sd['weight'].shape[1] > self.x_embedder.weight.shape[1]:
                img_embedder_sd['weight'] = img_embedder_sd['weight'][:, :self.x_embedder.weight.shape[1]]
        self.x_embedder.load_state_dict(img_embedder_sd, strict=False)
        
    def get_state_dict(self):
        if self.control_lora is not None:
            lora_sd = self.control_lora.get_state_dict(dtype=torch.float32)
        else:
            lora_sd = {}
        # todo make sure we match loras elseware. 
        img_embedder_sd = self.x_embedder.state_dict()
        for key, value in img_embedder_sd.items():
            lora_sd[f"transformer.x_embedder.{key}"] = value
        return lora_sd
    
    @property
    def is_active(self):
        return self.adapter_ref().is_active
