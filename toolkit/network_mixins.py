import json
import os
from collections import OrderedDict
from typing import Optional, Union, List, Type, TYPE_CHECKING, Dict, Any, Literal

import torch
from optimum.quanto import QTensor
from torch import nn
import weakref

from tqdm import tqdm

from toolkit.config_modules import NetworkConfig
from toolkit.lorm import extract_conv, extract_linear, count_parameters
from toolkit.metadata import add_model_hash_to_meta
from toolkit.paths import KEYMAPS_ROOT
from toolkit.saving import get_lora_keymap_from_model_keymap
from optimum.quanto import QBytesTensor

if TYPE_CHECKING:
    from toolkit.lycoris_special import LycorisSpecialNetwork, LoConSpecialModule
    from toolkit.lora_special import LoRASpecialNetwork, LoRAModule
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.models.DoRA import DoRAModule

Network = Union['LycorisSpecialNetwork', 'LoRASpecialNetwork']
Module = Union['LoConSpecialModule', 'LoRAModule', 'DoRAModule']

LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear',
    'QLinear'
    # 'GroupNorm',
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]

ExtractMode = Union[
    'existing'
    'fixed',
    'threshold',
    'ratio',
    'quantile',
    'percentage'
]


def broadcast_and_multiply(tensor, multiplier):
    # Determine the number of dimensions required
    num_extra_dims = tensor.dim() - multiplier.dim()

    # Unsqueezing the tensor to match the dimensionality
    for _ in range(num_extra_dims):
        multiplier = multiplier.unsqueeze(-1)

    try:
        # Multiplying the broadcasted tensor with the output tensor
        result = tensor * multiplier
    except RuntimeError as e:
        print(e)
        print(tensor.size())
        print(multiplier.size())
        raise e

    return result


def add_bias(tensor, bias):
    if bias is None:
        return tensor
    # add batch dim
    bias = bias.unsqueeze(0)
    bias = torch.cat([bias] * tensor.size(0), dim=0)
    # Determine the number of dimensions required
    num_extra_dims = tensor.dim() - bias.dim()

    # Unsqueezing the tensor to match the dimensionality
    for _ in range(num_extra_dims):
        bias = bias.unsqueeze(-1)

    # we may need to swap -1 for -2
    if bias.size(1) != tensor.size(1):
        if len(bias.size()) == 3:
            bias = bias.permute(0, 2, 1)
        elif len(bias.size()) == 4:
            bias = bias.permute(0, 3, 1, 2)

    # Multiplying the broadcasted tensor with the output tensor
    try:
        result = tensor + bias
    except RuntimeError as e:
        print(e)
        print(tensor.size())
        print(bias.size())
        raise e

    return result


class ExtractableModuleMixin:
    def extract_weight(
            self: Module,
            extract_mode: ExtractMode = "existing",
            extract_mode_param: Union[int, float] = None,
    ):
        device = self.lora_down.weight.device
        weight_to_extract = self.org_module[0].weight
        if extract_mode == "existing":
            extract_mode = 'fixed'
            extract_mode_param = self.lora_dim
            
        if isinstance(weight_to_extract, QBytesTensor):
            weight_to_extract = weight_to_extract.dequantize()
        
        weight_to_extract = weight_to_extract.clone().detach().float()

        if self.org_module[0].__class__.__name__ in CONV_MODULES:
            # do conv extraction
            down_weight, up_weight, new_dim, diff = extract_conv(
                weight=weight_to_extract,
                mode=extract_mode,
                mode_param=extract_mode_param,
                device=device
            )

        elif self.org_module[0].__class__.__name__ in LINEAR_MODULES:
            # do linear extraction
            down_weight, up_weight, new_dim, diff = extract_linear(
                weight=weight_to_extract,
                mode=extract_mode,
                mode_param=extract_mode_param,
                device=device,
            )
        else:
            raise ValueError(f"Unknown module type: {self.org_module[0].__class__.__name__}")

        self.lora_dim = new_dim

        # inject weights into the param
        self.lora_down.weight.data = down_weight.to(self.lora_down.weight.dtype).clone().detach()
        self.lora_up.weight.data = up_weight.to(self.lora_up.weight.dtype).clone().detach()

        # copy bias if we have one and are using them
        if self.org_module[0].bias is not None and self.lora_up.bias is not None:
            self.lora_up.bias.data = self.org_module[0].bias.data.clone().detach()

        # set up alphas
        self.alpha = (self.alpha * 0) + down_weight.shape[0]
        self.scale = self.alpha / self.lora_dim

        # assign them

        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            # scaler is a parameter update the value with 1.0
            self.scalar.data = torch.tensor(1.0).to(self.scalar.device, self.scalar.dtype)


class ToolkitModuleMixin:
    def __init__(
            self: Module,
            *args,
            network: Network,
            **kwargs
    ):
        self.network_ref: weakref.ref = weakref.ref(network)
        self.is_checkpointing = False
        self._multiplier: Union[float, list, torch.Tensor] = None

    def _call_forward(self: Module, x):
        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return 0.0  # added to original forward

        if hasattr(self, 'lora_mid') and self.lora_mid is not None:
            lx = self.lora_mid(self.lora_down(x))
        else:
            try:
                lx = self.lora_down(x)
            except RuntimeError as e:
                print(f"Error in {self.__class__.__name__} lora_down")
                raise e

        if isinstance(self.dropout, nn.Dropout) or isinstance(self.dropout, nn.Identity):
            lx = self.dropout(lx)
        # normal dropout
        elif self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.rank_dropout > 0 and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            scale = scale * self.scalar

        return lx * scale

    def lorm_forward(self: Network, x, *args, **kwargs):
        network: Network = self.network_ref()
        if not network.is_active:
            return self.org_forward(x, *args, **kwargs)
        
        orig_dtype = x.dtype
        
        if x.dtype != self.lora_down.weight.dtype:
            x = x.to(self.lora_down.weight.dtype)

        if network.lorm_train_mode == 'local':
            # we are going to predict input with both and do a loss on them
            inputs = x.detach()
            with torch.no_grad():
                # get the local prediction
                target_pred = self.org_forward(inputs, *args, **kwargs).detach()
            with torch.set_grad_enabled(True):
                # make a prediction with the lorm
                lorm_pred = self.lora_up(self.lora_down(inputs.requires_grad_(True)))

                local_loss = torch.nn.functional.mse_loss(target_pred.float(), lorm_pred.float())
                # backpropr
                local_loss.backward()

            network.module_losses.append(local_loss.detach())
            # return the original as we dont want our trainer to affect ones down the line
            return target_pred

        else:
            x = self.lora_up(self.lora_down(x))
            if x.dtype != orig_dtype:
                x = x.to(orig_dtype)

    def forward(self: Module, x, *args, **kwargs):
        skip = False
        network: Network = self.network_ref()
        if network.is_lorm:
            # we are doing lorm
            return self.lorm_forward(x, *args, **kwargs)

        # skip if not active
        if not network.is_active:
            skip = True

        # skip if is merged in
        if network.is_merged_in:
            skip = True

        # skip if multiplier is 0
        if network._multiplier == 0:
            skip = True

        if skip:
            # network is not active, avoid doing anything
            return self.org_forward(x, *args, **kwargs)

        # if self.__class__.__name__ == "DoRAModule":
        #     # return dora forward
        #     return self.dora_forward(x, *args, **kwargs)
        
        if self.__class__.__name__ == "LokrModule":
            return self._call_forward(x)

        org_forwarded = self.org_forward(x, *args, **kwargs)

        if isinstance(x, QTensor):
            x = x.dequantize()
        # always cast to float32
        lora_input = x.to(self.lora_down.weight.dtype)
        lora_output = self._call_forward(lora_input)
        multiplier = self.network_ref().torch_multiplier

        lora_output_batch_size = lora_output.size(0)
        multiplier_batch_size = multiplier.size(0)
        if lora_output_batch_size != multiplier_batch_size:
            num_interleaves = lora_output_batch_size // multiplier_batch_size
            # todo check if this is correct, do we just concat when doing cfg?
            multiplier = multiplier.repeat_interleave(num_interleaves)

        scaled_lora_output = broadcast_and_multiply(lora_output, multiplier)
        scaled_lora_output = scaled_lora_output.to(org_forwarded.dtype)

        if self.__class__.__name__ == "DoRAModule":
            # ref https://github.com/huggingface/peft/blob/1e6d1d73a0850223b0916052fd8d2382a90eae5a/src/peft/tuners/lora/layer.py#L417
            # x = dropout(x)
            # todo this wont match the dropout applied to the lora
            if isinstance(self.dropout, nn.Dropout) or isinstance(self.dropout, nn.Identity):
                lx = self.dropout(x)
            # normal dropout
            elif self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(x, p=self.dropout)
            else:
                lx = x
            lora_weight = self.lora_up.weight @ self.lora_down.weight
            # scale it here
            # todo handle our batch split scalers for slider training. For now take the mean of them
            scale = multiplier.mean()
            scaled_lora_weight = lora_weight * scale
            scaled_lora_output = scaled_lora_output + self.apply_dora(lx, scaled_lora_weight).to(org_forwarded.dtype)

        try:
            x = org_forwarded + scaled_lora_output
        except RuntimeError as e:
            print(e)
            print(org_forwarded.size())
            print(scaled_lora_output.size())
            raise e
        return x

    def enable_gradient_checkpointing(self: Module):
        self.is_checkpointing = True

    def disable_gradient_checkpointing(self: Module):
        self.is_checkpointing = False

    @torch.no_grad()
    def merge_out(self: Module, merge_out_weight=1.0):
        # make sure it is positive
        merge_out_weight = abs(merge_out_weight)
        # merging out is just merging in the negative of the weight
        self.merge_in(merge_weight=-merge_out_weight)

    @torch.no_grad()
    def merge_in(self: Module, merge_weight=1.0):
        if not self.can_merge_in:
            return
        # get up/down weight
        up_weight = self.lora_up.weight.clone().float()
        down_weight = self.lora_down.weight.clone().float()

        # extract weight from org_module
        org_sd = self.org_module[0].state_dict()
        # todo find a way to merge in weights when doing quantized model
        if 'weight._data' in org_sd:
            # quantized weight
            return

        weight_key = "weight"
        if 'weight._data' in org_sd:
            # quantized weight
            weight_key = "weight._data"

        orig_dtype = org_sd[weight_key].dtype
        weight = org_sd[weight_key].float()

        multiplier = merge_weight
        scale = self.scale
        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            scale = scale * self.scalar

        # merge weight
        if len(weight.size()) == 2:
            # linear
            weight = weight + multiplier * (up_weight @ down_weight) * scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                    weight
                    + multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # print(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + multiplier * conved * scale

        # set weight to org_module
        org_sd[weight_key] = weight.to(orig_dtype)
        self.org_module[0].load_state_dict(org_sd)

    def setup_lorm(self: Module, state_dict: Optional[Dict[str, Any]] = None):
        # LoRM (Low Rank Middle) is a method reduce the number of parameters in a module while keeping the inputs and
        # outputs the same. It is basically a LoRA but with the original module removed

        # if a state dict is passed, use those weights instead of extracting
        # todo load from state dict
        network: Network = self.network_ref()
        lorm_config = network.network_config.lorm_config.get_config_for_module(self.lora_name)

        extract_mode = lorm_config.extract_mode
        extract_mode_param = lorm_config.extract_mode_param
        parameter_threshold = lorm_config.parameter_threshold
        self.extract_weight(
            extract_mode=extract_mode,
            extract_mode_param=extract_mode_param
        )


class ToolkitNetworkMixin:
    def __init__(
            self: Network,
            *args,
            train_text_encoder: Optional[bool] = True,
            train_unet: Optional[bool] = True,
            is_sdxl=False,
            is_v2=False,
            is_ssd=False,
            is_vega=False,
            network_config: Optional[NetworkConfig] = None,
            is_lorm=False,
            **kwargs
    ):
        self.train_text_encoder = train_text_encoder
        self.train_unet = train_unet
        self.is_checkpointing = False
        self._multiplier: float = 1.0
        self.is_active: bool = False
        self.is_sdxl = is_sdxl
        self.is_ssd = is_ssd
        self.is_vega = is_vega
        self.is_v2 = is_v2
        self.is_v1 = not is_v2 and not is_sdxl and not is_ssd and not is_vega
        self.is_merged_in = False
        self.is_lorm = is_lorm
        self.network_config: NetworkConfig = network_config
        self.module_losses: List[torch.Tensor] = []
        self.lorm_train_mode: Literal['local', None] = None
        self.can_merge_in = not is_lorm

    def get_keymap(self: Network, force_weight_mapping=False):
        use_weight_mapping = False

        if self.is_ssd:
            keymap_tail = 'ssd'
            use_weight_mapping = True
        elif self.is_vega:
            keymap_tail = 'vega'
            use_weight_mapping = True
        elif self.is_sdxl:
            keymap_tail = 'sdxl'
        elif self.is_v2:
            keymap_tail = 'sd2'
        else:
            keymap_tail = 'sd1'
            # todo double check this
            # use_weight_mapping = True

        if force_weight_mapping:
            use_weight_mapping = True

        # load keymap
        keymap_name = f"stable_diffusion_locon_{keymap_tail}.json"
        if use_weight_mapping:
            keymap_name = f"stable_diffusion_{keymap_tail}.json"

        keymap_path = os.path.join(KEYMAPS_ROOT, keymap_name)

        keymap = None
        # check if file exists
        if os.path.exists(keymap_path):
            with open(keymap_path, 'r', encoding='utf-8') as f:
                keymap = json.load(f)['ldm_diffusers_keymap']

        if use_weight_mapping and keymap is not None:
            # get keymap from weights
            keymap = get_lora_keymap_from_model_keymap(keymap)

        # upgrade keymaps for DoRA
        if self.network_type.lower() == 'dora':
            if keymap is not None:
                new_keymap = {}
                for ldm_key, diffusers_key in keymap.items():
                    ldm_key = ldm_key.replace('.alpha', '.magnitude')
                    # ldm_key = ldm_key.replace('.lora_down.weight', '.lora_down')
                    # ldm_key = ldm_key.replace('.lora_up.weight', '.lora_up')

                    diffusers_key = diffusers_key.replace('.alpha', '.magnitude')
                    # diffusers_key = diffusers_key.replace('.lora_down.weight', '.lora_down')
                    # diffusers_key = diffusers_key.replace('.lora_up.weight', '.lora_up')

                    new_keymap[ldm_key] = diffusers_key

                keymap = new_keymap

        return keymap
    
    def get_state_dict(self: Network, extra_state_dict=None, dtype=torch.float16):
        keymap = self.get_keymap()

        save_keymap = {}
        if keymap is not None:
            for ldm_key, diffusers_key in keymap.items():
                #  invert them
                save_keymap[diffusers_key] = ldm_key

        state_dict = self.state_dict()
        save_dict = OrderedDict()

        for key in list(state_dict.keys()):
            v = state_dict[key]
            v = v.detach().clone().to("cpu").to(dtype)
            save_key = save_keymap[key] if key in save_keymap else key
            save_dict[save_key] = v
            del state_dict[key]

        if extra_state_dict is not None:
            # add extra items to state dict
            for key in list(extra_state_dict.keys()):
                v = extra_state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                save_dict[key] = v

        if self.peft_format:
            # lora_down = lora_A
            # lora_up = lora_B
            # no alpha

            new_save_dict = {}
            for key, value in save_dict.items():
                if key.endswith('.alpha'):
                    continue
                new_key = key
                new_key = new_key.replace('lora_down', 'lora_A')
                new_key = new_key.replace('lora_up', 'lora_B')
                # replace all $$ with .
                new_key = new_key.replace('$$', '.')
                new_save_dict[new_key] = value

            save_dict = new_save_dict
        
                
        if self.network_type.lower() == "lokr":
            new_save_dict = {}
            for key, value in save_dict.items():
                # lora_transformer_transformer_blocks_7_attn_to_v.lokr_w1 to lycoris_transformer_blocks_7_attn_to_v.lokr_w1
                new_key = key
                new_key = new_key.replace('lora_transformer_', 'lycoris_')
                new_save_dict[new_key] = value

            save_dict = new_save_dict
        
        if self.base_model_ref is not None:
            save_dict = self.base_model_ref().convert_lora_weights_before_save(save_dict)
        return save_dict

    def save_weights(
            self: Network,
            file, dtype=torch.float16,
            metadata=None,
            extra_state_dict: Optional[OrderedDict] = None
    ):
        save_dict = self.get_state_dict(extra_state_dict=extra_state_dict, dtype=dtype)
        
        if metadata is not None and len(metadata) == 0:
            metadata = None

        if metadata is None:
            metadata = OrderedDict()
        metadata = add_model_hash_to_meta(save_dict, metadata)
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            save_file(save_dict, file, metadata)
        else:
            torch.save(save_dict, file)

    def load_weights(self: Network, file, force_weight_mapping=False):
        # allows us to save and load to and from ldm weights
        keymap = self.get_keymap(force_weight_mapping)
        keymap = {} if keymap is None else keymap

        if isinstance(file, str):
            if os.path.splitext(file)[1] == ".safetensors":
                from safetensors.torch import load_file

                weights_sd = load_file(file)
            else:
                weights_sd = torch.load(file, map_location="cpu")
        else:
            # probably a state dict
            weights_sd = file
        
        if self.base_model_ref is not None:
            weights_sd = self.base_model_ref().convert_lora_weights_before_load(weights_sd)

        load_sd = OrderedDict()
        for key, value in weights_sd.items():
            load_key = keymap[key] if key in keymap else key
            # replace old double __ with single _
            if self.is_pixart:
                load_key = load_key.replace('__', '_')

            if self.peft_format:
                # lora_down = lora_A
                # lora_up = lora_B
                # no alpha
                if load_key.endswith('.alpha'):
                    continue
                load_key = load_key.replace('lora_A', 'lora_down')
                load_key = load_key.replace('lora_B', 'lora_up')
                # replace all . with $$
                load_key = load_key.replace('.', '$$')
                load_key = load_key.replace('$$lora_down$$', '.lora_down.')
                load_key = load_key.replace('$$lora_up$$', '.lora_up.')
            
            if self.network_type.lower() == "lokr":
                # lora_transformer_transformer_blocks_7_attn_to_v.lokr_w1 to lycoris_transformer_blocks_7_attn_to_v.lokr_w1
                load_key = load_key.replace('lycoris_', 'lora_transformer_')

            load_sd[load_key] = value

        # extract extra items from state dict
        current_state_dict = self.state_dict()
        extra_dict = OrderedDict()
        to_delete = []
        for key in list(load_sd.keys()):
            if key not in current_state_dict:
                extra_dict[key] = load_sd[key]
                to_delete.append(key)
        for key in to_delete:
            del load_sd[key]

        print(f"Missing keys: {to_delete}")
        if len(to_delete) > 0 and self.is_v1 and not force_weight_mapping and not (
                len(to_delete) == 1 and 'emb_params' in to_delete):
            print(" Attempting to load with forced keymap")
            return self.load_weights(file, force_weight_mapping=True)

        info = self.load_state_dict(load_sd, False)
        if len(extra_dict.keys()) == 0:
            extra_dict = None
        return extra_dict

    @torch.no_grad()
    def _update_torch_multiplier(self: Network):
        # builds a tensor for fast usage in the forward pass of the network modules
        # without having to set it in every single module every time it changes
        multiplier = self._multiplier
        # get first module
        try:
            first_module = self.get_all_modules()[0]
        except IndexError:
            raise ValueError("There are not any lora modules in this network. Check your config and try again")
        
        if hasattr(first_module, 'lora_down'):
            device = first_module.lora_down.weight.device
            dtype = first_module.lora_down.weight.dtype
        elif hasattr(first_module, 'lokr_w1'):
            device = first_module.lokr_w1.device
            dtype = first_module.lokr_w1.dtype
        elif hasattr(first_module, 'lokr_w1_a'):
            device = first_module.lokr_w1_a.device
            dtype = first_module.lokr_w1_a.dtype
        else:
            raise ValueError("Unknown module type")
        with torch.no_grad():
            tensor_multiplier = None
            if isinstance(multiplier, int) or isinstance(multiplier, float):
                tensor_multiplier = torch.tensor((multiplier,)).to(device, dtype=dtype)
            elif isinstance(multiplier, list):
                tensor_multiplier = torch.tensor(multiplier).to(device, dtype=dtype)
            elif isinstance(multiplier, torch.Tensor):
                tensor_multiplier = multiplier.clone().detach().to(device, dtype=dtype)

            self.torch_multiplier = tensor_multiplier.clone().detach()

    @property
    def multiplier(self) -> Union[float, List[float], List[List[float]]]:
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Union[float, List[float], List[List[float]]]):
        # it takes time to update all the multipliers, so we only do it if the value has changed
        if self._multiplier == value:
            return
        # if we are setting a single value but have a list, keep the list if every item is the same as value
        self._multiplier = value
        self._update_torch_multiplier()

    # called when the context manager is entered
    # ie: with network:
    def __enter__(self: Network):
        self.is_active = True

    def __exit__(self: Network, exc_type, exc_value, tb):
        self.is_active = False

    def force_to(self: Network, device, dtype):
        self.to(device, dtype)
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        for lora in loras:
            lora.to(device, dtype)

    def get_all_modules(self: Network) -> List[Module]:
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        return loras

    def _update_checkpointing(self: Network):
        for module in self.get_all_modules():
            if self.is_checkpointing:
                module.enable_gradient_checkpointing()
            else:
                module.disable_gradient_checkpointing()

    def enable_gradient_checkpointing(self: Network):
        # not supported
        self.is_checkpointing = True
        self._update_checkpointing()

    def disable_gradient_checkpointing(self: Network):
        # not supported
        self.is_checkpointing = False
        self._update_checkpointing()

    def merge_in(self, merge_weight=1.0):
        if self.network_type.lower() == 'dora':
            return
        self.is_merged_in = True
        for module in self.get_all_modules():
            module.merge_in(merge_weight)

    def merge_out(self: Network, merge_weight=1.0):
        if not self.is_merged_in:
            return
        self.is_merged_in = False
        for module in self.get_all_modules():
            module.merge_out(merge_weight)

    def extract_weight(
            self: Network,
            extract_mode: ExtractMode = "existing",
            extract_mode_param: Union[int, float] = None,
    ):
        if extract_mode_param is None:
            raise ValueError("extract_mode_param must be set")
        for module in tqdm(self.get_all_modules(), desc="Extracting weights"):
            module.extract_weight(
                extract_mode=extract_mode,
                extract_mode_param=extract_mode_param
            )

    def setup_lorm(self: Network, state_dict: Optional[Dict[str, Any]] = None):
        for module in tqdm(self.get_all_modules(), desc="Extracting LoRM"):
            module.setup_lorm(state_dict=state_dict)

    def calculate_lorem_parameter_reduction(self):
        params_reduced = 0
        for module in self.get_all_modules():
            num_orig_module_params = count_parameters(module.org_module[0])
            num_lorem_params = count_parameters(module.lora_down) + count_parameters(module.lora_up)
            params_reduced += (num_orig_module_params - num_lorem_params)

        return params_reduced
