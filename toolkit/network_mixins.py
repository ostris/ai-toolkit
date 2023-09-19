import json
import os
from collections import OrderedDict
from typing import Optional, Union, List, Type, TYPE_CHECKING, Dict, Any

import torch
from torch import nn
import weakref
from toolkit.metadata import add_model_hash_to_meta
from toolkit.paths import KEYMAPS_ROOT

if TYPE_CHECKING:
    from toolkit.lycoris_special import LycorisSpecialNetwork, LoConSpecialModule
    from toolkit.lora_special import LoRASpecialNetwork, LoRAModule

Network = Union['LycorisSpecialNetwork', 'LoRASpecialNetwork']
Module = Union['LoConSpecialModule', 'LoRAModule']

LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear'
    # 'GroupNorm',
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]


def broadcast_and_multiply(tensor, multiplier):
    # Determine the number of dimensions required
    num_extra_dims = tensor.dim() - multiplier.dim()

    # Unsqueezing the tensor to match the dimensionality
    for _ in range(num_extra_dims):
        multiplier = multiplier.unsqueeze(-1)

    # Multiplying the broadcasted tensor with the output tensor
    result = tensor * multiplier

    return result


class ToolkitModuleMixin:
    def __init__(
            self: Module,
            *args,
            network: Network,
            call_super_init: bool = True,
            **kwargs
    ):
        if call_super_init:
            super().__init__(*args, **kwargs)
        self.network_ref: weakref.ref = weakref.ref(network)
        self.is_checkpointing = False
        self.is_normalizing = False
        self.normalize_scaler = 1.0
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

    # this may get an additional positional arg or not

    def forward(self: Module, x, *args, **kwargs):
        if not self.network_ref().is_active:
            # network is not active, avoid doing anything
            return self.org_forward(x, *args, **kwargs)

        org_forwarded = self.org_forward(x, *args, **kwargs)
        lora_output = self._call_forward(x)
        multiplier = self.network_ref().torch_multiplier

        lora_output_batch_size = lora_output.size(0)
        multiplier_batch_size = multiplier.size(0)
        if lora_output_batch_size != multiplier_batch_size:
            num_interleaves = lora_output_batch_size // multiplier_batch_size
            multiplier = multiplier.repeat_interleave(num_interleaves)
        # multiplier = 1.0

        if self.is_normalizing:
            with torch.no_grad():

                # do this calculation without set multiplier and instead use same polarity, but with 1.0 multiplier
                if isinstance(multiplier, torch.Tensor):
                    norm_multiplier = multiplier.clone().detach() * 10
                    norm_multiplier = norm_multiplier.clamp(min=-1.0, max=1.0)
                else:
                    norm_multiplier = multiplier

                # get a dim array from orig forward that had index of all dimensions except the batch and channel

                # Calculate the target magnitude for the combined output
                orig_max = torch.max(torch.abs(org_forwarded))

                # Calculate the additional increase in magnitude that lora_output would introduce
                potential_max_increase = torch.max(
                    torch.abs(org_forwarded + lora_output * norm_multiplier) - torch.abs(org_forwarded))

                epsilon = 1e-6  # Small constant to avoid division by zero

                # Calculate the scaling factor for the lora_output
                # to ensure that the potential increase in magnitude doesn't change the original max
                normalize_scaler = orig_max / (orig_max + potential_max_increase + epsilon)
                normalize_scaler = normalize_scaler.detach()

                # save the scaler so it can be applied later
                self.normalize_scaler = normalize_scaler.clone().detach()

            lora_output = lora_output * normalize_scaler

        return org_forwarded + broadcast_and_multiply(lora_output, multiplier)

    def enable_gradient_checkpointing(self: Module):
        self.is_checkpointing = True

    def disable_gradient_checkpointing(self: Module):
        self.is_checkpointing = False

    @torch.no_grad()
    def apply_stored_normalizer(self: Module, target_normalize_scaler: float = 1.0):
        """
        Applied the previous normalization calculation to the module.
        This must be called before saving or normalization will be lost.
        It is probably best to call after each batch as well.
        We just scale the up down weights to match this vector
        :return:
        """
        # get state dict
        state_dict = self.state_dict()
        dtype = state_dict['lora_up.weight'].dtype
        device = state_dict['lora_up.weight'].device

        # todo should we do this at fp32?
        if isinstance(self.normalize_scaler, torch.Tensor):
            scaler = self.normalize_scaler.clone().detach()
        else:
            scaler = torch.tensor(self.normalize_scaler).to(device, dtype=dtype)

        total_module_scale = scaler / target_normalize_scaler
        num_modules_layers = 2  # up and down
        up_down_scale = torch.pow(total_module_scale, 1.0 / num_modules_layers) \
            .to(device, dtype=dtype)

        # apply the scaler to the up and down weights
        for key in state_dict.keys():
            if key.endswith('.lora_up.weight') or key.endswith('.lora_down.weight'):
                # do it inplace do params are updated
                state_dict[key] *= up_down_scale

        # reset the normalization scaler
        self.normalize_scaler = target_normalize_scaler


class ToolkitNetworkMixin:
    def __init__(
            self: Network,
            *args,
            train_text_encoder: Optional[bool] = True,
            train_unet: Optional[bool] = True,
            is_sdxl=False,
            is_v2=False,
            **kwargs
    ):
        self.train_text_encoder = train_text_encoder
        self.train_unet = train_unet
        self.is_checkpointing = False
        self._multiplier: float = 1.0
        self.is_active: bool = False
        self._is_normalizing: bool = False
        self.is_sdxl = is_sdxl
        self.is_v2 = is_v2
        # super().__init__(*args, **kwargs)

    def get_keymap(self: Network):
        if self.is_sdxl:
            keymap_tail = 'sdxl'
        elif self.is_v2:
            keymap_tail = 'sd2'
        else:
            keymap_tail = 'sd1'
        # load keymap
        keymap_name = f"stable_diffusion_locon_{keymap_tail}.json"
        keymap_path = os.path.join(KEYMAPS_ROOT, keymap_name)

        keymap = None
        # check if file exists
        if os.path.exists(keymap_path):
            with open(keymap_path, 'r') as f:
                keymap = json.load(f)['ldm_diffusers_keymap']

        return keymap

    def save_weights(
            self: Network,
            file, dtype=torch.float16,
            metadata=None,
            extra_state_dict: Optional[OrderedDict] = None
    ):
        keymap = self.get_keymap()

        save_keymap = {}
        if keymap is not None:
            for ldm_key, diffusers_key in keymap.items():
                #  invert them
                save_keymap[diffusers_key] = ldm_key

        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        save_dict = OrderedDict()

        for key in list(state_dict.keys()):
            v = state_dict[key]
            v = v.detach().clone().to("cpu").to(dtype)
            save_key = save_keymap[key] if key in save_keymap else key
            save_dict[save_key] = v

        if extra_state_dict is not None:
            # add extra items to state dict
            for key in list(extra_state_dict.keys()):
                v = extra_state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                save_dict[key] = v

        if metadata is None:
            metadata = OrderedDict()
        metadata = add_model_hash_to_meta(state_dict, metadata)
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            save_file(save_dict, file, metadata)
        else:
            torch.save(save_dict, file)

    def load_weights(self: Network, file):
        # allows us to save and load to and from ldm weights
        keymap = self.get_keymap()
        keymap = {} if keymap is None else keymap

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        load_sd = OrderedDict()
        for key, value in weights_sd.items():
            load_key = keymap[key] if key in keymap else key
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

        info = self.load_state_dict(load_sd, False)
        if len(extra_dict.keys()) == 0:
            extra_dict = None
        return extra_dict

    def _update_torch_multiplier(self: Network):
        # builds a tensor for fast usage in the forward pass of the network modules
        # without having to set it in every single module every time it changes
        multiplier = self._multiplier
        # get first module
        first_module = self.get_all_modules()[0]
        device = first_module.lora_down.weight.device
        dtype = first_module.lora_down.weight.dtype
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
    def multiplier(self) -> Union[float, List[float]]:
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Union[float, List[float]]):
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

    @property
    def is_normalizing(self: Network) -> bool:
        return self._is_normalizing

    @is_normalizing.setter
    def is_normalizing(self: Network, value: bool):
        self._is_normalizing = value
        for module in self.get_all_modules():
            module.is_normalizing = self._is_normalizing

    def apply_stored_normalizer(self: Network, target_normalize_scaler: float = 1.0):
        for module in self.get_all_modules():
            module.apply_stored_normalizer(target_normalize_scaler)
