import json
import os
from collections import OrderedDict
from typing import Optional, Union, List, Type, TYPE_CHECKING

import torch
from torch import nn

from toolkit.metadata import add_model_hash_to_meta
from toolkit.paths import KEYMAPS_ROOT

if TYPE_CHECKING:
    from toolkit.lycoris_special import LycorisSpecialNetwork, LoConSpecialModule
    from toolkit.lora_special import LoRASpecialNetwork, LoRAModule

Network = Union['LycorisSpecialNetwork', 'LoRASpecialNetwork']
Module = Union['LoConSpecialModule', 'LoRAModule']


class ToolkitModuleMixin:
    def __init__(
            self: Module,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.is_checkpointing = False
        self.is_normalizing = False
        self.normalize_scaler = 1.0

    # this allows us to set different multipliers on a per item in a batch basis
    # allowing us to run positive and negative weights in the same batch
    # really only useful for slider training for now
    def get_multiplier(self: Module, lora_up):
        with torch.no_grad():
            batch_size = lora_up.size(0)
            # batch will have all negative prompts first and positive prompts second
            # our multiplier list is for a prompt pair. So we need to repeat it for positive and negative prompts
            # if there is more than our multiplier, it is likely a batch size increase, so we need to
            # interleave the multipliers
            if isinstance(self.multiplier, list):
                if len(self.multiplier) == 0:
                    # single item, just return it
                    return self.multiplier[0]
                elif len(self.multiplier) == batch_size:
                    # not doing CFG
                    multiplier_tensor = torch.tensor(self.multiplier).to(lora_up.device, dtype=lora_up.dtype)
                else:

                    # we have a list of multipliers, so we need to get the multiplier for this batch
                    multiplier_tensor = torch.tensor(self.multiplier * 2).to(lora_up.device, dtype=lora_up.dtype)
                    # should be 1 for if total batch size was 1
                    num_interleaves = (batch_size // 2) // len(self.multiplier)
                    multiplier_tensor = multiplier_tensor.repeat_interleave(num_interleaves)

                # match lora_up rank
                if len(lora_up.size()) == 2:
                    multiplier_tensor = multiplier_tensor.view(-1, 1)
                elif len(lora_up.size()) == 3:
                    multiplier_tensor = multiplier_tensor.view(-1, 1, 1)
                elif len(lora_up.size()) == 4:
                    multiplier_tensor = multiplier_tensor.view(-1, 1, 1, 1)
                return multiplier_tensor.detach()

            else:
                return self.multiplier

    def _call_forward(self: Module, x):
        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return 0.0  # added to original forward

        if hasattr(self, 'lora_mid') and hasattr(self, 'cp') and self.cp:
            lx = self.lora_mid(self.lora_down(x))
        else:
            lx = self.lora_down(x)

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
            scale *= self.scalar

        return lx * scale

    def forward(self: Module, x):
        org_forwarded = self.org_forward(x)
        lora_output = self._call_forward(x)
        multiplier = self.get_multiplier(lora_output)

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

            lora_output *= normalize_scaler

        return org_forwarded + (lora_output * multiplier)

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
        super().__init__(*args, **kwargs)

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
                keymap = json.load(f)

        return keymap

    def save_weights(self: Network, file, dtype=torch.float16, metadata=None):
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

        info = self.load_state_dict(load_sd, False)
        return info

    @property
    def multiplier(self) -> Union[float, List[float]]:
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Union[float, List[float]]):
        self._multiplier = value
        self._update_lora_multiplier()

    def _update_lora_multiplier(self: Network):
        if self.is_active:
            if hasattr(self, 'unet_loras'):
                for lora in self.unet_loras:
                    lora.multiplier = self._multiplier
            if hasattr(self, 'text_encoder_loras'):
                for lora in self.text_encoder_loras:
                    lora.multiplier = self._multiplier
        else:
            if hasattr(self, 'unet_loras'):
                for lora in self.unet_loras:
                    lora.multiplier = 0
            if hasattr(self, 'text_encoder_loras'):
                for lora in self.text_encoder_loras:
                    lora.multiplier = 0

    # called when the context manager is entered
    # ie: with network:
    def __enter__(self: Network):
        self.is_active = True
        self._update_lora_multiplier()

    def __exit__(self: Network, exc_type, exc_value, tb):
        self.is_active = False
        self._update_lora_multiplier()

    def force_to(self: Network, device, dtype):
        self.to(device, dtype)
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        for lora in loras:
            lora.to(device, dtype)

    def get_all_modules(self: Network):
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

    # def enable_gradient_checkpointing(self: Network):
    #     # not supported
    #     self.is_checkpointing = True
    #     self._update_checkpointing()
    #
    # def disable_gradient_checkpointing(self: Network):
    #     # not supported
    #     self.is_checkpointing = False
    #     self._update_checkpointing()

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
