import math
import os
import re
import sys
from typing import List, Optional, Dict, Type, Union

import torch
from transformers import CLIPTextModel

from .paths import SD_SCRIPTS_ROOT
from .train_tools import get_torch_dtype

sys.path.append(SD_SCRIPTS_ROOT)

from networks.lora import LoRANetwork, get_block_index

from torch.utils.checkpoint import checkpoint

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
            self,
            lora_name,
            org_module: torch.nn.Module,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        # if limit_rank:
        #   self.lora_dim = min(lora_dim, in_dim, out_dim)
        #   if self.lora_dim != lora_dim:
        #     print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
        # else:
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier: Union[float, List[float]] = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.is_checkpointing = False
        self.is_normalizing = False
        self.normalize_scaler = 1.0

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    # this allows us to set different multipliers on a per item in a batch basis
    # allowing us to run positive and negative weights in the same batch
    # really only useful for slider training for now
    def get_multiplier(self, lora_up):
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

    def _call_forward(self, x):
        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return 0.0  # added to original forward

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
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

        return lx * scale

    def forward(self, x):
        org_forwarded = self.org_forward(x)
        lora_output = self._call_forward(x)

        if self.is_normalizing:
            with torch.no_grad():
                # do this calculation without multiplier
                # get a dim array from orig forward that had index of all dimensions except the batch and channel

                # Calculate the target magnitude for the combined output
                orig_max = torch.max(torch.abs(org_forwarded))

                # Calculate the additional increase in magnitude that lora_output would introduce
                potential_max_increase = torch.max(torch.abs(org_forwarded + lora_output) - torch.abs(org_forwarded))

                epsilon = 1e-6  # Small constant to avoid division by zero

                # Calculate the scaling factor for the lora_output
                # to ensure that the potential increase in magnitude doesn't change the original max
                normalize_scaler = orig_max / (orig_max + potential_max_increase + epsilon)
                normalize_scaler = normalize_scaler.detach()

                # save the scaler so it can be applied later
                self.normalize_scaler = normalize_scaler.clone().detach()

            lora_output *= normalize_scaler

        multiplier = self.get_multiplier(lora_output)

        return org_forwarded + (lora_output * multiplier)

    def enable_gradient_checkpointing(self):
        self.is_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.is_checkpointing = False

    @torch.no_grad()
    def apply_stored_normalizer(self, target_normalize_scaler: float = 1.0):
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

        total_module_scale = torch.tensor(self.normalize_scaler / target_normalize_scaler) \
            .to(device, dtype=dtype)
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


class LoRASpecialNetwork(LoRANetwork):
    NUM_OF_BLOCKS = 12  # フルモデル相当でのup,downの層の数

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
    LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

    def __init__(
            self,
            text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
            unet,
            multiplier: float = 1.0,
            lora_dim: int = 4,
            alpha: float = 1,
            dropout: Optional[float] = None,
            rank_dropout: Optional[float] = None,
            module_dropout: Optional[float] = None,
            conv_lora_dim: Optional[int] = None,
            conv_alpha: Optional[float] = None,
            block_dims: Optional[List[int]] = None,
            block_alphas: Optional[List[float]] = None,
            conv_block_dims: Optional[List[int]] = None,
            conv_block_alphas: Optional[List[float]] = None,
            modules_dim: Optional[Dict[str, int]] = None,
            modules_alpha: Optional[Dict[str, int]] = None,
            module_class: Type[object] = LoRAModule,
            varbose: Optional[bool] = False,
            train_text_encoder: Optional[bool] = True,
            train_unet: Optional[bool] = True,
    ) -> None:
        """
        LoRA network: すごく引数が多いが、パターンは以下の通り
        1. lora_dimとalphaを指定
        2. lora_dim、alpha、conv_lora_dim、conv_alphaを指定
        3. block_dimsとblock_alphasを指定 :  Conv2d3x3には適用しない
        4. block_dims、block_alphas、conv_block_dims、conv_block_alphasを指定 : Conv2d3x3にも適用する
        5. modules_dimとmodules_alphaを指定 (推論用)
        """
        # call the parent of the parent we are replacing (LoRANetwork) init
        super(LoRANetwork, self).__init__()

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.is_checkpointing = False
        self._multiplier: float = 1.0
        self.is_active: bool = False
        self._is_normalizing: bool = False
        # triggers the state updates
        self.multiplier = multiplier

        if modules_dim is not None:
            print(f"create LoRA network from weights")
        elif block_dims is not None:
            print(f"create LoRA network from block_dims")
            print(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")
            print(f"block_dims: {block_dims}")
            print(f"block_alphas: {block_alphas}")
            if conv_block_dims is not None:
                print(f"conv_block_dims: {conv_block_dims}")
                print(f"conv_block_alphas: {conv_block_alphas}")
        else:
            print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            print(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")
            if self.conv_lora_dim is not None:
                print(
                    f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}")

        # create module instances
        def create_modules(
                is_unet: bool,
                text_encoder_idx: Optional[int],  # None, 1, 2
                root_module: torch.nn.Module,
                target_replace_modules: List[torch.nn.Module],
        ) -> List[LoRAModule]:
            prefix = (
                self.LORA_PREFIX_UNET
                if is_unet
                else (
                    self.LORA_PREFIX_TEXT_ENCODER
                    if text_encoder_idx is None
                    else (self.LORA_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LORA_PREFIX_TEXT_ENCODER2)
                )
            )
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            elif is_unet and block_dims is not None:
                                # U-Netでblock_dims指定あり
                                block_idx = get_block_index(lora_name)
                                if is_linear or is_conv2d_1x1:
                                    dim = block_dims[block_idx]
                                    alpha = block_alphas[block_idx]
                                elif conv_block_dims is not None:
                                    dim = conv_block_dims[block_idx]
                                    alpha = conv_block_alphas[block_idx]
                            else:
                                # 通常、すべて対象とする
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if is_linear or is_conv2d_1x1 or (
                                        self.conv_lora_dim is not None or conv_block_dims is not None):
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                            )
                            loras.append(lora)
            return loras, skipped

        text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]

        # create LoRA for text encoder
        # 毎回すべてのモジュールを作るのは無駄なので要検討
        self.text_encoder_loras = []
        skipped_te = []
        if train_text_encoder:
            for i, text_encoder in enumerate(text_encoders):
                if len(text_encoders) > 1:
                    index = i + 1
                    print(f"create LoRA for Text Encoder {index}:")
                else:
                    index = None
                    print(f"create LoRA for Text Encoder:")

                text_encoder_loras, skipped = create_modules(False, index, text_encoder,
                                                             LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
                self.text_encoder_loras.extend(text_encoder_loras)
                skipped_te += skipped
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.conv_lora_dim is not None or conv_block_dims is not None:
            target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        if train_unet:
            self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)
        else:
            self.unet_loras = []
            skipped_un = []
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        skipped = skipped_te + skipped_un
        if varbose and len(skipped) > 0:
            print(
                f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} LoRA modules are skipped / block_lr_weightまたはdim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                print(f"\t{name}")

        self.up_lr_weight: List[float] = None
        self.down_lr_weight: List[float] = None
        self.mid_lr_weight: float = None
        self.block_lr = False

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    @property
    def multiplier(self) -> Union[float, List[float]]:
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Union[float, List[float]]):
        self._multiplier = value
        self._update_lora_multiplier()

    def _update_lora_multiplier(self):

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
    def __enter__(self):
        self.is_active = True
        self._update_lora_multiplier()

    def __exit__(self, exc_type, exc_value, tb):
        self.is_active = False
        self._update_lora_multiplier()

    def force_to(self, device, dtype):
        self.to(device, dtype)
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        for lora in loras:
            lora.to(device, dtype)

    def get_all_modules(self):
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        return loras

    def _update_checkpointing(self):
        for module in self.get_all_modules():
            if self.is_checkpointing:
                module.enable_gradient_checkpointing()
            else:
                module.disable_gradient_checkpointing()

    def enable_gradient_checkpointing(self):
        # not supported
        self.is_checkpointing = True
        self._update_checkpointing()

    def disable_gradient_checkpointing(self):
        # not supported
        self.is_checkpointing = False
        self._update_checkpointing()

    @property
    def is_normalizing(self) -> bool:
        return self._is_normalizing

    @is_normalizing.setter
    def is_normalizing(self, value: bool):
        self._is_normalizing = value
        for module in self.get_all_modules():
            module.is_normalizing = self._is_normalizing

    def apply_stored_normalizer(self, target_normalize_scaler: float = 1.0):
        for module in self.get_all_modules():
            module.apply_stored_normalizer(target_normalize_scaler)
