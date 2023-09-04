import json
import math
import os
import re
import sys
from typing import List, Optional, Dict, Type, Union

import torch
from transformers import CLIPTextModel

from .network_mixins import ToolkitNetworkMixin, ToolkitModuleMixin
from .paths import SD_SCRIPTS_ROOT

sys.path.append(SD_SCRIPTS_ROOT)

from networks.lora import LoRANetwork, get_block_index

from torch.utils.checkpoint import checkpoint

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear'
    # 'GroupNorm',
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]

class LoRAModule(ToolkitModuleMixin, torch.nn.Module):
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
        self.scalar = torch.tensor(1.0)

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


class LoRASpecialNetwork(ToolkitNetworkMixin, LoRANetwork):
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
            is_sdxl=False,
            is_v2=False,
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
        self.is_sdxl = is_sdxl
        self.is_v2 = is_v2

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
                        is_linear = child_module.__class__.__name__.in_(LINEAR_MODULES)
                        is_conv2d = child_module.__class__.__name__.in_(CONV_MODULES)
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
