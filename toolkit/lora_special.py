import copy
import json
import math
import weakref
import os
import re
import sys
from typing import List, Optional, Dict, Type, Union
import torch
from diffusers import UNet2DConditionModel, PixArtTransformer2DModel, AuraFlowTransformer2DModel
from transformers import CLIPTextModel
from toolkit.models.lokr import LokrModule

from .config_modules import NetworkConfig
from .lorm import count_parameters
from .network_mixins import ToolkitNetworkMixin, ToolkitModuleMixin, ExtractableModuleMixin

from toolkit.kohya_lora import LoRANetwork
from toolkit.models.DoRA import DoRAModule
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear',
    'QLinear',
    # 'GroupNorm',
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv',
    'QConv2d',
]

class LoRAModule(ToolkitModuleMixin, ExtractableModuleMixin, torch.nn.Module):
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
            network: 'LoRASpecialNetwork' = None,
            use_bias: bool = False,
            **kwargs
    ):
        self.can_merge_in = True
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)
        self.lora_name = lora_name
        self.orig_module_ref = weakref.ref(org_module)
        self.scalar = torch.tensor(1.0, device=org_module.weight.device)
        # check if parent has bias. if not force use_bias to False
        if org_module.bias is None:
            use_bias = False

        if org_module.__class__.__name__ in CONV_MODULES:
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

        if org_module.__class__.__name__ in CONV_MODULES:
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=use_bias)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=use_bias)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier: Union[float, List[float]] = multiplier
        # wrap the original module so it doesn't get weights updated
        self.org_module = [org_module]
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.is_checkpointing = False

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
        # del self.org_module


class LoRASpecialNetwork(ToolkitNetworkMixin, LoRANetwork):
    NUM_OF_BLOCKS = 12  # フルモデル相当でのup,downの層の数

    # UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    # UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "ResnetBlock2D"]
    UNET_TARGET_REPLACE_MODULE = ["UNet2DConditionModel"]
    # UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["UNet2DConditionModel"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    PEFT_PREFIX_UNET = "unet"
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
            use_text_encoder_1: bool = True,
            use_text_encoder_2: bool = True,
            train_unet: Optional[bool] = True,
            is_sdxl=False,
            is_v2=False,
            is_v3=False,
            is_pixart: bool = False,
            is_auraflow: bool = False,
            is_flux: bool = False,
            is_lumina2: bool = False,
            use_bias: bool = False,
            is_lorm: bool = False,
            ignore_if_contains = None,
            only_if_contains = None,
            parameter_threshold: float = 0.0,
            attn_only: bool = False,
            target_lin_modules=LoRANetwork.UNET_TARGET_REPLACE_MODULE,
            target_conv_modules=LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3,
            network_type: str = "lora",
            full_train_in_out: bool = False,
            transformer_only: bool = False,
            peft_format: bool = False,
            is_assistant_adapter: bool = False,
            is_transformer: bool = False,
            base_model: 'StableDiffusion' = None,
            **kwargs
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
        torch.nn.Module.__init__(self)
        ToolkitNetworkMixin.__init__(
            self,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            is_sdxl=is_sdxl,
            is_v2=is_v2,
            is_lorm=is_lorm,
            **kwargs
        )
        if ignore_if_contains is None:
            ignore_if_contains = []
        self.ignore_if_contains = ignore_if_contains
        self.transformer_only = transformer_only
        self.base_model_ref = None
        if base_model is not None:
            self.base_model_ref = weakref.ref(base_model)

        self.only_if_contains: Union[List, None] = only_if_contains

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
        self.torch_multiplier = None
        # triggers the state updates
        self.multiplier = multiplier
        self.is_sdxl = is_sdxl
        self.is_v2 = is_v2
        self.is_v3 = is_v3
        self.is_pixart = is_pixart
        self.is_auraflow = is_auraflow
        self.is_flux = is_flux
        self.is_lumina2 = is_lumina2
        self.network_type = network_type
        self.is_assistant_adapter = is_assistant_adapter
        if self.network_type.lower() == "dora":
            self.module_class = DoRAModule
            module_class = DoRAModule
        elif self.network_type.lower() == "lokr":
            self.module_class = LokrModule
            module_class = LokrModule
        self.network_config: NetworkConfig = kwargs.get("network_config", None)

        self.peft_format = peft_format
        self.is_transformer = is_transformer
        

        # always do peft for flux only for now
        if self.is_flux or self.is_v3 or self.is_lumina2 or is_transformer:
            # don't do peft format for lokr
            if self.network_type.lower() != "lokr":
                self.peft_format = True

        if self.peft_format:
            # no alpha for peft
            self.alpha = self.lora_dim
            alpha = self.alpha
            self.conv_alpha = self.conv_lora_dim
            conv_alpha = self.conv_alpha

        self.full_train_in_out = full_train_in_out

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
            unet_prefix = self.LORA_PREFIX_UNET
            if self.peft_format:
                unet_prefix = self.PEFT_PREFIX_UNET
            if is_pixart or is_v3 or is_auraflow or is_flux or is_lumina2 or self.is_transformer:
                unet_prefix = f"lora_transformer"
                if self.peft_format:
                    unet_prefix = "transformer"

            prefix = (
                unet_prefix
                if is_unet
                else (
                    self.LORA_PREFIX_TEXT_ENCODER
                    if text_encoder_idx is None
                    else (self.LORA_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LORA_PREFIX_TEXT_ENCODER2)
                )
            )
            loras = []
            skipped = []
            attached_modules = []
            lora_shape_dict = {}
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ in LINEAR_MODULES
                        is_conv2d = child_module.__class__.__name__ in CONV_MODULES
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)


                        lora_name = [prefix, name, child_name]
                        # filter out blank
                        lora_name = [x for x in lora_name if x and x != ""]
                        lora_name = ".".join(lora_name)
                        # if it doesnt have a name, it wil have two dots
                        lora_name.replace("..", ".")
                        clean_name = lora_name
                        if self.peft_format:
                            # we replace this on saving
                            lora_name = lora_name.replace(".", "$$")
                        else:
                            lora_name = lora_name.replace(".", "_")

                        skip = False
                        if any([word in clean_name for word in self.ignore_if_contains]):
                            skip = True

                        # see if it is over threshold
                        if count_parameters(child_module) < parameter_threshold:
                            skip = True
                        
                        if self.transformer_only and is_unet:
                            transformer_block_names = None
                            if base_model is not None:
                                transformer_block_names = base_model.get_transformer_block_names()
                            
                            if transformer_block_names is not None:
                                if not any([name in lora_name for name in transformer_block_names]):
                                    skip = True
                            else:
                                if self.is_pixart:
                                    if "transformer_blocks" not in lora_name:
                                        skip = True
                                if self.is_flux:
                                    if "transformer_blocks" not in lora_name:
                                        skip = True
                                if self.is_lumina2:
                                    if "layers$$" not in lora_name and "noise_refiner$$" not in lora_name and "context_refiner$$" not in lora_name:
                                        skip = True
                                if  self.is_v3:
                                    if "transformer_blocks" not in lora_name:
                                        skip = True
                                
                                # handle custom models
                                if hasattr(root_module, 'transformer_blocks'):
                                    if "transformer_blocks" not in lora_name:
                                        skip = True
                                        
                                if hasattr(root_module, 'blocks'):
                                    if "blocks" not in lora_name:
                                        skip = True
                                
                                if hasattr(root_module, 'single_blocks'):
                                    if "single_blocks" not in lora_name and "double_blocks" not in lora_name:
                                        skip = True

                        if (is_linear or is_conv2d) and not skip:

                            if self.only_if_contains is not None:
                                if not any([word in clean_name for word in self.only_if_contains]) and not any([word in lora_name for word in self.only_if_contains]):
                                    continue

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
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
                            
                            module_kwargs = {}
                            
                            if self.network_type.lower() == "lokr":
                                module_kwargs["factor"] = self.network_config.lokr_factor

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                network=self,
                                parent=module,
                                use_bias=use_bias,
                                **module_kwargs
                            )
                            loras.append(lora)
                            if self.network_type.lower() == "lokr":
                                try:
                                    lora_shape_dict[lora_name] = [list(lora.lokr_w1.weight.shape), list(lora.lokr_w2.weight.shape)]
                                except:
                                    pass
                            else:
                                lora_shape_dict[lora_name] = [list(lora.lora_down.weight.shape), list(lora.lora_up.weight.shape)]
            return loras, skipped

        text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]

        # create LoRA for text encoder
        # 毎回すべてのモジュールを作るのは無駄なので要検討
        self.text_encoder_loras = []
        skipped_te = []
        if train_text_encoder:
            for i, text_encoder in enumerate(text_encoders):
                if not use_text_encoder_1 and i == 0:
                    continue
                if not use_text_encoder_2 and i == 1:
                    continue
                if len(text_encoders) > 1:
                    index = i + 1
                    print(f"create LoRA for Text Encoder {index}:")
                else:
                    index = None
                    print(f"create LoRA for Text Encoder:")

                replace_modules = LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE

                if self.is_pixart:
                    replace_modules = ["T5EncoderModel"]

                text_encoder_loras, skipped = create_modules(False, index, text_encoder, replace_modules)
                self.text_encoder_loras.extend(text_encoder_loras)
                skipped_te += skipped
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = target_lin_modules
        if modules_dim is not None or self.conv_lora_dim is not None or conv_block_dims is not None:
            target_modules += target_conv_modules

        if is_v3:
            target_modules = ["SD3Transformer2DModel"]

        if is_pixart:
            target_modules = ["PixArtTransformer2DModel"]

        if is_auraflow:
            target_modules = ["AuraFlowTransformer2DModel"]

        if is_flux:
            target_modules = ["FluxTransformer2DModel"]
        
        if is_lumina2:
            target_modules = ["Lumina2Transformer2DModel"]

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

        if self.full_train_in_out:
            print("full train in out")
            # we are going to retrain the main in out layers for VAE change usually
            if self.is_pixart:
                transformer: PixArtTransformer2DModel = unet
                self.transformer_pos_embed = copy.deepcopy(transformer.pos_embed)
                self.transformer_proj_out = copy.deepcopy(transformer.proj_out)

                transformer.pos_embed = self.transformer_pos_embed
                transformer.proj_out = self.transformer_proj_out

            elif self.is_auraflow:
                transformer: AuraFlowTransformer2DModel = unet
                self.transformer_pos_embed = copy.deepcopy(transformer.pos_embed)
                self.transformer_proj_out = copy.deepcopy(transformer.proj_out)

                transformer.pos_embed = self.transformer_pos_embed
                transformer.proj_out = self.transformer_proj_out

            else:
                unet: UNet2DConditionModel = unet
                unet_conv_in: torch.nn.Conv2d = unet.conv_in
                unet_conv_out: torch.nn.Conv2d = unet.conv_out

                # clone these and replace their forwards with ours
                self.unet_conv_in = copy.deepcopy(unet_conv_in)
                self.unet_conv_out = copy.deepcopy(unet_conv_out)
                unet.conv_in = self.unet_conv_in
                unet.conv_out = self.unet_conv_out

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        # call Lora prepare_optimizer_params
        all_params = super().prepare_optimizer_params(text_encoder_lr, unet_lr, default_lr)

        if self.full_train_in_out:
            if self.is_pixart or self.is_auraflow or self.is_flux:
                all_params.append({"lr": unet_lr, "params": list(self.transformer_pos_embed.parameters())})
                all_params.append({"lr": unet_lr, "params": list(self.transformer_proj_out.parameters())})
            else:
                all_params.append({"lr": unet_lr, "params": list(self.unet_conv_in.parameters())})
                all_params.append({"lr": unet_lr, "params": list(self.unet_conv_out.parameters())})

        return all_params


