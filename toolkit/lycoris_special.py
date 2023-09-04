import os
from typing import Optional, Union, List, Type

from lycoris.kohya import LycorisNetwork, LoConModule
from torch import nn
from transformers import CLIPTextModel

from toolkit.network_mixins import ToolkitNetworkMixin, ToolkitModuleMixin


class LoConSpecialModule(ToolkitModuleMixin, LoConModule):
    def __init__(
            self,
            lora_name,
            org_module: nn.Module,
            multiplier=1.0,
            lora_dim=4, alpha=1,
            dropout=0., rank_dropout=0., module_dropout=0.,
            use_cp=False,
            **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim, alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            use_cp=use_cp,
            **kwargs,
        )


class LycorisSpecialNetwork(ToolkitNetworkMixin, LycorisNetwork):
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
            use_cp: Optional[bool] = False,
            network_module: Type[object] = LoConSpecialModule,
            **kwargs,
    ) -> None:
        # LyCORIS unique stuff
        if dropout is None:
            dropout = 0
        if rank_dropout is None:
            rank_dropout = 0
        if module_dropout is None:
            module_dropout = 0

        super().__init__(
            text_encoder,
            unet,
            multiplier=multiplier,
            lora_dim=lora_dim,
            conv_lora_dim=conv_lora_dim,
            alpha=alpha,
            conv_alpha=conv_alpha,
            use_cp=use_cp,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            network_module=network_module,
            **kwargs,
        )

