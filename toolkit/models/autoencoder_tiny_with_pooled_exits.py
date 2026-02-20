from typing import Optional, Tuple, Union
from diffusers import AutoencoderTiny
from diffusers.models.autoencoders.vae import (
    EncoderTiny,
    get_activation,
    AutoencoderTinyBlock,
    DecoderOutput
)
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.configuration_utils import register_to_config
import torch
import torch.nn as nn

class DecoderTinyWithPooledExits(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        upsampling_scaling_factor: int,
        act_fn: str,
        upsample_fn: str,
    ):
        super().__init__()
        layers = []
        self.ordered_layers = []
        l = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.ordered_layers.append(l)
        layers.append(l)
        l = get_activation(act_fn)
        self.ordered_layers.append(l)
        layers.append(l)

        pooled_exits = []
        

        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)
            num_channels = block_out_channels[i]

            for _ in range(num_block):
                l = AutoencoderTinyBlock(num_channels, num_channels, act_fn)
                layers.append(l)
                self.ordered_layers.append(l)

            if not is_final_block:
                l = nn.Upsample(
                    scale_factor=upsampling_scaling_factor, mode=upsample_fn
                )
                layers.append(l)
                self.ordered_layers.append(l)

            conv_out_channel = num_channels if not is_final_block else out_channels
            l = nn.Conv2d(
                num_channels,
                conv_out_channel,
                kernel_size=3,
                padding=1,
                bias=is_final_block,
            )
            layers.append(l)
            self.ordered_layers.append(l)

            if not is_final_block:
                p = nn.Conv2d(
                    conv_out_channel,
                    out_channels=3,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
                p._is_pooled_exit = True
                pooled_exits.append(p)
                self.ordered_layers.append(p)

        self.layers = nn.ModuleList(layers)
        self.pooled_exits = nn.ModuleList(pooled_exits)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, pooled_outputs=False) -> torch.Tensor:
        r"""The forward method of the `DecoderTiny` class."""
        # Clamp.
        x = torch.tanh(x / 3) * 3

        pooled_output_list = []

        for layer in self.ordered_layers:
            # see if is pooled exit
            try:
                if hasattr(layer, '_is_pooled_exit') and layer._is_pooled_exit:
                    if pooled_outputs:
                        pooled_output = layer(x)
                        pooled_output_list.append(pooled_output)
                else:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        x = self._gradient_checkpointing_func(layer, x)
                    else:
                        x = layer(x)
            except RuntimeError as e:
                raise e

        # scale image from [0, 1] to [-1, 1] to match diffusers convention
        x = x.mul(2).sub(1)

        if pooled_outputs:
            return x, pooled_output_list
        return x


class AutoencoderTinyWithPooledExits(AutoencoderTiny):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        act_fn: str = "relu",
        upsample_fn: str = "nearest",
        latent_channels: int = 4,
        upsampling_scaling_factor: int = 2,
        num_encoder_blocks: Tuple[int, ...] = (1, 3, 3, 3),
        num_decoder_blocks: Tuple[int, ...] = (3, 3, 3, 1),
        latent_magnitude: int = 3,
        latent_shift: float = 0.5,
        force_upcast: bool = False,
        scaling_factor: float = 1.0,
        shift_factor: float = 0.0,
    ):
        super(AutoencoderTiny, self).__init__()

        if len(encoder_block_out_channels) != len(num_encoder_blocks):
            raise ValueError(
                "`encoder_block_out_channels` should have the same length as `num_encoder_blocks`."
            )
        if len(decoder_block_out_channels) != len(num_decoder_blocks):
            raise ValueError(
                "`decoder_block_out_channels` should have the same length as `num_decoder_blocks`."
            )

        self.encoder = EncoderTiny(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_blocks=num_encoder_blocks,
            block_out_channels=encoder_block_out_channels,
            act_fn=act_fn,
        )

        self.decoder = DecoderTinyWithPooledExits(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_blocks=num_decoder_blocks,
            block_out_channels=decoder_block_out_channels,
            upsampling_scaling_factor=upsampling_scaling_factor,
            act_fn=act_fn,
            upsample_fn=upsample_fn,
        )

        self.latent_magnitude = latent_magnitude
        self.latent_shift = latent_shift
        self.scaling_factor = scaling_factor

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.spatial_scale_factor = 2**out_channels
        self.tile_overlap_factor = 0.125
        self.tile_sample_min_size = 512
        self.tile_latent_min_size = (
            self.tile_sample_min_size // self.spatial_scale_factor
        )

        self.register_to_config(block_out_channels=decoder_block_out_channels)
        self.register_to_config(force_upcast=False)
    
    @apply_forward_hook
    def decode_with_pooled_exits(
        self, x: torch.Tensor, generator: Optional[torch.Generator] = None, return_dict: bool = False
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        output, pooled_outputs = self.decoder(x, pooled_outputs=True)

        if not return_dict:
            return (output, pooled_outputs)

        return DecoderOutput(sample=output)
