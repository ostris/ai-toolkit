from typing import Union, Tuple, Literal, Optional

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from torch import Tensor
from tqdm import tqdm

from toolkit.config_modules import LoRMConfig

conv = nn.Conv2d
lin = nn.Linear
_size_2_t = Union[int, Tuple[int, int]]

ExtractMode = Union[
    'fixed',
    'threshold',
    'ratio',
    'quantile',
    'percentage'
]

LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear'
]
CONV_MODULES = [
    # 'Conv2d',
    # 'LoRACompatibleConv'
]

UNET_TARGET_REPLACE_MODULE = [
    "Transformer2DModel",
    # "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
]

LORM_TARGET_REPLACE_MODULE = UNET_TARGET_REPLACE_MODULE

UNET_TARGET_REPLACE_NAME = [
    "conv_in",
    "conv_out",
    "time_embedding.linear_1",
    "time_embedding.linear_2",
]

UNET_MODULES_TO_AVOID = [
]


# Low Rank Convolution
class LoRMCon2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            lorm_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 'same',
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.lorm_channels = lorm_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=lorm_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        # Kernel size on the up is always 1x1.
        # I don't think you could calculate a dual 3x3, or I can't at least

        self.up = nn.Conv2d(
            in_channels=lorm_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding='same',
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode='zeros',
            device=device,
            dtype=dtype
        )

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        x = input
        x = self.down(x)
        x = self.up(x)
        return x


class LoRMLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            lorm_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.lorm_features = lorm_features
        self.out_features = out_features

        self.down = nn.Linear(
            in_features=in_features,
            out_features=lorm_features,
            bias=False,
            device=device,
            dtype=dtype

        )
        self.up = nn.Linear(
            in_features=lorm_features,
            out_features=out_features,
            bias=bias,
            # bias=True,
            device=device,
            dtype=dtype
        )

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        x = input
        x = self.down(x)
        x = self.up(x)
        return x


def extract_conv(
        weight: Union[torch.Tensor, nn.Parameter],
        mode='fixed',
        mode_param=0,
        device='cpu'
) -> Tuple[Tensor, Tensor, int, Tensor]:
    weight = weight.to(device)
    out_ch, in_ch, kernel_size, _ = weight.shape

    U, S, Vh = torch.linalg.svd(weight.reshape(out_ch, -1))
    if mode == 'percentage':
        assert 0 <= mode_param <= 1  # Ensure it's a valid percentage.
        original_params = out_ch * in_ch * kernel_size * kernel_size
        desired_params = mode_param * original_params
        # Solve for lora_rank from the equation
        lora_rank = int(desired_params / (in_ch * kernel_size * kernel_size + out_ch))
    elif mode == 'fixed':
        lora_rank = mode_param
    elif mode == 'threshold':
        assert mode_param >= 0
        lora_rank = torch.sum(S > mode_param).item()
    elif mode == 'ratio':
        assert 1 >= mode_param >= 0
        min_s = torch.max(S) * mode_param
        lora_rank = torch.sum(S > min_s).item()
    elif mode == 'quantile' or mode == 'percentile':
        assert 1 >= mode_param >= 0
        s_cum = torch.cumsum(S, dim=0)
        min_cum_sum = mode_param * torch.sum(S)
        lora_rank = torch.sum(s_cum < min_cum_sum).item()
    else:
        raise NotImplementedError('Extract mode should be "fixed", "threshold", "ratio" or "quantile"')
    lora_rank = max(1, lora_rank)
    lora_rank = min(out_ch, in_ch, lora_rank)
    if lora_rank >= out_ch / 2:
        lora_rank = int(out_ch / 2)
        print(f"rank is higher than it should be")
        # print(f"Skipping layer as determined rank is too high")
        # return None, None, None, None
        # return weight, 'full'

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    diff = (weight - (U @ Vh).reshape(out_ch, in_ch, kernel_size, kernel_size)).detach()
    extract_weight_A = Vh.reshape(lora_rank, in_ch, kernel_size, kernel_size).detach()
    extract_weight_B = U.reshape(out_ch, lora_rank, 1, 1).detach()
    del U, S, Vh, weight
    return extract_weight_A, extract_weight_B, lora_rank, diff


def extract_linear(
        weight: Union[torch.Tensor, nn.Parameter],
        mode='fixed',
        mode_param=0,
        device='cpu',
) -> Tuple[Tensor, Tensor, int, Tensor]:
    weight = weight.to(device)
    out_ch, in_ch = weight.shape

    U, S, Vh = torch.linalg.svd(weight)

    if mode == 'percentage':
        assert 0 <= mode_param <= 1  # Ensure it's a valid percentage.
        desired_params = mode_param * out_ch * in_ch
        # Solve for lora_rank from the equation
        lora_rank = int(desired_params / (in_ch + out_ch))
    elif mode == 'fixed':
        lora_rank = mode_param
    elif mode == 'threshold':
        assert mode_param >= 0
        lora_rank = torch.sum(S > mode_param).item()
    elif mode == 'ratio':
        assert 1 >= mode_param >= 0
        min_s = torch.max(S) * mode_param
        lora_rank = torch.sum(S > min_s).item()
    elif mode == 'quantile':
        assert 1 >= mode_param >= 0
        s_cum = torch.cumsum(S, dim=0)
        min_cum_sum = mode_param * torch.sum(S)
        lora_rank = torch.sum(s_cum < min_cum_sum).item()
    else:
        raise NotImplementedError('Extract mode should be "fixed", "threshold", "ratio" or "quantile"')
    lora_rank = max(1, lora_rank)
    lora_rank = min(out_ch, in_ch, lora_rank)
    if lora_rank >= out_ch / 2:
        # print(f"rank is higher than it should be")
        lora_rank = int(out_ch / 2)
        # return weight, 'full'
        # print(f"Skipping layer as determined rank is too high")
        # return None, None, None, None

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    diff = (weight - U @ Vh).detach()
    extract_weight_A = Vh.reshape(lora_rank, in_ch).detach()
    extract_weight_B = U.reshape(out_ch, lora_rank).detach()
    del U, S, Vh, weight
    return extract_weight_A, extract_weight_B, lora_rank, diff


def replace_module_by_path(network, name, module):
    """Replace a module in a network by its name."""
    name_parts = name.split('.')
    current_module = network
    for part in name_parts[:-1]:
        current_module = getattr(current_module, part)
    try:
        setattr(current_module, name_parts[-1], module)
    except Exception as e:
        print(e)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def compute_optimal_bias(original_module, linear_down, linear_up, X):
    Y_original = original_module(X)
    Y_approx = linear_up(linear_down(X))
    E = Y_original - Y_approx

    optimal_bias = E.mean(dim=0)

    return optimal_bias


def format_with_commas(n):
    return f"{n:,}"


def print_lorm_extract_details(
        start_num_params: int,
        end_num_params: int,
        num_replaced: int,
):
    start_formatted = format_with_commas(start_num_params)
    end_formatted = format_with_commas(end_num_params)
    num_replaced_formatted = format_with_commas(num_replaced)

    width = max(len(start_formatted), len(end_formatted), len(num_replaced_formatted))

    print(f"Convert UNet result:")
    print(f" - converted: {num_replaced:>{width},} modules")
    print(f" -     start: {start_num_params:>{width},} params")
    print(f" -       end: {end_num_params:>{width},} params")


lorm_ignore_if_contains = [
    'proj_out', 'proj_in',
]

lorm_parameter_threshold = 1000000


@torch.no_grad()
def convert_diffusers_unet_to_lorm(
        unet: UNet2DConditionModel,
        config: LoRMConfig,
):
    print('Converting UNet to LoRM UNet')
    start_num_params = count_parameters(unet)
    named_modules = list(unet.named_modules())

    num_replaced = 0

    pbar = tqdm(total=len(named_modules), desc="UNet -> LoRM UNet")
    layer_names_replaced = []
    converted_modules = []
    ignore_if_contains = [
        'proj_out', 'proj_in',
    ]

    for name, module in named_modules:
        module_name = module.__class__.__name__
        if module_name in UNET_TARGET_REPLACE_MODULE:
            for child_name, child_module in module.named_modules():
                new_module: Union[LoRMCon2d, LoRMLinear, None] = None
                # if child name includes attn, skip it
                combined_name = combined_name = f"{name}.{child_name}"
                # if child_module.__class__.__name__ in LINEAR_MODULES and child_module.bias is None:
                #     pass

                lorm_config = config.get_config_for_module(combined_name)

                extract_mode = lorm_config.extract_mode
                extract_mode_param = lorm_config.extract_mode_param
                parameter_threshold = lorm_config.parameter_threshold

                if any([word in child_name for word in ignore_if_contains]):
                    pass

                elif child_module.__class__.__name__ in LINEAR_MODULES:
                    if count_parameters(child_module) > parameter_threshold:

                        dtype = child_module.weight.dtype
                        # extract and convert
                        down_weight, up_weight, lora_dim, diff = extract_linear(
                            weight=child_module.weight.clone().detach().float(),
                            mode=extract_mode,
                            mode_param=extract_mode_param,
                            device=child_module.weight.device,
                        )
                        if down_weight is None:
                            continue
                        down_weight = down_weight.to(dtype=dtype)
                        up_weight = up_weight.to(dtype=dtype)
                        bias_weight = None
                        if child_module.bias is not None:
                            bias_weight = child_module.bias.data.clone().detach().to(dtype=dtype)
                        # linear layer weights = (out_features, in_features)
                        new_module = LoRMLinear(
                            in_features=down_weight.shape[1],
                            lorm_features=lora_dim,
                            out_features=up_weight.shape[0],
                            bias=bias_weight is not None,
                            device=down_weight.device,
                            dtype=down_weight.dtype
                        )

                        # replace the weights
                        new_module.down.weight.data = down_weight
                        new_module.up.weight.data = up_weight
                        if bias_weight is not None:
                            new_module.up.bias.data = bias_weight
                        # else:
                        #     new_module.up.bias.data = torch.zeros_like(new_module.up.bias.data)

                        # bias_correction = compute_optimal_bias(
                        #     child_module,
                        #     new_module.down,
                        #     new_module.up,
                        #     torch.randn((1000, down_weight.shape[1])).to(device=down_weight.device, dtype=dtype)
                        #     )
                        # new_module.up.bias.data += bias_correction

                elif child_module.__class__.__name__ in CONV_MODULES:
                    if count_parameters(child_module) > parameter_threshold:
                        dtype = child_module.weight.dtype
                        down_weight, up_weight, lora_dim, diff = extract_conv(
                            weight=child_module.weight.clone().detach().float(),
                            mode=extract_mode,
                            mode_param=extract_mode_param,
                            device=child_module.weight.device,
                        )
                        if down_weight is None:
                            continue
                        down_weight = down_weight.to(dtype=dtype)
                        up_weight = up_weight.to(dtype=dtype)
                        bias_weight = None
                        if child_module.bias is not None:
                            bias_weight = child_module.bias.data.clone().detach().to(dtype=dtype)

                        new_module = LoRMCon2d(
                            in_channels=down_weight.shape[1],
                            lorm_channels=lora_dim,
                            out_channels=up_weight.shape[0],
                            kernel_size=child_module.kernel_size,
                            dilation=child_module.dilation,
                            padding=child_module.padding,
                            padding_mode=child_module.padding_mode,
                            stride=child_module.stride,
                            bias=bias_weight is not None,
                            device=down_weight.device,
                            dtype=down_weight.dtype
                        )
                        # replace the weights
                        new_module.down.weight.data = down_weight
                        new_module.up.weight.data = up_weight
                        if bias_weight is not None:
                            new_module.up.bias.data = bias_weight

                if new_module:
                    combined_name = f"{name}.{child_name}"
                    replace_module_by_path(unet, combined_name, new_module)
                    converted_modules.append(new_module)
                    num_replaced += 1
                    layer_names_replaced.append(
                        f"{combined_name} - {format_with_commas(count_parameters(child_module))}")

                pbar.update(1)
    pbar.close()
    end_num_params = count_parameters(unet)

    def sorting_key(s):
        # Extract the number part, remove commas, and convert to integer
        return int(s.split("-")[1].strip().replace(",", ""))

    sorted_layer_names_replaced = sorted(layer_names_replaced, key=sorting_key, reverse=True)
    for layer_name in sorted_layer_names_replaced:
        print(layer_name)

    print_lorm_extract_details(
        start_num_params=start_num_params,
        end_num_params=end_num_params,
        num_replaced=num_replaced,
    )

    return converted_modules
