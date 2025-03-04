# based heavily on https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from toolkit.network_mixins import ToolkitModuleMixin

from typing import TYPE_CHECKING, Union, List

from optimum.quanto import QBytesTensor, QTensor

if TYPE_CHECKING:

    from toolkit.lora_special import LoRASpecialNetwork


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    '''
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale.
    secon value is a value for weight.

    Becuase of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 127, 1   127 -> 127, 1    127 -> 127, 1   127 -> 127, 1   127 -> 127, 1
    128 -> 16, 8    128 -> 64, 2     128 -> 32, 4    128 -> 16, 8    128 -> 16, 8
    250 -> 125, 2   250 -> 125, 2    250 -> 125, 2   250 -> 125, 2   250 -> 125, 2
    360 -> 45, 8    360 -> 180, 2    360 -> 90, 4    360 -> 45, 8    360 -> 45, 8
    512 -> 32, 16   512 -> 256, 2    512 -> 128, 4   512 -> 64, 8    512 -> 32, 16
    1024 -> 32, 32  1024 -> 512, 2   1024 -> 256, 4  1024 -> 128, 8  1024 -> 64, 16
    '''

    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_weight_cp(t, wa, wb):
    rebuild2 = torch.einsum('i j k l, i p, j r -> p r k l',
                            t, wa, wb)  # [c, d, k1, k2]
    return rebuild2


def make_kron(w1, w2, scale):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)

    return rebuild*scale


class LokrModule(ToolkitModuleMixin, nn.Module):
    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.,
        rank_dropout=0.,
        module_dropout=0.,
        use_cp=False,
        decompose_both=False,
        network: 'LoRASpecialNetwork' = None,
        factor: int = -1,  # factorization factor
        **kwargs,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)
        factor = int(factor)
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.cp = False
        self.use_w1 = False
        self.use_w2 = False
        self.can_merge_in = True

        self.shape = org_module.weight.shape
        if org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels

            in_m, in_n = factorization(in_dim, factor)
            out_l, out_k = factorization(out_dim, factor)
            # ((a, b), (c, d), *k_size)
            shape = ((out_l, out_k), (in_m, in_n), *k_size)

            self.cp = use_cp and k_size != (1, 1)
            if decompose_both and lora_dim < max(shape[0][0], shape[1][0])/2:
                self.lokr_w1_a = nn.Parameter(
                    torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(torch.empty(
                    shape[0][0], shape[1][0]))  # a*c, 1-mode

            if lora_dim >= max(shape[0][1], shape[1][1])/2:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(
                    shape[0][1], shape[1][1], *k_size))
            elif self.cp:
                self.lokr_t2 = nn.Parameter(torch.empty(
                    lora_dim, lora_dim, shape[2], shape[3]))
                self.lokr_w2_a = nn.Parameter(
                    torch.empty(lora_dim, shape[0][1]))  # b, 1-mode
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][1]))  # d, 2-mode
            else:  # Conv2d not cp
                # bigger part. weight and LoRA. [b, dim] x [dim, d*k1*k2]
                self.lokr_w2_a = nn.Parameter(
                    torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(
                    lora_dim, shape[1][1]*shape[2]*shape[3]))
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d*k1*k2)) = (a, b)⊗(c, d*k1*k2) = (ac, bd*k1*k2)

            self.op = F.conv2d
            self.extra_args = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups
            }

        else:  # Linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features

            in_m, in_n = factorization(in_dim, factor)
            out_l, out_k = factorization(out_dim, factor)
            # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d
            shape = ((out_l, out_k), (in_m, in_n))

            # smaller part. weight scale
            if decompose_both and lora_dim < max(shape[0][0], shape[1][0])/2:
                self.lokr_w1_a = nn.Parameter(
                    torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(torch.empty(
                    shape[0][0], shape[1][0]))  # a*c, 1-mode

            if lora_dim < max(shape[0][1], shape[1][1])/2:
                # bigger part. weight and LoRA. [b, dim] x [dim, d]
                self.lokr_w2_a = nn.Parameter(
                    torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][1]))
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
            else:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(
                    torch.empty(shape[0][1], shape[1][1]))

            self.op = F.linear
            self.extra_args = {}

        self.dropout = dropout
        if dropout:
            print("[WARN]LoKr haven't implemented normal dropout yet.")
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        if self.use_w2 and self.use_w1:
            # use scale = 1
            alpha = lora_dim
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha))  # treat as constant

        if self.use_w2:
            torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            if self.cp:
                torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.constant_(self.lokr_w2_b, 0)

        if self.use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

        self.multiplier = multiplier
        self.org_module = [org_module]
        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a@self.lokr_w1_b,
            (self.lokr_w2 if self.use_w2
             else make_weight_cp(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b) if self.cp
             else self.lokr_w2_a@self.lokr_w2_b),
            torch.tensor(self.multiplier * self.scale)
        )
        assert torch.sum(torch.isnan(weight)) == 0, "weight is nan"

    # Same as locon.py
    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def get_weight(self, orig_weight=None):
        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a@self.lokr_w1_b,
            (self.lokr_w2 if self.use_w2
             else make_weight_cp(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b) if self.cp
             else self.lokr_w2_a@self.lokr_w2_b),
            torch.tensor(self.scale)
        )
        if orig_weight is not None:
            weight = weight.reshape(orig_weight.shape)
        if self.training and self.rank_dropout:
            drop = torch.rand(weight.size(0)) < self.rank_dropout
            weight *= drop.view(-1, [1] *
                                len(weight.shape[1:])).to(weight.device)
        return weight

    @torch.no_grad()
    def merge_in(self, merge_weight=1.0):
        if not self.can_merge_in:
            return

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

        scale = self.scale
        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            scale = scale * self.scalar

        lokr_weight = self.get_weight(weight)

        merged_weight = (
            weight
            + (lokr_weight * merge_weight).to(weight.device, dtype=weight.dtype)
        )

        # set weight to org_module
        org_sd[weight_key] = merged_weight.to(orig_dtype)
        self.org_module[0].load_state_dict(org_sd)

    def get_orig_weight(self):
        weight = self.org_module[0].weight
        if isinstance(weight, QTensor) or isinstance(weight, QBytesTensor):
            return weight.dequantize().data.detach()
        else:
            return weight.data.detach()

    def get_orig_bias(self):
        if hasattr(self.org_module[0], 'bias') and self.org_module[0].bias is not None:
            if isinstance(self.org_module[0].bias, QTensor) or isinstance(self.org_module[0].bias, QBytesTensor):
                return self.org_module[0].bias.dequantize().data.detach()
            else:
                return self.org_module[0].bias.data.detach()
        return None

    def _call_forward(self, x):
        if isinstance(x, QTensor) or isinstance(x, QBytesTensor):
            x = x.dequantize()

        orig_dtype = x.dtype

        orig_weight = self.get_orig_weight()
        lokr_weight = self.get_weight(orig_weight).to(dtype=orig_weight.dtype)
        multiplier = self.network_ref().torch_multiplier

        if x.dtype != orig_weight.dtype:
            x = x.to(dtype=orig_weight.dtype)

        # we do not currently support split batch multipliers for lokr. Just do a mean
        multiplier = torch.mean(multiplier)

        weight = (
            orig_weight
            + lokr_weight * multiplier
        )
        bias = self.get_orig_bias()
        if bias is not None:
            bias = bias.to(weight.device, dtype=weight.dtype)
        output = self.op(
            x,
            weight.view(self.shape),
            bias,
            **self.extra_args
        )
        return output.to(orig_dtype)
