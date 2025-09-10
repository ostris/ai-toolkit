import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F

from .math import attention, rope
from functools import lru_cache


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.use_compiled = use_compiled

    def _forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

    def forward(self, x: Tensor):
        return F.rms_norm(x, self.scale.shape, weight=self.scale, eps=1e-6)
        # if self.use_compiled:
        #     return torch.compile(self._forward)(x)
        # else:
        #     return self._forward(x)


def distribute_modulations(tensor: torch.Tensor, depth_single_blocks, depth_double_blocks):
    """
    Distributes slices of the tensor into the block_dict as ModulationOut objects.

    Args:
        tensor (torch.Tensor): Input tensor with shape [batch_size, vectors, dim].
    """
    batch_size, vectors, dim = tensor.shape

    block_dict = {}

    # HARD CODED VALUES! lookup table for the generated vectors
    # TODO: move this into chroma config!
    # Add 38 single mod blocks
    for i in range(depth_single_blocks):
        key = f"single_blocks.{i}.modulation.lin"
        block_dict[key] = None

    # Add 19 image double blocks
    for i in range(depth_double_blocks):
        key = f"double_blocks.{i}.img_mod.lin"
        block_dict[key] = None

    # Add 19 text double blocks
    for i in range(depth_double_blocks):
        key = f"double_blocks.{i}.txt_mod.lin"
        block_dict[key] = None

    # Add the final layer
    block_dict["final_layer.adaLN_modulation.1"] = None
    # 6.2b version
    # block_dict["lite_double_blocks.4.img_mod.lin"] = None
    # block_dict["lite_double_blocks.4.txt_mod.lin"] = None

    idx = 0  # Index to keep track of the vector slices

    for key in block_dict.keys():
        if "single_blocks" in key:
            # Single block: 1 ModulationOut
            block_dict[key] = ModulationOut(
                shift=tensor[:, idx : idx + 1, :],
                scale=tensor[:, idx + 1 : idx + 2, :],
                gate=tensor[:, idx + 2 : idx + 3, :],
            )
            idx += 3  # Advance by 3 vectors

        elif "img_mod" in key:
            # Double block: List of 2 ModulationOut
            double_block = []
            for _ in range(2):  # Create 2 ModulationOut objects
                double_block.append(
                    ModulationOut(
                        shift=tensor[:, idx : idx + 1, :],
                        scale=tensor[:, idx + 1 : idx + 2, :],
                        gate=tensor[:, idx + 2 : idx + 3, :],
                    )
                )
                idx += 3  # Advance by 3 vectors per ModulationOut
            block_dict[key] = double_block

        elif "txt_mod" in key:
            # Double block: List of 2 ModulationOut
            double_block = []
            for _ in range(2):  # Create 2 ModulationOut objects
                double_block.append(
                    ModulationOut(
                        shift=tensor[:, idx : idx + 1, :],
                        scale=tensor[:, idx + 1 : idx + 2, :],
                        gate=tensor[:, idx + 2 : idx + 3, :],
                    )
                )
                idx += 3  # Advance by 3 vectors per ModulationOut
            block_dict[key] = double_block

        elif "final_layer" in key:
            # Final layer: 1 ModulationOut
            block_dict[key] = [
                tensor[:, idx : idx + 1, :],
                tensor[:, idx + 1 : idx + 2, :],
            ]
            idx += 2  # Advance by 3 vectors

    return block_dict



class NerfEmbedder(nn.Module):
    """
    An embedder module that combines input features with a 2D positional
    encoding that mimics the Discrete Cosine Transform (DCT).

    This module takes an input tensor of shape (B, P^2, C), where P is the
    patch size, and enriches it with positional information before projecting
    it to a new hidden size.
    """
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        """
        Initializes the NerfEmbedder.

        Args:
            in_channels (int): The number of channels in the input tensor.
            hidden_size_input (int): The desired dimension of the output embedding.
            max_freqs (int): The number of frequency components to use for both
                             the x and y dimensions of the positional encoding.
                             The total number of positional features will be max_freqs^2.
        """
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        
        # A linear layer to project the concatenated input features and
        # positional encodings to the final output dimension.
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input)
        )

    @lru_cache(maxsize=4)
    def fetch_pos(self, patch_size, device, dtype):
        """
        Generates and caches 2D DCT-like positional embeddings for a given patch size.

        The LRU cache is a performance optimization that avoids recomputing the
        same positional grid on every forward pass.

        Args:
            patch_size (int): The side length of the square input patch.
            device: The torch device to create the tensors on.
            dtype: The torch dtype for the tensors.

        Returns:
            A tensor of shape (1, patch_size^2, max_freqs^2) containing the
            positional embeddings.
        """
        # Create normalized 1D coordinate grids from 0 to 1.
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        
        # Create a 2D meshgrid of coordinates.
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        
        # Reshape positions to be broadcastable with frequencies.
        # Shape becomes (patch_size^2, 1, 1).
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        
        # Create a 1D tensor of frequency values from 0 to max_freqs-1.
        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)
        
        # Reshape frequencies to be broadcastable for creating 2D basis functions.
        # freqs_x shape: (1, max_freqs, 1)
        # freqs_y shape: (1, 1, max_freqs)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        
        # A custom weighting coefficient, not part of standard DCT.
        # This seems to down-weight the contribution of higher-frequency interactions.
        coeffs = (1 + freqs_x * freqs_y) ** -1
        
        # Calculate the 1D cosine basis functions for x and y coordinates.
        # This is the core of the DCT formulation.
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        
        # Combine the 1D basis functions to create 2D basis functions by element-wise
        # multiplication, and apply the custom coefficients. Broadcasting handles the
        # combination of all (pos_x, freqs_x) with all (pos_y, freqs_y).
        # The result is flattened into a feature vector for each position.
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)
        
        return dct

    def forward(self, inputs):
        """
        Forward pass for the embedder.

        Args:
            inputs (Tensor): The input tensor of shape (B, P^2, C).

        Returns:
            Tensor: The output tensor of shape (B, P^2, hidden_size_input).
        """
        # Get the batch size, number of pixels, and number of channels.
        B, P2, C = inputs.shape
        # Store the original dtype to cast back to at the end.
        original_dtype = inputs.dtype
        # Force all operations within this module to run in fp32.
        with torch.autocast("cuda", enabled=False):
            # Infer the patch side length from the number of pixels (P^2).
            patch_size = int(P2 ** 0.5)

            inputs = inputs.float()
            # Fetch the pre-computed or cached positional embeddings.
            dct = self.fetch_pos(patch_size, inputs.device, torch.float32)
            
            # Repeat the positional embeddings for each item in the batch.
            dct = dct.repeat(B, 1, 1)
            
            # Concatenate the original input features with the positional embeddings
            # along the feature dimension.
            inputs = torch.cat([inputs, dct], dim=-1)
            
            # Project the combined tensor to the target hidden size.
            inputs = self.embedder.float()(inputs)
        
        return inputs.to(original_dtype)



class NerfGLUBlock(nn.Module):
    """
    A NerfBlock using a Gated Linear Unit (GLU) like MLP.
    """
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio, use_compiled):
        super().__init__()
        # The total number of parameters for the MLP is increased to accommodate
        # the gate, value, and output projection matrices.
        # We now need to generate parameters for 3 matrices.
        total_params = 3 * hidden_size_x**2 * mlp_ratio
        self.param_generator = nn.Linear(hidden_size_s, total_params)
        self.norm = RMSNorm(hidden_size_x, use_compiled)
        self.mlp_ratio = mlp_ratio
        # nn.init.zeros_(self.param_generator.weight)
        # nn.init.zeros_(self.param_generator.bias)


    def forward(self, x, s):
        batch_size, num_x, hidden_size_x = x.shape
        mlp_params = self.param_generator(s)

        # Split the generated parameters into three parts for the gate, value, and output projection.
        fc1_gate_params, fc1_value_params, fc2_params = mlp_params.chunk(3, dim=-1)

        # Reshape the parameters into matrices for batch matrix multiplication.
        fc1_gate = fc1_gate_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc1_value = fc1_value_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2 = fc2_params.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)

        # Normalize the generated weight matrices as in the original implementation.
        fc1_gate = torch.nn.functional.normalize(fc1_gate, dim=-2)
        fc1_value = torch.nn.functional.normalize(fc1_value, dim=-2)
        fc2 = torch.nn.functional.normalize(fc2, dim=-2)

        res_x = x
        x = self.norm(x)

        # Apply the final output projection.
        x = torch.bmm(torch.nn.functional.silu(torch.bmm(x, fc1_gate)) * torch.bmm(x, fc1_value), fc2)
        
        x = x + res_x
        return x


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, use_compiled):
        super().__init__()
        self.norm = RMSNorm(hidden_size, use_compiled=use_compiled)
        self.linear = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


class NerfFinalLayerConv(nn.Module):
    def __init__(self, hidden_size, out_channels, use_compiled):
        super().__init__()
        self.norm = RMSNorm(hidden_size, use_compiled=use_compiled)

        # replace nn.Linear with nn.Conv2d since linear is just pointwise conv
        self.conv = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # shape: [N, C, H, W] !
        # RMSNorm normalizes over the last dimension, but our channel dim (C) is at dim=1.
        # So, we permute the dimensions to make the channel dimension the last one.
        x_permuted = x.permute(0, 2, 3, 1)  # Shape becomes [N, H, W, C]

        # Apply normalization on the feature/channel dimension
        x_norm = self.norm(x_permuted)

        # Permute back to the original dimension order for the convolution
        x_norm_permuted = x_norm.permute(0, 3, 1, 2) # Shape becomes [N, C, H, W]

        # Apply the 3x3 convolution
        x = self.conv(x_norm_permuted)
        return x
    

class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList(
            [MLPEmbedder(hidden_dim, hidden_dim) for x in range(n_layers)]
        )
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for x in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.query_norm = RMSNorm(dim, use_compiled=use_compiled)
        self.key_norm = RMSNorm(dim, use_compiled=use_compiled)
        self.use_compiled = use_compiled

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim, use_compiled=use_compiled)
        self.proj = nn.Linear(dim, dim)
        self.use_compiled = use_compiled

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


def _modulation_shift_scale_fn(x, scale, shift):
    return (1 + scale) * x + shift


def _modulation_gate_fn(x, gate, gate_params):
    return x + gate * gate_params


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        use_compiled: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_compiled=use_compiled,
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_compiled=use_compiled,
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.use_compiled = use_compiled

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def modulation_gate_fn(self, x, gate, gate_params):
        if self.use_compiled:
            return torch.compile(_modulation_gate_fn)(x, gate, gate_params)
        else:
            return _modulation_gate_fn(x, gate, gate_params)

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        distill_vec: list[ModulationOut],
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = distill_vec

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        # replaced with compiled fn
        # img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_modulated = self.modulation_shift_scale_fn(
            img_modulated, img_mod1.scale, img_mod1.shift
        )
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        # replaced with compiled fn
        # txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_modulated = self.modulation_shift_scale_fn(
            txt_modulated, txt_mod1.scale, txt_mod1.shift
        )
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe, mask=mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        # replaced with compiled fn
        # img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        img = self.modulation_gate_fn(img, img_mod1.gate, self.img_attn.proj(img_attn))
        img = self.modulation_gate_fn(
            img,
            img_mod2.gate,
            self.img_mlp(
                self.modulation_shift_scale_fn(
                    self.img_norm2(img), img_mod2.scale, img_mod2.shift
                )
            ),
        )

        # calculate the txt bloks
        # replaced with compiled fn
        # txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        txt = self.modulation_gate_fn(txt, txt_mod1.gate, self.txt_attn.proj(txt_attn))
        txt = self.modulation_gate_fn(
            txt,
            txt_mod2.gate,
            self.txt_mlp(
                self.modulation_shift_scale_fn(
                    self.txt_norm2(txt), txt_mod2.scale, txt_mod2.shift
                )
            ),
        )

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim, use_compiled=use_compiled)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.use_compiled = use_compiled

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def modulation_gate_fn(self, x, gate, gate_params):
        if self.use_compiled:
            return torch.compile(_modulation_gate_fn)(x, gate, gate_params)
        else:
            return _modulation_gate_fn(x, gate, gate_params)

    def forward(
        self, x: Tensor, pe: Tensor, distill_vec: list[ModulationOut], mask: Tensor
    ) -> Tensor:
        mod = distill_vec
        # replaced with compiled fn
        # x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        x_mod = self.modulation_shift_scale_fn(self.pre_norm(x), mod.scale, mod.shift)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        # replaced with compiled fn
        # return x + mod.gate * output
        return self.modulation_gate_fn(x, mod.gate, output)


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.use_compiled = use_compiled

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def forward(self, x: Tensor, distill_vec: list[Tensor]) -> Tensor:
        shift, scale = distill_vec
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        # replaced with compiled fn
        # x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.modulation_shift_scale_fn(
            self.norm_final(x), scale[:, None, :], shift[:, None, :]
        )
        x = self.linear(x)
        return x
