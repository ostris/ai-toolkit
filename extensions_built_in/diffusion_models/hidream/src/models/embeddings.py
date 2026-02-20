import torch
from torch import nn
from typing import List
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()

# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
class EmbedND(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(2)
    
class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        out_channels=1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels * patch_size * patch_size, out_channels, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, latent):
        latent = self.proj(latent)
        return latent
    
class PooledEmbed(nn.Module):
    def __init__(self, text_emb_dim, hidden_size): 
        super().__init__()
        self.pooled_embedder = TimestepEmbedding(in_channels=text_emb_dim, time_embed_dim=hidden_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pooled_embed):
        return self.pooled_embedder(pooled_embed)
    
class TimestepEmbed(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.time_proj = Timesteps(num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=hidden_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, timesteps, wdtype):
        t_emb = self.time_proj(timesteps).to(dtype=wdtype)
        t_emb = self.timestep_embedder(t_emb)
        return t_emb
    
class OutEmbed(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, adaln_input):
        shift, scale = self.adaLN_modulation(adaln_input).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x