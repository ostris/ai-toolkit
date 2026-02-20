import torch
import torch.nn as nn

from toolkit.models.zipper_resampler import ContextualAlphaMask


# Conv1d MLP
# MLP that can alternately be used as a conv1d on dim 1
class MLPC(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            hidden_dim,
            do_conv=False,
            use_residual=True
    ):
        super().__init__()
        self.do_conv = do_conv
        if use_residual:
            assert in_dim == out_dim
        # dont normalize if using conv
        if not do_conv:
            self.layernorm = nn.LayerNorm(in_dim)

        if do_conv:
            self.fc1 = nn.Conv1d(in_dim, hidden_dim, 1)
            self.fc2 = nn.Conv1d(hidden_dim, out_dim, 1)
        else:
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        if not self.do_conv:
            x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class ZipperBlock(nn.Module):
    def __init__(
            self,
            in_size,
            in_tokens,
            out_size,
            out_tokens,
            hidden_size,
            hidden_tokens,
    ):
        super().__init__()
        self.in_size = in_size
        self.in_tokens = in_tokens
        self.out_size = out_size
        self.out_tokens = out_tokens
        self.hidden_size = hidden_size
        self.hidden_tokens = hidden_tokens
        # permute to (batch_size, out_size, in_tokens)

        self.zip_token = MLPC(
            in_dim=self.in_tokens,
            out_dim=self.out_tokens,
            hidden_dim=self.hidden_tokens,
            do_conv=True,  # no need to permute
            use_residual=False
        )

        # permute to (batch_size, out_tokens, out_size)

        # in shpae: (batch_size, in_tokens, in_size)
        self.zip_size = MLPC(
            in_dim=self.in_size,
            out_dim=self.out_size,
            hidden_dim=self.hidden_size,
            use_residual=False
        )

    def forward(self, x):
        x = self.zip_token(x)
        x = self.zip_size(x)
        return x






# CLIPFusionModule
# Fuses any size of vision and text embeddings into a single embedding.
# remaps tokens and vectors.
class CLIPFusionModule(nn.Module):
    def __init__(
            self,
            text_hidden_size: int = 768,
            text_tokens: int = 77,
            vision_hidden_size: int = 1024,
            vision_tokens: int = 257,
            num_blocks: int = 1,
    ):
        super(CLIPFusionModule, self).__init__()

        self.text_hidden_size = text_hidden_size
        self.text_tokens = text_tokens
        self.vision_hidden_size = vision_hidden_size
        self.vision_tokens = vision_tokens

        self.resampler = ZipperBlock(
            in_size=self.vision_hidden_size,
            in_tokens=self.vision_tokens,
            out_size=self.text_hidden_size,
            out_tokens=self.text_tokens,
            hidden_size=self.vision_hidden_size * 2,
            hidden_tokens=self.vision_tokens * 2
        )

        self.zipper_blocks = torch.nn.ModuleList([
            ZipperBlock(
                in_size=self.text_hidden_size * 2,
                in_tokens=self.text_tokens,
                out_size=self.text_hidden_size,
                out_tokens=self.text_tokens,
                hidden_size=self.text_hidden_size * 2,
                hidden_tokens=self.text_tokens * 2
            ) for i in range(num_blocks)
        ])

        self.ctx_alpha = ContextualAlphaMask(
            dim=self.text_hidden_size,
        )

        self.alpha = nn.Parameter(torch.zeros([text_tokens]) + 0.01)

    def forward(self, text_embeds, vision_embeds):
        # text_embeds = (batch_size, 77, 768)
        # vision_embeds = (batch_size, 257, 1024)
        # output = (batch_size, 77, 768)

        vision_embeds = self.resampler(vision_embeds)
        x = vision_embeds
        for i, block in enumerate(self.zipper_blocks):
            res = x
            x = torch.cat([text_embeds, x], dim=-1)
            x = block(x)
            x = x + res

        # alpha mask
        ctx_alpha = self.ctx_alpha(text_embeds)
        # reshape alpha to (1, 77, 1)
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)

        x = ctx_alpha * x * alpha

        x = x + text_embeds

        return x
