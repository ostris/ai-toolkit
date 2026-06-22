import torch
import torch.nn as nn


class ContextualAlphaMask(nn.Module):
    def __init__(
            self,
            dim: int = 768,
    ):
        super(ContextualAlphaMask, self).__init__()
        self.dim = dim

        half_dim = dim // 2
        quarter_dim = dim // 4

        self.fc1 = nn.Linear(self.dim, self.dim)
        self.fc2 = nn.Linear(self.dim, half_dim)
        self.norm1 = nn.LayerNorm(half_dim)
        self.fc3 = nn.Linear(half_dim, half_dim)
        self.fc4 = nn.Linear(half_dim, quarter_dim)
        self.norm2 = nn.LayerNorm(quarter_dim)
        self.fc5 = nn.Linear(quarter_dim, quarter_dim)
        self.fc6 = nn.Linear(quarter_dim, 1)
        # set fc6  weights to near zero
        self.fc6.weight.data.normal_(mean=0.0, std=0.0001)
        self.act_fn = nn.GELU()

    def forward(self, x):
        # x = (batch_size, 77, 768)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.fc3(x)
        x = self.act_fn(x)
        x = self.fc4(x)
        x = self.norm2(x)
        x = self.act_fn(x)
        x = self.fc5(x)
        x = self.act_fn(x)
        x = self.fc6(x)
        x = torch.sigmoid(x)
        return x


class ZipperModule(nn.Module):
    def __init__(
            self,
            in_size,
            in_tokens,
            out_size,
            out_tokens,
            hidden_size,
            hidden_tokens,
            use_residual=False,
    ):
        super().__init__()
        self.in_size = in_size
        self.in_tokens = in_tokens
        self.out_size = out_size
        self.out_tokens = out_tokens
        self.hidden_size = hidden_size
        self.hidden_tokens = hidden_tokens
        self.use_residual = use_residual

        self.act_fn = nn.GELU()
        self.layernorm = nn.LayerNorm(self.in_size)

        self.conv1 = nn.Conv1d(self.in_tokens, self.hidden_tokens, 1)
        # act
        self.fc1 = nn.Linear(self.in_size, self.hidden_size)
        # act
        self.conv2 = nn.Conv1d(self.hidden_tokens, self.out_tokens, 1)
        # act
        self.fc2 = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.conv1(x)
        x = self.act_fn(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.conv2(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class ZipperResampler(nn.Module):
    def __init__(
            self,
            in_size,
            in_tokens,
            out_size,
            out_tokens,
            hidden_size,
            hidden_tokens,
            num_blocks=1,
            is_conv_input=False,
    ):
        super().__init__()
        self.is_conv_input = is_conv_input

        module_list = []
        for i in range(num_blocks):

            this_in_size = in_size
            this_in_tokens = in_tokens
            this_out_size = out_size
            this_out_tokens = out_tokens
            this_hidden_size = hidden_size
            this_hidden_tokens = hidden_tokens
            use_residual = False

            # maintain middle sizes as hidden_size
            if i == 0:  # first block
                this_in_size = in_size
                this_in_tokens = in_tokens
                if num_blocks == 1:
                    this_out_size = out_size
                    this_out_tokens = out_tokens
                else:
                    this_out_size = hidden_size
                    this_out_tokens = hidden_tokens
            elif i == num_blocks - 1:  # last block
                this_out_size = out_size
                this_out_tokens = out_tokens
                if num_blocks == 1:
                    this_in_size = in_size
                    this_in_tokens = in_tokens
                else:
                    this_in_size = hidden_size
                    this_in_tokens = hidden_tokens
            else:  # middle blocks
                this_out_size = hidden_size
                this_out_tokens = hidden_tokens
                this_in_size = hidden_size
                this_in_tokens = hidden_tokens
                use_residual = True

            module_list.append(ZipperModule(
                in_size=this_in_size,
                in_tokens=this_in_tokens,
                out_size=this_out_size,
                out_tokens=this_out_tokens,
                hidden_size=this_hidden_size,
                hidden_tokens=this_hidden_tokens,
                use_residual=use_residual
            ))

        self.blocks = nn.ModuleList(module_list)

        self.ctx_alpha = ContextualAlphaMask(
            dim=out_size,
        )

    def forward(self, x):
        if self.is_conv_input:
            # flatten
            x = x.view(x.size(0), x.size(1), -1)
            # rearrange to (batch, tokens, size)
            x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)
        alpha = self.ctx_alpha(x)
        return x * alpha
