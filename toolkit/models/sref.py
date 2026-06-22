import torch
import torch.nn as nn


class SrefImageEncoder(torch.nn.Module):
    def __init__(
        self,
        input_features: int = 1152,
        input_tokens: int = 512,
        output_tokens: int = 512,
        output_features: int = 4096,
        intermediate_size: int = 4096,
        num_digits: int = 10,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.device = device
        self.dtype = dtype
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.output_features = output_features
        self.intermediate_size = intermediate_size
        self.num_digits = num_digits

        self.proj_in = nn.Linear(
            input_features, intermediate_size, dtype=dtype)
        # (bs, num_digits, intermediate_size)
        self.conv_pool = nn.Conv1d(input_tokens, num_digits, 1, dtype=dtype)
        self.linear_pool = nn.Linear(
            intermediate_size, 1, dtype=dtype)  # (bs, num_digits, 1)
        # do sigmoid for digits 0.0-1.0 = (0 to 10) Always floor when rounding digits so you get 0-9
        self.flatten = nn.Flatten()  # (bs, num_digits * intermediate_size)

        # a numeric sref would come in here with num_digits
        self.sref_in = nn.Linear(num_digits, intermediate_size, dtype=dtype)
        self.fc1 = nn.Linear(intermediate_size, intermediate_size, dtype=dtype)
        self.fc2 = nn.Linear(intermediate_size, intermediate_size, dtype=dtype)

        self.proj_out = nn.Linear(
            intermediate_size, output_features * output_tokens, dtype=dtype)

    def forward(self, siglip_embeds) -> torch.Tensor:
        x = self.proj_in(siglip_embeds)
        x = torch.nn.functional.silu(x)
        x = self.conv_pool(x)
        x = self.linear_pool(x)
        x = torch.sigmoid(x)

        sref = self.flatten(x)

        x = self.sref_in(sref)
        x = torch.nn.functional.silu(x)
        x = self.fc1(x)
        x = torch.nn.functional.silu(x)
        x = self.fc2(x)
        x = torch.nn.functional.silu(x)
        x = self.proj_out(x)

        return x
