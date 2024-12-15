import torch


class Decorator(torch.nn.Module):
    def __init__(
        self,
        num_tokens: int = 4,
        token_size: int = 4096,
    ) -> None:
        super().__init__()

        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(num_tokens, token_size)
        )
        # ensure it is float32
        self.weight.data = self.weight.data.float()

    def forward(self, text_embeds: torch.Tensor, is_unconditional=False) -> torch.Tensor:
        # make sure the param is float32
        if self.weight.dtype != text_embeds.dtype:
            self.weight.data = self.weight.data.float()
        # expand batch to match text_embeds
        batch_size = text_embeds.shape[0]
        decorator_embeds = self.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if is_unconditional:
            # zero pad the decorator embeds
            decorator_embeds = torch.zeros_like(decorator_embeds)

        if decorator_embeds.dtype != text_embeds.dtype:
            decorator_embeds = decorator_embeds.to(text_embeds.dtype)
        text_embeds = torch.cat((text_embeds, decorator_embeds), dim=-2)

        return text_embeds
