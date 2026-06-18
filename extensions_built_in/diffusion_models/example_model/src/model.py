"""A minimal diffusion transformer (DiT) used by the example model extension.

This file stands in for the situation where diffusers does NOT have your model.
You vendor the architecture yourself inside your extension's ``src/`` folder and
load the weights manually in your model class (see ``../example_model.py``).

The architecture here is intentionally tiny and boring:

    latents (B, C, h, w)
        -> patchify with a strided conv            (B, N_img, hidden)
    text embeds (B, L, text_dim)
        -> linear projection                       (B, L, hidden)
    concat [text | image] into one joint sequence  (B, L + N_img, hidden)
        -> N transformer blocks (self attention + mlp, adaLN-zero
           modulated by the timestep embedding)
        -> final modulated norm + linear
    take only the image tokens and unpatchify back to (B, C, h, w)

Real models add RoPE position embeddings, fancier attention, guidance
embeddings, etc. For real-world reference implementations in this repo see:
  - ../../chroma/src/model.py      (flux-style double/single stream blocks)
  - ../../ernie_image/transformer.py (diffusers ModelMixin based)
  - ../../ideogram4/src/transformer.py (packed single-sequence model)

GRADIENT CHECKPOINTING
======================
ai-toolkit enables gradient checkpointing on your model from
``jobs/process/BaseSDTrainProcess.py`` which does, in order of preference:

    if hasattr(unet, 'enable_gradient_checkpointing'):
        unet.enable_gradient_checkpointing()
    elif hasattr(unet, 'gradient_checkpointing'):
        unet.gradient_checkpointing = True

So a custom model only needs:
  1. a ``self.gradient_checkpointing`` flag (default False)
  2. (optionally) an ``enable_gradient_checkpointing()`` method
  3. to wrap each transformer block call in ``torch.utils.checkpoint.checkpoint``
     when the flag is set AND grads are enabled.

IMPORTANT: gate on ``torch.is_grad_enabled()``, NOT on ``self.training``.
Sampling runs under ``torch.no_grad()`` where checkpointing is pure overhead,
and some training setups (e.g. certain adapters) run the module in eval mode
while still needing gradients. ``torch.is_grad_enabled()`` handles both.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard sinusoidal embedding.

    in:  t   (B,) float tensor, the flow-matching time in [0, 1] (1 = pure noise)
    out: emb (B, dim)

    We scale t by 1000 before embedding so the sinusoids get a useful range,
    the same trick flux and friends use.
    """
    t = t.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ExampleTransformerBlock(nn.Module):
    """One DiT block: adaLN-zero modulated self-attention + MLP.

    in:  x         (B, S, hidden)   the joint [text | image] token sequence
         temb      (B, hidden)      the timestep embedding
         attn_mask (B, 1, 1, S)     bool, True = attend, False = padding
    out: x         (B, S, hidden)
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # adaLN-zero: timestep embedding -> shift/scale/gate for attn and mlp.
        # Zero-init so the block starts as identity (standard DiT trick).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.adaLN_modulation(temb).unsqueeze(1).chunk(6, dim=-1)
        )  # each (B, 1, hidden), broadcasts over the sequence

        # --- attention ---
        # ALWAYS default to torch's built-in scaled_dot_product_attention so the
        # model runs with no extra dependency. If the reference repo you are
        # porting hard-codes flash-attn (or xformers, sage, ...), do NOT carry
        # that requirement over -- make it OPTIONAL. The clean pattern is a
        # per-module ``attention_backend`` flag toggled in bulk from the parent
        # model (e.g. ``set_attention_backend("flash")``), branching to the
        # flash kernel only when explicitly selected AND the package is present.
        # See ../../ideogram4/src/transformer.py and ../../boogu_image/src for
        # working "native" (SDPA) + optional "flash" implementations.
        h = self.norm1(x) * (1 + scale_a) + shift_a
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        h = h.transpose(1, 2).reshape(b, s, d)
        x = x + gate_a * self.proj(h)

        # --- mlp ---
        h = self.norm2(x) * (1 + scale_m) + shift_m
        x = x + gate_m * self.mlp(h)
        return x


class ExampleTransformer2DModel(nn.Module):
    """The denoiser. Plain ``nn.Module`` on purpose.

    You could also subclass ``diffusers.ModelMixin``/``ConfigMixin`` (see
    ../../ernie_image/transformer.py) to get ``save_pretrained``,
    ``_gradient_checkpointing_func`` etc. for free, but a plain module shows
    exactly what ai-toolkit actually requires, which is very little:

      - a forward pass
      - ``device`` / ``dtype`` properties (BaseModel reads ``self.model.device``
        and ``self.model.dtype`` in a few places, e.g. save_device_state)
      - the gradient checkpointing flag described in the module docstring

    NOTE: the class NAME matters. ``ExampleModel.target_lora_modules`` lists
    "ExampleTransformer2DModel" -- that string is matched against module class
    names when deciding where to attach LoRA layers.
    """

    def __init__(
        self,
        in_channels: int = 16,    # VAE latent channels
        out_channels: int = 16,   # predicted velocity has the same channels
        patch_size: int = 2,      # latent pixels per token side
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 12,
        text_dim: int = 2048,     # width of the text encoder hidden states
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # latent (B, C, h, w) -> image tokens (B, N_img, hidden)
        self.x_embedder = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        # text encoder hidden states -> model width
        self.text_proj = nn.Linear(text_dim, hidden_size)
        # sinusoidal timestep embedding -> mlp
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # ``blocks`` is the repeated-layer ModuleList. The attribute name is
        # what ExampleModel.get_transformer_block_names() returns, which the
        # LoRA code uses for block targeting / "transformer only" training.
        self.blocks = nn.ModuleList(
            [
                ExampleTransformerBlock(hidden_size, num_heads)
                for _ in range(num_layers)
            ]
        )

        # final adaLN + projection back to patch pixels, zero-init so the
        # untrained model predicts zeros.
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.adaLN_out[-1].weight)
        nn.init.zeros_(self.adaLN_out[-1].bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # gradient checkpointing flag, flipped on by the trainer (see module
        # docstring). Off by default so inference pays no cost.
        self.gradient_checkpointing = False

    # the trainer prefers this method if it exists
    def enable_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, C, h, w) noisy latents
        timestep: torch.Tensor,               # (B,) flow time in [0, 1], 1 = pure noise
        encoder_hidden_states: torch.Tensor,  # (B, L, text_dim) padded text features
        attention_mask: torch.Tensor,         # (B, L) 1 = real text token, 0 = padding
    ) -> torch.Tensor:
        """Predict the flow-matching velocity.

        out: (B, C, h, w) velocity in the ai-toolkit convention
             (noise - clean), matching ExampleModel.get_loss_target().
        """
        b, c, h, w = hidden_states.shape
        p = self.patch_size
        gh, gw = h // p, w // p
        n_img = gh * gw

        # tokens
        img = self.x_embedder(hidden_states)              # (B, hidden, gh, gw)
        img = img.flatten(2).transpose(1, 2)              # (B, N_img, hidden)
        txt = self.text_proj(encoder_hidden_states)       # (B, L, hidden)
        x = torch.cat([txt, img], dim=1)                  # (B, L + N_img, hidden)

        # timestep conditioning
        temb = self.t_embedder(timestep_embedding(timestep, self.hidden_size))
        temb = temb.to(x.dtype)

        # joint attention mask: text padding is masked out, image tokens and
        # real text tokens attend everywhere. (B, 1, 1, S) bool for sdpa.
        img_mask = torch.ones(b, n_img, dtype=torch.bool, device=x.device)
        attn_mask = torch.cat([attention_mask.bool(), img_mask], dim=1)
        attn_mask = attn_mask[:, None, None, :]

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # Recompute this block's activations during backward instead
                # of storing them -- trades compute for a big VRAM saving.
                # use_reentrant=False is the modern, correct variant.
                x = checkpoint(block, x, temb, attn_mask, use_reentrant=False)
            else:
                x = block(x, temb, attn_mask)

        # final modulation + project, keep only the image tokens
        shift, scale = self.adaLN_out(temb).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale) + shift
        x = self.proj_out(x)[:, -n_img:]                  # (B, N_img, p*p*C)

        # unpatchify back to the latent layout
        x = x.view(b, gh, gw, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.out_channels, h, w)
        return x
