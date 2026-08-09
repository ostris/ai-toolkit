# PRXPixel (Photoroom PRX-7B, pixel-space text-to-image)

ai-toolkit integration for [`Photoroom/prxpixel-t2i`](https://huggingface.co/Photoroom/prxpixel-t2i),
a ~7B pixel-space diffusion transformer.

It is implemented **from scratch** so ai-toolkit does not depend on the (still
unmerged) diffusers PR [huggingface/diffusers#13928](https://github.com/huggingface/diffusers/pull/13928):
the transformer is vendored in [src/transformer_prx.py](src/transformer_prx.py)
and a minimal preview sampler lives in [src/pipeline.py](src/pipeline.py).

## What makes this model unusual

PRXPixel differs from a typical latent flow-matching model in three ways, each
handled in [prx_pixel_t2i.py](prx_pixel_t2i.py):

| Property | What it means | How it's handled |
|---|---|---|
| **Pixel space** | No VAE — the transformer denoises raw RGB (`in_channels=3`, `patch_size=16`) | A `FakeVAE` (identity, scaling 1) so encode/decode are no-ops; "latents" are the image in `[-1, 1]` |
| **x-prediction** | The model predicts the clean image `x0`, not the flow velocity | `get_noise_prediction` returns `x0`; `get_loss_target` is the clean latents. The `x0 → velocity` conversion only happens at sampling time |
| **noise_scale = 2.0** | Trains/samples from `randn * 2.0`, not unit noise | `get_latent_noise_from_latents` scales the training noise; the pipeline scales the starting noise |

Text is encoded by the Qwen3-VL text tower (`Qwen3VLTextModel`, hidden size
2048 → the transformer's `context_in_dim`), padded to 256 tokens.

The x-prediction objective follows *"Back to Basics: Let Denoising Generative
Models Denoise"* (https://arxiv.org/abs/2511.13720).

## Architecture (released checkpoint)

`depth=24`, `hidden_size=3584`, `num_heads=28`, `mlp_ratio=3.5`,
`in_channels=3`, `patch_size=16`, `context_in_dim=2048`, `bottleneck_size=768`,
`axes_dim=[64, 64]`, `resolution_embeds=True`, flow-matching scheduler with
`shift=3.0`.

## Train it

```yaml
model:
  arch: "prx_pixel"
  name_or_path: "/path/to/prxpixel-t2i"   # diffusers folder: transformer/,
                                          # text_encoder/, tokenizer/, scheduler/
  quantize: true        # optional: qfloat8 the transformer
  quantize_te: true     # optional: qfloat8 the Qwen3-VL text encoder
train:
  gradient_checkpointing: true
sample:
  guidance_scale: 5.0
  sample_steps: 28
```

Datasets bucket to multiples of 16px (`vae_scale_factor * patch_size`).
See [../example_model/README.md](../example_model/README.md) for the generic
lifecycle, registration and LoRA conventions.
