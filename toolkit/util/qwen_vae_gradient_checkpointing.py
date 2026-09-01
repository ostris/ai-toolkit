import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
    QwenImageDecoder3d,
    QwenImageEncoder3d,
)

# diffusers removed gradient checkpointing from AutoencoderKLQwenImage
# (_supports_gradient_checkpointing = False and no checkpoint calls left in the
# forwards). This patches it back in so vae.enable_gradient_checkpointing()
# works again.

_orig_encoder_forward = QwenImageEncoder3d.forward
_orig_decoder_forward = QwenImageDecoder3d.forward

_patched = False


def _checkpoint(module, x):
    return torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)


def _cache_is_fresh(feat_cache):
    return feat_cache is None or all(c is None for c in feat_cache)


# The checkpointed paths run cache-free. feat_cache is mutated in place during
# the forward, so recomputing a block during backward would see end-state cache
# contents and produce garbage. With a fresh (all None) cache the cached and
# cache-free paths are numerically identical for the chunk being processed, so
# this is exact for single-frame (image) inputs. Later video chunks arrive with
# a populated cache and fall through to the original un-checkpointed forward.


def _encoder_forward(self, x, feat_cache=None, feat_idx=[0]):
    if not (self.gradient_checkpointing and torch.is_grad_enabled() and _cache_is_fresh(feat_cache)):
        return _orig_encoder_forward(self, x, feat_cache, feat_idx)

    x = self.conv_in(x)
    for layer in self.down_blocks:
        x = _checkpoint(layer, x)
    x = _checkpoint(self.mid_block, x)
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    x = self.conv_out(x)
    return x


def _decoder_forward(self, x, feat_cache=None, feat_idx=[0]):
    if not (self.gradient_checkpointing and torch.is_grad_enabled() and _cache_is_fresh(feat_cache)):
        return _orig_decoder_forward(self, x, feat_cache, feat_idx)

    x = self.conv_in(x)
    x = _checkpoint(self.mid_block, x)
    for up_block in self.up_blocks:
        x = _checkpoint(up_block, x)
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    x = self.conv_out(x)
    return x


def patch_qwen_vae_gradient_checkpointing():
    global _patched
    if _patched:
        return
    _patched = True

    AutoencoderKLQwenImage._supports_gradient_checkpointing = True
    QwenImageEncoder3d.forward = _encoder_forward
    QwenImageDecoder3d.forward = _decoder_forward
