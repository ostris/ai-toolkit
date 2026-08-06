# Thin extension of the official diffusers AutoencoderKLWan.
# All model / tiling / patchify logic comes from diffusers so it stays in sync
# with upstream. The only thing added here is gradient checkpointing support:
# the encoder/decoder forwards are monkeypatched with copies of the upstream
# forwards that add checkpointing branches, and the subclass re-enables
# _supports_gradient_checkpointing (upstream has it turned off).

import torch

from diffusers.models.autoencoders.autoencoder_kl_wan import (
    CACHE_T,
    AutoencoderKLWan as AutoencoderKLWanBase,
    WanDecoder3d,
    WanEncoder3d,
)


# copied from diffusers WanEncoder3d.forward with gradient checkpointing added
def _wan_encoder_forward(self, x, feat_cache=None, feat_idx=[0]):
    use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None

    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_in(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_in(x)

    ## downsamples
    for layer in self.down_blocks:
        if use_ckpt:
            x = self._gradient_checkpointing_func(layer, x)
        elif feat_cache is not None:
            x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            x = layer(x)

    ## middle
    if use_ckpt:
        x = self._gradient_checkpointing_func(self.mid_block, x)
    else:
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_out(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_out(x)

    return x


# copied from diffusers WanDecoder3d.forward with gradient checkpointing added
def _wan_decoder_forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
    use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None

    ## conv1
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_in(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_in(x)

    ## middle
    if use_ckpt:
        x = self._gradient_checkpointing_func(self.mid_block, x)
    else:
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

    ## upsamples
    for up_block in self.up_blocks:
        if use_ckpt:
            x = self._gradient_checkpointing_func(up_block, x, None, [0], first_chunk)
        else:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_out(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_out(x)
    return x


WanEncoder3d.forward = _wan_encoder_forward
WanDecoder3d.forward = _wan_decoder_forward


class AutoencoderKLWan(AutoencoderKLWanBase):
    _supports_gradient_checkpointing = True
