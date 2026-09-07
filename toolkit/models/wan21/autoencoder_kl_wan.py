# Thin extension of the official diffusers AutoencoderKLWan.
# All model / tiling / patchify logic comes from diffusers so it stays in sync
# with upstream. The only thing added here is gradient checkpointing support:
# the encoder/decoder forwards are monkeypatched with copies of the upstream
# forwards that add checkpointing branches, and the subclass re-enables
# _supports_gradient_checkpointing (upstream has it turned off).
# The checkpointing branches only run cache-free (the chunked-cache path
# mutates shared lists, which is unsafe under checkpoint recompute), but
# upstream _encode always goes through the chunked cache — so when
# checkpointing is active, _encode runs the encoder cache-free over the whole
# clip in one pass instead. Causal convs make that pass identical to the
# chunked one; the only divergence is downsample3d WanResample, whose temporal
# conv upstream applies only via the cache path, so it gets a cache-free
# forward that reproduces the chunked semantics exactly:
# out = cat([x[:, :, :1], time_conv(x)], dim=2)
# (chunk 0 is returned as-is; later chunks run time_conv(cat([prev_last,
# chunk])), which tiles into exactly these stride-2 windows).

import torch

from diffusers.models.autoencoders.autoencoder_kl_wan import (
    CACHE_T,
    AutoencoderKLWan as AutoencoderKLWanBase,
    WanDecoder3d,
    WanEncoder3d,
    WanResample,
    patchify,
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


_wan_resample_stock_forward = WanResample.forward


# cache-free downsample3d matching the chunked-cache semantics (see module
# docstring); upstream's cache-free path skips the temporal conv entirely
def _wan_resample_forward(self, x, feat_cache=None, feat_idx=[0]):
    if self.mode != "downsample3d" or feat_cache is not None:
        return _wan_resample_stock_forward(self, x, feat_cache=feat_cache, feat_idx=feat_idx)
    b, c, t, h, w = x.size()
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = self.resample(x)
    x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
    x = torch.cat([x[:, :, :1], self.time_conv(x)], dim=2)
    return x


WanEncoder3d.forward = _wan_encoder_forward
WanDecoder3d.forward = _wan_decoder_forward
WanResample.forward = _wan_resample_forward


class AutoencoderKLWan(AutoencoderKLWanBase):
    _supports_gradient_checkpointing = True

    def _encode(self, x: torch.Tensor):
        # the chunked-cache encode never reaches the checkpointing branches in
        # _wan_encoder_forward (they need feat_cache=None), so run cache-free
        # in one pass when checkpointing is active
        if not (torch.is_grad_enabled() and self.encoder.gradient_checkpointing):
            return super()._encode(x)

        _, _, num_frame, height, width = x.shape
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)
        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)
        out = self.encoder(x)
        return self.quant_conv(out)
