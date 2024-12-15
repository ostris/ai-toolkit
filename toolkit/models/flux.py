
# forward that bypasses the guidance embedding so it can be avoided during training.
from functools import partial


def guidance_embed_bypass_forward(self, timestep, guidance, pooled_projection):
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(
        timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)
    pooled_projections = self.text_embedder(pooled_projection)
    conditioning = timesteps_emb + pooled_projections
    return conditioning

# bypass the forward function


def bypass_flux_guidance(transformer):
    if hasattr(transformer.time_text_embed, '_bfg_orig_forward'):
        return
    # dont bypass if it doesnt have the guidance embedding
    if not hasattr(transformer.time_text_embed, 'guidance_embedder'):
        return
    transformer.time_text_embed._bfg_orig_forward = transformer.time_text_embed.forward
    transformer.time_text_embed.forward = partial(
        guidance_embed_bypass_forward, transformer.time_text_embed
    )

# restore the forward function


def restore_flux_guidance(transformer):
    if not hasattr(transformer.time_text_embed, '_bfg_orig_forward'):
        return
    transformer.time_text_embed.forward = transformer.time_text_embed._bfg_orig_forward
    del transformer.time_text_embed._bfg_orig_forward
