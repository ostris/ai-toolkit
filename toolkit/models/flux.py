
# forward that bypasses the guidance embedding so it can be avoided during training.
from functools import partial
from typing import Optional
import torch
from diffusers import FluxTransformer2DModel


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

def new_device_to(self: FluxTransformer2DModel, *args, **kwargs):
    # Store original device if provided in args or kwargs
    device_in_kwargs = 'device' in kwargs
    device_in_args = any(isinstance(arg, (str, torch.device)) for arg in args)
    
    device = None
    # Remove device from kwargs if present
    if device_in_kwargs:
        device = kwargs['device']
        del kwargs['device']
    
    # Only filter args if we detected a device argument
    if device_in_args:
        args = list(args)
        for idx, arg in enumerate(args):
            if isinstance(arg, (str, torch.device)):
                device = arg
                del args[idx]
    
    self.pos_embed = self.pos_embed.to(device, *args, **kwargs)
    self.time_text_embed = self.time_text_embed.to(device, *args, **kwargs)
    self.context_embedder = self.context_embedder.to(device, *args, **kwargs)
    self.x_embedder = self.x_embedder.to(device, *args, **kwargs)
    for block in self.transformer_blocks:
        block.to(block._split_device, *args, **kwargs)
    for block in self.single_transformer_blocks:
        block.to(block._split_device, *args, **kwargs)
    
    self.norm_out = self.norm_out.to(device, *args, **kwargs)
    self.proj_out = self.proj_out.to(device, *args, **kwargs)
    
    
    
    return self

    


def split_gpu_double_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    if hidden_states.device != self._split_device:
        hidden_states = hidden_states.to(self._split_device)
    if encoder_hidden_states.device != self._split_device:
        encoder_hidden_states = encoder_hidden_states.to(self._split_device)
    if temb.device != self._split_device:
        temb = temb.to(self._split_device)
    if image_rotary_emb is not None and image_rotary_emb[0].device != self._split_device:
        # is a tuple of tensors
        image_rotary_emb = tuple([t.to(self._split_device) for t in image_rotary_emb])
    return self._pre_gpu_split_forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)


def split_gpu_single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    **kwargs
):
    if hidden_states.device != self._split_device:
        hidden_states = hidden_states.to(device=self._split_device)
    if temb.device != self._split_device:
        temb = temb.to(device=self._split_device)
    if image_rotary_emb is not None and image_rotary_emb[0].device != self._split_device:
        # is a tuple of tensors
        image_rotary_emb = tuple([t.to(self._split_device) for t in image_rotary_emb])
    
    hidden_state_out = self._pre_gpu_split_forward(hidden_states, temb, image_rotary_emb, joint_attention_kwargs, **kwargs)
    if hasattr(self, "_split_output_device"):
        return hidden_state_out.to(self._split_output_device)
    return hidden_state_out


def add_model_gpu_splitter_to_flux(
    transformer: FluxTransformer2DModel,
    # ~ 5 billion for all other params
    other_module_params: Optional[int] = 5e9,
    # since they are not trainable, multiply by smaller number
    other_module_param_count_scale: Optional[float] = 0.3
):
    gpu_id_list = [i for i in range(torch.cuda.device_count())]
    
    # if len(gpu_id_list) > 2:
    #     raise ValueError("Cannot split to more than 2 GPUs currently.")
    other_module_params *= other_module_param_count_scale
    
    # since we are not tuning the 
    total_params = sum(p.numel() for p in transformer.parameters()) + other_module_params
    
    params_per_gpu = total_params / len(gpu_id_list)
    
    current_gpu_idx = 0
    # text encoders, vae, and some non block layers will all be on gpu 0
    current_gpu_params = other_module_params

    for double_block in transformer.transformer_blocks:
        device = torch.device(f"cuda:{current_gpu_idx}")
        double_block._pre_gpu_split_forward = double_block.forward
        double_block.forward = partial(
            split_gpu_double_block_forward, double_block)
        double_block._split_device = device
        # add the params to the current gpu
        current_gpu_params += sum(p.numel() for p in double_block.parameters())
        # if the current gpu params are greater than the params per gpu, move to next gpu
        if current_gpu_params > params_per_gpu:
            current_gpu_idx += 1
            current_gpu_params = 0
            if current_gpu_idx >= len(gpu_id_list):
                current_gpu_idx = gpu_id_list[-1]
        
    for single_block in transformer.single_transformer_blocks:
        device = torch.device(f"cuda:{current_gpu_idx}")
        single_block._pre_gpu_split_forward = single_block.forward
        single_block.forward = partial(
            split_gpu_single_block_forward, single_block)
        single_block._split_device = device
        # add the params to the current gpu
        current_gpu_params += sum(p.numel() for p in single_block.parameters())
        # if the current gpu params are greater than the params per gpu, move to next gpu
        if current_gpu_params > params_per_gpu:
            current_gpu_idx += 1
            current_gpu_params = 0
            if current_gpu_idx >= len(gpu_id_list):
                current_gpu_idx = gpu_id_list[-1]
    
    # add output device to last layer
    transformer.single_transformer_blocks[-1]._split_output_device = torch.device("cuda:0")
    
    transformer._pre_gpu_split_to = transformer.to
    transformer.to = partial(new_device_to, transformer)
