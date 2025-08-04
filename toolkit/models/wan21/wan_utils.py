import torch
import torch.nn.functional as F


def add_first_frame_conditioning(
    latent_model_input,
    first_frame,
    vae
):
    """
    Adds first frame conditioning to a video diffusion model input.

    Args:
        latent_model_input: Original latent input (bs, channels, num_frames, height, width)
        first_frame: Tensor of first frame to condition on (bs, channels, height, width)
        vae: VAE model for encoding the conditioning

    Returns:
        conditioned_latent: The complete conditioned latent input (bs, 36, num_frames, height, width)
    """
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample)

    # Get number of frames from latent model input
    _, _, num_latent_frames, _, _ = latent_model_input.shape

    # Calculate original number of frames
    # For n original frames, there are (n-1)//4 + 1 latent frames
    # So to get n: n = (num_latent_frames-1)*4 + 1
    num_frames = (num_latent_frames - 1) * 4 + 1
    
    if len(first_frame.shape) == 3:
        # we have a single image
        first_frame = first_frame.unsqueeze(0)
    
    # if it doesnt match the batch size, we need to expand it
    if first_frame.shape[0] != latent_model_input.shape[0]:
        first_frame = first_frame.expand(latent_model_input.shape[0], -1, -1, -1)
        
    # resize first frame to match the latent model input
    vae_scale_factor = vae.config.scale_factor_spatial
    first_frame = F.interpolate(
        first_frame,
        size=(latent_model_input.shape[3] * vae_scale_factor, latent_model_input.shape[4] * vae_scale_factor),
        mode='bilinear',
        align_corners=False
    )

    # Add temporal dimension to first frame
    first_frame = first_frame.unsqueeze(2)

    # Create video condition with first frame and zeros for remaining frames
    zero_frame = torch.zeros_like(first_frame)
    video_condition = torch.cat([
        first_frame,
        *[zero_frame for _ in range(num_frames - 1)]
    ], dim=2)

    # Prepare for VAE encoding (bs, channels, num_frames, height, width)
    # video_condition = video_condition.permute(0, 2, 1, 3, 4)

    # Encode with VAE
    latent_condition = vae.encode(
        video_condition.to(device, dtype)
    ).latent_dist.sample()
    latent_condition = latent_condition.to(device, dtype)
    
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device, dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, dtype
    )
    latent_condition = (latent_condition - latents_mean) * latents_std
    

    # Create mask: 1 for conditioning frames, 0 for frames to generate
    batch_size = first_frame.shape[0]
    latent_height = latent_condition.shape[3]
    latent_width = latent_condition.shape[4]

    # Initialize mask for all frames
    mask_lat_size = torch.ones(
        batch_size, 1, num_frames, latent_height, latent_width)

    # Set all non-first frames to 0
    mask_lat_size[:, :, list(range(1, num_frames))] = 0

    # Special handling for first frame
    first_frame_mask = mask_lat_size[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(
        first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)

    # Combine first frame mask with rest
    mask_lat_size = torch.concat(
        [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)

    # Reshape and transpose for model input
    mask_lat_size = mask_lat_size.view(
        batch_size, -1, vae_scale_factor_temporal, latent_height, latent_width)
    mask_lat_size = mask_lat_size.transpose(1, 2)
    mask_lat_size = mask_lat_size.to(device, dtype)

    # Combine conditioning with latent input
    first_frame_condition = torch.concat(
        [mask_lat_size, latent_condition], dim=1)
    conditioned_latent = torch.cat(
        [latent_model_input, first_frame_condition], dim=1)

    return conditioned_latent


def add_first_frame_conditioning_v22(
    latent_model_input,
    first_frame,
    vae,
    last_frame=None
):
    """
    Overwrites first few time steps in latent_model_input with VAE-encoded first_frame,
    and returns the modified latent + binary mask (0=conditioned, 1=noise).

    Args:
        latent_model_input: torch.Tensor of shape (bs, 48, T, H, W)
        first_frame: torch.Tensor of shape (bs, 3, H*scale, W*scale)
        vae: VAE model with .encode() and .config.latents_mean/std

    Returns:
        latent: (bs, 48, T, H, W) - modified input latent
        mask: (bs, 1, T, H, W) - binary mask
    """
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    bs, _, T, H, W = latent_model_input.shape
    scale = vae.config.scale_factor_spatial
    target_h = H * scale
    target_w = W * scale

    # Ensure shape
    if first_frame.ndim == 3:
        first_frame = first_frame.unsqueeze(0)
    if first_frame.shape[0] != bs:
        first_frame = first_frame.expand(bs, -1, -1, -1)

    # Resize and encode
    first_frame_up = F.interpolate(first_frame, size=(target_h, target_w), mode="bilinear", align_corners=False)
    first_frame_up = first_frame_up.unsqueeze(2)  # (bs, 3, 1, H, W)
    encoded = vae.encode(first_frame_up).latent_dist.sample().to(dtype).to(device)

    # Normalize
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, dtype)
    encoded = (encoded - mean) * std

    # Replace in latent
    latent = latent_model_input.clone()
    latent[:, :, :encoded.shape[2]] = encoded  # typically first frame: [:, :, 0]

    # Mask: 0 where conditioned, 1 otherwise
    mask = torch.ones(bs, 1, T, H, W, device=device, dtype=dtype)
    mask[:, :, :encoded.shape[2]] = 0.0
    
    if last_frame is not None:
        # If last_frame is provided, encode it similarly
        last_frame_up = F.interpolate(last_frame, size=(target_h, target_w), mode="bilinear", align_corners=False)
        last_frame_up = last_frame_up.unsqueeze(2)
        last_encoded = vae.encode(last_frame_up).latent_dist.sample().to(dtype).to(device)
        last_encoded = (last_encoded - mean) * std
        latent[:, :, -last_encoded.shape[2]:] = last_encoded  # replace last
        mask[:, :, -last_encoded.shape[2]:] = 0.0  #
        # Ensure mask is still binary
        mask = mask.clamp(0.0, 1.0)

    return latent, mask