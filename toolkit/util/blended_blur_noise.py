import torch

cached_multipier = None

def get_multiplier(timesteps, num_timesteps=1000):
    global cached_multipier
    if cached_multipier is None:
        # creates a bell curve
        x = torch.arange(num_timesteps, dtype=torch.float32)
        y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)

        # Shift minimum to 0
        y_shifted = y - y.min()

        # Scale to make mean 1
        cached_multipier = y_shifted * (num_timesteps / y_shifted.sum())
    
    scale_list = []
    # get the idx multiplier for each timestep
    for i in range(timesteps.shape[0]):
        idx = min(int(timesteps[i].item()) - 1, 0)
        scale_list.append(cached_multipier[idx:idx + 1])
    
    scales = torch.cat(scale_list, dim=0)
    
    batch_multiplier = scales.view(-1, 1, 1, 1)
    
    return batch_multiplier


def get_blended_blur_noise(latents, noise, timestep):
    latent_chunks = torch.chunk(latents, latents.shape[0], dim=0)
    
    # timestep is 1000 to 0
    # timestep = timestep.to(latents.device, dtype=latents.dtype)
    
    # scale it so timestep 1000 is 0 and 0 is 2
    # blur_strength = value_map(timestep, 1000, 0, 0, 1.0)
    # blur_strength = timestep / 500.0
    # blur_strength = blur_strength.view(-1, 1, 1, 1)
    
    # scale to 2.0 max
    # blur_strength = get_multiplier(timestep).to(
    #     latents.device, dtype=latents.dtype
    # ) * 2.0
    
    # blur_strength = 2.0
    
    blurred_latent_chunks = []
    for i in range(len(latent_chunks)):
        latent_chunk = latent_chunks[i]
        # get two random scalers 0.1 to 0.9
        # scaler1 = random.uniform(0.2, 0.8)
        scaler1 = 0.25
        scaler2 = scaler1
        
        # shrink latents by 1/4 and bring them back for blurring using interpolation
        blur_latents = torch.nn.functional.interpolate(
            latent_chunk,
            size=(int(latents.shape[2] * scaler1), int(latents.shape[3] * scaler2)),
            mode='bilinear',
            align_corners=False
        )
        blur_latents = torch.nn.functional.interpolate(
            blur_latents,
            size=(latents.shape[2], latents.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        # only the difference of the blur from ground truth
        blur_latents = blur_latents - latent_chunk
        blurred_latent_chunks.append(blur_latents)
    
    blur_latents = torch.cat(blurred_latent_chunks, dim=0)
    
        
    # make random strength along batch 0 to 1
    blur_strength = torch.rand((latents.shape[0], 1, 1, 1), device=latents.device, dtype=latents.dtype) * 2
    
    blur_latents = blur_latents * blur_strength
    
    noise = noise + blur_latents
    return noise
    