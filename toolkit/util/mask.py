import torch
import numpy as np
import os
import torch.nn.functional as F
from PIL import Image
import time
import random


def generate_random_mask(
    batch_size,
    height=256,
    width=256,
    device='cuda',
    min_coverage=0.2,
    max_coverage=0.8,
    num_blobs_range=(1, 3)
):
    """
    Generate random blob masks for a batch of images.
    Fast GPU version with smooth, non-circular blob shapes.

    Args:
        batch_size (int): Number of masks to generate
        height (int): Height of the mask
        width (int): Width of the mask
        device (str): Device to run the computation on ('cuda' or 'cpu')
        min_coverage (float): Minimum percentage of the image to be covered (0-1)
        max_coverage (float): Maximum percentage of the image to be covered (0-1)
        num_blobs_range (tuple): Range of number of blobs (min, max)

    Returns:
        torch.Tensor: Binary masks with shape (batch_size, 1, height, width)
    """
    # Initialize masks on GPU
    masks = torch.zeros((batch_size, 1, height, width), device=device)

    # Pre-compute coordinate grid on GPU
    y_indices = torch.arange(height, device=device).view(
        height, 1).expand(height, width)
    x_indices = torch.arange(width, device=device).view(
        1, width).expand(height, width)

    # Prepare gaussian kernels for smoothing
    small_kernel = get_gaussian_kernel(7, 1.0).to(device)
    small_kernel = small_kernel.view(1, 1, 7, 7)

    large_kernel = get_gaussian_kernel(15, 2.5).to(device)
    large_kernel = large_kernel.view(1, 1, 15, 15)

    # Constants
    max_radius = min(height, width) // 3
    min_radius = min(height, width) // 8

    # For each mask in the batch
    for b in range(batch_size):
        # Determine number of blobs for this mask
        num_blobs = np.random.randint(
            num_blobs_range[0], num_blobs_range[1] + 1)

        # Target coverage for this mask
        target_coverage = np.random.uniform(min_coverage, max_coverage)

        # Initialize this mask
        mask = torch.zeros(1, 1, height, width, device=device)

        # Generate blobs with smoother edges
        for _ in range(num_blobs):
            # Create a low-frequency noise field first (for smooth organic shapes)
            noise_field = torch.zeros(height, width, device=device)

            # Use low-frequency sine waves to create base shape distortion
            # This creates smoother warping compared to pure random noise
            num_waves = np.random.randint(2, 5)
            for i in range(num_waves):
                freq_x = np.random.uniform(1.0, 3.0) * np.pi / width
                freq_y = np.random.uniform(1.0, 3.0) * np.pi / height
                phase_x = np.random.uniform(0, 2 * np.pi)
                phase_y = np.random.uniform(0, 2 * np.pi)
                amp = np.random.uniform(0.5, 1.0) * max_radius / (i+1.5)

                # Generate smooth wave patterns
                wave = torch.sin(x_indices * freq_x + phase_x) * \
                    torch.sin(y_indices * freq_y + phase_y) * amp
                noise_field += wave

            # Basic ellipse parameters
            center_y = np.random.randint(height//4, 3*height//4)
            center_x = np.random.randint(width//4, 3*width//4)
            radius = np.random.randint(min_radius, max_radius)

            # Squeeze and stretch the ellipse with random scaling
            scale_y = np.random.uniform(0.6, 1.4)
            scale_x = np.random.uniform(0.6, 1.4)

            # Random rotation
            theta = np.random.uniform(0, 2 * np.pi)
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)

            # Calculate elliptical distance field
            y_scaled = (y_indices - center_y) * scale_y
            x_scaled = (x_indices - center_x) * scale_x

            # Apply rotation
            rotated_y = y_scaled * cos_theta - x_scaled * sin_theta
            rotated_x = y_scaled * sin_theta + x_scaled * cos_theta

            # Compute distances
            distances = torch.sqrt(rotated_y**2 + rotated_x**2)

            # Apply the smooth noise field to the distance field
            perturbed_distances = distances + noise_field

            # Create base blob
            blob = (perturbed_distances < radius).float(
            ).unsqueeze(0).unsqueeze(0)

            # Apply strong smoothing for very smooth edges
            # Double smoothing to get really organic edges
            blob = F.pad(blob, (7, 7, 7, 7), mode='reflect')
            blob = F.conv2d(blob, large_kernel, padding=0)

            # Apply threshold to get a nice shape
            rand_threshold = np.random.uniform(0.3, 0.6)
            blob = (blob > rand_threshold).float()

            # Apply second smoothing pass
            blob = F.pad(blob, (3, 3, 3, 3), mode='reflect')
            blob = F.conv2d(blob, small_kernel, padding=0)
            blob = (blob > 0.5).float()

            # Add to mask
            mask = torch.maximum(mask, blob)

        # Ensure desired coverage
        current_coverage = mask.mean().item()

        # Scale if needed to match target coverage
        if current_coverage > 0:  # Avoid division by zero
            if current_coverage < target_coverage * 0.7:  # Too small
                # Dilate mask to increase coverage
                mask = F.pad(mask, (2, 2, 2, 2), mode='reflect')
                mask = F.max_pool2d(mask, kernel_size=5, stride=1, padding=0)
            elif current_coverage > target_coverage * 1.3:  # Too large
                # Erode mask to decrease coverage
                mask = F.pad(mask, (1, 1, 1, 1), mode='reflect')
                mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=0)
                mask = (mask > 0.7).float()

        # Final smooth and threshold
        mask = F.pad(mask, (3, 3, 3, 3), mode='reflect')
        mask = F.conv2d(mask, small_kernel, padding=0)
        mask = (mask > 0.5).float()

        # Add to batch
        masks[b] = mask

    return masks


def get_gaussian_kernel(kernel_size=5, sigma=1.0):
    """
    Returns a 2D Gaussian kernel.
    """
    # Create 1D kernels
    x = torch.linspace(-sigma * 2, sigma * 2, kernel_size)
    x = x.view(1, -1).repeat(kernel_size, 1)
    y = x.transpose(0, 1)

    # 2D Gaussian
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian /= gaussian.sum()

    return gaussian


def save_masks_as_images(masks, suffix="", output_dir="output"):
    """
    Save generated masks as RGB JPG images using PIL.
    """
    os.makedirs(output_dir, exist_ok=True)

    batch_size = masks.shape[0]
    for i in range(batch_size):
        # Convert mask to numpy array
        mask = masks[i, 0].cpu().numpy()

        # Scale to 0-255 range and convert to uint8
        mask_255 = (mask * 255).astype(np.uint8)

        # Create RGB image (white mask on black background)
        rgb_mask = np.stack([mask_255, mask_255, mask_255], axis=2)

        # Convert to PIL Image and save
        img = Image.fromarray(rgb_mask)
        img.save(os.path.join(output_dir, f"mask_{i:03d}{suffix}.jpg"), quality=95)


def random_dialate_mask(mask, max_percent=0.05):
    """
    Randomly dialates a binary mask with a kernel of random size.
    
    Args:
        mask (torch.Tensor): Input mask of shape [batch_size, channels, height, width]
        max_percent (float): Maximum kernel size as a percentage of the mask size
        
    Returns:
        torch.Tensor: Dialated mask with the same shape as input
    """
    
    size = mask.shape[-1]
    max_size = int(size * max_percent)
    
    # Handle case where max_size is too small
    if max_size < 3:
        max_size = 3
    
    batch_chunks = torch.chunk(mask, mask.shape[0], dim=0)
    out_chunks = []
    
    for i in range(len(batch_chunks)):
        chunk = batch_chunks[i]
        
        # Ensure kernel size is odd for proper padding
        kernel_size = np.random.randint(1, max_size)
        
        # If kernel_size is less than 2, keep the original mask
        if kernel_size < 2:
            out_chunks.append(chunk)
            continue
            
        # Make sure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create normalized dilation kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device) / (kernel_size * kernel_size)
        
        # Pad the mask for convolution
        padding = kernel_size // 2
        padded_mask = F.pad(chunk, (padding, padding, padding, padding), mode='constant', value=0)
        
        # Apply convolution
        dilated = F.conv2d(padded_mask, kernel)
        
        # Random threshold for varied dilation effect
        threshold = np.random.uniform(0.2, 0.8)
        
        # Apply threshold
        dilated = (dilated > threshold).float()
        
        out_chunks.append(dilated)
    
    return torch.cat(out_chunks, dim=0)


if __name__ == "__main__":
    # Parameters
    batch_size = 20
    height = 256
    width = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Generating {batch_size} random blob masks on {device}...")

    for i in range(5):
        # time it
        start = time.time()
        masks = generate_random_mask(
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
            min_coverage=0.2,
            max_coverage=0.8,
            num_blobs_range=(1, 3)
        )
        dialation = random_dialate_mask(masks)
        print(f"Generated {batch_size} masks with shape: {masks.shape}")
        end = time.time()
        # print time in milliseconds
        print(f"Time taken: {(end - start)*1000:.2f} ms")

    print(f"Saving masks to 'output' directory...")
    save_masks_as_images(masks)
    save_masks_as_images(dialation, suffix="_dilated" )

    print("Done!")
