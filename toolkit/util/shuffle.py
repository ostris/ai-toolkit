import torch
import random
import numpy as np

def shuffle_tensor_along_axis(tensor, axis=0, seed=None):
    """
    Shuffle a tensor along a specified axis without affecting the global random state.
    
    Args:
        tensor (torch.Tensor): The input tensor to shuffle
        axis (int, optional): The axis along which to shuffle. Defaults to 0.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        
    Returns:
        torch.Tensor: The shuffled tensor
    """
    # Clone the tensor to avoid in-place modifications
    shuffled_tensor = tensor.clone()
    
    # Store original random states
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    
    try:
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Get the size of the dimension to shuffle
        dim_size = tensor.shape[axis]
        
        # Generate random indices for shuffling
        indices = torch.randperm(dim_size)
        
        # Create a slice object to shuffle along the specified axis
        slices = [slice(None)] * tensor.dim()
        slices[axis] = indices
        
        # Apply the shuffle
        shuffled_tensor = tensor[slices]
    
    finally:
        # Restore original random states
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(py_state)
        
    return shuffled_tensor