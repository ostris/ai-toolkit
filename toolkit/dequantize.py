

from functools import partial
from optimum.quanto.tensor import QTensor
import torch


def hacked_state_dict(self, *args, **kwargs):
    orig_state_dict = self.orig_state_dict(*args, **kwargs)
    new_state_dict = {}
    for key, value in orig_state_dict.items():
        if key.endswith("._scale"):
            continue
        if key.endswith(".input_scale"):
            continue
        if key.endswith(".output_scale"):
            continue
        if key.endswith("._data"):
            key = key[:-6]
            scale = orig_state_dict[key + "._scale"]
            # scale is the original dtype
            dtype = scale.dtype
            scale = scale.float()
            value = value.float()
            dequantized = value * scale
            
            # handle input and output scaling if they exist
            input_scale = orig_state_dict.get(key + ".input_scale")
            
            if input_scale is not None:
                # make sure the tensor is 1.0
                if input_scale.item() != 1.0:
                    raise ValueError("Input scale is not 1.0, cannot dequantize")
                
            output_scale = orig_state_dict.get(key + ".output_scale")
            
            if output_scale is not None:
                # make sure the tensor is 1.0
                if output_scale.item() != 1.0:
                    raise ValueError("Output scale is not 1.0, cannot dequantize")
            
            new_state_dict[key] = dequantized.to('cpu', dtype=dtype)
        else:
            new_state_dict[key] = value
    return new_state_dict

# hacks the state dict so we can dequantize before saving
def patch_dequantization_on_save(model):
    model.orig_state_dict = model.state_dict
    model.state_dict = partial(hacked_state_dict, model)
  
  
def dequantize_parameter(module: torch.nn.Module, param_name: str) -> bool:
    """
    Convert a quantized parameter back to a regular Parameter with floating point values.
    
    Args:
        module: The module containing the parameter to unquantize
        param_name: Name of the parameter to unquantize (e.g., 'weight', 'bias')
    
    Returns:
        bool: True if parameter was unquantized, False if it was already unquantized
    """
    
    # Check if the parameter exists
    if not hasattr(module, param_name):
        raise AttributeError(f"Module has no parameter named '{param_name}'")
    
    param = getattr(module, param_name)
    
    # If it's not a parameter or not quantized, nothing to do
    if not isinstance(param, torch.nn.Parameter):
        raise TypeError(f"'{param_name}' is not a Parameter")
    if not isinstance(param, QTensor):
        return False
        
    # Convert to float tensor while preserving device and requires_grad
    with torch.no_grad():
        float_tensor = param.float()
        new_param = torch.nn.Parameter(
            float_tensor,
            requires_grad=param.requires_grad
        )
    
    # Replace the parameter
    setattr(module, param_name, new_param)
    
    return True