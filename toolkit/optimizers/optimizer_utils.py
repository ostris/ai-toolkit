import torch
from torch import Tensor
from typing import Optional
from optimum.quanto import QBytesTensor


def compute_scale_for_dtype(tensor, dtype):
    """
    Compute appropriate scale for the given tensor and target dtype.
    
    Args:
        tensor: Input tensor to be quantized
        dtype: Target dtype for quantization
    Returns:
        Appropriate scale factor for the quantization
    """
    if dtype == torch.int8:
        abs_max = torch.max(torch.abs(tensor))
        return abs_max / 127.0 if abs_max > 0 else 1.0
    elif dtype == torch.uint8:
        max_val = torch.max(tensor)
        min_val = torch.min(tensor)
        range_val = max_val - min_val
        return range_val / 255.0 if range_val > 0 else 1.0
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # For float8, we typically want to preserve the magnitude of the values
        # while fitting within the representable range of the format
        abs_max = torch.max(torch.abs(tensor))
        if dtype == torch.float8_e4m3fn:
            # e4m3fn has range [-448, 448] with no infinities
            max_representable = 448.0
        else:  # torch.float8_e5m2
            # e5m2 has range [-57344, 57344] with infinities
            max_representable = 57344.0
        
        return abs_max / max_representable if abs_max > 0 else 1.0
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")

def quantize_tensor(tensor, dtype):
    """
    Quantize a floating-point tensor to the target dtype with appropriate scaling.
    
    Args:
        tensor: Input tensor (float)
        dtype: Target dtype for quantization
    Returns:
        quantized_data: Quantized tensor
        scale: Scale factor used
    """
    scale = compute_scale_for_dtype(tensor, dtype)
    
    if dtype == torch.int8:
        quantized_data = torch.clamp(torch.round(tensor / scale), -128, 127).to(dtype)
    elif dtype == torch.uint8:
        quantized_data = torch.clamp(torch.round(tensor / scale), 0, 255).to(dtype)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # For float8, we scale and then cast directly to the target type
        # The casting operation will handle the appropriate rounding
        scaled_tensor = tensor / scale
        quantized_data = scaled_tensor.to(dtype)
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")
        
    return quantized_data, scale


def update_parameter(target, result_float):
    """
    Updates a parameter tensor, handling both regular torch.Tensor and QBytesTensor cases
    with proper rescaling for quantized tensors.
    
    Args:
        target: The parameter to update (either torch.Tensor or QBytesTensor)
        result_float: The new values to assign (torch.Tensor)
    """
    if isinstance(target, QBytesTensor):
        # Get the target dtype from the existing quantized tensor
        target_dtype = target._data.dtype
        
        # Handle device placement
        device = target._data.device
        result_float = result_float.to(device)
        
        # Compute new quantized values and scale
        quantized_data, new_scale = quantize_tensor(result_float, target_dtype)
        
        # Update the internal tensors with newly computed values
        target._data.copy_(quantized_data)
        target._scale.copy_(new_scale)
    else:
        # Regular tensor update
        target.copy_(result_float)


def get_format_params(dtype: torch.dtype) -> tuple[int, int]:
    """
    Returns (mantissa_bits, total_bits) for each format.
    mantissa_bits excludes the implicit leading 1.
    """
    if dtype == torch.float32:
        return 23, 32
    elif dtype == torch.bfloat16:
        return 7, 16
    elif dtype == torch.float16:
        return 10, 16
    elif dtype == torch.float8_e4m3fn:
        return 3, 8
    elif dtype == torch.float8_e5m2:
        return 2, 8
    elif dtype == torch.int8:
        return 0, 8  # Int8 doesn't have mantissa bits
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
def copy_stochastic_bf16(target: torch.Tensor, source: torch.Tensor):
    # adapted from https://github.com/Nerogar/OneTrainer/blob/411532e85f3cf2b52baa37597f9c145073d54511/modules/util/bf16_stochastic_rounding.py#L5
    # create a random 16 bit integer
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


def copy_stochastic(target: torch.Tensor, source: torch.Tensor, eps: Optional[float] = None) -> None:
    with torch.no_grad():
        # assert if target is on cpu, throw error
        assert target.device.type != 'cpu', "Target is on cpu!"
        assert source.device.type != 'cpu', "Source is on cpu!"
        
        if target.dtype == torch.float32:
            target.copy_(source)
            return
        if target.dtype == torch.bfloat16:
            copy_stochastic_bf16(target, source)
            return

        mantissa_bits, _ = get_format_params(target.dtype)
        round_factor = 2 ** (23 - mantissa_bits)

        # Add uniform noise for stochastic rounding
        noise = torch.rand_like(source, device=source.device) - 0.5
        rounded = torch.round(source * round_factor + noise)
        result_float = rounded / round_factor

        # Clamp for float8
        if target.dtype == torch.float8_e4m3fn:
            result_float.clamp_(-448.0, 448.0)
        elif target.dtype == torch.float8_e5m2:
            result_float.clamp_(-57344.0, 57344.0)

        update_parameter(target, result_float)


class Auto8bitTensor:
    def __init__(self, data: Tensor, *args, **kwargs):
        if isinstance(data, dict):  # Add constructor from state dict
            self._load_from_state_dict(data)
        else:
            abs_max = data.abs().max().item()
            scale = abs_max / 127.0 if abs_max > 0 else 1.0

            self.quantized = (data / scale).round().clamp(-127, 127).to(torch.int8)
            self.scale = scale
            self.orig_dtype = data.dtype

    def dequantize(self) -> Tensor:
        return self.quantized.to(dtype=torch.float32) * self.scale

    def to(self, *args, **kwargs):
        # Handle the dtype argument whether it's positional or keyword
        dtype = None
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
            args = args[1:]
        elif 'dtype' in kwargs:
            dtype = kwargs['dtype']
            del kwargs['dtype']

        if dtype is not None:
            # First dequantize then convert to requested dtype
            return self.dequantize().to(dtype=dtype, *args, **kwargs)

        # If no dtype specified, just pass through to parent
        return self.dequantize().to(*args, **kwargs)

    def state_dict(self):
        """Returns a dictionary containing the current state of the tensor."""
        return {
            'quantized': self.quantized,
            'scale': self.scale,
            'orig_dtype': self.orig_dtype
        }

    def _load_from_state_dict(self, state_dict):
        """Loads the tensor state from a state dictionary."""
        self.quantized = state_dict['quantized']
        self.scale = state_dict['scale']
        self.orig_dtype = state_dict['orig_dtype']

    def __str__(self):
        return f"Auto8bitTensor({self.dequantize()})"


def stochastic_grad_accummulation(param):
    if hasattr(param, "_accum_grad"):
        grad_fp32 = param._accum_grad.clone().to(torch.float32)
        grad_fp32.add_(param.grad.to(torch.float32))
        copy_stochastic(param._accum_grad, grad_fp32)
        del grad_fp32
        del param.grad
    else:
        param._accum_grad = param.grad.clone()
        del param.grad
