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


def copy_stochastic(
    target: torch.Tensor,
    source: torch.Tensor,
    eps: Optional[float] = None
) -> None:
    """
    Performs stochastic rounding from source tensor to target tensor.

    Args:
        target: Destination tensor (determines the target format)
        source: Source tensor (typically float32)
        eps: Optional minimum value for stochastic rounding (for numerical stability)
    """
    with torch.no_grad():
        # If target is float32, just copy directly
        if target.dtype == torch.float32:
            target.copy_(source)
            return

        # Special handling for int8
        if target.dtype == torch.int8:
            # Scale the source values to utilize the full int8 range
            scaled = source * 127.0  # Scale to [-127, 127]

            # Add random noise for stochastic rounding
            noise = torch.rand_like(scaled) - 0.5
            rounded = torch.round(scaled + noise)

            # Clamp to int8 range
            clamped = torch.clamp(rounded, -127, 127)
            target.copy_(clamped.to(torch.int8))
            return

        mantissa_bits, _ = get_format_params(target.dtype)

        # Convert source to int32 view
        source_int = source.view(dtype=torch.int32)

        # Calculate number of bits to round
        bits_to_round = 23 - mantissa_bits  # 23 is float32 mantissa bits

        # Create random integers for stochastic rounding
        rand = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << bits_to_round),
        )

        # Add random values to the bits that will be rounded off
        result = source_int.clone()
        result.add_(rand)

        # Mask to keep only the bits we want
        # Create mask with 1s in positions we want to keep
        mask = (-1) << bits_to_round
        result.bitwise_and_(mask)

        # Handle minimum value threshold if specified
        if eps is not None:
            eps_int = torch.tensor(
                eps, dtype=torch.float32).view(dtype=torch.int32)
            zero_mask = (result.abs() < eps_int)
            result[zero_mask] = torch.sign(source_int[zero_mask]) * eps_int

        # Convert back to float32 view
        result_float = result.view(dtype=torch.float32)

        # Special handling for float8 formats
        if target.dtype == torch.float8_e4m3fn:
            result_float.clamp_(-448.0, 448.0)
        elif target.dtype == torch.float8_e5m2:
            result_float.clamp_(-57344.0, 57344.0)

        # Copy the result to the target tensor
        update_parameter(target, result_float)
        # target.copy_(result_float)
        del result, rand, source_int


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
