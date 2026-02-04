"""
AdamW with FP8 State Storage

Uses PyTorch native float8 for optimizer state (m, v) storage.
Uses e4m3fn for exp_avg_sq (needs more precision) and e5m2 for exp_avg.

Key differences from bitsandbytes:
- bitsandbytes: uint8 linear quantization + absmax scaling
- This: float8 native format with per-tensor scaling

Usage:
    optimizer = AdamWFP8(model.parameters(), lr=1e-4)
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional, Tuple, Dict, Any


class AdamWFP8(Optimizer):
    """
    AdamW optimizer with FP8 state storage.
    
    Uses mixed FP8 precision:
    - exp_avg (m): float8_e5m2 (larger dynamic range for momentum)
    - exp_avg_sq (v): float8_e4m3fn (more precision for variance)
    
    Both states use per-tensor scaling to handle small values.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-4)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.0)
        amsgrad: Whether to use AMSGrad variant (default: False)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)
        
        # Check FP8 support - use e4m3fn for better precision
        self._check_fp8_support()
        self._fp8_dtype_m = torch.float8_e5m2      # momentum: larger range
        self._fp8_dtype_v = torch.float8_e4m3fn    # variance: more precision
    
    def _check_fp8_support(self):
        """Verify PyTorch FP8 support is available."""
        if not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError(
                "PyTorch float8_e4m3fn not available. "
                "Requires PyTorch >= 2.1. Current version: " + torch.__version__
            )
    
    def _to_fp8_scaled(self, tensor: torch.Tensor, fp8_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert tensor to FP8 with per-tensor scaling.
        Returns (fp8_tensor, scale) where original = fp8_tensor * scale
        """
        # Compute scale factor to map values into FP8 representable range
        # e4m3fn max ~448, e5m2 max ~57344
        if fp8_dtype == torch.float8_e4m3fn:
            max_fp8 = 448.0
        else:
            max_fp8 = 57344.0
        
        abs_max = tensor.abs().max().clamp(min=1e-12)
        scale = abs_max / max_fp8
        
        # Scale down, convert to FP8
        scaled = tensor / scale
        fp8_tensor = scaled.to(fp8_dtype)
        
        return fp8_tensor, scale
    
    def _from_fp8_scaled(self, fp8_tensor: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Convert FP8 tensor back using stored scale."""
        return fp8_tensor.to(dtype) * scale
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamWFP8 does not support sparse gradients")
                
                amsgrad = group['amsgrad']
                state = self.state[p]
                
                # State initialization - directly in FP8 (zeros are exactly representable)
                if len(state) == 0:
                    state['step'] = 0
                    # FP8 zeros - no precision loss, saves memory from start
                    state['exp_avg'] = torch.zeros_like(p, dtype=self._fp8_dtype_m)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=self._fp8_dtype_v)
                    state['scale_m'] = torch.tensor(1.0, device=p.device)
                    state['scale_v'] = torch.tensor(1.0, device=p.device)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, dtype=self._fp8_dtype_v)
                        state['scale_max_v'] = torch.tensor(1.0, device=p.device)
                
                state['step'] += 1
                step = state['step']
                
                # Upcast FP8 states to fp32 for computation
                exp_avg = self._from_fp8_scaled(state['exp_avg'], state['scale_m'], torch.float32)
                exp_avg_sq = self._from_fp8_scaled(state['exp_avg_sq'], state['scale_v'], torch.float32)
                if amsgrad:
                    max_exp_avg_sq = self._from_fp8_scaled(state['max_exp_avg_sq'], state['scale_max_v'], torch.float32)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Decoupled weight decay (AdamW style)
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Cast grad to fp32 for computation
                grad_fp32 = grad.to(torch.float32)
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)
                
                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                # Update parameters (cast to param dtype)
                update = exp_avg / denom * step_size
                p.data.add_(update.to(p.dtype), alpha=-1)
                
                # Downcast states to FP8 for storage (with scaling)
                state['exp_avg'], state['scale_m'] = self._to_fp8_scaled(exp_avg, self._fp8_dtype_m)
                state['exp_avg_sq'], state['scale_v'] = self._to_fp8_scaled(exp_avg_sq, self._fp8_dtype_v)
                if amsgrad:
                    state['max_exp_avg_sq'], state['scale_max_v'] = self._to_fp8_scaled(max_exp_avg_sq, self._fp8_dtype_v)

        
        return loss
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict with FP8 states converted to fp32 for saving."""
        state_dict = super().state_dict()
        
        # Convert FP8 states to fp32 for compatibility
        for param_id, param_state in state_dict['state'].items():
            for key, scale_key in [('exp_avg', 'scale_m'), ('exp_avg_sq', 'scale_v'), ('max_exp_avg_sq', 'scale_max_v')]:
                if key in param_state:
                    scale = param_state.get(scale_key, torch.tensor(1.0))
                    param_state[key] = param_state[key].to(torch.float32) * scale
                    if scale_key in param_state:
                        del param_state[scale_key]
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict, states will be converted to FP8 on next step."""
        # Keep states in fp32, they will be converted on next step() call
        super().load_state_dict(state_dict)
