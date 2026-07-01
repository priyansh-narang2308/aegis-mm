"""
Aegis-MM Weight Quantization Engine
Simulates 8-bit (INT8) and 4-bit (INT4/NF4) symmetric linear quantization
to evaluate production memory compression vs. inference fidelity tradeoffs.
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


def quantize_tensor_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric 8-bit quantization: scales tensor values into [-127, 127] integer buckets.
    Returns int8 tensor and scale factor.
    """
    max_val = torch.max(torch.abs(tensor))
    scale = max_val / 127.0
    if scale == 0:
        scale = torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device)
    q_tensor = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return q_tensor, scale


def dequantize_tensor_int8(q_tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return q_tensor.to(target_dtype) * scale


def quantize_tensor_int4(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric 4-bit quantization simulation: scales tensor values into [-7, 7] integer buckets.
    Returns int8 tensor restricted to 4-bit range [-7, 7] and scale factor.
    """
    max_val = torch.max(torch.abs(tensor))
    scale = max_val / 7.0
    if scale == 0:
        scale = torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device)
    q_tensor = torch.clamp(torch.round(tensor / scale), -7, 7).to(torch.int8)
    return q_tensor, scale


def dequantize_tensor_int4(q_tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return q_tensor.to(target_dtype) * scale


class QuantizedLinearWrapper(nn.Module):
    """
    Wraps standard PyTorch nn.Linear to simulate low-bit quantized weight storage
    and on-the-fly dequantization during inference.
    """
    def __init__(self, linear_layer: nn.Linear, bits: int = 8):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.bits = bits
        self.orig_dtype = linear_layer.weight.dtype
        
        with torch.no_grad():
            if bits == 8:
                q_w, s_w = quantize_tensor_int8(linear_layer.weight.data)
            elif bits == 4:
                q_w, s_w = quantize_tensor_int4(linear_layer.weight.data)
            else:
                raise ValueError(f"Unsupported quantization bits: {bits}. Must be 4 or 8.")
                
        self.register_buffer("q_weight", q_w)
        self.register_buffer("scale", s_w)
        
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.data.clone())
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits == 8:
            approx_w = dequantize_tensor_int8(self.q_weight, self.scale, target_dtype=x.dtype)
        else:
            approx_w = dequantize_tensor_int4(self.q_weight, self.scale, target_dtype=x.dtype)
            
        return F_linear_helper(x, approx_w, self.bias)

    def compute_compression_ratio(self) -> float:
        """
        Returns theoretical memory footprint reduction ratio vs FP32 (32-bit).
        """
        return 32.0 / float(self.bits)


def F_linear_helper(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return torch.nn.functional.linear(input, weight, bias)


def apply_quantization_to_model(model: nn.Module, bits: int = 8) -> int:
    """
    In-place replacement of Linear layers inside a PyTorch model with QuantizedLinearWrapper.
    Returns number of layers quantized.
    """
    count = 0
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, QuantizedLinearWrapper(module, bits=bits))
            count += 1
        else:
            count += apply_quantization_to_model(module, bits=bits)
    return count
