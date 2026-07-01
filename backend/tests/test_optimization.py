"""
Unit tests for Aegis-MM Quantization & Tradeoff Benchmarking.
Verifies int8/int4 quantization wrappers, model conversion, and tradeoff table generation.
"""
import pytest
import torch
import torch.nn as nn
from src.optimization.quantization import quantize_tensor_int8, quantize_tensor_int4, QuantizedLinearWrapper, apply_quantization_to_model
from src.optimization.bench_tradeoff import run_tradeoff_benchmark


def test_int8_quantization_precision():
    x = torch.randn(10, 10)
    q_x, s_x = quantize_tensor_int8(x)
    assert q_x.dtype == torch.int8
    
    approx = q_x.to(torch.float32) * s_x
    mae = torch.mean(torch.abs(x - approx)).item()
    assert mae < 0.05  # INT8 quantization error should be minimal


def test_int4_quantization_precision():
    x = torch.randn(10, 10)
    q_x, s_x = quantize_tensor_int4(x)
    assert q_x.dtype == torch.int8
    assert torch.max(q_x) <= 7 and torch.min(q_x) >= -7


def test_model_quantization_wrapper():
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
    count = apply_quantization_to_model(model, bits=8)
    assert count == 2
    assert isinstance(model[0], QuantizedLinearWrapper)
    
    x = torch.randn(2, 64)
    out = model(x)
    assert out.shape == (2, 16)


def test_tradeoff_benchmark_table():
    res = run_tradeoff_benchmark(num_iterations=5)
    assert "FP32 (Full Precision)" in res
    assert "INT8 (8-Bit Quantized)" in res
    assert "INT4 (4-Bit Quantized)" in res
    assert res["INT8 (8-Bit Quantized)"]["compression_ratio"] == "4x"
    assert res["INT4 (4-Bit Quantized)"]["compression_ratio"] == "8x"
