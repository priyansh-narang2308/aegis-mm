
import copy
import torch
from typing import Dict, Any, List
from src.fusion.multimodal_guardrail import MultimodalGuardrailFusionNet
from src.optimization.quantization import apply_quantization_to_model


def run_tradeoff_benchmark(num_iterations: int = 50) -> Dict[str, Any]:
    """
    Executes comparative benchmarks across precision modes and returns structured tradeoff table.
    """
    # 1. Base FP32 Model
    model_fp32 = MultimodalGuardrailFusionNet()
    model_fp32.eval()
    
    model_int8 = copy.deepcopy(model_fp32)
    apply_quantization_to_model(model_int8, bits=8)
    model_int8.eval()
    
    model_int4 = copy.deepcopy(model_fp32)
    apply_quantization_to_model(model_int4, bits=4)
    model_int4.eval()
    
    # Synthetic input vectors
    v = torch.randn(1, 128)
    a = torch.randn(1, 128)
    c = torch.randn(1, 128)
    t = torch.randn(1, 32)
    
    # Base FP32 output baseline
    with torch.no_grad():
        base_out = model_fp32(v, a, c, t)
        
    models = {
        "FP32 (Full Precision)": (model_fp32, 32),
        "INT8 (8-Bit Quantized)": (model_int8, 8),
        "INT4 (4-Bit Quantized)": (model_int4, 4)
    }
    
    results = {}
    for name, (model, bits) in models.items():
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(v, a, c, t)
                
        start = time.perf_counter()
        for _ in range(num_iterations):
            with torch.no_grad():
                out = model(v, a, c, t)
        elapsed_ms = ((time.perf_counter() - start) / num_iterations) * 1000.0
        
        # Calculate fidelity error relative to FP32 baseline
        mae_risk = torch.abs(out["overall_risk_score"] - base_out["overall_risk_score"]).item()
        
        # Calculate theoretical parameter memory reduction
        param_count = sum(p.numel() for p in model.parameters())
        # Quantized wrappers store int8 buffers
        buffer_count = sum(b.numel() for b in model.buffers() if b.dtype == torch.int8)
        
        if bits == 32:
            mem_kb = (param_count * 4) / 1024.0
        elif bits == 8:
            mem_kb = (buffer_count * 1) / 1024.0
        elif bits == 4:
            mem_kb = (buffer_count * 0.5) / 1024.0
            
        compression = 32.0 / float(bits)
        
        results[name] = {
            "bits": bits,
            "latency_ms": round(elapsed_ms, 3),
            "mae_fidelity_error": round(mae_risk, 5),
            "memory_kb": round(mem_kb, 2),
            "compression_ratio": f"{int(compression)}x"
        }
        
    return results


def format_markdown_tradeoff_table(results: Dict[str, Any]) -> str:
    table = "\n### Quantization Quality vs. Speed Tradeoff Matrix\n\n"
    table += "| Precision Mode | Memory Footprint (KB) | Compression | Fidelity Error (MAE) | Forward Latency (ms) |\n"
    table += "| :--- | :---: | :---: | :---: | :---: |\n"
    for name, data in results.items():
        table += f"| **{name}** | {data['memory_kb']} KB | **{data['compression_ratio']}** | `{data['mae_fidelity_error']}` | **{data['latency_ms']} ms** |\n"
    return table


if __name__ == "__main__":
    res = run_tradeoff_benchmark()
    print(format_markdown_tradeoff_table(res))
