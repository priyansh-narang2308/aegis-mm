"""
Unit tests for Aegis-MM Parameter-Efficient Fine-Tuning (LoRA) Module.
Verifies exact zero-initialization invariants, weight merging, and execution speed.
"""
import time
import pytest
import torch
from src.peft.lora_layer import LoRALinear
from src.peft.cognitive_alignment import CognitiveAlignmentModule


def test_lora_initialization_invariant():
    """
    Hu et al. 2021 invariant: B matrix is zero initialized, so at initialization,
    LoRA forward pass MUST exactly equal base layer forward pass.
    """
    layer = LoRALinear(in_features=64, out_features=32, rank=8, alpha=16.0)
    layer.eval()
    
    x = torch.randn(4, 64)
    with torch.no_grad():
        base_out = layer.base_layer(x)
        lora_out = layer(x)
        
    assert torch.allclose(base_out, lora_out, atol=1e-6)


def test_lora_weight_merging_invariant():
    """
    Verifies that merging B*A into base weights produces exact same forward pass
    while eliminating auxiliary matrix multiplication overhead.
    """
    layer = LoRALinear(in_features=64, out_features=32, rank=8, alpha=16.0)
    # Simulate non-zero B weights during training
    with torch.no_grad():
        layer.lora_B.normal_(mean=0.0, std=0.1)
        
    layer.eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        out_unmerged = layer(x)
        
        layer.merge_weights()
        assert layer._merged is True
        out_merged = layer(x)
        
        layer.unmerge_weights()
        assert layer._merged is False
        out_unmerged_restored = layer(x)
        
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5)
    assert torch.allclose(out_unmerged, out_unmerged_restored, atol=1e-6)


def test_cognitive_alignment_shapes_and_speed():
    model = CognitiveAlignmentModule(
        vision_dim=128, audio_dim=128, telemetry_dim=32, hidden_dim=128, lora_rank=8
    )
    model.eval()
    
    batch_size = 4
    v = torch.randn(batch_size, 128)
    a = torch.randn(batch_size, 128)
    t = torch.randn(batch_size, 32)
    
    start = time.perf_counter()
    with torch.no_grad():
        out = model(v, a, t)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    
    assert out["cognitive_embedding"].shape == (batch_size, 128)
    assert out["ai_fluency_score"].shape == (batch_size, 1)
    assert out["integrity_confidence"].shape == (batch_size, 1)
    print(f"\nCognitive Alignment LoRA forward pass latency: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 20.0
