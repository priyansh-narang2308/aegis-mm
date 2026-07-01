"""
Unit tests for Aegis-MM Multimodal Late-Fusion Transformer Network.
Verifies cross-modal attention shapes, backward pass stability, and execution latency.
"""
import time
import pytest
import torch
from src.fusion.cross_attention import CrossModalAttentionBlock
from src.fusion.multimodal_guardrail import MultimodalGuardrailFusionNet


def test_cross_modal_attention_shapes():
    block = CrossModalAttentionBlock(query_dim=128, kv_dim=128, hidden_dim=128)
    block.eval()
    
    q = torch.randn(4, 128)
    kv = torch.randn(4, 128)
    with torch.no_grad():
        out = block(q, kv)
        
    assert out.shape == (4, 128)


def test_multimodal_guardrail_fusion_shapes():
    fusion_net = MultimodalGuardrailFusionNet(
        vision_dim=128, audio_dim=128, cognitive_dim=128, telemetry_dim=32, hidden_dim=128
    )
    fusion_net.eval()
    
    batch_size = 4
    v = torch.randn(batch_size, 128)
    a = torch.randn(batch_size, 128)
    c = torch.randn(batch_size, 128)
    t = torch.randn(batch_size, 32)
    
    with torch.no_grad():
        out = fusion_net(v, a, c, t)
        
    assert out["fused_embedding"].shape == (batch_size, 128)
    assert out["overall_risk_score"].shape == (batch_size, 1)
    assert out["synchronization_discrepancy"].shape == (batch_size, 1)
    assert out["multimodal_confidence"].shape == (batch_size, 1)


def test_multimodal_fusion_gradient_flow():
    fusion_net = MultimodalGuardrailFusionNet(
        vision_dim=128, audio_dim=128, cognitive_dim=128, telemetry_dim=32, hidden_dim=128
    )
    fusion_net.train()
    
    v = torch.randn(2, 128, requires_grad=True)
    a = torch.randn(2, 128, requires_grad=True)
    c = torch.randn(2, 128, requires_grad=True)
    t = torch.randn(2, 32, requires_grad=True)
    
    out = fusion_net(v, a, c, t)
    loss = out["overall_risk_score"].mean() + out["synchronization_discrepancy"].mean()
    loss.backward()
    
    assert v.grad is not None
    assert a.grad is not None
    assert c.grad is not None
    assert t.grad is not None


def test_multimodal_fusion_latency():
    fusion_net = MultimodalGuardrailFusionNet(
        vision_dim=128, audio_dim=128, cognitive_dim=128, telemetry_dim=32, hidden_dim=128
    )
    fusion_net.eval()
    
    v = torch.randn(1, 128)
    a = torch.randn(1, 128)
    c = torch.randn(1, 128)
    t = torch.randn(1, 32)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = fusion_net(v, a, c, t)
            
    start = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        with torch.no_grad():
            _ = fusion_net(v, a, c, t)
    elapsed_ms = ((time.perf_counter() - start) / iterations) * 1000.0
    
    print(f"\nAverage Multimodal Late-Fusion forward pass latency: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 15.0, f"Fusion latency {elapsed_ms:.2f}ms exceeds streaming SLA!"
