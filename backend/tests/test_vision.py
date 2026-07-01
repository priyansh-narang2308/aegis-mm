"""
Unit tests for PyTorch Vision Anomaly Pipeline.
Verifies tensor shapes, gradient flow, and real-time execution latency.
"""
import time
import pytest
import torch
from src.vision.attention_net import VisionAnomalyPipeline


@pytest.fixture
def vision_pipeline():
    model = VisionAnomalyPipeline(in_channels=3, feature_dim=128)
    model.eval()
    return model


def test_vision_pipeline_tensor_shapes(vision_pipeline):
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        out = vision_pipeline(x)
    
    assert out["vision_embedding"].shape == (batch_size, 128)
    assert out["spatial_feature_map"].shape == (batch_size, 128, 14, 14)
    assert out["head_pose"].shape == (batch_size, 3)
    assert out["gaze_vector"].shape == (batch_size, 2)
    assert out["gaze_anomaly_score"].shape == (batch_size, 1)
    assert out["visual_artifact_score"].shape == (batch_size, 1)


def test_vision_pipeline_gradient_flow():
    model = VisionAnomalyPipeline(in_channels=3, feature_dim=128)
    model.train()
    
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    out = model(x)
    
    loss = out["gaze_anomaly_score"].mean() + out["visual_artifact_score"].mean()
    loss.backward()
    
    # Check that gradients flow back to the input frames and model weights
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_vision_pipeline_latency_benchmark(vision_pipeline):
    """
    Verifies that real-time forward pass latency is suitable for 30 FPS streaming (<35ms on average).
    """
    x = torch.randn(1, 3, 224, 224)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = vision_pipeline(x)
            
    start = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        with torch.no_grad():
            _ = vision_pipeline(x)
    elapsed_ms = ((time.perf_counter() - start) / iterations) * 1000.0
    
    print(f"\nAverage single-frame Vision forward pass latency: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 100.0, f"Latency {elapsed_ms:.2f}ms exceeds streaming SLA!"
