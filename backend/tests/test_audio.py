"""
Unit tests for PyTorch Audio Spectral Anomaly Pipeline.
Verifies tensor shapes, determinism, and real-time execution speed.
"""
import time
import pytest
import torch
from src.audio.spectral_net import SpectralAudioNet


@pytest.fixture
def audio_pipeline():
    model = SpectralAudioNet(sample_rate=16000, n_mels=64, feature_dim=128)
    model.eval()
    return model


def test_audio_pipeline_raw_waveform_shape(audio_pipeline):
    batch_size = 4
    # 1 second of audio at 16kHz
    raw_audio = torch.randn(batch_size, 16000)
    
    with torch.no_grad():
        out = audio_pipeline(raw_audio)
        
    assert out["audio_embedding"].shape == (batch_size, 128)
    assert out["spectrogram"].shape[0] == batch_size
    assert out["spectrogram"].shape[1] == 1
    assert out["spectrogram"].shape[2] == 64
    assert out["synthetic_speech_score"].shape == (batch_size, 1)
    assert out["acoustic_anomaly_score"].shape == (batch_size, 1)


def test_audio_pipeline_spectrogram_input_shape(audio_pipeline):
    batch_size = 2
    precomputed_spec = torch.randn(batch_size, 1, 64, 100)
    
    with torch.no_grad():
        out = audio_pipeline(precomputed_spec)
        
    assert out["audio_embedding"].shape == (batch_size, 128)
    assert out["synthetic_speech_score"].shape == (batch_size, 1)


def test_audio_pipeline_determinism(audio_pipeline):
    x = torch.randn(2, 16000)
    with torch.no_grad():
        out1 = audio_pipeline(x)
        out2 = audio_pipeline(x)
        
    assert torch.allclose(out1["audio_embedding"], out2["audio_embedding"], atol=1e-6)
    assert torch.allclose(out1["synthetic_speech_score"], out2["synthetic_speech_score"], atol=1e-6)


def test_audio_pipeline_latency(audio_pipeline):
    x = torch.randn(1, 16000)
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = audio_pipeline(x)
            
    start = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        with torch.no_grad():
            _ = audio_pipeline(x)
    elapsed_ms = ((time.perf_counter() - start) / iterations) * 1000.0
    
    print(f"\nAverage single-chunk Audio forward pass latency: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 50.0, f"Audio latency {elapsed_ms:.2f}ms exceeds streaming SLA!"
