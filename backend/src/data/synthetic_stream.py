"""
Aegis-MM Synthetic Stream Generator
Generates realistic multi-channel streaming frames (Vision, Audio, Telemetry)
on-the-fly with configurable adversarial anomaly injections for zero-dataset testing.
"""
import time
import math
import torch
from typing import Dict, Any, Optional


class SyntheticStreamGenerator:
    """
    Simulates a live candidate interview stream producing synchronized 30 FPS video frames,
    16kHz audio waveform chunks, and behavioral telemetry vectors.
    Supports injecting synthetic adversarial anomalies on demand.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.frame_idx = 0

    def get_next_frame(self, anomaly_type: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Generates next stream packet.
        Args:
            anomaly_type: None, 'gaze_drift', 'tts_audio', or 'prompt_injection'
        Returns:
            Dict containing video_frame (1, 3, 224, 224), audio_chunk (1, 16000), and telemetry (1, 32).
        """
        self.frame_idx += 1
        g = torch.Generator()
        g.manual_seed(self.seed + self.frame_idx)

        # Base natural tensors
        frame = torch.randn(1, 3, 224, 224, generator=g) * 0.1
        audio = torch.randn(1, 16000, generator=g) * 0.05
        telemetry = torch.zeros(1, 32)
        
        # Base normal behavior values
        telemetry[0, 0] = 35.0  # Normal ping latency ms
        telemetry[0, 1] = 1.5   # Normal typing cadence
        telemetry[0, 2] = 0.0   # Focus loss count

        # Inject adversarial anomalies if requested
        if anomaly_type == "gaze_drift":
            # Simulate high spatial variance and asymmetric feature shift
            frame[:, :, :50, :50] += 2.5
            telemetry[0, 2] = 5.0  # Multiple rapid focus shifts off-screen
        elif anomaly_type == "tts_audio":
            # Inject high-frequency periodic harmonic tone typical of TTS vocoders
            t = torch.linspace(0, 1, 16000)
            harmonic = torch.sin(2 * math.pi * 3000 * t) * 0.8
            audio += harmonic
        elif anomaly_type == "prompt_injection":
            # Simulate sudden massive keystroke burst + copy-paste latency drop
            telemetry[0, 1] = 45.0 # Unnatural typing burst (copy-paste speed)
            telemetry[0, 3] = 1.0  # Clipboard paste event detected

        return {
            "frame_id": torch.tensor([self.frame_idx]),
            "timestamp": time.time(),
            "video_frame": frame,
            "audio_chunk": audio,
            "telemetry": telemetry,
            "injected_anomaly": anomaly_type or "none"
        }
