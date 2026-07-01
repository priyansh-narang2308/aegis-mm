"""
Unit tests for Aegis-MM Data Pipeline & Synthetic Stream Generator.
Verifies packet generation, anomaly injection, and dataset loader interfaces.
"""
import pytest
import torch
from src.data.synthetic_stream import SyntheticStreamGenerator
from src.data.loader import ASVSpoofAudioDataset, FaceForensicsVideoDataset


def test_synthetic_stream_generation():
    generator = SyntheticStreamGenerator(seed=100)
    packet1 = generator.get_next_frame(anomaly_type="none")
    packet2 = generator.get_next_frame(anomaly_type="gaze_drift")
    
    assert packet1["frame_id"].item() == 1
    assert packet2["frame_id"].item() == 2
    assert packet1["video_frame"].shape == (1, 3, 224, 224)
    assert packet1["audio_chunk"].shape == (1, 16000)
    assert packet1["telemetry"].shape == (1, 32)
    assert packet2["injected_anomaly"] == "gaze_drift"
    assert packet2["telemetry"][0, 2].item() == 5.0  # Focus loss anomaly value


def test_benchmark_dataset_loaders():
    audio_ds = ASVSpoofAudioDataset(root_dir="/tmp/mock")
    video_ds = FaceForensicsVideoDataset(root_dir="/tmp/mock")
    
    a_wav, a_lbl = audio_ds[0]
    v_frm, v_lbl = video_ds[0]
    
    assert a_wav.shape == (16000,)
    assert a_lbl.shape == (1,)
    assert v_frm.shape == (3, 224, 224)
    assert v_lbl.shape == (1,)
