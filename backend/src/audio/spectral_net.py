
import torch
import torch.nn as nn
from typing import Dict, Union
from .spectral_features import MelSpectrogramExtractor, TemporalSpectralConvBlock


class SpectralAudioNet(nn.Module):
    """
    End-to-end audio proctoring pipeline. Takes raw waveforms (B x Samples)
    or precomputed Mel-spectrograms (B x 1 x F x T) and returns acoustic risk telemetry.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        feature_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.mel_extractor = MelSpectrogramExtractor(sample_rate=sample_rate, n_mels=n_mels)
        self.conv_block = TemporalSpectralConvBlock(in_channels=1, out_channels=64)
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.SiLU(inplace=True)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Predicts synthetic voice cloning probability [0.0, 1.0]
        self.synthetic_speech_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Predicts background whispering or earpiece acoustic anomalies [0.0, 1.0]
        self.acoustic_anomaly_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_input: Raw waveform (B, num_samples) OR log-Mel spectrogram (B, 1, n_mels, time_steps)
        Returns:
            Dict containing audio embedding, synthetic speech score, and acoustic anomaly score.
        """
        if audio_input.dim() == 2:
            # Convert raw 1D waveform to 4D log-Mel spectrogram
            spectrogram = self.mel_extractor(audio_input)
        elif audio_input.dim() == 3:
            spectrogram = audio_input.unsqueeze(1)
        else:
            spectrogram = audio_input
            
        feat = self.conv_block(spectrogram)
        feat = self.stage2(feat)
        
        pooled = self.pool(feat).flatten(1)  # (B, feature_dim)
        proj = self.projection(pooled)
        
        synth_score = self.synthetic_speech_head(proj)
        anomaly_score = self.acoustic_anomaly_head(proj)
        
        return {
            "audio_embedding": pooled,                # (B, feature_dim)
            "spectrogram": spectrogram,               # (B, 1, n_mels, T)
            "synthetic_speech_score": synth_score,    # (B, 1)
            "acoustic_anomaly_score": anomaly_score,  # (B, 1)
        }
