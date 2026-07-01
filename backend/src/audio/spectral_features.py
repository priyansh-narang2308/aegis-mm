"""
Aegis-MM Audio Spectral Feature Extractor
Converts raw 16kHz audio waveform chunks into log-Mel spectrograms and extracts
temporal-spectral feature maps using Depthwise Separable 1D/2D Convolutions.
"""
import torch
import torch.nn as nn
import torchaudio.transforms as T


class MelSpectrogramExtractor(nn.Module):
    """
    Transforms raw waveform batches (B x Samples) into normalized log-Mel spectrograms (B x n_mels x Time).
    """
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 512, n_mels: int = 64):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, num_samples) or (B, 1, num_samples)
        Returns:
            log_mel: (B, 1, n_mels, time_steps)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        mel_spec = self.mel_transform(waveform)
        # Log compression for numerical stability and human perception alignment
        log_mel = torch.log(mel_spec + 1e-6)
        return log_mel


class TemporalSpectralConvBlock(nn.Module):
    """
    Processes 2D spectrogram feature maps across time and frequency axes
    to isolate synthetic TTS artifacts and high-frequency acoustic anomalies.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, time_steps)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x
