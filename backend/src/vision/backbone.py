"""
Aegis-MM Spatial Attention Vision Backbone
Lightweight convolutional feature extractor using Depthwise Separable Convolutions
and Spatial Self-Attention for real-time (sub-30ms) frame analysis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """
    Lightweight convolution block reducing FLOPs for streaming inference.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class SpatialAttentionBlock(nn.Module):
    """
    Computes spatial attention weights across feature map (H x W) to highlight
    salient facial regions (eyes, mouth, gaze vectors).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B x N x C'
        proj_key = self.key_conv(x).view(B, -1, H * W)                       # B x C' x N
        
        energy = torch.bmm(proj_query, proj_key)                             # B x N x N
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(B, -1, H * W)                   # B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))              # B x C x N
        out = out.view(B, C, H, W)
        
        return x + self.gamma * out


class SpatialAttentionBackbone(nn.Module):
    """
    Extracts dense spatial representations from B x 3 x H x W image tensors.
    Returns feature maps and global pooled embedding vector.
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.stage1 = DepthwiseSeparableConv2d(32, 64, stride=2)
        self.stage2 = DepthwiseSeparableConv2d(64, 128, stride=2)
        self.attn = SpatialAttentionBlock(128)
        self.stage3 = DepthwiseSeparableConv2d(128, feature_dim, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor):
        # x: (B, 3, 224, 224)
        x = self.stem(x)       # (B, 32, 112, 112)
        x = self.stage1(x)     # (B, 64, 56, 56)
        x = self.stage2(x)     # (B, 128, 28, 28)
        x = self.attn(x)       # Spatial attention over facial region
        x_map = self.stage3(x) # (B, feature_dim, 14, 14)
        
        pooled = self.pool(x_map).flatten(1) # (B, feature_dim)
        return pooled, x_map
