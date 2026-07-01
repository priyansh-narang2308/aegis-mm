"""
Aegis-MM Unified Vision Anomaly Pipeline
Combines Spatial Attention Backbone with Gaze/Head-Pose estimation and
visual deepfake/integrity scoring into a unified PyTorch forward pass.
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from .backbone import SpatialAttentionBackbone
from .gaze_net import GazeHeadPoseEstimator


class VisionAnomalyPipeline(nn.Module):
    """
    End-to-end vision proctoring pipeline processing raw frame batches (B x 3 x H x W).
    Outputs embedded representations, gaze telemetry, and visual risk indices.
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = SpatialAttentionBackbone(in_channels=in_channels, feature_dim=feature_dim)
        self.gaze_estimator = GazeHeadPoseEstimator(feature_dim=feature_dim, dropout=dropout)
        
        # Deepfake visual discrepancy / lip-sync visual artifact branch
        self.visual_artifact_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: Input video frame tensor of shape (B, 3, H, W)
        Returns:
            Dict containing vision embedding, spatial feature map, head pose,
            gaze vector, gaze anomaly score, and visual artifact risk.
        """
        embedding, spatial_map = self.backbone(frames)
        gaze_telemetry = self.gaze_estimator(embedding)
        visual_artifact_score = self.visual_artifact_head(embedding)
        
        # Combine into unified telemetry dictionary
        output = {
            "vision_embedding": embedding,              # (B, feature_dim)
            "spatial_feature_map": spatial_map,         # (B, feature_dim, H', W')
            "head_pose": gaze_telemetry["head_pose"],   # (B, 3)
            "gaze_vector": gaze_telemetry["gaze_vector"],# (B, 2)
            "gaze_anomaly_score": gaze_telemetry["gaze_anomaly_score"], # (B, 1)
            "visual_artifact_score": visual_artifact_score,             # (B, 1)
        }
        return output
