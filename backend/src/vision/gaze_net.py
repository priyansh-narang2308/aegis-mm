"""
Aegis-MM Gaze & Head-Pose Estimator
Estimates 3D head rotation Euler angles (Pitch, Yaw, Roll) and 2D gaze vectors
from spatial feature embeddings to detect off-screen reading and fraud.
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict


class GazeHeadPoseEstimator(nn.Module):
    """
    Multitask projection head taking backbone embeddings (B x feature_dim)
    and predicting head orientation, gaze vector, and gaze drift anomaly score.
    """
    def __init__(self, feature_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Predicts Euler angles (Pitch, Yaw, Roll normalized between -1.0 and 1.0)
        self.head_pose_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 3),
            nn.Tanh()
        )
        
        # Predicts 2D screen gaze coordinates (gx, gy in [-1.0, 1.0])
        self.gaze_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 2),
            nn.Tanh()
        )
        
        # Predicts likelihood of off-screen gaze drift [0.0, 1.0]
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embedding: (B, feature_dim) tensor from SpatialAttentionBackbone
        Returns:
            Dictionary containing:
            - head_pose: (B, 3) Pitch, Yaw, Roll
            - gaze_vector: (B, 2) Gaze x, y coordinates
            - gaze_anomaly_score: (B, 1) Probability of suspicious off-screen focus
        """
        feat = self.shared_mlp(embedding)
        
        head_pose = self.head_pose_head(feat)
        gaze_vector = self.gaze_head(feat)
        anomaly_score = self.anomaly_classifier(feat)
        
        return {
            "head_pose": head_pose,
            "gaze_vector": gaze_vector,
            "gaze_anomaly_score": anomaly_score,
        }
