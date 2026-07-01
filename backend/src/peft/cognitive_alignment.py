"""
Aegis-MM Cognitive Alignment & AI Fluency Module
Applies Parameter-Efficient LoRA projection layers to cross-modal embeddings
to compute real-time AI Fluency scores and candidate integrity confidence.
"""
import torch
import torch.nn as nn
from typing import Dict
from .lora_layer import LoRALinear


class CognitiveAlignmentModule(nn.Module):
    """
    Evaluates candidate cognitive alignment by projecting joint vision-audio-telemetry
    vectors through parameter-efficient LoRA layers.
    Distinguishes high-level AI Fluency (productive synthesis) from adversarial fraud.
    """
    def __init__(
        self,
        vision_dim: int = 128,
        audio_dim: int = 128,
        telemetry_dim: int = 32,
        hidden_dim: int = 128,
        lora_rank: int = 8,
        lora_alpha: float = 16.0
    ):
        super().__init__()
        in_dim = vision_dim + audio_dim + telemetry_dim
        
        # LoRA-adapted projection layers
        self.lora_proj1 = LoRALinear(
            in_features=in_dim,
            out_features=hidden_dim,
            rank=lora_rank,
            alpha=lora_alpha
        )
        self.act1 = nn.SiLU(inplace=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.lora_proj2 = LoRALinear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rank=lora_rank,
            alpha=lora_alpha
        )
        self.act2 = nn.SiLU(inplace=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # AI Fluency Index [0.0, 1.0] (Measures structural AI usage competence)
        self.fluency_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Session Integrity Confidence [0.0, 1.0]
        self.integrity_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        vision_embed: torch.Tensor,
        audio_embed: torch.Tensor,
        telemetry: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vision_embed: (B, vision_dim)
            audio_embed: (B, audio_dim)
            telemetry: (B, telemetry_dim) e.g. latency, keystroke cadence, focus shift rate
        Returns:
            Dict containing cognitive latent embedding, AI fluency score, and integrity score.
        """
        x = torch.cat([vision_embed, audio_embed, telemetry], dim=-1)
        
        h1 = self.norm1(self.act1(self.lora_proj1(x)))
        h2 = self.norm2(self.act2(self.lora_proj2(h1)))
        
        fluency = self.fluency_head(h2)
        integrity = self.integrity_head(h2)
        
        return {
            "cognitive_embedding": h2,
            "ai_fluency_score": fluency,
            "integrity_confidence": integrity,
        }
