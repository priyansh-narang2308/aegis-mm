"""
Aegis-MM Low-Rank Adaptation (LoRA) PyTorch Module
Implements Parameter-Efficient Fine-Tuning linear adaptation layer:
    h = W_0 * x + (alpha / r) * (B * A * x)
Inspired by Hu et al., 2021 (LoRA: Low-Rank Adaptation of Large Language Models).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation linear layer that injects trainable rank-r decomposition matrices
    alongside frozen or adaptable pre-trained projection weights.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Base dense projection (simulating base pre-trained VLM/LLM projection)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA Low-Rank Matrices
        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, rank))
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_lora_parameters(self):
        """
        Hu et al. 2021 initialization:
        A ~ N(0, 1/r) or Kaiming Uniform
        B = 0 (Ensures delta_W = B * A = 0 at initialization)
        """
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.rank > 0 and not getattr(self, "_merged", False):
            lora_out = (
                self.lora_dropout(x) @ self.lora_A.T
            ) @ self.lora_B.T
            return base_out + lora_out * self.scaling
        return base_out

    def merge_weights(self):
        """
        Merges LoRA adapter matrix B * A into base_layer weight matrix for zero-latency inference.
        """
        if self.rank > 0 and not getattr(self, "_merged", False):
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self._merged = True

    def unmerge_weights(self):
        """
        Unmerges LoRA adapter matrix from base_layer weight matrix.
        """
        if self.rank > 0 and getattr(self, "_merged", False):
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self._merged = False
