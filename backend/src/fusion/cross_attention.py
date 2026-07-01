"""
Aegis-MM Cross-Modal Attention Layer
Enables bidirectional attention between diverse sensory streams (Vision <-> Audio)
to uncover temporal and cross-modal synchronization discrepancies.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionBlock(nn.Module):
    """
    Cross-modal multi-head attention block allowing one modality sequence/vector
    to query representations from another modality.
    """
    def __init__(self, query_dim: int, kv_dim: int, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, query_dim) or (B, N_q, query_dim)
            key_value: (B, kv_dim) or (B, N_kv, kv_dim)
        Returns:
            Fused representation aligned with query shape (B, hidden_dim)
        """
        # Ensure 3D sequence tensors for multi-head attention math
        q_3d = query.unsqueeze(1) if query.dim() == 2 else query
        kv_3d = key_value.unsqueeze(1) if key_value.dim() == 2 else key_value
        
        B, N_q, _ = q_3d.shape
        _, N_kv, _ = kv_3d.shape
        
        # Linear projections split into multi-head format: (B, N, heads, head_dim) -> (B, heads, N, head_dim)
        q = self.q_proj(q_3d).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_3d).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_3d).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)  # (B, heads, N_q, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.hidden_dim)
        context = self.out_proj(context)
        
        # Residual connection + LayerNorm
        res_q = self.q_proj(q_3d)
        h1 = self.norm1(res_q + context)
        
        # FFN + Residual
        out = self.norm2(h1 + self.ffn(h1))
        
        return out.squeeze(1) if query.dim() == 2 else out
