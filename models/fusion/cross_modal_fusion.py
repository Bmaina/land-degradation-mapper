"""
cross_modal_fusion.py
=====================
Cross-modal attention fusion for Sentinel-1 SAR and Sentinel-2 optical
feature streams prior to the shared ViT backbone.

Three fusion strategies:
  1. concat         – channel concatenation (baseline)
  2. gated          – learned per-channel gating (soft selection)
  3. cross_modal_attention – cross-attention between modalities (default)

The default cross-modal attention is inspired by:
  "Cross-Attention is All You Need: Adapting Pretrained Vision Transformers for
   Multi-Sensor Remote Sensing" (Scheibenreif et al., 2023)
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FusionStrategy = Literal["concat", "gated", "cross_modal_attention"]


# ═══════════════════════════════════════════════════════════════════════════════
class ConcatFusion(nn.Module):
    """Simplest baseline: concatenate along channel dim, project to fused_dim."""

    def __init__(self, s1_dim: int, s2_dim: int, fused_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(s1_dim + s2_dim, fused_dim, 1, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.GELU(),
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([s1, s2], dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
class GatedFusion(nn.Module):
    """
    Gated fusion: each modality generates a sigmoid gate that weights its own
    contribution.  Allows the model to suppress noisy channels (e.g., cloud-
    affected optical data) while amplifying informative SAR return.
    """

    def __init__(self, s1_dim: int, s2_dim: int, fused_dim: int):
        super().__init__()
        self.s1_proj = nn.Conv2d(s1_dim, fused_dim, 1)
        self.s2_proj = nn.Conv2d(s2_dim, fused_dim, 1)
        self.gate_s1 = nn.Sequential(nn.Conv2d(fused_dim * 2, fused_dim, 1), nn.Sigmoid())
        self.gate_s2 = nn.Sequential(nn.Conv2d(fused_dim * 2, fused_dim, 1), nn.Sigmoid())
        self.out_norm = nn.LayerNorm([fused_dim, 1, 1])  # broadcast-compatible

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        f1 = self.s1_proj(s1)
        f2 = self.s2_proj(s2)
        combined = torch.cat([f1, f2], dim=1)
        g1 = self.gate_s1(combined)
        g2 = self.gate_s2(combined)
        fused = g1 * f1 + g2 * f2
        return fused


# ═══════════════════════════════════════════════════════════════════════════════
class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion (default):

      • S1 features act as *queries*, attending to S2 key-value pairs  → s1→s2
      • S2 features act as *queries*, attending to S1 key-value pairs  → s2→s1
      • Both enriched representations are summed and projected.

    This allows each modality to selectively pull complementary information from
    the other (e.g., SAR texture highlighting areas of optical ambiguity).
    """

    def __init__(
        self,
        s1_dim: int,
        s2_dim: int,
        fused_dim: int,
        num_heads: int = 8,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.fused_dim = fused_dim

        # Project each modality to common dimension
        self.s1_proj = nn.Conv2d(s1_dim, fused_dim, 1)
        self.s2_proj = nn.Conv2d(s2_dim, fused_dim, 1)

        # Cross-attention modules
        self.s1_to_s2_attn = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.s2_to_s1_attn = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward after cross-attention
        self.ff = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, fused_dim),
        )

        self.norm1 = nn.LayerNorm(fused_dim)
        self.norm2 = nn.LayerNorm(fused_dim)
        self.norm_out = nn.LayerNorm(fused_dim)

    def _to_seq(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """(B, C, H, W) → (B, H*W, C)  and return H, W."""
        B, C, H, W = x.shape
        return x.flatten(2).permute(0, 2, 1), H, W

    def _to_spatial(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, H*W, C) → (B, C, H, W)."""
        return x.permute(0, 2, 1).reshape(x.shape[0], -1, H, W)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        s1 : (B, s1_dim, H, W)
        s2 : (B, s2_dim, H, W)   – must match s1 spatial dims

        Returns
        -------
        fused : (B, fused_dim, H, W)
        """
        f1 = self.s1_proj(s1)   # (B, D, H, W)
        f2 = self.s2_proj(s2)

        seq1, H, W = self._to_seq(f1)    # (B, N, D)
        seq2, _,  _ = self._to_seq(f2)

        # S1 queries attend to S2 context
        x1, _ = self.s1_to_s2_attn(query=seq1, key=seq2, value=seq2)
        x1 = self.norm1(x1 + seq1)      # residual

        # S2 queries attend to S1 context
        x2, _ = self.s2_to_s1_attn(query=seq2, key=seq1, value=seq1)
        x2 = self.norm2(x2 + seq2)

        # Fuse enriched streams
        fused_seq = self.ff(torch.cat([x1, x2], dim=-1))    # (B, N, D)
        fused_seq = self.norm_out(fused_seq)
        return self._to_spatial(fused_seq, H, W)             # (B, D, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
def build_fusion(
    strategy: FusionStrategy,
    s1_dim: int,
    s2_dim: int,
    fused_dim: int,
    **kwargs,
) -> nn.Module:
    """Factory function for fusion modules."""
    if strategy == "concat":
        return ConcatFusion(s1_dim, s2_dim, fused_dim)
    elif strategy == "gated":
        return GatedFusion(s1_dim, s2_dim, fused_dim)
    elif strategy == "cross_modal_attention":
        return CrossModalAttentionFusion(s1_dim, s2_dim, fused_dim, **kwargs)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy!r}")


# ═══════════════════════════════════════════════════════════════════════════════
class SpatiotemporalAttention(nn.Module):
    """
    Lightweight temporal self-attention applied over a sequence of fused
    feature maps from different acquisition dates.

    Each time step produces a spatial feature map (B, D, H, W).
    We treat time as the sequence dimension and apply self-attention,
    then aggregate via weighted mean.

    Input:  (B, T, D, H, W)  — T time steps
    Output: (B, D, H, W)     — temporally integrated features
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm   = nn.LayerNorm(dim)
        self.ff     = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.norm2  = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D, H, W = x.shape
        # Global spatial pooling → one token per time step
        tokens = x.flatten(3).mean(-1)          # (B, T, D)
        x_attn, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(x_attn + tokens)
        tokens = self.norm2(self.ff(tokens) + tokens)

        # Soft-max weights for temporal aggregation
        weights = tokens.softmax(dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, T, D, 1, 1)
        return (x * weights).sum(dim=1)         # (B, D, H, W)
