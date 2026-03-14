"""
dofa_vit.py
===========
DOFA (Dynamic One-For-All) Vision Transformer backbone for geospatial
foundation modelling, adapted for Sentinel-1/2 multi-sensor fusion.

Reference:
  Xiong et al. (2024) "DOFA: A Universal Model for Geospatial Intelligence"
  arXiv:2403.15356

Architecture:
  ViT-Base/16 pre-trained on diverse Earth Observation data (optical, SAR,
  hyperspectral) using a self-supervised masked autoencoder objective.
  Input wavelength metadata conditions the patch embedding → enables
  zero-shot transfer across arbitrary sensor configurations.
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────

def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Truncated normal initialisation (Timm-style)."""
    with torch.no_grad():
        return tensor.normal_(0, std).clamp_(-2 * std, 2 * std)


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
    return x.div(keep) * noise


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ── Wavelength-Conditioned Patch Embedding ───────────────────────────────────

class WavelengthConditionedPatchEmbed(nn.Module):
    """
    Dynamic patch embedding conditioned on per-channel centre wavelengths.
    Each channel produces its own kernel via a small wavelength MLP,
    enabling the backbone to generalise across sensor configurations.
    """

    def __init__(
        self,
        img_size: int  = 224,
        patch_size: int = 16,
        in_chans: int   = 12,
        embed_dim: int  = 768,
        wavelength_mlp_hidden: int = 128,
    ):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim  = embed_dim

        # Learnable wavelength → kernel weight MLP
        self.wl_mlp = nn.Sequential(
            nn.Linear(1, wavelength_mlp_hidden),
            nn.GELU(),
            nn.Linear(wavelength_mlp_hidden, patch_size * patch_size * embed_dim),
        )
        # Shared bias
        self.proj_bias = nn.Parameter(torch.zeros(embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)

    def forward(
        self,
        x: torch.Tensor,                    # (B, C, H, W)
        wavelengths: torch.Tensor,          # (C,) in nm
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.patch_size

        # Generate per-channel conv kernels from wavelengths
        wl_norm = wavelengths.unsqueeze(-1).float() / 2500.0  # rough normalise
        kernels = self.wl_mlp(wl_norm)                         # (C, P*P*D)
        kernels = kernels.view(C, self.embed_dim, P, P)        # (C, D, P, P)

        # Unfold image into patches and apply dynamic kernel
        # x: (B, C, H, W) → (B*C, 1, H, W)
        x_per_ch = x.view(B * C, 1, H, W)
        kernels_bc = kernels.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(
            B * C, self.embed_dim, P, P
        )

        # Grouped convolution (each channel has its own kernel)
        # Workaround: loop over channels for simplicity
        patches_list: List[torch.Tensor] = []
        for i in range(C):
            ch = x[:, i:i+1, :, :]                         # (B, 1, H, W)
            k  = kernels[i].unsqueeze(0)                    # (1, D, P, P)
            k  = k.expand(B, -1, -1, -1).reshape(B, self.embed_dim, P, P)

            # (B, D, nH, nW) but we need grouped conv per sample
            out = F.conv2d(
                ch.reshape(1, B, H, W),
                k.reshape(B, self.embed_dim, P, P),
                padding=0, stride=P, groups=B
            ).squeeze(0)                                    # (B*D, nH, nW)
            nH = H // P;  nW = W // P
            out = out.view(B, self.embed_dim, nH, nW)
            patches_list.append(out)

        # Sum contributions from all channels
        tokens = sum(patches_list) + self.proj_bias.view(1, -1, 1, 1)  # (B, D, nH, nW)
        tokens = tokens.flatten(2).transpose(1, 2)                      # (B, N, D)

        # Prepend CLS token and add positional embedding
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        return tokens                                       # (B, N+1, D)


# ── Multi-Head Self-Attention ────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int  = 12,
        qkv_bias: bool  = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.qkv        = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


# ── MLP Block ────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float    = 4.0,
        drop: float         = 0.0,
        attn_drop: float    = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ── DOFA ViT Backbone ────────────────────────────────────────────────────────

class DOFAViTBackbone(nn.Module):
    """
    DOFA ViT-Base/16 backbone.

    Produces hierarchical feature maps by collecting intermediate block outputs
    at quarters of the depth (for use with UperNet decoder).

    Parameters
    ----------
    img_size    : int        input spatial resolution (must be divisible by 16)
    in_chans    : int        total input channels (S1+S2 after fusion projection)
    embed_dim   : int        transformer hidden dimension
    depth       : int        number of transformer blocks
    num_heads   : int
    wavelengths : Tensor     centre wavelengths (nm) for each input channel
    """

    # Approximate centre wavelengths (nm) – SAR assigned synthetic values
    WAVELENGTHS_S1 = torch.tensor([5_600.0, 5_600.0])              # C-band SAR (VV, VH)
    WAVELENGTHS_S2 = torch.tensor([492, 560, 665, 704, 740,
                                   783, 833, 865, 1610, 2190], dtype=torch.float32)

    def __init__(
        self,
        img_size: int   = 224,
        in_chans: int   = 12,        # 2 SAR + 10 optical
        embed_dim: int  = 768,
        depth: int      = 12,
        num_heads: int  = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.depth       = depth
        self.img_size    = img_size
        self.num_patches = (img_size // 16) ** 2

        self.patch_embed = WavelengthConditionedPatchEmbed(
            img_size=img_size,
            patch_size=16,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[i],
            )
            for i in range(depth)
        ])

        # Indices at which to extract intermediate features (for FPN decoder)
        self.out_indices = {2, 5, 8, 11}  # depths 3, 6, 9, 12 (0-indexed)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_pretrained(self, ckpt_path: str, strict: bool = False):
        """Load DOFA pre-trained weights with key remapping."""
        state = torch.load(ckpt_path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        missing, unexpected = self.load_state_dict(state, strict=strict)
        logger.info(
            "Loaded DOFA weights: %d missing, %d unexpected keys",
            len(missing), len(unexpected)
        )

    def forward(
        self,
        x: torch.Tensor,               # (B, C, H, W)  fused channels
        wavelengths: Optional[torch.Tensor] = None,  # (C,)
    ) -> List[torch.Tensor]:
        """
        Returns
        -------
        features : list of 4 tensors, each (B, H/16, W/16, D)
                   at successive depths for UperNet decoder
        """
        if wavelengths is None:
            wavelengths = torch.cat([
                self.WAVELENGTHS_S1, self.WAVELENGTHS_S2
            ]).to(x.device)

        tokens = self.patch_embed(x, wavelengths)   # (B, N+1, D)
        B, _, D = tokens.shape
        H = W   = self.img_size // 16

        features: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            if i in self.out_indices:
                # Discard CLS, reshape to spatial
                feat = self.norm(tokens[:, 1:, :])  # (B, N, D)
                feat = feat.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
                features.append(feat)

        return features   # 4 × (B, D, H', W')
