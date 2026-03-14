"""
land_deg_mapper.py
==================
Full LandDegMapper model:

  Sentinel-1 (2 ch) ──┐
                       ├── CrossModalAttentionFusion ──► DOFAViT ──► UperNet ──► Degradation Map
  Sentinel-2 (10 ch) ──┘
        + spectral      (fused_dim channels)        (4 feat maps)  (6 classes)
          indices (5 ch)

Optionally integrates multi-temporal composites via SpatiotemporalAttention.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.dofa_vit import DOFAViTBackbone
from models.fusion.cross_modal_fusion import build_fusion, SpatiotemporalAttention
from models.segmentation.upernet_decoder import UperNetDecoder


class LandDegMapper(nn.Module):
    """
    End-to-end land degradation mapping model.

    Parameters
    ----------
    num_classes      : int   degradation class count (default 6)
    fusion_strategy  : str   one of concat | gated | cross_modal_attention
    use_spectral_idx : bool  append 5 spectral indices to S2 features
    use_temporal     : bool  apply spatiotemporal attention over T time steps
    pretrained_ckpt  : str   path to DOFA pre-trained weights
    freeze_epochs    : int   epochs to freeze backbone (linear probe phase)
    """

    #: Approximate channel counts
    S1_CHANNELS      = 2
    S2_CHANNELS      = 10
    IDX_CHANNELS     = 5    # NDVI, NDWI, BSI, NDRE, EVI

    def __init__(
        self,
        num_classes: int       = 6,
        embed_dim: int         = 768,
        img_size: int          = 224,
        fusion_strategy: str   = "cross_modal_attention",
        s1_proj_dim: int       = 256,
        s2_proj_dim: int       = 256,
        fused_dim: int         = 512,
        depth: int             = 12,
        num_heads: int         = 12,
        use_spectral_idx: bool = True,
        use_temporal: bool     = False,
        n_time_steps: int      = 4,
        pretrained_ckpt: Optional[str] = None,
        freeze_epochs: int     = 5,
        drop_rate: float       = 0.1,
        attn_drop: float       = 0.05,
    ):
        super().__init__()
        self.num_classes      = num_classes
        self.img_size         = img_size
        self.use_spectral_idx = use_spectral_idx
        self.use_temporal     = use_temporal
        self.freeze_epochs    = freeze_epochs
        self._current_epoch   = 0

        s2_in = self.S2_CHANNELS + (self.IDX_CHANNELS if use_spectral_idx else 0)

        # 1. Modality projectors (bring to common spatial feature dim)
        self.s1_projector = nn.Sequential(
            nn.Conv2d(self.S1_CHANNELS, s1_proj_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(s1_proj_dim), nn.GELU(),
            nn.Conv2d(s1_proj_dim, s1_proj_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(s1_proj_dim), nn.GELU(),
        )
        self.s2_projector = nn.Sequential(
            nn.Conv2d(s2_in, s2_proj_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(s2_proj_dim), nn.GELU(),
            nn.Conv2d(s2_proj_dim, s2_proj_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(s2_proj_dim), nn.GELU(),
        )

        # 2. Cross-modal fusion
        self.fusion = build_fusion(
            strategy=fusion_strategy,
            s1_dim=s1_proj_dim,
            s2_dim=s2_proj_dim,
            fused_dim=fused_dim,
        )

        # fused_dim channels → pretend they are sensor channels for ViT
        in_chans_vit = fused_dim // 32  # compress to plausible channel count
        self.pre_vit_proj = nn.Conv2d(fused_dim, in_chans_vit, 1)

        # 3. DOFA ViT backbone
        self.backbone = DOFAViTBackbone(
            img_size=img_size,
            in_chans=in_chans_vit,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop,
        )
        if pretrained_ckpt:
            self.backbone.load_pretrained(pretrained_ckpt)

        # 4. Optional temporal attention
        if use_temporal:
            self.temporal_attn = SpatiotemporalAttention(embed_dim)

        # 5. UperNet decoder
        feat_channels = [embed_dim] * 4   # all ViT intermediate features same dim
        self.decoder = UperNetDecoder(
            in_channels=feat_channels,
            num_classes=num_classes,
            fpn_out_ch=256,
            head_ch=512,
            use_aux_head=True,
        )

        self._freeze_backbone()

    # ------------------------------------------------------------------
    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def on_epoch_start(self, epoch: int):
        self._current_epoch = epoch
        if epoch == self.freeze_epochs:
            self._unfreeze_backbone()

    # ------------------------------------------------------------------
    def forward(
        self,
        s1: torch.Tensor,                         # (B, 2, H, W)
        s2: torch.Tensor,                         # (B, 10+5, H, W)
        wavelengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with:
          "logits"       : (B, num_classes, H, W)  – segmentation predictions
          "aux_logits"   : (B, num_classes, H', W') – auxiliary head (train only)
          "uncertainty"  : (B, 1, H, W)            – softmax entropy map
        """
        B, _, H, W = s1.shape

        # Project each modality
        f1 = self.s1_projector(s1)   # (B, s1_proj_dim, H, W)
        f2 = self.s2_projector(s2)   # (B, s2_proj_dim, H, W)

        # Cross-modal fusion
        fused = self.fusion(f1, f2)                  # (B, fused_dim, H, W)
        vit_in = self.pre_vit_proj(fused)             # (B, in_chans_vit, H, W)

        # ViT backbone → multi-scale features
        features = self.backbone(vit_in, wavelengths)  # list of 4 × (B, D, h, w)

        # UperNet decoder → logits
        logits, aux_logits = self.decoder(features, img_size=H)

        # Pixel-wise uncertainty = softmax entropy
        probs = logits.softmax(dim=1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1, keepdim=True)

        out = {"logits": logits, "uncertainty": entropy}
        if aux_logits is not None:
            out["aux_logits"] = aux_logits
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        tta: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference wrapper with optional test-time augmentation (TTA).
        TTA applies 4 rotations + horizontal flip → 8 augmentations,
        averages softmax probabilities.
        """
        self.eval()
        if not tta:
            return self.forward(s1, s2)

        aug_probs: List[torch.Tensor] = []
        for k in range(4):                         # 4 rotations
            for flip in [False, True]:
                s1_aug = torch.rot90(s1, k, [2, 3])
                s2_aug = torch.rot90(s2, k, [2, 3])
                if flip:
                    s1_aug = s1_aug.flip(-1)
                    s2_aug = s2_aug.flip(-1)
                out = self.forward(s1_aug, s2_aug)
                probs = out["logits"].softmax(dim=1)

                # Reverse augmentation on probabilities
                if flip:
                    probs = probs.flip(-1)
                probs = torch.rot90(probs, -k, [2, 3])
                aug_probs.append(probs)

        mean_probs = torch.stack(aug_probs).mean(0)
        logits     = mean_probs.log()
        entropy    = -(mean_probs * (mean_probs + 1e-8).log()).sum(1, keepdim=True)
        return {"logits": logits, "uncertainty": entropy}

    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        def count(mod):
            return sum(p.numel() for p in mod.parameters() if p.requires_grad)
        return {
            "s1_projector":    count(self.s1_projector),
            "s2_projector":    count(self.s2_projector),
            "fusion":          count(self.fusion),
            "backbone":        count(self.backbone),
            "decoder":         count(self.decoder),
            "total_trainable": count(self),
        }
