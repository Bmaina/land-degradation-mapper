"""
upernet_decoder.py
==================
UperNet segmentation head adapted for ViT backbone feature pyramids.

UperNet: "Unified Perceptual Parsing for Scene Understanding"
         Xiao et al. (ECCV 2018) – adapted for ViT multi-scale features.

Architecture:
  - FPN neck: lateral + top-down path over 4 ViT intermediate outputs
  - PPM head: pooling pyramid module for global context
  - Segmentation head: 1×1 conv → bilinear upsample → num_classes
  - Auxiliary head (optional): intermediate supervision at block 9 output
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building Blocks ──────────────────────────────────────────────────────────

def conv_bn_relu(in_c: int, out_c: int, k: int = 3, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


# ── Pyramid Pooling Module ────────────────────────────────────────────────────

class PPM(nn.Module):
    """
    Pyramid Pooling Module from PSPNet.
    Aggregates multi-scale context via adaptive average pooling
    at pool sizes [1, 2, 3, 6].
    """

    POOL_SIZES = [1, 2, 3, 6]

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for s in self.POOL_SIZES
        ])
        self.bottleneck = conv_bn_relu(
            in_channels + len(self.POOL_SIZES) * out_channels, out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w    = x.shape[2], x.shape[3]
        pooled  = [F.interpolate(stage(x), (h, w), mode="bilinear", align_corners=False)
                   for stage in self.stages]
        return self.bottleneck(torch.cat([x] + pooled, dim=1))


# ── FPN Neck ─────────────────────────────────────────────────────────────────

class FPNNeck(nn.Module):
    """
    Feature Pyramid Network neck.
    Takes 4 multi-scale feature maps from the ViT backbone and fuses them
    into a single representative feature map via top-down pathway.
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        # Lateral 1×1 projections
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        # 3×3 smooth after merging
        self.smooth = nn.ModuleList([
            conv_bn_relu(out_channels, out_channels) for _ in in_channels
        ])
        self.out_channels = out_channels

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        features : list of (B, C_i, H_i, W_i) from coarse to fine
                   (ViT outputs are all same spatial resolution,
                   but having the FPN lets us extend to hierarchical backbones)

        Returns
        -------
        fpn_outs : list of (B, out_channels, H_i, W_i) – same length
        """
        laterals = [l(f) for l, f in zip(self.lateral, features)]

        # Top-down fusion (fine → coarse)
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2], laterals[i - 1].shape[3]
            upsampled = F.interpolate(laterals[i], (h, w), mode="nearest")
            laterals[i - 1] = laterals[i - 1] + upsampled

        return [s(l) for s, l in zip(self.smooth, laterals)]


# ── UperNet Decoder ───────────────────────────────────────────────────────────

class UperNetDecoder(nn.Module):
    """
    UperNet segmentation decoder.

    Accepts 4 multi-scale ViT feature maps; produces per-pixel class logits.

    Parameters
    ----------
    in_channels    : list of channel counts for each feature level
    num_classes    : number of output semantic classes
    fpn_out_ch     : intermediate FPN channels
    head_ch        : segmentation head channels
    use_aux_head   : whether to add auxiliary classification head
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        fpn_out_ch: int  = 256,
        head_ch: int     = 512,
        use_aux_head: bool = True,
    ):
        super().__init__()
        self.fpn       = FPNNeck(in_channels, fpn_out_ch)
        self.ppm       = PPM(in_channels[-1], fpn_out_ch)
        self.num_levels = len(in_channels)
        self.use_aux   = use_aux_head

        # Merge all FPN levels + PPM output
        total_in = fpn_out_ch * (self.num_levels + 1)
        self.fuse = conv_bn_relu(total_in, head_ch)
        self.dropout = nn.Dropout2d(0.1)
        self.seg_head = nn.Conv2d(head_ch, num_classes, 1)

        # Auxiliary head on second-to-last feature (index 2)
        if use_aux_head:
            self.aux_head = nn.Sequential(
                conv_bn_relu(in_channels[2], fpn_out_ch),
                nn.Dropout2d(0.1),
                nn.Conv2d(fpn_out_ch, num_classes, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        features: List[torch.Tensor],
        img_size: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        features : 4 × (B, D, H', W')  from DOFAViT
        img_size : original image resolution for final upsample

        Returns
        -------
        logits     : (B, num_classes, H, W)
        aux_logits : (B, num_classes, H', W') if use_aux_head else None
        """
        # PPM on deepest (most semantic) feature
        ppm_out = self.ppm(features[-1])     # (B, F, H', W')

        # FPN over all features
        fpn_outs = self.fpn(features)         # list of (B, F, H', W')

        # Upsample all to finest resolution
        target_h, target_w = fpn_outs[0].shape[2], fpn_outs[0].shape[3]
        merged = [
            F.interpolate(f, (target_h, target_w), mode="bilinear", align_corners=False)
            for f in fpn_outs + [ppm_out]
        ]

        fused  = self.fuse(torch.cat(merged, dim=1))     # (B, head_ch, H', W')
        fused  = self.dropout(fused)
        logits = self.seg_head(fused)                    # (B, C, H', W')

        # Upsample to original image resolution
        if img_size is not None:
            logits = F.interpolate(logits, (img_size, img_size),
                                   mode="bilinear", align_corners=False)

        aux_logits = None
        if self.use_aux and self.training:
            aux_logits = self.aux_head(features[2])
            if img_size is not None:
                aux_logits = F.interpolate(aux_logits, (img_size, img_size),
                                           mode="bilinear", align_corners=False)

        return logits, aux_logits
