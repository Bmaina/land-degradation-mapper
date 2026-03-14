"""
losses.py  &  metrics.py
========================
Custom loss functions and evaluation metrics for semantic segmentation
of land degradation classes.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., 2017).
    Downweights easy negatives to focus training on hard, mis-classified pixels.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = 255,
    ):
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha
        self.reduction    = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)
        targets : (B, H, W) long
        """
        C = logits.size(1)
        log_p = F.log_softmax(logits, dim=1)  # (B, C, H, W)
        p     = log_p.exp()

        # One-hot targets: (B, H, W) → (B, C, H, W)
        targets_oh = F.one_hot(
            targets.clamp(0, C - 1), C
        ).permute(0, 3, 1, 2).float()

        # p_t: probability of true class
        p_t = (p * targets_oh).sum(dim=1)     # (B, H, W)

        focal_weight = (1 - p_t) ** self.gamma
        ce = -(log_p * targets_oh).sum(dim=1)  # per-pixel CE

        loss = focal_weight * ce

        # Class weights
        if self.alpha is not None:
            alpha_t = (self.alpha.to(logits.device) * targets_oh).sum(dim=1)
            loss = alpha_t * loss

        # Ignore index mask
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            loss = loss[mask]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════════════════════════════
#  Dice Loss
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """Soft multi-class Dice loss."""

    def __init__(
        self,
        smooth: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.smooth       = smooth
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.size(1)
        probs = logits.softmax(dim=1)  # (B, C, H, W)

        # Mask ignore pixels
        mask = (targets != self.ignore_index)
        targets_masked = targets.clone()
        targets_masked[~mask] = 0

        targets_oh = F.one_hot(targets_masked, C).permute(0, 3, 1, 2).float()
        targets_oh[:, :, ~mask] = 0  # zero out ignored positions

        dice_per_class = []
        for c in range(C):
            p = probs[:, c].flatten()
            t = targets_oh[:, c].flatten()
            intersection = (p * t).sum()
            dice = (2 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_per_class.append(1 - dice)

        dice_tensor = torch.stack(dice_per_class)
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            return (dice_tensor * w).sum() / w.sum()
        return dice_tensor.mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Combined Focal + Dice Loss
# ═══════════════════════════════════════════════════════════════════════════════

class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice loss.

    L = (1 - dice_weight) * L_focal  +  dice_weight * L_dice

    Focal focuses on hard pixels; Dice ensures class balance.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.5,
        ignore_index: int  = 255,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            ignore_index=ignore_index,
        )
        self.dice = DiceLoss(
            class_weights=class_weights,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            (1 - self.dice_weight) * self.focal(logits, targets)
            + self.dice_weight    * self.dice(logits, targets)
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Segmentation Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class SegmentationMetrics:
    """
    Computes:
      • Per-class IoU (Intersection over Union)
      • Mean IoU (mIoU)
      • Overall Accuracy (OA)
      • Per-class F1 (Dice)
      • Macro-F1

    Uses confusion matrix accumulation for efficiency.
    """

    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device      = device
        self.conf_matrix = torch.zeros(num_classes, num_classes,
                                       dtype=torch.long, device=device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255):
        """preds, targets: (B, H, W) long."""
        mask  = targets != ignore_index
        preds   = preds[mask].long()
        targets = targets[mask].long()

        idx = targets * self.num_classes + preds
        self.conf_matrix += torch.bincount(
            idx, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        cm   = self.conf_matrix.float()
        tp   = cm.diag()
        fp   = cm.sum(0) - tp
        fn   = cm.sum(1) - tp

        iou     = tp / (tp + fp + fn + 1e-8)
        f1      = 2 * tp / (2 * tp + fp + fn + 1e-8)
        oa      = tp.sum() / (cm.sum() + 1e-8)

        per_cls: Dict[str, float] = {}
        for c in range(self.num_classes):
            per_cls[f"iou_cls{c}"] = iou[c].item()
            per_cls[f"f1_cls{c}"]  = f1[c].item()

        return {
            "mean_iou":         iou.mean().item(),
            "overall_accuracy": oa.item(),
            "f1_macro":         f1.mean().item(),
            **per_cls,
        }

    def reset(self):
        self.conf_matrix.zero_()
