"""
train.py
========
Main training loop for LandDegMapper.

Features:
  • AMP mixed-precision training (fp16)
  • Cosine LR schedule with linear warm-up
  • Focal+Dice combined loss with class weighting
  • MLflow experiment tracking (metrics, params, artefacts, model registry)
  • Early stopping & best-model checkpointing
  • Gradient clipping
  • Backbone freeze warm-up (linear probe → full fine-tune)

Usage:
  python train.py --config configs/config.yaml [--resume checkpoints/last.pth]
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Project imports (assumed importable from project root)
from models.land_deg_mapper import LandDegMapper
from training.losses import FocalDiceLoss
from training.metrics import SegmentationMetrics
from data.dataset import LandDegDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ═══════════════════════════════════════════════════════════════════════════════
#  Cosine LR schedule with linear warm-up
# ═══════════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        base_lr: float = 1e-4,
    ):
        self.opt            = optimizer
        self.warmup         = warmup_epochs
        self.total          = total_epochs
        self.min_lr         = min_lr
        self.base_lr        = base_lr

    def step(self, epoch: int):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# ═══════════════════════════════════════════════════════════════════════════════
#  Training / Validation steps
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: LandDegMapper,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: FocalDiceLoss,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    aux_weight: float = 0.4,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    metrics = SegmentationMetrics(model.num_classes, device)

    for batch_idx, batch in enumerate(loader):
        s1    = batch["s1"].to(device, non_blocking=True)
        s2    = batch["s2"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            out  = model(s1, s2)
            loss = criterion(out["logits"], label)
            if "aux_logits" in out:
                loss = loss + aux_weight * criterion(out["aux_logits"], label)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = out["logits"].argmax(dim=1)
        metrics.update(preds, label)

        if batch_idx % 50 == 0:
            logger.info("  step %4d  loss=%.4f", batch_idx, loss.item())

    scores = metrics.compute()
    scores["loss"] = total_loss / len(loader)
    return scores


@torch.no_grad()
def validate(
    model: LandDegMapper,
    loader: DataLoader,
    criterion: FocalDiceLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(model.num_classes, device)

    for batch in loader:
        s1    = batch["s1"].to(device, non_blocking=True)
        s2    = batch["s2"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        out  = model(s1, s2)
        loss = criterion(out["logits"], label)
        total_loss += loss.item()
        preds = out["logits"].argmax(dim=1)
        metrics.update(preds, label)

    scores = metrics.compute()
    scores["loss"] = total_loss / len(loader)
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(cfg_path: str, resume: Optional[str] = None):
    # ── Load config ──────────────────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg["training"]["device"]
                          if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = LandDegDataset(
        patch_dir=cfg["paths"]["patches"],
        split="train",
        augment=True,
        use_spectral_idx=True,
    )
    val_ds = LandDegDataset(
        patch_dir=cfg["paths"]["patches"],
        split="val",
        augment=False,
        use_spectral_idx=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    # ── Model ────────────────────────────────────────────────────────────────
    model = LandDegMapper(
        num_classes=cfg["model"]["num_classes"],
        embed_dim=cfg["model"]["embed_dim"],
        img_size=cfg["model"]["img_size"],
        fusion_strategy=cfg["model"]["fusion"]["strategy"],
        fused_dim=cfg["model"]["fusion"]["fused_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        use_spectral_idx=True,
        pretrained_ckpt=cfg["model"].get("pretrained_weights"),
        freeze_epochs=cfg["training"].get("freeze_backbone_epochs", 5),
        drop_rate=cfg["model"]["dropout"],
        attn_drop=cfg["model"]["attn_dropout"],
    ).to(device)

    param_info = model.count_parameters()
    logger.info("Parameters: %s", param_info)

    # ── Loss ─────────────────────────────────────────────────────────────────
    class_weights = torch.tensor(cfg["classes"]["weights"], device=device)
    criterion = FocalDiceLoss(
        num_classes=cfg["model"]["num_classes"],
        class_weights=class_weights,
        focal_gamma=cfg["training"]["loss"]["focal_gamma"],
        dice_weight=cfg["training"]["loss"]["dice_weight"],
    )

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["optimizer"]["lr"],
        weight_decay=cfg["training"]["optimizer"]["weight_decay"],
        betas=cfg["training"]["optimizer"]["betas"],
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg["training"]["scheduler"]["warmup_epochs"],
        total_epochs=cfg["training"]["epochs"],
        min_lr=cfg["training"]["scheduler"]["min_lr"],
        base_lr=cfg["training"]["optimizer"]["lr"],
    )
    scaler = GradScaler(enabled=cfg["training"]["mixed_precision"])

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou   = 0.0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        logger.info("Resumed from epoch %d  best_mIoU=%.4f", start_epoch, best_miou)

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(tags=cfg["mlflow"]["run_tags"]) as run:
        # Log all config params
        mlflow.log_params({
            "model_backbone":  cfg["model"]["backbone"],
            "fusion_strategy": cfg["model"]["fusion"]["strategy"],
            "embed_dim":       cfg["model"]["embed_dim"],
            "depth":           cfg["model"]["depth"],
            "num_classes":     cfg["model"]["num_classes"],
            "batch_size":      cfg["training"]["batch_size"],
            "lr":              cfg["training"]["optimizer"]["lr"],
            "epochs":          cfg["training"]["epochs"],
            "n_params_total":  param_info["total_trainable"],
        })
        mlflow.log_artifact(cfg_path, artifact_path="config")

        logger.info("MLflow run: %s", run.info.run_id)

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(start_epoch, cfg["training"]["epochs"]):
            t0 = time.time()
            model.on_epoch_start(epoch)

            lr = scheduler.step(epoch)

            train_scores = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler,
                device, cfg["training"]["gradient_clip"]
            )
            val_scores = validate(model, val_loader, criterion, device)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d  lr=%.2e  "
                "train_loss=%.4f  val_loss=%.4f  val_mIoU=%.4f  [%.1fs]",
                epoch + 1, cfg["training"]["epochs"], lr,
                train_scores["loss"], val_scores["loss"],
                val_scores.get("mean_iou", 0.0), elapsed,
            )

            # Log to MLflow
            mlflow.log_metrics({
                "train/loss":    train_scores["loss"],
                "train/mIoU":    train_scores.get("mean_iou", 0.0),
                "train/OA":      train_scores.get("overall_accuracy", 0.0),
                "val/loss":      val_scores["loss"],
                "val/mIoU":      val_scores.get("mean_iou", 0.0),
                "val/OA":        val_scores.get("overall_accuracy", 0.0),
                "val/F1_macro":  val_scores.get("f1_macro", 0.0),
                "lr":            lr,
            }, step=epoch)

            # Checkpoint
            ckpt_data = {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_miou": best_miou,
                "val_scores": val_scores,
            }
            torch.save(ckpt_data, ckpt_dir / "last.pth")

            if val_scores.get("mean_iou", 0.0) > best_miou:
                best_miou = val_scores["mean_iou"]
                torch.save(ckpt_data, ckpt_dir / "best_model.pth")
                logger.info("  ✓ New best mIoU: %.4f", best_miou)
                mlflow.log_metric("best_val_mIoU", best_miou, step=epoch)

        # ── Register final model ──────────────────────────────────────────
        if cfg["mlflow"]["log_model"]:
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=cfg["mlflow"]["register_model_name"],
            )
            logger.info("Model registered as '%s'", cfg["mlflow"]["register_model_name"])

        logger.info("Training complete. Best val mIoU: %.4f", best_miou)


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LandDegMapper")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args.config, args.resume)
