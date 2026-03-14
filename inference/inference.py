"""
inference.py
============
Sliding-window inference pipeline for large-scale land degradation mapping.

Produces:
  • Degradation risk map (GeoTIFF, 6-class)
  • Probability maps per class
  • Uncertainty map (softmax entropy)
  • Vector hotspot shapefile (high-degradation polygons)

Designed for processing full Sentinel tiles (100 km × 100 km, ~10,000 × 10,000 px).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import torch
import torch.nn.functional as F
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import shape

from models.land_deg_mapper import LandDegMapper
from data.sentinel_preprocessor import (
    Sentinel1Preprocessor,
    Sentinel2Preprocessor,
    PatchExtractor,
)

logger = logging.getLogger(__name__)


# ── Class metadata ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Stable Vegetation",
    "Moderate Stress",
    "Severe Stress",
    "Bare / Exposed Soil",
    "Active Degradation",
    "Recovery / Revegetation",
]

# GeoTIFF colour table (RGBA) – for QGIS / GEE visualisation
CLASS_COLORS = [
    (46, 139, 87,  255),   # Stable     – dark green
    (154, 205, 50, 255),   # Mod stress – yellow-green
    (255, 165, 0,  255),   # Sev stress – orange
    (210, 180, 140,255),   # Bare soil  – tan
    (139, 0,   0,  255),   # Active deg – dark red
    (0,  128, 255, 255),   # Recovery   – blue
]


# ═══════════════════════════════════════════════════════════════════════════════

class DegradationMapper:
    """
    End-to-end inference on a Sentinel tile pair (S1 + S2).

    Parameters
    ----------
    model_path  : str      path to best_model.pth checkpoint
    patch_size  : int      inference patch size (must match training)
    patch_stride: int      sliding-window stride (typically patch_size // 2)
    batch_size  : int      patches per forward pass
    tta         : bool     test-time augmentation
    device      : str      cuda | cpu | mps
    """

    def __init__(
        self,
        model_path: str,
        patch_size: int  = 224,
        patch_stride: int = 112,
        batch_size: int  = 32,
        tta: bool        = True,
        device: str      = "cuda",
    ):
        self.patch_size   = patch_size
        self.patch_stride = patch_stride
        self.batch_size   = batch_size
        self.tta          = tta
        self.device       = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        logger.info("Loading checkpoint: %s", model_path)
        ckpt  = torch.load(model_path, map_location=self.device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        self.model = LandDegMapper(num_classes=6)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        logger.info("Model loaded on %s", self.device)

        # Preprocessors
        self.s1_proc = Sentinel1Preprocessor()
        self.s2_proc = Sentinel2Preprocessor()
        self.extractor = PatchExtractor(patch_size, patch_stride, min_valid_frac=0.5)

    # ------------------------------------------------------------------
    def _stitch_predictions(
        self,
        patch_results: list,
        h: int, w: int,
        n_classes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted averaging of overlapping patch predictions.
        Centre pixels get higher weight (Gaussian weighting).
        """
        P = self.patch_size
        prob_accum  = np.zeros((n_classes, h, w), dtype=np.float32)
        weight_map  = np.zeros((h, w), dtype=np.float32)

        # Gaussian weight kernel (favours patch centre)
        yy, xx = np.mgrid[:P, :P]
        sig = P / 4
        kernel = np.exp(-((xx - P/2)**2 + (yy - P/2)**2) / (2 * sig**2))

        for patch in patch_results:
            r, c    = patch["row"], patch["col"]
            probs   = patch["probs"]          # (C, P, P)
            prob_accum[:, r:r+P, c:c+P] += probs * kernel[None]
            weight_map[r:r+P, c:c+P]   += kernel

        weight_map = np.maximum(weight_map, 1e-6)
        prob_map   = prob_accum / weight_map[None]          # (C, H, W)
        pred_map   = prob_map.argmax(axis=0).astype(np.uint8)  # (H, W)
        return pred_map, prob_map

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_patches(self, patches: list) -> list:
        """Batch-process patches through the model."""
        results = []
        for i in range(0, len(patches), self.batch_size):
            batch = patches[i : i + self.batch_size]
            s1_t  = torch.from_numpy(
                np.stack([p["s1"] for p in batch])
            ).to(self.device)
            s2_t  = torch.from_numpy(
                np.stack([p["s2"] for p in batch])
            ).to(self.device)

            out   = self.model.predict(s1_t, s2_t, tta=self.tta)
            probs = out["logits"].softmax(dim=1).cpu().numpy()  # (B, C, P, P)
            unc   = out["uncertainty"].squeeze(1).cpu().numpy()  # (B, P, P)

            for j, p in enumerate(batch):
                results.append({
                    "row":   p["row"],
                    "col":   p["col"],
                    "probs": probs[j],
                    "unc":   unc[j],
                })
        return results

    # ------------------------------------------------------------------
    def run(
        self,
        s1_vv_path: str,
        s1_vh_path: str,
        s2_band_paths: Dict[str, str],
        s2_scl_path: Optional[str],
        output_dir: str,
        tile_id: str = "tile",
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, str]:
        """
        Full inference on one Sentinel tile.

        Returns
        -------
        dict mapping output type → file path
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Preprocessing Sentinel-1 …")
        s1 = self.s1_proc.process(Path(s1_vv_path), Path(s1_vh_path))

        logger.info("Preprocessing Sentinel-2 …")
        s2_bands, s2_idx = self.s2_proc.process(
            {k: Path(v) for k, v in s2_band_paths.items()},
            scl_path=Path(s2_scl_path) if s2_scl_path else None,
        )
        s2_full = np.concatenate([s2_bands, s2_idx], axis=0)  # (15, H, W)

        # Resample S1 to S2 spatial extent if needed
        H, W = s2_full.shape[1], s2_full.shape[2]
        if s1.shape[1:] != (H, W):
            s1_t   = torch.from_numpy(s1).unsqueeze(0).float()
            s1_t   = F.interpolate(s1_t, (H, W), mode="bilinear", align_corners=False)
            s1     = s1_t.squeeze(0).numpy()

        logger.info("Extracting patches (H=%d W=%d) …", H, W)
        patches = self.extractor.extract(s1, s2_full)
        logger.info("Running inference on %d patches (TTA=%s) …", len(patches), self.tta)

        results = self._run_patches(patches)
        pred_map, prob_map = self._stitch_predictions(results, H, W, 6)

        # Uncertainty map from entropy
        unc_map = np.zeros((H, W), dtype=np.float32)
        for r in results:
            r0, c0 = r["row"], r["col"]
            P      = r["unc"].shape[0]
            unc_map[r0:r0+P, c0:c0+P] = np.maximum(
                unc_map[r0:r0+P, c0:c0+P], r["unc"]
            )

        # ── Save GeoTIFF ─────────────────────────────────────────────────
        with rasterio.open(list(s2_band_paths.values())[0]) as ref:
            transform = ref.transform
            crs       = ref.crs

        # Degradation class map
        deg_path = out_dir / f"{tile_id}_degradation.tif"
        with rasterio.open(
            deg_path, "w",
            driver="GTiff", height=H, width=W,
            count=1, dtype=np.uint8,
            crs=crs, transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(pred_map, 1)
            dst.write_colormap(1, {i: c for i, c in enumerate(CLASS_COLORS)})

        # Probability maps (float32, one band per class)
        prob_path = out_dir / f"{tile_id}_probabilities.tif"
        with rasterio.open(
            prob_path, "w",
            driver="GTiff", height=H, width=W,
            count=6, dtype=np.float32,
            crs=crs, transform=transform,
            compress="lzw",
        ) as dst:
            for c in range(6):
                dst.write(prob_map[c], c + 1)
                dst.set_band_description(c + 1, CLASS_NAMES[c])

        # Uncertainty map
        unc_path = out_dir / f"{tile_id}_uncertainty.tif"
        with rasterio.open(
            unc_path, "w",
            driver="GTiff", height=H, width=W,
            count=1, dtype=np.float32,
            crs=crs, transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(unc_map, 1)

        # ── Vectorise hotspots (Active Degradation class = 4) ─────────────
        hotspot_mask  = (pred_map == 4).astype(np.uint8)
        hotspot_polys = []
        for geom, val in shapes(hotspot_mask, transform=transform):
            if val == 1:
                hotspot_polys.append(shape(geom))

        vector_path = out_dir / f"{tile_id}_hotspots.gpkg"
        if hotspot_polys:
            gdf = gpd.GeoDataFrame(
                {"geometry": hotspot_polys,
                 "area_ha": [p.area / 1e4 for p in hotspot_polys]},
                crs=crs,
            )
            gdf = gdf[gdf.area_ha > 0.5]  # filter tiny polygons
            gdf.to_file(vector_path, driver="GPKG")
            logger.info("Hotspots: %d polygons  (%.0f ha total)",
                        len(gdf), gdf.area_ha.sum())

        outputs = {
            "degradation_map": str(deg_path),
            "probability_maps": str(prob_path),
            "uncertainty_map":  str(unc_path),
            "hotspot_vectors":  str(vector_path),
        }
        logger.info("Outputs written to: %s", out_dir)
        return outputs


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run land degradation inference")
    parser.add_argument("--model",   required=True, help="checkpoint path")
    parser.add_argument("--s1_vv",   required=True)
    parser.add_argument("--s1_vh",   required=True)
    parser.add_argument("--s2_dir",  required=True, help="dir with S2 band TIFs")
    parser.add_argument("--scl",     default=None,  help="SCL band path")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--tile_id", default="tile")
    parser.add_argument("--tta",     action="store_true")
    args = parser.parse_args()

    import glob, os
    band_paths = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(args.s2_dir, "*.tif"))
    }

    mapper = DegradationMapper(args.model, tta=args.tta)
    outputs = mapper.run(
        s1_vv_path=args.s1_vv,
        s1_vh_path=args.s1_vh,
        s2_band_paths=band_paths,
        s2_scl_path=args.scl,
        output_dir=args.out_dir,
        tile_id=args.tile_id,
    )
    print(json.dumps(outputs, indent=2))
