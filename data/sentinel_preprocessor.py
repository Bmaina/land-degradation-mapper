"""
sentinel_preprocessor.py
========================
Preprocessing pipeline for Sentinel-1 SAR and Sentinel-2 multispectral imagery.

Sentinel-1  → terrain-corrected γ₀ backscatter (dB), speckle-filtered
Sentinel-2  → surface reflectance (L2A), cloud-masked, normalised indices
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance

logger = logging.getLogger(__name__)

# ── Sentinel-2 Band Indices ──────────────────────────────────
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_BAND_IDX = {b: i for i, b in enumerate(S2_BANDS)}

# ── Normalisation statistics (ImageNet-style, computed over Africa) ──────────
S2_MEAN = np.array([0.0897, 0.1216, 0.1200, 0.1558, 0.2152,
                    0.2428, 0.2640, 0.2740, 0.2160, 0.1580], dtype=np.float32)
S2_STD  = np.array([0.0412, 0.0420, 0.0600, 0.0540, 0.0630,
                    0.0680, 0.0750, 0.0720, 0.0780, 0.0620], dtype=np.float32)
S1_MEAN = np.array([-12.5, -19.8], dtype=np.float32)   # VV, VH (dB)
S1_STD  = np.array([  4.2,   4.5], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
class Sentinel1Preprocessor:
    """
    Sentinel-1 GRD preprocessing:
      1. Convert linear backscatter → dB
      2. Lee speckle filter
      3. Z-score normalisation
    """

    def __init__(self, target_resolution: int = 10):
        self.target_resolution = target_resolution

    # ------------------------------------------------------------------
    def linear_to_db(self, arr: np.ndarray) -> np.ndarray:
        """σ⁰ linear → dB (clip to avoid log(0))."""
        arr = np.clip(arr, 1e-6, None)
        return 10.0 * np.log10(arr).astype(np.float32)

    # ------------------------------------------------------------------
    def lee_filter(self, img: np.ndarray, window: int = 7) -> np.ndarray:
        """
        Simplified Lee speckle filter per channel.
        Equivalent of ESA SNAP's Lee filter.
        """
        filtered = np.empty_like(img)
        for c in range(img.shape[0]):
            band = img[c].astype(np.float64)
            band_mean = uniform_filter(band, window)
            band_sq   = uniform_filter(band ** 2, window)
            band_var  = band_sq - band_mean ** 2

            overall_var = float(np.var(band))
            noise_var   = overall_var / (1 + overall_var)  # approx

            weight  = band_var / (band_var + noise_var + 1e-10)
            filtered[c] = (band_mean + weight * (band - band_mean)).astype(np.float32)
        return filtered

    # ------------------------------------------------------------------
    def normalise(self, arr: np.ndarray) -> np.ndarray:
        for c in range(arr.shape[0]):
            arr[c] = (arr[c] - S1_MEAN[c]) / (S1_STD[c] + 1e-8)
        return arr

    # ------------------------------------------------------------------
    def process(self, vv_path: Path, vh_path: Path) -> np.ndarray:
        """Return (2, H, W) float32 array: [VV_norm, VH_norm]."""
        with rasterio.open(vv_path) as src:
            vv = src.read(1).astype(np.float32)
        with rasterio.open(vh_path) as src:
            vh = src.read(1).astype(np.float32)

        stack = np.stack([vv, vh], axis=0)          # (2, H, W) linear
        stack = self.linear_to_db(stack)             # → dB
        stack = self.lee_filter(stack)               # despeckle
        stack = self.normalise(stack)                # Z-normalise
        logger.debug("S1 processed: shape=%s  min=%.2f  max=%.2f",
                     stack.shape, stack.min(), stack.max())
        return stack


# ═══════════════════════════════════════════════════════════════════════════════
class Sentinel2Preprocessor:
    """
    Sentinel-2 L2A preprocessing:
      1. Cloud masking via SCL (Scene Classification Layer)
      2. Bilinear resample all bands to 10 m
      3. Compute spectral indices (NDVI, NDWI, BSI, NDRE, EVI)
      4. Z-score normalisation
    """

    # SCL classes to mask (cloud, shadow, etc.)
    SCL_MASK_CLASSES = {1, 3, 8, 9, 10, 11}   # saturated, cloud shadow, cloud, cirrus, snow

    def __init__(self, target_resolution: int = 10):
        self.target_resolution = target_resolution

    # ------------------------------------------------------------------
    def read_band(self, path: Path, target_shape: Tuple[int, int]) -> np.ndarray:
        with rasterio.open(path) as src:
            data = src.read(
                1,
                out_shape=target_shape,
                resampling=Resampling.bilinear,
            ).astype(np.float32) / 10_000.0   # DN → reflectance [0–1]
        return data

    # ------------------------------------------------------------------
    def cloud_mask(self, scl: np.ndarray) -> np.ndarray:
        """Return boolean mask: True = valid pixel."""
        valid = np.ones(scl.shape, dtype=bool)
        for cls in self.SCL_MASK_CLASSES:
            valid &= (scl != cls)
        return valid

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            idx = np.where((a + b) == 0, 0.0, (a - b) / (a + b))
        return idx.astype(np.float32)

    def compute_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute key spectral indices for vegetation/degradation monitoring."""
        nir, red   = bands["B8"],  bands["B4"]
        swir1, grn = bands["B11"], bands["B3"]
        re1        = bands["B5"]   # Red-Edge 1

        indices = {
            "NDVI": self._safe_index(nir, red),                # veg greenness
            "NDWI": self._safe_index(grn, nir),                # water/moisture
            "BSI":  self._safe_index(swir1 + red, nir + grn),  # bare soil
            "NDRE": self._safe_index(nir, re1),                 # red-edge veg
            "EVI":  2.5 * (nir - red) / (nir + 6*red - 7.5*bands["B2"] + 1 + 1e-8),
        }
        return indices

    # ------------------------------------------------------------------
    def normalise(self, arr: np.ndarray) -> np.ndarray:
        for c in range(arr.shape[0]):
            arr[c] = (arr[c] - S2_MEAN[c]) / (S2_STD[c] + 1e-8)
        return arr

    # ------------------------------------------------------------------
    def process(
        self,
        band_paths: Dict[str, Path],
        scl_path: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        bands_norm  : (10, H, W) normalised surface reflectance
        indices_arr : (5,  H, W) spectral indices [NDVI, NDWI, BSI, NDRE, EVI]
        """
        # Use B4 as reference for target shape
        with rasterio.open(band_paths["B4"]) as ref:
            h, w = ref.height, ref.width

        bands: Dict[str, np.ndarray] = {}
        for b in S2_BANDS:
            bands[b] = self.read_band(band_paths[b], (h, w))

        # Cloud mask (set invalid to NaN, interpolated later in patching)
        mask = np.ones((h, w), dtype=bool)
        if scl_path is not None:
            with rasterio.open(scl_path) as src:
                scl = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
            mask = self.cloud_mask(scl)
        cloud_frac = 1.0 - mask.mean()
        logger.info("Cloud fraction: %.1f%%", cloud_frac * 100)

        band_arr = np.stack([bands[b] for b in S2_BANDS], axis=0)
        band_arr[:, ~mask] = np.nan

        indices = self.compute_indices(bands)
        idx_arr = np.stack(list(indices.values()), axis=0)
        idx_arr[:, ~mask] = np.nan

        band_arr_norm = self.normalise(band_arr)

        logger.debug("S2 processed: shape=%s  cloud=%.1f%%",
                     band_arr_norm.shape, cloud_frac * 100)
        return band_arr_norm, idx_arr


# ═══════════════════════════════════════════════════════════════════════════════
class PatchExtractor:
    """
    Extract fixed-size patches with optional overlap for training
    and sliding-window for inference.
    """

    def __init__(
        self,
        patch_size: int = 224,
        stride: int = 112,
        min_valid_frac: float = 0.8,
    ):
        self.patch_size = patch_size
        self.stride     = stride
        self.min_valid  = min_valid_frac

    # ------------------------------------------------------------------
    def extract(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        label: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Parameters
        ----------
        s1    : (2, H, W)
        s2    : (15, H, W)  10 bands + 5 indices
        label : (H, W) int8, optional

        Returns
        -------
        List of patch dicts with keys: s1, s2, label, row, col
        """
        _, H, W = s1.shape
        P, S    = self.patch_size, self.stride
        patches = []

        for r in range(0, H - P + 1, S):
            for c in range(0, W - P + 1, S):
                s2_patch = s2[:, r:r+P, c:c+P]

                # Skip patches dominated by NaN (cloud)
                valid_frac = np.isfinite(s2_patch).mean()
                if valid_frac < self.min_valid:
                    continue

                # Replace remaining NaN with channel median
                s2_patch = np.where(
                    np.isfinite(s2_patch), s2_patch,
                    np.nanmedian(s2_patch, axis=(1, 2), keepdims=True)
                )

                patch = {
                    "s1":  s1[:, r:r+P, c:c+P].copy(),
                    "s2":  s2_patch.copy(),
                    "row": r,
                    "col": c,
                }
                if label is not None:
                    patch["label"] = label[r:r+P, c:c+P].copy()
                patches.append(patch)

        logger.debug("Extracted %d valid patches from (%d, %d) image", len(patches), H, W)
        return patches
