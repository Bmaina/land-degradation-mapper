"""
dataset.py
==========
PyTorch Dataset for loading preprocessed patch files.
Expects patches saved as .npz files with keys: s1, s2, label
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob

class LandDegDataset(Dataset):
    def __init__(self, patch_dir, split="train", augment=False, use_spectral_idx=True):
        self.patch_dir = Path(patch_dir)
        self.split = split
        self.augment = augment
        self.use_spectral_idx = use_spectral_idx
        self.files = sorted(glob.glob(str(self.patch_dir / split / "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        s1 = torch.from_numpy(data["s1"]).float()
        s2_bands = torch.from_numpy(data["s2_bands"]).float()
        s2_idx   = torch.from_numpy(data["s2_indices"]).float()
        s2 = torch.cat([s2_bands, s2_idx], dim=0) if self.use_spectral_idx else s2_bands
        label = torch.from_numpy(data["label"]).long()
        if self.augment:
            if torch.rand(1) > 0.5:
                s1, s2, label = s1.flip(-1), s2.flip(-1), label.flip(-1)
            k = torch.randint(0, 4, (1,)).item()
            s1  = torch.rot90(s1,  k, [1, 2])
            s2  = torch.rot90(s2,  k, [1, 2])
            label = torch.rot90(label.unsqueeze(0), k, [1, 2]).squeeze(0)
        return {"s1": s1, "s2": s2, "label": label}
