"""Dataset loader for ADNI 3-class SFCN inputs [B,1,160,192,160]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    CLASS_NAMES,
    LABELS_CSV,
    LABEL_TO_IDX,
    PREPROCESSED_DIR,
    PREPROCESSED_META_NAME,
    PREPROCESSED_NPY_NAME,
    PREPROCESS_PIPELINE_VERSION,
    TARGET_SHAPE,
)


def official_augment(vol: np.ndarray) -> np.ndarray:
    """Peng et al. 2021: random shift 0-2 voxels + 50% sagittal flip."""
    out = vol.copy()
    shifts = np.random.randint(0, 3, size=3)
    out = np.roll(out, shifts, axis=(0, 1, 2))
    if np.random.rand() < 0.5:
        out = out[::-1, :, :]
    return out.astype(np.float32)


class ADNIDataset(Dataset):
    def __init__(
        self,
        eids: list[str],
        labels_df: pd.DataFrame,
        preprocessed_dir: Path = PREPROCESSED_DIR,
        training: bool = False,
    ):
        self.eids = eids
        self.labels = labels_df.set_index(labels_df["eid"].astype(str))
        self.preprocessed_dir = preprocessed_dir
        self.training = training

    def __len__(self) -> int:
        return len(self.eids)

    def __getitem__(self, idx: int) -> dict:
        eid = self.eids[idx]
        vol = np.load(self.preprocessed_dir / eid / PREPROCESSED_NPY_NAME).astype(np.float32)
        if vol.shape != TARGET_SHAPE:
            raise ValueError(f"{eid}: shape {vol.shape} != {TARGET_SHAPE}")
        if self.training:
            vol = official_augment(vol)
        x = torch.from_numpy(vol[None, ...])

        row = self.labels.loc[str(eid)]
        label = str(row["label"]).strip().upper()
        if label not in LABEL_TO_IDX:
            raise ValueError(f"{eid}: unknown label {label!r}, expected {CLASS_NAMES}")
        y = torch.tensor(LABEL_TO_IDX[label], dtype=torch.long)

        return {"eid": eid, "x": x, "y": y, "label": label}


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    df["eid"] = df["eid"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    return df


def available_eids(
    preprocessed_dir: Path = PREPROCESSED_DIR,
    require_current_pipeline: bool = True,
) -> list[str]:
    import json

    df = load_labels()
    eids = []
    for eid in df["eid"]:
        eid = str(eid)
        npy_path = preprocessed_dir / eid / PREPROCESSED_NPY_NAME
        if not npy_path.exists():
            continue
        if require_current_pipeline:
            meta_path = preprocessed_dir / eid / PREPROCESSED_META_NAME
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("pipeline_version") != PREPROCESS_PIPELINE_VERSION:
                continue
        eids.append(eid)
    return sorted(eids)
