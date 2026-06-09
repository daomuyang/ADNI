#!/usr/bin/env python3
"""Lightweight integration checks for ADNI pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

REQUIRED = [
    "TARGET_SHAPE", "PREPROCESSED_NPY_NAME", "PREPROCESS_PIPELINE_VERSION",
    "CLASS_NAMES", "NUM_CLASSES", "PRETRAINED_PATH", "SFCN_CHANNELS",
]


def check_config() -> None:
    import config
    missing = [n for n in REQUIRED if not hasattr(config, n)]
    if missing:
        raise RuntimeError(f"config.py missing: {missing}")
    print("OK config")


def check_verify() -> None:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "verify_preprocess.py")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"verify failed:\n{proc.stdout}\n{proc.stderr}")
    print(proc.stdout.strip())


def check_dataset() -> None:
    from config import TARGET_SHAPE
    from dataset import ADNIDataset, available_eids, load_labels

    eids = available_eids()
    if not eids:
        raise RuntimeError("no preprocessed subjects")
    ds = ADNIDataset(eids[:3], load_labels(), training=False)
    b = ds[0]
    assert b["x"].shape == (1, *TARGET_SHAPE)
    assert b["y"].dtype == torch.long
    print(f"OK dataset ({len(eids)} subjects)")


def check_model() -> None:
    from config import NUM_CLASSES, SFCN_CHANNELS
    from dp_model.model_files.sfcn import SFCN
    from train import flatten_logprob

    x = torch.randn(2, 1, 160, 192, 160)
    out = flatten_logprob(SFCN(channel_number=SFCN_CHANNELS, output_dim=NUM_CLASSES)(x))
    assert out.shape == (2, NUM_CLASSES)
    print("OK SFCN forward")


def main() -> int:
    for fn in [check_config, check_verify, check_dataset, check_model]:
        fn()
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
