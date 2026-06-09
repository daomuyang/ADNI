#!/usr/bin/env python3
"""Verify all preprocessed ADNI subjects share the same pipeline run."""

from __future__ import annotations

import json
import sys

import numpy as np

from config import (
    LABELS_CSV,
    PREPROCESSED_DIR,
    PREPROCESSED_META_NAME,
    PREPROCESSED_NPY_NAME,
    PREPROCESS_PIPELINE_VERSION,
    SFCN_NORMALIZATION_METHOD,
    TARGET_SHAPE,
)


def main() -> int:
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-existing", action="store_true",
        help="Only verify subjects that already have preprocessed .npy files",
    )
    parser.add_argument(
        "--current-version-only", action="store_true",
        help="With --only-existing, verify only subjects at current pipeline_version",
    )
    args = parser.parse_args()

    df = pd.read_csv(LABELS_CSV)
    eids = df["eid"].astype(str).tolist()
    if args.only_existing:
        existing = [
            eid for eid in eids
            if (PREPROCESSED_DIR / eid / PREPROCESSED_NPY_NAME).exists()
        ]
        if args.current_version_only:
            filtered = []
            for eid in existing:
                meta_path = PREPROCESSED_DIR / eid / PREPROCESSED_META_NAME
                if not meta_path.exists():
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("pipeline_version") == PREPROCESS_PIPELINE_VERSION:
                    filtered.append(eid)
            eids = filtered
            print(f"Checking {len(eids)} subjects at {PREPROCESS_PIPELINE_VERSION!r}")
        else:
            eids = existing
        if not eids:
            print("FAILED: no matching preprocessed subjects found")
            return 1
    errors = []
    versions = set()
    run_ids = set()

    for eid in eids:
        meta_path = PREPROCESSED_DIR / eid / PREPROCESSED_META_NAME
        npy_path = PREPROCESSED_DIR / eid / PREPROCESSED_NPY_NAME
        if not npy_path.exists():
            errors.append(f"{eid}: missing {PREPROCESSED_NPY_NAME}")
            continue
        if not meta_path.exists():
            errors.append(f"{eid}: missing {PREPROCESSED_META_NAME}")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        pv = meta.get("pipeline_version")
        rid = meta.get("run_id")
        versions.add(pv)
        run_ids.add(rid)
        if pv != PREPROCESS_PIPELINE_VERSION:
            errors.append(f"{eid}: pipeline_version={pv!r} != {PREPROCESS_PIPELINE_VERSION!r}")
        if meta.get("normalization") != SFCN_NORMALIZATION_METHOD:
            errors.append(f"{eid}: normalization mismatch")
        vol = np.load(npy_path)
        if tuple(vol.shape) != TARGET_SHAPE:
            errors.append(f"{eid}: shape={vol.shape} != {TARGET_SHAPE}")
        brain = vol[np.abs(vol) > 1e-6]
        if float(np.abs(vol).max()) < 1e-4 or brain.size < 1000:
            errors.append(f"{eid}: volume near-empty")

    if len(versions) > 1:
        errors.append(f"multiple pipeline_version values: {versions}")

    summary_path = PREPROCESSED_DIR / "preprocess_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        if summary.get("pipeline_version") != PREPROCESS_PIPELINE_VERSION:
            errors.append("preprocess_summary.json pipeline_version mismatch")
    else:
        errors.append("missing preprocess_summary.json")

    if errors:
        print("FAILED:")
        for e in errors:
            print(" ", e)
        return 1

    rid_msg = run_ids.pop() if len(run_ids) == 1 else f"{len(run_ids)} runs"
    print(f"OK: {len(eids)} subjects, pipeline_version={PREPROCESS_PIPELINE_VERSION!r}, run_id={rid_msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
