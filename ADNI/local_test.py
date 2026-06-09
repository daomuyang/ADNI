#!/usr/bin/env python3
"""
Small-scale local pipeline test before running on cloud server.

Steps:
  1. Preprocess N subjects (--preprocess-limit)
  2. 2-fold CV with few epochs (--epochs)
  3. Predict + eval on processed subjects
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 72}\n{desc}\n$ {' '.join(cmd)}\n{'=' * 72}")
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ADNI local small-scale pipeline test")
    parser.add_argument("--preprocess-limit", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--freeze-epochs", type=int, default=4)
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_preprocess:
        run(
            [py, "preprocess.py", "--limit", str(args.preprocess_limit), "--overwrite", "--workers", "1", "--no-nifti"],
            f"Preprocess first {args.preprocess_limit} subjects (v9 pipeline)",
        )
        run(
            [py, "verify_preprocess.py", "--only-existing", "--current-version-only"],
            "Verify v9 preprocessed subjects",
        )

    run(
        [
            py, "train.py",
            "--cv-only",
            "--folds", str(args.folds),
            "--epochs", str(args.epochs),
            "--freeze-epochs", str(args.freeze_epochs),
        ],
        f"Train {args.folds}-fold CV, {args.epochs} epochs",
    )
    run([py, "predict.py", "--eval", "--output", "outputs/local_submission.csv"], "Predict + eval")

    summary_path = ROOT / "outputs" / "cv_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print("\nLOCAL TEST SUMMARY")
        print(f"  mean_acc: {summary.get('mean_acc', float('nan')):.3f}")
        print(f"  mean_balanced_acc: {summary.get('mean_balanced_acc', float('nan')):.3f}")
        print(f"  mean_f1_macro: {summary.get('mean_f1_macro', float('nan')):.3f}")

    print("\nLOCAL TEST PASSED — ready for full run on server:")
    print("  python preprocess.py --overwrite --workers 1 --no-nifti")
    print("  python train.py")
    print("  python predict.py --output outputs/submission.csv --eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
