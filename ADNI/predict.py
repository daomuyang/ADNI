#!/usr/bin/env python3
"""Generate ADNI submission CSV: ID, Pre (CN/MCI/AD)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    IDX_TO_LABEL,
    MODELS_DIR,
    NUM_CLASSES,
    OUTPUTS_DIR,
    PRETRAINED_PATH,
    SFCN_CHANNELS,
    USE_TTA,
)
from dataset import ADNIDataset, available_eids, load_labels
from dp_model.model_files.sfcn import SFCN
from train import flatten_logprob, get_device, load_pretrained


def find_checkpoints() -> list[Path]:
    paths: list[Path] = []
    root = MODELS_DIR / "diagnosis"
    cv = root / "cv"
    if cv.exists():
        for fold in sorted(cv.glob("fold_*")):
            ckpt = fold / "best_model.pt"
            if ckpt.exists():
                paths.append(ckpt)
    final_ckpt = root / "final" / "fold_0" / "best_model.pt"
    if final_ckpt.exists() and final_ckpt not in paths:
        paths.append(final_ckpt)
    return paths


def load_models(ckpts: list[Path], device: torch.device) -> list[SFCN]:
    models = []
    for ckpt in ckpts:
        m = SFCN(channel_number=SFCN_CHANNELS, output_dim=NUM_CLASSES)
        load_pretrained(m, PRETRAINED_PATH)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        m.to(device).eval()
        models.append(m)
    return models


@torch.no_grad()
def forward_probs(x: torch.Tensor, models: list[SFCN], use_tta: bool) -> np.ndarray:
    views = [x]
    if use_tta:
        views.append(torch.flip(x, dims=[-3]))  # sagittal flip

    model_probs = []
    for m in models:
        view_probs = []
        for v in views:
            logp = flatten_logprob(m(v))
            view_probs.append(torch.exp(logp).cpu().numpy())
        model_probs.append(np.mean(view_probs, axis=0))
    return np.mean(model_probs, axis=0)[0]


@torch.no_grad()
def predict(eids: list[str], labels_df: pd.DataFrame, models: list[SFCN], device: torch.device) -> pd.DataFrame:
    ds = ADNIDataset(eids, labels_df, training=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    rows = []
    for batch in tqdm(loader, desc="predict"):
        x = batch["x"].to(device)
        eid = batch["eid"][0]
        mean_prob = forward_probs(x, models, USE_TTA)
        pred_idx = int(np.argmax(mean_prob))
        rows.append({
            "ID": eid,
            "Pre": IDX_TO_LABEL[pred_idx],
            "prob_CN": float(mean_prob[0]),
            "prob_MCI": float(mean_prob[1]),
            "prob_AD": float(mean_prob[2]),
        })
    return pd.DataFrame(rows)


def eval_predictions(merged: pd.DataFrame) -> dict:
    y_true = merged["label_true"].values
    y_pred = merged["Pre"].values
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES)).tolist()
    report = classification_report(
        y_true, y_pred, labels=list(CLASS_NAMES), output_dict=True, zero_division=0,
    )
    return {
        "n": len(merged),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion": cm,
        "per_class": report,
        "pred_counts": merged["Pre"].value_counts().to_dict(),
        "true_counts": merged["label_true"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "submission.csv")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    global USE_TTA
    if args.no_tta:
        USE_TTA = False

    device = get_device()
    ckpts = find_checkpoints()
    if not ckpts:
        raise FileNotFoundError("No checkpoints. Run: python train.py")

    print(f"Models: {len(ckpts)} | device: {device} | TTA: {USE_TTA}")

    labels_df = load_labels()
    eids = available_eids()
    models = load_models(ckpts, device)

    df = predict(eids, labels_df, models, device).sort_values("ID")
    submit = df[["ID", "Pre"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submit.to_csv(args.output, index=False)
    df.to_csv(args.output.with_name(args.output.stem + "_with_probs.csv"), index=False)

    print(f"Saved {len(submit)} rows -> {args.output}")
    print("Pre counts:", submit["Pre"].value_counts().to_dict())

    if args.eval:
        merged = submit.merge(
            labels_df.rename(columns={"eid": "ID", "label": "label_true"}),
            on="ID",
        )
        summary = eval_predictions(merged)
        print("Eval vs labels:")
        print(f"  acc={summary['acc']:.3f} bal={summary['balanced_acc']:.3f} F1={summary['f1_macro']:.3f}")
        print("  confusion (rows=true, cols=pred):", summary["confusion"])
        for cls in CLASS_NAMES:
            r = summary["per_class"].get(cls, {})
            print(f"  {cls}: P={r.get('precision', 0):.3f} R={r.get('recall', 0):.3f} F1={r.get('f1-score', 0):.3f}")
        with open(OUTPUTS_DIR / "predict_eval.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
