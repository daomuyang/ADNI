#!/usr/bin/env python3
"""
Fine-tune SFCN for ADNI 3-class diagnosis (CN/MCI/AD).

Transfer learning from UKBiobank brain-age SFCN (Peng et al. MedIA 2021).
Small-sample adaptations: freeze backbone, differential LR, class weights, label smoothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOP_PATIENCE,
    FREEZE_BACKBONE_EPOCHS,
    LABEL_SMOOTHING,
    LABEL_TO_IDX,
    LR_BACKBONE_MULT,
    LR_DECAY_EVERY,
    LR_DECAY_FACTOR,
    LR_INIT,
    MODELS_DIR,
    N_FOLDS,
    NUM_CLASSES,
    NUM_EPOCHS,
    OUTPUTS_DIR,
    PRETRAINED_PATH,
    RANDOM_SEED,
    SFCN_CHANNELS,
    SGD_MOMENTUM,
    USE_CLASS_WEIGHTS,
    VAL_RATIO,
    WEIGHT_DECAY,
)
from dataset import ADNIDataset, available_eids, load_labels
from dp_model.model_files.sfcn import SFCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained(model: SFCN, path: Path) -> dict[str, int]:
    if not path.exists():
        logger.warning("Pretrained not found: %s — random init", path)
        return {"loaded": 0, "missing": len(model.state_dict()), "unexpected": 0}
    state = torch.load(path, map_location="cpu", weights_only=False)
    mapped = {k.replace("module.", ""): v for k, v in state.items()}
    model_state = model.state_dict()
    compatible = {
        k: v for k, v in mapped.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    skipped = len(mapped) - len(compatible)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    logger.info(
        "Loaded %s | matched=%d skipped_shape=%d missing=%d unexpected=%d",
        path.name, len(compatible), skipped, len(missing), len(unexpected),
    )
    return {
        "loaded": len(compatible),
        "skipped_shape": skipped,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def flatten_logprob(output) -> torch.Tensor:
    x = output[0]
    if x.dim() == 5:
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
    return x.reshape(x.size(0), -1)


def split_stats(name: str, eids: list[str], labels_df: pd.DataFrame) -> dict[str, Any]:
    sub = labels_df.set_index("eid").loc[eids]
    counts = sub["label"].value_counts().to_dict()
    return {"name": name, "n": len(eids), "label_counts": counts}


def log_split(name: str, eids: list[str], labels_df: pd.DataFrame) -> dict[str, Any]:
    stats = split_stats(name, eids, labels_df)
    counts = stats["label_counts"]
    parts = " ".join(f"{k}={counts.get(k, 0)}" for k in CLASS_NAMES)
    logger.info("%s | n=%d | %s", name, stats["n"], parts)
    return stats


def backbone_params(model: SFCN) -> list[torch.nn.Parameter]:
    return list(model.feature_extractor.parameters())


def head_params(model: SFCN) -> list[torch.nn.Parameter]:
    return list(model.classifier.parameters())


def set_backbone_trainable(model: SFCN, trainable: bool) -> None:
    for p in backbone_params(model):
        p.requires_grad = trainable


def make_class_weights(train_eids: list[str], labels_df: pd.DataFrame, device: torch.device) -> torch.Tensor | None:
    if not USE_CLASS_WEIGHTS:
        return None
    y = labels_df.set_index("eid").loc[train_eids, "label"].map(LABEL_TO_IDX).values
    weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def smooth_targets(y: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return F.one_hot(y, num_classes).float()
    with torch.no_grad():
        smooth = torch.full((y.size(0), num_classes), smoothing / (num_classes - 1), device=y.device)
        smooth.scatter_(1, y.unsqueeze(1), 1.0 - smoothing)
    return smooth


def classification_loss(
    logp: torch.Tensor,
    y: torch.Tensor,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    if label_smoothing > 0:
        target = smooth_targets(y, logp.size(1), label_smoothing)
        loss = -(target * logp).sum(dim=1)
        if class_weights is not None:
            loss = loss * class_weights[y]
        return loss.mean()
    return F.nll_loss(logp, y, weight=class_weights)


def _diag_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(true, pred, labels=list(range(NUM_CLASSES))).tolist()
    report = classification_report(
        true, pred, labels=list(range(NUM_CLASSES)), target_names=list(CLASS_NAMES),
        output_dict=True, zero_division=0,
    )
    return {
        "acc": float(accuracy_score(true, pred)),
        "balanced_acc": float(balanced_accuracy_score(true, pred)),
        "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "confusion": cm,
        "per_class": report,
        "pred_counts": {CLASS_NAMES[i]: int((pred == i).sum()) for i in range(NUM_CLASSES)},
        "true_counts": {CLASS_NAMES[i]: int((true == i).sum()) for i in range(NUM_CLASSES)},
    }


@torch.no_grad()
def evaluate(
    model: SFCN,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    pred_list, true_list, losses = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logp = flatten_logprob(model(x))
        losses.append(float(classification_loss(logp, y, class_weights, 0.0).item()))
        pred_list.extend(logp.argmax(dim=1).cpu().numpy().tolist())
        true_list.extend(y.cpu().numpy().tolist())
    pred = np.array(pred_list)
    true = np.array(true_list)
    metrics = _diag_metrics(pred, true)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train_epoch(
    model: SFCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    desc: str,
) -> float:
    model.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logp = flatten_logprob(model(x))
        loss = classification_loss(logp, y, class_weights, label_smoothing)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


def build_optimizer(model: SFCN, lr: float, backbone_mult: float) -> torch.optim.SGD:
    return torch.optim.SGD(
        [
            {"params": backbone_params(model), "lr": lr * backbone_mult},
            {"params": head_params(model), "lr": lr},
        ],
        momentum=SGD_MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )


def fit_fold(
    train_eids: list[str],
    val_eids: list[str],
    labels_df: pd.DataFrame,
    fold_id: int,
    device: torch.device,
    out_dir: Path,
    num_epochs: int,
    freeze_epochs: int,
) -> dict[str, Any]:
    log_split(f"fold {fold_id} train", train_eids, labels_df)
    log_split(f"fold {fold_id} val", val_eids, labels_df)

    model = SFCN(channel_number=SFCN_CHANNELS, output_dim=NUM_CLASSES)
    load_pretrained(model, PRETRAINED_PATH)
    model = model.to(device)
    class_weights = make_class_weights(train_eids, labels_df, device)

    set_backbone_trainable(model, False)
    optimizer = build_optimizer(model, LR_INIT, LR_BACKBONE_MULT)

    train_loader = DataLoader(
        ADNIDataset(train_eids, labels_df, training=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ADNIDataset(val_eids, labels_df, training=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    best_score = -float("inf")
    best_state = None
    best_metrics: dict[str, Any] = {}
    patience = 0
    history = []

    epoch_bar = tqdm(range(1, num_epochs + 1), desc=f"fold {fold_id}", unit="ep")
    for epoch in epoch_bar:
        if epoch == freeze_epochs + 1:
            set_backbone_trainable(model, True)
            logger.info("fold %d epoch %d: unfreeze backbone", fold_id, epoch)

        if epoch > 1 and (epoch - 1) % LR_DECAY_EVERY == 0:
            for pg in optimizer.param_groups:
                pg["lr"] *= LR_DECAY_FACTOR
            logger.info(
                "fold %d epoch %d: lr backbone=%.6f head=%.6f",
                fold_id, epoch,
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
            )

        train_loss = train_epoch(
            model, train_loader, optimizer, device, class_weights, LABEL_SMOOTHING,
            desc=f"fold{fold_id} train ep{epoch}",
        )
        val_metrics = evaluate(model, val_loader, device, class_weights)
        score = val_metrics["balanced_acc"]
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        epoch_bar.set_postfix(
            train=f"{train_loss:.3f}",
            val_acc=f"{val_metrics['acc']:.3f}",
            bal=f"{val_metrics['balanced_acc']:.3f}",
        )
        logger.info(
            "fold %d ep %d | train=%.4f val_loss=%.4f acc=%.3f bal=%.3f F1=%.3f "
            "pred %s true %s",
            fold_id, epoch, train_loss, val_metrics["loss"],
            val_metrics["acc"], val_metrics["balanced_acc"], val_metrics["f1_macro"],
            val_metrics["pred_counts"], val_metrics["true_counts"],
        )

        if score > best_score:
            best_score = score
            best_metrics = val_metrics.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                logger.info("fold %d early stop at epoch %d", fold_id, epoch)
                break

    fold_dir = out_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if best_state:
        torch.save(best_state, fold_dir / "best_model.pt")
    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    payload = {
        "fold": fold_id,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "epochs": len(history),
        "train_split": split_stats(f"fold{fold_id} train", train_eids, labels_df),
        "val_split": split_stats(f"fold{fold_id} val", val_eids, labels_df),
    }
    with open(fold_dir / "best_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    mets = [r["best_metrics"] for r in results if r.get("best_metrics")]
    summary: dict[str, Any] = {"n_folds": len(results), "folds": results}
    for k in ("acc", "balanced_acc", "f1_macro", "f1_weighted", "loss"):
        vals = [m[k] for m in mets if k in m]
        if vals:
            summary[f"mean_{k}"] = float(np.mean(vals))
            summary[f"std_{k}"] = float(np.std(vals))
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    logger.info("=" * 72)
    logger.info("DIAGNOSIS CROSS-VALIDATION SUMMARY (%d folds)", summary["n_folds"])
    for fold in summary["folds"]:
        m = fold.get("best_metrics", {})
        logger.info(
            "  fold %s | acc=%.3f bal=%.3f F1=%.3f",
            fold.get("fold"), m.get("acc", float("nan")),
            m.get("balanced_acc", float("nan")), m.get("f1_macro", float("nan")),
        )
    logger.info(
        "  MEAN | acc=%.3f±%.3f bal=%.3f±%.3f F1=%.3f±%.3f",
        summary.get("mean_acc", float("nan")), summary.get("std_acc", 0),
        summary.get("mean_balanced_acc", float("nan")), summary.get("std_balanced_acc", 0),
        summary.get("mean_f1_macro", float("nan")), summary.get("std_f1_macro", 0),
    )
    logger.info("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--freeze-epochs", type=int, default=FREEZE_BACKBONE_EPOCHS)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    labels_df = load_labels()
    eids = available_eids()
    if len(eids) < 10:
        raise RuntimeError(f"Only {len(eids)} preprocessed subjects. Run preprocess.py first.")

    y_labels = labels_df.set_index("eid").loc[eids, "label"].map(LABEL_TO_IDX).values
    log_split("Full dataset", eids, labels_df)

    logger.info(
        "Device: %s | subjects: %d | epochs: %d | freeze: %d | lr: %.4f",
        device, len(eids), args.epochs, args.freeze_epochs, LR_INIT,
    )
    logger.info("Pretrained: %s", PRETRAINED_PATH.name)

    task_dir = MODELS_DIR / "diagnosis"
    all_summaries: dict[str, Any] = {}

    cv_results = []
    if not args.skip_cv:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_SEED)
        for fold_id, (tr_idx, va_idx) in enumerate(skf.split(eids, y_labels)):
            tr = [eids[i] for i in tr_idx]
            va = [eids[i] for i in va_idx]
            cv_results.append(
                fit_fold(tr, va, labels_df, fold_id, device, task_dir / "cv", args.epochs, args.freeze_epochs)
            )
        summary = aggregate_results(cv_results)
        all_summaries["cv"] = summary
        with open(OUTPUTS_DIR / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print_summary(summary)

    if args.cv_only:
        return

    try:
        tr, ho = train_test_split(
            eids, test_size=VAL_RATIO, random_state=RANDOM_SEED, stratify=y_labels,
        )
    except ValueError:
        tr, ho = train_test_split(eids, test_size=VAL_RATIO, random_state=RANDOM_SEED)

    final = fit_fold(tr, ho, labels_df, 0, device, task_dir / "final", args.epochs, args.freeze_epochs)
    all_summaries["final"] = final
    with open(OUTPUTS_DIR / "final_summary.json", "w") as f:
        json.dump(final, f, indent=2)

    m = final.get("best_metrics", {})
    logger.info(
        "FINAL holdout | acc=%.3f bal=%.3f F1=%.3f",
        m.get("acc", float("nan")), m.get("balanced_acc", float("nan")),
        m.get("f1_macro", float("nan")),
    )

    with open(OUTPUTS_DIR / "train_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    logger.info("TRAINING FINISHED — Next: python predict.py --eval")


if __name__ == "__main__":
    main()
