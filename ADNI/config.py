"""ADNI CN/MCI/AD diagnosis — SFCN transfer from UKBiobank_deep_pretrain."""
import os
from pathlib import Path

ADNI_ROOT = Path(__file__).resolve().parent
RAW_DIR = ADNI_ROOT / "ADNI_data"
LABELS_CSV = RAW_DIR / "selected_ADNI_105_info.csv"
PREPROCESSED_DIR = ADNI_ROOT / "preprocessed"
MODELS_DIR = ADNI_ROOT / "models"
OUTPUTS_DIR = ADNI_ROOT / "outputs"
PRETRAINED_DIR = MODELS_DIR / "pretrained"

# UKBB brain-age SFCN weights for transfer learning (Peng et al. MedIA 2021)
PRETRAINED_PATH = PRETRAINED_DIR / "run_20190719_00_epoch_best_mae.p"

# SFCN input (official: [B, 1, 160, 192, 160])
TARGET_SHAPE = (160, 192, 160)
TARGET_VOXEL_SIZE = (1.0, 1.0, 1.0)
PREPROCESSED_NPY_NAME = "T1_MNI_brain_160.npy"
PREPROCESSED_META_NAME = "preprocess_meta.json"
PREPROCESS_PIPELINE_VERSION = "2026-06-06-v9_adni_n4_robustfov"

SFCN_NORMALIZATION_METHOD = "sfcn_official_divide_by_mean"
SFCN_NORMALIZATION_SOURCE = "UKBiobank_deep_pretrain/examples.ipynb"

PREPROCESS_WORKERS = max(1, min(8, os.cpu_count() or 4))
# ADNI elderly brains: slightly lower BET fraction retains atrophic tissue (default UKB uses 0.5)
BET_FRACTION = 0.4

# 3-class diagnosis
CLASS_NAMES = ("CN", "MCI", "AD")
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# SFCN backbone (same as UKBB age model)
SFCN_CHANNELS = [32, 64, 128, 256, 256, 64]

# Fine-tune hyperparams (Peng SGD + small-sample ADNI adaptations)
RANDOM_SEED = 42
N_FOLDS = 5
BATCH_SIZE = 4
NUM_EPOCHS = 100
LR_INIT = 0.005
LR_BACKBONE_MULT = 0.1  # differential LR: backbone slower than head
LR_DECAY_FACTOR = 0.5
LR_DECAY_EVERY = 30
WEIGHT_DECAY = 0.001
SGD_MOMENTUM = 0.9
EARLY_STOP_PATIENCE = 15
VAL_RATIO = 0.2
FREEZE_BACKBONE_EPOCHS = 8  # train classifier head first
LABEL_SMOOTHING = 0.05
USE_CLASS_WEIGHTS = True
USE_TTA = True  # test-time augmentation at inference
