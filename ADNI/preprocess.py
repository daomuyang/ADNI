#!/usr/bin/env python3
"""
ADNI T1 preprocessing for SFCN transfer learning.

Combines ADNI best-practice FSL steps with UKBiobank_deep_pretrain inference:
  - ADNI: N4 + robustfov + BET + FLIRT (SANCHES-Pedro/adni_preprocessing style)
  - SFCN: divide-by-mean + crop_center (examples.ipynb)

Pipeline:
  1. fslreorient2std (if available)
  2. Resample to 1 mm isotropic
  3. N4 bias field correction (SimpleITK)
  4. FSL robustfov (crop neck/empty FoV before skull-strip)
  5. FSL BET (-R -f 0.5 -g 0)
  6. FSL FLIRT affine to MNI152 1 mm brain (linear only, matches UKBB pretrain)
  7. data / data.mean(); crop_center -> 160×192×160

Outputs: ADNI/preprocessed/<eid>/T1_MNI_brain_160.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from config import (
    BET_FRACTION,
    LABELS_CSV,
    PREPROCESSED_DIR,
    PREPROCESS_PIPELINE_VERSION,
    PREPROCESS_WORKERS,
    RANDOM_SEED,
    RAW_DIR,
    SFCN_NORMALIZATION_METHOD,
    SFCN_NORMALIZATION_SOURCE,
    TARGET_SHAPE,
    TARGET_VOXEL_SIZE,
    PREPROCESSED_NPY_NAME,
    PREPROCESSED_META_NAME,
)
from dp_model.dp_utils import crop_center

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STANDARD_MNI_NAMES = {
    "T1_brain_linearto_MNI.nii.gz",
    "T1_brain_to_MNI.nii.gz",
    "T1_unbiased_brain_linearto_MNI.nii.gz",
}

_RUN_ID: str | None = None


def set_seed(seed: int, sitk_threads: int | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    n_threads = sitk_threads if sitk_threads is not None else 1
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n_threads)


def fsl_install_message() -> str:
    return (
        "SFCN preprocessing requires FSL flirt, bet, FSLDIR, and "
        "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz.\n\n"
        "Try: conda install -y -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ "
        "-c conda-forge fsl-flirt fsl-bet2 fsl-data_standard"
    )


def check_fsl_dependencies() -> tuple[Path, Path, Path, Path | None]:
    flirt = shutil.which("flirt")
    bet = shutil.which("bet")
    robustfov = shutil.which("robustfov")
    fsldir = os.environ.get("FSLDIR")
    missing: list[str] = []
    if not flirt:
        missing.append("flirt")
    if not bet:
        missing.append("bet")
    if not fsldir:
        missing.append("FSLDIR")
        mni = None
    else:
        mni = Path(fsldir) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"
        if not mni.exists():
            missing.append(str(mni))
    if missing:
        raise RuntimeError(
            "Missing FSL dependency: " + ", ".join(missing) + "\n\n" + fsl_install_message()
        )
    return Path(flirt), Path(bet), mni, Path(robustfov) if robustfov else None  # type: ignore[arg-type]


def fsl_context() -> dict[str, str | dict[str, str]]:
    flirt, bet, mni, robustfov = check_fsl_dependencies()
    env = os.environ.copy()
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    ctx: dict[str, str | dict[str, str]] = {
        "flirt": str(flirt), "bet": str(bet), "mni": str(mni), "env": env,
    }
    if robustfov:
        ctx["robustfov"] = str(robustfov)
    return ctx


def run_command(cmd: list[str | Path], env: dict[str, str]) -> None:
    result = subprocess.run(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(stderr or f"command failed: {' '.join(map(str, cmd))}")


def save_sitk(image: sitk.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(image, sitk.sitkFloat32), str(path))
    return path


def image_shape(path: Path) -> tuple[int, int, int]:
    return tuple(int(v) for v in nib.load(str(path)).shape[:3])


def find_existing_mni_file(case_dir: Path) -> Path | None:
    by_name = {p.name: p for p in case_dir.rglob("*.nii*")}
    for name in STANDARD_MNI_NAMES:
        if name in by_name:
            return by_name[name]
    return None


def read_nifti_as_sitk(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def n4_bias_field_correction(image: sitk.Image) -> sitk.Image:
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([40, 40, 20, 10])
    return corrector.Execute(image, mask)


def resample_to_spacing(image: sitk.Image, spacing: tuple[float, float, float]) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[i] * original_spacing[i] / spacing[i]))
        for i in range(3)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def array_stats(data: np.ndarray) -> dict[str, float]:
    return {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
    }


def sfcn_official_scale(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32, copy=True)
    mean = float(data.mean())
    if abs(mean) <= 1e-6:
        raise RuntimeError("divide-by-mean failed: image mean too small")
    return data / mean


def official_postprocess_from_mni(mni_path: Path) -> tuple[np.ndarray, dict]:
    """Match examples.ipynb: divide-by-mean on full MNI volume, then crop_center."""
    img = nib.load(str(mni_path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    raw_stats = array_stats(data)
    if not all(data.shape[i] >= TARGET_SHAPE[i] for i in range(3)):
        raise RuntimeError(
            f"MNI volume {data.shape} smaller than target {TARGET_SHAPE}; "
            "cannot use official crop_center"
        )
    scaled = sfcn_official_scale(data)
    cropped = crop_center(scaled, TARGET_SHAPE)
    normalized_stats = array_stats(cropped)
    if tuple(cropped.shape) != TARGET_SHAPE:
        raise RuntimeError(f"final shape is {cropped.shape}, expected {TARGET_SHAPE}")
    return cropped.astype(np.float32), {
        "normalization_method": SFCN_NORMALIZATION_METHOD,
        "normalization_source": SFCN_NORMALIZATION_SOURCE,
        "crop_method": "dp_utils.crop_center",
        "mni_shape_before_crop": list(data.shape),
        "raw_min": raw_stats["min"],
        "raw_max": raw_stats["max"],
        "raw_mean": raw_stats["mean"],
        "raw_std": raw_stats["std"],
        "normalized_min": normalized_stats["min"],
        "normalized_max": normalized_stats["max"],
        "normalized_mean": normalized_stats["mean"],
        "normalized_std": normalized_stats["std"],
    }


def run_fsl_pipeline(
    raw_path: Path,
    case_dir: Path,
    work_dir: Path,
    eid: str,
    fsl: dict[str, str | dict[str, str]],
) -> tuple[Path, dict]:
    env = fsl["env"]  # type: ignore[assignment]
    details: dict = {
        "used_existing_mni_file": 0,
        "used_fslreorient2std": 0,
        "used_robustfov": 0,
        "used_n4": 1,
        "input_shape": shape_text(image_shape(raw_path)),
        "shape_after_reorient": "",
        "shape_after_1mm": "",
        "shape_after_n4": "",
        "shape_after_robustfov": "",
        "shape_after_bet": "",
        "shape_after_flirt": "",
    }

    reorient_cmd = shutil.which("fslreorient2std")
    if reorient_cmd:
        reorient_path = work_dir / f"{eid}_reorient.nii.gz"
        run_command([reorient_cmd, raw_path, reorient_path], env)
        current_path = reorient_path
        details["used_fslreorient2std"] = 1
        details["shape_after_reorient"] = shape_text(image_shape(reorient_path))
    else:
        current_path = raw_path

    one_mm_path = work_dir / f"{eid}_1mm.nii.gz"
    image = resample_to_spacing(read_nifti_as_sitk(current_path), spacing=TARGET_VOXEL_SIZE)
    save_sitk(image, one_mm_path)
    details["shape_after_1mm"] = shape_text(image_shape(one_mm_path))

    n4_path = work_dir / f"{eid}_n4.nii.gz"
    image = n4_bias_field_correction(read_nifti_as_sitk(one_mm_path))
    save_sitk(image, n4_path)
    details["shape_after_n4"] = shape_text(image_shape(n4_path))

    bet_input = n4_path
    if fsl.get("robustfov"):
        roi_path = work_dir / f"{eid}_roi.nii.gz"
        run_command([fsl["robustfov"], "-i", n4_path, "-r", roi_path], env)
        bet_input = roi_path
        details["used_robustfov"] = 1
        details["shape_after_robustfov"] = shape_text(image_shape(roi_path))

    brain_path = work_dir / f"{eid}_brain_1mm.nii.gz"
    run_command(
        [fsl["bet"], bet_input, brain_path, "-R", "-f", str(BET_FRACTION), "-g", "0"], env,
    )
    if not brain_path.exists():
        raise RuntimeError("BET finished but output file is missing")
    details["shape_after_bet"] = shape_text(image_shape(brain_path))

    flirt_path = work_dir / f"{eid}_brain_linearto_MNI.nii.gz"
    mat_path = work_dir / f"{eid}_brain_to_MNI.mat"
    run_command(
        [
            fsl["flirt"],
            "-in", brain_path,
            "-ref", fsl["mni"],
            "-out", flirt_path,
            "-omat", mat_path,
            "-dof", "12",
            "-cost", "corratio",
            "-interp", "trilinear",
        ],
        env,
    )
    if not flirt_path.exists() or not mat_path.exists():
        raise RuntimeError("FLIRT finished but output image or matrix is missing")
    details["shape_after_flirt"] = shape_text(image_shape(flirt_path))
    return flirt_path, details


def shape_text(values: tuple[int, ...] | list[int] | str) -> str:
    if isinstance(values, str):
        return values
    return "x".join(str(int(v)) for v in values)


def preprocess_subject(
    eid: str,
    raw_path: Path,
    case_dir: Path,
    out_dir: Path,
    fsl: dict[str, str | dict[str, str]],
    save_nifti: bool = True,
    run_meta: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    details: dict = {
        "source": str(raw_path),
        "used_existing_mni_file": 0,
        "used_fslreorient2std": 0,
        "used_robustfov": 0,
        "used_n4": 0,
        "input_shape": "",
        "shape_after_reorient": "",
        "shape_after_1mm": "",
        "shape_after_n4": "",
        "shape_after_robustfov": "",
        "shape_after_bet": "",
        "shape_after_flirt": "",
    }

    existing_mni = find_existing_mni_file(case_dir)
    if existing_mni is not None:
        details["source"] = str(existing_mni)
        details["used_existing_mni_file"] = 1
        details["input_shape"] = shape_text(image_shape(existing_mni))
        data, norm_meta = official_postprocess_from_mni(existing_mni)
    else:
        with tempfile.TemporaryDirectory(prefix=f"sfcn_{eid}_") as temp_dir:
            flirt_path, step_details = run_fsl_pipeline(
                raw_path, case_dir, Path(temp_dir), eid, fsl
            )
            details.update(step_details)
            data, norm_meta = official_postprocess_from_mni(flirt_path)

    if float(np.abs(data).max()) < 1e-4:
        raise RuntimeError(
            f"Preprocess produced near-empty volume for {raw_path} "
            "(check BET / FLIRT registration)"
        )

    npy_path = out_dir / PREPROCESSED_NPY_NAME
    np.save(npy_path, data.astype(np.float32))

    meta = {
        "source": details["source"],
        "shape": list(data.shape),
        "mean": norm_meta["normalized_mean"],
        "std": norm_meta["normalized_std"],
        "min": norm_meta["normalized_min"],
        "max": norm_meta["normalized_max"],
        "normalization": SFCN_NORMALIZATION_METHOD,
        "normalization_source": SFCN_NORMALIZATION_SOURCE,
        "pipeline_version": PREPROCESS_PIPELINE_VERSION,
        "used_existing_mni_file": details.get("used_existing_mni_file", 0),
        "used_fslreorient2std": details.get("used_fslreorient2std", 0),
        "used_robustfov": details.get("used_robustfov", 0),
        "used_n4": details.get("used_n4", 0),
        "crop_method": norm_meta.get("crop_method", "dp_utils.crop_center"),
        "mni_shape_before_crop": norm_meta.get("mni_shape_before_crop", []),
        "input_shape": details.get("input_shape", ""),
        "shape_after_reorient": details.get("shape_after_reorient", ""),
        "shape_after_1mm": details.get("shape_after_1mm", ""),
        "shape_after_n4": details.get("shape_after_n4", ""),
        "shape_after_robustfov": details.get("shape_after_robustfov", ""),
        "shape_after_bet": details.get("shape_after_bet", ""),
        "shape_after_flirt": details.get("shape_after_flirt", ""),
        "raw_mean": norm_meta["raw_mean"],
        "raw_std": norm_meta["raw_std"],
    }
    if run_meta:
        meta.update(run_meta)
    if save_nifti:
        nii_out = out_dir / "T1_MNI_brain_160.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(nii_out))
        meta["nifti"] = str(nii_out)

    with open(out_dir / PREPROCESSED_META_NAME, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def subject_needs_processing(eid: str, overwrite: bool, only_outdated: bool) -> bool:
    out_dir = PREPROCESSED_DIR / eid
    done_flag = out_dir / PREPROCESSED_NPY_NAME
    meta_path = out_dir / PREPROCESSED_META_NAME
    if overwrite:
        return True
    if only_outdated:
        if not done_flag.exists() or not meta_path.exists():
            return True
        with open(meta_path) as f:
            meta = json.load(f)
        return (
            meta.get("pipeline_version") != PREPROCESS_PIPELINE_VERSION
            or not meta.get("run_id")
        )
    return not done_flag.exists()


def collect_subjects() -> list[tuple[str, Path, Path]]:
    """ADNI: use relative_path from CSV (supports .nii and .nii.gz)."""
    df = pd.read_csv(LABELS_CSV)
    subjects: list[tuple[str, Path, Path]] = []
    for _, row in df.iterrows():
        eid = str(row["eid"])
        rel = str(row["relative_path"])
        raw_path = RAW_DIR / rel
        case_dir = raw_path.parent
        if raw_path.exists():
            subjects.append((eid, raw_path, case_dir))
        else:
            logger.warning("Missing image for eid=%s: %s", eid, raw_path)
    subjects.sort(key=lambda x: x[0])
    return subjects


def preprocess_worker(job: dict) -> tuple[str, bool, str | None]:
    idx = job["idx"]
    eid = job["eid"]
    try:
        set_seed(job["seed"] + idx, sitk_threads=1)
        preprocess_subject(
            eid=eid,
            raw_path=Path(job["raw_path"]),
            case_dir=Path(job["case_dir"]),
            out_dir=PREPROCESSED_DIR / eid,
            fsl=job["fsl"],
            save_nifti=job["save_nifti"],
            run_meta=job["run_meta"],
        )
        return eid, True, None
    except Exception as e:
        return eid, False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ADNI T1 volumes (SFCN FSL pipeline)")
    parser.add_argument("--limit", type=int, default=0, help="Process only N subjects (0=all)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-outdated", action="store_true")
    parser.add_argument("--no-nifti", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=PREPROCESS_WORKERS)
    args = parser.parse_args()

    workers = max(1, int(args.workers))
    set_seed(args.seed, sitk_threads=1 if workers > 1 else None)
    fsl = fsl_context()

    logger.info("Pipeline version: %s", PREPROCESS_PIPELINE_VERSION)
    logger.info("Normalization: %s", SFCN_NORMALIZATION_METHOD)
    logger.info("Workers: %d", workers)

    global _RUN_ID
    _RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_meta = {"run_id": _RUN_ID, "random_seed": args.seed, "overwrite": args.overwrite}

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    subjects = collect_subjects()
    if args.limit > 0:
        subjects = subjects[: args.limit]

    todo = [
        (eid, raw_path, case_dir)
        for eid, raw_path, case_dir in subjects
        if subject_needs_processing(eid, args.overwrite, args.only_outdated)
    ]
    logger.info(
        "Subjects in CSV: %d | to process: %d | skip: %d",
        len(subjects), len(todo), len(subjects) - len(todo),
    )

    ok, fail = 0, 0
    skip = len(subjects) - len(todo)

    if workers == 1:
        for idx, (eid, raw_path, case_dir) in enumerate(tqdm(todo, desc="preprocess")):
            try:
                set_seed(args.seed + idx, sitk_threads=None)
                preprocess_subject(
                    eid=eid, raw_path=raw_path, case_dir=case_dir,
                    out_dir=PREPROCESSED_DIR / eid, fsl=fsl,
                    save_nifti=not args.no_nifti, run_meta=run_meta,
                )
                ok += 1
            except Exception as e:
                logger.error("Failed %s: %s", eid, e)
                fail += 1
    else:
        jobs = [
            {
                "idx": idx, "eid": eid, "raw_path": str(raw_path),
                "case_dir": str(case_dir), "save_nifti": not args.no_nifti,
                "run_meta": run_meta, "seed": args.seed, "fsl": fsl,
            }
            for idx, (eid, raw_path, case_dir) in enumerate(todo)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(preprocess_worker, job) for job in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="preprocess"):
                eid, success, error = fut.result()
                if success:
                    ok += 1
                else:
                    logger.error("Failed %s: %s", eid, error)
                    fail += 1

    summary = {
        "ok": ok, "skip": skip, "fail": fail, "total": len(subjects),
        "pipeline_version": PREPROCESS_PIPELINE_VERSION,
        "normalization": SFCN_NORMALIZATION_METHOD,
        "run_id": _RUN_ID, "random_seed": args.seed,
        "overwrite": args.overwrite, "workers": workers,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(PREPROCESSED_DIR / "preprocess_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Done: %s", summary)


if __name__ == "__main__":
    main()
