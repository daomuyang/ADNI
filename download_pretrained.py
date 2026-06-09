#!/usr/bin/env python3
"""Download UKBiobank_deep_pretrain SFCN brain-age weights for transfer learning."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from config import PRETRAINED_DIR, PRETRAINED_PATH

WEIGHT = (
    PRETRAINED_PATH.name,
    "brain_age/run_20190719_00_epoch_best_mae.p",
    [
        "https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/raw/master/brain_age/run_20190719_00_epoch_best_mae.p",
        "https://media.githubusercontent.com/media/ha-ha-ha-han/UKBiobank_deep_pretrain/master/brain_age/run_20190719_00_epoch_best_mae.p",
    ],
)


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code != 200:
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest.stat().st_size > 1000
    except Exception:
        pass
    try:
        subprocess.run(["curl", "-L", "-o", str(dest), url], check=True, timeout=300)
        return dest.stat().st_size > 1000
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    name, relpath, urls = WEIGHT
    dest = PRETRAINED_DIR / name

    if dest.exists() and not args.force:
        print(f"OK exists: {dest}")
        return

    for url in urls:
        print(f"Downloading {name} from {url} ...")
        if download(url, dest):
            print(f"Saved {dest}")
            return

    repo = PRETRAINED_DIR / "UKBiobank_deep_pretrain"
    if not repo.exists():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain.git", str(repo)],
                check=True, timeout=600, cwd=str(PRETRAINED_DIR),
            )
        except Exception as e:
            print("git clone failed:", e, file=sys.stderr)
    src = repo / relpath
    if src.exists():
        shutil.copy2(src, dest)
        print(f"Copied {src} -> {dest}")
        return

    print(f"FAILED: {name}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
