"""
DESI BAO data auto-fetch — CobayaSampler/bao_data git clone into project cache.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tav_shared.dataset_ledger import mark_processed
from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

BAO_DATA_REPO_URL = "https://github.com/CobayaSampler/bao_data.git"
DATASETS_DIR = TAU_SUPERBLOCK_ROOT / "datasets" / "desi"
BAO_DATA_DIR = DATASETS_DIR / "bao_data"
DEFAULT_COBAYA_ROOT = BAO_DATA_DIR / "desi_bao_dr2"
DESI_DATASET_ID = "desi:bao_data_dr2"


def _run_git(args: list[str], *, cwd: Path | None = None) -> None:
    cmd = ["git", *args]
    print(f"[DESI FETCH] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_desi_bao_data(
    *,
    force_refresh: bool = False,
    auto_fetch: bool = True,
) -> Path:
    """
    Return ``desi_bao_dr2`` directory — cache hit, git pull, or fresh clone.

    Pipeline stage: **pull → cache** under ``datasets/desi/bao_data/``.
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if DEFAULT_COBAYA_ROOT.is_dir() and any(DEFAULT_COBAYA_ROOT.glob("desi_gaussian_bao_*_mean.txt")):
        if not force_refresh:
            print(f"[DESI FETCH] Using cached DR2 tables: {DEFAULT_COBAYA_ROOT}")
            return DEFAULT_COBAYA_ROOT

    if not auto_fetch:
        raise FileNotFoundError(
            f"DESI DR2 tables not cached at {DEFAULT_COBAYA_ROOT}. "
            "Set auto_fetch or clone manually:\n"
            f"  git clone {BAO_DATA_REPO_URL} {BAO_DATA_DIR}"
        )

    if BAO_DATA_DIR.is_dir() and (BAO_DATA_DIR / ".git").is_dir():
        print(f"[DESI FETCH] Updating existing clone: {BAO_DATA_DIR}")
        _run_git(["pull", "--ff-only"], cwd=BAO_DATA_DIR)
    else:
        if BAO_DATA_DIR.exists():
            raise FileNotFoundError(
                f"{BAO_DATA_DIR} exists but is not a git repo — remove or fix manually."
            )
        print(f"[DESI FETCH] Cloning {BAO_DATA_REPO_URL} → {BAO_DATA_DIR}")
        _run_git(["clone", "--depth", "1", BAO_DATA_REPO_URL, str(BAO_DATA_DIR)])

    if not DEFAULT_COBAYA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Clone succeeded but {DEFAULT_COBAYA_ROOT} is missing — check bao_data layout."
        )

    mark_processed([DESI_DATASET_ID], note="desi bao_data auto-fetch")
    print(f"[DESI FETCH] DR2 tables ready: {DEFAULT_COBAYA_ROOT}")
    return DEFAULT_COBAYA_ROOT