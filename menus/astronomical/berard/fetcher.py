"""
Asset fetcher for claygatlin/tau-cosmology-berard-framework.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tav_shared.tav_project_paths import PROJECT_ROOT, TAU_SUPERBLOCK_ROOT

BERARD_REPO_URL = "https://github.com/claygatlin/tau-cosmology-berard-framework"
TAU_COSMOLOGY_REPO_URL = "https://github.com/claygatlin/tau-cosmology"
DATASETS_DIR = TAU_SUPERBLOCK_ROOT / "datasets" / "berard_framework"
TEXT_DIR = DATASETS_DIR / "text"
QUANTUM_DIR = DATASETS_DIR / "quantum_counts"
FIGURES_DIR = DATASETS_DIR / "figures"
NGC3198_BENCH_DIR = DATASETS_DIR / "ngc3198_benchmark"
DATA_DIR = DATASETS_DIR / "data"
MANIFEST_PATH = DATASETS_DIR / "manifest.json"

PERSONAL_FILES_CANDIDATES: tuple[Path, ...] = (
    PROJECT_ROOT.parent / "Personal_Files" / "berard-framework",
    TAU_SUPERBLOCK_ROOT / "Personal_Files" / "berard-framework",
    Path.home() / "Personal_Files" / "berard-framework",
)

TEXT_ASSETS: tuple[str, ...] = (
    "README.md",
    "summaries/Executive Summary.txt",
    "notes/ARCHITECTURE OVERVIEW — BERARD FRAMEWORK.txt",
    "notes/ROADMAP — BERARD FRAMEWORK.md",
    "notes/METHODOLOGY NOTE — BERARD FRAMEWORK.md",
    "notes/DESIGN PHILOSOPHY — BERARD FRAMEWORK.txt",
)

SAMPLE_COUNTS: tuple[str, ...] = (
    "neg_reps3_1054_262k_counts.json",
    "reps3_1054_262k_counts.json",
    "reps3_1054_131k_counts.json",
)


@dataclass
class BerardSyncSummary:
    text_files: list[str]
    quantum_files: list[str]
    personal_files: list[str]
    tau_cosmology_files: list[str]
    figure_files: list[str]
    source: str
    tau_cosmology_source: str


def _ensure_dirs() -> None:
    for path in (DATASETS_DIR, TEXT_DIR, QUANTUM_DIR, FIGURES_DIR, NGC3198_BENCH_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for candidate in PERSONAL_FILES_CANDIDATES:
        candidate.mkdir(parents=True, exist_ok=True)


def _clone_berard_repo(dest: Path) -> Path:
    return _clone_repo(dest, BERARD_REPO_URL)


def sync_berard_assets(
    *,
    force_refresh: bool = False,
    include_large: bool = False,
    clone_dir: Path | None = None,
) -> BerardSyncSummary:
    """
    Sync textual + sample quantum assets from GitHub into datasets/berard_framework.
    """
    _ensure_dirs()
    text_out: list[str] = []
    quantum_out: list[str] = []
    personal_out: list[str] = []

    repo = clone_dir
    if repo is None:
        repo = Path("/tmp/tau-cosmology-berard-framework")
    if force_refresh or not (repo / "README.md").is_file():
        repo = _clone_berard_repo(repo)

    for rel in TEXT_ASSETS:
        src = repo / rel
        if not src.is_file():
            continue
        dest_name = Path(rel).name
        dest = TEXT_DIR / dest_name
        if force_refresh or not dest.is_file() or dest.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dest)
        text_out.append(str(dest))
        for personal_root in PERSONAL_FILES_CANDIDATES:
            pdest = personal_root / dest_name
            shutil.copy2(dest, pdest)
            personal_out.append(str(pdest))

    for name in SAMPLE_COUNTS:
        src = repo / name
        if not src.is_file():
            continue
        dest = QUANTUM_DIR / name
        if force_refresh or not dest.is_file():
            shutil.copy2(src, dest)
        quantum_out.append(str(dest))

    if include_large:
        large_name = "periodic_berard_131k_full.json"
        src = repo / large_name
        if src.is_file():
            dest = QUANTUM_DIR / large_name
            if force_refresh or not dest.is_file():
                shutil.copy2(src, dest)
            quantum_out.append(str(dest))

    tau_files, tau_src, fig_files = _sync_tau_cosmology_assets(
        force_refresh=force_refresh,
    )

    manifest: dict[str, Any] = {
        "repository": BERARD_REPO_URL,
        "tau_cosmology_repository": TAU_COSMOLOGY_REPO_URL,
        "text_assets": text_out,
        "quantum_counts": quantum_out,
        "personal_files": personal_out,
        "tau_cosmology_assets": tau_files,
        "figure_assets": fig_files,
        "notebooks_available_on_github": [
            "Tiny_Tau_v2.ipynb",
            "Tiny_Tau_v21.91.ipynb",
            "Tiny_Tau_v22.1.ipynb",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return BerardSyncSummary(
        text_files=text_out,
        quantum_files=quantum_out,
        personal_files=personal_out,
        tau_cosmology_files=tau_files,
        figure_files=fig_files,
        source=str(repo),
        tau_cosmology_source=tau_src,
    )


def _sync_tau_cosmology_assets(
    *,
    force_refresh: bool = False,
    clone_dir: Path | None = None,
) -> tuple[list[str], str, list[str]]:
    """Sync tau-cosmology NGC 3198 benchmark + PREREG + reference plots."""
    _ensure_dirs()
    copied: list[str] = []
    figures: list[str] = []
    repo = clone_dir or Path("/tmp/tau-cosmology")
    if force_refresh or not (repo / "PREREGISTRATION.md").is_file():
        repo = _clone_repo(repo, TAU_COSMOLOGY_REPO_URL)

    for name in ("PREREGISTRATION.md", "README.md", "test_units.py"):
        src = repo / name
        if src.is_file():
            dst = TEXT_DIR / name
            shutil.copy2(src, dst)
            copied.append(str(dst))

    bench = repo / "NGC_3198_Rotation-Curve_Benchmark"
    if bench.is_dir():
        for src in bench.iterdir():
            if src.suffix.lower() in {".png", ".pdf", ".md", ".py"}:
                dst = NGC3198_BENCH_DIR / src.name
                shutil.copy2(src, dst)
                copied.append(str(dst))
                if src.suffix.lower() == ".png":
                    fig_dst = FIGURES_DIR / f"ngc3198_{src.name}"
                    shutil.copy2(src, fig_dst)
                    figures.append(str(fig_dst))

    bf_fig = Path("/tmp/tau-cosmology-berard-framework/figures")
    if bf_fig.is_dir():
        for src in bf_fig.glob("*.pdf"):
            dst = FIGURES_DIR / src.name
            if force_refresh or not dst.is_file():
                shutil.copy2(src, dst)
            figures.append(str(dst))

    return copied, str(repo), figures


def _clone_repo(dest: Path, url: str) -> Path:
    if (dest / ".git").is_dir():
        subprocess.run(
            ["git", "-C", str(dest), "pull", "--ff-only"],
            check=False,
            capture_output=True,
            text=True,
        )
        return dest
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )
    return dest


def list_cached_quantum_counts() -> list[Path]:
    _ensure_dirs()
    return sorted(QUANTUM_DIR.glob("*_counts.json"))


def load_framework_readme() -> str:
    _ensure_dirs()
    readme = TEXT_DIR / "README.md"
    if readme.is_file():
        return readme.read_text(encoding="utf-8", errors="replace")
    return ""


def knowledge_index() -> dict[str, Any]:
    """Lightweight knowledge base index for RAG-style retrieval hooks."""
    _ensure_dirs()
    docs: list[dict[str, str]] = []
    for path in sorted(TEXT_DIR.glob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            docs.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "chars": str(len(text)),
                    "preview": text[:400].replace("\n", " "),
                }
            )
    return {
        "repository": BERARD_REPO_URL,
        "documents": docs,
        "manifest": str(MANIFEST_PATH) if MANIFEST_PATH.is_file() else None,
        "quantum_count_files": [str(p) for p in list_cached_quantum_counts()],
    }