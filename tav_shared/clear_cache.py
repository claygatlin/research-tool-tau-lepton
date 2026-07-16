"""
List and remove on-disk dataset caches per research-tool module.

Deletes downloaded files only. Pull/batch ledgers and processed flags are left
intact so completed work is not re-run; use Reset to Initial Run for that.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Set

from tav_shared.dataset_registry import DatasetProvider, TargetState, get_provider, list_modules
from tav_shared.tav_project_paths import DATASETS_ROOT

CLEAR_CACHE_MENU_LABEL = "Clear Dataset Cache"

_LEDGER_SKIP_NAMES = frozenset({"processed.txt"})
_INTAKE_LEDGER_NAMES = frozenset({"done.txt", "batch_done.txt", "record_metadata.json"})

_EXTRA_CACHE_MODULES: dict[str, tuple[str, Path]] = {
    "BERARD_FRAMEWORK": ("Berard Framework", DATASETS_ROOT / "berard_framework"),
    "TSB_CASIMIR": ("Casimir Tau-SB Scan", DATASETS_ROOT / "casimir"),
}


@dataclass
class ModuleCacheEntry:
    module_tag: str
    title: str
    target_count: int
    total_bytes: int
    paths: List[Path] = field(default_factory=list)


@dataclass
class ClearCacheResult:
    module_tag: str
    title: str
    deleted_paths: List[str] = field(default_factory=list)
    bytes_freed: int = 0
    errors: List[str] = field(default_factory=list)


def format_size(num_bytes: int) -> str:
    """Human-readable byte size."""
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _is_cache_ledger(path: Path) -> bool:
    return path.name in _INTAKE_LEDGER_NAMES


def _scan_intake_cache(intake_dir: Path) -> List[Path]:
    if not intake_dir.is_dir():
        return []
    paths: List[Path] = []
    for child in sorted(intake_dir.iterdir()):
        if child.name in _LEDGER_SKIP_NAMES or _is_cache_ledger(child):
            continue
        if child.is_file() or child.is_dir():
            paths.append(child)
    return paths


def _collect_target_paths(provider: DatasetProvider) -> tuple[List[Path], int]:
    paths: List[Path] = []
    seen: Set[Path] = set()
    target_count = 0

    for target in provider.list_fetchable():
        if target.state in {
            TargetState.CACHED,
            TargetState.PROCESSED,
            TargetState.ARCHIVED,
        }:
            target_count += 1
        if target.path is None or not target.path.exists():
            continue
        resolved = target.path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(target.path)

    intake_paths = _scan_intake_cache(provider.intake_dir)
    for path in intake_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            paths.append(path)

    return paths, target_count


def _extra_module_paths(module_tag: str, data_dir: Path) -> tuple[List[Path], int]:
    if not data_dir.is_dir():
        return [], 0
    paths = _scan_intake_cache(data_dir)
    if not paths:
        return [], 0
    file_count = sum(1 for path in paths if path.is_file())
    dir_count = sum(1 for path in paths if path.is_dir())
    target_count = max(file_count + dir_count, 1)
    return paths, target_count


def list_modules_with_cache() -> List[ModuleCacheEntry]:
    """Return modules that currently have downloaded dataset files on disk."""
    entries: List[ModuleCacheEntry] = []
    covered_dirs: Set[Path] = set()

    for provider in list_modules():
        paths, target_count = _collect_target_paths(provider)
        if provider.intake_dir.is_dir():
            covered_dirs.add(provider.intake_dir.resolve())
        if not paths:
            continue
        total_bytes = sum(_path_size(path) for path in paths)
        entries.append(
            ModuleCacheEntry(
                module_tag=provider.module_tag,
                title=provider.title,
                target_count=target_count,
                total_bytes=total_bytes,
                paths=paths,
            )
        )

    for module_tag, (title, data_dir) in _EXTRA_CACHE_MODULES.items():
        if any(entry.module_tag == module_tag for entry in entries):
            continue
        if data_dir.resolve() in covered_dirs:
            continue
        paths, target_count = _extra_module_paths(module_tag, data_dir)
        if not paths:
            continue
        total_bytes = sum(_path_size(path) for path in paths)
        entries.append(
            ModuleCacheEntry(
                module_tag=module_tag,
                title=title,
                target_count=target_count,
                total_bytes=total_bytes,
                paths=paths,
            )
        )

    entries.sort(key=lambda entry: entry.title.lower())
    return entries


def _delete_path(path: Path) -> int:
    if not path.exists():
        return 0
    size = _path_size(path)
    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    return size


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    ordered: List[Path] = []
    seen: Set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def clear_module_cache(module_tag: str) -> ClearCacheResult:
    """Delete cached dataset files for one module; ledgers and processed flags unchanged."""
    provider = get_provider(module_tag)
    extra = _EXTRA_CACHE_MODULES.get(module_tag)
    title = provider.title if provider is not None else (extra[0] if extra else module_tag)

    if provider is not None:
        paths, _ = _collect_target_paths(provider)
    elif extra is not None:
        paths, _ = _extra_module_paths(module_tag, extra[1])
    else:
        raise KeyError(f"Unknown module tag: {module_tag!r}")

    result = ClearCacheResult(module_tag=module_tag, title=title)

    for path in _unique_paths(paths):
        if _is_cache_ledger(path):
            continue
        try:
            freed = _delete_path(path)
            result.deleted_paths.append(str(path))
            result.bytes_freed += freed
            print(f"[CLEAR CACHE] Removed {path}")
        except OSError as exc:
            msg = f"{path}: {exc}"
            result.errors.append(msg)
            print(f"[CLEAR CACHE] Failed to remove {path}: {exc}")

    print(
        f"[CLEAR CACHE] {title} ({module_tag}): "
        f"removed {len(result.deleted_paths)} path(s), "
        f"freed {format_size(result.bytes_freed)}"
    )
    return result