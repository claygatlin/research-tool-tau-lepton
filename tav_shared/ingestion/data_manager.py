"""Simple data fetch/cache helper for ingestion workflows."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

# Keep existing default log destination while avoiding duplicate root handlers.
logging.basicConfig(
    level=logging.INFO,
    filename="research_tool.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ResearchTool")

DATA_MANAGER_PARAM_KEY = "data_manager"
DATA_MANAGER_LEGACY_PARAM_KEY = "__data_manager__"

_RUNTIME_DATA_MANAGER: "DataManager | None" = None
_RUNTIME_CONFIG: tuple[bool, str] | None = None


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _runtime_config_from_params(
    params: dict[str, Any] | None,
    *,
    use_mock: bool | None,
    cache_dir: str | os.PathLike[str] | None,
) -> tuple[bool, str]:
    options = params or {}
    env_use_mock = os.getenv("TAV_USE_MOCK_DATA")
    env_cache_dir = os.getenv("TAV_CACHE_DIR")

    resolved_use_mock = _to_bool(
        use_mock if use_mock is not None else options.get("use_mock", env_use_mock),
        default=False,
    )
    raw_cache = cache_dir or options.get("cache_dir") or env_cache_dir or "./cache"
    resolved_cache = str(Path(raw_cache).expanduser().resolve())
    return resolved_use_mock, resolved_cache


class DataManager:
    """Fetch data from cache, mock generator, or source files."""

    def __init__(self, use_mock: bool = False, cache_dir: str | os.PathLike[str] = "./cache"):
        self.use_mock = bool(use_mock)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized DataManager (Mode: %s)", "Mock" if self.use_mock else "Real")

    def fetch(
        self,
        data_key: str,
        source_path: str | os.PathLike[str] | None = None,
        *,
        force_refresh: bool = False,
    ) -> Any:
        """Load data from cache when possible, otherwise from mock/source."""
        cache_file = self.cache_dir / f"{str(data_key).strip()}.pkl"

        if cache_file.is_file() and not force_refresh:
            logger.info("Loading %s from cache.", data_key)
            with cache_file.open("rb") as fh:
                return pickle.load(fh)

        if self.use_mock:
            logger.warning("Generating mock data for %s", data_key)
            data = np.random.normal(0.0, 1.0, (10, 5))
        else:
            logger.info("Fetching real data from %s", source_path)
            data = self._load_from_source(source_path)

        with cache_file.open("wb") as fh:
            pickle.dump(data, fh)
        return data

    def _load_from_source(self, path: str | os.PathLike[str] | None) -> Any:
        """Load supported local formats (.txt/.dat/.csv/.tsv/.npy/.npz/.pkl/.root)."""
        if not path:
            raise FileNotFoundError("Source path not provided.")

        src = Path(path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Source path {src} not found.")

        suffix = src.suffix.lower()
        if suffix in {".txt", ".dat", ".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else "," if suffix == ".csv" else None
            try:
                return np.loadtxt(src, delimiter=delimiter)
            except Exception:
                # genfromtxt is more tolerant of sparse/missing values.
                return np.genfromtxt(src, delimiter=delimiter)

        if suffix == ".npy":
            return np.load(src, allow_pickle=False)

        if suffix == ".npz":
            with np.load(src, allow_pickle=False) as archive:
                keys = list(archive.keys())
                if len(keys) == 1:
                    return archive[keys[0]]
                return {key: archive[key] for key in keys}

        if suffix in {".pkl", ".pickle"}:
            with src.open("rb") as fh:
                return pickle.load(fh)

        if suffix == ".root":
            try:
                import uproot
            except ImportError as exc:
                raise ValueError(
                    "Reading .root files requires uproot. Install with: pip install uproot"
                ) from exc

            with uproot.open(src) as root_file:
                tree_keys = [key for key, val in root_file.items() if hasattr(val, "arrays")]
                if not tree_keys:
                    raise ValueError(f"No TTree found in ROOT file: {src}")
                tree = root_file[tree_keys[0]]
                return tree.arrays(library="np")

        raise ValueError(f"Unsupported source format: {suffix}")


def get_runtime_data_manager(
    params: dict[str, Any] | None = None,
    *,
    use_mock: bool | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> DataManager:
    """Return a shared runtime DataManager instance for all flow handlers."""
    global _RUNTIME_DATA_MANAGER, _RUNTIME_CONFIG

    cfg = _runtime_config_from_params(params, use_mock=use_mock, cache_dir=cache_dir)
    if _RUNTIME_DATA_MANAGER is None or _RUNTIME_CONFIG != cfg:
        _RUNTIME_DATA_MANAGER = DataManager(use_mock=cfg[0], cache_dir=cfg[1])
        _RUNTIME_CONFIG = cfg
    return _RUNTIME_DATA_MANAGER


def inject_data_manager(params: dict[str, Any] | None) -> dict[str, Any]:
    """Attach the shared runtime DataManager onto action params."""
    out = dict(params or {})
    manager = get_runtime_data_manager(out)
    out.setdefault(DATA_MANAGER_PARAM_KEY, manager)
    out.setdefault(DATA_MANAGER_LEGACY_PARAM_KEY, manager)
    return out