"""CMS Open Data — programmatic fetch + native NanoAOD analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from menus.particle.cern.fetcher import DATASETS_DIR, list_cached_targets
from menus.particle.cern.manifest import CERN_MANIFEST
from tav_research.data_pull_common import COMMON_BATCH_HINT
from tav_shared.llm_analysis import append_llm_entry_fields

MENU_LABEL = "CMS Open Data (NanoAOD / Electrons)"

_PREFERRED_DEFAULT_KEY = "cms_nanoaod_dimu"


def discover_cms_open_data_targets(*, include_uncached: bool = False) -> list[dict[str, Any]]:
    """
    Discover CMS NanoAOD datasets under ``datasets/cern/``.

    Returns rows with ``key``, ``label``, ``path``, and ``cached`` for the
    entry-form picker and fetch resolver.
    """
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_paths: set[str] = set()

    for item in list_cached_targets():
        key = str(item.get("key") or "")
        spec = CERN_MANIFEST.get(key, {})
        if spec.get("analyze") != "cms_nanoaod":
            continue
        sample = item.get("sample_root")
        if item.get("cached") and sample:
            path = Path(sample)
            label = f"{key} — {path.name}"
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "path": str(path),
                    "cached": True,
                }
            )
            seen_keys.add(key)
            seen_paths.add(str(path.resolve()))
        elif include_uncached and key not in seen_keys:
            rows.append(
                {
                    "key": key,
                    "label": f"{key} — (auto-fetch from opendata.cern.ch)",
                    "path": None,
                    "cached": False,
                }
            )
            seen_keys.add(key)

    if DATASETS_DIR.is_dir():
        for root_file in sorted(DATASETS_DIR.rglob("*.root")):
            resolved = str(root_file.resolve())
            if resolved in seen_paths:
                continue
            try:
                rel = root_file.relative_to(DATASETS_DIR)
            except ValueError:
                rel = Path(root_file.name)
            manifest_key = rel.parts[0] if rel.parts else root_file.stem
            spec = CERN_MANIFEST.get(manifest_key, {})
            if manifest_key in CERN_MANIFEST and spec.get("analyze") != "cms_nanoaod":
                continue
            if manifest_key in seen_keys:
                continue
            rows.append(
                {
                    "key": manifest_key if manifest_key in CERN_MANIFEST else resolved,
                    "label": rel.as_posix(),
                    "path": resolved,
                    "cached": True,
                }
            )
            seen_paths.add(resolved)

    rows.sort(key=lambda row: (0 if row.get("cached") else 1, str(row.get("label") or "")))
    return rows


def _default_dataset_label(targets: list[dict[str, Any]]) -> str:
    if not targets:
        return _PREFERRED_DEFAULT_KEY
    for preferred in (_PREFERRED_DEFAULT_KEY, "cms_nanoaod_higgs_zz"):
        for row in targets:
            if row.get("key") == preferred:
                return str(row["label"])
    return str(targets[0]["label"])


def resolve_cms_open_data_query(query: str) -> str:
    """Map picker label, manifest key, recid, or path to a run_analysis target."""
    q = (query or "").strip()
    if not q:
        targets = discover_cms_open_data_targets()
        if targets:
            row = targets[0]
            return str(row["path"] or row["key"])
        return _PREFERRED_DEFAULT_KEY

    catalog = discover_cms_open_data_targets(include_uncached=True)
    for row in catalog:
        if q == row.get("label") or q == row.get("key"):
            return str(row.get("path") or row["key"])

    if " — " in q:
        head = q.split(" — ", 1)[0].strip()
        for row in catalog:
            if head == row.get("key"):
                return str(row.get("path") or row["key"])
        return head

    return q


def entry_instructions() -> list[str]:
    targets = discover_cms_open_data_targets()
    n_cached = sum(1 for row in targets if row.get("cached"))
    lines = [
        f"Auto-discovered {n_cached} cached CMS dataset(s) under datasets/cern/.",
        "Use UP/DOWN to pick a dataset; ENTER confirms.",
        "Uncached manifest keys auto-fetch from opendata.cern.ch before analysis.",
        "Uses CERN fetcher + uproot (native venv).",
        COMMON_BATCH_HINT,
    ]
    if targets:
        lines.append("Cached:")
        for row in targets[:6]:
            if row.get("cached"):
                lines.append(f"  • {row['label']}")
    return lines


def entry_fields() -> list[dict]:
    targets = discover_cms_open_data_targets(include_uncached=True)
    cached = [row for row in targets if row.get("cached")]
    picker_rows = cached or targets
    choices = [str(row["label"]) for row in picker_rows]
    if not choices:
        choices = [_PREFERRED_DEFAULT_KEY]

    default_label = _default_dataset_label(cached or picker_rows)
    n_cached = len(cached)

    return append_llm_entry_fields(
        [
            {
                "key": "query",
                "label": "CMS dataset (datasets/cern/)",
                "default": default_label,
                "required": False,
                "choices": choices,
                "hint": (
                    f"{n_cached} cached .root file(s) found — pick one or leave default"
                ),
            },
            {
                "key": "batch_file",
                "label": "Batch list file (optional)",
                "default": "",
                "required": False,
                "hint": "Plain-text .txt/.csv list; overrides single dataset when set",
            },
            {
                "key": "entry_stop",
                "label": "Max events",
                "default": "50000",
                "required": False,
                "hint": "Uproot entry limit (0 = full chunked scan via CERN menu)",
            },
            {
                "key": "force_refresh",
                "label": "Force refresh fetch",
                "default": "no",
                "required": False,
                "hint": "yes = re-download from CERN when target is not cached",
                "choices": ["no", "yes"],
            },
        ]
    )


def fetch_and_graph(query: str, params: dict[str, Any] | None = None) -> None:
    from menus.particle.cern.analyzer import run_analysis
    from menus.particle.cern.fetcher import ensure_cern_target, resolve_root_path
    from menus.particle.cern.manifest import resolve_target_key

    params = params or {}
    q = resolve_cms_open_data_query(query)
    force = str(params.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
    try:
        entry_stop = int(params.get("entry_stop") or 50000)
    except ValueError:
        entry_stop = 50000

    key = resolve_target_key(q)
    if key and resolve_root_path(key) is None:
        print(f"[TAV ENGINE] Auto-fetching CERN target: {key}")
        ensure_cern_target(key, force_refresh=force, auto_fetch=True)

    if key:
        q = key

    print(f"[TAV ENGINE] CMS dataset resolved: {q}")
    report = run_analysis(
        q,
        analysis="cms_nanoaod",
        entry_stop=entry_stop,
        output_prefix="cms_open_data",
    )
    print(f"[TAV ENGINE] CMS Open Data report: {report.get('report_path')}")