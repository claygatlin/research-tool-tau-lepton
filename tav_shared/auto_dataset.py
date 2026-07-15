"""
Auto dataset pipeline wired into every analysis run.

Stages (unchanged from Dataset Manager design):
  1. **pull**   — fetch from remote / git / wget
  2. **cache**  — store under ``datasets/<module>/``
  3. **store**  — ``mark_processed`` after successful fetch/run
  4. **archive** — ``delete_previous_archived_runs`` before moving to finishedA/

Mock/synthetic data is allowed **only** for explicit toy actions (TSB Research
``Generate Mocks``, Casimir ``Example`` scan). All other modules require live data.
"""

from __future__ import annotations

from typing import Any, Callable

from tav_shared.dataset_registry import fetch_selected, get_provider

# (module_tag, action) pairs that intentionally use synthetic data
MOCK_ONLY_ACTIONS: set[tuple[str, str]] = {
    ("TSB_RESEARCH", "Generate Mocks"),
    ("TSB_CASIMIR", "Example Casimir Tau-Search"),
}

_ModuleEnsure = Callable[[str, dict[str, Any]], None]

_MODULE_ENSURES: dict[str, _ModuleEnsure] = {}


def _register_ensures() -> None:
    if _MODULE_ENSURES:
        return

    def _desi(action: str, options: dict[str, Any]) -> None:
        if action == "Injection/Recovery Tests":
            return
        from menus.astronomical.desi.fetcher import ensure_desi_bao_data

        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        dr2 = ensure_desi_bao_data(force_refresh=force, auto_fetch=True)
        if not (options.get("cobaya_path") or "").strip():
            options["cobaya_path"] = str(dr2.parent)

    def _frb(action: str, options: dict[str, Any]) -> None:
        if action == "Pull Datasets from Open Archives":
            return
        from menus.astronomical.frb.fetcher import ensure_frb_datasets

        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        restore = str(options.get("restore_archived", "yes")).lower() in {"yes", "y", "true", "1"}
        print("[TAV ENGINE] Auto-fetch: FRB + void catalogs")
        ensure_frb_datasets(force_refresh=force, restore_archived=restore, auto_fetch=True)

    def _sparc(action: str, options: dict[str, Any]) -> None:
        if action in {"Pull Batch from Repository", "Pull Single Galaxy CSV"}:
            return
        galaxy = (options.get("galaxy_name") or options.get("query") or "").strip()
        if not galaxy:
            return
        from menus.astronomical.sparc.fetcher import ensure_galaxy_csv

        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        print(f"[TAV ENGINE] Auto-fetch: SPARC galaxy {galaxy}")
        ensure_galaxy_csv(galaxy, force_refresh=force)

    def _registry(module_tag: str) -> _ModuleEnsure:
        def _ensure(action: str, options: dict[str, Any]) -> None:
            provider = get_provider(module_tag)
            if provider is None:
                return
            targets = _default_registry_targets(module_tag, action, options)
            if not targets:
                return
            print(f"[TAV ENGINE] Auto-fetch via registry: {module_tag} ({len(targets)} target(s))")
            result = fetch_selected(module_tag, targets, options)
            if result.fetched:
                print(f"[TAV ENGINE] Fetched: {', '.join(result.fetched[:6])}")
            if result.failed:
                for key, msg in result.failed.items():
                    raise RuntimeError(f"Dataset fetch failed for {key}: {msg}")

        return _ensure

    def _cern(action: str, options: dict[str, Any]) -> None:
        if action == "Pull Datasets from Open Archives":
            return
        from menus.particle.cern.manifest import ACTION_DEFAULT_TARGETS

        targets = ACTION_DEFAULT_TARGETS.get(action, [])
        if not targets:
            return
        from menus.particle.cern.fetcher import pull_selected_targets

        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        restore = str(options.get("restore_archived", "yes")).lower() in {"yes", "y", "true", "1"}
        print(f"[TAV ENGINE] Auto-fetch: CERN Open Data ({', '.join(targets)})")
        result = pull_selected_targets(
            targets,
            params={"restore_archived": "yes" if restore else "no"},
            force_refresh=force,
        )
        if result.failed:
            for key, msg in result.failed.items():
                raise RuntimeError(f"CERN fetch failed for {key}: {msg}")

    _MODULE_ENSURES["TAU_SB_DESI"] = _desi
    _MODULE_ENSURES["FRB_COSMIC_WEB_TAV"] = _frb
    _MODULE_ENSURES["SPARC"] = _sparc
    _MODULE_ENSURES["CERN_OPENDATA"] = _cern
    for tag in ("EMPIRICAL_TESTS", "LHCB_TAV_ECHO", "TAV_DATA_INTEGRATOR", "PLANCK_CMB_TAV"):
        _MODULE_ENSURES[tag] = _registry(tag)


def _default_registry_targets(module_tag: str, action: str, options: dict[str, Any]) -> list[str]:
    provider = get_provider(module_tag)
    if provider is None:
        return []
    remote = [t.id for t in provider.list_fetchable() if t.state.name in {"REMOTE", "ARCHIVED"}]
    if module_tag == "EMPIRICAL_TESTS":
        try:
            from menus.empirical_tests.extension import _ACTION_CONFIG

            sources = (_ACTION_CONFIG.get(action) or {}).get("sources") or []
            if sources:
                return [f"empirical:{key}" for key in sources]
        except ImportError:
            pass
        return remote[:3] if remote else []
    if module_tag == "LHCB_TAV_ECHO":
        return [t.id for t in provider.list_fetchable() if "lhcb:" in t.id][:1]
    if module_tag == "CERN_OPENDATA":
        from menus.particle.cern.manifest import ACTION_DEFAULT_TARGETS

        return [f"cern:{key}" for key in ACTION_DEFAULT_TARGETS.get(action, [])]
    return remote


def ensure_datasets_before_run(
    module_tag: str,
    action: str,
    options: dict[str, Any] | None = None,
) -> None:
    """
    Pull required datasets before ``run_action``. Raises on missing live data.
    """
    if options is None:
        options = {}
    if (module_tag, action) in MOCK_ONLY_ACTIONS:
        print(f"[TAV ENGINE] Skipping live-data fetch for mock/toy action: {action}")
        return

    _register_ensures()
    ensure_fn = _MODULE_ENSURES.get(module_tag)
    if ensure_fn is None:
        ext_hook = options.pop("_extension_ensure", None)
        if callable(ext_hook):
            ext_hook(action, options)
        return

    print(f"[TAV ENGINE] Dataset pipeline — pull→cache for {module_tag} / {action}")
    ensure_fn(action, options)