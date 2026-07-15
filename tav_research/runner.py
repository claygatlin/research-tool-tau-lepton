"""
Post-menu execution — run captured actions with optional LLM review.
"""

from __future__ import annotations

import curses
import os
import sys

import matplotlib

from tav_shared.auto_dataset import ensure_datasets_before_run
from tav_shared.run_output import capture_run_log
from tav_shared.llm_analysis import run_post_action_llm_review

from tav_research.data_pull import fetch_and_graph, is_batch_list_file, _iter_resumable_batch_lines
from tav_research.dataset_manager import DATASET_MANAGER_TAG, run_dataset_manager_action
from tav_research.curses_shell import restore_terminal
from tav_research.main_loop import main_curses
from tav_research.registry import MODULE_EXTENSIONS, PLANCK_IMPORT_ERROR, planck_cmb_extension

# Force Tkinter backend for pop-up graphing (same as legacy research_tool.py).
matplotlib.use("TkAgg")


def load_api_keys() -> None:
    """Populate LLM API key env vars from config/api_keys.env or .env."""
    from tav_shared.remote_ai.load_keys import load_api_keys as _load_remote_keys
    from tav_shared.tav_resonance_bootstrap import bootstrap_tav_resonance

    _load_remote_keys()
    bootstrap_tav_resonance(quiet=True)


def _inject_gateway_live_models(params: dict) -> dict:
    """Merge gateway live-model list from env into run/review options."""
    from tav_shared.llm_analysis import load_gateway_live_models

    out = dict(params)
    live = load_gateway_live_models()
    if live:
        out["gateway_live_models"] = live
    return out


def _prepare_module_run(module_tag: str, action: str, params: dict) -> tuple[dict, dict]:
    """Apply module-specific Tau defaults and build action-relevant log options."""
    ext = MODULE_EXTENSIONS.get(module_tag)
    prepared = params
    if ext is not None and hasattr(ext, "prepare_run_options"):
        prepared = ext.prepare_run_options(action, params)
    if ext is not None and hasattr(ext, "options_for_run_log"):
        log_options = ext.options_for_run_log(action, prepared)
    else:
        log_options = prepared
    return prepared, log_options


def _run_with_optional_llm_review(
    module_tag: str,
    action: str,
    params: dict,
    runner,
    *,
    log_options: dict | None = None,
) -> None:
    """Execute a captured run and optionally query LLM providers on the log."""
    review_params = _inject_gateway_live_models(params)
    with capture_run_log(module_tag, action, log_options or params) as log_path:
        result = runner()
    if isinstance(result, str) and result.strip():
        review_params["llm_report_paths"] = result.strip()
    run_post_action_llm_review(
        review_params,
        module_tag=module_tag,
        action=action,
        log_path=log_path,
    )


def main() -> None:
    load_api_keys()
    try:
        while True:
            repo, query, params = curses.wrapper(main_curses)
            params = params or {}
            if repo is None:
                break

            if repo == DATASET_MANAGER_TAG:
                params["dataset_manager_action"] = query
                _run_with_optional_llm_review(
                    "dataset_manager",
                    query,
                    params,
                    lambda: run_dataset_manager_action(params),
                )
            elif repo in MODULE_EXTENSIONS:
                prepared, log_options = _prepare_module_run(repo, query, params)
                ext = MODULE_EXTENSIONS[repo]
                ensure_datasets_before_run(repo, query, prepared)
                _run_with_optional_llm_review(
                    repo,
                    query,
                    prepared,
                    lambda: ext.run_action(query, options=prepared),
                    log_options=log_options,
                )
            elif repo == "Planck CMB Tav-Scan" and planck_cmb_extension is None:
                print("\n[TAV ENGINE] Planck CMB module unavailable.")
                if PLANCK_IMPORT_ERROR is not None:
                    print(f"[ERROR] Import failed: {PLANCK_IMPORT_ERROR}")
                print("[TAV ENGINE] Install healpy in the project venv:")
                print("  ./venv/bin/pip install healpy")
                print("[TAV ENGINE] Then run: ./venv/bin/python research_tool.py")
            elif repo and query:
                if os.path.isfile(query) and is_batch_list_file(query):
                    for b_query in _iter_resumable_batch_lines(query, params):
                        item_params = {**params, "_record_batch_done": b_query}
                        _run_with_optional_llm_review(
                            repo,
                            b_query,
                            item_params,
                            lambda q=b_query, p=item_params: fetch_and_graph(repo, q, p),
                        )
                else:
                    _run_with_optional_llm_review(
                        repo,
                        query,
                        params,
                        lambda: fetch_and_graph(repo, query, params),
                    )
            else:
                print("\n[TAV ENGINE] Extraction aborted.")
            break
    except Exception as exc:
        detail = str(exc).strip()
        label = type(exc).__name__
        print(f"Terminal Error: {detail or f'{label} (no message)'}")
        sys.exit(1)
    finally:
        restore_terminal()