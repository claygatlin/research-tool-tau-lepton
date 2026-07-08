"""
Modular bridge: Remote AI Processing ↔ research_tool.py

Actions:
  - Test API Connections
  - Analyze Artifact / Report
  - Custom Data Processing Prompt
  - Provider Configuration Status
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from tav_shared.llm_analysis import analyze_run_output, parse_provider_list
from tav_shared.remote_ai.api_registry import provider_status
from tav_shared.remote_ai.connection_test import test_api_connections
from tav_shared.remote_ai.load_keys import load_api_keys
from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

MODULE_TAG = "REMOTE_AI"

MENU_ACTIONS = [
    "Test API Connections",
    "Analyze Artifact / Report",
    "Custom Data Processing Prompt",
    "Provider Configuration Status",
]

_ACTION_KEYS: Dict[str, str] = {
    "Test API Connections": "test_connections",
    "Analyze Artifact / Report": "analyze_artifact",
    "Custom Data Processing Prompt": "custom_prompt",
    "Provider Configuration Status": "provider_status",
}

ARTIFACTS_DIR = TAU_SUPERBLOCK_ROOT / "artifacts" / "remote_ai"


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def entry_fields(action: str) -> list[dict]:
    provider_field = {
        "key": "llm_providers",
        "label": "LLM providers",
        "default": "all",
        "required": False,
        "hint": "all | grok,gemini,openai,claude,openrouter,ollama,nvidia_nim",
    }
    if action == "Test API Connections":
        return [provider_field]
    if action == "Analyze Artifact / Report":
        return [
            provider_field,
            {
                "key": "artifact_path",
                "label": "Artifact path (JSON, log, .txt)",
                "default": "artifacts/prime_past/best_tav_bbn_config.json",
                "required": True,
                "hint": "Relative to TauSuperblock root or absolute path",
            },
            {
                "key": "analysis_focus",
                "label": "Analysis focus (optional)",
                "default": "scientific review",
                "required": False,
            },
        ]
    if action == "Custom Data Processing Prompt":
        return [
            provider_field,
            {
                "key": "input_path",
                "label": "Input data file (optional)",
                "default": "",
                "required": False,
                "hint": "CSV/JSON/log excerpt appended to your prompt",
            },
            {
                "key": "custom_prompt",
                "label": "Processing instruction",
                "default": "Summarize tensions and suggest next experiments.",
                "required": True,
            },
        ]
    return []


def entry_instructions(action: str) -> list[str]:
    base = [
        "Remote AI — Grok, Gemini, OpenAI, Claude, OpenRouter, Ollama, NVIDIA NIM.",
        "Keys: config/api_keys.env or environment variables (see config/api_keys.env.example).",
        "Outputs: artifacts/remote_ai/",
    ]
    if action == "Test API Connections":
        base.append("Sends a minimal ping prompt to each configured provider.")
    if action == "Analyze Artifact / Report":
        base.append("Runs multi-provider verbose review on an existing artifact.")
    return base


def _resolve_path(raw: str) -> Path:
    path = Path(raw.strip())
    if not path.is_absolute():
        path = TAU_SUPERBLOCK_ROOT / path
    return path


def run_action(
    selection: str,
    show_plots: bool = True,
    options: dict | None = None,
) -> str | None:
    options = options or {}
    action = _ACTION_KEYS.get(selection)
    if action is None:
        print(f"[TAV ENGINE] Unknown Remote AI action: {selection}")
        return None

    load_api_keys()
    print(f"\n[TAV ENGINE] Remote AI Processing — {selection}")

    if action == "test_connections":
        providers = parse_provider_list(options)
        result = test_api_connections(providers=providers, verbose=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        out = ARTIFACTS_DIR / f"connection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[TAV ENGINE] Connection test saved: {out}")
        return str(out)

    if action == "provider_status":
        print("\n=== Remote AI Provider Status ===")
        for row in provider_status():
            mark = "configured" if row["configured"] == "yes" else "missing key"
            print(f"  {row['label']}: {mark}")
            print(f"    env: {row['env']}  model: {row['model']}")
        return None

    if action == "analyze_artifact":
        path = _resolve_path(options.get("artifact_path") or "")
        if not path.is_file():
            raise FileNotFoundError(f"Artifact not found: {path}")
        focus = (options.get("analysis_focus") or "scientific review").strip()
        options = {**options, "llm_analysis": "yes", "analysis_focus": focus}
        bundle = analyze_run_output(
            module_tag=MODULE_TAG,
            action=f"Analyze Artifact ({path.name})",
            log_path=path,
            report_paths=[path],
            options=options,
        )
        return bundle.get("report_path")

    if action == "custom_prompt":
        prompt = (options.get("custom_prompt") or "").strip()
        if not prompt:
            raise ValueError("custom_prompt is required")
        input_path = (options.get("input_path") or "").strip()
        context = ""
        if input_path:
            path = _resolve_path(input_path)
            if path.is_file():
                context = path.read_text(encoding="utf-8", errors="replace")
                context = context[:48000]
        full_prompt = (
            f"{prompt}\n\n--- INPUT DATA ---\n{context}"
            if context
            else prompt
        )
        from tav_shared.llm_analysis import query_all_providers

        providers = parse_provider_list(options)
        results = query_all_providers(full_prompt, providers=providers)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = ARTIFACTS_DIR / f"custom_prompt_{stamp}.json"
        payload = {
            "timestamp": stamp,
            "prompt": prompt,
            "input_path": input_path or None,
            "providers": [r.to_dict() for r in results],
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        for row in results:
            if row.analysis:
                print(f"\n--- {row.provider.upper()} ---\n{row.analysis[:2000]}")
        print(f"[TAV ENGINE] Custom prompt results: {out}")
        return str(out)

    return None