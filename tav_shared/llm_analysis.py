#!/usr/bin/env python3
"""
Multi-provider verbose analysis of Tau-Superblock module run output.

Queries Grok (xAI), Gemini (Google), ChatGPT (OpenAI), and Claude (Anthropic)
when the corresponding API keys are set, then saves combined reports under
artifacts/.

Environment variables
---------------------
XAI_API_KEY       — Grok / xAI
GEMINI_API_KEY    — Google Gemini (GOOGLE_API_KEY also accepted)
OPENAI_API_KEY    — ChatGPT / OpenAI
ANTHROPIC_API_KEY — Claude / Anthropic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from tav_shared.run_output import ARTIFACTS_DIR, write_test_output

MAX_OUTPUT_CHARS = 48_000
MAX_REPORT_CHARS = 24_000
REQUEST_TIMEOUT_S = 120

PROVIDER_ALIASES = {
    "grok": "grok",
    "xai": "grok",
    "gemini": "gemini",
    "google": "gemini",
    "openai": "openai",
    "chatgpt": "openai",
    "gpt": "openai",
    "claude": "claude",
    "anthropic": "claude",
    "openrouter": "openrouter",
    "ollama": "ollama",
    "nvidia": "nvidia_nim",
    "nvidia_nim": "nvidia_nim",
    "nim": "nvidia_nim",
    "nvapi": "nvidia_nim",
    "minimax": "nvidia_nim",
}

DEFAULT_MODELS = {
    "grok": "grok-3-mini",
    "gemini": "gemini-flash-latest",
    "openai": "gpt-4o-mini",
    "claude": "claude-sonnet-4-6",
}


@dataclass
class ProviderResult:
    provider: str
    model: str
    analysis: str = ""
    error: str | None = None
    latency_s: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisBundle:
    module_tag: str
    action: str
    timestamp: str
    prompt_chars: int
    providers: list[ProviderResult] = field(default_factory=list)
    report_path: str | None = None
    text_report_path: str | None = None
    ade_comparative_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_tag": self.module_tag,
            "action": self.action,
            "timestamp": self.timestamp,
            "prompt_chars": self.prompt_chars,
            "providers": [p.to_dict() for p in self.providers],
            "report_path": self.report_path,
            "text_report_path": self.text_report_path,
            "ade_comparative_summary": self.ade_comparative_summary,
        }


def llm_analysis_entry_fields() -> list[dict]:
    """Form fields for research_tool.py module entry screens."""
    return [
        {
            "key": "llm_analysis",
            "label": "AI verbose review",
            "default": "no",
            "required": False,
            "hint": "Query Grok, Gemini, ChatGPT, Claude on run output (needs API keys)",
            "choices": ["no", "yes"],
        },
        {
            "key": "llm_providers",
            "label": "LLM providers",
            "default": "all",
            "required": False,
            "hint": "all | grok,gemini,openai,claude,openrouter,ollama,nvidia_nim",
        },
    ]


def append_llm_entry_fields(fields: list[dict]) -> list[dict]:
    """Append AI review fields without duplicating keys."""
    existing = {f.get("key") for f in fields}
    return fields + [f for f in llm_analysis_entry_fields() if f.get("key") not in existing]


def should_run_llm_analysis(options: dict | None) -> bool:
    raw = (options or {}).get("llm_analysis", "")
    return str(raw).strip().lower() in {"yes", "y", "true", "1", "on"}


def parse_provider_list(options: dict | None) -> list[str]:
    raw = str((options or {}).get("llm_providers") or "all").strip().lower()
    if not raw or raw in {"all", "*"}:
        return ["grok", "gemini", "openai", "claude", "openrouter", "ollama", "nvidia_nim"]
    selected: list[str] = []
    for token in raw.replace(";", ",").split(","):
        key = PROVIDER_ALIASES.get(token.strip().lower())
        if key and key not in selected:
            selected.append(key)
    return selected or ["grok", "gemini", "openai", "claude", "openrouter", "ollama", "nvidia_nim"]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n...[truncated {len(text) - limit} characters]..."


def _read_report_excerpts(report_paths: Iterable[str | Path] | None) -> str:
    if not report_paths:
        return ""
    chunks: list[str] = []
    budget = MAX_REPORT_CHARS
    for raw in report_paths:
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                text = json.dumps(payload, indent=2, sort_keys=True)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, json.JSONDecodeError) as exc:
            chunks.append(f"--- {path.name} (read error: {exc}) ---")
            continue
        piece = f"--- {path.name} ---\n{text}"
        if len(piece) > budget:
            piece = _truncate(piece, budget)
            budget = 0
        else:
            budget -= len(piece)
        chunks.append(piece)
        if budget <= 0:
            break
    return "\n\n".join(chunks)


def _model_row_by_name(ranking: list[dict[str, Any]], needle: str) -> dict[str, Any] | None:
    needle_l = needle.lower()
    for row in ranking:
        if needle_l in str(row.get("name", "")).lower():
            return row
    return None


def _ade_param_labels(n_params: int) -> list[str]:
    deg = max(0, int(n_params) - 5)
    return [f"poly_c{i}" for i in range(deg)] + [
        "ade_amplitude_frac",
        "ade_omega",
        "ade_phase_rad",
        "ade_z_star",
    ]


def evaluate_ade_comparative_results_from_payload(payload: dict[str, Any]) -> str:
    """
    Summarize ΛCDM vs aDE vs Tau-SB model comparison for LLM review.

    Expects a Tau-SB DESI JSON report with a ``model_comparison`` block produced
    by :meth:`TauSBScanner.compare_models`.
    """
    mc = payload.get("model_comparison")
    if not isinstance(mc, dict):
        return ""

    ranking = mc.get("ranking") or []
    if not ranking:
        return ""

    ade = _model_row_by_name(ranking, "ade")
    lcdm = _model_row_by_name(ranking, "lcdm") or _model_row_by_name(ranking, "ΛCDM")
    tau = _model_row_by_name(ranking, "tau-sb")
    if ade is None:
        return ""

    n_data = int(payload.get("n_data") or ade.get("n_data") or 0)
    best_model = str(mc.get("best_model", ranking[0].get("name", "unknown")))
    delta_aic = mc.get("delta_aic") or {}
    delta_bic = mc.get("delta_bic") or {}
    bayes = mc.get("bayes_factors_vs_tau_sb") or {}

    lines = [
        "=== aDE COMPARATIVE MODEL EVALUATION (pre-computed) ===",
        "Three-way Gaussian χ² fit on the loaded BAO vector:",
        "  • ΛCDM — polynomial distance–redshift baseline only",
        "  • aDE — poly baseline + toy ultralight axion+Λ modulation "
        "(envelope×cos(ωz+φ) in z-space)",
        "  • Tau-SB — poly baseline + log-periodic 1/7 track in s-space + hierarchical step",
        "",
        f"Dataset: {payload.get('data_label', 'n/a')} | n_data={n_data}",
        f"AIC-ranked best model: {best_model}",
        "",
        "Information criteria (lower is better):",
    ]

    for row in ranking:
        name = str(row.get("name", "model"))
        chi2 = float(row.get("chi2", float("nan")))
        k = int(row.get("n_params", 0))
        dof = max(n_data - k, 1) if n_data else 0
        red = chi2 / dof if dof else float("nan")
        daic = float(delta_aic.get(name, float("nan")))
        dbic = float(delta_bic.get(name, float("nan"))) if delta_bic else float("nan")
        lines.append(
            f"  {name}: χ²={chi2:.2f}, reduced χ²={red:.2f}, "
            f"k={k}, AIC={float(row.get('aic', float('nan'))):.2f}, "
            f"BIC={float(row.get('bic', float('nan'))):.2f}, "
            f"ΔAIC={daic:.2f}"
            + (f", ΔBIC={dbic:.2f}" if math.isfinite(dbic) else "")
        )

    if tau is not None:
        tau_name = str(tau.get("name"))
        ade_name = str(ade.get("name"))
        d_aic_tau_ade = float(delta_aic.get(tau_name, float("nan"))) - float(
            delta_aic.get(ade_name, 0.0)
        )
        bf_vs_ade = float(bayes.get("vs_ade", float("nan")))
        lines.extend(
            [
                "",
                "Tau-SB vs aDE head-to-head:",
                f"  ΔAIC(Tau-SB − aDE) = {d_aic_tau_ade:+.2f} "
                f"({'Tau-SB preferred' if d_aic_tau_ade < -2 else 'aDE preferred' if d_aic_tau_ade > 2 else 'inconclusive'})",
                f"  BF(Tau-SB/aDE) from ΔBIC ≈ {bf_vs_ade:.4g} "
                f"({'>1 favors Tau-SB' if bf_vs_ade > 1 else '<1 favors aDE' if bf_vs_ade < 1 else 'neutral'})",
            ]
        )

    params = ade.get("params") or []
    if params:
        labels = _ade_param_labels(int(ade.get("n_params", len(params))))
        lines.append("")
        lines.append("aDE best-fit oscillation parameters (after poly baseline):")
        for label, val in zip(labels[len(labels) - 4 :], params[len(params) - 4 :], strict=False):
            lines.append(f"  {label} = {float(val):.6g}")

    lines.extend(
        [
            "",
            "Reviewer focus for aDE:",
            "  1. Does aDE win only by extra flexibility (k=+2 vs ΛCDM) or by meaningful residual structure?",
            "  2. Is the aDE modulation cosmologically interpretable vs a Tau-SB 1/7 prior?",
            "  3. Are reduced χ² values near unity (single-channel) or inflated (mixed DM/DH vector)?",
            "  4. Would nested sampling or cross-validation favor the simpler model?",
        ]
    )
    return "\n".join(lines)


def evaluate_ade_comparative_results(
    report_paths: Iterable[str | Path] | None = None,
    *,
    payload: dict[str, Any] | None = None,
) -> str:
    """
    Load Tau-SB DESI report JSON and return an aDE comparison block for LLM prompts.
    """
    if payload is not None:
        return evaluate_ade_comparative_results_from_payload(payload)

    for raw in report_paths or []:
        path = Path(raw)
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("model_comparison") or "tau_sb_desi" in path.name.lower():
            block = evaluate_ade_comparative_results_from_payload(data)
            if block:
                return block
    return ""


def _module_context_notes(module_tag: str, action: str) -> str:
    tag = (module_tag or "").upper()
    if tag == "TAU_SB_DESI":
        return (
            "Context for DESI runs:\n"
            "- Data are live DESI DR2 Gaussian BAO tables from the CobayaSampler/bao_data repo "
            "(desi_bao_dr2/), not mock/example data.\n"
            "- The action name 'Cobaya' refers to that data repository path, NOT the Cobaya MCMC sampler.\n"
            "- mcmc_steps appears only when pipelines_enabled includes emcee_mcmc; otherwise MCMC did not run.\n"
            "- Two separate 1/7 checks may appear: Lomb-Scargle in s-space and tav-resonance on residuals.\n"
            "- M₀=313.1 MeV and n_hier≈45.8 are fixed framework constants, not fitted outputs.\n"
            "- model_compare fits ΛCDM (poly), aDE (axion+Λ z-space modulation), and Tau-SB; "
            "see the aDE COMPARATIVE MODEL EVALUATION block when present.\n"
        )
    return ""


def build_analysis_prompt(
    module_tag: str,
    action: str,
    output_text: str,
    *,
    report_excerpts: str = "",
    run_options: str = "",
    ade_comparison: str = "",
) -> str:
    body = _truncate(output_text.strip(), MAX_OUTPUT_CHARS)
    reports = _truncate(report_excerpts.strip(), MAX_REPORT_CHARS) if report_excerpts else "(none)"
    options_block = run_options.strip() or "(see log header)"
    context = _module_context_notes(module_tag, action)
    header = (
        "You are a senior physics and data-analysis reviewer for the Tau-Superblock "
        "(Tav Topology / Tau cylinder) research framework.\n\n"
        "Analyze the module run output below in **verbose** detail. Structure your response with:\n"
        "1. Executive summary (3–6 sentences)\n"
        "2. Key numerical findings and units\n"
        "3. Consistency with Tau-SB hypotheses (1/7 tracks, hierarchical binding, mass-gap anchor)\n"
        "4. Statistical rigor and limitations\n"
        "5. Data-quality caveats and systematic risks\n"
        "6. Concrete follow-up experiments or diagnostics\n"
        "7. Overall evidence verdict (weak / moderate / strong) with justification\n"
    )
    if ade_comparison.strip():
        header += (
            "8. **aDE comparative verdict** — ΛCDM vs aDE (axion+Λ) vs Tau-SB: "
            "which model is statistically preferred, whether aDE's extra parameters "
            "are justified, and implications for dynamical dark energy vs Tau-SB 1/7 tracks\n"
        )
    header += "\n"

    parts = [
        header,
        f"Module: {module_tag}\n",
        f"Action: {action}\n",
    ]
    if context:
        parts.append(context)
    if ade_comparison.strip():
        parts.append(ade_comparison.strip() + "\n\n")
    parts.extend(
        [
            "=== RUN OPTIONS (action-relevant only) ===\n"
            f"{options_block}\n\n"
            "=== TERMINAL / LOG OUTPUT ===\n"
            f"{body}\n\n"
            "=== ATTACHED REPORT EXCERPTS ===\n"
            f"{reports}\n",
        ]
    )
    return "".join(parts)


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    provider: str,
) -> dict[str, Any]:
    response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S)
    if response.status_code >= 400:
        detail = response.text[:1200]
        raise RuntimeError(f"{provider} HTTP {response.status_code}: {detail}")
    return response.json()


def query_grok(prompt: str, *, model: str | None = None) -> ProviderResult:
    provider = "grok"
    model = model or DEFAULT_MODELS[provider]
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        return ProviderResult(
            provider=provider,
            model=model,
            skipped=True,
            skip_reason="XAI_API_KEY not set",
        )
    started = time.monotonic()
    try:
        data = _post_json(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert scientific reviewer."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            provider=provider,
        )
        text = data["choices"][0]["message"]["content"]
        return ProviderResult(
            provider=provider,
            model=model,
            analysis=str(text).strip(),
            latency_s=time.monotonic() - started,
        )
    except Exception as exc:
        return ProviderResult(
            provider=provider,
            model=model,
            error=str(exc),
            latency_s=time.monotonic() - started,
        )


def query_gemini(prompt: str, *, model: str | None = None) -> ProviderResult:
    provider = "gemini"
    model = model or DEFAULT_MODELS[provider]
    api_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not api_key:
        return ProviderResult(
            provider=provider,
            model=model,
            skipped=True,
            skip_reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
        )
    started = time.monotonic()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )
    try:
        data = _post_json(
            url,
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": api_key,
            },
            payload={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
            provider=provider,
        )
        candidates = data.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "\n".join(str(part.get("text", "")) for part in parts).strip()
        if not text:
            raise RuntimeError(f"empty Gemini response: {json.dumps(data)[:800]}")
        return ProviderResult(
            provider=provider,
            model=model,
            analysis=text,
            latency_s=time.monotonic() - started,
        )
    except Exception as exc:
        return ProviderResult(
            provider=provider,
            model=model,
            error=str(exc),
            latency_s=time.monotonic() - started,
        )


def query_openai(prompt: str, *, model: str | None = None) -> ProviderResult:
    provider = "openai"
    model = model or DEFAULT_MODELS[provider]
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return ProviderResult(
            provider=provider,
            model=model,
            skipped=True,
            skip_reason="OPENAI_API_KEY not set",
        )
    started = time.monotonic()
    try:
        data = _post_json(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert scientific reviewer."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            provider=provider,
        )
        text = data["choices"][0]["message"]["content"]
        return ProviderResult(
            provider=provider,
            model=model,
            analysis=str(text).strip(),
            latency_s=time.monotonic() - started,
        )
    except Exception as exc:
        return ProviderResult(
            provider=provider,
            model=model,
            error=str(exc),
            latency_s=time.monotonic() - started,
        )


def query_claude(prompt: str, *, model: str | None = None) -> ProviderResult:
    provider = "claude"
    model = model or DEFAULT_MODELS[provider]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return ProviderResult(
            provider=provider,
            model=model,
            skipped=True,
            skip_reason="ANTHROPIC_API_KEY not set",
        )
    started = time.monotonic()
    try:
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload={
                "model": model,
                "max_tokens": 4096,
                "temperature": 0.2,
                "system": "You are an expert scientific reviewer.",
                "messages": [{"role": "user", "content": prompt}],
            },
            provider=provider,
        )
        blocks = data.get("content") or []
        text = "\n".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if not text:
            raise RuntimeError(f"empty Claude response: {json.dumps(data)[:800]}")
        return ProviderResult(
            provider=provider,
            model=model,
            analysis=text,
            latency_s=time.monotonic() - started,
        )
    except Exception as exc:
        return ProviderResult(
            provider=provider,
            model=model,
            error=str(exc),
            latency_s=time.monotonic() - started,
        )


_PROVIDER_FN = {
    "grok": query_grok,
    "gemini": query_gemini,
    "openai": query_openai,
    "claude": query_claude,
}


def _extended_provider_fn(name: str):
    """Resolve built-in and registry-backed provider query functions."""
    fn = _PROVIDER_FN.get(name)
    if fn is not None:
        return fn
    try:
        from tav_shared.remote_ai.api_registry import get_provider

        spec = get_provider(name)
        if spec is not None:
            return spec.query_fn
    except ImportError:
        pass
    return None


def query_all_providers(
    prompt: str,
    *,
    providers: list[str] | None = None,
    models: dict[str, str] | None = None,
) -> list[ProviderResult]:
    """Query each requested provider; skipped providers return skip_reason."""
    want = providers or list(_PROVIDER_FN)
    models = models or {}
    results: list[ProviderResult] = []
    for name in want:
        fn = _extended_provider_fn(name)
        if fn is None:
            results.append(
                ProviderResult(
                    provider=name,
                    model="",
                    skipped=True,
                    skip_reason=f"unknown provider '{name}'",
                )
            )
            continue
        results.append(fn(prompt, model=models.get(name)))
    return results


def analyze_run_output(
    *,
    module_tag: str,
    action: str,
    output_text: str = "",
    log_path: str | Path | None = None,
    report_paths: list[str | Path] | None = None,
    options: dict | None = None,
) -> dict[str, Any]:
    """
    Query Grok, Gemini, ChatGPT, and Claude with run output; save verbose analyses.

    Returns a dict with JSON/text report paths and per-provider results.
    """
    options = options or {}
    if log_path and not output_text.strip():
        path = Path(log_path)
        if path.is_file():
            output_text = path.read_text(encoding="utf-8", errors="replace")

    if not output_text.strip():
        raise ValueError("No output text or log_path provided for LLM analysis")

    report_excerpts = _read_report_excerpts(report_paths)
    run_options = str(options.get("run_options_for_prompt") or _format_run_options_for_prompt(options))
    ade_comparison = ""
    if str(module_tag).upper() == "TAU_SB_DESI":
        ade_comparison = evaluate_ade_comparative_results(report_paths)
    prompt = build_analysis_prompt(
        module_tag,
        action,
        output_text,
        report_excerpts=report_excerpts,
        run_options=run_options,
        ade_comparison=ade_comparison,
    )
    providers = parse_provider_list(options)

    print(f"\n[TAV ENGINE] LLM verbose analysis — {module_tag} / {action}")
    print(f"[TAV ENGINE] Providers: {', '.join(providers)} | prompt chars: {len(prompt)}")

    results = query_all_providers(prompt, providers=providers)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle = AnalysisBundle(
        module_tag=module_tag,
        action=action,
        timestamp=stamp,
        prompt_chars=len(prompt),
        providers=results,
        ade_comparative_summary=ade_comparison or None,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch if ch.isalnum() else "_" for ch in f"{module_tag}_{action}".lower()).strip("_")
    json_path = ARTIFACTS_DIR / f"llm_analysis_{slug}_{stamp}.json"
    from menus.astronomical.desi.json_util import write_json

    write_json(json_path, bundle.to_dict(), indent=2, sort_keys=True)
    bundle.report_path = str(json_path)

    text_lines = [
        "=== TAV-SUPERBLOCK LLM VERBOSE ANALYSIS ===",
        f"Module: {module_tag}",
        f"Action: {action}",
        f"Timestamp (UTC): {stamp}",
        "",
    ]
    for row in results:
        text_lines.append("=" * 72)
        label = row.provider.upper()
        if row.skipped:
            text_lines.append(f"## {label} — SKIPPED ({row.skip_reason})")
            text_lines.append("")
            continue
        if row.error:
            text_lines.append(f"## {label} — ERROR ({row.model})")
            text_lines.append(row.error)
            text_lines.append("")
            continue
        text_lines.append(f"## {label} — {row.model} ({row.latency_s:.1f}s)")
        text_lines.append(row.analysis)
        text_lines.append("")

    text_path = write_test_output(f"llm_analysis_{slug}", "\n".join(text_lines))
    bundle.text_report_path = str(text_path)

    ok = [r for r in results if r.analysis and not r.error and not r.skipped]
    skipped = [r for r in results if r.skipped]
    failed = [r for r in results if r.error]
    print(
        f"[TAV ENGINE] LLM analysis complete: {len(ok)} ok, "
        f"{len(skipped)} skipped, {len(failed)} failed"
    )
    for row in results:
        if row.analysis and not row.error:
            preview = row.analysis.splitlines()[0][:100]
            print(f"  {row.provider}: {preview}...")
        elif row.skipped:
            print(f"  {row.provider}: skipped ({row.skip_reason})")
        elif row.error:
            print(f"  {row.provider}: error ({row.error[:120]})")
    print(f"[TAV ENGINE] LLM JSON report: {bundle.report_path}")
    print(f"[TAV ENGINE] LLM text report: {bundle.text_report_path}")

    return bundle.to_dict()


def maybe_analyze_with_llms(
    options: dict | None,
    *,
    module_tag: str,
    action: str,
    output_text: str = "",
    log_path: str | Path | None = None,
    report_paths: list[str | Path] | None = None,
) -> dict[str, Any] | None:
    """Run multi-LLM analysis when options['llm_analysis'] is enabled."""
    if not should_run_llm_analysis(options):
        return None
    try:
        return analyze_run_output(
            module_tag=module_tag,
            action=action,
            output_text=output_text,
            log_path=log_path,
            report_paths=report_paths,
            options=options,
        )
    except Exception as exc:
        print(f"[TAV ENGINE] LLM analysis failed: {exc}")
        return None


# Public alias for extensions importing a single entry point.
query_llm_analysis = analyze_run_output


def _discover_report_paths(options: dict | None, log_path: str | Path | None = None) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()

    def _add(path_like: str | Path | None) -> None:
        if not path_like:
            return
        path = Path(path_like)
        key = str(path)
        if key in seen:
            return
        if path.is_file():
            seen.add(key)
            paths.append(path)

    raw = (options or {}).get("report_paths") or (options or {}).get("llm_report_paths") or ""
    for token in str(raw).replace(";", ",").split(","):
        _add(token.strip())

    if log_path:
        try:
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("report:"):
                _add(stripped.split(":", 1)[1].strip())

    return paths


def _format_run_options_for_prompt(options: dict | None) -> str:
    if not options:
        return ""
    skip = {"report_paths", "llm_report_paths", "dataset_manager_mode", "module_tag", "target_ids"}
    lines: list[str] = []
    for key, value in sorted(options.items()):
        if key in skip:
            continue
        text = str(value).strip()
        if text:
            lines.append(f"  {key}: {text}")
    return "\n".join(lines)


def run_post_action_llm_review(
    options: dict | None,
    *,
    module_tag: str,
    action: str,
    log_path: str | Path | None,
) -> dict[str, Any] | None:
    """Hook for research_tool.py after capture_run_log completes."""
    report_paths = _discover_report_paths(options, log_path)
    prompt_options = dict(options or {})
    prompt_options["run_options_for_prompt"] = _format_run_options_for_prompt(options)
    return maybe_analyze_with_llms(
        prompt_options,
        module_tag=module_tag,
        action=action,
        log_path=log_path,
        report_paths=report_paths,
    )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-LLM verbose analysis of Tau-SB run output")
    parser.add_argument("--module", required=True, help="Module tag, e.g. SPARC")
    parser.add_argument("--action", required=True, help="Action name")
    parser.add_argument("--log", help="Path to captured run log")
    parser.add_argument("--text", help="Inline output text (instead of --log)")
    parser.add_argument("--report", action="append", default=[], help="Report file to attach (repeatable)")
    parser.add_argument(
        "--providers",
        default="all",
        help="Comma-separated: grok,gemini,openai,claude or all",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    options = {"llm_providers": args.providers, "llm_analysis": "yes"}
    analyze_run_output(
        module_tag=args.module,
        action=args.action,
        output_text=args.text or "",
        log_path=args.log,
        report_paths=args.report,
        options=options,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())