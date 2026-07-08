"""Ping remote AI providers with a minimal connectivity prompt."""

from __future__ import annotations

from typing import Any

from tav_shared.llm_analysis import ProviderResult, parse_provider_list, query_all_providers
from tav_shared.remote_ai.api_registry import get_provider, list_providers, provider_status

_PING_PROMPT = (
    "Reply with exactly one line: OK — remote AI connection test for TauSuperblock."
)


def test_api_connections(
    *,
    providers: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Test configured remote AI endpoints.

    ``providers`` may be None (all registered), a provider id list, or parsed from
    options via ``parse_provider_list``.
    """
    if providers is None:
        want = [spec.id for spec in list_providers()]
    else:
        want = parse_provider_list({"llm_providers": ",".join(providers)})

    if verbose:
        print("\n" + "=" * 72)
        print("REMOTE AI — API CONNECTION TEST")
        print("=" * 72)
        for row in provider_status():
            flag = "✓" if row["configured"] == "yes" else "✗"
            print(f"  [{flag}] {row['label']:<32} env: {row['env']}")

    # Extend query_all_providers for openrouter/ollama via registry
    results: list[ProviderResult] = []
    for name in want:
        spec = get_provider(name)
        if spec is None:
            results.append(
                ProviderResult(
                    provider=name,
                    model="",
                    skipped=True,
                    skip_reason=f"unknown provider '{name}'",
                )
            )
            continue
        if not any(__import__("os").environ.get(v, "").strip() for v in spec.env_vars):
            results.append(
                ProviderResult(
                    provider=spec.id,
                    model=spec.default_model,
                    skipped=True,
                    skip_reason=f"{' or '.join(spec.env_vars)} not set",
                )
            )
            continue
        results.append(spec.query_fn(_PING_PROMPT, model=spec.default_model))

    ok = [r for r in results if r.analysis and not r.error and not r.skipped]
    failed = [r for r in results if r.error]
    skipped = [r for r in results if r.skipped]

    if verbose:
        print("\n--- Results ---")
        for row in results:
            if row.skipped:
                print(f"  {row.provider}: SKIPPED ({row.skip_reason})")
            elif row.error:
                print(f"  {row.provider}: ERROR — {row.error[:160]}")
            else:
                preview = (row.analysis or "").splitlines()[0][:80]
                print(f"  {row.provider}: OK ({row.latency_s:.1f}s) — {preview}")

        print(
            f"\nSummary: {len(ok)} connected, {len(skipped)} skipped, {len(failed)} failed"
        )

    return {
        "providers_tested": want,
        "connected": len(ok),
        "skipped": len(skipped),
        "failed": len(failed),
        "results": [r.to_dict() for r in results],
        "status": provider_status(),
    }