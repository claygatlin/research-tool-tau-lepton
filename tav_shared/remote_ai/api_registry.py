"""
Registry of remote AI providers used by research_tool.py.

Each provider maps to an environment variable and optional default model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from tav_shared.llm_analysis import DEFAULT_MODELS, query_claude, query_gemini, query_grok, query_openai


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    env_vars: tuple[str, ...]
    default_model: str
    query_fn: Callable[..., object]


def _has_key(*names: str) -> bool:
    return any(os.environ.get(name, "").strip() for name in names)


def list_providers() -> list[ProviderSpec]:
    return [
        ProviderSpec(
            id="grok",
            label="Grok (xAI)",
            env_vars=("XAI_API_KEY",),
            default_model=DEFAULT_MODELS["grok"],
            query_fn=query_grok,
        ),
        ProviderSpec(
            id="gemini",
            label="Gemini (Google)",
            env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            default_model=DEFAULT_MODELS["gemini"],
            query_fn=query_gemini,
        ),
        ProviderSpec(
            id="openai",
            label="ChatGPT (OpenAI)",
            env_vars=("OPENAI_API_KEY",),
            default_model=DEFAULT_MODELS["openai"],
            query_fn=query_openai,
        ),
        ProviderSpec(
            id="claude",
            label="Claude (Anthropic)",
            env_vars=("ANTHROPIC_API_KEY",),
            default_model=DEFAULT_MODELS["claude"],
            query_fn=query_claude,
        ),
        ProviderSpec(
            id="openrouter",
            label="OpenRouter (multi-model gateway)",
            env_vars=("OPENROUTER_API_KEY",),
            default_model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            query_fn=query_openrouter,
        ),
        ProviderSpec(
            id="ollama",
            label="Ollama (local / remote)",
            env_vars=("OLLAMA_BASE_URL",),
            default_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            query_fn=query_ollama,
        ),
        ProviderSpec(
            id="nvidia_nim",
            label="NVIDIA NIM (integrate.api.nvidia.com)",
            env_vars=("NVIDIA_API_KEY", "NVAPI_KEY"),
            default_model=os.environ.get("NVIDIA_NIM_MODEL", "minimaxai/minimax-m3"),
            query_fn=query_nvidia_nim,
        ),
    ]


def provider_status() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in list_providers():
        configured = _has_key(*spec.env_vars)
        rows.append(
            {
                "id": spec.id,
                "label": spec.label,
                "configured": "yes" if configured else "no",
                "env": " | ".join(spec.env_vars),
                "model": spec.default_model,
            }
        )
    return rows


def get_provider(provider_id: str) -> ProviderSpec | None:
    for spec in list_providers():
        if spec.id == provider_id:
            return spec
    return None


def query_openrouter(prompt: str, *, model: str | None = None):
    """OpenRouter OpenAI-compatible chat completions."""
    import time

    from tav_shared.llm_analysis import ProviderResult, _post_json

    provider = "openrouter"
    model = model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return ProviderResult(
            provider=provider,
            model=model,
            skipped=True,
            skip_reason="OPENROUTER_API_KEY not set",
        )
    started = time.monotonic()
    try:
        data = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/TauSuperblock",
                "X-Title": "TauSuperblock research_tool",
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


def query_ollama(prompt: str, *, model: str | None = None):
    """Ollama local or remote generate API."""
    import time

    from tav_shared.llm_analysis import ProviderResult, _post_json

    provider = "ollama"
    model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    if not base:
        return ProviderResult(
            provider=provider,
            model=model,
            skipped=True,
            skip_reason="OLLAMA_BASE_URL not set",
        )
    started = time.monotonic()
    try:
        data = _post_json(
            f"{base}/api/chat",
            headers={"Content-Type": "application/json"},
            payload={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            provider=provider,
        )
        text = (data.get("message") or {}).get("content", "")
        if not text:
            raise RuntimeError(f"empty Ollama response: {str(data)[:400]}")
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


def query_nvidia_nim(prompt: str, *, model: str | None = None):
    """
    NVIDIA integrate API — OpenAI-compatible chat completions.

    Default model: minimaxai/minimax-m3 (MiniMax-M3 via NIM).
    """
    import time

    from tav_shared.llm_analysis import ProviderResult, _post_json

    provider = "nvidia_nim"
    model = model or os.environ.get("NVIDIA_NIM_MODEL", "minimaxai/minimax-m3")
    api_key = (
        os.environ.get("NVIDIA_API_KEY", "").strip()
        or os.environ.get("NVAPI_KEY", "").strip()
    )
    base_url = os.environ.get(
        "NVIDIA_NIM_BASE_URL",
        "https://integrate.api.nvidia.com/v1/chat/completions",
    ).strip()

    if not api_key:
        return ProviderResult(
            provider=provider,
            model=model,
            skipped=True,
            skip_reason="NVIDIA_API_KEY or NVAPI_KEY not set",
        )
    started = time.monotonic()
    try:
        data = _post_json(
            base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            payload={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert scientific reviewer."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": int(os.environ.get("NVIDIA_NIM_MAX_TOKENS", "8192")),
                "temperature": float(os.environ.get("NVIDIA_NIM_TEMPERATURE", "0.2")),
                "top_p": float(os.environ.get("NVIDIA_NIM_TOP_P", "0.95")),
                "stream": False,
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