#!/usr/bin/env bash
# Install Research/tav-resonance into the active research Python environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG="${ROOT}/tav-resonance"
PY="${RESEARCH_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in "${ROOT}/.venv/bin/python" "./venv/bin/python" "./.venv/bin/python"; do
    if [[ -x "$c" ]]; then PY="$c"; break; fi
  done
fi
PY="${PY:-python3}"
echo "Installing tav-resonance from ${PKG}"
echo "Using interpreter: ${PY}"
exec "$PY" -m pip install -e "$PKG"