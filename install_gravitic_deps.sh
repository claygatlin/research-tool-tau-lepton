#!/usr/bin/env bash
# Install GWOSC/LISA data-access stack for the research tool:
#   - gwdatafind (LIGO GWDataFind Python client)
#   - requests-pelican (Pelican/OSDF HTTP adapter)
#   - pelican CLI (OSDF client binary, bundled under Research/third_party/pelican/)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REQ="${ROOT}/scripts/research_tool/requirements-gravitic.txt"
PY="${RESEARCH_PYTHON:-}"

if [[ -z "$PY" ]]; then
  for c in "${ROOT}/.venv/bin/python" "${ROOT}/scripts/research_tool/.venv/bin/python"; do
    if [[ -x "$c" ]]; then PY="$c"; break; fi
  done
fi
PY="${PY:-python3}"

echo "[gravitic] Research root: ${ROOT}"
echo "[gravitic] Python: ${PY}"
echo "[gravitic] Installing Python packages from ${REQ}"
"$PY" -m pip install -r "$REQ"

echo "[gravitic] Installing Pelican CLI (v${PELICAN_VERSION:-7.25.0})"
export PYTHONPATH="${ROOT}/scripts/research_tool:${PYTHONPATH:-}"
"$PY" - <<'PY'
from menus.gravitic.common.pelican_tools import install_pelican_binary, pelican_status

path = install_pelican_binary()
status = pelican_status()
print(f"[gravitic] Pelican binary: {path}")
print(f"[gravitic] Pelican version: {status.get('version')}")
PY

echo "[gravitic] Done."
echo "[gravitic] Optional env:"
echo "  export GWDATAFIND_SERVER=https://datafind.gwosc.org"
echo "  export PELICAN_BIN=${ROOT}/third_party/pelican/bin/pelican"