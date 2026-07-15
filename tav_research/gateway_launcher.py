"""Spawn TAU-SB Gateway GUI as a detached background process."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

GATEWAY_GUI_MENU_LABEL = "TAU-SB Gateway (GUI)"

_LOG_DIR = Path.home() / "Research" / "cache" / "logs"
_LOG_PATH = _LOG_DIR / "gateway.log"

_gateway_proc: subprocess.Popen | None = None
_log_handle: object | None = None


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _research_root() -> Path:
    return _scripts_dir().parent


def _python_executable() -> Path:
    venv = _research_root() / ".venv" / "bin" / "python"
    if venv.is_file():
        return venv
    return Path(sys.executable)


def _gateway_script() -> Path:
    return _scripts_dir() / "gateway.py"


def gateway_log_path() -> Path:
    return _LOG_PATH


def is_gateway_running() -> bool:
    global _gateway_proc
    if _gateway_proc is not None and _gateway_proc.poll() is None:
        return True
    return False


def _prepare_spawn_env() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
        if Path("/tmp/.X11-unix").is_dir():
            env.setdefault("DISPLAY", ":0")
    scripts = str(_scripts_dir())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        from research_bridge import load_research_keys

        load_research_keys(verbose=False)
    except Exception:
        pass
    return env


def spawn_gateway_gui() -> tuple[int | None, Path, str | None]:
    """
    Start gateway.py --gui in a new session. Output goes to gateway.log only.

    Returns (pid, log_path, error_message). error_message is None on success.
    """
    global _gateway_proc, _log_handle

    if is_gateway_running():
        assert _gateway_proc is not None
        return _gateway_proc.pid, _LOG_PATH, None

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = _prepare_spawn_env()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log_handle = _LOG_PATH.open("a", encoding="utf-8")
    log_handle.write(f"\n--- Gateway spawn {stamp} ---\n")
    log_handle.flush()

    cmd = [str(_python_executable()), str(_gateway_script()), "--gui"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(_research_root()),
            env=env,
            start_new_session=True,
        )
    except Exception as exc:
        log_handle.close()
        return None, _LOG_PATH, str(exc)

    _gateway_proc = proc
    _log_handle = log_handle
    return proc.pid, _LOG_PATH, None