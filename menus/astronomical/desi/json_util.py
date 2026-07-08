"""JSON helpers for Tau-SB DESI artifacts (NumPy-safe serialization)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def numpy_json_default(obj: Any) -> Any:
    """
    ``json.dumps(..., default=numpy_json_default)`` callback.

    Prefer this over a bare ``lambda x: x.tolist() if isinstance(x, np.ndarray) else x``,
    which leaves ``np.float64`` / ``np.int64`` / ``np.bool_`` unconverted.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class NumpyEncoder(json.JSONEncoder):
    """Serialize NumPy scalars and arrays in ``json.dumps``."""

    def default(self, obj: Any) -> Any:
        try:
            return numpy_json_default(obj)
        except TypeError:
            return super().default(obj)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert payloads to JSON-safe Python types.

    Drops matplotlib figure objects and other non-serializable plot handles.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    type_name = type(obj).__name__
    if type_name in {"Figure", "Axes", "AxesSubplot"} or type_name.endswith("Figure"):
        return None

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in {
                "acf_plot",
                "qq_plot",
                "residuals_vs_fitted_plot",
                "figures",
                "gamma_sensitivity_plot",
                "jackknife_plot",
                "rd_vs_gamma_plot",
            }:
                continue
            cleaned = sanitize_for_json(value)
            if cleaned is not None:
                out[str(key)] = cleaned
        return out

    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]

    return obj


def json_dumps(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` with NumPy-safe ``default`` and plot-handle stripping."""
    cleaned = sanitize_for_json(obj)
    if "default" not in kwargs and "cls" not in kwargs:
        kwargs["default"] = numpy_json_default
    return json.dumps(cleaned, **kwargs)


def write_json(path: str | Path, obj: Any, **kwargs: Any) -> Path:
    """Write JSON artifact with NumPy-safe encoding."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json_dumps(obj, **kwargs)
    if not text.endswith("\n"):
        text += "\n"
    out.write_text(text, encoding="utf-8")
    return out