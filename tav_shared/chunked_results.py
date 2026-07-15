"""
Chunked JSON persistence for large Tav analysis payloads.

Splits oversized result dicts into upload-friendly parts (default 8 MB) with a
manifest for reassembly. Used by CERN full-dataset runs and any pipeline whose
serialized report exceeds a size ceiling.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from menus.astronomical.desi.json_util import (
    numpy_json_default,
    sanitize_for_json,
    write_json,
)

CHUNKED_FORMAT_VERSION = "tav_chunked_results_v1"
PAYLOAD_FORMAT_VERSION = "tav_chunked_payload_v1"

# Full CMS NanoAOD skims (e.g. 61M dimu) exceed single-file JSON limits.
LARGE_DATASET_EVENT_THRESHOLD = 100_000
DEFAULT_MAX_CHUNK_MB = 8.0


def should_chunk_tav_results(
    *,
    n_events: int = 0,
    chunk_results: bool | None = None,
) -> bool:
    """Return True when chunked persistence should be used."""
    if chunk_results is not None:
        return bool(chunk_results)
    return int(n_events) >= LARGE_DATASET_EVENT_THRESHOLD


def _serialize_results(results: dict[str, Any], *, indent: int) -> bytes:
    clean = sanitize_for_json(results)
    text = json.dumps(
        clean,
        indent=indent,
        sort_keys=True,
        default=numpy_json_default,
    )
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


def _chunk_summary(results: dict[str, Any]) -> dict[str, Any]:
    mult = results.get("multiplicity_7fold") or {}
    return {
        "action": results.get("action"),
        "verdict": results.get("verdict"),
        "timestamp": results.get("timestamp"),
        "seven_periodic": results.get("seven_periodic"),
        "total_muons_binned": results.get("total_muons_binned"),
        "n_events": mult.get("n_events"),
        "n_pt_bins": results.get("n_pt_bins"),
    }


def save_tav_results_chunked(
    results: dict[str, Any],
    output_dir: str | Path,
    base_filename: str,
    *,
    max_size_mb: float = DEFAULT_MAX_CHUNK_MB,
    indent: int = 2,
) -> list[str]:
    """
    Save Tav results to JSON; split into <= ``max_size_mb`` chunks when needed.

    Returns created file paths. Single-file saves return one ``.json`` path.
    Multi-part saves return ``[manifest, part001, part002, ...]``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = str(base_filename).strip()
    if base.endswith(".json"):
        base = base[:-5]

    encoded = _serialize_results(results, indent=indent)
    max_bytes = max(1, int(float(max_size_mb) * 1024 * 1024))

    if len(encoded) <= max_bytes:
        single_path = out_dir / f"{base}.json"
        single_path.write_bytes(encoded)
        return [str(single_path.resolve())]

    # Base64 (+33%) and JSON envelope expand each part; keep files under max_bytes.
    envelope_reserve = 16_384
    raw_chunk_bytes = max(4096, int((max_bytes - envelope_reserve) * 0.70))

    chunk_paths: list[Path] = []
    for idx, offset in enumerate(range(0, len(encoded), raw_chunk_bytes)):
        part_bytes = encoded[offset : offset + raw_chunk_bytes]
        part_path = out_dir / f"{base}_part{idx + 1:03d}.json"
        write_json(
            part_path,
            {
                "format": PAYLOAD_FORMAT_VERSION,
                "chunk_index": idx,
                "encoding": "base64",
                "byte_length": len(part_bytes),
                "payload": base64.b64encode(part_bytes).decode("ascii"),
            },
            indent=indent,
            sort_keys=True,
        )
        chunk_paths.append(part_path)

    manifest_path = out_dir / f"{base}_manifest.json"
    write_json(
        manifest_path,
        {
            "format": CHUNKED_FORMAT_VERSION,
            "base_filename": base,
            "n_chunks": len(chunk_paths),
            "total_bytes": len(encoded),
            "max_size_mb": float(max_size_mb),
            "chunks": [p.name for p in chunk_paths],
            "summary": _chunk_summary(results),
        },
        indent=indent,
        sort_keys=True,
    )

    return [str(manifest_path.resolve())] + [
        str(p.resolve()) for p in chunk_paths
    ]


def load_tav_results_chunked(path: str | Path) -> dict[str, Any]:
    """
    Load Tav results from a single JSON file or a chunked manifest + parts.
    """
    entry = Path(path)
    raw = json.loads(entry.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {entry}")

    fmt = raw.get("format")
    if fmt != CHUNKED_FORMAT_VERSION:
        return raw

    base = entry.parent
    parts: list[bytes] = []
    for idx, name in enumerate(raw.get("chunks") or []):
        part_path = base / str(name)
        part = json.loads(part_path.read_text(encoding="utf-8"))
        if part.get("format") != PAYLOAD_FORMAT_VERSION:
            raise ValueError(f"Unexpected chunk format in {part_path}")
        if int(part.get("chunk_index", -1)) != idx:
            raise ValueError(f"Chunk order mismatch at {part_path}")
        payload = part.get("payload")
        if not isinstance(payload, str):
            raise ValueError(f"Missing payload in {part_path}")
        parts.append(base64.b64decode(payload.encode("ascii")))

    merged = b"".join(parts)
    loaded = json.loads(merged.decode("utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Reassembled Tav results are not a JSON object")
    return loaded