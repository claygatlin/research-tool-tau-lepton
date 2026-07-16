"""Native uproot analysis for cached CERN Open Data ROOT files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp, compose_dataset_slug
from menus.particle.cern.fetcher import resolve_root_path
from menus.particle.cern.mod7_phase import mod7_phase_residues, weighted_mod7_histogram
from menus.particle.cern.tav_superblock_cms_muon_analyzer import (
    DEFAULT_PT_BINS,
    tav_7fold_muon_analysis,
)
from tav_research.curses_shell import ChunkedScanProgress, optional_chunked_scan_progress
from tav_shared.ingestion import get_runtime_data_manager
from tav_shared.ingestion.pipeline import validate_and_normalize_cms_dimuon_batch


PT_REDUCTION_FLATTEN = "flatten"
PT_REDUCTION_LEADING = "leading"
PT_REDUCTION_CHOICES: tuple[str, ...] = (PT_REDUCTION_FLATTEN, PT_REDUCTION_LEADING)


def _numpy_scalar_float(value: Any) -> float:
    """Coerce 0-d or 1-element numpy values without scalar-conversion errors."""
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return 0.0
    return float(arr[0])


def _is_awkward_array(arr: Any) -> bool:
    return hasattr(arr, "type") and hasattr(arr, "layout")


def _jagged_event_count(batch_data: Any) -> int:
    """Number of events in one uproot iterate batch of a jagged branch."""
    if _is_awkward_array(batch_data):
        return int(ak.num(batch_data, axis=0))
    arr = np.asarray(batch_data, dtype=object)
    if arr.dtype == object:
        return int(arr.size)
    return int(len(batch_data))


def _flatten_branch(arr: Any) -> np.ndarray:
    """Collapse jagged NanoAOD branches to a 1-D float array (all objects)."""
    flat = ak.to_numpy(ak.flatten(arr, axis=None))
    flat = np.asarray(flat, dtype=float)
    return flat[np.isfinite(flat)]


def _leading_jagged_awkward(arr: Any) -> np.ndarray:
    """
    Leading object pT per event from awkward jagged NanoAOD.

    Equivalent to::

        leading = electrons_pt[electrons_pt.counts > 0, 0]
    """
    if arr is None:
        return np.array([], dtype=float)
    try:
        counts = ak.num(arr, axis=1)
    except (ValueError, TypeError):
        return _flatten_branch(arr)
    mask = counts > 0
    if not bool(ak.any(mask)):
        return np.array([], dtype=float)
    leading = arr[mask, 0]
    out = np.asarray(ak.to_numpy(leading), dtype=float).ravel()
    return out[np.isfinite(out)]


def reduce_jagged_pt_branch(
    arr: Any,
    *,
    mode: str = PT_REDUCTION_FLATTEN,
) -> np.ndarray:
    """
    Reduce jagged ``Electron_pt`` / ``Muon_pt`` / ``Photon_pt`` for analysis.

    - ``flatten``: all objects — use for global periodograms and pT histograms.
    - ``leading``: first object per event — use for per-event 0-D kinematic cuts.
    """
    reduction = (mode or PT_REDUCTION_FLATTEN).strip().lower()
    if reduction == PT_REDUCTION_LEADING:
        return _leading_jagged_awkward(arr)
    return _flatten_branch(arr)


def _per_event_array(arr: Any) -> np.ndarray:
    """One scalar per event (e.g. nMuon)."""
    vals = ak.to_numpy(arr)
    return np.asarray(vals, dtype=float).ravel()


def _tree_branch_names(tree: Any) -> set[str]:
    """Return branch names available on a NanoAOD Events tree."""
    try:
        return {str(key) for key in tree.keys()}
    except Exception:
        return set()


_LEPTON_BRANCH_PAIRS: dict[str, tuple[str, str]] = {
    "Muon": ("nMuon", "Muon_pt"),
    "Electron": ("nElectron", "Electron_pt"),
}

PILEUP_NTREAINT_BRANCHES: tuple[str, ...] = (
    "Pileup_nTrueInt",
    "Pileup_nPU",
    "fixedGridRhoFastest",
)

PHOTON_PT_BRANCHES: tuple[str, ...] = (
    "Photon_pt",
    "Electron_pt",
)


def resolve_pileup_branch(tree: Any) -> str | None:
    """Return the first available NanoAOD pileup branch on ``tree``."""
    available = _tree_branch_names(tree)
    for name in PILEUP_NTREAINT_BRANCHES:
        if name in available:
            return name
    return None


def resolve_photon_branch(tree: Any) -> str | None:
    """Return Photon_pt or Electron_pt for vector-boson / gamma crosschecks."""
    available = _tree_branch_names(tree)
    for name in PHOTON_PT_BRANCHES:
        if name in available:
            return name
    return None


def _cms_chunk_scan_branches(
    tree: Any,
    *,
    primary_lepton: str = "Muon",
    include_crosscheck: bool = True,
    include_electrons: bool | None = None,
) -> tuple[list[str], str, str | None]:
    """
    Pick uproot.iterate branch list for a CMS NanoAOD file.

    Dimu skims (e.g. Run2012BC_DoubleMuParked_Muons.root) often contain only
    ``nMuon`` and ``Muon_pt`` — never request missing electron branches.

    Returns ``(branches, primary_kind, crosscheck_kind)`` where kinds are
    ``"muon"`` / ``"electron"`` / ``None``.
    """
    available = _tree_branch_names(tree)
    primary_key = str(primary_lepton or "Muon").strip().title()
    if primary_key not in _LEPTON_BRANCH_PAIRS:
        raise ValueError(f"Unsupported primary lepton: {primary_lepton!r}")

    n_branch, pt_branch = _LEPTON_BRANCH_PAIRS[primary_key]
    branches = [name for name in (n_branch, pt_branch) if name in available]
    if len(branches) != 2:
        raise KeyError(
            f"NanoAOD tree missing {primary_key} branches "
            f"({n_branch}, {pt_branch}); found: {sorted(available)[:12]}"
        )

    primary_kind = primary_key.lower()
    cross_kind: str | None = None
    want_cross = include_crosscheck
    if include_electrons is not None and primary_key == "Muon":
        want_cross = bool(include_electrons)
    if want_cross:
        other_key = "Electron" if primary_key == "Muon" else "Muon"
        o_n, o_pt = _LEPTON_BRANCH_PAIRS[other_key]
        other_branches = [name for name in (o_n, o_pt) if name in available]
        if len(other_branches) == 2:
            branches.extend(other_branches)
            cross_kind = other_key.lower()
    return branches, primary_kind, cross_kind


def _flatten_jagged_np(jagged_batch: Any) -> np.ndarray:
    """Flatten one uproot iterate batch of jagged ``Electron_pt`` / ``Muon_pt``."""
    if _is_awkward_array(jagged_batch):
        return _flatten_branch(jagged_batch)
    arr = np.asarray(jagged_batch, dtype=object)
    if arr.size == 0:
        return np.array([], dtype=float)
    if arr.dtype != object:
        flat = np.asarray(arr, dtype=float).ravel()
        return flat[np.isfinite(flat)]
    parts: list[np.ndarray] = []
    for item in arr:
        if item is None:
            continue
        piece = np.asarray(item, dtype=float).ravel()
        if piece.size:
            parts.append(piece[np.isfinite(piece)])
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def _leading_jagged_np(jagged_batch: Any) -> np.ndarray:
    """
    Leading object pT per event from uproot iterate batches.

    Matches awkward selection ``branch[branch.counts > 0, 0]`` for 0-D expectations.
    """
    if _is_awkward_array(jagged_batch):
        return _leading_jagged_awkward(jagged_batch)
    arr = np.asarray(jagged_batch, dtype=object)
    if arr.size == 0:
        return np.array([], dtype=float)
    if arr.dtype != object:
        flat = np.asarray(arr, dtype=float).ravel()
        return flat[np.isfinite(flat)]
    leading: list[float] = []
    for item in arr:
        if item is None:
            continue
        piece = np.asarray(item, dtype=float).ravel()
        if piece.size:
            val = _numpy_scalar_float(piece[0])
            if np.isfinite(val):
                leading.append(val)
    return np.asarray(leading, dtype=float)


def _reduce_jagged_np(
    jagged_batch: Any,
    *,
    mode: str = PT_REDUCTION_FLATTEN,
) -> np.ndarray:
    """Dispatch flatten vs leading for chunked uproot iterate batches."""
    reduction = (mode or PT_REDUCTION_FLATTEN).strip().lower()
    if reduction == PT_REDUCTION_LEADING:
        return _leading_jagged_np(jagged_batch)
    return _flatten_jagged_np(jagged_batch)


def scan_cms_nanoaod_chunked(
    root_path: str | Path,
    *,
    tree_name: str = "Events",
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    pt_bins: int = 20,
    pt_max_gev: float = 200.0,
    primary_lepton: str = "Muon",
    include_crosscheck: bool = True,
    include_electrons: bool | None = None,
    pt_reduction: str = PT_REDUCTION_FLATTEN,
    crosscheck_pt_reduction: str | None = None,
    progress_every: int = 5_000_000,
    progress: ChunkedScanProgress | None = None,
    use_curses_progress: bool = True,
    progress_title: str = "CMS Full-Dataset 7-Fold Scan",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Chunked full-file NanoAOD scan — accumulates global pT histogram + multiplicity.

    Designed for the 61M-event ``Run2012BC_DoubleMuParked_Muons.root`` skim.
    Set ``primary_lepton='Electron'`` for electron-only or electron-primary skims.
    """
    path = Path(root_path)
    primary_key = str(primary_lepton or "Muon").strip().title()
    n_primary_branch, pt_primary_branch = _LEPTON_BRANCH_PAIRS[primary_key]
    primary_reduction = (pt_reduction or PT_REDUCTION_FLATTEN).strip().lower()
    cross_reduction = (
        (crosscheck_pt_reduction or pt_reduction or PT_REDUCTION_FLATTEN).strip().lower()
    )

    bin_edges = np.linspace(0.0, pt_max_gev, int(pt_bins) + 1)
    primary_hist = np.zeros(int(pt_bins), dtype=np.float64)
    cross_hist: np.ndarray | None = None
    n_primary_chunks: list[np.ndarray] = []
    total_events = 0
    total_primary = 0
    branches: list[str] = []
    primary_kind = "muon"
    cross_kind: str | None = None

    n_entries_in_file = 0
    effective_stop = 0

    with optional_chunked_scan_progress(
        progress,
        title=progress_title,
        use_curses_progress=use_curses_progress,
    ) as live_progress:
        if verbose and live_progress is None:
            print(f"[CERN CMS] Starting chunked scan: {path}")

        with uproot.open(path) as handle:
            tree = handle[tree_name]
            branches, primary_kind, cross_kind = _cms_chunk_scan_branches(
                tree,
                primary_lepton=primary_key,
                include_crosscheck=include_crosscheck,
                include_electrons=include_electrons,
            )
            if cross_kind is not None:
                cross_hist = np.zeros(int(pt_bins), dtype=np.float64)
            n_entries_in_file = int(tree.num_entries)
            effective_stop = (
                n_entries_in_file
                if not entry_stop or entry_stop <= 0
                else min(int(entry_stop), n_entries_in_file)
            )

            cross_pt_branch = (
                _LEPTON_BRANCH_PAIRS["Muon" if cross_kind == "muon" else "Electron"][1]
                if cross_kind
                else None
            )

            if live_progress is not None:
                live_progress.begin(
                    path=str(path),
                    total_events=effective_stop,
                    chunk_size=int(chunk_size),
                )
            elif verbose:
                print(f"[CERN CMS] Total events in file: {n_entries_in_file:,}")
                print(f"[CERN CMS] Primary lepton: {primary_key}")
                print(f"[CERN CMS] Branches: {', '.join(branches)}")
                if include_crosscheck and cross_kind is None:
                    other = "Muon" if primary_key == "Electron" else "Electron"
                    print(
                        f"[CERN CMS] {other} cross-check skipped "
                        f"(branches not in this skim)."
                    )

            last_report = 0
            lepton_label = primary_key

            for batch in tree.iterate(
                branches,
                step_size=int(chunk_size),
                entry_stop=effective_stop,
                library="np",
            ):
                n_primary = np.asarray(batch[n_primary_branch], dtype=np.float32)
                pt_flat = _reduce_jagged_np(
                    batch[pt_primary_branch],
                    mode=primary_reduction,
                )

                if pt_flat.size:
                    hist_chunk, _ = np.histogram(pt_flat, bins=bin_edges)
                    primary_hist += hist_chunk
                    total_primary += int(pt_flat.size)

                n_primary_chunks.append(n_primary)
                total_events += int(n_primary.size)

                if cross_kind and cross_pt_branch and cross_pt_branch in batch:
                    cross_flat = _reduce_jagged_np(
                        batch[cross_pt_branch],
                        mode=cross_reduction,
                    )
                    if cross_flat.size and cross_hist is not None:
                        c_chunk, _ = np.histogram(cross_flat, bins=bin_edges)
                        cross_hist += c_chunk

                if live_progress is not None:
                    live_progress.update(total_events, total_primary)
                if live_progress is None or not live_progress.is_interactive:
                    if (
                        verbose
                        and progress_every > 0
                        and total_events - last_report >= progress_every
                    ):
                        print(
                            f"[CERN CMS]   Processed {total_events:,} / {effective_stop:,} events..."
                        )
                        last_report = total_events

        if live_progress is not None:
            live_progress.finish(total_events, total_primary)
        if verbose and (live_progress is None or not live_progress.is_interactive):
            print(
                f"[CERN CMS] Finished {total_events:,} events "
                f"({total_primary:,} {lepton_label.lower()}s binned)."
            )

    n_primary_per_event = (
        np.concatenate(n_primary_chunks)
        if n_primary_chunks
        else np.array([], dtype=np.float32)
    )

    empty_hist = np.zeros(int(pt_bins), dtype=np.float64)
    if primary_kind == "muon":
        muon_pt_hist = primary_hist
        n_muon_per_event = n_primary_per_event
        electron_pt_hist = cross_hist
        n_muons_binned = total_primary
        n_electrons_binned = 0
        n_electron_per_event = np.array([], dtype=np.float32)
    else:
        electron_pt_hist = primary_hist
        n_electron_per_event = n_primary_per_event
        muon_pt_hist = cross_hist if cross_hist is not None else empty_hist
        n_muon_per_event = np.array([], dtype=np.float32)
        n_muons_binned = 0
        n_electrons_binned = total_primary

    return {
        "path": str(path),
        "tree": tree_name,
        "chunk_size": int(chunk_size),
        "n_entries_in_file": n_entries_in_file,
        "entry_stop": effective_stop,
        "n_events_processed": total_events,
        "primary_lepton": primary_kind,
        "n_primary_binned": total_primary,
        "primary_pt_hist": primary_hist,
        "n_primary_per_event": n_primary_per_event,
        "crosscheck_pt_hist": cross_hist,
        "crosscheck_lepton": cross_kind,
        "bin_edges": bin_edges,
        "branches_read": branches,
        "pt_bins": int(pt_bins),
        "pt_max_gev": float(pt_max_gev),
        "pt_reduction": primary_reduction,
        "crosscheck_pt_reduction": cross_reduction if cross_kind else None,
        # Backward-compatible aliases
        "n_muons_binned": n_muons_binned,
        "muon_pt_hist": muon_pt_hist,
        "n_muon_per_event": n_muon_per_event,
        "electron_pt_hist": electron_pt_hist,
        "n_electrons_binned": n_electrons_binned,
        "n_electron_per_event": n_electron_per_event,
        "electron_crosscheck": cross_kind == "electron",
    }


def _dimuon_kinematics_from_jagged(
    pt_jagged: Any,
    eta_jagged: Any,
    phi_jagged: Any,
) -> dict[str, np.ndarray]:
    """Build per-dimuon-event kinematics from one awkward/numpy jagged batch."""
    n_events = len(pt_jagged)
    leading: list[float] = []
    subleading: list[float] = []
    dphi_out: list[float] = []
    dr_out: list[float] = []
    sys_pt_out: list[float] = []

    for idx in range(n_events):
        pt_row = np.asarray(pt_jagged[idx], dtype=float).ravel()
        if pt_row.size < 2:
            continue
        eta_row = np.asarray(eta_jagged[idx], dtype=float).ravel()
        phi_row = np.asarray(phi_jagged[idx], dtype=float).ravel()
        n = min(pt_row.size, eta_row.size, phi_row.size)
        if n < 2:
            continue
        pt_row, eta_row, phi_row = pt_row[:n], eta_row[:n], phi_row[:n]
        order = np.argsort(pt_row)[::-1]
        pt1, pt2 = float(pt_row[order[0]]), float(pt_row[order[1]])
        eta1, eta2 = float(eta_row[order[0]]), float(eta_row[order[1]])
        phi1, phi2 = float(phi_row[order[0]]), float(phi_row[order[1]])
        dphi = abs(phi1 - phi2)
        dphi = min(dphi, 2.0 * np.pi - dphi)
        deta = eta1 - eta2
        delta_r = float(np.hypot(deta, dphi))
        px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
        py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
        system_pt = float(np.hypot(px, py))
        leading.append(pt1)
        subleading.append(pt2)
        dphi_out.append(float(dphi))
        dr_out.append(delta_r)
        sys_pt_out.append(system_pt)

    return {
        "leading_pt": np.asarray(leading, dtype=float),
        "subleading_pt": np.asarray(subleading, dtype=float),
        "delta_phi": np.asarray(dphi_out, dtype=float),
        "delta_r": np.asarray(dr_out, dtype=float),
        "system_pt": np.asarray(sys_pt_out, dtype=float),
    }


def extract_pileup_ntrueint_histogram(
    root_path: str | Path,
    *,
    tree_name: str = "Events",
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    max_bin: int = 100,
    verbose: bool = False,
    optional: bool = False,
) -> dict[str, Any] | None:
    """
    Build per-event ``nTrueInt`` occupancy histogram from a NanoAOD ROOT file.

    Used as the data target (or MC source) profile for ``TavMCWeighting``.

    When ``optional=True``, returns ``None`` if the file has no pileup branch
    (e.g. dimuon skims) instead of raising ``KeyError``.
    """
    resolved = resolve_root_path(str(root_path))
    path = Path(resolved) if resolved is not None else Path(root_path)
    if not path.is_file():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    hist = np.zeros(int(max_bin) + 1, dtype=np.float64)
    total_events = 0
    n_entries_in_file = 0
    effective_stop = 0
    pileup_branch: str | None = None

    with uproot.open(path) as handle:
        tree = handle[tree_name]
        pileup_branch = resolve_pileup_branch(tree)
        if pileup_branch is None:
            if optional:
                return None
            raise KeyError(
                f"No pileup branch in {path.name}; tried {PILEUP_NTREAINT_BRANCHES}"
            )
        n_entries_in_file = int(tree.num_entries)
        effective_stop = (
            n_entries_in_file
            if not entry_stop or entry_stop <= 0
            else min(int(entry_stop), n_entries_in_file)
        )
        if verbose:
            print(
                f"[CERN CMS] Pileup profile ({pileup_branch}): "
                f"{path.name} ({effective_stop:,} events)"
            )
        for batch in tree.iterate(
            [pileup_branch],
            step_size=int(chunk_size),
            entry_stop=effective_stop,
            library="np",
        ):
            ntrue = np.asarray(batch[pileup_branch], dtype=np.float64).ravel()
            ntrue = ntrue[np.isfinite(ntrue)]
            ntrue = np.clip(np.rint(ntrue), 0, int(max_bin)).astype(int)
            if ntrue.size:
                chunk_hist = np.bincount(ntrue, minlength=int(max_bin) + 1)
                hist += chunk_hist[: hist.size]
            total_events += int(ntrue.size)

    return {
        "path": str(path),
        "pileup_branch": pileup_branch,
        "histogram": hist,
        "n_events": total_events,
        "n_entries_in_file": n_entries_in_file,
        "entry_stop": effective_stop,
        "max_bin": int(max_bin),
    }


def extract_pileup_ntrueint_histogram_optional(
    root_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Like :func:`extract_pileup_ntrueint_histogram` but never raises on missing pileup."""
    return extract_pileup_ntrueint_histogram(root_path, optional=True, **kwargs)


def extract_photon_pt_histogram_from_nanoaod(
    root_path: str | Path,
    *,
    tree_name: str = "Events",
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    pt_bins: int = 20,
    pt_max_gev: float = 200.0,
    pt_reduction: str = PT_REDUCTION_FLATTEN,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Chunked pT histogram from ``Photon_pt`` or ``Electron_pt`` NanoAOD branches.

    Used by Option 4 to map a secondary gamma / EM dataset into the validation
    configuration when the primary dimu skim lacks photon branches.

    ``pt_reduction='flatten'`` (default) flattens all electrons/photons for
    global periodograms. Use ``'leading'`` for one pT per event (0-D kinematics).
    """
    resolved = resolve_root_path(str(root_path))
    path = Path(resolved) if resolved is not None else Path(root_path)
    if not path.is_file():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    bin_edges = np.linspace(0.0, pt_max_gev, int(pt_bins) + 1)
    photon_hist = np.zeros(int(pt_bins), dtype=np.float64)
    total_events = 0
    total_objects = 0
    n_entries_in_file = 0
    effective_stop = 0
    photon_branch: str | None = None
    channel = "undetermined"
    reduction = (pt_reduction or PT_REDUCTION_FLATTEN).strip().lower()
    leading_per_event: list[np.ndarray] = []

    with uproot.open(path) as handle:
        tree = handle[tree_name]
        photon_branch = resolve_photon_branch(tree)
        if photon_branch is None:
            raise KeyError(
                f"No photon/electron pT branch in {path.name}; "
                f"tried {PHOTON_PT_BRANCHES}"
            )
        channel = "photon" if photon_branch == "Photon_pt" else "electron_em"
        n_entries_in_file = int(tree.num_entries)
        effective_stop = (
            n_entries_in_file
            if not entry_stop or entry_stop <= 0
            else min(int(entry_stop), n_entries_in_file)
        )
        if verbose:
            print(
                f"[CERN CMS] Photon kinematics ({photon_branch}, {reduction}): "
                f"{path.name} ({effective_stop:,} events)"
            )
        for batch in tree.iterate(
            [photon_branch],
            step_size=int(chunk_size),
            entry_stop=effective_stop,
            library="ak",
        ):
            branch_data = batch[photon_branch]
            pt_values = reduce_jagged_pt_branch(branch_data, mode=reduction)
            if pt_values.size:
                chunk_hist, _ = np.histogram(pt_values, bins=bin_edges)
                photon_hist += chunk_hist
                total_objects += int(pt_values.size)
                if reduction == PT_REDUCTION_LEADING:
                    leading_per_event.append(pt_values)
            total_events += _jagged_event_count(branch_data)

    leading_pt: np.ndarray | None = None
    if leading_per_event:
        leading_pt = np.concatenate(leading_per_event)

    return {
        "path": str(path),
        "tree": tree_name,
        "photon_branch": photon_branch,
        "channel": channel,
        "pt_reduction": reduction,
        "photon_pt_hist": photon_hist.tolist(),
        "leading_pt_per_event": leading_pt.tolist() if leading_pt is not None else None,
        "bin_edges": bin_edges.tolist(),
        "n_events_processed": total_events,
        "n_photon_objects_binned": total_objects,
        "n_entries_in_file": n_entries_in_file,
        "entry_stop": effective_stop,
        "pt_bins": int(pt_bins),
        "pt_max_gev": float(pt_max_gev),
    }


def read_electron_pt_from_tree(
    tree: Any,
    *,
    entry_start: int = 0,
    entry_stop: int | None = None,
    pt_reduction: str = PT_REDUCTION_FLATTEN,
    library: str = "ak",
) -> dict[str, Any]:
    """
    Read jagged ``Electron_pt`` from an uproot tree with safe reduction.

    Example (awkward)::

        electrons_pt = tree.arrays([\"Electron_pt\"], library=\"ak\")[\"Electron_pt\"]
        flat = reduce_jagged_pt_branch(electrons_pt, mode=\"flatten\")
        leading = reduce_jagged_pt_branch(electrons_pt, mode=\"leading\")
    """
    branches = _tree_branch_names(tree)
    if "Electron_pt" not in branches:
        raise KeyError("Electron_pt branch not present on tree")
    data = tree.arrays(
        ["Electron_pt"],
        entry_start=int(entry_start),
        entry_stop=entry_stop,
        library=library,
    )
    electrons_pt = data["Electron_pt"]
    mode = (pt_reduction or PT_REDUCTION_FLATTEN).strip().lower()
    reduced = reduce_jagged_pt_branch(electrons_pt, mode=mode)
    flat = reduce_jagged_pt_branch(electrons_pt, mode=PT_REDUCTION_FLATTEN)
    leading = reduce_jagged_pt_branch(electrons_pt, mode=PT_REDUCTION_LEADING)
    return {
        "electron_pt_jagged": electrons_pt,
        "pt_reduction": mode,
        "electron_pt_reduced": reduced,
        "flat_electrons_pt": flat,
        "leading_electron_pt": leading,
        "n_events": int(ak.num(electrons_pt, axis=0)),
        "n_electrons_flat": int(flat.size),
        "n_events_with_electron": int(leading.size),
    }


def accumulate_weighted_validation_observables(
    root_path: str | Path,
    event_weight_fn: Any,
    *,
    tree_name: str = "Events",
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    pt_bins: int = 20,
    pt_max_gev: float = 200.0,
    primary_lepton: str = "Muon",
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Chunked per-event weighted mod-7 + weighted pT histogram accumulation.

    ``event_weight_fn`` accepts a 1-D ``nTrueInt`` array and returns per-event weights.
    """
    resolved = resolve_root_path(str(root_path))
    path = Path(resolved) if resolved is not None else Path(root_path)
    if not path.is_file():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    primary_key = str(primary_lepton or "Muon").strip().title()
    n_branch, pt_branch = _LEPTON_BRANCH_PAIRS[primary_key]
    bin_edges = np.linspace(0.0, float(pt_max_gev), int(pt_bins) + 1)
    pt_hist = np.zeros(int(pt_bins), dtype=np.float64)
    mod7_hist = np.zeros(7, dtype=np.float64)
    total_event_weight = 0.0
    total_muons_weighted = 0.0
    n_events = 0
    pileup_branch: str | None = None
    pileup_hist = np.zeros(101, dtype=np.float64)

    with uproot.open(path) as handle:
        tree = handle[tree_name]
        available = _tree_branch_names(tree)
        if n_branch not in available or pt_branch not in available:
            raise KeyError(f"Missing {primary_key} branches in {path.name}")
        pileup_branch = resolve_pileup_branch(tree)
        branches = [n_branch, pt_branch]
        if pileup_branch:
            branches.append(pileup_branch)

        n_entries_in_file = int(tree.num_entries)
        effective_stop = (
            n_entries_in_file
            if not entry_stop or entry_stop <= 0
            else min(int(entry_stop), n_entries_in_file)
        )
        if verbose:
            print(
                f"[CERN CMS] Weighted scan: {path.name} "
                f"({effective_stop:,} events, pileup={pileup_branch or 'none'})"
            )

        for batch in tree.iterate(
            branches,
            step_size=int(chunk_size),
            entry_stop=effective_stop,
            library="np",
        ):
            n_primary = np.asarray(batch[n_branch], dtype=np.float64).ravel()
            n_chunk = int(n_primary.size)
            if n_chunk == 0:
                continue

            if pileup_branch and pileup_branch in batch:
                ntrue = np.asarray(batch[pileup_branch], dtype=np.float64).ravel()
                ntrue = np.clip(np.rint(ntrue), 0, 100).astype(int)
                if ntrue.size == n_chunk:
                    pileup_hist += np.bincount(ntrue, minlength=pileup_hist.size)[:101]
                    event_weights = np.asarray(event_weight_fn(ntrue), dtype=np.float64)
                else:
                    event_weights = np.ones(n_chunk, dtype=np.float64)
            else:
                event_weights = np.ones(n_chunk, dtype=np.float64)

            event_weights = np.where(np.isfinite(event_weights), event_weights, 0.0)
            mod7_hist += weighted_mod7_histogram(n_primary, event_weights)

            pt_flat = _flatten_jagged_np(batch[pt_branch])
            if pt_flat.size:
                n_per = np.clip(np.rint(n_primary), 0, 64).astype(np.int64)
                if int(n_per.sum()) == pt_flat.size:
                    muon_weights = np.repeat(event_weights, n_per)
                else:
                    muon_weights = np.ones(pt_flat.size, dtype=np.float64)
                chunk_hist, _ = np.histogram(
                    pt_flat,
                    bins=bin_edges,
                    weights=muon_weights,
                )
                pt_hist += chunk_hist
                total_muons_weighted += float(muon_weights.sum())

            total_event_weight += float(event_weights.sum())
            n_events += n_chunk

    mod7_fractions = (
        (mod7_hist / total_event_weight).tolist()
        if total_event_weight > 0
        else [0.0] * 7
    )
    uniform = total_event_weight / 7.0 if total_event_weight > 0 else 0.0
    chi2_mod7 = (
        float(np.sum((mod7_hist - uniform) ** 2 / max(uniform, 1e-9)))
        if uniform > 0
        else 0.0
    )

    return {
        "path": str(path),
        "primary_lepton": primary_key,
        "pileup_branch": pileup_branch,
        "n_events_processed": n_events,
        "total_event_weight": total_event_weight,
        "total_muons_weighted": total_muons_weighted,
        "mod7_histogram": mod7_hist.tolist(),
        "mod7_fractions": mod7_fractions,
        "chi2_mod7_vs_uniform": chi2_mod7,
        "muon_pt_hist": pt_hist,
        "bin_edges": bin_edges,
        "pileup_histogram": pileup_hist.tolist(),
    }


def extract_dimuon_kinematics_from_nanoaod(
    root_path: str | Path,
    *,
    tree_name: str = "Events",
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Chunked NanoAOD read of dimuon kinematics for geometric-floor / recoil-cut scans.

    Returns a normalized event table with ``leading_pt``, ``delta_phi``, ``delta_r``,
    and ``system_pt`` per dimuon event (``nMuon >= 2``).
    """
    resolved = resolve_root_path(str(root_path))
    path = Path(resolved) if resolved is not None else Path(root_path)
    if not path.is_file():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    required = ("nMuon", "Muon_pt", "Muon_eta", "Muon_phi")
    accum = {
        "leading_pt": [],
        "subleading_pt": [],
        "delta_phi": [],
        "delta_r": [],
        "system_pt": [],
        "n_muon_per_event": [],
        "n_true_int": [],
    }
    total_events = 0
    n_entries_in_file = 0
    effective_stop = 0

    with uproot.open(path) as handle:
        tree = handle[tree_name]
        available = _tree_branch_names(tree)
        missing = [name for name in required if name not in available]
        if missing:
            raise KeyError(
                f"Dimuon kinematics branches missing in {path.name}: {missing}"
            )
        n_entries_in_file = int(tree.num_entries)
        effective_stop = (
            n_entries_in_file
            if not entry_stop or entry_stop <= 0
            else min(int(entry_stop), n_entries_in_file)
        )
        pileup_branch = resolve_pileup_branch(tree)
        read_branches = list(required)
        if pileup_branch:
            read_branches.append(pileup_branch)
        if verbose:
            print(
                f"[CERN CMS] Dimuon kinematics: {path.name} "
                f"({effective_stop:,} / {n_entries_in_file:,} events, "
                f"pileup={pileup_branch or 'none'})"
            )
        for batch in tree.iterate(
            read_branches,
            step_size=int(chunk_size),
            entry_stop=effective_stop,
            library="np",
        ):
            n_muon = np.asarray(batch["nMuon"], dtype=np.int32)
            total_events += int(n_muon.size)
            dimuon_mask = n_muon >= 2
            if not np.any(dimuon_mask):
                continue
            pt_batch = np.asarray(batch["Muon_pt"], dtype=object)[dimuon_mask]
            eta_batch = np.asarray(batch["Muon_eta"], dtype=object)[dimuon_mask]
            phi_batch = np.asarray(batch["Muon_phi"], dtype=object)[dimuon_mask]
            chunk = _dimuon_kinematics_from_jagged(pt_batch, eta_batch, phi_batch)
            for key in ("leading_pt", "subleading_pt", "delta_phi", "delta_r", "system_pt"):
                accum[key].extend(chunk[key].tolist())
            n_in_chunk = int(chunk["leading_pt"].size)
            if n_in_chunk:
                accum["n_muon_per_event"].extend(
                    n_muon[dimuon_mask][:n_in_chunk].astype(int).tolist()
                )
            if pileup_branch and pileup_branch in batch:
                ntrue = np.asarray(batch[pileup_branch], dtype=np.float64).ravel()[dimuon_mask]
                n_in_chunk = int(chunk["leading_pt"].size)
                if ntrue.size >= n_in_chunk:
                    accum["n_true_int"].extend(
                        np.clip(np.rint(ntrue[:n_in_chunk]), 0, 100).astype(int).tolist()
                    )

    merged: dict[str, Any] = {}
    for key, vals in accum.items():
        if not vals:
            continue
        if key in {"n_true_int", "n_muon_per_event"}:
            merged[key] = np.asarray(vals, dtype=int)
        else:
            merged[key] = np.asarray(vals, dtype=float)
    if merged.get("n_true_int", np.array([])).size == 0:
        merged.pop("n_true_int", None)
    n_dimuon = int(merged["leading_pt"].size)
    if verbose:
        print(
            f"[CERN CMS] Dimuon events extracted: {n_dimuon:,} "
            f"from {total_events:,} scanned"
        )

    raw_payload = {
        "path": str(path),
        "tree": tree_name,
        "n_entries_in_file": n_entries_in_file,
        "entry_stop": effective_stop,
        "n_events_scanned": total_events,
        "n_dimuon_events": n_dimuon,
        **merged,
    }
    validated, report = validate_and_normalize_cms_dimuon_batch(
        raw_payload,
        filter_anomalies=True,
        strict=False,
        return_columnar=True,
        data_manager=get_runtime_data_manager(),
    )
    if validated:
        raw_payload = validated
    raw_payload["n_dimuon_events"] = report.n_accepted
    raw_payload["ingestion_validation"] = report.as_dict()
    if verbose and report.n_rejected:
        print(
            f"[CERN CMS] Ingestion validation: "
            f"{report.n_accepted:,} accepted, {report.n_rejected:,} rejected"
        )
    return raw_payload


def analyze_cms_nanoaod_full(
    root_path: str | Path,
    *,
    tree_name: str = "Events",
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    output_prefix: str = "tav_full_scan",
    pt_bins: int = 20,
    pt_max_gev: float = 200.0,
    save_tav_plots: bool = True,
    progress: ChunkedScanProgress | None = None,
    use_curses_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full-dataset chunked scan → ``tav_7fold_muon_analysis``."""
    resolved = resolve_root_path(str(root_path))
    if resolved is None:
        raise FileNotFoundError(f"No ROOT file for {root_path!r}")
    path = Path(resolved)
    scan = scan_cms_nanoaod_chunked(
        path,
        tree_name=tree_name,
        chunk_size=chunk_size,
        entry_stop=entry_stop,
        pt_bins=pt_bins,
        pt_max_gev=pt_max_gev,
        progress=progress,
        use_curses_progress=use_curses_progress,
        verbose=verbose,
    )

    dataset_slug = compose_dataset_slug(output_prefix, path.stem, "full")
    tav_results = tav_7fold_muon_analysis(
        muon_pt_hist=scan["muon_pt_hist"],
        bin_edges=scan["bin_edges"],
        n_muon_per_event=scan["n_muon_per_event"],
        photon_pt_hist=scan.get("electron_pt_hist"),
        output_dir=output_prefix,
        save_plots=save_tav_plots,
        verbose=verbose,
        dataset_slug=dataset_slug,
        n_events_processed=int(scan["n_events_processed"]),
        chunk_results=True,
    )

    summary: dict[str, Any] = {
        "action": "CMS Full-Dataset 7-Fold Muon Scan",
        "path": str(path),
        "scan": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in scan.items()
            if k != "n_muon_per_event"
        },
        "n_events_processed": scan["n_events_processed"],
        "timestamp": artifact_timestamp(),
        "tav_7fold_muon": tav_results,
        "tav_verdict": tav_results.get("verdict"),
        "seven_periodic": tav_results.get("seven_periodic"),
        "report_path": tav_results.get("report_path"),
        "report_paths": tav_results.get("report_paths"),
        "report_chunked": tav_results.get("report_chunked"),
    }
    if tav_results.get("plot_paths"):
        summary["tav_plot_paths"] = tav_results["plot_paths"]

    if verbose and tav_results.get("seven_periodic"):
        sp = tav_results["seven_periodic"]
        print("\n=== FULL DATASET TAV RESULTS ===")
        print(f"7-periodic significance: {sp.get('significance_sigma', 0):.2f} σ")
        print(f"Global verdict: {tav_results.get('verdict')}")
        if tav_results.get("report_chunked"):
            print(
                f"Results saved (chunked): {tav_results.get('report_path')} "
                f"({len(tav_results.get('report_paths') or []) - 1} parts)"
            )
        else:
            print(f"Results saved: {tav_results.get('report_path')}")

    return summary


def extract_cms_muon_inputs(
    data: Any,
    *,
    pt_bins: int = DEFAULT_PT_BINS,
    pt_max_gev: float = 200.0,
    photon_pt_reduction: str = PT_REDUCTION_FLATTEN,
) -> dict[str, Any]:
    """
    Build inputs for ``tav_7fold_muon_analysis`` from awkward NanoAOD arrays.
    """
    muon_pt = _flatten_branch(data["Muon_pt"]) if "Muon_pt" in data.fields else np.array([])
    n_muon = _per_event_array(data["nMuon"]) if "nMuon" in data.fields else None

    photon_hist = None
    photon_reduction = (photon_pt_reduction or PT_REDUCTION_FLATTEN).strip().lower()
    for branch in ("Photon_pt", "Electron_pt"):
        if branch in data.fields:
            photon_pt = reduce_jagged_pt_branch(
                data[branch],
                mode=photon_reduction,
            )
            if photon_pt.size:
                _, photon_edges = np.histogram(photon_pt, bins=pt_bins, range=(0.0, pt_max_gev))
                photon_hist, _ = np.histogram(photon_pt, bins=photon_edges)
                break

    if muon_pt.size:
        bin_edges = np.linspace(0.0, pt_max_gev, pt_bins + 1)
        muon_hist, _ = np.histogram(muon_pt, bins=bin_edges)
    else:
        bin_edges = np.linspace(0.0, pt_max_gev, pt_bins + 1)
        muon_hist = np.zeros(pt_bins, dtype=float)

    return {
        "muon_pt_hist": muon_hist,
        "bin_edges": bin_edges,
        "n_muon_per_event": n_muon,
        "photon_pt_hist": photon_hist,
        "photon_pt_reduction": photon_reduction,
        "n_muons_flat": int(muon_pt.size),
        "n_events": int(n_muon.size) if n_muon is not None else None,
    }


def analyze_cms_nanoaod(
    root_path: str | Path,
    *,
    tree_name: str = "Events",
    entry_stop: int = 50000,
    output_prefix: str = "cern_cms_nanoaod",
    run_tav_7fold: bool = True,
    save_tav_plots: bool = True,
    pt_bins: int = DEFAULT_PT_BINS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Read CMS NanoAOD, summarize branches, and run Tav 7-fold muon analysis."""
    path = Path(root_path)
    with uproot.open(path) as file:
        tree = file[tree_name]
        n_entries_in_file = int(tree.num_entries)
        branches = file[tree_name].keys()
        cols: list[str] = []
        for cand in (
            "Muon_pt",
            "Electron_pt",
            "Photon_pt",
            "nMuon",
            "nElectron",
        ):
            if cand in branches:
                cols.append(cand)
        if not cols:
            cols = list(branches)[:6]
        data = tree.arrays(cols, library="ak", entry_stop=entry_stop)

    summary: dict[str, Any] = {
        "path": str(path),
        "tree": tree_name,
        "branches_read": cols,
        "entry_stop": entry_stop,
        "n_entries_in_file": n_entries_in_file,
        "timestamp": artifact_timestamp(),
    }
    for col in cols:
        if col.startswith("n"):
            arr = _per_event_array(data[col])
        elif col in ("Electron_pt", "Photon_pt", "Muon_pt"):
            arr = reduce_jagged_pt_branch(
                data[col],
                mode=PT_REDUCTION_FLATTEN,
            )
        else:
            arr = _flatten_branch(data[col])
        summary[f"{col}_mean"] = float(np.mean(arr)) if arr.size else None
        summary[f"{col}_max"] = float(np.max(arr)) if arr.size else None

    tav_inputs = extract_cms_muon_inputs(data, pt_bins=pt_bins)
    n_events_read = int(tav_inputs["n_events"] or entry_stop)
    summary["n_events_processed"] = n_events_read
    summary["muon_pt_histogram"] = tav_inputs["muon_pt_hist"].tolist()
    summary["muon_pt_bin_edges"] = tav_inputs["bin_edges"].tolist()

    if run_tav_7fold:
        dataset_slug = compose_dataset_slug(output_prefix, path.stem)
        tav_results = tav_7fold_muon_analysis(
            muon_pt_hist=tav_inputs["muon_pt_hist"],
            bin_edges=tav_inputs["bin_edges"],
            n_muon_per_event=tav_inputs["n_muon_per_event"],
            photon_pt_hist=tav_inputs["photon_pt_hist"],
            output_dir=output_prefix,
            save_plots=save_tav_plots,
            verbose=verbose,
            dataset_slug=dataset_slug,
        )
        summary["tav_7fold_muon"] = tav_results
        summary["tav_verdict"] = tav_results.get("verdict")
        if tav_results.get("plot_paths"):
            summary["tav_plot_paths"] = tav_results["plot_paths"]
        summary["report_path"] = tav_results.get("report_path")
        if verbose and n_entries_in_file > n_events_read:
            print(
                f"[CERN CMS] Sampled {n_events_read:,} / {n_entries_in_file:,} events "
                f"(set entry_stop=0 or use CMS Full-Dataset 7-Fold Scan for entire file)"
            )
    else:
        out = artifact_path(TestSlug.CERN, compose_dataset_slug(output_prefix), "report", "json")
        from menus.astronomical.desi.json_util import write_json

        write_json(out, summary, indent=2, sort_keys=True)
        summary["report_path"] = str(out)

    return summary


def _analyze_alice_esd_classic(
    root_path: Path,
    *,
    entry_stop: int = 50000,
) -> dict[str, Any]:
    """Summarize legacy ALICE ESD (esdTree) when O2 AO2D trees are absent."""
    with uproot.open(root_path) as file:
        tree_name = "esdTree" if "esdTree" in file else next(iter(file.keys()))
        tree = file[tree_name]
        cols = [c for c in tree.keys() if "fNtracks" in c]
        if not cols:
            raise KeyError(f"No readable summary branches in {tree_name}")
        cols = [cols[0]]
        data = tree.arrays(cols, library="np", entry_stop=entry_stop)

    ntracks = np.asarray(data[cols[0]], dtype=float)
    return {
        "track_tree": tree_name,
        "n_events": int(ntracks.size),
        "n_tracks": int(np.nansum(ntracks)),
        "ntracks_mean": float(np.nanmean(ntracks)) if ntracks.size else None,
        "ntracks_max": float(np.nanmax(ntracks)) if ntracks.size else None,
        "pt_mean": None,
        "eta_mean": None,
        "analysis_mode": "esd_classic",
    }


def analyze_alice_root(
    root_path: str | Path,
    *,
    entry_stop: int = 50000,
    output_prefix: str = "cern_alice_root",
) -> dict[str, Any]:
    """Load ALICE AO2D (O2track) or legacy ESD (esdTree) ROOT samples."""
    path = Path(root_path)
    try:
        from menus.particle.alice.data_pull import load_alice_o2_root

        df, track_key = load_alice_o2_root(str(path), entry_stop=entry_stop)
        summary = {
            "path": str(path),
            "track_tree": track_key,
            "n_tracks": int(len(df)),
            "pt_mean": float(df["Pt"].mean()) if len(df) else None,
            "eta_mean": float(df["Eta"].mean()) if len(df) else None,
            "analysis_mode": "o2track",
            "timestamp": artifact_timestamp(),
        }
    except (KeyError, ValueError) as exc:
        print(f"[CERN OPENDATA] O2 track read unavailable ({exc}); using esdTree fallback")
        summary = _analyze_alice_esd_classic(path, entry_stop=entry_stop)
        summary["path"] = str(path)
        summary["timestamp"] = artifact_timestamp()
    out = artifact_path(TestSlug.CERN, compose_dataset_slug(output_prefix), "report", "json")
    from menus.astronomical.desi.json_util import write_json

    write_json(out, summary, indent=2, sort_keys=True)
    summary["report_path"] = str(out)
    return summary


def run_analysis(
    target_or_path: str,
    *,
    analysis: str = "cms_nanoaod",
    entry_stop: int = 50000,
    output_prefix: str = "cern_analysis",
    run_tav_7fold: bool = True,
    save_tav_plots: bool = True,
    full_scan: bool = False,
    chunk_size: int = 500_000,
    pt_bins: int = 20,
    progress: ChunkedScanProgress | None = None,
    use_curses_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    root = resolve_root_path(target_or_path)
    if root is None:
        raise FileNotFoundError(
            f"No cached ROOT for {target_or_path!r}. "
            "Run Pull Datasets from Open Archives first."
        )
    if analysis == "alice_esd":
        return analyze_alice_root(root, entry_stop=entry_stop, output_prefix=output_prefix)
    if full_scan or entry_stop <= 0:
        return analyze_cms_nanoaod_full(
            root,
            chunk_size=chunk_size,
            entry_stop=None if entry_stop <= 0 else entry_stop,
            output_prefix=output_prefix,
            pt_bins=pt_bins,
            save_tav_plots=save_tav_plots,
            progress=progress,
            use_curses_progress=use_curses_progress,
            verbose=verbose,
        )
    return analyze_cms_nanoaod(
        root,
        entry_stop=entry_stop,
        output_prefix=output_prefix,
        run_tav_7fold=run_tav_7fold,
        save_tav_plots=save_tav_plots,
        pt_bins=pt_bins,
        verbose=verbose,
    )