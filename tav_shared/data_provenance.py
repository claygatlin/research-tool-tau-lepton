#!/usr/bin/env python3
"""
Data provenance registry for Tav-Superblock / research_tool modules.

Reports whether each menu action uses genuine repository data (cached locally),
remote-only targets, synthetic validation data, or purely theoretical inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT as PROJECT_ROOT

DATASETS_DESI = PROJECT_ROOT / "datasets" / "desi"
COBAYA_DR2 = DATASETS_DESI / "bao_data" / "desi_bao_dr2"
SPARC_DATA = PROJECT_ROOT / "data" / "sparc"
SPARC_DATASETS = PROJECT_ROOT / "datasets" / "sparc"
PANTHEON = DATASETS_DESI / "pantheon_plus" / "Pantheon+SH0ES.dat"

DataClass = Literal["real_cached", "real_remote", "synthetic", "theoretical", "mixed"]


@dataclass
class ProvenanceEntry:
    module: str
    action: str
    data_class: DataClass
    repository: str | None
    local_path: str | None
    data_available: bool
    notes: str


def _count_glob(directory: Path, pattern: str) -> int:
    if not directory.is_dir():
        return 0
    return len(list(directory.glob(pattern)))


def dataset_availability() -> dict[str, Any]:
    """Snapshot of whether key on-disk datasets are present."""
    desi_tables = _count_glob(COBAYA_DR2, "desi_gaussian_bao_*_mean.txt")
    sparc_count = _count_glob(SPARC_DATA, "*.csv") + _count_glob(SPARC_DATASETS, "*.csv")
    legacy_bao = _count_glob(DATASETS_DESI / "bao_data", "sdss_*.dat")
    return {
        "desi_dr2_cobaya_tables": desi_tables,
        "desi_dr2_path": str(COBAYA_DR2),
        "desi_dr2_available": desi_tables > 0,
        "legacy_bao_tables": legacy_bao,
        "sparc_galaxy_csv_count": sparc_count,
        "sparc_available": sparc_count > 0,
        "pantheon_plus_available": PANTHEON.is_file(),
        "pantheon_plus_path": str(PANTHEON),
    }


def _entries_tsb_research(avail: dict[str, Any]) -> list[ProvenanceEntry]:
    module = "Tau-Superblock Research Engine"
    mocks_path = str(PROJECT_ROOT / "artifacts" / "mocks")
    desi_ok = avail["desi_dr2_available"]
    return [
        ProvenanceEntry(
            module,
            "Generate Mocks",
            "synthetic",
            None,
            mocks_path,
            True,
            "Cylinder oscillation template on s-grid with SoundHorizon anchor and synthetic residual diagnostics.",
        ),
        ProvenanceEntry(
            module,
            "Generate Full DESI Summary Dashboard",
            "mixed",
            "CobayaSampler/bao_data desi_bao_dr2",
            str(PROJECT_ROOT / "artifacts" / "desi_dashboard"),
            desi_ok,
            "ACF, Q-Q, residuals vs fitted, SoundHorizon curve; DESI fit residuals when cached.",
        ),
        ProvenanceEntry(
            module,
            "Evolve Cylinder State",
            "theoretical",
            None,
            None,
            True,
            "Clockwork evolution with twistor, SoundHorizon, and exclusion coupling.",
        ),
        ProvenanceEntry(
            module,
            "Run Residual Diagnostics",
            "mixed",
            "CobayaSampler/bao_data desi_bao_dr2",
            str(COBAYA_DR2),
            desi_ok,
            "Default: DESI fit residuals; optional cylinder-evolution synthetic mode.",
        ),
        ProvenanceEntry(
            module,
            "Calibrate SoundHorizon",
            "theoretical",
            None,
            None,
            True,
            "Φ-anchored r_d calibration; compares γ-swing vs legacy pipeline.",
        ),
    ]


def _entries_desi(avail: dict[str, Any]) -> list[ProvenanceEntry]:
    repo = "CobayaSampler/bao_data + data.desi.lbl.gov"
    path = str(COBAYA_DR2)
    ok = avail["desi_dr2_available"]
    real: DataClass = "real_cached" if ok else "real_remote"
    return [
        ProvenanceEntry("TAU_SB_DESI", "Real DESI DR2 Scan (Cobaya)", real, repo, path, ok,
                        "Primary production scan; fails if DR2 tables missing."),
        ProvenanceEntry("TAU_SB_DESI", "Periodicity Scan (1/7 Tracks)", real, repo, path, ok,
                        "Lomb-Scargle on DR2 residuals."),
        ProvenanceEntry("TAU_SB_DESI", "Model Compare (ΛCDM vs aDE vs Tau-SB)", real, repo, path, ok,
                        "χ² comparison on DR2 vector."),
        ProvenanceEntry("TAU_SB_DESI", "MCMC Posteriors (emcee)", real, repo, path, ok,
                        "MCMC on DR2 data."),
        ProvenanceEntry("TAU_SB_DESI", "Healpy Dipole Fit", "mixed", repo, path, ok,
                        "Real BAO vector; healpy map uses synthetic sky patches unless user supplies RA/Dec."),
        ProvenanceEntry("TAU_SB_DESI", "Change-Point Detection", real, repo, path, ok,
                        "Changepoint search on DR2 expansion residuals."),
        ProvenanceEntry("TAU_SB_DESI", "Injection/Recovery Tests", "synthetic", None, None, True,
                        "Tau-SB signal injected into DR2 noise model (validation only)."),
        ProvenanceEntry("TAU_SB_DESI", "Joint DESI+SN+Planck Fit", "mixed", repo, path, ok,
                        "DESI DR2 + Pantheon+ table if cached."),
        ProvenanceEntry("TAU_SB_DESI", "Batch Tracer Scan (All DR2)", real, repo, path, ok,
                        "Loops all available DR2 tracers."),
        ProvenanceEntry("TAU_SB_DESI", "Full Production Pipeline", "mixed", repo, path, ok,
                        "DR2 scan + optional synthetic injection / mock MCMC debias trials."),
    ]


def _entries_sparc(avail: dict[str, Any]) -> list[ProvenanceEntry]:
    repo = "https://astroweb.case.edu/SPARC/"
    path = str(SPARC_DATASETS if SPARC_DATASETS.is_dir() else SPARC_DATA)
    ok = avail["sparc_available"]
    dc: DataClass = "real_cached" if ok else "real_remote"
    return [
        ProvenanceEntry("SPARC", "Pull Batch from Repository", dc, repo, path, ok,
                        "Downloads MassModels_Lelli2016c.mrt; per-galaxy CSVs."),
        ProvenanceEntry("SPARC", "Rotation Curve Analysis", dc, repo, path, ok,
                        "Uses cached galaxy CSVs under data/sparc or datasets/sparc."),
    ]


def _entries_integrator() -> list[ProvenanceEntry]:
    return [
        ProvenanceEntry(
            "INTEGRATOR", "Fetch Planck Maps", "real_remote",
            "ESA Planck Legacy Archive", str(PROJECT_ROOT / "datasets" / "integrator"),
            False, "Download on fetch; cached FITS after pull.",
        ),
        ProvenanceEntry(
            "INTEGRATOR", "Run Integrator Pipeline", "mixed",
            "Planck + SPARC master table", None, True,
            "Mixed cached maps/tables depending on prior fetches.",
        ),
    ]


def _entries_frb() -> list[ProvenanceEntry]:
    return [
        ProvenanceEntry(
            "FRB_WEB", "Fetch CHIME/FRB Catalog", "real_remote",
            "CHIME/FRB + CDS mirrors", str(PROJECT_ROOT / "datasets" / "frb"),
            False, "Network download; void catalog from public URL.",
        ),
        ProvenanceEntry(
            "FRB_WEB", "FRB Web Analysis", "mixed", "CHIME/FRB", None, True,
            "Real catalog when fetched; some sightline grids are deterministic synthetic.",
        ),
    ]


def _entries_planck() -> list[ProvenanceEntry]:
    return [
        ProvenanceEntry(
            "PLANCK_CMB", "Fetch / Analyze CMB", "real_remote",
            "Planck Legacy Archive", str(PROJECT_ROOT / "datasets" / "planck"),
            False, "Requires integrator or planck fetch first.",
        ),
    ]


def provenance_entries(
    *,
    scope: Literal["all", "tsb_research", "desi", "sparc"] = "all",
) -> list[ProvenanceEntry]:
    avail = dataset_availability()
    entries: list[ProvenanceEntry] = []
    if scope in ("all", "tsb_research"):
        entries.extend(_entries_tsb_research(avail))
    if scope in ("all", "desi"):
        entries.extend(_entries_desi(avail))
    if scope in ("all", "sparc"):
        entries.extend(_entries_sparc(avail))
    if scope == "all":
        entries.extend(_entries_integrator())
        entries.extend(_entries_frb())
        entries.extend(_entries_planck())
    return entries


def data_provenance_report(
    *,
    scope: Literal["all", "tsb_research", "desi", "sparc"] = "all",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build a JSON-serializable provenance report for research-tool menu actions.

    Parameters
    ----------
    scope:
        Limit to one module family or ``all``.
    verbose:
        Print a human-readable table to stdout.
    """
    avail = dataset_availability()
    entries = provenance_entries(scope=scope)
    by_class: dict[str, int] = {}
    for entry in entries:
        by_class[entry.data_class] = by_class.get(entry.data_class, 0) + 1

    report: dict[str, Any] = {
        "scope": scope,
        "dataset_availability": avail,
        "summary_by_data_class": by_class,
        "entries": [asdict(e) for e in entries],
    }

    if verbose:
        print("\n=== Data Provenance Report ===")
        print(f"Scope: {scope}")
        print("\nLocal dataset cache:")
        print(f"  DESI DR2 Cobaya tables : {avail['desi_dr2_cobaya_tables']} ({avail['desi_dr2_path']})")
        print(f"  Legacy BAO tables      : {avail['legacy_bao_tables']}")
        print(f"  SPARC galaxy CSVs      : {avail['sparc_galaxy_csv_count']}")
        print(f"  Pantheon+ available    : {avail['pantheon_plus_available']}")
        print("\nActions:")
        for entry in entries:
            status = "OK" if entry.data_available else "MISSING CACHE"
            print(
                f"  [{entry.module}] {entry.action}\n"
                f"    class={entry.data_class} | {status} | {entry.notes}"
            )
        print("=== End Provenance Report ===\n")

    return report


if __name__ == "__main__":
    data_provenance_report(verbose=True)