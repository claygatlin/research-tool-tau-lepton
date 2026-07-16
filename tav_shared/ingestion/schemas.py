"""
Pydantic schemas for Tau-Superblock ingestion validation.

Defines mandatory fields and types for CMS dimuon tracks, BBN empirical
payloads, and fractal-tau likelihood inputs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tav_shared.ingestion.config import load_ingestion_bounds


def _pair_bounds(key: str, section: str) -> tuple[float, float]:
    bounds = load_ingestion_bounds().get(section, {})
    raw = bounds.get(key, [0.0, 1.0e30])
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return float(raw[0]), float(raw[1])
    return 0.0, 1.0e30


def _int_pair_bounds(key: str, section: str) -> tuple[int, int]:
    lo, hi = _pair_bounds(key, section)
    return int(lo), int(hi)


class CMSDimuonEventRecord(BaseModel):
    """Single validated dimuon event (canonical internal names)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    leading_pt_gev: float = Field(..., ge=0.0)
    subleading_pt_gev: float = Field(..., ge=0.0)
    delta_phi_rad: float = Field(..., ge=0.0)
    delta_r: float = Field(..., ge=0.0)
    system_pt_gev: float = Field(..., ge=0.0)
    n_muon: int = Field(..., ge=2)
    n_true_int: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _physical_ranges(self) -> CMSDimuonEventRecord:
        b = load_ingestion_bounds()["cms_dimuon"]
        checks: list[tuple[str, float, tuple[float, float]]] = [
            ("leading_pt_gev", self.leading_pt_gev, tuple(b["leading_pt_gev"])),
            ("subleading_pt_gev", self.subleading_pt_gev, tuple(b["subleading_pt_gev"])),
            ("delta_phi_rad", self.delta_phi_rad, tuple(b["delta_phi_rad"])),
            ("delta_r", self.delta_r, tuple(b["delta_r"])),
            ("system_pt_gev", self.system_pt_gev, tuple(b["system_pt_gev"])),
        ]
        for name, value, (lo, hi) in checks:
            if not (float(lo) <= value <= float(hi)):
                raise ValueError(f"{name}={value} outside [{lo}, {hi}]")
        n_lo, n_hi = _int_pair_bounds("n_muon", "cms_dimuon")
        if not (n_lo <= self.n_muon <= n_hi):
            raise ValueError(f"n_muon={self.n_muon} outside [{n_lo}, {n_hi}]")
        if self.n_true_int is not None:
            pu_lo, pu_hi = _int_pair_bounds("n_true_int", "cms_dimuon")
            if not (pu_lo <= self.n_true_int <= pu_hi):
                raise ValueError(
                    f"n_true_int={self.n_true_int} outside [{pu_lo}, {pu_hi}]"
                )
        if self.subleading_pt_gev > self.leading_pt_gev + 1e-9:
            raise ValueError("subleading_pt_gev exceeds leading_pt_gev")
        return self


class CMSDimuonBatch(BaseModel):
    """Batch of dimuon events plus optional metadata."""

    model_config = ConfigDict(extra="allow")

    events: list[CMSDimuonEventRecord] = Field(default_factory=list)
    source_format: str = "internal"
    path: str | None = None


class BBNAbundanceObservation(BaseModel):
    """Value + uncertainty pair for a BBN observable."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    value: float
    uncertainty: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def _abundance_bounds(self) -> BBNAbundanceObservation:
        b = load_ingestion_bounds()["bbn"]
        lo = float(b["abundance_min"])
        hi = float(b["abundance_max"])
        unc_lo = float(b["uncertainty_min"])
        if not (lo <= self.value <= hi):
            raise ValueError(f"abundance value {self.value} outside [{lo}, {hi}]")
        if self.uncertainty < unc_lo:
            raise ValueError(f"uncertainty {self.uncertainty} below {unc_lo}")
        return self


class NeutronLifetimeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bottle: float
    beam: float
    pdg: float
    unc: float = Field(..., gt=0.0)
    tension_sigma: float

    @model_validator(mode="after")
    def _lifetime_bounds(self) -> NeutronLifetimeRecord:
        b = load_ingestion_bounds()["bbn"]
        lo, hi = tuple(b["neutron_lifetime_s"])
        for label, val in (
            ("bottle", self.bottle),
            ("beam", self.beam),
            ("pdg", self.pdg),
        ):
            if not (float(lo) <= val <= float(hi)):
                raise ValueError(f"neutron_lifetime.{label}={val} outside [{lo}, {hi}]")
        max_tension = float(b["tension_sigma_max"])
        if abs(self.tension_sigma) > max_tension:
            raise ValueError(
                f"tension_sigma={self.tension_sigma} exceeds {max_tension}"
            )
        return self


class BBNEmpiricalPayload(BaseModel):
    """Canonical BBN confrontation anchors."""

    model_config = ConfigDict(extra="ignore")

    neutron_lifetime: NeutronLifetimeRecord
    bbn_abundances: dict[str, BBNAbundanceObservation]

    @field_validator("bbn_abundances")
    @classmethod
    def _required_keys(
        cls, value: dict[str, BBNAbundanceObservation]
    ) -> dict[str, BBNAbundanceObservation]:
        required = {"D_H", "Y_p_He4", "Li7_H_obs", "Li7_H_std_theory"}
        missing = required - set(value.keys())
        if missing:
            raise ValueError(f"bbn_abundances missing keys: {sorted(missing)}")
        return value


class FractalTauParamsSchema(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    winding_density: float
    fractal_level: float
    phase_slip_alpha: float

    @model_validator(mode="after")
    def _param_bounds(self) -> FractalTauParamsSchema:
        b = load_ingestion_bounds()["fractal_tau"]
        for key in ("winding_density", "fractal_level", "phase_slip_alpha"):
            lo, hi = tuple(b[key])
            val = getattr(self, key)
            if not (float(lo) <= val <= float(hi)):
                raise ValueError(f"{key}={val} outside [{lo}, {hi}]")
        return self


class FractalTauObservedSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mass_gap_mev: float
    sigma_mass_gap_mev: float = Field(default=5.0, gt=0.0)
    sigma_qcd_mev: float = Field(default=5.0, gt=0.0)
    spectral_dim: float
    sigma_spectral_dim: float = Field(default=0.15, gt=0.0)
    n_hier: float = Field(default=3.0, gt=0.0)
    torsion_factor: float | None = Field(default=None, gt=0.0)
    likelihood_mode: str = "dual"

    @model_validator(mode="after")
    def _observed_bounds(self) -> FractalTauObservedSchema:
        b = load_ingestion_bounds()["fractal_tau"]
        lo, hi = tuple(b["mass_gap_mev"])
        if not (float(lo) <= self.mass_gap_mev <= float(hi)):
            raise ValueError(
                f"mass_gap_mev={self.mass_gap_mev} outside [{lo}, {hi}]"
            )
        lo_ds, hi_ds = tuple(b["spectral_dim"])
        if not (float(lo_ds) <= self.spectral_dim <= float(hi_ds)):
            raise ValueError(
                f"spectral_dim={self.spectral_dim} outside [{lo_ds}, {hi_ds}]"
            )
        return self


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "cms_dimuon_event": CMSDimuonEventRecord,
    "bbn_empirical": BBNEmpiricalPayload,
    "fractal_tau_params": FractalTauParamsSchema,
    "fractal_tau_observed": FractalTauObservedSchema,
}


def schema_for(name: str) -> type[BaseModel]:
    try:
        return SCHEMA_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown ingestion schema: {name}") from exc