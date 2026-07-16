"""
Ingestion validation pipeline: schema enforcement, anomaly filtering, reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from tav_shared.ingestion.config import load_ingestion_bounds
from tav_shared.ingestion.normalize import (
    cms_batch_to_columnar,
    detect_source_format,
    load_payload_from_path,
    normalize_bbn_payload,
    normalize_cms_dimuon_batch,
    normalize_fractal_observed,
    normalize_fractal_params,
)
from tav_shared.ingestion.schemas import (
    BBNEmpiricalPayload,
    CMSDimuonEventRecord,
    FractalTauObservedSchema,
    FractalTauParamsSchema,
    schema_for,
)
from tav_shared.ingestion.data_manager import DataManager, get_runtime_data_manager

ModelT = TypeVar("ModelT", bound=BaseModel)


def _resolve_data_manager(data_manager: DataManager | None) -> DataManager:
    """Resolve a shared runtime manager for every ingestion pipeline call."""
    return data_manager or get_runtime_data_manager()


def _load_source_payload(
    source: str,
    *,
    data_manager: DataManager,
) -> Any:
    """Prefer existing JSON/CSV loader, then fall back to DataManager loaders."""
    try:
        return load_payload_from_path(source)
    except ValueError:
        key = Path(source).stem or "ingestion_source"
        return data_manager.fetch(key, source)


@dataclass
class IngestionValidationReport:
    """Summary of an ingestion validation pass."""

    schema_name: str
    source_format: str
    n_received: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    ok: bool = True
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    normalized: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "source_format": self.source_format,
            "n_received": self.n_received,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "ok": self.ok,
            "anomalies": self.anomalies,
            "missing_fields": self.missing_fields,
            "messages": self.messages,
        }


def _record_anomaly(
    report: IngestionValidationReport,
    *,
    index: int | None,
    reason: str,
    detail: Any = None,
) -> None:
    report.anomalies.append(
        {
            "index": index,
            "reason": reason,
            "detail": str(detail) if detail is not None else None,
        }
    )


def validate_generic_records(
    records: Sequence[Mapping[str, Any]],
    schema_name: str,
    *,
    filter_invalid: bool = True,
) -> IngestionValidationReport:
    """Validate a sequence of dict records against a registered schema."""
    model_cls = schema_for(schema_name)
    report = IngestionValidationReport(
        schema_name=schema_name,
        source_format="records",
        n_received=len(records),
    )
    accepted: list[dict[str, Any]] = []
    for idx, row in enumerate(records):
        try:
            model = model_cls.model_validate(dict(row))
            accepted.append(model.model_dump())
        except ValidationError as exc:
            report.n_rejected += 1
            _record_anomaly(report, index=idx, reason="schema_validation", detail=exc)
            if not filter_invalid:
                report.ok = False
        except ValueError as exc:
            report.n_rejected += 1
            _record_anomaly(report, index=idx, reason="physical_range", detail=exc)
            if not filter_invalid:
                report.ok = False

    report.n_accepted = len(accepted)
    report.normalized = {"records": accepted}
    if report.n_accepted == 0 and report.n_received > 0:
        report.ok = False
        report.messages.append("All records rejected by validation")
    return report


def validate_and_normalize_cms_dimuon_batch(
    raw: Mapping[str, Any] | list[Mapping[str, Any]] | str | Any,
    *,
    source_path: str | None = None,
    filter_anomalies: bool = True,
    strict: bool = False,
    return_columnar: bool = True,
    data_manager: DataManager | None = None,
) -> tuple[dict[str, Any], IngestionValidationReport]:
    """
    Validate CMS dimuon kinematics before solver / recoil-cut pipelines.

    Returns normalized payload (columnar by default) and a validation report.
    """
    manager = _resolve_data_manager(data_manager)
    if isinstance(raw, (str,)):
        payload = _load_source_payload(raw, data_manager=manager)
        source_path = source_path or raw
    else:
        payload = raw

    fmt = detect_source_format(payload, path=source_path)
    try:
        normalized_batch = normalize_cms_dimuon_batch(payload, source_format=fmt)
    except (TypeError, ValueError) as exc:
        report = IngestionValidationReport(
            schema_name="cms_dimuon_event",
            source_format=fmt,
            ok=False,
            messages=[str(exc)],
        )
        if strict:
            raise ValueError(f"CMS dimuon normalization failed: {exc}") from exc
        return {}, report

    events = normalized_batch.get("events", [])
    report = IngestionValidationReport(
        schema_name="cms_dimuon_event",
        source_format=str(normalized_batch.get("source_format", fmt)),
        n_received=len(events),
    )

    accepted: list[dict[str, Any]] = []
    for idx, row in enumerate(events):
        try:
            model = CMSDimuonEventRecord.model_validate(row)
            accepted.append(model.model_dump())
        except (ValidationError, ValueError) as exc:
            report.n_rejected += 1
            _record_anomaly(report, index=idx, reason="cms_dimuon", detail=exc)

    report.n_accepted = len(accepted)
    min_required = int(
        load_ingestion_bounds()["cms_dimuon"].get("min_events_after_filter", 10)
    )
    if report.n_accepted < min_required:
        report.ok = False
        report.messages.append(
            f"Accepted events ({report.n_accepted}) below minimum ({min_required})"
        )

    if filter_anomalies:
        normalized_batch["events"] = accepted
    elif report.n_rejected > 0:
        report.ok = False

    if return_columnar:
        out = cms_batch_to_columnar(normalized_batch)
    else:
        out = normalized_batch
    out["ingestion_validation"] = report.as_dict()
    report.normalized = {"n_dimuon_events": report.n_accepted}

    if strict and not report.ok:
        raise ValueError(
            f"CMS dimuon ingestion validation failed: {report.messages or report.anomalies[:3]}"
        )
    return out, report


def validate_and_normalize_bbn_payload(
    raw: Mapping[str, Any] | str | Any,
    *,
    source_path: str | None = None,
    strict: bool = True,
    data_manager: DataManager | None = None,
) -> tuple[dict[str, Any], IngestionValidationReport]:
    """Validate BBN empirical confrontation anchors."""
    manager = _resolve_data_manager(data_manager)
    if isinstance(raw, (str,)):
        payload = _load_source_payload(raw, data_manager=manager)
        source_path = source_path or raw
    else:
        payload = raw

    fmt = detect_source_format(payload, path=source_path)
    report = IngestionValidationReport(
        schema_name="bbn_empirical",
        source_format=fmt,
        n_received=1,
    )

    try:
        canonical = normalize_bbn_payload(payload)
        model = BBNEmpiricalPayload.model_validate(canonical)
        out = _bbn_model_to_legacy_dict(model)
        report.n_accepted = 1
        report.normalized = out
    except (TypeError, ValueError, ValidationError) as exc:
        report.n_rejected = 1
        report.ok = False
        report.n_accepted = 0
        _record_anomaly(report, index=0, reason="bbn_empirical", detail=exc)
        report.messages.append(str(exc))
        if strict:
            raise ValueError(f"BBN ingestion validation failed: {exc}") from exc
        return dict(payload) if isinstance(payload, Mapping) else {}, report

    out["ingestion_validation"] = report.as_dict()
    return out, report


def _bbn_model_to_legacy_dict(model: BBNEmpiricalPayload) -> dict[str, Any]:
    """Preserve legacy tuple layout for confrontation scanners."""
    nl = model.neutron_lifetime
    abund = model.bbn_abundances
    return {
        "neutron_lifetime": {
            "bottle": nl.bottle,
            "beam": nl.beam,
            "pdg": nl.pdg,
            "unc": nl.unc,
            "tension_sigma": nl.tension_sigma,
        },
        "bbn_abundances": {
            key: (obs.value, obs.uncertainty)
            for key, obs in abund.items()
        },
    }


def validate_fractal_tau_for_likelihood(
    params: Mapping[str, float] | Sequence[float] | Any,
    observed: Mapping[str, float] | Any | None = None,
    *,
    observed_defaults: Mapping[str, float] | None = None,
    strict: bool = True,
    data_manager: DataManager | None = None,
) -> tuple[dict[str, float], dict[str, Any], IngestionValidationReport]:
    """
    Validate fractal-tau parameters and observed anchors before log_likelihood.

    Returns (validated_params, validated_observed, report).
    """
    _resolve_data_manager(data_manager)
    report = IngestionValidationReport(
        schema_name="fractal_tau_params",
        source_format=detect_source_format(params),
        n_received=2,
    )
    validated_params: dict[str, float] = {}
    validated_observed: dict[str, Any] = {}

    try:
        p_norm = normalize_fractal_params(params)
        p_model = FractalTauParamsSchema.model_validate(p_norm)
        validated_params = p_model.model_dump()
        report.n_accepted += 1
    except (TypeError, ValueError, ValidationError) as exc:
        report.n_rejected += 1
        report.ok = False
        _record_anomaly(report, index=0, reason="fractal_params", detail=exc)
        report.messages.append(f"params: {exc}")
        if strict:
            raise ValueError(f"Fractal tau param validation failed: {exc}") from exc

    try:
        o_norm = normalize_fractal_observed(observed, defaults=observed_defaults)
        o_model = FractalTauObservedSchema.model_validate(o_norm)
        validated_observed = {
            k: v for k, v in o_model.model_dump().items() if v is not None
        }
        report.n_accepted += 1
    except (TypeError, ValueError, ValidationError) as exc:
        report.n_rejected += 1
        report.ok = False
        _record_anomaly(report, index=1, reason="fractal_observed", detail=exc)
        report.messages.append(f"observed: {exc}")
        if strict:
            raise ValueError(f"Fractal tau observed validation failed: {exc}") from exc

    report.normalized = {
        "params": validated_params,
        "observed": validated_observed,
    }
    return validated_params, validated_observed, report