"""Shared post-reduction physics checks for the CLI and review notebook."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .models import AxisymmetricRFField
from .pipeline import ReductionResult
from .quality import qualify_axisymmetric_field
from .steady_state import prepare_fill_treatment
from .workflow import (
    PRODUCTION_QUALITY_GATE_VERSION,
    AxisymmetricGridPlan,
    TailQualification,
    production_quality_gates,
    qualify_evanescent_tail,
    resolve_production_quality_status,
)


@dataclass(frozen=True)
class ReductionAssessment:
    """Measured fields after fill treatment, tail evidence and qualification.

    No extrapolated fields are produced here. Call ``apply_tail_policy`` only
    after inspecting ``quality``; the modal policy still refuses a failed tail.
    """

    measured_field: AxisymmetricRFField
    tail: TailQualification
    quality: dict[str, Any]


def assess_reduced_field(
    reduction: ReductionResult,
    plan: AxisymmetricGridPlan,
    *,
    simulated_duration_s: float,
    cathode_origin_report: Mapping[str, Any],
    source_integrity_gate: Mapping[str, Any],
    fill_policy: str = "auto",
    drive_start_time_s: float | None = None,
    tail_policy: str = "modal",
    tail_fit_start_m: float = 0.03,
    wall_guard_cells: int = 2,
    axis_guard_cells: int = 1,
    beam_core_radius_m: float = 2.0e-3,
    requested_quality_status: str = "auto",
) -> ReductionAssessment:
    """Apply one fill treatment and evaluate one consistent set of quality gates.

    Tail diagnostics are evaluated on the same amplitude reference as Maxwell
    checks and the output fields. Callers supply evidence of source integrity;
    this function never infers a verified checksum from a file name.
    """
    variants = {"modal": "modal", "zero": "zero_tail_control", "none": "measured_only"}
    if tail_policy not in variants:
        raise ValueError("tail_policy must be 'modal', 'zero', or 'none'")
    reduction_report = reduction.report.as_dict()
    field, fill_record = prepare_fill_treatment(
        reduction.field,
        reduction_report["temporal"],
        simulated_duration_s=simulated_duration_s,
        policy=fill_policy,
        drive_start_time_s=drive_start_time_s,
    )
    tail = qualify_evanescent_tail(
        field, target_z_max_m=plan.target_z_max_m, fit_start_z_m=tail_fit_start_m,
    )
    numerical = qualify_axisymmetric_field(
        field, plan.measured_vacuum_mask,
        wall_guard_cells=wall_guard_cells,
        axis_guard_cells=axis_guard_cells,
        tail_start_z_m=tail_fit_start_m,
        beam_core_radius_m=beam_core_radius_m,
    )
    gates = production_quality_gates(
        reduction_report, numerical, tail.as_dict(), cathode_origin_report,
    )
    gates["fill_treatment"] = {
        "passed": bool(fill_record["production_eligible"]),
        "metric": "stationary saved window or explicitly selected transient/model control",
        "policy": fill_policy,
        "steady_state_verified": False,
        "gate_version": PRODUCTION_QUALITY_GATE_VERSION,
        "name": "fill_treatment",
    }
    gates["source_integrity"] = dict(source_integrity_gate)
    quality = {
        "status": resolve_production_quality_status(requested_quality_status, gates),
        "gates": gates,
        "failed_gates": [name for name, gate in gates.items() if not gate["passed"]],
        "numerical_measured_support": numerical,
        "tail_extension": tail.as_dict(),
        "field_reduction": reduction_report,
        "cathode_origin": dict(cathode_origin_report),
        "axisymmetry_policy": {
            "production_representation": "cylindrical_vector_m0",
            "nonaxisymmetric_content_is_diagnostic_not_veto": True,
            "reason": "RF-Track production remains axisymmetric by explicit project decision",
        },
        "variant": variants[tail_policy],
        "fill_transient": fill_record,
    }
    return ReductionAssessment(field, tail, quality)
