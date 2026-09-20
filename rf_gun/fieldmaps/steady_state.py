"""Conditional single-pole extrapolation of an RF filling transient.

The short saved window measures a local complex envelope and slope. It does not
independently establish Q, source turn-on/ramp, or the steady-state amplitude.
Only under the explicit assumptions of one on-resonance mode, zero initial field,
and an ideal constant step drive does A=A_inf*(1-exp(-(t-t_start)/tau)) imply
u/(exp(u)-1)=g*(t-t_start), with g=d(log|A|)/dt and tau=2*Q_loaded/omega.

Evaluate this relation at the PHASOR REFERENCE TIME, not the solver stop time.
The old v1 workflow used an unsigned complex-slope norm at ~99.30 ns but inverted
it at 100 ns; its 2.1278x factor was an assumption, not an independent calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .models import AxisymmetricRFField

#: Model identifier stamped into metadata so a consumer can tell which correction ran.
FILL_MODEL = "single_pole_cavity_fill_v2"

#: Below this per-cycle drift the export is treated as converged and left alone.
NEGLIGIBLE_DRIFT_PER_CYCLE = 1.0e-5

#: Sanity bounds on the inferred loaded Q.  Outside these the single-pole picture is not
#: trustworthy and the fit is reported as failed rather than silently applied.
Q_LOADED_MIN = 50.0
Q_LOADED_MAX = 1.0e6

#: Refuse to invent more than this much amplitude; beyond it the export is too far from
#: steady state for a one-parameter extrapolation to be defensible.
MAX_CORRECTION_FACTOR = 10.0


def _solve_normalized_time(gt: float) -> float:
    """Invert ``u / (exp(u) - 1) = gt`` for ``u > 0`` by bisection.

    The left-hand side decreases monotonically from 1 (u -> 0) toward 0 (u -> inf), so a
    bracketed bisection is unconditionally convergent and needs no external solver.
    """

    if not (0.0 < gt < 1.0):
        raise ValueError(
            f"normalized growth g*t must lie strictly between 0 and 1 to imply a fill "
            f"transient (got {gt!r}); g*t >= 1 is not reachable by a single-pole fill"
        )

    def lhs(u: float) -> float:
        return u * math.exp(-u) / (-math.expm1(-u))

    lo, hi = 1.0e-12, 1.0
    while lhs(hi) > gt:
        hi *= 2.0
        if hi > 1.0e4:
            raise ValueError("fill-transient solve failed to bracket a root")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lhs(mid) > gt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class FillTransientFit:
    """Result of fitting the single-pole fill model to a measured envelope drift."""

    drift_per_cycle: float
    frequency_hz: float
    simulated_duration_s: float
    growth_rate_per_s: float
    normalized_time: float
    tau_s: float
    q_loaded: float
    fill_fraction: float
    correction_factor: float
    model: str
    converged_export: bool
    passed: bool
    checks: dict[str, Any]
    reference_time_s: float
    drive_start_time_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "drift_per_cycle": self.drift_per_cycle,
            "frequency_hz": self.frequency_hz,
            "simulated_duration_s": self.simulated_duration_s,
            "reference_time_s": self.reference_time_s,
            "drive_start_time_s": self.drive_start_time_s,
            "elapsed_drive_time_s": self.reference_time_s - self.drive_start_time_s,
            "steady_state_verified": False,
            "assumptions": [
                "single on-resonance mode", "zero initial cavity field",
                "ideal constant step drive at drive_start_time_s",
                "one common real envelope for E and H",
            ],
            "growth_rate_per_s": self.growth_rate_per_s,
            "normalized_time_t_over_tau": self.normalized_time if math.isfinite(self.normalized_time) else None,
            "tau_s": self.tau_s,
            "q_loaded": self.q_loaded,
            "fill_fraction": self.fill_fraction,
            "correction_factor": self.correction_factor,
            "converged_export": self.converged_export,
            "passed": self.passed,
            "checks": dict(self.checks),
        }


def fit_fill_transient(
    drift_per_cycle: float,
    frequency_hz: float,
    simulated_duration_s: float,
    *,
    reference_time_s: float | None = None,
    drive_start_time_s: float = 0.0,
) -> FillTransientFit:
    """Fit ``A(t) = A_inf (1 - exp(-t/tau))`` from the measured per-cycle envelope drift.

    ``drift_per_cycle`` must be SIGNED amplitude growth, Re(<P,P'>/<P,P>)/f,
    at ``reference_time_s``. The unsigned temporal quality metric is unsuitable.
    Omitting the reference retains the endpoint convention for direct callers.
    ``passed`` only checks model admissibility; it does not validate assumptions.
    """

    drift = float(drift_per_cycle)
    frequency = float(frequency_hz)
    duration = float(simulated_duration_s)
    reference = duration if reference_time_s is None else float(reference_time_s)
    start = float(drive_start_time_s)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("simulated_duration_s must be finite and positive")
    if not math.isfinite(drift):
        raise ValueError("drift_per_cycle must be finite")
    if not math.isfinite(reference) or reference <= 0.0 or reference > duration:
        raise ValueError("reference_time_s must be positive and no later than simulated_duration_s")
    if not math.isfinite(start) or not 0.0 <= start < reference:
        raise ValueError("drive_start_time_s must lie in [0, reference_time_s)")
    if drift < -NEGLIGIBLE_DRIFT_PER_CYCLE:
        raise ValueError("negative amplitude drift is decay, not a converged or step-filling cavity")

    omega = 2.0 * math.pi * frequency

    if abs(drift) <= NEGLIGIBLE_DRIFT_PER_CYCLE:
        # A locally stationary window is not proof of global steady state.
        return FillTransientFit(
            drift_per_cycle=drift,
            frequency_hz=frequency,
            simulated_duration_s=duration,
            reference_time_s=reference,
            drive_start_time_s=start,
            growth_rate_per_s=drift * frequency,
            normalized_time=math.inf,
            tau_s=0.0,
            q_loaded=0.0,
            fill_fraction=1.0,
            correction_factor=1.0,
            model=FILL_MODEL,
            converged_export=True,
            passed=True,
            checks={
                "drift_above_noise_floor": False,
                "locally_stationary_window": True,
                "noise_floor_per_cycle": NEGLIGIBLE_DRIFT_PER_CYCLE,
            },
        )

    growth_rate = drift * frequency
    elapsed = reference - start
    gt = growth_rate * elapsed
    u = _solve_normalized_time(gt)
    tau = elapsed / u
    q_loaded = omega * tau / 2.0
    fill_fraction = -math.expm1(-u)
    correction = 1.0 / fill_fraction

    checks = {
        "drift_above_noise_floor": True,
        "normalized_growth_in_range": 0.0 < gt < 1.0,
        "q_loaded_plausible": Q_LOADED_MIN <= q_loaded <= Q_LOADED_MAX,
        "correction_within_bound": correction <= MAX_CORRECTION_FACTOR,
        "q_loaded_bounds": [Q_LOADED_MIN, Q_LOADED_MAX],
        "max_correction_factor": MAX_CORRECTION_FACTOR,
    }
    passed = bool(
        checks["normalized_growth_in_range"]
        and checks["q_loaded_plausible"]
        and checks["correction_within_bound"]
    )

    return FillTransientFit(
        drift_per_cycle=drift,
        frequency_hz=frequency,
        simulated_duration_s=duration,
        reference_time_s=reference,
        drive_start_time_s=start,
        growth_rate_per_s=growth_rate,
        normalized_time=u,
        tau_s=tau,
        q_loaded=q_loaded,
        fill_fraction=fill_fraction,
        correction_factor=correction,
        model=FILL_MODEL,
        converged_export=False,
        passed=passed,
        checks=checks,
    )


def apply_fill_correction(
    field: AxisymmetricRFField,
    fit: FillTransientFit,
) -> AxisymmetricRFField:
    """Scale every E and B component to the extrapolated steady-state amplitude.

    Refuses failed, repeated, or temporally incompatible corrections. This
    changes the assumed amplitude reference, not the source-power calibration.
    """

    if not fit.passed:
        raise ValueError(
            "refusing to apply a fill correction from a failed fit: "
            f"{ {k: v for k, v in fit.checks.items() if v is False} }"
        )
    if field.metadata.get("fill_correction_applied"):
        raise ValueError("fill correction has already been applied to this field")
    if not math.isclose(field.frequency_hz, fit.frequency_hz, rel_tol=1e-12):
        raise ValueError("fill correction frequency does not match the field")
    if "reference_time_s" in field.metadata and not math.isclose(
        float(field.metadata["reference_time_s"]), fit.reference_time_s,
        rel_tol=0.0, abs_tol=1e-15,
    ):
        raise ValueError("fill correction reference_time_s does not match the field phasors")
    if fit.converged_export:
        return field

    factor = float(fit.correction_factor)
    metadata = dict(field.metadata)
    metadata.update(
        {
            "fill_correction_applied": True,
            "fill_correction": fit.as_dict(),
            "amplitude_reference": "model_estimated_steady_state",
            "steady_state_verified": False,
        }
    )
    return AxisymmetricRFField(
        frequency_hz=field.frequency_hz,
        r_m=field.r_m,
        z_m=field.z_m,
        Er_Vpm=field.Er_Vpm * factor,
        Ez_Vpm=field.Ez_Vpm * factor,
        Btheta_T=field.Btheta_T * factor,
        Bz_T=field.Bz_T * factor,
        metadata=metadata,
    )


def prepare_fill_treatment(
    field: AxisymmetricRFField,
    temporal: Mapping[str, Any],
    *,
    simulated_duration_s: float,
    policy: str = "auto",
    drive_start_time_s: float | None = None,
) -> tuple[AxisymmetricRFField, dict[str, Any]]:
    """Apply only an explicitly requested step model; keep controls usable.

    Auto leaves the samples unchanged and flags a drifting export as analysis
    only. Off is an explicit raw-amplitude control and never invokes inversion.
    Single-pole is a provisional model, accepted only with a declared start time
    and broadly consistent, mainly real E/H envelope growth. The 25% envelope
    consistency tolerance is a diagnostic guard, not an uncertainty estimate.
    """
    if policy not in {"auto", "off", "single-pole"}:
        raise ValueError("unknown fill-correction policy")
    if field.metadata.get("fill_correction_applied"):
        raise ValueError("fill correction has already been applied to this field")
    total = float(temporal["production_metrics"]["max_family_combined_fractional_drift_per_cycle"])
    if not math.isfinite(total) or total < 0.0:
        raise ValueError("temporal drift norm must be finite and non-negative")
    reference = float(field.metadata["reference_time_s"])
    stationary = total <= NEGLIGIBLE_DRIFT_PER_CYCLE
    record: dict[str, Any] = {
        "policy": policy,
        "applied": False,
        "applied_correction_factor": 1.0,
        "reference_time_s": reference,
        "simulated_duration_s": float(simulated_duration_s),
        "complex_drift_norm_per_cycle": total,
        "locally_stationary_window": stationary,
        "steady_state_verified": False,
        "production_eligible": policy != "auto" or stationary,
        "status": "stationary_window" if stationary else "uncorrected_transient",
    }
    metadata = dict(field.metadata)
    metadata.update({
        "fill_correction_applied": False,
        "amplitude_reference": record["status"],
        "steady_state_verified": False,
    })
    if policy == "single-pole":
        if drive_start_time_s is None:
            raise ValueError("single-pole requires an explicit assumed drive_start_time_s")
        families = temporal["family_aggregate"]
        gamma = [complex(
            families[name]["signed_amplitude_drift_per_cycle"],
            families[name]["phase_drift_rad_per_cycle"],
        ) for name in ("E", "H")]
        common = sum(gamma) / 2.0  # Equal family weight; never mix E and H units.
        residuals = [float(families[name]["nonseparable_drift_per_cycle"]) for name in ("E", "H")]
        if not all(math.isfinite(x) for g in gamma for x in (g.real, g.imag)) or not all(
            math.isfinite(x) and x >= 0.0 for x in residuals
        ):
            raise ValueError("non-finite or invalid common-envelope diagnostics")
        scale = max(abs(common.real), NEGLIGIBLE_DRIFT_PER_CYCLE)
        consistency = {
            "electric_magnetic_growth_agreement": abs(gamma[0].real - gamma[1].real) <= 0.25 * scale,
            "small_phase_rotation": max(abs(g.imag) for g in gamma) <= 0.25 * scale,
            "common_spatial_envelope": max(residuals) <= 0.25 * scale,
        }
        if not all(consistency.values()):
            raise ValueError(f"single-pole common-envelope assumption fails: {consistency}")
        fit = fit_fill_transient(
            common.real, field.frequency_hz, simulated_duration_s,
            reference_time_s=reference, drive_start_time_s=drive_start_time_s,
        )
        working = apply_fill_correction(replace(field, metadata=metadata), fit)
        record.update(fit.as_dict())
        record.update({
            "applied": not fit.converged_export,
            "applied_correction_factor": fit.correction_factor,
            "status": "stationary_window" if fit.converged_export else "model_estimated_steady_state",
            "envelope_consistency": consistency,
            "electric_amplitude_drift_per_cycle": gamma[0].real,
            "magnetic_amplitude_drift_per_cycle": gamma[1].real,
        })
        metadata = dict(working.metadata)
    else:
        working = replace(field, metadata=metadata)
    metadata["fill_treatment"] = record
    return replace(working, metadata=metadata), record
