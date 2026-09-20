"""Optional diagnostic comparisons; not imported by the production artifact path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .models import AxisymmetricRFField


def compare_tail_control(
    selected: AxisymmetricRFField,
    control: AxisymmetricRFField,
    *,
    measured_z_max_m: float,
) -> dict[str, float]:
    """Compare tail voltage only when both maps share their measured fields.

    A grid, frequency, phase or amplitude mismatch is an error: it would make
    the apparent tail difference include an unrelated change to the input map.
    The tolerance allows roundoff from complex64 export and inverse scaling.
    """
    if not np.isclose(selected.frequency_hz, control.frequency_hz, rtol=1e-12, atol=0):
        raise ValueError("tail control frequency differs from the selected map")
    if not np.array_equal(selected.r_m, control.r_m) or not np.array_equal(selected.z_m, control.z_m):
        raise ValueError("tail control must use the same r/z grid as the selected map")
    if not np.isfinite(measured_z_max_m) or not selected.z_m[0] <= measured_z_max_m < selected.z_m[-1]:
        raise ValueError("measured_z_max_m must leave a nonempty measured region and downstream tail")
    measured = selected.z_m <= measured_z_max_m + 1e-15
    for name in ("Er_Vpm", "Ez_Vpm", "Btheta_T", "Bz_T"):
        reference = getattr(selected, name)[measured]
        candidate = getattr(control, name)[measured]
        tolerance = 5 * np.finfo(np.float32).eps * float(np.max(np.abs(reference)))
        if not np.allclose(reference, candidate, rtol=0, atol=tolerance):
            raise ValueError(f"tail control has different measured {name}; rebuild a matching control")
    selected_voltage = np.trapezoid(selected.Ez_Vpm[:, 0], selected.z_m)
    control_voltage = np.trapezoid(control.Ez_Vpm[:, 0], control.z_m)
    difference = abs(selected_voltage - control_voltage)
    return {
        "selected_axis_voltage_magnitude_kV": float(abs(selected_voltage) / 1e3),
        "zero_control_axis_voltage_magnitude_kV": float(abs(control_voltage) / 1e3),
        "complex_voltage_difference_V": float(difference),
        "difference_fraction_of_selected_percent": float(1e2 * difference / max(abs(selected_voltage), 1e-30)),
    }


@dataclass(frozen=True)
class ComplexFieldComparison:
    common_gain: complex
    raw_nrmse: float
    aligned_nrmse: float
    complex_correlation_magnitude: float
    amplitude_correlation: float
    median_candidate_to_reference_amplitude: float
    active_sample_count: int


def _selection(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray | None,
    active_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.complex128)
    cand = np.asarray(candidate, dtype=np.complex128)
    if ref.shape != cand.shape:
        raise ValueError(f"reference shape {ref.shape} differs from candidate shape {cand.shape}")
    if not 0.0 <= active_fraction < 1.0:
        raise ValueError("active_fraction must lie in [0, 1)")
    active = np.isfinite(ref) & np.isfinite(cand)
    peak = max(float(np.max(np.abs(ref[active]))) if np.any(active) else 0.0, 0.0)
    if peak > 0.0:
        active &= np.abs(ref) >= active_fraction * peak
    if mask is not None:
        active &= np.broadcast_to(np.asarray(mask, dtype=bool), ref.shape)
    if not np.any(active):
        raise ValueError("comparison has no finite active samples")
    return ref[active], cand[active], active


def optimal_common_complex_gain(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> complex:
    """Least-squares scalar ``g`` minimizing ``||reference - g*candidate||``."""

    reference_array = np.asarray(reference, dtype=np.complex128)
    candidate_array = np.asarray(candidate, dtype=np.complex128)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("reference and candidate must have matching shapes")
    weight_array = (
        np.ones(reference_array.shape, dtype=np.float64)
        if weights is None
        else np.broadcast_to(np.asarray(weights, dtype=np.float64), reference_array.shape)
    )
    ref = reference_array.reshape(-1)
    cand = candidate_array.reshape(-1)
    weight = weight_array.reshape(-1)
    denominator = np.sum(weight * np.abs(cand) ** 2)
    if denominator <= 0.0:
        raise ValueError("candidate has zero weighted norm")
    return complex(np.sum(weight * np.conj(cand) * ref) / denominator)


def compare_complex_fields(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    active_fraction: float = 0.01,
) -> ComplexFieldComparison:
    """Compare spatial complex fields with one diagnostic global alignment."""

    ref, cand, active = _selection(reference, candidate, mask, active_fraction)
    if weights is None:
        selected_weights = np.ones(ref.shape, dtype=np.float64)
    else:
        selected_weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), active.shape)[active]
    gain = optimal_common_complex_gain(ref, cand, weights=selected_weights)
    reference_norm = float(np.sqrt(np.sum(selected_weights * np.abs(ref) ** 2)))
    if reference_norm == 0.0:
        raise ValueError("reference has zero weighted norm")
    raw_nrmse = float(np.sqrt(np.sum(selected_weights * np.abs(ref - cand) ** 2)) / reference_norm)
    aligned_nrmse = float(
        np.sqrt(np.sum(selected_weights * np.abs(ref - gain * cand) ** 2)) / reference_norm
    )
    inner = np.sum(selected_weights * np.conj(ref) * cand)
    candidate_norm = float(np.sqrt(np.sum(selected_weights * np.abs(cand) ** 2)))
    complex_correlation = float(abs(inner) / (reference_norm * candidate_norm)) if candidate_norm else 0.0
    ref_amp = np.abs(ref)
    cand_amp = np.abs(cand)
    if ref_amp.size > 1 and np.std(ref_amp) > 0.0 and np.std(cand_amp) > 0.0:
        amplitude_correlation = float(np.corrcoef(ref_amp, cand_amp)[0, 1])
    else:
        amplitude_correlation = 1.0 if np.allclose(ref_amp, cand_amp) else 0.0
    nonzero_ref = ref_amp > 0.0
    median_ratio = float(np.median(cand_amp[nonzero_ref] / ref_amp[nonzero_ref]))
    return ComplexFieldComparison(
        common_gain=gain,
        raw_nrmse=raw_nrmse,
        aligned_nrmse=aligned_nrmse,
        complex_correlation_magnitude=complex_correlation,
        amplitude_correlation=amplitude_correlation,
        median_candidate_to_reference_amplitude=median_ratio,
        active_sample_count=int(ref.size),
    )


def compare_component_sets(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
    active_fraction: float = 0.01,
) -> dict[str, ComplexFieldComparison]:
    """Apply the same transparent metric definition component by component."""

    if set(reference) != set(candidate):
        raise ValueError("reference and candidate component names differ")
    return {
        name: compare_complex_fields(
            reference[name], candidate[name], mask=mask, active_fraction=active_fraction
        )
        for name in reference
    }
