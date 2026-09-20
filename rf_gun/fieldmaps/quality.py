"""Numerical qualification diagnostics for axisymmetric RF field maps.

The routines in this module are deliberately independent of RF-Track and of
Matplotlib.  Inputs and outputs use SI units.  Phasors follow the project
convention

``F(t) = Re(F_hat * exp(+i * omega * (t - t_ref)))``.

Consequently, in vacuum, ``curl(E) = -i*omega*B`` and
``curl(B) = +i*omega/c**2*E``.  The public diagnostic functions return plain
Python dictionaries/lists/scalars so their results can be written directly to
the JSON quality record in a compact field artifact.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.ndimage import binary_erosion

from .coordinates import MU0_HPM
from .models import AxisymmetricRFField

SPEED_OF_LIGHT_MPS = 299_792_458.0
EPSILON0_FPM = 1.0 / (MU0_HPM * SPEED_OF_LIGHT_MPS**2)
QUALITY_SCHEMA_VERSION = "axisymmetric_vacuum_quality_v1"


def _vacuum_mask(field: AxisymmetricRFField, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.ones(field.shape, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape != field.shape:
        raise ValueError(f"vacuum_mask has shape {result.shape}; expected {field.shape}")
    if not np.any(result):
        raise ValueError("vacuum_mask selects no cells")
    return result


def guarded_vacuum_mask(
    field: AxisymmetricRFField,
    vacuum_mask: np.ndarray | None = None,
    *,
    wall_guard_cells: int = 2,
    axis_guard_cells: int = 1,
) -> np.ndarray:
    """Return the vacuum cells safe for finite-difference Maxwell metrics.

    ``wall_guard_cells`` erodes material and array boundaries using a
    four-connected stencil.  ``axis_guard_cells`` is the number of radial
    samples removed starting with ``r=0``; a value of one excludes only the
    coordinate singularity.  Axis regularity is evaluated separately on the
    un-eroded mask.
    """

    if int(wall_guard_cells) != wall_guard_cells or wall_guard_cells < 0:
        raise ValueError("wall_guard_cells must be a non-negative integer")
    if int(axis_guard_cells) != axis_guard_cells or axis_guard_cells < 0:
        raise ValueError("axis_guard_cells must be a non-negative integer")
    mask = _vacuum_mask(field, vacuum_mask).copy()
    if wall_guard_cells:
        structure = np.array([[False, True, False], [True, True, True], [False, True, False]])
        mask = binary_erosion(
            mask,
            structure=structure,
            iterations=int(wall_guard_cells),
            border_value=0,
        )
    mask[:, : int(axis_guard_cells)] = False
    if not np.any(mask):
        raise ValueError(
            "wall/axis guards remove every vacuum cell; reduce the guards or use a larger grid"
        )
    return mask


def _cylindrical_weights(field: AxisymmetricRFField, mask: np.ndarray) -> np.ndarray:
    # Constant factors 2*pi*dr*dz cancel in every RMS.  The r factor is the
    # physically important cylindrical-volume weighting.
    weights = np.broadcast_to(field.r_m[None, :], field.shape).astype(np.float64, copy=True)
    weights *= mask
    if float(np.sum(weights)) == 0.0:
        # A mask containing only the mathematical axis has zero cylindrical
        # measure.  Uniform weights still give a useful, explicitly reported
        # diagnostic rather than a division by zero.
        weights = mask.astype(np.float64)
    return weights


def _weighted_rms(values: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    selected_weights = weights[mask]
    denominator = float(np.sum(selected_weights))
    if denominator <= 0.0:
        return 0.0
    return float(np.sqrt(np.sum(selected_weights * np.abs(values[mask]) ** 2) / denominator))


def _array_summary(
    values: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    *,
    units: str,
) -> dict[str, float | str]:
    magnitudes = np.abs(np.asarray(values)[mask])
    if magnitudes.size == 0:
        raise ValueError("diagnostic mask selects no samples")
    return {
        "units": units,
        "spatial_phasor_rms": _weighted_rms(values, mask, weights),
        "median_abs": float(np.median(magnitudes)),
        "p95_abs": float(np.percentile(magnitudes, 95.0)),
        "max_abs": float(np.max(magnitudes)),
    }


def field_norm_diagnostics(
    field: AxisymmetricRFField,
    vacuum_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return component, combined-field, and mean energy-density norms.

    Spatial RMS values use cylindrical-volume weights proportional to ``r``.
    They are phasor-amplitude norms; the RMS of the corresponding sinusoid in
    time is smaller by ``sqrt(2)``.
    """

    mask = _vacuum_mask(field, vacuum_mask)
    weights = _cylindrical_weights(field, mask)
    components = {
        "Er": _array_summary(field.Er_Vpm, mask, weights, units="V/m"),
        "Ez": _array_summary(field.Ez_Vpm, mask, weights, units="V/m"),
        "Btheta": _array_summary(field.Btheta_T, mask, weights, units="T"),
        "Bz": _array_summary(field.Bz_T, mask, weights, units="T"),
    }
    e2 = np.abs(field.Er_Vpm) ** 2 + np.abs(field.Ez_Vpm) ** 2
    b2 = np.abs(field.Btheta_T) ** 2 + np.abs(field.Bz_T) ** 2
    e_magnitude = np.sqrt(e2)
    b_magnitude = np.sqrt(b2)
    e_rms = _weighted_rms(e_magnitude, mask, weights)
    b_rms = _weighted_rms(b_magnitude, mask, weights)
    return {
        "weighting": "cylindrical_volume_proportional_to_r_drdz",
        "phasor_convention": "peak_complex_amplitude",
        "vacuum_cell_count": int(np.count_nonzero(mask)),
        "components": components,
        "combined": {
            "E_spatial_phasor_rms_Vpm": e_rms,
            "E_max_Vpm": float(np.max(e_magnitude[mask])),
            "B_spatial_phasor_rms_T": b_rms,
            "B_max_T": float(np.max(b_magnitude[mask])),
            "cB_spatial_phasor_rms_Vpm": SPEED_OF_LIGHT_MPS * b_rms,
            "cB_to_E_rms_ratio": SPEED_OF_LIGHT_MPS * b_rms / e_rms if e_rms > 0.0 else 0.0,
        },
        "time_average_energy_density": {
            "electric_Jpm3": 0.25 * EPSILON0_FPM * e_rms**2,
            "magnetic_Jpm3": 0.25 / MU0_HPM * b_rms**2,
        },
    }


def _derivative(values: np.ndarray, coordinate: np.ndarray, axis: int) -> np.ndarray:
    edge_order = 2 if coordinate.size >= 3 else 1
    return np.gradient(values, coordinate, axis=axis, edge_order=edge_order)


def _radial_cylindrical_derivative(
    radial_component: np.ndarray,
    r_m: np.ndarray,
) -> np.ndarray:
    """Evaluate ``(1/r) d(r F_r)/dr``, including its regular axis limit."""

    derivative = _derivative(radial_component, r_m, axis=1)
    result = np.empty_like(radial_component, dtype=np.result_type(radial_component, np.complex128))
    result[:, 1:] = derivative[:, 1:] + radial_component[:, 1:] / r_m[None, 1:]
    result[:, 0] = 2.0 * derivative[:, 0]
    return result


def _equation_summary(
    residual: np.ndarray,
    terms: Mapping[str, np.ndarray],
    mask: np.ndarray,
    weights: np.ndarray,
    *,
    units: str,
    family_reference_rms: float,
) -> dict[str, Any]:
    residual_stats = _array_summary(residual, mask, weights, units=units)
    residual_rms = float(residual_stats["spatial_phasor_rms"])
    term_rms = {name: _weighted_rms(value, mask, weights) for name, value in terms.items()}
    balance_reference = float(np.sqrt(sum(value * value for value in term_rms.values())))
    balance_normalized = residual_rms / balance_reference if balance_reference > 0.0 else 0.0
    family_normalized = residual_rms / family_reference_rms if family_reference_rms > 0.0 else 0.0
    return {
        **residual_stats,
        "term_rms": term_rms,
        "balance_reference_rms": balance_reference,
        "normalized_rms": balance_normalized,
        "family_normalized_rms": family_normalized,
    }


def maxwell_residual_diagnostics(
    field: AxisymmetricRFField,
    vacuum_mask: np.ndarray | None = None,
    *,
    wall_guard_cells: int = 2,
    axis_guard_cells: int = 1,
    active_field_fraction: float = 1.0e-3,
) -> dict[str, Any]:
    """Evaluate vacuum Maxwell residuals for the stored axisymmetric components.

    The primary equations are those of an axisymmetric TM-like representation
    with ``Etheta=Br=0``.  Non-zero ``Bz`` cannot be fully qualified without the
    omitted components, so its three residuals are reported separately under
    ``representation_consistency`` rather than hidden.
    """

    if not 0.0 <= active_field_fraction < 1.0:
        raise ValueError("active_field_fraction must lie in [0, 1)")
    metric_mask = guarded_vacuum_mask(
        field,
        vacuum_mask,
        wall_guard_cells=wall_guard_cells,
        axis_guard_cells=axis_guard_cells,
    )
    electromagnetic_amplitude = np.sqrt(
        np.abs(field.Er_Vpm) ** 2
        + np.abs(field.Ez_Vpm) ** 2
        + (SPEED_OF_LIGHT_MPS * np.abs(field.Btheta_T)) ** 2
        + (SPEED_OF_LIGHT_MPS * np.abs(field.Bz_T)) ** 2
    )
    peak = float(np.max(electromagnetic_amplitude[metric_mask]))
    active_mask = metric_mask.copy()
    if peak > 0.0 and active_field_fraction > 0.0:
        active_mask &= electromagnetic_amplitude >= active_field_fraction * peak
    if not np.any(active_mask):
        active_mask = metric_mask.copy()

    omega = 2.0 * np.pi * field.frequency_hz
    k0 = omega / SPEED_OF_LIGHT_MPS
    d_er_dz = _derivative(field.Er_Vpm, field.z_m, axis=0)
    d_ez_dr = _derivative(field.Ez_Vpm, field.r_m, axis=1)
    d_ez_dz = _derivative(field.Ez_Vpm, field.z_m, axis=0)
    div_er = _radial_cylindrical_derivative(field.Er_Vpm, field.r_m)
    d_bt_dz = _derivative(field.Btheta_T, field.z_m, axis=0)
    curl_bt_z = _radial_cylindrical_derivative(field.Btheta_T, field.r_m)
    d_bz_dr = _derivative(field.Bz_T, field.r_m, axis=1)
    d_bz_dz = _derivative(field.Bz_T, field.z_m, axis=0)

    residuals = {
        "faraday_theta": (
            d_er_dz - d_ez_dr + 1j * omega * field.Btheta_T,
            {"dEr_dz": d_er_dz, "minus_dEz_dr": -d_ez_dr, "iomega_Btheta": 1j * omega * field.Btheta_T},
            "V/m^2",
            "dEr/dz - dEz/dr + i*omega*Btheta = 0",
        ),
        "ampere_r": (
            -d_bt_dz - 1j * omega / SPEED_OF_LIGHT_MPS**2 * field.Er_Vpm,
            {"minus_dBtheta_dz": -d_bt_dz, "minus_iomega_Er_over_c2": -1j * omega / SPEED_OF_LIGHT_MPS**2 * field.Er_Vpm},
            "T/m",
            "-dBtheta/dz - i*omega*Er/c^2 = 0",
        ),
        "ampere_z": (
            curl_bt_z - 1j * omega / SPEED_OF_LIGHT_MPS**2 * field.Ez_Vpm,
            {"radial_curl_Btheta": curl_bt_z, "minus_iomega_Ez_over_c2": -1j * omega / SPEED_OF_LIGHT_MPS**2 * field.Ez_Vpm},
            "T/m",
            "(1/r)d(r*Btheta)/dr - i*omega*Ez/c^2 = 0",
        ),
        "gauss_e": (
            div_er + d_ez_dz,
            {"radial_divergence_Er": div_er, "dEz_dz": d_ez_dz},
            "V/m^2",
            "(1/r)d(r*Er)/dr + dEz/dz = 0",
        ),
    }
    consistency = {
        "faraday_z_from_Bz": (
            1j * omega * field.Bz_T,
            {"iomega_Bz": 1j * omega * field.Bz_T},
            "V/m^2",
            "i*omega*Bz = 0 when Etheta = 0",
        ),
        "ampere_theta_from_Bz": (
            -d_bz_dr,
            {"minus_dBz_dr": -d_bz_dr},
            "T/m",
            "-dBz/dr = 0 when Br = Etheta = 0",
        ),
        "gauss_b_from_Bz": (
            d_bz_dz,
            {"dBz_dz": d_bz_dz},
            "T/m",
            "dBz/dz = 0 when Br = 0",
        ),
    }

    def summarize_for(mask: np.ndarray) -> dict[str, Any]:
        weights = _cylindrical_weights(field, mask)
        electric_magnitude = np.sqrt(np.abs(field.Er_Vpm) ** 2 + np.abs(field.Ez_Vpm) ** 2)
        e_rms = _weighted_rms(electric_magnitude, mask, weights)
        b_magnitude = np.sqrt(np.abs(field.Btheta_T) ** 2 + np.abs(field.Bz_T) ** 2)
        b_rms = _weighted_rms(b_magnitude, mask, weights)
        faraday_reference = float(np.hypot(k0 * e_rms, omega * b_rms))
        ampere_reference = float(
            np.hypot(k0 * b_rms, omega / SPEED_OF_LIGHT_MPS**2 * e_rms)
        )
        primary: dict[str, Any] = {}
        for name, (residual, terms, units, equation) in residuals.items():
            family = faraday_reference if name in {"faraday_theta", "gauss_e"} else ampere_reference
            entry = _equation_summary(
                residual,
                terms,
                mask,
                weights,
                units=units,
                family_reference_rms=family,
            )
            entry["equation"] = equation
            primary[name] = entry
        representation: dict[str, Any] = {}
        for name, (residual, terms, units, equation) in consistency.items():
            family = faraday_reference if name == "faraday_z_from_Bz" else ampere_reference
            entry = _equation_summary(
                residual,
                terms,
                mask,
                weights,
                units=units,
                family_reference_rms=family,
            )
            entry["equation"] = equation
            representation[name] = entry
        faraday_rms = primary["faraday_theta"]["spatial_phasor_rms"]
        ampere_rms = float(
            np.hypot(
                primary["ampere_r"]["spatial_phasor_rms"],
                primary["ampere_z"]["spatial_phasor_rms"],
            )
        )
        return {
            "sample_count": int(np.count_nonzero(mask)),
            "reference_scales": {
                "vacuum_wavenumber_per_m": k0,
                "faraday_or_divE_Vpm2": faraday_reference,
                "ampere_or_divB_Tpm": ampere_reference,
            },
            "primary": primary,
            "representation_consistency": representation,
            "combined": {
                "faraday_family_normalized_rms": faraday_rms / faraday_reference if faraday_reference > 0.0 else 0.0,
                "ampere_family_normalized_rms": ampere_rms / ampere_reference if ampere_reference > 0.0 else 0.0,
            },
        }

    return {
        "phasor_convention": "Re(F*exp(+i*omega*(t-t_ref)))",
        "vacuum_equations": {
            "faraday": "curl(E) = -i*omega*B",
            "ampere": "curl(B) = +i*omega*E/c^2",
            "gauss_e": "div(E) = 0",
            "gauss_b": "div(B) = 0",
        },
        "representation_assumptions": ["axisymmetric", "Etheta=0", "Br=0"],
        "wall_guard_cells": int(wall_guard_cells),
        "axis_guard_cells": int(axis_guard_cells),
        "active_field_fraction": float(active_field_fraction),
        "guarded_vacuum_cell_count": int(np.count_nonzero(metric_mask)),
        "active_guarded_cell_count": int(np.count_nonzero(active_mask)),
        "guarded_vacuum": summarize_for(metric_mask),
        "active_guarded_vacuum": summarize_for(active_mask),
    }


def _axis_polynomial_slope(
    values: np.ndarray,
    r_m: np.ndarray,
    *,
    point_count: int,
) -> np.ndarray:
    scale = float(r_m[1] - r_m[0])
    degree = min(3, point_count - 1)
    design = np.vander(r_m[:point_count] / scale, N=degree + 1, increasing=True)
    coefficients, _, rank, _ = np.linalg.lstsq(design, values[:, :point_count].T, rcond=None)
    if rank != degree + 1:
        raise ValueError("near-axis coordinates do not define the regularity polynomial")
    return coefficients[1] / scale


def axis_regularity_diagnostics(
    field: AxisymmetricRFField,
    vacuum_mask: np.ndarray | None = None,
    *,
    axis_fit_points: int = 4,
    longitudinal_guard_cells: int = 1,
) -> dict[str, Any]:
    """Check odd-component zeros and even-component radial slopes at ``r=0``."""

    if axis_fit_points < 3 or axis_fit_points > field.r_m.size:
        raise ValueError("axis_fit_points must be between 3 and the radial grid size")
    if longitudinal_guard_cells < 0:
        raise ValueError("longitudinal_guard_cells must be non-negative")
    mask = _vacuum_mask(field, vacuum_mask)
    valid_rows = np.all(mask[:, :axis_fit_points], axis=1)
    if longitudinal_guard_cells:
        valid_rows[:longitudinal_guard_cells] = False
        valid_rows[-longitudinal_guard_cells:] = False
    if not np.any(valid_rows):
        raise ValueError("no longitudinal rows contain the requested near-axis vacuum stencil")
    norms = field_norm_diagnostics(field, mask)["combined"]
    e_scale = float(norms["E_spatial_phasor_rms_Vpm"])
    b_scale = float(norms["B_spatial_phasor_rms_T"])
    k0 = 2.0 * np.pi * field.frequency_hz / SPEED_OF_LIGHT_MPS

    def odd(values: np.ndarray, scale: float, units: str) -> dict[str, Any]:
        axis_values = np.asarray(values)[valid_rows, 0]
        rms = float(np.sqrt(np.mean(np.abs(axis_values) ** 2)))
        return {
            "expected": "zero_at_axis",
            "units": units,
            "axis_rms": rms,
            "axis_max_abs": float(np.max(np.abs(axis_values))),
            "fraction_of_combined_field_rms": rms / scale if scale > 0.0 else 0.0,
        }

    def even(values: np.ndarray, scale: float, units: str) -> dict[str, Any]:
        slopes = _axis_polynomial_slope(values, field.r_m, point_count=axis_fit_points)[valid_rows]
        rms = float(np.sqrt(np.mean(np.abs(slopes) ** 2)))
        reference = k0 * scale
        return {
            "expected": "zero_radial_derivative_at_axis",
            "units": units,
            "radial_slope_rms": rms,
            "radial_slope_max_abs": float(np.max(np.abs(slopes))),
            "normalized_to_k0_times_combined_field_rms": rms / reference if reference > 0.0 else 0.0,
        }

    return {
        "axis_r_m": float(field.r_m[0]),
        "axis_fit_points": int(axis_fit_points),
        "longitudinal_guard_cells": int(longitudinal_guard_cells),
        "evaluated_row_count": int(np.count_nonzero(valid_rows)),
        "odd_components": {
            "Er": odd(field.Er_Vpm, e_scale, "V/m"),
            "Btheta": odd(field.Btheta_T, b_scale, "T"),
        },
        "even_components": {
            "Ez": even(field.Ez_Vpm, e_scale, "V/m^2"),
            "Bz": even(field.Bz_T, b_scale, "T/m"),
        },
    }


def _radial_rms_profile(
    values: np.ndarray,
    field: AxisymmetricRFField,
    mask: np.ndarray,
) -> np.ndarray:
    radial_weights = np.broadcast_to(field.r_m[None, :], field.shape) * mask
    numerator = np.sum(radial_weights * np.abs(values) ** 2, axis=1)
    denominator = np.sum(radial_weights, axis=1)
    profile = np.zeros(field.z_m.size, dtype=np.float64)
    valid = denominator > 0.0
    profile[valid] = np.sqrt(numerator[valid] / denominator[valid])
    # A row containing only r=0 has zero cylindrical weight; retain its value.
    axis_only = (~valid) & mask[:, 0]
    profile[axis_only] = np.abs(values[axis_only, 0])
    return profile


def _remaining_trapezoid(z_m: np.ndarray, values: np.ndarray) -> np.ndarray:
    segments = 0.5 * (values[:-1] + values[1:]) * np.diff(z_m)
    result = np.zeros(values.size, dtype=np.float64)
    result[:-1] = np.cumsum(segments[::-1])[::-1]
    return result


def _decay_fit(z_m: np.ndarray, amplitude: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    floor = 1.0e-12 * float(np.max(amplitude)) if amplitude.size else 0.0
    usable = selected & np.isfinite(amplitude) & (amplitude > floor)
    if np.count_nonzero(usable) < 3:
        return {"available": False, "reason": "fewer_than_three_nonzero_tail_samples"}
    z = z_m[usable]
    log_amplitude = np.log(amplitude[usable])
    slope, intercept = np.polyfit(z, log_amplitude, 1)
    predicted = slope * z + intercept
    ss_residual = float(np.sum((log_amplitude - predicted) ** 2))
    ss_total = float(np.sum((log_amplitude - np.mean(log_amplitude)) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0.0 else 1.0
    return {
        "available": True,
        "log_amplitude_slope_per_m": float(slope),
        "decay_constant_per_m": float(-slope),
        "decay_length_m": float(-1.0 / slope) if slope < 0.0 else None,
        "log_amplitude_r_squared": r_squared,
        "sample_count": int(z.size),
        "is_decaying": bool(slope < 0.0),
    }


def tail_profile_diagnostics(
    field: AxisymmetricRFField,
    vacuum_mask: np.ndarray | None = None,
    *,
    tail_start_z_m: float | None = None,
    near_zero_fraction: float = 1.0e-2,
    max_profile_points: int = 512,
) -> dict[str, Any]:
    """Describe field decay toward the measured downstream boundary.

    The endpoint tests are relative to each profile's maximum on the measured
    grid.  They deliberately report values and booleans rather than deciding
    whether extrapolation is physically acceptable; that decision also depends
    on omitted integrated voltage and beam-observable convergence.
    """

    if not 0.0 < near_zero_fraction < 1.0:
        raise ValueError("near_zero_fraction must lie strictly between zero and one")
    if max_profile_points < 2:
        raise ValueError("max_profile_points must be at least two")
    mask = _vacuum_mask(field, vacuum_mask)
    z = field.z_m
    if tail_start_z_m is None:
        tail_start = float(z[0] + 0.8 * (z[-1] - z[0]))
    else:
        tail_start = float(tail_start_z_m)
    if not z[0] <= tail_start < z[-1]:
        raise ValueError("tail_start_z_m must lie within the grid and before its endpoint")
    tail_selection = z >= tail_start

    e_combined = np.sqrt(np.abs(field.Er_Vpm) ** 2 + np.abs(field.Ez_Vpm) ** 2)
    b_combined = np.sqrt(np.abs(field.Btheta_T) ** 2 + np.abs(field.Bz_T) ** 2)
    values = {
        "Er_Vpm": field.Er_Vpm,
        "Ez_Vpm": field.Ez_Vpm,
        "Btheta_T": field.Btheta_T,
        "Bz_T": field.Bz_T,
        "E_combined_Vpm": e_combined,
        "B_combined_T": b_combined,
    }
    profiles = {name: _radial_rms_profile(value, field, mask) for name, value in values.items()}
    endpoint: dict[str, Any] = {}
    for name, profile in profiles.items():
        maximum = float(np.max(profile))
        end_value = float(profile[-1])
        ratio = end_value / maximum if maximum > 0.0 else 0.0
        tail_max_ratio = float(np.max(profile[tail_selection]) / maximum) if maximum > 0.0 else 0.0
        endpoint[name] = {
            "radial_rms_at_end": end_value,
            "global_radial_rms_max": maximum,
            "end_to_global_max_fraction": ratio,
            "tail_max_to_global_max_fraction": tail_max_ratio,
            "end_below_near_zero_threshold": bool(ratio <= near_zero_fraction),
        }

    ez_axis = np.abs(field.Ez_Vpm[:, 0])
    remaining_voltage = _remaining_trapezoid(z, ez_axis)
    tail_index = int(np.searchsorted(z, tail_start, side="left"))
    total_voltage = float(remaining_voltage[0])
    tail_voltage = float(remaining_voltage[tail_index])
    profile_indices = np.unique(
        np.linspace(0, z.size - 1, min(z.size, max_profile_points), dtype=np.int64)
    )
    return {
        "measured_z_min_m": float(z[0]),
        "measured_z_max_m": float(z[-1]),
        "tail_start_z_m": tail_start,
        "near_zero_fraction": float(near_zero_fraction),
        "radial_profile_definition": "cylindrical_r_weighted_phasor_rms",
        "endpoint": endpoint,
        "on_axis_Ez_abs_integral": {
            "total_measured_V": total_voltage,
            "tail_measured_V": tail_voltage,
            "tail_fraction_of_measured": tail_voltage / total_voltage if total_voltage > 0.0 else 0.0,
            "remaining_at_each_profile_z_V": remaining_voltage[profile_indices].tolist(),
        },
        "decay_fit_E_combined": _decay_fit(z, profiles["E_combined_Vpm"], tail_selection),
        "decay_fit_B_combined": _decay_fit(z, profiles["B_combined_T"], tail_selection),
        "profiles": {
            "z_m": z[profile_indices].tolist(),
            **{name: profile[profile_indices].tolist() for name, profile in profiles.items()},
            "Ez_axis_abs_Vpm": ez_axis[profile_indices].tolist(),
        },
    }


def qualify_axisymmetric_field(
    field: AxisymmetricRFField,
    vacuum_mask: np.ndarray | None = None,
    *,
    wall_guard_cells: int = 2,
    axis_guard_cells: int = 1,
    active_field_fraction: float = 1.0e-3,
    axis_fit_points: int = 4,
    tail_start_z_m: float | None = None,
    near_zero_fraction: float = 1.0e-2,
    beam_core_radius_m: float | None = None,
) -> dict[str, Any]:
    """Build the complete JSON-serializable numerical qualification record."""

    field.validate()
    mask = _vacuum_mask(field, vacuum_mask)
    maxwell_vacuum = maxwell_residual_diagnostics(
        field,
        mask,
        wall_guard_cells=wall_guard_cells,
        axis_guard_cells=axis_guard_cells,
        active_field_fraction=active_field_fraction,
    )
    result = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "frequency_hz": float(field.frequency_hz),
        "coordinate_system": "right_handed_cylindrical_beam_frame_r_z",
        "phasor_convention": "Re(F*exp(+i*omega*(t-t_ref)))",
        "norms": field_norm_diagnostics(field, mask),
        "axis_regularity": axis_regularity_diagnostics(
            field,
            mask,
            axis_fit_points=axis_fit_points,
            longitudinal_guard_cells=wall_guard_cells,
        ),
        "maxwell_vacuum": maxwell_vacuum,
        "tail": tail_profile_diagnostics(
            field,
            mask,
            tail_start_z_m=tail_start_z_m,
            near_zero_fraction=near_zero_fraction,
        ),
    }
    if beam_core_radius_m is not None:
        radius = float(beam_core_radius_m)
        if not np.isfinite(radius) or radius <= 0.0 or radius > float(field.r_m[-1]):
            raise ValueError("beam_core_radius_m must be finite, positive, and inside r_m support")
        core_mask = mask & (field.r_m[None, :] <= radius)
        result["maxwell_beam_core"] = {
            "radius_m": radius,
            "reason": (
                "production Maxwell gate ROI; full-aperture residuals remain reported under "
                "maxwell_vacuum because off-axis coupler/material boundaries are not represented "
                "by the cylindrical beam-channel mask"
            ),
            **maxwell_residual_diagnostics(
                field,
                core_mask,
                wall_guard_cells=wall_guard_cells,
                axis_guard_cells=axis_guard_cells,
                active_field_fraction=active_field_fraction,
            ),
        }
    return result


__all__ = [
    "EPSILON0_FPM",
    "QUALITY_SCHEMA_VERSION",
    "SPEED_OF_LIGHT_MPS",
    "axis_regularity_diagnostics",
    "field_norm_diagnostics",
    "guarded_vacuum_mask",
    "maxwell_residual_diagnostics",
    "qualify_axisymmetric_field",
    "tail_profile_diagnostics",
]
