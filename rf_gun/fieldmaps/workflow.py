"""Project-level planning and tail checks for the XFdtd cavity artifact.

This module deliberately contains no RF-Track dependency.  It turns the raw
component-specific XFdtd support into one uniform RF-Track grid, constructs the
known vacuum channel mask, and makes the unmeasured downstream interval an
explicit (and testable) modeling choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..aperture import R_CAV_MM, R2_MM, aperture_radius_profile_mm
from .coordinates import MU0_HPM, XFD_TD_CATHODE_TO_BEAM, FrameTransform
from .harmonic import HarmonicFitConfig, fit_fixed_frequency_phasor
from .models import AxisymmetricRFField
from .tail import (
    EvanescentTailModel,
    circular_waveguide_tm01_decay_length_m,
    extend_axisymmetric_with_modal_tail,
    fit_evanescent_tail,
)
from .xfdtd_h5 import H5Selection, XFdtdH5


CAVITY_SOURCE_SHA256 = "095ead03c4d20133e55d63c39bc61608304c54ea965d09d8571edd3046072ab7"
PRODUCTION_QUALITY_GATE_VERSION = "xfdtd_m0_gate_v2"
REQUIRED_PRODUCTION_GATE_NAMES = frozenset(
    {
        "temporal_global_fit",
        "temporal_significant_family_block_fit",
        "temporal_envelope_drift_per_cycle",
        "maxwell_faraday_beam_core",
        "maxwell_ampere_beam_core",
        "axis_Er_zero",
        "axis_Btheta_zero",
        "downstream_tail",
        "cathode_origin",
    }
)


@dataclass(frozen=True)
class BeamFrameSupport:
    """Common six-component rectangular support in cathode-centered beam axes."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    z_min_m: float
    z_max_m: float
    full_azimuth_radius_m: float


@dataclass(frozen=True)
class AxisymmetricGridPlan:
    """Uniform measured/full grids and their physical-vacuum masks."""

    r_m: np.ndarray
    measured_z_m: np.ndarray
    full_z_m: np.ndarray
    measured_vacuum_mask: np.ndarray
    full_vacuum_mask: np.ndarray
    measured_aperture_radius_m: np.ndarray
    full_aperture_radius_m: np.ndarray
    source_support: BeamFrameSupport
    requested_dr_m: float
    requested_dz_m: float
    delta_cathode_chamfer_mm: float

    def __post_init__(self) -> None:
        for name in (
            "r_m",
            "measured_z_m",
            "full_z_m",
            "measured_aperture_radius_m",
            "full_aperture_radius_m",
        ):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        for name in ("measured_vacuum_mask", "full_vacuum_mask"):
            values = np.array(getattr(self, name), dtype=bool, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def dr_m(self) -> float:
        return float(self.r_m[1] - self.r_m[0])

    @property
    def dz_m(self) -> float:
        return float(self.full_z_m[1] - self.full_z_m[0])

    @property
    def measured_z_max_m(self) -> float:
        return float(self.measured_z_m[-1])

    @property
    def target_z_max_m(self) -> float:
        return float(self.full_z_m[-1])

    @property
    def missing_length_m(self) -> float:
        return self.target_z_max_m - self.measured_z_max_m

    def as_metadata(self) -> dict[str, Any]:
        return {
            "requested_dr_m": self.requested_dr_m,
            "requested_dz_m": self.requested_dz_m,
            "realized_dr_m": self.dr_m,
            "realized_dz_m": self.dz_m,
            "r_count": int(self.r_m.size),
            "measured_z_count": int(self.measured_z_m.size),
            "full_z_count": int(self.full_z_m.size),
            "measured_z_max_m": self.measured_z_max_m,
            "target_z_max_m": self.target_z_max_m,
            "missing_length_m": self.missing_length_m,
            "delta_cathode_chamfer_mm": self.delta_cathode_chamfer_mm,
            "source_common_full_azimuth_radius_m": self.source_support.full_azimuth_radius_m,
            "source_common_z_min_m": self.source_support.z_min_m,
            "source_common_z_max_m": self.source_support.z_max_m,
        }


@dataclass(frozen=True)
class TailQualification:
    """Measured end-field and below-cutoff-tail evidence."""

    model: EvanescentTailModel
    metrics: Mapping[str, float]
    thresholds: Mapping[str, float]
    checks: Mapping[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.passed else "fail",
            "model": self.model.as_metadata(),
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "checks": dict(self.checks),
        }


@dataclass(frozen=True)
class CathodeOriginPhysicalCheck:
    """Physical evidence locating the emitting surface on an XFdtd Yee grid.

    ``beam_z_m`` and ``normal_field_phasor_Vpm`` contain only the fitted on-axis
    line used by the check.  The material/vacuum indices identify the adjacent
    rising-field pair selected as the physical conductor boundary.  A declared
    origin passes only when that pair brackets beam ``z=0``, the material sample
    is sufficiently suppressed, and the vacuum sample carries a resolvable RF
    signal.
    """

    source_path: str
    normal_native_component: str
    native_to_beam_normal_sign: float
    fixed_native_indices: Mapping[str, int]
    fixed_native_coordinates_m: Mapping[str, float]
    beam_transverse_offset_m: tuple[float, float]
    beam_z_m: np.ndarray
    normal_field_phasor_Vpm: np.ndarray
    material_sample_index: int | None
    vacuum_sample_index: int | None
    transition_detection: str
    maximum_material_to_vacuum_ratio: float
    minimum_vacuum_fraction_of_line_peak: float
    origin_search_half_width_m: float
    frequency_hz: float
    reference_time_s: float
    time_sample_count: int
    fit_global_nrmse: float
    fit_active_nrmse_p95: float
    fit_peak_fractional_drift_per_cycle: float
    logical_bytes_read: int
    estimated_decompressed_bytes: int | None

    def __post_init__(self) -> None:
        z = np.array(self.beam_z_m, dtype=np.float64, copy=True)
        phasor = np.array(self.normal_field_phasor_Vpm, dtype=np.complex128, copy=True)
        if z.ndim != 1 or phasor.shape != z.shape or z.size < 2:
            raise ValueError("cathode-origin line arrays must be matching 1-D arrays with >=2 samples")
        if not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0.0):
            raise ValueError("beam_z_m must be finite and strictly increasing")
        if not np.all(np.isfinite(phasor.real)) or not np.all(np.isfinite(phasor.imag)):
            raise ValueError("normal_field_phasor_Vpm contains non-finite values")
        for index in (self.material_sample_index, self.vacuum_sample_index):
            if index is not None and not 0 <= int(index) < z.size:
                raise ValueError("transition sample index lies outside the fitted line")
        if (self.material_sample_index is None) != (self.vacuum_sample_index is None):
            raise ValueError("material and vacuum transition indices must either both exist or both be absent")
        if self.material_sample_index is not None:
            if int(self.vacuum_sample_index) != int(self.material_sample_index) + 1:
                raise ValueError("material/vacuum transition samples must be adjacent")
        z.setflags(write=False)
        phasor.setflags(write=False)
        object.__setattr__(self, "beam_z_m", z)
        object.__setattr__(self, "normal_field_phasor_Vpm", phasor)
        object.__setattr__(self, "fixed_native_indices", MappingProxyType(dict(self.fixed_native_indices)))
        object.__setattr__(
            self,
            "fixed_native_coordinates_m",
            MappingProxyType(dict(self.fixed_native_coordinates_m)),
        )

    @property
    def transition_midpoint_m(self) -> float | None:
        if self.material_sample_index is None:
            return None
        left = float(self.beam_z_m[self.material_sample_index])
        right = float(self.beam_z_m[self.vacuum_sample_index])
        return 0.5 * (left + right)

    @property
    def transition_resolution_m(self) -> float | None:
        if self.material_sample_index is None:
            return None
        return float(
            self.beam_z_m[self.vacuum_sample_index]
            - self.beam_z_m[self.material_sample_index]
        )

    @property
    def material_sample_beam_z_m(self) -> float | None:
        if self.material_sample_index is None:
            return None
        return float(self.beam_z_m[self.material_sample_index])

    @property
    def vacuum_sample_beam_z_m(self) -> float | None:
        if self.vacuum_sample_index is None:
            return None
        return float(self.beam_z_m[self.vacuum_sample_index])

    @property
    def material_field_amplitude_Vpm(self) -> float | None:
        if self.material_sample_index is None:
            return None
        return float(abs(self.normal_field_phasor_Vpm[self.material_sample_index]))

    @property
    def vacuum_field_amplitude_Vpm(self) -> float | None:
        if self.vacuum_sample_index is None:
            return None
        return float(abs(self.normal_field_phasor_Vpm[self.vacuum_sample_index]))

    @property
    def line_peak_amplitude_Vpm(self) -> float:
        return float(np.max(np.abs(self.normal_field_phasor_Vpm)))

    @property
    def field_suppression_ratio(self) -> float | None:
        material = self.material_field_amplitude_Vpm
        vacuum = self.vacuum_field_amplitude_Vpm
        if material is None or vacuum is None or vacuum <= 0.0:
            return None
        return material / vacuum

    @property
    def vacuum_fraction_of_line_peak(self) -> float | None:
        vacuum = self.vacuum_field_amplitude_Vpm
        peak = self.line_peak_amplitude_Vpm
        if vacuum is None or peak <= 0.0:
            return None
        return vacuum / peak

    @property
    def transition_brackets_zero(self) -> bool:
        material_z = self.material_sample_beam_z_m
        vacuum_z = self.vacuum_sample_beam_z_m
        return bool(
            material_z is not None
            and vacuum_z is not None
            and material_z <= 0.0 <= vacuum_z
        )

    @property
    def material_field_is_suppressed(self) -> bool:
        ratio = self.field_suppression_ratio
        return bool(
            ratio is not None and ratio <= self.maximum_material_to_vacuum_ratio
        )

    @property
    def vacuum_signal_is_resolved(self) -> bool:
        fraction = self.vacuum_fraction_of_line_peak
        return bool(
            fraction is not None
            and fraction >= self.minimum_vacuum_fraction_of_line_peak
        )

    @property
    def origin_physically_bracketed(self) -> bool:
        return bool(
            self.transition_brackets_zero
            and self.material_field_is_suppressed
            and self.vacuum_signal_is_resolved
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible report; the plottable line stays on the object."""

        return {
            "status": "pass" if self.origin_physically_bracketed else "fail",
            "source_path": self.source_path,
            "origin_physically_bracketed": self.origin_physically_bracketed,
            "normal_native_component": self.normal_native_component,
            "native_to_beam_normal_sign": self.native_to_beam_normal_sign,
            "on_axis_sample": {
                "fixed_native_indices": dict(self.fixed_native_indices),
                "fixed_native_coordinates_m": dict(self.fixed_native_coordinates_m),
                "beam_transverse_offset_m": list(self.beam_transverse_offset_m),
                "line_sample_count": int(self.beam_z_m.size),
            },
            "transition": {
                "detection": self.transition_detection,
                "material_sample_index": self.material_sample_index,
                "material_sample_beam_z_m": self.material_sample_beam_z_m,
                "material_field_amplitude_Vpm": self.material_field_amplitude_Vpm,
                "vacuum_sample_index": self.vacuum_sample_index,
                "vacuum_sample_beam_z_m": self.vacuum_sample_beam_z_m,
                "vacuum_field_amplitude_Vpm": self.vacuum_field_amplitude_Vpm,
                "midpoint_beam_z_m": self.transition_midpoint_m,
                "resolution_m": self.transition_resolution_m,
                "field_suppression_ratio": self.field_suppression_ratio,
                "vacuum_fraction_of_line_peak": self.vacuum_fraction_of_line_peak,
                "brackets_zero": self.transition_brackets_zero,
                "material_field_is_suppressed": self.material_field_is_suppressed,
                "vacuum_signal_is_resolved": self.vacuum_signal_is_resolved,
            },
            "thresholds": {
                "maximum_material_to_vacuum_ratio": self.maximum_material_to_vacuum_ratio,
                "minimum_vacuum_fraction_of_line_peak": self.minimum_vacuum_fraction_of_line_peak,
                "origin_search_half_width_m": self.origin_search_half_width_m,
            },
            "harmonic_fit": {
                "frequency_hz": self.frequency_hz,
                "reference_time_s": self.reference_time_s,
                "time_sample_count": self.time_sample_count,
                "global_nrmse": self.fit_global_nrmse,
                "active_nrmse_p95": self.fit_active_nrmse_p95,
                "peak_fractional_drift_per_cycle": self.fit_peak_fractional_drift_per_cycle,
            },
            "bounded_read": {
                "logical_bytes": self.logical_bytes_read,
                "estimated_decompressed_bytes": self.estimated_decompressed_bytes,
            },
        }


def _signed_axis_for_beam_direction(
    transform: FrameTransform,
    beam_axis: int,
) -> tuple[int, float]:
    """Return the one native axis/sign mapping to a beam axis."""

    rotation = np.asarray(transform.R_target_from_native, dtype=np.float64)
    magnitude = np.abs(rotation)
    permutation = magnitude > 0.5
    if not (
        np.all(np.sum(permutation, axis=0) == 1)
        and np.all(np.sum(permutation, axis=1) == 1)
        and np.allclose(magnitude[permutation], 1.0, atol=1.0e-12, rtol=0.0)
        and np.allclose(magnitude[~permutation], 0.0, atol=1.0e-12, rtol=0.0)
    ):
        raise ValueError("cathode-origin check requires a signed-axis-permutation transform")
    native_axis = int(np.flatnonzero(permutation[int(beam_axis)])[0])
    return native_axis, float(rotation[int(beam_axis), native_axis])


def _detect_cathode_transition_pair(
    beam_z_m: np.ndarray,
    amplitude_Vpm: np.ndarray,
    *,
    search_half_width_m: float,
    minimum_vacuum_fraction_of_line_peak: float,
) -> tuple[int | None, int | None, str]:
    """Locate the material-to-vacuum rising edge nearest the declared origin."""

    z = np.asarray(beam_z_m, dtype=np.float64)
    amplitude = np.asarray(amplitude_Vpm, dtype=np.float64)
    midpoints = 0.5 * (z[:-1] + z[1:])
    candidates = np.flatnonzero(np.abs(midpoints) <= float(search_half_width_m))
    if candidates.size == 0:
        return None, None, "no_adjacent_pair_in_search_window"
    peak = float(np.max(amplitude))
    active_threshold = float(minimum_vacuum_fraction_of_line_peak) * peak
    crossings = candidates[
        (amplitude[candidates] < active_threshold)
        & (amplitude[candidates + 1] >= active_threshold)
    ]
    tiny = max(np.finfo(np.float64).tiny, peak * 1.0e-15)
    if crossings.size:
        # Prefer the physical threshold crossing closest to the declared origin;
        # contrast is a deterministic tie-breaker on symmetric grids.
        ratios = (amplitude[crossings] + tiny) / (amplitude[crossings + 1] + tiny)
        order = np.lexsort((ratios, np.abs(midpoints[crossings])))
        material = int(crossings[order[0]])
        return material, material + 1, "active_threshold_rising_edge"
    # Preserve a useful failed report even when the line never crosses the
    # resolvable-signal threshold: select its strongest local rising contrast.
    ratios = (amplitude[candidates] + tiny) / (amplitude[candidates + 1] + tiny)
    order = np.lexsort((np.abs(midpoints[candidates]), ratios))
    material = int(candidates[order[0]])
    return material, material + 1, "maximum_rising_contrast_fallback"


def check_xfdtd_cathode_origin(
    source_path: str | Path,
    *,
    transform: FrameTransform = XFD_TD_CATHODE_TO_BEAM,
    harmonic: HarmonicFitConfig | None = None,
    maximum_material_to_vacuum_ratio: float = 1.0e-3,
    minimum_vacuum_fraction_of_line_peak: float = 1.0e-3,
    origin_search_half_width_m: float = 1.0e-3,
    max_logical_read_bytes: int = 64 * 1024 * 1024,
) -> CathodeOriginPhysicalCheck:
    """Verify the declared cathode origin using the physical normal-E jump.

    Only one field selection is performed: every exact electric timestamp on
    the nearest-on-axis line of the native E component that maps to beam ``Ez``.
    Component-specific Yee coordinates determine both the transverse indices and
    beam-z positions; the signed vector transform determines the component sign.
    """

    maximum_ratio = float(maximum_material_to_vacuum_ratio)
    minimum_fraction = float(minimum_vacuum_fraction_of_line_peak)
    search_half_width = float(origin_search_half_width_m)
    if not np.isfinite(maximum_ratio) or maximum_ratio < 0.0:
        raise ValueError("maximum_material_to_vacuum_ratio must be finite and non-negative")
    if not np.isfinite(minimum_fraction) or not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_vacuum_fraction_of_line_peak must lie in (0, 1]")
    if not np.isfinite(search_half_width) or search_half_width <= 0.0:
        raise ValueError("origin_search_half_width_m must be finite and positive")

    native_axis, normal_sign = _signed_axis_for_beam_direction(transform, 2)
    axis_names = "xyz"
    native_component = axis_names[native_axis]
    with XFdtdH5(source_path, max_logical_read_bytes=max_logical_read_bytes) as reader:
        manifest = reader.manifest()
        native_axes = reader.coordinates("E", native_component, frame="native")
        component_shape = reader.component_shape("E", native_component)
        indices: list[int | slice] = []
        fixed_indices: dict[str, int] = {}
        fixed_coordinates: dict[str, float] = {}
        for axis_index, (axis_name, coordinate) in enumerate(zip(axis_names, native_axes, strict=True)):
            if axis_index == native_axis:
                indices.append(slice(0, component_shape[axis_index + 1]))
                continue
            nearest = int(np.argmin(np.abs(coordinate - transform.origin_native_m[axis_index])))
            indices.append(nearest)
            fixed_indices[axis_name] = nearest
            fixed_coordinates[axis_name] = float(coordinate[nearest])
        selection = H5Selection(
            slice(0, component_shape[0]), indices[0], indices[1], indices[2]
        )
        estimate = reader.estimate_read("E", native_component, selection)
        samples = reader.read_component("E", native_component, selection)
        exact_time_s = reader.times("E")

    expected_shape = (exact_time_s.size, native_axes[native_axis].size)
    if samples.shape != expected_shape:
        raise ValueError(
            f"normal-E line has shape {samples.shape}; expected {expected_shape}"
        )
    drive_frequency = float(manifest.drive_frequency_hz)
    if harmonic is None:
        fit_config = HarmonicFitConfig(
            drive_frequency,
            include_offset=True,
            include_linear_envelope=True,
        )
    else:
        if not np.isclose(
            float(harmonic.frequency_hz), drive_frequency, rtol=1.0e-12, atol=0.0
        ):
            raise ValueError(
                "cathode-origin harmonic frequency must match the serialized XFdtd drive frequency"
            )
        fit_config = harmonic
    fitted = fit_fixed_frequency_phasor(
        exact_time_s,
        normal_sign * samples,
        fit_config,
        axis=0,
    )
    native_longitudinal_coordinate = np.asarray(native_axes[native_axis], dtype=np.float64)
    beam_z = normal_sign * (
        native_longitudinal_coordinate - transform.origin_native_m[native_axis]
    )
    order = np.argsort(beam_z)
    beam_z = beam_z[order]
    phasor = np.asarray(fitted.phasor)[order]
    if np.any(np.diff(beam_z) <= 0.0):
        raise ValueError("normal-component Yee coordinate does not map to a unique beam-z line")
    material_index, vacuum_index, detection = _detect_cathode_transition_pair(
        beam_z,
        np.abs(phasor),
        search_half_width_m=search_half_width,
        minimum_vacuum_fraction_of_line_peak=minimum_fraction,
    )

    native_axis_point = np.array(transform.origin_native_m, dtype=np.float64, copy=True)
    for axis_name, coordinate in fixed_coordinates.items():
        native_axis_point[axis_names.index(axis_name)] = coordinate
    beam_offset = transform.transform_points(native_axis_point)[:2]
    return CathodeOriginPhysicalCheck(
        source_path=str(Path(source_path).resolve()),
        normal_native_component=f"E{native_component}",
        native_to_beam_normal_sign=normal_sign,
        fixed_native_indices=fixed_indices,
        fixed_native_coordinates_m=fixed_coordinates,
        beam_transverse_offset_m=(float(beam_offset[0]), float(beam_offset[1])),
        beam_z_m=beam_z,
        normal_field_phasor_Vpm=phasor,
        material_sample_index=material_index,
        vacuum_sample_index=vacuum_index,
        transition_detection=detection,
        maximum_material_to_vacuum_ratio=maximum_ratio,
        minimum_vacuum_fraction_of_line_peak=minimum_fraction,
        origin_search_half_width_m=search_half_width,
        frequency_hz=drive_frequency,
        reference_time_s=float(fitted.reference_time_s),
        time_sample_count=int(exact_time_s.size),
        fit_global_nrmse=float(fitted.global_nrmse),
        fit_active_nrmse_p95=float(fitted.active_nrmse_p95),
        fit_peak_fractional_drift_per_cycle=float(
            fitted.peak_fractional_drift_per_cycle
        ),
        logical_bytes_read=int(estimate.logical_bytes),
        estimated_decompressed_bytes=(
            None
            if estimate.estimated_decompressed_bytes is None
            else int(estimate.estimated_decompressed_bytes)
        ),
    )



def _axis_forcing_fraction(reduction_report: Mapping[str, Any], component: str) -> float | None:
    """Pre-forcing on-axis residual for ``component``, or None for an older report."""

    record = (
        (reduction_report or {}).get("symmetry", {}).get("axis_forcing", {}).get(component, {})
    )
    value = record.get("pre_forcing_fraction_of_combined_field_rms")
    return None if value is None else float(value)


def production_quality_gates(
    reduction_report: Mapping[str, Any],
    numerical_diagnostics: Mapping[str, Any],
    tail_diagnostics: Mapping[str, Any],
    cathode_origin_diagnostics: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Evaluate the single production gate set used by CLI and notebook builds.

    Keeping this public and RF-Track independent prevents an interactive writer
    from conferring ``qualified`` status with a weaker subset of checks than the
    automated preparation command.
    """

    try:
        temporal = reduction_report["temporal"]
        active_maxwell = numerical_diagnostics["maxwell_beam_core"][
            "active_guarded_vacuum"
        ]["combined"]
        axis = numerical_diagnostics["axis_regularity"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "production quality gates require temporal, beam-core active-vacuum Maxwell, "
            "and axis-regularity diagnostics"
        ) from exc

    production_temporal = temporal.get("production_metrics", {})
    temporal_global = production_temporal.get(
        "max_family_combined_nrmse", temporal.get("max_global_nrmse")
    )
    temporal_block = production_temporal.get(
        "max_significant_family_block_nrmse"
    )
    temporal_drift = production_temporal.get(
        "max_family_combined_fractional_drift_per_cycle",
        temporal.get("max_peak_fractional_drift_per_cycle"),
    )
    if temporal_global is None or temporal_block is None or temporal_drift is None:
        raise ValueError("temporal diagnostics do not contain production qualification metrics")

    gates: dict[str, dict[str, Any]] = {
        "temporal_global_fit": {
            "value": float(temporal_global),
            "maximum": 0.02,
            "metric": "max E/H-family cylindrical-energy-weighted NRMSE",
        },
        "temporal_significant_family_block_fit": {
            "value": float(temporal_block),
            "maximum": 0.02,
            "metric": (
                "worst vector-family block NRMSE among blocks with >=1e-4 family signal; "
                "per-component local p95 remains diagnostic because component nodal planes "
                "make its relative denominator ill-conditioned"
            ),
        },
        "temporal_envelope_drift_per_cycle": {
            "value": float(temporal_drift),
            "maximum": 0.02,
            "metric": "max E/H-family cylindrical-energy-weighted phasor drift per cycle",
        },
        "maxwell_faraday_beam_core": {
            "value": float(active_maxwell["faraday_family_normalized_rms"]),
            "maximum": 0.20,
        },
        "maxwell_ampere_beam_core": {
            "value": float(active_maxwell["ampere_family_normalized_rms"]),
            "maximum": 0.20,
        },
        # Gate the PRE-forcing projection residual when the reducer recorded it. The stored
        # axis values are hard-zeroed, so gating those would always pass no matter how
        # inconsistent the m=0 projection actually was.
        "axis_Er_zero": {
            "value": float(
                _axis_forcing_fraction(reduction_report, "Er")
                if _axis_forcing_fraction(reduction_report, "Er") is not None
                else axis["odd_components"]["Er"]["fraction_of_combined_field_rms"]
            ),
            "maximum": 1.0e-3,
            "metric": (
                "projected |Er| on axis before the exact-zero forcing, relative to the "
                "combined field RMS"
            ),
        },
        "axis_Btheta_zero": {
            "value": float(
                _axis_forcing_fraction(reduction_report, "Btheta")
                if _axis_forcing_fraction(reduction_report, "Btheta") is not None
                else axis["odd_components"]["Btheta"]["fraction_of_combined_field_rms"]
            ),
            "maximum": 1.0e-3,
            "metric": (
                "projected |Btheta| on axis before the exact-zero forcing, relative to the "
                "combined field RMS"
            ),
        },
        "downstream_tail": {
            "passed": str(tail_diagnostics.get("status")) == "pass",
        },
        "cathode_origin": {
            "passed": (
                str(cathode_origin_diagnostics.get("status")) == "pass"
                and bool(cathode_origin_diagnostics.get("origin_physically_bracketed", False))
            ),
            "metric": "normal-E material/vacuum transition physically brackets beam z=0",
        },
    }
    for name, gate in gates.items():
        if "passed" not in gate:
            value = float(gate["value"])
            maximum = float(gate["maximum"])
            if not np.isfinite(value):
                gate["passed"] = False
                gate["failure"] = "non_finite_value"
            else:
                gate["passed"] = bool(value <= maximum)
        gate["gate_version"] = PRODUCTION_QUALITY_GATE_VERSION
        gate["name"] = name
    return gates


def resolve_production_quality_status(
    requested: str,
    gates: Mapping[str, Mapping[str, Any]],
) -> str:
    """Resolve ``auto|qualified|analysis_only`` against objective gate results."""

    normalized = str(requested).strip().lower()
    if normalized not in {"auto", "qualified", "analysis_only"}:
        raise ValueError("requested quality status must be auto, qualified, or analysis_only")

    # Absence of evidence is not a pass. Resolving purely on "nothing failed" makes an empty
    # or truncated gate mapping resolve to 'qualified', so a build that never evaluated the
    # production checks would publish as production. Require the full required set, plus the
    # current gate version on each, before any qualified verdict is reachable.
    missing = sorted(REQUIRED_PRODUCTION_GATE_NAMES - set(gates))
    stale = sorted(
        name
        for name in REQUIRED_PRODUCTION_GATE_NAMES & set(gates)
        if str(gates[name].get("gate_version", PRODUCTION_QUALITY_GATE_VERSION))
        != PRODUCTION_QUALITY_GATE_VERSION
    )
    if normalized == "qualified" and (missing or stale):
        raise RuntimeError(
            "qualified status was requested but the gate set is incomplete: "
            f"missing={missing}, stale={stale}"
        )

    failed = [name for name, gate in gates.items() if not bool(gate.get("passed", False))]
    if missing or stale:
        return "analysis_only"
    if normalized == "qualified" and failed:
        raise RuntimeError(
            "qualified status was requested but objective gates failed: "
            + ", ".join(failed)
        )
    if normalized == "analysis_only":
        return "analysis_only"
    return "qualified" if not failed else "analysis_only"


def _signed_axis_interval(axis: np.ndarray, sign: float) -> tuple[float, float]:
    values = sign * np.asarray(axis, dtype=np.float64)
    return float(np.min(values)), float(np.max(values))


def common_beam_frame_support(
    source_path: str,
    *,
    transform: FrameTransform = XFD_TD_CATHODE_TO_BEAM,
) -> BeamFrameSupport:
    """Find the intersection supported by every staggered E/H component.

    The present XFdtd-to-beam transform is a signed axis permutation.  Refusing
    a general oblique rotation here is intentional: the axis-aligned bounds of
    an obliquely transformed box would not be sufficient to guarantee that all
    requested cylindrical points are inside the native box.
    """

    rotation = np.asarray(transform.R_target_from_native)
    nonzero = np.abs(rotation) > 0.5
    if not (
        np.all(np.sum(nonzero, axis=0) == 1)
        and np.all(np.sum(nonzero, axis=1) == 1)
        and np.allclose(np.abs(rotation[nonzero]), 1.0, atol=1.0e-12)
    ):
        raise ValueError("common_beam_frame_support requires a signed-axis-permutation transform")

    intervals: list[list[tuple[float, float]]] = [[], [], []]
    with XFdtdH5(source_path) as reader:
        for quantity in ("E", "H"):
            for component in "xyz":
                native_axes = reader.coordinates(
                    quantity, component, frame="cathode_centered"
                )
                for beam_axis in range(3):
                    native_axis = int(np.flatnonzero(nonzero[beam_axis])[0])
                    sign = float(rotation[beam_axis, native_axis])
                    intervals[beam_axis].append(
                        _signed_axis_interval(native_axes[native_axis], sign)
                    )

    common = []
    for beam_axis_intervals in intervals:
        lower = max(item[0] for item in beam_axis_intervals)
        upper = min(item[1] for item in beam_axis_intervals)
        if lower >= upper:
            raise ValueError("the six XFdtd components have no common beam-frame support")
        common.append((lower, upper))
    x, y, z = common
    full_radius = min(-x[0], x[1], -y[0], y[1])
    if full_radius <= 0.0:
        raise ValueError("common XFdtd support does not enclose the beam axis")
    return BeamFrameSupport(
        x_min_m=x[0],
        x_max_m=x[1],
        y_min_m=y[0],
        y_max_m=y[1],
        z_min_m=z[0],
        z_max_m=z[1],
        full_azimuth_radius_m=float(full_radius),
    )


def _uniform_axis_to_endpoint(endpoint_m: float, requested_step_m: float) -> np.ndarray:
    endpoint = float(endpoint_m)
    step = float(requested_step_m)
    if not np.isfinite(endpoint) or endpoint <= 0.0:
        raise ValueError("endpoint_m must be finite and positive")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("requested_step_m must be finite and positive")
    intervals = max(1, int(np.round(endpoint / step)))
    return np.linspace(0.0, endpoint, intervals + 1, dtype=np.float64)


def _vacuum_mask(
    r_m: np.ndarray,
    z_m: np.ndarray,
    *,
    delta_cathode_chamfer_mm: float,
    wall_margin_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    aperture = aperture_radius_profile_mm(
        np.asarray(z_m) * 1.0e3, float(delta_cathode_chamfer_mm)
    ) * 1.0e-3
    usable_radius = aperture - float(wall_margin_m)
    if np.any(usable_radius <= 0.0):
        raise ValueError("wall_margin_m removes the complete vacuum channel")
    # A tiny tolerance avoids classifying a mathematically coincident wall point
    # differently solely because of binary rounding.  RF-Track separately applies
    # the same aperture profile as the actual particle-loss boundary.
    tolerance = 8.0 * np.finfo(np.float64).eps * max(float(np.max(aperture)), 1.0)
    mask = np.asarray(r_m)[None, :] <= usable_radius[:, None] + tolerance
    return np.asarray(mask, dtype=bool), np.asarray(aperture, dtype=np.float64)


def plan_axisymmetric_grid(
    source_path: str,
    *,
    target_z_max_m: float,
    requested_dr_m: float = 100.0e-6,
    requested_dz_m: float = 50.0e-6,
    r_max_m: float = R_CAV_MM * 1.0e-3,
    delta_cathode_chamfer_mm: float = 0.0,
    wall_margin_m: float = 0.0,
    transform: FrameTransform = XFD_TD_CATHODE_TO_BEAM,
) -> AxisymmetricGridPlan:
    """Create one grid whose measured prefix and tail extension stay aligned."""

    support = common_beam_frame_support(source_path, transform=transform)
    requested_radius = float(r_max_m)
    if requested_radius > support.full_azimuth_radius_m + 2.0e-12:
        raise ValueError(
            f"r_max_m={requested_radius:.9g} exceeds the common full-azimuth source support "
            f"{support.full_azimuth_radius_m:.9g} m"
        )
    if support.z_max_m <= 0.0 or support.z_min_m > 0.0:
        raise ValueError(
            f"source support [{support.z_min_m:.9g}, {support.z_max_m:.9g}] m does not contain z=0"
        )
    full_z = _uniform_axis_to_endpoint(target_z_max_m, requested_dz_m)
    measured_count = int(np.searchsorted(full_z, support.z_max_m + 2.0e-12, side="right"))
    if measured_count < 3:
        raise ValueError("fewer than three planned z samples lie inside the measured source")
    measured_z = full_z[:measured_count]
    r = _uniform_axis_to_endpoint(requested_radius, requested_dr_m)
    measured_mask, measured_aperture = _vacuum_mask(
        r,
        measured_z,
        delta_cathode_chamfer_mm=delta_cathode_chamfer_mm,
        wall_margin_m=wall_margin_m,
    )
    full_mask, full_aperture = _vacuum_mask(
        r,
        full_z,
        delta_cathode_chamfer_mm=delta_cathode_chamfer_mm,
        wall_margin_m=wall_margin_m,
    )
    return AxisymmetricGridPlan(
        r_m=r,
        measured_z_m=measured_z,
        full_z_m=full_z,
        measured_vacuum_mask=measured_mask,
        full_vacuum_mask=full_mask,
        measured_aperture_radius_m=measured_aperture,
        full_aperture_radius_m=full_aperture,
        source_support=support,
        requested_dr_m=float(requested_dr_m),
        requested_dz_m=float(requested_dz_m),
        delta_cathode_chamfer_mm=float(delta_cathode_chamfer_mm),
    )


def _time_averaged_energy_per_length(field: AxisymmetricRFField) -> np.ndarray:
    electric_squared = np.abs(field.Er_Vpm) ** 2 + np.abs(field.Ez_Vpm) ** 2
    magnetic_squared = np.abs(field.Btheta_T) ** 2 + np.abs(field.Bz_T) ** 2
    epsilon0 = 1.0 / (MU0_HPM * 299_792_458.0**2)
    energy_density = 0.25 * (
        epsilon0 * electric_squared + magnetic_squared / MU0_HPM
    )
    return np.trapezoid(
        energy_density * (2.0 * np.pi * field.r_m[None, :]),
        field.r_m,
        axis=1,
    )


def qualify_evanescent_tail(
    field: AxisymmetricRFField,
    *,
    target_z_max_m: float,
    fit_start_z_m: float = 30.0e-3,
    pipe_radius_m: float = R2_MM * 1.0e-3,
    maximum_endpoint_field_fraction: float = 0.01,
    maximum_endpoint_energy_fraction: float = 1.0e-3,
    maximum_decay_length_relative_error: float = 0.10,
    minimum_log_amplitude_r_squared: float = 0.995,
    maximum_tail_voltage_fraction: float = 1.0e-3,
    maximum_fit_phase_change_deg: float = 5.0,
) -> TailQualification:
    """Fit the on-axis tail and gate a marked modal extension.

    Axisymmetry is intentionally absent from these gates.  The project policy is
    to report non-axisymmetric E/B content while still using the cylindrical m=0
    projection for RF-Track.
    """

    model = fit_evanescent_tail(
        field.z_m,
        field.Ez_Vpm[:, 0],
        fit_start_z_m=float(fit_start_z_m),
    )
    analytic_length = circular_waveguide_tm01_decay_length_m(
        float(pipe_radius_m), field.frequency_hz
    )
    e_magnitude = np.sqrt(np.abs(field.Er_Vpm) ** 2 + np.abs(field.Ez_Vpm) ** 2)
    b_magnitude = np.sqrt(np.abs(field.Btheta_T) ** 2 + np.abs(field.Bz_T) ** 2)
    endpoint_e_fraction = float(np.max(e_magnitude[-1]) / np.max(e_magnitude))
    endpoint_b_fraction = float(np.max(b_magnitude[-1]) / np.max(b_magnitude))
    energy_per_length = _time_averaged_energy_per_length(field)
    endpoint_energy_fraction = float(energy_per_length[-1] / np.max(energy_per_length))
    decay_error = abs(model.decay_length_m - analytic_length) / analytic_length
    fit_phase_change_deg = abs(
        np.rad2deg(model.phase_constant_per_m * (model.anchor_z_m - model.fit_start_z_m))
    )
    missing_voltage = model.integrated_reference_tail(float(target_z_max_m))
    measured_voltage = np.trapezoid(field.Ez_Vpm[:, 0], field.z_m)
    measured_voltage_abs = abs(complex(measured_voltage))
    tail_voltage_fraction = (
        abs(missing_voltage) / measured_voltage_abs if measured_voltage_abs > 0.0 else np.inf
    )
    endpoint_reference_mismatch = abs(
        model.reference_phasor - complex(field.Ez_Vpm[-1, 0])
    ) / max(abs(complex(field.Ez_Vpm[-1, 0])), np.finfo(float).tiny)

    metrics = MappingProxyType(
        {
            "measured_z_max_m": float(field.z_m[-1]),
            "target_z_max_m": float(target_z_max_m),
            "missing_length_m": float(target_z_max_m - field.z_m[-1]),
            "endpoint_e_fraction_of_map_peak": endpoint_e_fraction,
            "endpoint_b_fraction_of_map_peak": endpoint_b_fraction,
            "endpoint_energy_per_length_fraction_of_peak": endpoint_energy_fraction,
            "fitted_decay_length_m": model.decay_length_m,
            "analytic_tm01_decay_length_m": analytic_length,
            "decay_length_relative_error": float(decay_error),
            "log_amplitude_r_squared": model.log_amplitude_r_squared,
            "fit_phase_change_deg": float(fit_phase_change_deg),
            "modal_tail_integrated_voltage_V": float(abs(missing_voltage)),
            "measured_complex_voltage_magnitude_V": float(measured_voltage_abs),
            "modal_tail_voltage_fraction": float(tail_voltage_fraction),
            "fit_anchor_to_last_sample_relative_mismatch": float(endpoint_reference_mismatch),
        }
    )
    thresholds = MappingProxyType(
        {
            "maximum_endpoint_field_fraction": float(maximum_endpoint_field_fraction),
            "maximum_endpoint_energy_fraction": float(maximum_endpoint_energy_fraction),
            "maximum_decay_length_relative_error": float(maximum_decay_length_relative_error),
            "minimum_log_amplitude_r_squared": float(minimum_log_amplitude_r_squared),
            "maximum_tail_voltage_fraction": float(maximum_tail_voltage_fraction),
            "maximum_fit_phase_change_deg": float(maximum_fit_phase_change_deg),
        }
    )
    checks = MappingProxyType(
        {
            "electric_field_near_zero_at_measured_end": endpoint_e_fraction
            <= maximum_endpoint_field_fraction,
            "magnetic_field_near_zero_at_measured_end": endpoint_b_fraction
            <= maximum_endpoint_field_fraction,
            "stored_energy_near_zero_at_measured_end": endpoint_energy_fraction
            <= maximum_endpoint_energy_fraction,
            "decay_matches_below_cutoff_tm01": decay_error
            <= maximum_decay_length_relative_error,
            "single_exponential_fit_quality": model.log_amplitude_r_squared
            >= minimum_log_amplitude_r_squared,
            "tail_voltage_is_negligible": tail_voltage_fraction
            <= maximum_tail_voltage_fraction,
            "tail_phase_is_stationary": fit_phase_change_deg <= maximum_fit_phase_change_deg,
        }
    )
    return TailQualification(model, metrics, thresholds, checks)


def extend_axisymmetric_with_zero_tail(
    field: AxisymmetricRFField,
    target_z_m: float,
) -> AxisymmetricRFField:
    """Create the explicit zero-field control for a missing downstream interval."""

    target = float(target_z_m)
    if target <= field.z_m[-1]:
        raise ValueError("target_z_m must lie beyond the measured grid")
    intervals_float = (target - field.z_m[-1]) / field.hz_m
    extra_count = int(np.round(intervals_float))
    if extra_count < 1 or not np.isclose(
        intervals_float, extra_count, rtol=0.0, atol=2.0e-8
    ):
        raise ValueError("target_z_m must align with the field's uniform z grid")
    extra_z = field.z_m[-1] + field.hz_m * np.arange(1, extra_count + 1)
    shape = (extra_count, field.r_m.size)

    def extend(values: np.ndarray) -> np.ndarray:
        return np.concatenate((values, np.zeros(shape, dtype=values.dtype)), axis=0)

    metadata = dict(field.metadata)
    metadata["spatial_support"] = {
        "measured_z_max_m": float(field.z_m[-1]),
        "requested_z_max_m": target,
        "artifact_z_max_m": float(extra_z[-1]),
        "extension": "explicit_zero_field_control",
    }
    return AxisymmetricRFField(
        frequency_hz=field.frequency_hz,
        r_m=field.r_m,
        z_m=np.concatenate((field.z_m, extra_z)),
        Er_Vpm=extend(field.Er_Vpm),
        Ez_Vpm=extend(field.Ez_Vpm),
        Btheta_T=extend(field.Btheta_T),
        Bz_T=extend(field.Bz_T),
        metadata=metadata,
    )


def apply_tail_policy(
    field: AxisymmetricRFField,
    plan: AxisymmetricGridPlan,
    qualification: TailQualification,
    *,
    policy: str = "modal",
) -> AxisymmetricRFField:
    """Apply a qualified modal tail, the zero control, or keep measured support."""

    normalized = str(policy).strip().lower()
    if normalized == "none":
        return field
    if not np.array_equal(field.r_m, plan.r_m) or not np.array_equal(
        field.z_m, plan.measured_z_m
    ):
        raise ValueError("field grids do not match the supplied AxisymmetricGridPlan")
    if normalized == "modal":
        if not qualification.passed:
            failed = [name for name, passed in qualification.checks.items() if not passed]
            raise ValueError(f"modal-tail qualification failed: {failed}")
        extended = extend_axisymmetric_with_modal_tail(
            field,
            plan.target_z_max_m,
            qualification.model,
            require_aligned=True,
        )
    elif normalized == "zero":
        extended = extend_axisymmetric_with_zero_tail(field, plan.target_z_max_m)
    else:
        raise ValueError("tail policy must be 'modal', 'zero', or 'none'")
    if not np.allclose(extended.z_m, plan.full_z_m, rtol=0.0, atol=2.0e-12):
        raise RuntimeError("tail extension did not reproduce the planned uniform full grid")
    # The analytic aperture remains authoritative: no field may survive in metal.
    components = {}
    for name in ("Er_Vpm", "Ez_Vpm", "Btheta_T", "Bz_T"):
        values = np.array(getattr(extended, name), copy=True)
        values[~plan.full_vacuum_mask] = 0.0
        components[name] = values
    metadata = dict(extended.metadata)
    metadata["tail_qualification"] = qualification.as_dict()
    return AxisymmetricRFField(
        frequency_hz=extended.frequency_hz,
        r_m=extended.r_m,
        z_m=extended.z_m,
        metadata=metadata,
        **components,
    )
