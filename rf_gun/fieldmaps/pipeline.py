"""Out-of-core XFdtd volume reduction to an axisymmetric RF field artifact."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.interpolate import RegularGridInterpolator

from .artifact import write_axisymmetric_artifact
from .coordinates import (
    MU0_HPM,
    XFD_TD_CATHODE_TO_BEAM,
    FrameTransform,
    one_sided_surface_limit,
)
from .harmonic import HarmonicFitConfig, HarmonicFitResult, fit_fixed_frequency_phasor
from .models import AxisymmetricRFField
from .sections import coordinate_slice_for_bounds
from .symmetry import cylindrical_samples_from_cartesian, project_cylindrical_m0
from .xfdtd_h5 import H5Selection, XFdtdH5, XFdtdManifest

ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ReductionConfig:
    """Numerical choices for an out-of-core cylindrical reduction."""

    r_m: np.ndarray
    z_m: np.ndarray
    theta_rad: np.ndarray
    vacuum_mask: np.ndarray
    harmonic: HarmonicFitConfig | None = None
    transform: FrameTransform = field(default_factory=lambda: XFD_TD_CATHODE_TO_BEAM)
    longitudinal_block_points: int = 64
    interpolation_guard_cells: int = 1
    wall_guard_cells: int = 1
    beam_core_radius_m: float | None = None
    surface_sample_count: int = 3
    surface_polynomial_order: int = 1
    #: Fraction of the local cell within which a node counts as lying ON the conductor plane.
    #: Yee tangential-E / normal-H components have a node exactly there, carrying the boundary
    #: zero; admitting it turns a long extrapolation into an anchored fit. Components that
    #: straddle the boundary sit half a cell away and are untouched at this fraction.
    surface_on_plane_cell_fraction: float = 0.1
    m_max: int = 4
    output_dtype: str = "complex64"
    max_logical_read_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        r = np.array(self.r_m, dtype=np.float64, copy=True)
        z = np.array(self.z_m, dtype=np.float64, copy=True)
        theta = np.array(self.theta_rad, dtype=np.float64, copy=True)
        vacuum_mask = np.array(self.vacuum_mask, dtype=bool, copy=True)
        if r.ndim != 1 or r.size < 2 or r[0] != 0.0 or np.any(np.diff(r) <= 0.0):
            raise ValueError("r_m must be increasing, contain at least two points, and start exactly at zero")
        if z.ndim != 1 or z.size < 2 or np.any(z < 0.0) or np.any(np.diff(z) <= 0.0):
            raise ValueError("z_m must be an increasing, non-negative 1-D grid")
        for name, axis in (("r_m", r), ("z_m", z)):
            spacing = np.diff(axis)
            if not np.allclose(spacing, spacing[0], rtol=1.0e-8, atol=max(1e-15, spacing[0] * 1e-8)):
                raise ValueError(f"{name} must be uniform")
        if theta.ndim != 1 or theta.size < 2 * int(self.m_max) + 1:
            raise ValueError("theta_rad has too few points for the requested m_max")
        wrapped = np.mod(theta, 2.0 * np.pi)
        if not np.all(np.isfinite(theta)) or np.unique(wrapped).size != theta.size:
            raise ValueError("theta_rad must contain finite, distinct azimuths")
        if self.longitudinal_block_points < 1:
            raise ValueError("longitudinal_block_points must be positive")
        if self.interpolation_guard_cells < 0:
            raise ValueError("interpolation_guard_cells must be non-negative")
        if vacuum_mask.shape != (z.size, r.size) or not np.any(vacuum_mask):
            raise ValueError(
                f"vacuum_mask must have shape {(z.size, r.size)} and select at least one cell"
            )
        if self.wall_guard_cells < 0:
            raise ValueError("wall_guard_cells must be non-negative")
        if self.beam_core_radius_m is not None:
            beam_core_radius = float(self.beam_core_radius_m)
            if not np.isfinite(beam_core_radius) or not 0.0 < beam_core_radius <= r[-1]:
                raise ValueError("beam_core_radius_m must lie in (0, r_m[-1]]")
        if self.surface_sample_count < 2:
            raise ValueError("surface_sample_count must be at least two")
        if not 0 <= self.surface_polynomial_order < self.surface_sample_count:
            raise ValueError("surface polynomial order must be below surface_sample_count")
        if self.output_dtype not in {"complex64", "complex128"}:
            raise ValueError("output_dtype must be 'complex64' or 'complex128'")
        if self.max_logical_read_bytes <= 0:
            raise ValueError("max_logical_read_bytes must be positive")
        r.setflags(write=False)
        z.setflags(write=False)
        theta.setflags(write=False)
        vacuum_mask.setflags(write=False)
        object.__setattr__(self, "r_m", r)
        object.__setattr__(self, "z_m", z)
        object.__setattr__(self, "theta_rad", theta)
        object.__setattr__(self, "vacuum_mask", vacuum_mask)

    def as_metadata(self) -> dict[str, Any]:
        harmonic = self.harmonic
        return {
            "r_min_m": float(self.r_m[0]),
            "r_max_m": float(self.r_m[-1]),
            "r_count": int(self.r_m.size),
            "dr_m": float(self.r_m[1] - self.r_m[0]),
            "z_min_m": float(self.z_m[0]),
            "z_max_m": float(self.z_m[-1]),
            "z_count": int(self.z_m.size),
            "dz_m": float(self.z_m[1] - self.z_m[0]),
            "theta_rad": self.theta_rad.tolist(),
            "vacuum_mask_shape": list(self.vacuum_mask.shape),
            "vacuum_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(self.vacuum_mask, dtype=np.uint8).tobytes()
            ).hexdigest(),
            "origin_native_m": self.transform.origin_native_m.tolist(),
            "R_rf_from_native": self.transform.R_target_from_native.tolist(),
            "longitudinal_block_points": self.longitudinal_block_points,
            "interpolation_guard_cells": self.interpolation_guard_cells,
            "wall_guard_cells": self.wall_guard_cells,
            "beam_core_radius_m": self.beam_core_radius_m,
            "surface_sample_count": self.surface_sample_count,
            "surface_polynomial_order": self.surface_polynomial_order,
            "surface_on_plane_cell_fraction": self.surface_on_plane_cell_fraction,
            "m_max": self.m_max,
            "output_dtype": self.output_dtype,
            "max_logical_read_bytes": self.max_logical_read_bytes,
            "harmonic": None
            if harmonic is None
            else {
                "frequency_hz": harmonic.frequency_hz,
                "reference_time_s": harmonic.reference_time_s,
                "include_offset": harmonic.include_offset,
                "include_linear_envelope": harmonic.include_linear_envelope,
                "active_amplitude_fraction": harmonic.active_amplitude_fraction,
            },
        }


@dataclass(frozen=True)
class ReductionReport:
    source: Mapping[str, Any]
    temporal: Mapping[str, Any]
    symmetry: Mapping[str, Any]
    surface: Mapping[str, Any]
    support: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": dict(self.source),
            "temporal": dict(self.temporal),
            "symmetry": dict(self.symmetry),
            "surface": dict(self.surface),
            "support": dict(self.support),
        }


@dataclass(frozen=True)
class ReductionResult:
    field: AxisymmetricRFField
    report: ReductionReport


@dataclass(frozen=True)
class ReductionArtifactResult:
    reduction: ReductionResult
    payload_sha256: str
    artifact_path: Path


def _notify(callback: ProgressCallback | None, **state: Any) -> None:
    if callback is not None:
        callback(MappingProxyType(state))


def _shared_harmonic_config(manifest: XFdtdManifest, config: ReductionConfig) -> HarmonicFitConfig:
    source_frequency = manifest.drive_frequency_hz
    if config.harmonic is None:
        base = HarmonicFitConfig(
            source_frequency,
            include_offset=True,
            include_linear_envelope=True,
        )
    else:
        base = config.harmonic
        tolerance = max(1.0e-3, 1.0e-12 * source_frequency)
        if abs(base.frequency_hz - source_frequency) > tolerance:
            raise ValueError(
                f"harmonic frequency {base.frequency_hz:.9g} Hz conflicts with source "
                f"frequency {source_frequency:.9g} Hz"
            )
    if base.reference_time_s is not None:
        return base
    reference = 0.5 * (
        min(float(manifest.electric_time_s[0]), float(manifest.magnetic_h_time_s[0]))
        + max(float(manifest.electric_time_s[-1]), float(manifest.magnetic_h_time_s[-1]))
    )
    return replace(base, reference_time_s=reference)


def _target_points_native_centered(
    r_m: np.ndarray,
    z_m: np.ndarray,
    theta_rad: np.ndarray,
    transform: FrameTransform,
) -> np.ndarray:
    theta = theta_rad[:, None, None]
    radius = r_m[None, None, :]
    longitudinal = z_m[None, :, None]
    shape = (theta_rad.size, z_m.size, r_m.size)
    beam = np.empty(shape + (3,), dtype=np.float64)
    beam[..., 0] = np.broadcast_to(radius * np.cos(theta), shape)
    beam[..., 1] = np.broadcast_to(radius * np.sin(theta), shape)
    beam[..., 2] = np.broadcast_to(longitudinal, shape)
    # The reader uses cathode-centered native coordinates, so only the inverse
    # rotation is required here; adding the native origin would be incorrect.
    return transform.inverse_vectors(beam)


def _expand_slice(index: slice, required_indices: np.ndarray, size: int) -> slice:
    if required_indices.size == 0:
        return index
    start = min(int(index.start), int(np.min(required_indices)))
    stop = max(int(index.stop), int(np.max(required_indices)) + 1)
    return slice(max(0, start), min(size, stop), 1)


def _bounded_slices(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    points: np.ndarray,
    *,
    guard_cells: int,
) -> tuple[slice, slice, slice]:
    flat = points.reshape(-1, 3)
    slices = []
    for dimension, axis in enumerate(axes):
        lower = float(np.min(flat[:, dimension]))
        upper = float(np.max(flat[:, dimension]))
        tolerance = max(2.0e-14, 1.0e-10 * max(abs(axis[0]), abs(axis[-1]), 1.0))
        if lower < axis[0] - tolerance or upper > axis[-1] + tolerance:
            raise ValueError(
                f"target coordinate range [{lower:.9g}, {upper:.9g}] m exceeds component "
                f"support [{axis[0]:.9g}, {axis[-1]:.9g}] m on native axis {dimension}"
            )
        lower = max(lower, float(axis[0]))
        upper = min(upper, float(axis[-1]))
        slices.append(
            coordinate_slice_for_bounds(axis, (lower, upper), guard_cells=guard_cells)
        )
    return tuple(slices)  # type: ignore[return-value]


def _sample_component_time_series(
    reader: XFdtdH5,
    quantity: str,
    component: str,
    points_native_centered_m: np.ndarray,
    surface_mask: np.ndarray,
    config: ReductionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    axes = reader.coordinates(quantity, component, frame="cathode_centered")
    spatial_slices = list(
        _bounded_slices(
            axes,
            points_native_centered_m,
            guard_cells=config.interpolation_guard_cells,
        )
    )
    # Beam z = -native y, so the downstream vacuum side is native y < 0. A node within a small
    # fraction of a cell of native y = 0 lies on the conductor plane itself, not inside the
    # metal: for tangential E and normal H that node carries the exact boundary zero, and
    # excluding it forced a ~99 um extrapolation through strongly curved data that produced a
    # tangential field ON the emission plane larger than the one half a cell downstream.
    _native_y = axes[1]
    _cell = float(np.min(np.abs(np.diff(_native_y)))) if _native_y.size > 1 else 0.0
    _surface_tolerance = float(config.surface_on_plane_cell_fraction) * _cell
    vacuum_y_indices = np.flatnonzero(
        (_native_y < 0.0) | (np.abs(_native_y) <= _surface_tolerance)
    )
    surface_indices = np.flatnonzero(surface_mask.reshape(-1))
    if surface_indices.size:
        if vacuum_y_indices.size < config.surface_sample_count:
            raise ValueError(
                f"{quantity}{component} has only {vacuum_y_indices.size} native-y samples on the "
                f"downstream vacuum side; need {config.surface_sample_count}"
            )
        nearest_order = np.argsort(np.abs(axes[1][vacuum_y_indices]))[: config.surface_sample_count]
        surface_y_indices = vacuum_y_indices[nearest_order]
        spatial_slices[1] = _expand_slice(spatial_slices[1], surface_y_indices, axes[1].size)
    else:
        surface_y_indices = np.empty(0, dtype=np.int64)
    selected_axes = tuple(axis[index] for axis, index in zip(axes, spatial_slices, strict=True))
    points = np.array(points_native_centered_m.reshape(-1, 3), copy=True)
    for dimension, axis in enumerate(axes):
        points[:, dimension] = np.clip(points[:, dimension], axis[0], axis[-1])
    times = reader.times(quantity)
    sampled = np.empty((times.size,) + points_native_centered_m.shape[:-1], dtype=np.float32)
    selection_spatial = tuple(spatial_slices)
    for frame_index in range(times.size):
        selection = H5Selection(frame_index, *selection_spatial)
        block = reader.read_component(quantity, component, selection)
        interpolator = RegularGridInterpolator(
            selected_axes,
            block,
            method="linear",
            bounds_error=True,
        )
        frame_values = np.asarray(interpolator(points), dtype=np.float64)
        if surface_indices.size:
            surface_points = np.array(points[surface_indices], copy=True)
            vacuum_samples = []
            for native_y_index in surface_y_indices:
                surface_points[:, 1] = axes[1][native_y_index]
                vacuum_samples.append(np.asarray(interpolator(surface_points), dtype=np.float64))
            beam_z_samples = -axes[1][surface_y_indices]
            frame_values[surface_indices] = one_sided_surface_limit(
                beam_z_samples,
                np.stack(vacuum_samples, axis=0),
                surface_m=0.0,
                vacuum_side="positive",
                n_points=config.surface_sample_count,
                polynomial_order=config.surface_polynomial_order,
                axis=0,
                surface_tolerance_m=_surface_tolerance,
            )
        sampled[frame_index] = frame_values.reshape(points_native_centered_m.shape[:-1])
    return times, sampled


def _temporal_summary(
    result: HarmonicFitResult,
    z_start: float,
    z_end: float,
    sampled: np.ndarray,
    spatial_weights: np.ndarray,
) -> dict[str, Any]:
    """Summarize one fit and retain additive, physically weighted metrics.

    Per-component relative errors remain useful diagnostics, but a component or
    downstream block whose true signal is almost zero must not dominate the
    production decision merely because its own tiny denominator magnifies
    numerical noise.  The additive sums below allow the completed reduction to
    form separate E- and H-family residual/drift metrics with cylindrical-volume
    weighting while still preserving every local worst case.
    """

    values = np.asarray(sampled)
    weights = np.asarray(spatial_weights, dtype=np.float64)
    if values.ndim < 2 or weights.shape != values.shape[1:]:
        raise ValueError("sampled values and temporal spatial_weights have incompatible shapes")
    mean = np.mean(values, axis=0, dtype=np.float64)
    signal_sumsq = 0.0
    for frame in values:
        difference = np.asarray(frame, dtype=np.float64) - mean
        signal_sumsq += float(np.sum(weights * difference * difference, dtype=np.float64))
    residual_sumsq = float(
        result.sample_count
        * np.sum(weights * np.asarray(result.residual_rms, dtype=np.float64) ** 2)
    )
    phasor_sumsq = float(
        np.sum(weights * np.abs(np.asarray(result.phasor)) ** 2, dtype=np.float64)
    )
    slope_change = np.asarray(result.phasor_slope_per_s) / float(result.frequency_hz)
    slope_change_sumsq = float(
        np.sum(weights * np.abs(slope_change) ** 2, dtype=np.float64)
    )
    # ||P'||/||P|| is unsigned: phase rotation and decay also increase it.
    # The complex projection separates amplitude growth from phase drift and
    # lets us check whether one common envelope describes the spatial field.
    overlap = np.sum(weights * np.conj(result.phasor) * slope_change)
    return {
        "z_start_m": z_start,
        "z_end_m": z_end,
        "sample_count": result.sample_count,
        "global_nrmse": result.global_nrmse,
        "active_nrmse_p95": result.active_nrmse_p95,
        "peak_fractional_drift_per_cycle": result.peak_fractional_drift_per_cycle,
        "cylindrical_weighted_signal_sumsq": signal_sumsq,
        "cylindrical_weighted_residual_sumsq": residual_sumsq,
        "cylindrical_weighted_phasor_sumsq": phasor_sumsq,
        "cylindrical_weighted_slope_change_per_cycle_sumsq": slope_change_sumsq,
        "cylindrical_weighted_phasor_slope_overlap_real": float(overlap.real),
        "cylindrical_weighted_phasor_slope_overlap_imag": float(overlap.imag),
        "cylindrical_spatial_weight_sum_m": float(np.sum(weights, dtype=np.float64)),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(np.sqrt(numerator / denominator)) if denominator > 0.0 else 0.0


def _interior_metric_mask(vacuum_mask: np.ndarray, wall_guard_cells: int) -> np.ndarray:
    if wall_guard_cells == 0:
        return np.array(vacuum_mask, copy=True)
    interior = binary_erosion(
        vacuum_mask,
        iterations=wall_guard_cells,
        border_value=1,
    )
    if not np.any(interior):
        raise ValueError(
            f"wall_guard_cells={wall_guard_cells} removes the complete vacuum ROI; "
            "use a finer mask or a smaller guard"
        )
    return np.asarray(interior, dtype=bool)


def reduce_xfdtd_to_axisymmetric(
    source_path: str | Path,
    config: ReductionConfig,
    *,
    progress: ProgressCallback | None = None,
) -> ReductionResult:
    """Reduce the six staggered XFdtd components using bounded longitudinal slabs.

    Each raw time frame is interpolated separately onto common cylindrical sample
    points.  The all-frame phasor fit is then performed at those points, so no
    component shares another component's Yee coordinates or timestamps.
    """

    output_dtype = np.dtype(config.output_dtype)
    names = ("Er", "Etheta", "Ez", "Br", "Btheta", "Bz")
    electric_names = ("Er", "Etheta", "Ez")
    magnetic_names = ("Br", "Btheta", "Bz")
    component_total = {name: 0.0 for name in names}
    component_nonaxis = {name: 0.0 for name in names}
    core_component_total = {name: 0.0 for name in names}
    core_component_nonaxis = {name: 0.0 for name in names}
    mode_power = {
        name: {mode: 0.0 for mode in range(-config.m_max, config.m_max + 1)} for name in names
    }
    temporal_blocks: dict[str, list[dict[str, Any]]] = {
        f"{quantity}{component}": [] for quantity in ("E", "H") for component in "xyz"
    }
    output = {
        "Er": np.empty((config.z_m.size, config.r_m.size), dtype=output_dtype),
        "Ez": np.empty((config.z_m.size, config.r_m.size), dtype=output_dtype),
        "Btheta": np.empty((config.z_m.size, config.r_m.size), dtype=output_dtype),
        "Bz": np.empty((config.z_m.size, config.r_m.size), dtype=output_dtype),
    }
    omitted_m0_power = {"Etheta": 0.0, "Br": 0.0}
    metric_mask = _interior_metric_mask(config.vacuum_mask, config.wall_guard_cells)
    with XFdtdH5(source_path, max_logical_read_bytes=config.max_logical_read_bytes) as reader:
        manifest = reader.manifest()
        if not np.allclose(
            config.transform.origin_native_m,
            manifest.cathode_origin_native_m,
            rtol=0.0,
            atol=2.0e-9,
        ):
            raise ValueError(
                "configured native origin does not match the cathode-centered coordinate groups: "
                f"{config.transform.origin_native_m} versus {manifest.cathode_origin_native_m}"
            )
        harmonic = _shared_harmonic_config(manifest, config)
        _notify(progress, stage="start", source=str(Path(source_path)), blocks=int(np.ceil(config.z_m.size / config.longitudinal_block_points)))
        for block_number, start in enumerate(
            range(0, config.z_m.size, config.longitudinal_block_points)
        ):
            stop = min(config.z_m.size, start + config.longitudinal_block_points)
            z_block = config.z_m[start:stop]
            points_native = _target_points_native_centered(
                config.r_m, z_block, config.theta_rad, config.transform
            )
            surface_mask = np.broadcast_to(
                np.isclose(z_block[None, :, None], 0.0, rtol=0.0, atol=1.0e-15),
                points_native.shape[:-1],
            )
            block_vacuum_mask = config.vacuum_mask[start:stop]
            block_metric_mask = metric_mask[start:stop]
            theta_metric_mask = np.broadcast_to(
                block_metric_mask[None, :, :], points_native.shape[:-1]
            )
            theta_metric_weights = np.broadcast_to(
                config.r_m[None, None, :], points_native.shape[:-1]
            ) * theta_metric_mask
            native_phasors: dict[str, np.ndarray] = {}
            for quantity in ("E", "H"):
                for component in "xyz":
                    key = f"{quantity}{component}"
                    _notify(
                        progress,
                        stage="component",
                        block=block_number,
                        z_start_m=float(z_block[0]),
                        z_end_m=float(z_block[-1]),
                        component=key,
                    )
                    times, sampled = _sample_component_time_series(
                        reader,
                        quantity,
                        component,
                        points_native,
                        surface_mask,
                        config,
                    )
                    fit = fit_fixed_frequency_phasor(
                        times,
                        sampled,
                        harmonic,
                        axis=0,
                        metric_mask=theta_metric_mask,
                    )
                    native_phasors[key] = fit.phasor
                    temporal_blocks[key].append(
                        _temporal_summary(
                            fit,
                            float(z_block[0]),
                            float(z_block[-1]),
                            sampled,
                            theta_metric_weights,
                        )
                    )
            electric_native = np.stack(
                [native_phasors[f"E{component}"] for component in "xyz"], axis=-1
            )
            magnetic_h_native = np.stack(
                [native_phasors[f"H{component}"] for component in "xyz"], axis=-1
            )
            vector_vacuum_mask = np.broadcast_to(
                block_vacuum_mask[None, :, :, None], electric_native.shape
            )
            electric_native = np.where(vector_vacuum_mask, electric_native, 0.0)
            magnetic_h_native = np.where(vector_vacuum_mask, magnetic_h_native, 0.0)
            electric_beam = config.transform.transform_vectors(electric_native)
            magnetic_b_beam = MU0_HPM * config.transform.transform_vectors(magnetic_h_native)
            cylindrical = cylindrical_samples_from_cartesian(
                electric_beam,
                magnetic_b_beam,
                config.theta_rad,
                theta_axis=0,
            )
            volume_weights = (
                np.broadcast_to(config.r_m[None, :], (z_block.size, config.r_m.size))
                * block_metric_mask
            )
            projection = project_cylindrical_m0(
                cylindrical,
                config.theta_rad,
                m_max=config.m_max,
                weights=volume_weights,
            )
            if config.beam_core_radius_m is not None:
                core_weights = volume_weights * (
                    config.r_m[None, :] <= float(config.beam_core_radius_m)
                )
                core_projection = project_cylindrical_m0(
                    cylindrical,
                    config.theta_rad,
                    m_max=config.m_max,
                    weights=core_weights,
                )
                for name in names:
                    core_component_total[name] += core_projection.component_total_power[name]
                    core_component_nonaxis[name] += (
                        core_projection.component_nonaxisymmetric_power[name]
                    )
            output["Er"][start:stop] = projection.Er
            output["Ez"][start:stop] = projection.Ez
            output["Btheta"][start:stop] = projection.Btheta
            output["Bz"][start:stop] = projection.Bz
            for name in names:
                component_total[name] += projection.component_total_power[name]
                component_nonaxis[name] += projection.component_nonaxisymmetric_power[name]
                for mode, power in projection.mode_power[name].items():
                    mode_power[name][mode] += power
            omitted_m0_power["Etheta"] += (
                config.theta_rad.size * projection.mode_power["Etheta"].get(0, 0.0)
            )
            omitted_m0_power["Br"] += config.theta_rad.size * projection.mode_power["Br"].get(0, 0.0)
            _notify(progress, stage="block_complete", block=block_number, z_stop_index=stop)

    # Er and Btheta must vanish on axis for a genuine m=0 field. They are forced to exactly
    # zero below so RF-Track receives a clean axis, but that forcing makes any downstream
    # "is the axis zero?" gate tautological. Record how far the *projected* values were from
    # zero before the forcing -- that residual is the real measure of projection consistency.
    _axis_e_scale = float(np.sqrt(np.mean(np.abs(output["Ez"][config.vacuum_mask]) ** 2))) \
        if np.any(config.vacuum_mask) else 0.0
    _axis_b_scale = float(np.sqrt(np.mean(np.abs(output["Btheta"][config.vacuum_mask]) ** 2))) \
        if np.any(config.vacuum_mask) else 0.0
    _axis_forcing = {}
    for _name, _scale in (("Er", _axis_e_scale), ("Btheta", _axis_b_scale)):
        _pre = np.asarray(output[_name][:, 0])
        _rms = float(np.sqrt(np.mean(np.abs(_pre) ** 2))) if _pre.size else 0.0
        _axis_forcing[_name] = {
            "pre_forcing_axis_rms": _rms,
            "pre_forcing_axis_max_abs": float(np.max(np.abs(_pre))) if _pre.size else 0.0,
            "pre_forcing_fraction_of_combined_field_rms": (_rms / _scale) if _scale > 0.0 else 0.0,
            "forced_to_zero": True,
        }

    output["Er"][:, 0] = 0.0
    output["Btheta"][:, 0] = 0.0
    for values in output.values():
        values[~config.vacuum_mask] = 0.0
    total_e = sum(component_total[name] for name in electric_names)
    total_b = sum(component_total[name] for name in magnetic_names)
    symmetry = {
        "axis_forcing": _axis_forcing,
        "production_representation": "cylindrical_vector_m0",
        "policy": (
            "nonaxisymmetric content is reported but does not veto the user-requested m0 "
            "RF-Track representation"
        ),
        "theta_count": int(config.theta_rad.size),
        "m_max": config.m_max,
        "component_nonaxisymmetric_fraction": {
            name: _ratio(component_nonaxis[name], component_total[name]) for name in names
        },
        "combined_e_nonaxisymmetric_fraction": _ratio(
            sum(component_nonaxis[name] for name in electric_names), total_e
        ),
        "combined_b_nonaxisymmetric_fraction": _ratio(
            sum(component_nonaxis[name] for name in magnetic_names), total_b
        ),
        "omitted_m0_Etheta_fraction_of_total_e": _ratio(omitted_m0_power["Etheta"], total_e),
        "omitted_m0_Br_fraction_of_total_b": _ratio(omitted_m0_power["Br"], total_b),
        "mode_power_fraction": {
            name: {
                str(mode): (
                    power / sum(mode_power[name].values())
                    if sum(mode_power[name].values()) > 0.0
                    else 0.0
                )
                for mode, power in mode_power[name].items()
            }
            for name in names
        },
    }
    if config.beam_core_radius_m is not None:
        core_total_e = sum(core_component_total[name] for name in electric_names)
        core_total_b = sum(core_component_total[name] for name in magnetic_names)
        symmetry["beam_core"] = {
            "radius_m": float(config.beam_core_radius_m),
            "component_nonaxisymmetric_fraction": {
                name: _ratio(core_component_nonaxis[name], core_component_total[name])
                for name in names
            },
            "combined_e_nonaxisymmetric_fraction": _ratio(
                sum(core_component_nonaxis[name] for name in electric_names), core_total_e
            ),
            "combined_b_nonaxisymmetric_fraction": _ratio(
                sum(core_component_nonaxis[name] for name in magnetic_names), core_total_b
            ),
        }
    significance_threshold = 1.0e-4
    family_aggregate: dict[str, dict[str, Any]] = {}
    for family, prefixes in (("E", ("E",)), ("H", ("H",))):
        entries = [
            item
            for component, blocks in temporal_blocks.items()
            if component.startswith(prefixes)
            for item in blocks
        ]
        signal_total = float(
            sum(item["cylindrical_weighted_signal_sumsq"] for item in entries)
        )
        residual_total = float(
            sum(item["cylindrical_weighted_residual_sumsq"] for item in entries)
        )
        phasor_total = float(
            sum(item["cylindrical_weighted_phasor_sumsq"] for item in entries)
        )
        slope_total = float(
            sum(
                item["cylindrical_weighted_slope_change_per_cycle_sumsq"]
                for item in entries
            )
        )
        combined_nrmse = (
            float(np.sqrt(residual_total / signal_total))
            if signal_total > 0.0
            else (0.0 if residual_total == 0.0 else float("inf"))
        )
        combined_drift = (
            float(np.sqrt(slope_total / phasor_total))
            if phasor_total > 0.0
            else (0.0 if slope_total == 0.0 else float("inf"))
        )
        family_aggregate[family] = {
            "cylindrical_weighting": "r (constant dtheta*dr*dz factors cancel)",
            "signal_sumsq": signal_total,
            "residual_sumsq": residual_total,
            "phasor_sumsq": phasor_total,
            "slope_change_per_cycle_sumsq": slope_total,
            "combined_nrmse": combined_nrmse,
            "combined_fractional_drift_per_cycle": combined_drift,
        }
        overlap = sum(
            complex(
                item["cylindrical_weighted_phasor_slope_overlap_real"],
                item["cylindrical_weighted_phasor_slope_overlap_imag"],
            )
            for item in entries
        )
        gamma = overlap / phasor_total if phasor_total > 0.0 else 0j
        family_aggregate[family].update({
            "signed_amplitude_drift_per_cycle": float(gamma.real),
            "phase_drift_rad_per_cycle": float(gamma.imag),
            "nonseparable_drift_per_cycle": float(np.sqrt(max(
                0.0, combined_drift**2 - abs(gamma)**2
            ))),
        })
        for item in entries:
            item["signal_fraction_of_family"] = (
                float(item["cylindrical_weighted_signal_sumsq"] / signal_total)
                if signal_total > 0.0
                else 0.0
            )
            item["production_significant"] = bool(
                item["signal_fraction_of_family"] >= significance_threshold
            )

    all_entries = [item for blocks in temporal_blocks.values() for item in blocks]
    significant_entries = [item for item in all_entries if item["production_significant"]]
    family_block_aggregate: dict[str, list[dict[str, Any]]] = {}
    significant_family_blocks: list[dict[str, Any]] = []
    for family in ("E", "H"):
        component_blocks = [temporal_blocks[f"{family}{component}"] for component in "xyz"]
        block_count = len(component_blocks[0])
        if any(len(blocks) != block_count for blocks in component_blocks):
            raise RuntimeError(f"{family} temporal component block counts do not match")
        signal_total = float(family_aggregate[family]["signal_sumsq"])
        blocks_out: list[dict[str, Any]] = []
        for block_index in range(block_count):
            entries = [blocks[block_index] for blocks in component_blocks]
            signal = float(
                sum(item["cylindrical_weighted_signal_sumsq"] for item in entries)
            )
            residual = float(
                sum(item["cylindrical_weighted_residual_sumsq"] for item in entries)
            )
            phasor = float(
                sum(item["cylindrical_weighted_phasor_sumsq"] for item in entries)
            )
            slope = float(
                sum(
                    item["cylindrical_weighted_slope_change_per_cycle_sumsq"]
                    for item in entries
                )
            )
            signal_fraction = signal / signal_total if signal_total > 0.0 else 0.0
            block = {
                "z_start_m": float(entries[0]["z_start_m"]),
                "z_end_m": float(entries[0]["z_end_m"]),
                "signal_sumsq": signal,
                "residual_sumsq": residual,
                "phasor_sumsq": phasor,
                "slope_change_per_cycle_sumsq": slope,
                "combined_nrmse": _ratio(residual, signal),
                "combined_fractional_drift_per_cycle": _ratio(slope, phasor),
                "signal_fraction_of_family": float(signal_fraction),
                "production_significant": bool(signal_fraction >= significance_threshold),
            }
            blocks_out.append(block)
            if block["production_significant"]:
                significant_family_blocks.append(block)
        family_block_aggregate[family] = blocks_out
    temporal = {
        "frequency_hz": harmonic.frequency_hz,
        "reference_time_s": harmonic.reference_time_s,
        "include_offset": harmonic.include_offset,
        "include_linear_envelope": harmonic.include_linear_envelope,
        "components": temporal_blocks,
        "family_aggregate": family_aggregate,
        "family_block_aggregate": family_block_aggregate,
        "production_significance_min_signal_fraction_of_family": significance_threshold,
        "production_metrics": {
            "max_family_combined_nrmse": max(
                family_aggregate[family]["combined_nrmse"] for family in ("E", "H")
            ),
            "max_significant_component_active_nrmse_p95_diagnostic": max(
                (item["active_nrmse_p95"] for item in significant_entries), default=0.0
            ),
            "max_significant_family_block_nrmse": max(
                (item["combined_nrmse"] for item in significant_family_blocks), default=0.0
            ),
            "max_family_combined_fractional_drift_per_cycle": max(
                family_aggregate[family]["combined_fractional_drift_per_cycle"]
                for family in ("E", "H")
            ),
            "significant_component_block_count": int(len(significant_entries)),
            "excluded_negligible_component_block_count": int(
                len(all_entries) - len(significant_entries)
            ),
            "significant_family_block_count": int(len(significant_family_blocks)),
        },
        # Preserve unfiltered extrema as visible diagnostics.  They deliberately
        # do not drive production qualification because locally normalized noise
        # in a vanishing component/tail block can be arbitrarily large.
        "max_global_nrmse": max(
            item["global_nrmse"] for item in all_entries
        ),
        "max_active_nrmse_p95": max(
            item["active_nrmse_p95"] for item in all_entries
        ),
        "max_peak_fractional_drift_per_cycle": max(
            item["peak_fractional_drift_per_cycle"] for item in all_entries
        ),
    }
    support = {
        "measured_z_min_m": float(config.z_m[0]),
        "measured_z_max_m": float(config.z_m[-1]),
        "r_max_m": float(config.r_m[-1]),
        "extension": "none",
        "vacuum_cell_fraction": float(np.mean(config.vacuum_mask)),
        "metric_cell_fraction_after_wall_guard": float(np.mean(metric_mask)),
        "wall_guard_cells": config.wall_guard_cells,
        "h_to_b_conversion": "mu0_times_H_in_vacuum_mask_only",
    }
    source = {
        "path": str(Path(source_path).resolve()),
        "file_size_bytes": manifest.file_size_bytes,
        "schema_version": manifest.schema_version,
        "solver": manifest.root_attributes.get("solver"),
        "source_run_id": manifest.root_attributes.get("source_run_id"),
        "frequency_hz": manifest.drive_frequency_hz,
        "source_power_w": manifest.source_power_w,
        "cathode_origin_native_m": manifest.cathode_origin_native_m.tolist(),
    }
    surface = {
        "method": "one_sided_vacuum_polynomial_limit",
        "vacuum_direction": "positive_beam_z_negative_native_y",
        "sample_count": config.surface_sample_count,
        "polynomial_order": config.surface_polynomial_order,
        "applied": bool(np.any(np.isclose(config.z_m, 0.0, atol=1e-15, rtol=0.0))),
    }
    report = ReductionReport(
        source=MappingProxyType(source),
        temporal=MappingProxyType(temporal),
        symmetry=MappingProxyType(symmetry),
        surface=MappingProxyType(surface),
        support=MappingProxyType(support),
    )
    metadata = {
        "representation": "cylindrical_vector_m0",
        "phasor_convention": "Re(F*exp(+i*2*pi*f*(t-t_ref)))",
        "reference_time_s": harmonic.reference_time_s,
        "source_power_w": manifest.source_power_w,
        "normalization": "absolute_source_fields_no_component_normalization",
        "reduction_config": config.as_metadata(),
        "quality_summary": {
            "temporal": {
                "max_global_nrmse": temporal["max_global_nrmse"],
                "max_active_nrmse_p95": temporal["max_active_nrmse_p95"],
                "max_peak_fractional_drift_per_cycle": temporal[
                    "max_peak_fractional_drift_per_cycle"
                ],
            },
            "symmetry": symmetry,
        },
        "spatial_support": support,
    }
    axisymmetric = AxisymmetricRFField(
        frequency_hz=manifest.drive_frequency_hz,
        r_m=config.r_m,
        z_m=config.z_m,
        Er_Vpm=output["Er"],
        Ez_Vpm=output["Ez"],
        Btheta_T=output["Btheta"],
        Bz_T=output["Bz"],
        metadata=metadata,
    )
    _notify(progress, stage="complete", symmetry=symmetry, temporal=temporal)
    return ReductionResult(axisymmetric, report)


def reduce_xfdtd_to_artifact(
    source_path: str | Path,
    artifact_path: str | Path,
    config: ReductionConfig,
    *,
    quality: Mapping[str, Any],
    provenance: Mapping[str, Any],
    progress: ProgressCallback | None = None,
    preview: Mapping[str, np.ndarray] | None = None,
    vacuum_mask: np.ndarray | None = None,
    aperture_radius_m: np.ndarray | None = None,
    overwrite: bool = False,
) -> ReductionArtifactResult:
    """Run the bounded reduction and atomically serialize its qualified result."""

    reduction = reduce_xfdtd_to_axisymmetric(source_path, config, progress=progress)
    quality_record = dict(quality)
    quality_record["field_reduction"] = reduction.report.as_dict()
    provenance_record = dict(provenance)
    actual_config = config.as_metadata()
    if "processing_config" in provenance_record and provenance_record["processing_config"] != actual_config:
        raise ValueError("provenance processing_config differs from the reduction configuration used")
    provenance_record["processing_config"] = actual_config
    provenance_record.setdefault("source_path", str(Path(source_path).resolve()))
    provenance_record.setdefault("source_file_size_bytes", Path(source_path).stat().st_size)
    if vacuum_mask is not None and not np.array_equal(np.asarray(vacuum_mask, dtype=bool), config.vacuum_mask):
        raise ValueError("artifact vacuum_mask differs from the mask used during field reduction")
    payload = write_axisymmetric_artifact(
        artifact_path,
        reduction.field,
        transform=config.transform,
        quality=quality_record,
        provenance=provenance_record,
        preview=preview,
        vacuum_mask=config.vacuum_mask,
        aperture_radius_m=aperture_radius_m,
        overwrite=overwrite,
    )
    return ReductionArtifactResult(reduction, payload, Path(artifact_path).resolve())
