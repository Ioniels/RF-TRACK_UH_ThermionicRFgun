"""Azimuthal-mode analysis and explicit cylindrical ``m=0`` projection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from .coordinates import cartesian_to_cylindrical_vectors


@dataclass(frozen=True)
class AzimuthalModeResult:
    mode_numbers: np.ndarray
    coefficients: np.ndarray
    theta_axis: int
    sample_count: int

    def coefficient(self, mode: int) -> np.ndarray:
        matches = np.flatnonzero(self.mode_numbers == int(mode))
        if matches.size != 1:
            raise KeyError(f"mode {mode} was not fitted")
        return self.coefficients[int(matches[0])]


def azimuthal_fourier_modes(
    samples: np.ndarray,
    theta_rad: np.ndarray,
    *,
    m_max: int = 4,
    axis: int = 0,
) -> AzimuthalModeResult:
    """Least-squares Fourier coefficients on uniform or non-uniform angular samples."""

    values = np.asarray(samples)
    theta = np.asarray(theta_rad, dtype=np.float64)
    theta_axis = normalize_axis_index(axis, values.ndim)
    if theta.ndim != 1 or theta.size != values.shape[theta_axis]:
        raise ValueError("theta_rad must be 1-D and match samples along axis")
    if not np.all(np.isfinite(theta)) or np.unique(np.mod(theta, 2.0 * np.pi)).size != theta.size:
        raise ValueError("theta_rad must contain finite, distinct azimuths")
    if m_max < 0:
        raise ValueError("m_max must be non-negative")
    mode_numbers = np.arange(-int(m_max), int(m_max) + 1, dtype=np.int64)
    if theta.size < mode_numbers.size:
        raise ValueError(
            f"{theta.size} angular samples cannot constrain {mode_numbers.size} Fourier modes"
        )
    design = np.exp(1j * theta[:, None] * mode_numbers[None, :])
    moved = np.moveaxis(values, theta_axis, 0)
    flat = moved.reshape(theta.size, -1)
    coefficients, _, rank, _ = np.linalg.lstsq(design, flat, rcond=None)
    if rank != mode_numbers.size:
        raise ValueError("azimuthal Fourier design is rank deficient")
    shaped = coefficients.reshape((mode_numbers.size,) + moved.shape[1:])
    return AzimuthalModeResult(mode_numbers, shaped, theta_axis, int(theta.size))


def project_axisymmetric_m0(
    samples: np.ndarray,
    theta_rad: np.ndarray,
    *,
    axis: int = 0,
    fit_m_max: int = 0,
) -> np.ndarray:
    """Return the cylindrical-vector m=0 coefficient."""

    return azimuthal_fourier_modes(samples, theta_rad, m_max=fit_m_max, axis=axis).coefficient(0)


def _weighted_power(values: np.ndarray, weights: np.ndarray | None) -> float:
    magnitude2 = np.abs(values) ** 2
    if weights is None:
        return float(np.sum(magnitude2))
    weight = np.asarray(weights, dtype=np.float64)
    try:
        return float(np.sum(magnitude2 * np.broadcast_to(weight, magnitude2.shape)))
    except ValueError as exc:
        raise ValueError(f"weights shape {weight.shape} is not broadcastable to {magnitude2.shape}") from exc


def weighted_nonaxisymmetric_fraction(
    samples: np.ndarray,
    m0: np.ndarray,
    *,
    axis: int = 0,
    weights: np.ndarray | None = None,
) -> float:
    """RMS amplitude fraction left after removing an m=0 projection."""

    values = np.moveaxis(np.asarray(samples), axis, 0)
    projected = np.asarray(m0)
    if values.shape[1:] != projected.shape:
        raise ValueError(f"m0 shape {projected.shape} does not match sample shape {values.shape[1:]}")
    residual = values - projected[None, ...]
    if weights is None:
        expanded_weights = None
    else:
        spatial_weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), projected.shape)
        expanded_weights = np.broadcast_to(spatial_weights[None, ...], values.shape)
    denominator = _weighted_power(values, expanded_weights)
    numerator = _weighted_power(residual, expanded_weights)
    return float(np.sqrt(numerator / denominator)) if denominator > 0.0 else 0.0


@dataclass(frozen=True)
class CylindricalSamples:
    Er: np.ndarray
    Etheta: np.ndarray
    Ez: np.ndarray
    Br: np.ndarray
    Btheta: np.ndarray
    Bz: np.ndarray
    theta_axis: int = 0

    def __post_init__(self) -> None:
        arrays = [np.asarray(getattr(self, name)) for name in ("Er", "Etheta", "Ez", "Br", "Btheta", "Bz")]
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError("all cylindrical components must have the same shape")
        if arrays[0].ndim < 1:
            raise ValueError("cylindrical sample arrays must have an azimuthal dimension")
        axis = normalize_axis_index(self.theta_axis, arrays[0].ndim)
        for name, array in zip(("Er", "Etheta", "Ez", "Br", "Btheta", "Bz"), arrays, strict=True):
            object.__setattr__(self, name, array)
        object.__setattr__(self, "theta_axis", axis)


def cylindrical_samples_from_cartesian(
    electric_xyz: np.ndarray,
    magnetic_b_xyz: np.ndarray,
    theta_rad: np.ndarray,
    *,
    theta_axis: int = 0,
) -> CylindricalSamples:
    """Rotate vectors before averaging, preserving radial/azimuthal parity."""

    electric = cartesian_to_cylindrical_vectors(electric_xyz, theta_rad, theta_axis=theta_axis)
    magnetic = cartesian_to_cylindrical_vectors(magnetic_b_xyz, theta_rad, theta_axis=theta_axis)
    return CylindricalSamples(
        Er=electric[..., 0],
        Etheta=electric[..., 1],
        Ez=electric[..., 2],
        Br=magnetic[..., 0],
        Btheta=magnetic[..., 1],
        Bz=magnetic[..., 2],
        theta_axis=theta_axis,
    )


@dataclass(frozen=True)
class AxisymmetricProjection:
    Er: np.ndarray
    Etheta: np.ndarray
    Ez: np.ndarray
    Br: np.ndarray
    Btheta: np.ndarray
    Bz: np.ndarray
    component_nonaxisymmetric_fraction: Mapping[str, float]
    combined_e_nonaxisymmetric_fraction: float
    combined_b_nonaxisymmetric_fraction: float
    component_total_power: Mapping[str, float]
    component_nonaxisymmetric_power: Mapping[str, float]
    mode_power: Mapping[str, Mapping[int, float]]
    mode_power_fraction: Mapping[str, Mapping[int, float]]


def _combined_fraction(
    samples: CylindricalSamples,
    projected: Mapping[str, np.ndarray],
    names: tuple[str, ...],
    weights: np.ndarray | None,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for name in names:
        values = np.moveaxis(np.asarray(getattr(samples, name)), samples.theta_axis, 0)
        residual = values - projected[name][None, ...]
        expanded = None
        if weights is not None:
            spatial = np.broadcast_to(np.asarray(weights, dtype=np.float64), projected[name].shape)
            expanded = np.broadcast_to(spatial[None, ...], values.shape)
        numerator += _weighted_power(residual, expanded)
        denominator += _weighted_power(values, expanded)
    return float(np.sqrt(numerator / denominator)) if denominator > 0.0 else 0.0


def project_cylindrical_m0(
    samples: CylindricalSamples,
    theta_rad: np.ndarray,
    *,
    m_max: int = 4,
    weights: np.ndarray | None = None,
) -> AxisymmetricProjection:
    """Project all cylindrical vector components and retain auditable asymmetry metrics."""

    names = ("Er", "Etheta", "Ez", "Br", "Btheta", "Bz")
    modes = {
        name: azimuthal_fourier_modes(
            getattr(samples, name), theta_rad, m_max=m_max, axis=samples.theta_axis
        )
        for name in names
    }
    projected = {name: result.coefficient(0) for name, result in modes.items()}
    component_fraction = {
        name: weighted_nonaxisymmetric_fraction(
            getattr(samples, name), projected[name], axis=samples.theta_axis, weights=weights
        )
        for name in names
    }
    component_total_power: dict[str, float] = {}
    component_nonaxis_power: dict[str, float] = {}
    for name in names:
        values = np.moveaxis(np.asarray(getattr(samples, name)), samples.theta_axis, 0)
        residual = values - projected[name][None, ...]
        expanded = None
        if weights is not None:
            spatial = np.broadcast_to(np.asarray(weights, dtype=np.float64), projected[name].shape)
            expanded = np.broadcast_to(spatial[None, ...], values.shape)
        component_total_power[name] = _weighted_power(values, expanded)
        component_nonaxis_power[name] = _weighted_power(residual, expanded)
    spectra: dict[str, Mapping[int, float]] = {}
    raw_mode_power: dict[str, Mapping[int, float]] = {}
    for name, result in modes.items():
        powers = np.array([_weighted_power(coefficient, weights) for coefficient in result.coefficients])
        total = float(np.sum(powers))
        raw_mode_power[name] = MappingProxyType(
            {
                int(mode): float(power)
                for mode, power in zip(result.mode_numbers, powers, strict=True)
            }
        )
        spectra[name] = MappingProxyType(
            {
                int(mode): (float(power / total) if total > 0.0 else 0.0)
                for mode, power in zip(result.mode_numbers, powers, strict=True)
            }
        )
    return AxisymmetricProjection(
        **projected,
        component_nonaxisymmetric_fraction=MappingProxyType(component_fraction),
        combined_e_nonaxisymmetric_fraction=_combined_fraction(
            samples, projected, ("Er", "Etheta", "Ez"), weights
        ),
        combined_b_nonaxisymmetric_fraction=_combined_fraction(
            samples, projected, ("Br", "Btheta", "Bz"), weights
        ),
        component_total_power=MappingProxyType(component_total_power),
        component_nonaxisymmetric_power=MappingProxyType(component_nonaxis_power),
        mode_power=MappingProxyType(raw_mode_power),
        mode_power_fraction=MappingProxyType(spectra),
    )


def enforce_axis_regularity(
    projection: AxisymmetricProjection,
    *,
    radial_axis: int,
) -> AxisymmetricProjection:
    """Set the mathematically odd ``Er`` and ``Btheta`` components to zero at r=0."""

    er = np.array(projection.Er, copy=True)
    btheta = np.array(projection.Btheta, copy=True)
    index = [slice(None)] * er.ndim
    index[normalize_axis_index(radial_axis, er.ndim)] = 0
    er[tuple(index)] = 0.0
    btheta[tuple(index)] = 0.0
    return replace(projection, Er=er, Btheta=btheta)
