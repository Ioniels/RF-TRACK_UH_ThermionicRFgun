"""Coordinate, vector, and cathode-boundary utilities for XFdtd maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

MU0_HPM = 4.0e-7 * np.pi


@dataclass(frozen=True)
class FrameTransform:
    """Proper rigid transform from a native solver frame to the beam frame."""

    origin_native_m: np.ndarray
    R_target_from_native: np.ndarray

    def __post_init__(self) -> None:
        origin = np.array(self.origin_native_m, dtype=np.float64, copy=True)
        rotation = np.array(self.R_target_from_native, dtype=np.float64, copy=True)
        if origin.shape != (3,):
            raise ValueError(f"origin_native_m must have shape (3,), got {origin.shape}")
        if rotation.shape != (3, 3):
            raise ValueError(f"R_target_from_native must have shape (3, 3), got {rotation.shape}")
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(rotation)):
            raise ValueError("frame transform contains non-finite values")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("R_target_from_native must be orthonormal")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(determinant, 1.0, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError(f"R_target_from_native must be right-handed (det=+1), got {determinant}")
        origin.setflags(write=False)
        rotation.setflags(write=False)
        object.__setattr__(self, "origin_native_m", origin)
        object.__setattr__(self, "R_target_from_native", rotation)

    def transform_points(self, points_native_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_native_m, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError("points_native_m must have final dimension 3")
        return np.einsum(
            "ij,...j->...i", self.R_target_from_native, points - self.origin_native_m
        )

    def inverse_points(self, points_target_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_target_m, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError("points_target_m must have final dimension 3")
        return np.einsum("ji,...j->...i", self.R_target_from_native, points) + self.origin_native_m

    def transform_vectors(self, vectors_native: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors_native)
        if vectors.shape[-1:] != (3,):
            raise ValueError("vectors_native must have final dimension 3")
        return np.einsum("ij,...j->...i", self.R_target_from_native, vectors)

    def inverse_vectors(self, vectors_target: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors_target)
        if vectors.shape[-1:] != (3,):
            raise ValueError("vectors_target must have final dimension 3")
        return np.einsum("ji,...j->...i", self.R_target_from_native, vectors)


XFD_TD_CATHODE_TO_BEAM = FrameTransform(
    origin_native_m=np.array([0.0, 0.012998784322142597, 0.3460047357082367]),
    R_target_from_native=np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    ),
)
# Spelling-compatible alias for callers that write the solver name as XDFDTD.
XDFDTD_CATHODE_TO_BEAM = XFD_TD_CATHODE_TO_BEAM


def magnetic_flux_density_from_h(H_Apm: np.ndarray, *, relative_permeability: float = 1.0) -> np.ndarray:
    """Convert magnetic-field intensity to B in a region of known permeability.

    The XFdtd artifact stores H, not B.  The pipeline should call this only in a
    validated material region; the default is vacuum.
    """

    mu_r = float(relative_permeability)
    if not np.isfinite(mu_r) or mu_r <= 0.0:
        raise ValueError("relative_permeability must be finite and positive")
    return np.asarray(H_Apm) * (MU0_HPM * mu_r)


def cartesian_to_cylindrical_vectors(
    vectors_xyz: np.ndarray,
    theta_rad: np.ndarray,
    *,
    theta_axis: int = 0,
) -> np.ndarray:
    """Transform Cartesian vector samples to ``(radial, azimuthal, longitudinal)``.

    ``theta_rad`` identifies the azimuth of the samples along ``theta_axis``.
    The final array dimension must contain the three Cartesian components.
    """

    vectors = np.asarray(vectors_xyz)
    if vectors.shape[-1:] != (3,):
        raise ValueError("vectors_xyz must have final dimension 3")
    axis = normalize_axis_index(theta_axis, vectors.ndim - 1)
    theta = np.asarray(theta_rad, dtype=np.float64)
    if theta.ndim != 1 or vectors.shape[axis] != theta.size:
        raise ValueError("theta_rad must be 1-D and match vectors along theta_axis")
    moved = np.moveaxis(vectors, axis, 0)
    expand = (theta.size,) + (1,) * (moved.ndim - 2)
    cosine = np.cos(theta).reshape(expand)
    sine = np.sin(theta).reshape(expand)
    radial = moved[..., 0] * cosine + moved[..., 1] * sine
    azimuthal = -moved[..., 0] * sine + moved[..., 1] * cosine
    cylindrical = np.stack((radial, azimuthal, moved[..., 2]), axis=-1)
    return np.moveaxis(cylindrical, 0, axis)


def one_sided_surface_limit(
    coordinate_m: np.ndarray,
    values: np.ndarray,
    *,
    surface_m: float = 0.0,
    vacuum_side: str = "positive",
    n_points: int = 3,
    polynomial_order: int = 1,
    axis: int = 0,
    surface_tolerance_m: float = 0.0,
) -> np.ndarray:
    """Extrapolate a field to a surface using only samples on the vacuum side.

    This deliberately refuses to average across a PEC interface.  Samples nearest
    the surface are selected on the requested side and a real-coordinate polynomial
    is fitted to each real/complex field value independently.

    ``surface_tolerance_m`` admits a sample sitting essentially *on* the surface
    (``|distance| <= surface_tolerance_m``).  On a Yee grid the tangential-E and normal-H
    components have a node exactly on the conductor plane, where the boundary condition makes
    them zero; excluding it forces a long extrapolation through strongly curved data and
    produces a spurious field on the surface.  Components that straddle the boundary (normal
    E, tangential H) have their nearest node half a cell away, so with a tolerance well below
    half a cell they are untouched and still genuinely one-sided.  Keep the tolerance small:
    it must never admit a sample from inside the metal.
    """

    coordinate = np.asarray(coordinate_m, dtype=np.float64)
    data = np.asarray(values)
    data_axis = normalize_axis_index(axis, data.ndim)
    if coordinate.ndim != 1 or data.shape[data_axis] != coordinate.size:
        raise ValueError("coordinate_m must be 1-D and match values along axis")
    if vacuum_side not in {"positive", "negative"}:
        raise ValueError("vacuum_side must be 'positive' or 'negative'")
    if n_points < 1 or polynomial_order < 0 or polynomial_order >= n_points:
        raise ValueError("require n_points >= 1 and 0 <= polynomial_order < n_points")
    distance = coordinate - float(surface_m)
    tolerance = abs(float(surface_tolerance_m))
    on_surface = np.abs(distance) <= tolerance
    vacuum = distance > 0.0 if vacuum_side == "positive" else distance < 0.0
    eligible = np.flatnonzero(vacuum | on_surface)
    if eligible.size < n_points:
        raise ValueError(
            f"only {eligible.size} samples lie on the {vacuum_side} vacuum side; need {n_points}"
        )
    order = eligible[np.argsort(np.abs(distance[eligible]))[:n_points]]
    x = distance[order]
    coordinate_scale = float(np.max(np.abs(x)))
    if coordinate_scale == 0.0:
        raise ValueError("vacuum-side samples collapse onto the surface coordinate")
    if tolerance > 0.0:
        # An admitted on-surface sample is the boundary value itself; snap its tiny residual
        # offset to exactly zero so it anchors the fit instead of tilting it.
        x = np.where(np.abs(x) <= tolerance, 0.0, x)
    moved = np.moveaxis(data, data_axis, 0)[order]
    design = np.vander(x / coordinate_scale, N=polynomial_order + 1, increasing=True)
    coefficients, _, rank, _ = np.linalg.lstsq(design, moved.reshape(n_points, -1), rcond=None)
    if rank != polynomial_order + 1:
        raise ValueError("vacuum-side coordinates do not define the requested polynomial")
    return coefficients[0].reshape(moved.shape[1:])


def pair_meridional_halves(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    parity: str,
) -> np.ndarray:
    """Project paired signed-plane samples onto even or odd cylindrical parity."""

    plus = np.asarray(positive)
    minus = np.asarray(negative)
    if plus.shape != minus.shape:
        raise ValueError("positive and negative half-plane samples must have the same shape")
    if parity == "even":
        return 0.5 * (plus + minus)
    if parity == "odd":
        return 0.5 * (plus - minus)
    raise ValueError("parity must be 'even' or 'odd'")
