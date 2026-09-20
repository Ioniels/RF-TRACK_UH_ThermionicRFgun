"""Bounded section extraction and Yee-lattice co-location primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .xfdtd_h5 import H5Selection, XFdtdH5

SPATIAL_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class LinearStencil:
    indices: np.ndarray
    weights: np.ndarray
    target_coordinate_m: float

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if indices.ndim != 1 or weights.shape != indices.shape or indices.size not in {1, 2}:
            raise ValueError("a linear stencil must contain one or two index/weight pairs")
        if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("linear stencil weights must sum to one")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "weights", weights)


def linear_stencil(
    coordinates_m: np.ndarray,
    target_coordinate_m: float,
    *,
    allow_extrapolation: bool = False,
) -> LinearStencil:
    """Return the one- or two-node stencil for a monotonic coordinate axis."""

    coordinates = np.asarray(coordinates_m, dtype=np.float64)
    target = float(target_coordinate_m)
    if coordinates.ndim != 1 or coordinates.size < 1:
        raise ValueError("coordinates_m must be a non-empty 1-D array")
    if not np.all(np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("coordinates_m must be finite and strictly increasing")
    exact = np.flatnonzero(np.isclose(coordinates, target, rtol=0.0, atol=2.0e-15))
    if exact.size:
        return LinearStencil(np.array([exact[0]]), np.array([1.0]), target)
    upper = int(np.searchsorted(coordinates, target, side="right"))
    lower = upper - 1
    if lower < 0 or upper >= coordinates.size:
        if not allow_extrapolation:
            raise ValueError(
                f"target {target:.9g} m is outside [{coordinates[0]:.9g}, {coordinates[-1]:.9g}] m"
            )
        if lower < 0:
            lower, upper = 0, min(1, coordinates.size - 1)
        else:
            lower, upper = max(0, coordinates.size - 2), coordinates.size - 1
    if lower == upper:
        return LinearStencil(np.array([lower]), np.array([1.0]), target)
    fraction = (target - coordinates[lower]) / (coordinates[upper] - coordinates[lower])
    return LinearStencil(
        np.array([lower, upper]),
        np.array([1.0 - fraction, fraction]),
        target,
    )


@dataclass(frozen=True)
class AxisAlignedSection:
    quantity: str
    component: str
    fixed_axis: str
    fixed_coordinate_m: float
    free_axes: tuple[str, str]
    coordinates_m: tuple[np.ndarray, np.ndarray]
    time_s: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ComponentBlock:
    """One bounded time-dependent component with its own Yee coordinates."""

    quantity: str
    component: str
    coordinate_frame: str
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    time_s: np.ndarray
    values: np.ndarray


def coordinate_slice_for_bounds(
    coordinates_m: np.ndarray,
    bounds_m: tuple[float, float],
    *,
    guard_cells: int = 1,
) -> slice:
    """Index slice covering physical bounds plus interpolation guard cells."""

    coordinate = np.asarray(coordinates_m, dtype=np.float64)
    if coordinate.ndim != 1 or coordinate.size == 0 or np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("coordinates_m must be a non-empty increasing 1-D array")
    lower, upper = (float(bounds_m[0]), float(bounds_m[1]))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("bounds_m must be finite and ordered")
    if lower < coordinate[0] or upper > coordinate[-1]:
        raise ValueError(
            f"bounds [{lower:.9g}, {upper:.9g}] m exceed coordinate support "
            f"[{coordinate[0]:.9g}, {coordinate[-1]:.9g}] m"
        )
    guard = int(guard_cells)
    if guard < 0:
        raise ValueError("guard_cells must be non-negative")
    start = max(0, int(np.searchsorted(coordinate, lower, side="right")) - 1 - guard)
    stop = min(coordinate.size, int(np.searchsorted(coordinate, upper, side="left")) + 1 + guard)
    if stop <= start:
        stop = min(coordinate.size, start + 1)
    return slice(start, stop, 1)


def extract_component_block(
    reader: XFdtdH5,
    quantity: str,
    component: str,
    *,
    x_bounds_m: tuple[float, float],
    y_bounds_m: tuple[float, float],
    z_bounds_m: tuple[float, float],
    coordinate_frame: str = "cathode_centered",
    time_bounds: slice | None = None,
    guard_cells: int = 1,
) -> ComponentBlock:
    """Read a physically bounded 3-D region, never the complete source volume."""

    axes = reader.coordinates(quantity, component, frame=coordinate_frame)
    spatial_slices = tuple(
        coordinate_slice_for_bounds(axis, bounds, guard_cells=guard_cells)
        for axis, bounds in zip(
            axes, (x_bounds_m, y_bounds_m, z_bounds_m), strict=True
        )
    )
    shape = reader.component_shape(quantity, component)
    time_slice = _explicit_slice(time_bounds, shape[0], "time")
    selection = H5Selection(time_slice, *spatial_slices)
    values = reader.read_component(quantity, component, selection)
    selected_axes = tuple(np.asarray(axis[index]) for axis, index in zip(axes, spatial_slices, strict=True))
    return ComponentBlock(
        quantity=str(quantity).upper(),
        component=str(component).lower(),
        coordinate_frame=coordinate_frame,
        x_m=selected_axes[0],
        y_m=selected_axes[1],
        z_m=selected_axes[2],
        time_s=reader.times(quantity)[time_slice],
        values=values,
    )


def _explicit_slice(index: slice | None, size: int, name: str) -> slice:
    if index is None:
        return slice(0, size, 1)
    if not isinstance(index, slice):
        raise TypeError(f"{name} bound must be a slice")
    start = 0 if index.start is None else int(index.start)
    stop = size if index.stop is None else int(index.stop)
    step = 1 if index.step is None else int(index.step)
    normalized = slice(start, stop, step)
    start_n, stop_n, step_n = normalized.indices(size)
    if step_n <= 0 or start_n >= stop_n:
        raise ValueError(f"{name} bound is empty or reversed")
    return slice(start_n, stop_n, step_n)


def extract_axis_aligned_section(
    reader: XFdtdH5,
    quantity: str,
    component: str,
    *,
    fixed_axis: str,
    coordinate_m: float,
    coordinate_frame: str = "cathode_centered",
    time_bounds: slice | None = None,
    spatial_bounds: Mapping[str, slice] | None = None,
) -> AxisAlignedSection:
    """Read and linearly interpolate one component onto an axis-aligned plane.

    Only the one or two source planes needed by the interpolation are read.  Free
    axes may be restricted further with index slices in ``spatial_bounds``.
    """

    fixed = str(fixed_axis).lower()
    if fixed not in SPATIAL_AXES:
        raise ValueError("fixed_axis must be x, y, or z")
    coordinate_axes = reader.coordinates(quantity, component, frame=coordinate_frame)
    dataset_shape = reader.component_shape(quantity, component)
    fixed_index = SPATIAL_AXES.index(fixed)
    stencil = linear_stencil(coordinate_axes[fixed_index], coordinate_m)
    bounds = dict(spatial_bounds or {})
    unknown = set(bounds) - set(SPATIAL_AXES)
    if unknown:
        raise ValueError(f"unknown spatial bound axes: {sorted(unknown)}")
    spatial_slices = [
        _explicit_slice(bounds.get(axis), int(dataset_shape[index + 1]), axis)
        for index, axis in enumerate(SPATIAL_AXES)
    ]
    first = int(np.min(stencil.indices))
    last = int(np.max(stencil.indices))
    spatial_slices[fixed_index] = slice(first, last + 1, 1)
    time_slice = _explicit_slice(time_bounds, int(dataset_shape[0]), "time")
    selection = H5Selection(time_slice, *spatial_slices)
    block = reader.read_component(quantity, component, selection)
    local_indices = stencil.indices - first
    selected = np.take(block, local_indices, axis=fixed_index + 1)
    weight_shape = [1] * selected.ndim
    weight_shape[fixed_index + 1] = stencil.weights.size
    interpolated = np.sum(selected * stencil.weights.reshape(weight_shape), axis=fixed_index + 1)
    free_axes = tuple(axis for axis in SPATIAL_AXES if axis != fixed)
    free_coordinates = tuple(
        np.asarray(coordinate_axes[SPATIAL_AXES.index(axis)][spatial_slices[SPATIAL_AXES.index(axis)]])
        for axis in free_axes
    )
    time = reader.times(quantity)[time_slice]
    return AxisAlignedSection(
        quantity=str(quantity).upper(),
        component=str(component).lower(),
        fixed_axis=fixed,
        fixed_coordinate_m=float(coordinate_m),
        free_axes=free_axes,  # type: ignore[arg-type]
        coordinates_m=free_coordinates,  # type: ignore[arg-type]
        time_s=time,
        values=interpolated,
    )


@dataclass(frozen=True)
class RectilinearPhasor:
    """One complex component on its native rectilinear Yee lattice."""

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        axes = tuple(np.asarray(axis, dtype=np.float64) for axis in (self.x_m, self.y_m, self.z_m))
        expected = tuple(axis.size for axis in axes)
        values = np.asarray(self.values)
        if values.shape != expected:
            raise ValueError(f"values shape {values.shape} does not match coordinate shape {expected}")
        for axis in axes:
            if axis.ndim != 1 or np.any(np.diff(axis) <= 0.0):
                raise ValueError("rectilinear coordinate axes must be 1-D and strictly increasing")
        object.__setattr__(self, "x_m", axes[0])
        object.__setattr__(self, "y_m", axes[1])
        object.__setattr__(self, "z_m", axes[2])
        object.__setattr__(self, "values", values)


def colocate_rectilinear(
    field: RectilinearPhasor,
    target_points_m: np.ndarray,
    *,
    bounds_error: bool = True,
    fill_value: complex = np.nan + 1j * np.nan,
) -> np.ndarray:
    """Interpolate a Yee-staggered phasor to explicit common physical points."""

    points = np.asarray(target_points_m, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError("target_points_m must have final dimension 3")
    original_shape = points.shape[:-1]
    interpolator = RegularGridInterpolator(
        (field.x_m, field.y_m, field.z_m),
        field.values,
        method="linear",
        bounds_error=bounds_error,
        fill_value=fill_value,
    )
    return np.asarray(interpolator(points.reshape(-1, 3))).reshape(original_shape)


def colocate_components(
    fields: Mapping[str, RectilinearPhasor],
    target_points_m: np.ndarray,
    *,
    bounds_error: bool = True,
) -> dict[str, np.ndarray]:
    """Co-locate independently staggered components without sharing their axes."""

    return {
        name: colocate_rectilinear(field, target_points_m, bounds_error=bounds_error)
        for name, field in fields.items()
    }


def common_coordinate_overlap(fields: Mapping[str, RectilinearPhasor]) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper XYZ limits shared by every supplied component."""

    if not fields:
        raise ValueError("at least one field is required")
    lower = np.max(
        np.array([[field.x_m[0], field.y_m[0], field.z_m[0]] for field in fields.values()]), axis=0
    )
    upper = np.min(
        np.array([[field.x_m[-1], field.y_m[-1], field.z_m[-1]] for field in fields.values()]), axis=0
    )
    if np.any(lower >= upper):
        raise ValueError(f"fields have no common 3-D overlap: lower={lower}, upper={upper}")
    return lower, upper
