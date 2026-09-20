"""Lazy and bounded access to XFdtd full-volume E/H HDF5 exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import h5py
import numpy as np

from .models import ValidationIssue, ValidationReport
from ..provenance import sha256_file as _sha256_file

XDFDTD_SCHEMA_VERSION = 1
ARRAY_AXES = ("time", "x", "y", "z")
QUANTITIES = {"E": "V/m", "H": "A/m"}
COMPONENTS = ("x", "y", "z")
COORDINATE_FRAME_TRANSLATION_ATOL_M = 2.0e-12


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def sha256_file(path: str | Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    """Compatibility spelling for the shared streaming file hash."""
    return _sha256_file(path, chunk_size=block_bytes)


@dataclass(frozen=True)
class ComponentManifest:
    quantity: str
    component: str
    dataset_path: str
    shape: tuple[int, int, int, int]
    dtype: str
    chunks: tuple[int, int, int, int] | None
    compression: str | None
    units: str
    native_coordinate_group: str
    cathode_centered_coordinate_group: str


@dataclass(frozen=True)
class XFdtdManifest:
    path: Path
    file_size_bytes: int
    schema_version: int
    array_axes: tuple[str, str, str, str]
    root_attributes: Mapping[str, Any]
    components: Mapping[str, ComponentManifest]
    electric_time_s: np.ndarray
    magnetic_h_time_s: np.ndarray
    cathode_origin_native_m: np.ndarray

    @property
    def drive_frequency_hz(self) -> float:
        return float(self.root_attributes["serialized_drive_frequency_hz"])

    @property
    def source_power_w(self) -> float:
        return float(self.root_attributes["port_power_w"])


SelectionIndex = int | slice


@dataclass(frozen=True)
class H5Selection:
    """Explicit selection in the serialized ``(time, x, y, z)`` order."""

    time: SelectionIndex
    x: SelectionIndex
    y: SelectionIndex
    z: SelectionIndex

    def as_tuple(self) -> tuple[SelectionIndex, SelectionIndex, SelectionIndex, SelectionIndex]:
        return (self.time, self.x, self.y, self.z)


@dataclass(frozen=True)
class ReadEstimate:
    selected_shape: tuple[int, ...]
    logical_bytes: int
    touched_chunks: int | None
    estimated_decompressed_bytes: int | None


def _normalize_index(index: SelectionIndex, size: int, name: str) -> tuple[SelectionIndex, int, tuple[int, int]]:
    if isinstance(index, (int, np.integer)):
        resolved = int(index)
        if resolved < 0:
            resolved += size
        if not 0 <= resolved < size:
            raise IndexError(f"{name} index {index} is outside [0, {size})")
        return resolved, 0, (resolved, resolved + 1)
    if not isinstance(index, slice):
        raise TypeError(f"{name} selection must be an integer or slice")
    if index.start is None or index.stop is None:
        raise ValueError(f"{name} slice must have explicit start and stop bounds")
    step = 1 if index.step is None else int(index.step)
    if step <= 0:
        raise ValueError(f"{name} slice step must be positive")
    start, stop, step = index.indices(size)
    length = max(0, (stop - start + step - 1) // step)
    if length == 0:
        raise ValueError(f"{name} selection is empty")
    return slice(start, stop, step), length, (start, stop)


def _validated_selection(
    selection: H5Selection,
    shape: tuple[int, int, int, int],
) -> tuple[tuple[SelectionIndex, ...], tuple[int, ...], tuple[tuple[int, int], ...]]:
    indices: list[SelectionIndex] = []
    output_shape: list[int] = []
    extents: list[tuple[int, int]] = []
    for name, index, size in zip(ARRAY_AXES, selection.as_tuple(), shape, strict=True):
        normalized, length, extent = _normalize_index(index, size, name)
        indices.append(normalized)
        extents.append(extent)
        if not isinstance(normalized, int):
            output_shape.append(length)
    full_spatial = all(extents[i] == (0, shape[i]) for i in range(1, 4))
    if full_spatial:
        raise ValueError(
            "full-volume field reads are disabled; bound at least one spatial axis "
            "and process the source in sections or blocks"
        )
    return tuple(indices), tuple(output_shape), tuple(extents)


def _infer_cathode_origin(handle: h5py.File) -> np.ndarray:
    origins: list[np.ndarray] = []
    for label in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        offsets = []
        for axis in COMPONENTS:
            native = np.asarray(
                handle[f"coordinates/native/{label}/{axis}_m"], dtype=np.float64
            )
            centered = np.asarray(
                handle[f"coordinates/cathode_centered/{label}/{axis}_m"],
                dtype=np.float64,
            )
            if native.shape != centered.shape or native.size == 0:
                raise ValueError(
                    f"{label} {axis} native and cathode-centered coordinate arrays "
                    "must have the same non-empty shape"
                )
            pointwise_offset = native - centered
            offset = float(np.median(pointwise_offset))
            maximum_residual = float(np.max(np.abs(pointwise_offset - offset)))
            if not np.isfinite(offset) or not np.isfinite(maximum_residual):
                raise ValueError(
                    f"{label} {axis} native/cathode-centered coordinate offset is not finite"
                )
            if maximum_residual > COORDINATE_FRAME_TRANSLATION_ATOL_M:
                raise ValueError(
                    f"{label} {axis} native and cathode-centered coordinates are not "
                    "related by one rigid translation: maximum pointwise residual "
                    f"{maximum_residual:.6g} m exceeds "
                    f"{COORDINATE_FRAME_TRANSLATION_ATOL_M:.6g} m"
                )
            offsets.append(offset)
        origins.append(np.asarray(offsets))
    stacked = np.stack(origins)
    if not np.allclose(stacked, stacked[0], atol=2.0e-12, rtol=0.0):
        raise ValueError("coordinate groups do not encode one consistent cathode origin")
    origin = np.asarray(np.median(stacked, axis=0), dtype=np.float64)
    origin.setflags(write=False)
    return origin


def _manifest_from_handle(path: Path, handle: h5py.File) -> XFdtdManifest:
    attrs = MappingProxyType({key: _plain(value) for key, value in handle.attrs.items()})
    components: dict[str, ComponentManifest] = {}
    for quantity, expected_units in QUANTITIES.items():
        for component in COMPONENTS:
            dataset_path = f"/fields/{quantity}/{component}"
            dataset = handle[dataset_path]
            key = f"{quantity}{component}"
            components[key] = ComponentManifest(
                quantity=quantity,
                component=component,
                dataset_path=dataset_path,
                shape=tuple(int(value) for value in dataset.shape),
                dtype=str(dataset.dtype),
                chunks=None if dataset.chunks is None else tuple(int(value) for value in dataset.chunks),
                compression=dataset.compression,
                units=_text(dataset.attrs.get("units", "")),
                native_coordinate_group=_text(dataset.attrs.get("native_coordinate_group", "")),
                cathode_centered_coordinate_group=_text(
                    dataset.attrs.get("cathode_centered_coordinate_group", "")
                ),
            )
            if components[key].units != expected_units:
                # Retain the actual value in the manifest; the validator reports it.
                pass
    axes = tuple(part.strip() for part in _text(handle.attrs.get("array_axis_order", "")).split(","))
    if len(axes) != 4:
        axes = tuple(axes)  # type: ignore[assignment]
    electric_time = np.array(handle["/time/E_s"], dtype=np.float64, copy=True)
    magnetic_time = np.array(handle["/time/H_s"], dtype=np.float64, copy=True)
    electric_time.setflags(write=False)
    magnetic_time.setflags(write=False)
    return XFdtdManifest(
        path=path.resolve(),
        file_size_bytes=path.stat().st_size,
        schema_version=int(handle.attrs.get("schema_version", -1)),
        array_axes=axes,  # type: ignore[arg-type]
        root_attributes=attrs,
        components=MappingProxyType(components),
        electric_time_s=electric_time,
        magnetic_h_time_s=magnetic_time,
        cathode_origin_native_m=_infer_cathode_origin(handle),
    )


def inspect_xfdtd_h5(path: str | Path) -> XFdtdManifest:
    """Read only metadata, coordinate axes, and the two short timestamp arrays."""

    source = Path(path)
    with h5py.File(source, "r") as handle:
        return _manifest_from_handle(source, handle)


def validate_xfdtd_h5(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ValidationReport:
    """Validate the source schema without reading any 4-D field dataset.

    Supplying ``expected_sha256`` requests a streaming whole-file integrity check,
    which is intentionally opt-in because the production cavity file is 24 GB.
    """

    issues: list[ValidationIssue] = []
    try:
        manifest = inspect_xfdtd_h5(path)
    except Exception as exc:
        return ValidationReport((ValidationIssue("error", "open_or_schema", str(exc)),))
    if manifest.schema_version != XDFDTD_SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                "error",
                "schema_version",
                f"expected {XDFDTD_SCHEMA_VERSION}, found {manifest.schema_version}",
            )
        )
    if tuple(manifest.array_axes) != ARRAY_AXES:
        issues.append(
            ValidationIssue("error", "array_axis_order", f"expected {ARRAY_AXES}, found {manifest.array_axes}")
        )
    reference_shape: tuple[int, ...] | None = None
    for key, component in manifest.components.items():
        if len(component.shape) != 4:
            issues.append(ValidationIssue("error", "component_rank", f"{key} shape is {component.shape}"))
        if reference_shape is None:
            reference_shape = component.shape
        elif component.shape != reference_shape:
            issues.append(
                ValidationIssue("error", "component_shape", f"{key} shape differs from {reference_shape}")
            )
        expected_units = QUANTITIES[component.quantity]
        if component.units != expected_units:
            issues.append(
                ValidationIssue(
                    "error", "component_units", f"{key} units are {component.units!r}, expected {expected_units!r}"
                )
            )
        if component.dtype != "float32":
            issues.append(
                ValidationIssue("warning", "component_dtype", f"{key} dtype is {component.dtype}, expected float32")
            )
    for name, time in (("E", manifest.electric_time_s), ("H", manifest.magnetic_h_time_s)):
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
            issues.append(ValidationIssue("error", "time_axis", f"{name} timestamps are not finite/increasing"))
        if reference_shape is not None and time.size != reference_shape[0]:
            issues.append(
                ValidationIssue(
                    "error", "time_length", f"{name} has {time.size} timestamps for {reference_shape[0]} frames"
                )
            )
    required_attrs = (
        "serialized_drive_frequency_hz",
        "port_power_w",
        "simulation_timestep_s",
        "solver",
        "source_run_id",
    )
    for attribute in required_attrs:
        if attribute not in manifest.root_attributes:
            issues.append(ValidationIssue("error", "missing_attribute", attribute))
    for attribute in ("serialized_drive_frequency_hz", "port_power_w", "simulation_timestep_s"):
        if attribute in manifest.root_attributes:
            try:
                value = float(manifest.root_attributes[attribute])
            except (TypeError, ValueError):
                value = float("nan")
            if not np.isfinite(value) or value <= 0.0:
                issues.append(
                    ValidationIssue("error", "invalid_attribute", f"{attribute} must be finite and positive")
                )
    try:
        with h5py.File(path, "r") as handle:
            for key, component in manifest.components.items():
                label = f"{component.quantity}{component.component}"
                for frame, group_path in (
                    ("native", component.native_coordinate_group),
                    ("cathode_centered", component.cathode_centered_coordinate_group),
                ):
                    expected_group = f"/coordinates/{frame}/{label}"
                    if group_path != expected_group:
                        issues.append(
                            ValidationIssue(
                                "error",
                                "coordinate_group",
                                f"{key} points to {group_path!r}, expected {expected_group!r}",
                            )
                        )
                    for spatial_index, axis in enumerate(COMPONENTS, start=1):
                        coordinate_path = f"{expected_group}/{axis}_m"
                        if coordinate_path not in handle:
                            issues.append(
                                ValidationIssue("error", "missing_coordinate", coordinate_path)
                            )
                            continue
                        dataset = handle[coordinate_path]
                        values = np.asarray(dataset, dtype=np.float64)
                        if values.shape != (component.shape[spatial_index],):
                            issues.append(
                                ValidationIssue(
                                    "error",
                                    "coordinate_length",
                                    f"{coordinate_path} shape {values.shape} does not match field axis",
                                )
                            )
                        if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
                            issues.append(
                                ValidationIssue(
                                    "error", "coordinate_axis", f"{coordinate_path} is not finite/increasing"
                                )
                            )
                        units = dataset.attrs.get("units", "")
                        if _text(units) != "m":
                            issues.append(
                                ValidationIssue(
                                    "error", "coordinate_units", f"{coordinate_path} units are {_text(units)!r}"
                                )
                            )
            for quantity in QUANTITIES:
                time_dataset = handle[f"/time/{quantity}_s"]
                if _text(time_dataset.attrs.get("units", "")) != "s":
                    issues.append(
                        ValidationIssue("error", "time_units", f"/time/{quantity}_s units must be 's'")
                    )
    except Exception as exc:
        issues.append(ValidationIssue("error", "coordinate_validation", str(exc)))
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            issues.append(ValidationIssue("error", "invalid_expected_sha256", expected_sha256))
        else:
            actual = sha256_file(path)
            if actual != expected:
                issues.append(
                    ValidationIssue("error", "sha256_mismatch", f"expected {expected}, computed {actual}")
                )
    return ValidationReport(tuple(issues))


class XFdtdH5:
    """Context-managed lazy reader that refuses accidental full-volume loads."""

    def __init__(self, path: str | Path, *, max_logical_read_bytes: int = 512 * 1024 * 1024):
        self.path = Path(path)
        self.max_logical_read_bytes = int(max_logical_read_bytes)
        if self.max_logical_read_bytes <= 0:
            raise ValueError("max_logical_read_bytes must be positive")
        self._handle: h5py.File | None = None

    def __enter__(self) -> "XFdtdH5":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self.open()
        assert self._handle is not None
        return self._handle

    def open(self) -> None:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def manifest(self) -> XFdtdManifest:
        return _manifest_from_handle(self.path, self.handle)

    def times(self, quantity: str) -> np.ndarray:
        normalized = str(quantity).upper()
        if normalized not in QUANTITIES:
            raise ValueError("quantity must be 'E' or 'H'")
        result = np.array(self.handle[f"/time/{normalized}_s"], dtype=np.float64, copy=True)
        result.setflags(write=False)
        return result

    def coordinates(
        self,
        quantity: str,
        component: str,
        *,
        frame: str = "cathode_centered",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized_q = str(quantity).upper()
        normalized_c = str(component).lower()
        if normalized_q not in QUANTITIES or normalized_c not in COMPONENTS:
            raise ValueError("quantity/component must be E|H and x|y|z")
        if frame not in {"native", "cathode_centered"}:
            raise ValueError("frame must be 'native' or 'cathode_centered'")
        label = f"{normalized_q}{normalized_c}"
        result = []
        for axis in COMPONENTS:
            values = np.array(
                self.handle[f"/coordinates/{frame}/{label}/{axis}_m"], dtype=np.float64, copy=True
            )
            values.setflags(write=False)
            result.append(values)
        return tuple(result)  # type: ignore[return-value]

    def estimate_read(self, quantity: str, component: str, selection: H5Selection) -> ReadEstimate:
        dataset = self._dataset(quantity, component)
        _, output_shape, extents = _validated_selection(selection, tuple(dataset.shape))
        logical_bytes = int(np.prod(output_shape, dtype=np.int64)) * dataset.dtype.itemsize
        if dataset.chunks is None:
            touched_chunks = None
            decompressed = None
        else:
            chunk_counts = []
            for extent, chunk in zip(extents, dataset.chunks, strict=True):
                start, stop = extent
                first = start // chunk
                last = (stop - 1) // chunk
                chunk_counts.append(last - first + 1)
            touched_chunks = int(np.prod(chunk_counts, dtype=np.int64))
            decompressed = touched_chunks * int(np.prod(dataset.chunks, dtype=np.int64)) * dataset.dtype.itemsize
        return ReadEstimate(output_shape, logical_bytes, touched_chunks, decompressed)

    def component_shape(self, quantity: str, component: str) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self._dataset(quantity, component).shape)

    def read_component(self, quantity: str, component: str, selection: H5Selection) -> np.ndarray:
        dataset = self._dataset(quantity, component)
        normalized, _, _ = _validated_selection(selection, tuple(dataset.shape))
        estimate = self.estimate_read(quantity, component, selection)
        if estimate.logical_bytes > self.max_logical_read_bytes:
            raise MemoryError(
                f"requested read is {estimate.logical_bytes / 2**20:.1f} MiB, above the "
                f"configured {self.max_logical_read_bytes / 2**20:.1f} MiB limit"
            )
        return np.asarray(dataset[normalized])

    def _dataset(self, quantity: str, component: str) -> h5py.Dataset:
        normalized_q = str(quantity).upper()
        normalized_c = str(component).lower()
        if normalized_q not in QUANTITIES or normalized_c not in COMPONENTS:
            raise ValueError("quantity/component must be E|H and x|y|z")
        return self.handle[f"/fields/{normalized_q}/{normalized_c}"]
