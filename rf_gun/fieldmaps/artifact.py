"""Versioned, digest-verified artifacts for axisymmetric RF-Track fields."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import h5py
import numpy as np

from .coordinates import FrameTransform
from .models import AxisymmetricRFField, FieldMapValidationError

ARTIFACT_SCHEMA_VERSION = "rf_gun_axisymmetric_rf_field_v1"
PHASOR_CONVENTION = "Re(F*exp(+i*2*pi*f*(t-t_ref)))"
QUALITY_STATUSES = {"qualified", "analysis_only"}
_REQUIRED_PAYLOAD_DATASETS = frozenset(
    {
        "/grid/r_m",
        "/grid/z_m",
        "/fields/Er_Vpm",
        "/fields/Ez_Vpm",
        "/fields/Btheta_T",
        "/fields/Bz_T",
        "/transform/origin_native_m",
        "/transform/R_rf_from_native",
    }
)


class ArtifactValidationError(FieldMapValidationError):
    """Raised for a corrupt, unsupported, or unqualified artifact."""


def json_compatible(value: Any) -> Any:
    """Convert supported scientific values to strict JSON-compatible records.

    Complex values retain real/imaginary parts; unsupported objects fail rather
    than being silently converted to display strings in artifact provenance.
    """
    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(json_compatible(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _hash_prefix(name: str, dtype: np.dtype[Any], shape: tuple[int, ...]) -> hashlib._Hash:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.dtype(dtype).str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(shape).encode("ascii"))
    digest.update(b"\0")
    return digest


def _hash_array(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = _hash_prefix(name, array.dtype, array.shape)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _hash_dataset(dataset: h5py.Dataset) -> str:
    digest = _hash_prefix(dataset.name, dataset.dtype, tuple(dataset.shape))
    if dataset.ndim == 0:
        digest.update(np.ascontiguousarray(dataset[()]).tobytes())
        return digest.hexdigest()
    rows_per_block = max(1, (8 * 1024 * 1024) // max(dataset.dtype.itemsize * int(np.prod(dataset.shape[1:])), 1))
    for start in range(0, dataset.shape[0], rows_per_block):
        block = np.ascontiguousarray(dataset[start : start + rows_per_block])
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def _storage_array(value: np.ndarray, *, complex_field: bool = False) -> np.ndarray:
    if complex_field:
        source = np.asarray(value)
        # Honor ReductionConfig/CLI's advertised precision choice.  The former
        # unconditional c8 cast silently discarded half of a complex128 build.
        dtype = np.dtype("<c8") if source.dtype == np.complex64 else np.dtype("<c16")
        return np.asarray(source, dtype=dtype)
    array = np.asarray(value)
    if array.dtype.kind == "b":
        return np.asarray(array, dtype=np.uint8)
    if array.dtype.kind == "c":
        return np.asarray(array, dtype=np.dtype("<c8"))
    if array.dtype.kind in "iu":
        return np.asarray(array)
    return np.asarray(array, dtype=np.dtype("<f8"))


def _chunks_for(array: np.ndarray) -> tuple[int, ...] | None:
    if array.ndim == 0 or array.size < 1024:
        return None
    if array.ndim == 1:
        return (min(array.shape[0], 8192),)
    return tuple(min(size, limit) for size, limit in zip(array.shape, (256,) * array.ndim, strict=True))


def _create_payload_dataset(
    handle: h5py.File,
    path: str,
    value: np.ndarray,
    digests: dict[str, str],
    *,
    units: str | None = None,
    complex_field: bool = False,
) -> h5py.Dataset:
    array = _storage_array(value, complex_field=complex_field)
    chunks = _chunks_for(array)
    kwargs: dict[str, Any] = {}
    if chunks is not None:
        kwargs.update(chunks=chunks, compression="gzip", compression_opts=4, shuffle=True, fletcher32=True)
    dataset = handle.create_dataset(path, data=array, **kwargs)
    if units is not None:
        dataset.attrs["units"] = units
    digest = _hash_array(dataset.name, array)
    dataset.attrs["sha256"] = digest
    digests[dataset.name] = digest
    return dataset


def _write_json(group: h5py.Group, name: str, value: Any) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=canonical_json(value), dtype=dtype)


def _read_json(handle: h5py.File, path: str) -> Any:
    value = handle[path][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _validate_source_sha(provenance: Mapping[str, Any]) -> None:
    source_sha = str(provenance.get("source_sha256", "")).lower()
    if len(source_sha) != 64 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ArtifactValidationError("provenance.source_sha256 must be a 64-character hexadecimal digest")


def _validate_quality_record(quality: Mapping[str, Any]) -> None:
    """Reject contradictory production labels before either writing or loading.

    ``require_qualified=False`` permits a deliberately exploratory
    ``analysis_only`` artifact; it must never turn a bare or failing record that
    merely says ``qualified`` into a trusted production input.
    """

    status = str(quality.get("status", ""))
    if status not in QUALITY_STATUSES:
        raise ArtifactValidationError(f"quality.status must be one of {sorted(QUALITY_STATUSES)}")
    if status != "qualified":
        return
    gates = quality.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ArtifactValidationError(
            "a qualified artifact must contain a non-empty quality.gates mapping"
        )
    for name, gate in gates.items():
        if not isinstance(gate, Mapping):
            raise ArtifactValidationError(f"quality gate {name!r} must be a mapping")
        passed = gate.get("passed")
        if not isinstance(passed, (bool, np.bool_)) or not bool(passed):
            raise ArtifactValidationError(
                f"qualified artifact has missing, non-boolean, or failing gate {name!r}"
            )


def _expected_payload_datasets(handle: h5py.File) -> set[str]:
    """Return the complete set of numeric datasets consumed by the reader.

    Digest-table coverage is part of the schema, not merely an optimisation for
    verification. Without this check a corrupted file could remove (for
    example) ``/fields/Bz_T`` from the digest table, update the aggregate hash,
    and leave the altered field itself unchecked.
    """

    expected = set(_REQUIRED_PAYLOAD_DATASETS)
    for optional_path in ("/grid/vacuum_mask", "/grid/aperture_radius_m"):
        if optional_path in handle:
            if not isinstance(handle[optional_path], h5py.Dataset):
                raise ArtifactValidationError(f"{optional_path} must be a dataset")
            expected.add(optional_path)
    if "/preview" in handle:
        preview_group = handle["/preview"]
        if not isinstance(preview_group, h5py.Group):
            raise ArtifactValidationError("/preview must be a group")
        for name, node in preview_group.items():
            if not isinstance(node, h5py.Dataset):
                raise ArtifactValidationError(
                    f"preview entry {name!r} must be a direct dataset"
                )
            expected.add(node.name)
    return expected


def _validate_payload_digest_table(
    handle: h5py.File, payload_digests: Any
) -> Mapping[str, str]:
    if not isinstance(payload_digests, Mapping):
        raise ArtifactValidationError("payload_digests_json must encode a mapping")
    normalized: dict[str, str] = {}
    for dataset_path, digest in payload_digests.items():
        if not isinstance(dataset_path, str) or not isinstance(digest, str):
            raise ArtifactValidationError("payload digest paths and values must be strings")
        lowered = digest.lower()
        if len(lowered) != 64 or any(char not in "0123456789abcdef" for char in lowered):
            raise ArtifactValidationError(
                f"payload digest for {dataset_path!r} is not a SHA-256 hexadecimal digest"
            )
        normalized[dataset_path] = lowered

    expected = _expected_payload_datasets(handle)
    actual = set(normalized)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ArtifactValidationError(
            "payload digest table does not exactly cover the artifact's numeric payload: "
            + "; ".join(details)
        )
    return normalized


@dataclass(frozen=True)
class AxisymmetricFieldArtifact:
    field: AxisymmetricRFField
    transform: FrameTransform
    quality: Mapping[str, Any]
    provenance: Mapping[str, Any]
    preview: Mapping[str, np.ndarray]
    vacuum_mask: np.ndarray | None
    aperture_radius_m: np.ndarray | None
    payload_digests: Mapping[str, str]
    payload_sha256: str


def write_axisymmetric_artifact(
    path: str | Path,
    field: AxisymmetricRFField,
    *,
    transform: FrameTransform,
    quality: Mapping[str, Any],
    provenance: Mapping[str, Any],
    preview: Mapping[str, np.ndarray] | None = None,
    vacuum_mask: np.ndarray | None = None,
    aperture_radius_m: np.ndarray | None = None,
    overwrite: bool = False,
) -> str:
    """Atomically write a compact field artifact and return its payload digest."""

    field.validate()
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    quality_data = dict(quality)
    _validate_quality_record(quality_data)
    provenance_data = dict(provenance)
    _validate_source_sha(provenance_data)
    if "processing_config" in provenance_data:
        calculated = canonical_json_sha256(provenance_data["processing_config"])
        supplied = provenance_data.get("processing_config_sha256")
        if supplied is not None and supplied != calculated:
            raise ArtifactValidationError("processing_config_sha256 does not match processing_config")
        provenance_data["processing_config_sha256"] = calculated
    if vacuum_mask is not None:
        mask = np.asarray(vacuum_mask, dtype=bool)
        if mask.shape != field.shape:
            raise ArtifactValidationError(f"vacuum_mask shape {mask.shape} does not match {field.shape}")
    else:
        mask = None
    if aperture_radius_m is not None:
        aperture = np.asarray(aperture_radius_m, dtype=np.float64)
        if aperture.shape not in {(field.z_m.size,), field.shape}:
            raise ArtifactValidationError(
                f"aperture_radius_m shape must be {(field.z_m.size,)} or {field.shape}, got {aperture.shape}"
            )
        if not np.all(np.isfinite(aperture)) or np.any(aperture < 0.0):
            raise ArtifactValidationError("aperture_radius_m must be finite and non-negative")
    else:
        aperture = None
    previews = dict(preview or {})
    for name in previews:
        if not name or "/" in name:
            raise ArtifactValidationError("preview names must be non-empty single HDF5 path segments")
    metadata_record = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "phasor_convention": PHASOR_CONVENTION,
        "frequency_hz": field.frequency_hz,
        "field": dict(field.metadata),
        "quality": quality_data,
        "provenance": provenance_data,
    }
    metadata_sha = canonical_json_sha256(metadata_record)

    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    digests: dict[str, str] = {}
    try:
        with h5py.File(temporary_path, "w") as handle:
            handle.attrs["schema_version"] = ARTIFACT_SCHEMA_VERSION
            handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            handle.attrs["phasor_convention"] = PHASOR_CONVENTION
            handle.attrs["frequency_hz"] = field.frequency_hz
            _create_payload_dataset(handle, "/grid/r_m", field.r_m, digests, units="m")
            _create_payload_dataset(handle, "/grid/z_m", field.z_m, digests, units="m")
            if mask is not None:
                _create_payload_dataset(handle, "/grid/vacuum_mask", mask, digests, units="1")
            if aperture is not None:
                _create_payload_dataset(
                    handle, "/grid/aperture_radius_m", aperture, digests, units="m"
                )
            for name, values, units in (
                ("Er_Vpm", field.Er_Vpm, "V/m"),
                ("Ez_Vpm", field.Ez_Vpm, "V/m"),
                ("Btheta_T", field.Btheta_T, "T"),
                ("Bz_T", field.Bz_T, "T"),
            ):
                _create_payload_dataset(
                    handle, f"/fields/{name}", values, digests, units=units, complex_field=True
                )
            _create_payload_dataset(
                handle,
                "/transform/origin_native_m",
                transform.origin_native_m,
                digests,
                units="m",
            )
            _create_payload_dataset(
                handle,
                "/transform/R_rf_from_native",
                transform.R_target_from_native,
                digests,
                units="1",
            )
            for name, values in previews.items():
                _create_payload_dataset(handle, f"/preview/{name}", np.asarray(values), digests)
            metadata_group = handle.require_group("/metadata")
            _write_json(metadata_group, "field_json", dict(field.metadata))
            quality_group = handle.require_group("/quality")
            _write_json(quality_group, "report_json", quality_data)
            provenance_group = handle.require_group("/provenance")
            _write_json(provenance_group, "record_json", provenance_data)
            payload_sha = canonical_json_sha256(
                {"dataset_digests": digests, "metadata_sha256": metadata_sha}
            )
            handle.attrs["payload_digests_json"] = canonical_json(digests)
            handle.attrs["metadata_sha256"] = metadata_sha
            handle.attrs["payload_sha256"] = payload_sha
            handle.flush()
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary_path, destination)
        return payload_sha
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _require_units(handle: h5py.File, path: str, expected: str) -> None:
    actual = handle[path].attrs.get("units", "")
    if isinstance(actual, bytes):
        actual = actual.decode("utf-8")
    if actual != expected:
        raise ArtifactValidationError(f"{path} units are {actual!r}, expected {expected!r}")


def read_axisymmetric_artifact(
    path: str | Path,
    *,
    verify: str = "payload",
    require_qualified: bool = True,
) -> AxisymmetricFieldArtifact:
    """Load an artifact, optionally verifying every stored numeric payload."""

    if verify not in {"none", "metadata", "payload"}:
        raise ValueError("verify must be 'none', 'metadata', or 'payload'")
    with h5py.File(path, "r") as handle:
        schema = handle.attrs.get("schema_version", "")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if schema != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"unsupported field artifact schema {schema!r}; expected {ARTIFACT_SCHEMA_VERSION!r}"
            )
        convention = handle.attrs.get("phasor_convention", "")
        if isinstance(convention, bytes):
            convention = convention.decode("utf-8")
        if convention != PHASOR_CONVENTION:
            raise ArtifactValidationError(f"unsupported phasor convention {convention!r}")
        for dataset, units in (
            ("/grid/r_m", "m"),
            ("/grid/z_m", "m"),
            ("/fields/Er_Vpm", "V/m"),
            ("/fields/Ez_Vpm", "V/m"),
            ("/fields/Btheta_T", "T"),
            ("/fields/Bz_T", "T"),
            ("/transform/origin_native_m", "m"),
            ("/transform/R_rf_from_native", "1"),
        ):
            if dataset not in handle:
                raise ArtifactValidationError(f"artifact is missing {dataset}")
            _require_units(handle, dataset, units)
        quality = _read_json(handle, "/quality/report_json")
        if not isinstance(quality, Mapping):
            raise ArtifactValidationError("artifact quality report must be a mapping")
        _validate_quality_record(quality)
        status = str(quality.get("status", ""))
        if require_qualified and status != "qualified":
            raise ArtifactValidationError(
                f"artifact status is {status!r}; production loading requires 'qualified'"
            )
        provenance = _read_json(handle, "/provenance/record_json")
        _validate_source_sha(provenance)
        metadata = _read_json(handle, "/metadata/field_json")
        metadata_record = {
            "schema_version": schema,
            "phasor_convention": convention,
            "frequency_hz": float(handle.attrs["frequency_hz"]),
            "field": metadata,
            "quality": quality,
            "provenance": provenance,
        }
        metadata_sha = str(handle.attrs.get("metadata_sha256", ""))
        if verify in {"metadata", "payload"} and canonical_json_sha256(metadata_record) != metadata_sha:
            raise ArtifactValidationError("artifact metadata digest mismatch")
        digest_json = handle.attrs.get("payload_digests_json", "{}")
        if isinstance(digest_json, bytes):
            digest_json = digest_json.decode("utf-8")
        try:
            decoded_digests = json.loads(str(digest_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError("payload_digests_json is not valid JSON") from exc
        payload_digests = _validate_payload_digest_table(handle, decoded_digests)
        payload_sha = str(handle.attrs.get("payload_sha256", ""))
        if verify in {"metadata", "payload"}:
            if canonical_json_sha256(
                {"dataset_digests": payload_digests, "metadata_sha256": metadata_sha}
            ) != payload_sha:
                raise ArtifactValidationError("canonical payload digest does not match digest table")
            for dataset_path, expected_digest in payload_digests.items():
                if dataset_path not in handle:
                    raise ArtifactValidationError(f"digested dataset {dataset_path} is missing")
                stored = handle[dataset_path].attrs.get("sha256", "")
                if isinstance(stored, bytes):
                    stored = stored.decode("utf-8")
                if stored != expected_digest:
                    raise ArtifactValidationError(f"digest attribute mismatch for {dataset_path}")
                if verify == "payload":
                    actual = _hash_dataset(handle[dataset_path])
                    if actual != expected_digest:
                        raise ArtifactValidationError(f"payload digest mismatch for {dataset_path}")
        frequency = float(handle.attrs["frequency_hz"])
        field = AxisymmetricRFField(
            frequency_hz=frequency,
            r_m=np.asarray(handle["/grid/r_m"]),
            z_m=np.asarray(handle["/grid/z_m"]),
            Er_Vpm=np.asarray(handle["/fields/Er_Vpm"]),
            Ez_Vpm=np.asarray(handle["/fields/Ez_Vpm"]),
            Btheta_T=np.asarray(handle["/fields/Btheta_T"]),
            Bz_T=np.asarray(handle["/fields/Bz_T"]),
            metadata=metadata,
        )
        transform = FrameTransform(
            origin_native_m=np.asarray(handle["/transform/origin_native_m"]),
            R_target_from_native=np.asarray(handle["/transform/R_rf_from_native"]),
        )
        preview = {
            name: np.asarray(dataset)
            for name, dataset in (handle["/preview"].items() if "/preview" in handle else [])
        }
        vacuum_mask = (
            np.asarray(handle["/grid/vacuum_mask"], dtype=bool)
            if "/grid/vacuum_mask" in handle
            else None
        )
        aperture = (
            np.asarray(handle["/grid/aperture_radius_m"], dtype=np.float64)
            if "/grid/aperture_radius_m" in handle
            else None
        )
    return AxisymmetricFieldArtifact(
        field=field,
        transform=transform,
        quality=MappingProxyType(quality),
        provenance=MappingProxyType(provenance),
        preview=MappingProxyType(preview),
        vacuum_mask=vacuum_mask,
        aperture_radius_m=aperture,
        payload_digests=MappingProxyType(payload_digests),
        payload_sha256=payload_sha,
    )


def load_axisymmetric_artifact(
    path: str | Path,
    *,
    verify: str = "payload",
    require_qualified: bool = True,
) -> AxisymmetricRFField:
    """Production convenience loader returning only the validated field object."""

    return read_axisymmetric_artifact(
        path, verify=verify, require_qualified=require_qualified
    ).field
