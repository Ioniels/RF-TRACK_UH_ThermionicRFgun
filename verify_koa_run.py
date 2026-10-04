#!/usr/bin/env python3
"""Fail-closed completion and input-identity checks for the KOA launchers.

The SLURM scripts pass the *same* argument array to this helper and to the
simulation driver.  Consequently a completed directory is reused only when
its structured marker, immutable field artifact, parser-resolved arguments,
code identity, validation report, and required outputs still agree.

This module intentionally keeps the marker logic outside shell heredocs so it
is shared by every study and can be unit tested without starting RF-Track.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


from rf_gun.provenance import canonical_json_sha256, sha256_file


#: 3: Bout_* is the fixed-s exit-plane beam with plane-crossing x, y; 2 is still accepted.
TRANSPORT_MARKER_SCHEMA_VERSION = 3
SUPPORTED_TRANSPORT_MARKER_SCHEMA_VERSIONS = (2, 3)
MACROPULSE_MARKER_SCHEMA_VERSION = 2
TRANSPORT_MARKER_NAME = ".run_complete"
MACROPULSE_MARKER_NAME = ".macropulse_complete"
_SHA256_HEX_LENGTH = 64
_EPHEMERAL_SCAN_TAG_KEYS = frozenset({"slurm_job_id", "slurm_array_job_id"})
_REQUIRED_TRANSPORT_TAG_KEYS = frozenset(
    {
        "campaign",
        "study",
        "stage",
        "task_id",
        "git_commit",
        "git_state",
        "code_sha256",
        "rftrack_version",
        "slurm_job_id",
        "slurm_array_job_id",
        "slurm_array_task_id",
        "python_stack_sha256",
    }
)
_REQUIRED_MACROPULSE_TAG_KEYS = (
    _REQUIRED_TRANSPORT_TAG_KEYS - {"rftrack_version"}
) | {"numpy_version", "scipy_version", "h5py_version"}
_CODE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".csv"})
_MACROPULSE_REQUIRED_OUTPUTS = (
    "study_config.json",
    "study_results.json",
    "back_bombardment_events.h5",
    "back_bombardment_heat_source.h5",
    "back_bombardment_macropulse.h5",
)


class CompletionVerificationError(RuntimeError):
    """A run directory is incomplete, corrupt, stale, or for another request."""


def python_stack_identity() -> dict[str, Any]:
    """Stable numerical-runtime identity without importing banner/noisy extension modules."""

    distributions = (
        "numpy",
        "scipy",
        "h5py",
        "matplotlib",
        "openpmd-beamphysics",
        "RF-Track",
    )
    packages: dict[str, str] = {}
    for distribution in distributions:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    rftrack_files: dict[str, str] = {}
    try:
        rftrack_distribution = importlib.metadata.distribution("RF-Track")
        for relative_file in rftrack_distribution.files or ():
            filename = Path(str(relative_file)).name
            if filename == "RF_Track.py" or (
                filename.startswith("_RF_Track")
                and Path(filename).suffix.lower() in {".so", ".pyd", ".dll", ".dylib"}
            ):
                resolved_file = Path(rftrack_distribution.locate_file(relative_file))
                if resolved_file.is_file():
                    rftrack_files[str(relative_file)] = sha256_file(resolved_file)
    except importlib.metadata.PackageNotFoundError:
        pass
    payload = {
        "schema_version": 1,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": packages,
        "rftrack_files": rftrack_files,
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise CompletionVerificationError(f"{label} is not a SHA-256 hexadecimal digest")
    return str(value).lower()


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CompletionVerificationError(f"missing {label}: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionVerificationError(f"cannot read valid JSON {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompletionVerificationError(f"{label} at {path} must contain a JSON object")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_repo_module(filename: str, module_name: str):
    module_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise CompletionVerificationError(f"cannot import parser from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_remainder_separator(arguments: Sequence[str]) -> list[str]:
    values = list(arguments)
    if values and values[0] == "--":
        values.pop(0)
    if not values:
        raise CompletionVerificationError("the driver argument list after '--' is empty")
    return values


def resolved_transport_arguments(driver_arguments: Sequence[str]) -> dict[str, Any]:
    """Resolve the runner's current defaults/presets without doing any tracking."""

    arguments = _strip_remainder_separator(driver_arguments)
    module = _load_repo_module("run_thermionic_tm010.py", "_koa_transport_driver")
    previous_argv = sys.argv
    try:
        sys.argv = ["run_thermionic_tm010.py", *arguments]
        namespace = module.parse_args()
    except SystemExit as exc:
        raise CompletionVerificationError(
            f"transport driver arguments are invalid (argparse exit {exc.code})"
        ) from exc
    finally:
        sys.argv = previous_argv

    module.apply_preset(namespace)
    # Importing this lightweight module does not force the lazy RF-Track binding to load.
    from rf_gun.finesse_presets import apply_finesse_preset_to_args

    apply_finesse_preset_to_args(namespace, namespace.finesse)
    if (bool(namespace.save_screen_hdf5) or bool(namespace.save_openpmd_beam)) and not bool(
        namespace.store_screen_phase_space
    ):
        namespace.store_screen_phase_space = True
    if bool(namespace.cathode_backstop_enabled) and not bool(namespace.save_lost_particles):
        namespace.save_lost_particles = True

    if namespace.threads is None:
        scheduler_threads = os.environ.get("SLURM_CPUS_PER_TASK")
        try:
            namespace.threads = max(1, int(scheduler_threads)) if scheduler_threads else 1
        except ValueError as exc:
            raise CompletionVerificationError(
                f"SLURM_CPUS_PER_TASK={scheduler_threads!r} is not a valid thread count"
            ) from exc
    else:
        namespace.threads = max(1, int(namespace.threads))

    from rf_gun.io import to_json_safe

    resolved = to_json_safe(vars(namespace))
    if not isinstance(resolved, dict):  # defensive: vars(namespace) is always a mapping
        raise CompletionVerificationError("resolved transport arguments are not a mapping")
    return resolved


def resolved_macropulse_arguments(driver_arguments: Sequence[str]) -> dict[str, Any]:
    arguments = _strip_remainder_separator(driver_arguments)
    module = _load_repo_module(
        "run_back_bombardment_macropulse.py", "_koa_macropulse_driver"
    )
    try:
        namespace = module.parse_args(arguments)
    except SystemExit as exc:
        raise CompletionVerificationError(
            f"macropulse driver arguments are invalid (argparse exit {exc.code})"
        ) from exc
    # Avoid importing rf_gun merely to sanitize argparse's Paths.
    result: dict[str, Any] = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            result[key] = str(value)
    return result


def _split_scan_tags(tags: Any, *, label: str) -> dict[str, str]:
    if not isinstance(tags, list):
        raise CompletionVerificationError(f"{label} must be a list of key=value strings")
    result: dict[str, str] = {}
    for item in tags:
        if not isinstance(item, str) or "=" not in item:
            raise CompletionVerificationError(f"{label} contains invalid tag {item!r}")
        key, value = item.split("=", 1)
        if not key or not value:
            raise CompletionVerificationError(f"{label} contains empty key/value in {item!r}")
        if key in result:
            raise CompletionVerificationError(f"{label} repeats key {key!r}")
        result[key] = value
    return result


def _normalize_transport_arguments_for_resume(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_copy = dict(actual)
    expected_copy = dict(expected)
    actual_tags = _split_scan_tags(actual_copy.get("scan_tags"), label="stored scan_tags")
    expected_tags = _split_scan_tags(expected_copy.get("scan_tags"), label="requested scan_tags")
    for label, tags in (("stored", actual_tags), ("requested", expected_tags)):
        missing = sorted(_REQUIRED_TRANSPORT_TAG_KEYS - set(tags))
        if missing:
            raise CompletionVerificationError(f"{label} scan_tags are missing {missing}")
        _validate_provenance_tags(
            tags, label=f"{label} scan_tags", expected_stage="transport", require_rftrack=True
        )

    # A new resubmission necessarily has new job IDs.  Require the old run to record them, but
    # exclude only their values from the stable physics/campaign comparison.
    for key in _EPHEMERAL_SCAN_TAG_KEYS:
        if key in expected_tags and key not in actual_tags:
            raise CompletionVerificationError(f"stored scan_tags are missing scheduler tag {key!r}")
        actual_tags.pop(key, None)
        expected_tags.pop(key, None)
    actual_copy["scan_tags"] = [f"{key}={actual_tags[key]}" for key in sorted(actual_tags)]
    expected_copy["scan_tags"] = [f"{key}={expected_tags[key]}" for key in sorted(expected_tags)]
    return actual_copy, expected_copy


def _validate_provenance_tags(
    tags: Mapping[str, str],
    *,
    label: str,
    expected_stage: str,
    require_rftrack: bool,
) -> None:
    _require_sha256(tags.get("code_sha256"), f"{label} code_sha256")
    _require_sha256(tags.get("python_stack_sha256"), f"{label} python_stack_sha256")
    commit = str(tags.get("git_commit", "")).lower()
    if len(commit) not in {40, 64} or any(character not in "0123456789abcdef" for character in commit):
        raise CompletionVerificationError(f"{label} git_commit is not a Git object digest")
    if tags.get("git_state") not in {"clean", "dirty"}:
        raise CompletionVerificationError(f"{label} git_state must be 'clean' or 'dirty'")
    if tags.get("stage") != expected_stage:
        raise CompletionVerificationError(
            f"{label} stage must be {expected_stage!r}"
        )
    if require_rftrack:
        version = str(tags.get("rftrack_version", "")).strip().lower()
        if version in {"", "unknown", "unavailable", "unset", "none"}:
            raise CompletionVerificationError(f"{label} has no usable RF-Track version")


def _first_difference(actual: Any, expected: Any, path: str = "args") -> str | None:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            return (
                f"{path} keys differ: missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(actual_keys):
            difference = _first_difference(actual[key], expected[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path} lengths differ: stored={len(actual)}, requested={len(expected)}"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            difference = _first_difference(actual_item, expected_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(
        expected, (int, float)
    ) and not isinstance(expected, bool):
        if float(actual) == float(expected):
            return None
    elif actual == expected:
        return None
    return f"{path} differs: stored={actual!r}, requested={expected!r}"


def inspect_artifact(
    path: str | Path,
    *,
    tail_policy: str,
    expected_dr_um: float | None = None,
    expected_dz_um: float | None = None,
    expected_aperture_delta_mm: float | None = None,
) -> dict[str, Any]:
    from rf_gun.fieldmaps import (
        PRODUCTION_QUALITY_GATE_VERSION,
        REQUIRED_PRODUCTION_GATE_NAMES,
        read_axisymmetric_artifact,
    )

    artifact_path = Path(path)
    try:
        artifact = read_axisymmetric_artifact(
            artifact_path, verify="payload", require_qualified=True
        )
    except Exception as exc:
        raise CompletionVerificationError(
            f"field artifact failed qualified full-payload verification: {artifact_path}: {exc}"
        ) from exc
    requested_policy = str(tail_policy).strip().lower()
    expected_variant = {"modal": "modal", "zero-control": "zero_tail_control"}.get(
        requested_policy
    )
    if expected_variant is None:
        raise CompletionVerificationError("tail policy must be 'modal' or 'zero-control'")
    actual_variant = str(artifact.quality.get("variant", ""))
    if actual_variant != expected_variant:
        raise CompletionVerificationError(
            f"artifact variant {actual_variant!r} does not match requested tail policy "
            f"{requested_policy!r} (expected {expected_variant!r})"
        )
    quality_gates = artifact.quality.get("gates")
    if not isinstance(quality_gates, Mapping):
        raise CompletionVerificationError(
            "qualified field artifact has no production quality-gate mapping"
        )
    required_gates = set(REQUIRED_PRODUCTION_GATE_NAMES) | {"source_integrity"}
    missing_gates = sorted(required_gates - set(quality_gates))
    if missing_gates:
        raise CompletionVerificationError(
            "qualified field artifact is missing current production gates: "
            + ", ".join(missing_gates)
        )
    stale_gates = sorted(
        name
        for name in required_gates
        if not isinstance(quality_gates[name], Mapping)
        or quality_gates[name].get("gate_version") != PRODUCTION_QUALITY_GATE_VERSION
    )
    if stale_gates:
        raise CompletionVerificationError(
            f"field artifact was not qualified with {PRODUCTION_QUALITY_GATE_VERSION}: "
            + ", ".join(stale_gates)
        )

    dr_um = float(artifact.field.hr_m * 1.0e6)
    dz_um = float(artifact.field.hz_m * 1.0e6)
    processing_config = artifact.provenance.get("processing_config", {})
    if isinstance(processing_config, Mapping) and "tail_policy" in processing_config:
        expected_processing_policy = {
            "modal": "modal",
            "zero-control": "zero",
        }[requested_policy]
        if processing_config.get("tail_policy") != expected_processing_policy:
            raise CompletionVerificationError(
                "artifact processing tail policy disagrees with its requested/qualified variant"
            )
    grid_plan = (
        processing_config.get("grid_plan", {})
        if isinstance(processing_config, Mapping)
        else {}
    )
    # The reducer fits each uniform axis exactly to its physical endpoint, so its *realized*
    # spacing is generally a few parts in 1e5 away from the requested nominal spacing.  For
    # production artifacts verify both halves of that contract: the stored requested value must
    # equal the scan point, and the actual array spacing must equal the stored realized value.
    for stem, actual_um, expected_um in (
        ("dr", dr_um, expected_dr_um),
        ("dz", dz_um, expected_dz_um),
    ):
        if expected_um is None:
            continue
        nominal_tolerance_um = max(1.0e-10, abs(float(expected_um)) * 1.0e-10)
        requested_key = f"requested_{stem}_m"
        realized_key = f"realized_{stem}_m"
        if isinstance(grid_plan, Mapping) and requested_key in grid_plan:
            try:
                stored_requested_um = float(grid_plan[requested_key]) * 1.0e6
            except (TypeError, ValueError) as exc:
                raise CompletionVerificationError(
                    f"artifact grid plan {requested_key} is not numeric"
                ) from exc
            if abs(stored_requested_um - float(expected_um)) > nominal_tolerance_um:
                raise CompletionVerificationError(
                    f"artifact grid plan requested_{stem}={stored_requested_um:.12g} um does "
                    f"not match convergence point {float(expected_um):.12g} um"
                )
            if realized_key not in grid_plan:
                raise CompletionVerificationError(
                    f"artifact grid plan is missing {realized_key}"
                )
            try:
                expected_actual_um = float(grid_plan[realized_key]) * 1.0e6
            except (TypeError, ValueError) as exc:
                raise CompletionVerificationError(
                    f"artifact grid plan {realized_key} is not numeric"
                ) from exc
        else:
            # Small synthetic/older fixtures have no grid-plan record. They can still be checked
            # directly, but production preprocessing always writes the two-value contract above.
            expected_actual_um = float(expected_um)
        realized_tolerance_um = max(1.0e-10, abs(expected_actual_um) * 1.0e-10)
        if abs(actual_um - expected_actual_um) > realized_tolerance_um:
            raise CompletionVerificationError(
                f"artifact actual {stem}={actual_um:.12g} um does not match its expected "
                f"realized spacing {expected_actual_um:.12g} um"
            )

    metadata = artifact.field.metadata
    source_power = metadata.get("source_power_w")
    try:
        source_power_w = float(source_power)
    except (TypeError, ValueError) as exc:
        raise CompletionVerificationError(
            "qualified artifact field metadata has no finite source_power_w"
        ) from exc
    if not math.isfinite(source_power_w) or not (source_power_w > 0.0):
        raise CompletionVerificationError(
            "qualified artifact field metadata has no finite positive source_power_w"
        )
    source_manifest = artifact.provenance.get("source_manifest", {})
    if isinstance(source_manifest, Mapping) and "source_power_w" in source_manifest:
        try:
            manifest_power_w = float(source_manifest["source_power_w"])
        except (TypeError, ValueError) as exc:
            raise CompletionVerificationError(
                "artifact source manifest source_power_w is not numeric"
            ) from exc
        if not math.isfinite(manifest_power_w) or not math.isclose(
            manifest_power_w, source_power_w, rel_tol=1.0e-12, abs_tol=0.0
        ):
            raise CompletionVerificationError(
                "artifact field metadata and source manifest disagree on source power"
            )
    source_sha256 = _require_sha256(
        artifact.provenance.get("source_sha256"), "artifact provenance source_sha256"
    )
    processing_config_sha256 = _require_sha256(
        artifact.provenance.get("processing_config_sha256"),
        "artifact provenance processing_config_sha256",
    )
    if expected_aperture_delta_mm is not None:
        import numpy as np

        if artifact.aperture_radius_m is None:
            raise CompletionVerificationError(
                "qualified artifact has no stored aperture-radius profile"
            )
        from rf_gun.aperture import aperture_radius_profile_mm

        expected_aperture = aperture_radius_profile_mm(
            artifact.field.z_m * 1.0e3, float(expected_aperture_delta_mm)
        ) * 1.0e-3
        stored_aperture = artifact.aperture_radius_m
        if stored_aperture.ndim == 2:
            if artifact.vacuum_mask is None:
                raise CompletionVerificationError(
                    "two-dimensional artifact aperture requires a vacuum mask"
                )
            stored_aperture = np.max(
                np.where(artifact.vacuum_mask, artifact.field.r_m[None, :], 0.0),
                axis=1,
            )
        if stored_aperture.shape != expected_aperture.shape or not np.allclose(
            stored_aperture, expected_aperture, rtol=0.0, atol=2.0e-9
        ):
            raise CompletionVerificationError(
                "artifact aperture geometry does not match requested "
                f"delta_cathode_chamfer_mm={float(expected_aperture_delta_mm):.12g}"
            )
    return {
        "path": str(artifact_path.resolve()),
        "file_sha256": sha256_file(artifact_path),
        "payload_sha256": _require_sha256(artifact.payload_sha256, "artifact payload_sha256"),
        "source_sha256": source_sha256,
        "processing_config_sha256": processing_config_sha256,
        "source_power_w": source_power_w,
        "variant": actual_variant,
        "tail_policy": requested_policy,
        "frequency_hz": float(artifact.field.frequency_hz),
        "nr": int(artifact.field.r_m.size),
        "nz": int(artifact.field.z_m.size),
        "r_min_m": float(artifact.field.r_m[0]),
        "r_max_m": float(artifact.field.r_m[-1]),
        "z_min_m": float(artifact.field.z_m[0]),
        "z_max_m": float(artifact.field.z_m[-1]),
        "dr_um": dr_um,
        "dz_um": dz_um,
        "aperture_profile_present": artifact.aperture_radius_m is not None,
    }


def _safe_marker_member(run_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CompletionVerificationError(f"marker {label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or value != relative.as_posix() or ".." in relative.parts:
        raise CompletionVerificationError(f"marker {label} must be a normalized relative path")
    root = run_dir.resolve()
    candidate = (run_dir / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CompletionVerificationError(f"marker {label} escapes its run directory") from exc
    return candidate


def _verify_output_manifest(
    run_dir: Path, marker: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    outputs = marker.get("required_outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise CompletionVerificationError("completion marker has no required_outputs manifest")
    checked: dict[str, Mapping[str, Any]] = {}
    bound_paths: set[str] = set()
    for name, record in outputs.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise CompletionVerificationError("required_outputs entries must be named mappings")
        relative_file = record.get("file")
        path = _safe_marker_member(run_dir, relative_file, f"required_outputs.{name}.file")
        if relative_file in bound_paths:
            raise CompletionVerificationError(
                f"required_outputs binds {relative_file!r} more than once"
            )
        bound_paths.add(str(relative_file))
        expected_sha = _require_sha256(
            record.get("sha256"), f"required_outputs.{name}.sha256"
        )
        try:
            actual_sha = sha256_file(path)
        except OSError as exc:
            raise CompletionVerificationError(f"required output {name!r} is missing: {path}") from exc
        if actual_sha != expected_sha:
            raise CompletionVerificationError(
                f"required output {name!r} digest mismatch: {path}"
            )
        expected_schema = record.get("schema_version")
        if expected_schema is not None:
            try:
                import h5py

                with h5py.File(path, "r") as handle:
                    actual_schema = handle.attrs.get("schema_version", "")
                if isinstance(actual_schema, bytes):
                    actual_schema = actual_schema.decode("utf-8")
            except Exception as exc:
                raise CompletionVerificationError(
                    f"required HDF5 output {name!r} cannot be inspected: {path}: {exc}"
                ) from exc
            if actual_schema != expected_schema:
                raise CompletionVerificationError(
                    f"required HDF5 output {name!r} schema is {actual_schema!r}, "
                    f"expected {expected_schema!r}"
                )
        checked[name] = record
    actual_files: set[str] = set()
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {
            TRANSPORT_MARKER_NAME,
            MACROPULSE_MARKER_NAME,
            "resource_usage.txt",
        }:
            continue
        try:
            relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
        except ValueError as exc:
            raise CompletionVerificationError(
                f"output file escapes its run directory: {path}"
            ) from exc
        actual_files.add(relative)
    if actual_files != bound_paths:
        raise CompletionVerificationError(
            "completion output manifest does not exactly cover the run bundle: "
            f"unbound={sorted(actual_files - bound_paths)}, "
            f"missing={sorted(bound_paths - actual_files)}"
        )
    return checked


def _verify_requested_transport_outputs(
    outputs: Mapping[str, Mapping[str, Any]], arguments: Mapping[str, Any], z_max_m: float
) -> None:
    bound_paths = {str(record.get("file", "")) for record in outputs.values()}
    if not bool(arguments.get("screens_enabled", True)):
        expected_screen_count = 0
    elif arguments.get("screens_z"):
        expected_screen_count = len(arguments["screens_z"])
    else:
        expected_screen_count = max(0, int(arguments.get("n_screens", 0)))
    if bool(arguments.get("save_openpmd_beam")):
        # the CLI appends a screen at z_max for the exit-plane Bout when none is requested there
        requested_z = (
            [float(z) for z in arguments["screens_z"]]
            if bool(arguments.get("screens_enabled", True)) and arguments.get("screens_z")
            else ([float(z_max_m)] if expected_screen_count > 0 else [])
        )
        if not any(abs(z - float(z_max_m)) <= 1.0e-6 for z in requested_z):
            expected_screen_count += 1

    if bool(arguments.get("save_openpmd_beam")):
        for prefix in ("B0_", "Bout_"):
            if not any(
                Path(path).parent.as_posix() == "screen_distributions_hdf5"
                and Path(path).name.startswith(prefix)
                and Path(path).suffix.lower() == ".h5"
                for path in bound_paths
            ):
                raise CompletionVerificationError(
                    f"completion marker does not bind requested {prefix.rstrip('_')} beam HDF5"
                )

    if bool(arguments.get("save_screen_hdf5")):
        screen_paths = {
            path
            for path in bound_paths
            if Path(path).parent.as_posix() == "screen_distributions_hdf5"
            and Path(path).name.startswith("screen_")
            and Path(path).suffix.lower() == ".h5"
        }
        if len(screen_paths) != expected_screen_count:
            raise CompletionVerificationError(
                "completion marker binds "
                f"{len(screen_paths)} per-screen HDF5 files, expected {expected_screen_count}"
            )

    if bool(arguments.get("save_figures")):
        required_figure_names = {
            "field_maps.png",
            "on_axis_field_profile.png",
            "screen_spectra.png",
            "emission_history.png",
        }
        if expected_screen_count > 0:
            required_figure_names.update(
                {
                    "initial_phase_space_x_px.png",
                    "beam_moments_evolution.png",
                    "beam_twiss_evolution.png",
                }
            )
        if bool(arguments.get("emission_field_iteration")):
            required_figure_names.update(
                {
                    "emission_iteration_convergence.png",
                    "emission_iteration_waveforms.png",
                    "emission_iteration_near_cathode.png",
                }
            )
        missing_figures = sorted(
            name for name in required_figure_names if f"figures/{name}" not in bound_paths
        )
        if missing_figures:
            raise CompletionVerificationError(
                f"completion marker does not bind requested PNG figures: {missing_figures}"
            )
    if bool(arguments.get("save_lost_particles")) and "lost_particle_diagnostics.json" not in bound_paths:
        raise CompletionVerificationError(
            "completion marker does not bind requested lost-particle diagnostics"
        )
    if bool(arguments.get("emission_field_iteration")) and "emission_iteration.npz" not in bound_paths:
        raise CompletionVerificationError(
            "completion marker does not bind requested emission-iteration data"
        )


def _resolved_transport_identity_from_run_config(
    config: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        hardcoded = config["hardcoded_parameters"]
        derived = config["derived_parameters"]
        cavity = hardcoded["cavity"]
        field_provenance = hardcoded["field_provenance"]
        grid = field_provenance["grid"]
        magnetic = field_provenance["magnetic_field_source"]
        z_support = grid["z_support_m"]
        return {
            "frequency_hz": float(cavity["f_hz"]),
            "grid_shape": [int(grid["nz"]), int(grid["nr"])],
            "hr_m": float(grid["realized_hr_m"]),
            "hz_m": float(grid["realized_hz_m"]),
            "z_min_m": float(derived["z_min_m"]),
            "z_max_m": float(derived["z_max_m"]),
            "field_map_z0_m": float(z_support[0]),
            "field_support_z_m": [float(z_support[0]), float(z_support[1])],
            "rf_magnetic_field_enabled": bool(magnetic["enabled"]),
            "rf_forward_power_w": float(cavity["rf_forward_power_w"]),
            "transport_phase_deg": float(derived["transport_phase_deg"]),
            "phasor_mode_effective": str(cavity["phasor_mode"]),
        }
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise CompletionVerificationError(
            "run_config.json lacks the resolved artifact-backed transport identity"
        ) from exc


def _externally_expected_artifact_identity(
    artifact: Mapping[str, Any], arguments: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconstruct artifact-owned resolved values without trusting the run's JSON files."""

    try:
        z_min_m = float(arguments["z_min"])
        explicit_z_max = arguments.get("z_max")
        if explicit_z_max is None:
            from rf_gun.aperture import (
                DEFAULT_DELTA_CATHODE_CHAMFER_MM,
                exit_tube_end_z_m,
            )

            delta_value = arguments.get("delta_cathode_chamfer_mm")
            delta_mm = (
                DEFAULT_DELTA_CATHODE_CHAMFER_MM
                if delta_value is None
                else float(delta_value)
            )
            z_max_m = exit_tube_end_z_m(delta_mm) + float(arguments["ext_zmax"])
        else:
            z_max_m = float(explicit_z_max)
        support_min_m = float(artifact["z_min_m"])
        support_max_m = float(artifact["z_max_m"])
        hz_m = float(artifact["dz_um"]) * 1.0e-6
    except (KeyError, TypeError, ValueError) as exc:
        raise CompletionVerificationError(
            "cannot reconstruct the requested artifact-backed tracking domain"
        ) from exc

    if not all(math.isfinite(value) for value in (z_min_m, z_max_m, hz_m)):
        raise CompletionVerificationError("requested artifact-backed domain is not finite")
    if hz_m <= 0.0 or z_max_m <= z_min_m:
        raise CompletionVerificationError("requested artifact-backed domain is invalid")
    tolerance = max(1.0e-12, abs(hz_m) * 1.0e-7)
    if z_min_m < support_min_m - tolerance or z_max_m > support_max_m + tolerance:
        raise CompletionVerificationError(
            "requested tracking domain is outside the selected artifact support"
        )
    start_index = int(round((z_min_m - support_min_m) / hz_m))
    stop_index = int(round((z_max_m - support_min_m) / hz_m))
    selected_start_m = support_min_m + start_index * hz_m
    selected_stop_m = support_min_m + stop_index * hz_m
    if (
        abs(selected_start_m - z_min_m) > tolerance
        or abs(selected_stop_m - z_max_m) > tolerance
        or stop_index <= start_index
    ):
        raise CompletionVerificationError(
            "requested tracking-domain endpoints do not coincide with artifact z planes"
        )
    return {
        "frequency_hz": float(artifact["frequency_hz"]),
        "grid_shape": [stop_index - start_index + 1, int(artifact["nr"])],
        "hr_m": float(artifact["dr_um"]) * 1.0e-6,
        "hz_m": hz_m,
        "z_min_m": z_min_m,
        "z_max_m": z_max_m,
        "field_map_z0_m": selected_start_m,
        "field_support_z_m": [selected_start_m, selected_stop_m],
        "rf_magnetic_field_enabled": arguments.get("rf_magnetic_field") == "artifact",
        "rf_forward_power_w": float(arguments["bl_p_fwd_w"]),
        "phasor_mode_effective": "artifact",
    }


def _compare_resolved_to_external_identity(
    resolved: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for key, expected_value in expected.items():
        if key not in resolved:
            raise CompletionVerificationError(
                f"canonical_config.resolved is missing artifact-owned field {key!r}"
            )
        actual_value = resolved[key]
        if isinstance(expected_value, list):
            if not isinstance(actual_value, list) or len(actual_value) != len(expected_value):
                raise CompletionVerificationError(
                    f"canonical_config.resolved.{key} has the wrong shape"
                )
            for index, (actual_item, expected_item) in enumerate(
                zip(actual_value, expected_value, strict=True)
            ):
                if isinstance(expected_item, (int, float)) and not isinstance(expected_item, bool):
                    try:
                        actual_numeric = float(actual_item)
                    except (TypeError, ValueError) as exc:
                        raise CompletionVerificationError(
                            f"canonical_config.resolved.{key}[{index}] is not numeric"
                        ) from exc
                    tolerance = max(1.0e-14, abs(float(expected_item)) * 1.0e-12)
                    if abs(actual_numeric - float(expected_item)) > tolerance:
                        raise CompletionVerificationError(
                            f"canonical_config.resolved.{key}[{index}] disagrees with artifact/request"
                        )
                elif actual_item != expected_item:
                    raise CompletionVerificationError(
                        f"canonical_config.resolved.{key}[{index}] disagrees with artifact/request"
                    )
        elif isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            try:
                actual_numeric = float(actual_value)
            except (TypeError, ValueError) as exc:
                raise CompletionVerificationError(
                    f"canonical_config.resolved.{key} is not numeric"
                ) from exc
            tolerance = max(1.0e-14, abs(float(expected_value)) * 1.0e-12)
            if abs(actual_numeric - float(expected_value)) > tolerance:
                raise CompletionVerificationError(
                    f"canonical_config.resolved.{key} disagrees with artifact/request"
                )
        elif actual_value != expected_value:
            raise CompletionVerificationError(
                f"canonical_config.resolved.{key} disagrees with artifact/request"
            )


def verify_transport_completion(
    run_dir: str | Path,
    artifact_path: str | Path,
    driver_arguments: Sequence[str],
) -> dict[str, Any]:
    run_path = Path(run_dir)
    marker_path = run_path / TRANSPORT_MARKER_NAME
    marker = _read_json_mapping(marker_path, "transport completion marker")
    if marker.get("schema_version") not in SUPPORTED_TRANSPORT_MARKER_SCHEMA_VERSIONS:
        raise CompletionVerificationError(
            f"unsupported/legacy transport marker schema {marker.get('schema_version')!r}; "
            f"accepted {list(SUPPORTED_TRANSPORT_MARKER_SCHEMA_VERSIONS)}"
        )
    if marker.get("stage") != "transport" or marker.get("status") != "complete":
        raise CompletionVerificationError("transport marker stage/status is not transport/complete")
    if marker.get("validation_status") != "ok":
        raise CompletionVerificationError("transport marker validation_status is not 'ok'")

    expected_arguments = resolved_transport_arguments(driver_arguments)
    expected_artifact_arg = expected_arguments.get("field_artifact")
    if expected_artifact_arg is None:
        raise CompletionVerificationError("KOA production verification requires --field-artifact")
    if Path(str(expected_artifact_arg)).resolve() != Path(artifact_path).resolve():
        raise CompletionVerificationError(
            "the --artifact verifier input and runner --field-artifact do not resolve to the same file"
        )
    tail_policy = str(expected_arguments.get("field_tail_policy"))
    from rf_gun.aperture import DEFAULT_DELTA_CATHODE_CHAMFER_MM

    aperture_delta_value = expected_arguments.get("delta_cathode_chamfer_mm")
    aperture_delta_mm = (
        DEFAULT_DELTA_CATHODE_CHAMFER_MM
        if aperture_delta_value is None
        else float(aperture_delta_value)
    )
    artifact = inspect_artifact(
        artifact_path,
        tail_policy=tail_policy,
        expected_aperture_delta_mm=aperture_delta_mm,
    )

    config_path = _safe_marker_member(
        run_path, marker.get("run_config_file"), "run_config_file"
    )
    if config_path.relative_to(run_path.resolve()).as_posix() != "run_config.json":
        raise CompletionVerificationError("run_config_file must be 'run_config.json'")
    marker_config_sha = _require_sha256(
        marker.get("run_config_sha256"), "marker run_config_sha256"
    )
    if sha256_file(config_path) != marker_config_sha:
        raise CompletionVerificationError("run_config.json digest does not match completion marker")
    config = _read_json_mapping(config_path, "run configuration")
    try:
        stored_identity = config["hardcoded_parameters"]["run_identity"]
        actual_arguments = stored_identity["args"]
    except (KeyError, TypeError) as exc:
        raise CompletionVerificationError(
            "run_config.json has no hardcoded_parameters.run_identity.args mapping"
        ) from exc
    if not isinstance(actual_arguments, dict):
        raise CompletionVerificationError("stored run_identity.args is not a mapping")
    stored_tags_for_runtime = _split_scan_tags(
        actual_arguments.get("scan_tags"), label="stored scan_tags"
    )
    try:
        rftrack_runtime = config["hardcoded_parameters"]["runtime_environment"]["rftrack"]
        runtime_version = str(rftrack_runtime["version"])
        runtime_module_files = rftrack_runtime["module_files"]
    except (KeyError, TypeError) as exc:
        raise CompletionVerificationError(
            "run_config.json lacks the RF-Track runtime identity"
        ) from exc
    if runtime_version != stored_tags_for_runtime.get("rftrack_version"):
        raise CompletionVerificationError(
            "run_config RF-Track version disagrees with its submitted identity tag"
        )
    if not isinstance(runtime_module_files, Mapping) or not runtime_module_files:
        raise CompletionVerificationError(
            "run_config RF-Track runtime identity has no module-file digests"
        )
    for label, record in runtime_module_files.items():
        if not isinstance(label, str) or not isinstance(record, Mapping):
            raise CompletionVerificationError(
                "run_config RF-Track module-file identity is malformed"
            )
        _require_sha256(record.get("sha256"), f"RF-Track module {label} sha256")
    actual_normalized, expected_normalized = _normalize_transport_arguments_for_resume(
        actual_arguments, expected_arguments
    )
    difference = _first_difference(actual_normalized, expected_normalized)
    if difference is not None:
        raise CompletionVerificationError(f"transport request identity mismatch: {difference}")

    if stored_identity.get("run_family") != expected_arguments.get("run_family"):
        raise CompletionVerificationError("run_config run_family does not match requested run family")
    if marker.get("run_family") != expected_arguments.get("run_family"):
        raise CompletionVerificationError("completion marker run_family does not match requested run family")
    marker_tags = marker.get("scan_tags")
    marker_tag_args = {"scan_tags": marker_tags}
    config_tag_args = {"scan_tags": actual_arguments.get("scan_tags")}
    marker_normalized, config_normalized = _normalize_transport_arguments_for_resume(
        marker_tag_args, config_tag_args
    )
    if marker_normalized != config_normalized:
        raise CompletionVerificationError("completion marker scan_tags do not match run_config")

    field_identity = marker.get("field_identity")
    if not isinstance(field_identity, dict):
        raise CompletionVerificationError("completion marker field_identity is missing")
    expected_b_mode = str(expected_arguments.get("rf_magnetic_field"))
    expected_b_enabled = expected_b_mode == "artifact"
    expected_power = float(expected_arguments.get("bl_p_fwd_w"))
    if not math.isfinite(expected_power) or expected_power <= 0.0:
        raise CompletionVerificationError("requested RF forward power must be finite and positive")
    expected_scale = (expected_power / float(artifact["source_power_w"])) ** 0.5
    exact_field_values = {
        "kind": "qualified_field_artifact",
        "path": artifact["path"],
        "file_sha256": artifact["file_sha256"],
        "payload_sha256": artifact["payload_sha256"],
        "payload_verification": "payload",
        "source_sha256": artifact["source_sha256"],
        "processing_config_sha256": artifact["processing_config_sha256"],
        "rf_magnetic_field_mode": expected_b_mode,
        "rf_magnetic_field_enabled": expected_b_enabled,
        "field_tail_policy": tail_policy,
        "artifact_variant": artifact["variant"],
    }
    for key, expected in exact_field_values.items():
        if field_identity.get(key) != expected:
            raise CompletionVerificationError(
                f"completion marker field_identity.{key}={field_identity.get(key)!r}, "
                f"expected {expected!r}"
            )
    for key, expected in (
        ("source_power_w", float(artifact["source_power_w"])),
        ("target_power_w", expected_power),
        ("common_field_scale", expected_scale),
    ):
        try:
            actual = float(field_identity[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompletionVerificationError(f"field_identity.{key} is missing/non-numeric") from exc
        if abs(actual - expected) > max(1.0e-14, abs(expected) * 1.0e-12):
            raise CompletionVerificationError(
                f"field_identity.{key}={actual:.12g}, expected {expected:.12g}"
            )
    if marker.get("field_artifact_sha256") != artifact["file_sha256"]:
        raise CompletionVerificationError("top-level artifact file digest does not match selected artifact")
    if marker.get("field_artifact_payload_sha256") != artifact["payload_sha256"]:
        raise CompletionVerificationError("top-level artifact payload digest does not match selected artifact")
    expected_field_input_sha = canonical_json_sha256(field_identity)
    if marker.get("field_input_sha256") != expected_field_input_sha:
        raise CompletionVerificationError("field_input_sha256 does not match field_identity")

    canonical_config = marker.get("canonical_config")
    if not isinstance(canonical_config, dict):
        raise CompletionVerificationError("completion marker canonical_config is missing")
    canonical_config_sha = _require_sha256(
        marker.get("canonical_config_sha256"), "canonical_config_sha256"
    )
    if canonical_json_sha256(canonical_config) != canonical_config_sha:
        raise CompletionVerificationError("canonical_config_sha256 does not hash canonical_config")
    canonical_arguments = canonical_config.get("arguments")
    if not isinstance(canonical_arguments, dict):
        raise CompletionVerificationError("canonical_config.arguments is missing")
    stored_arguments_without_output = {
        key: value for key, value in actual_arguments.items() if key != "output"
    }
    difference = _first_difference(
        canonical_arguments,
        stored_arguments_without_output,
        "canonical_config.arguments",
    )
    if difference is not None:
        raise CompletionVerificationError(
            f"canonical completion configuration disagrees with run_config: {difference}"
        )
    canonical_resolved = canonical_config.get("resolved")
    if not isinstance(canonical_resolved, dict):
        raise CompletionVerificationError("canonical_config.resolved is missing")
    expected_resolved = _resolved_transport_identity_from_run_config(config)
    difference = _first_difference(
        canonical_resolved, expected_resolved, "canonical_config.resolved"
    )
    if difference is not None:
        raise CompletionVerificationError(
            f"canonical resolved configuration disagrees with run_config: {difference}"
        )
    expected_identity = _externally_expected_artifact_identity(artifact, expected_arguments)
    _compare_resolved_to_external_identity(canonical_resolved, expected_identity)
    outputs = _verify_output_manifest(run_path, marker)
    canonical_output_names = {
        "run_config": "run_config.json",
        "run_results": "run_results.json",
        "validation": "validation.json",
    }
    for required_name, required_file in canonical_output_names.items():
        if required_name not in outputs:
            raise CompletionVerificationError(
                f"completion output manifest is missing {required_name!r}"
            )
        if outputs[required_name].get("file") != required_file:
            raise CompletionVerificationError(
                f"completion output {required_name!r} must bind {required_file!r}"
            )
    if outputs["run_config"].get("sha256") != marker_config_sha:
        raise CompletionVerificationError("run_config digest disagrees inside completion marker")
    validation_path = _safe_marker_member(
        run_path, marker.get("validation_file"), "validation_file"
    )
    if validation_path.relative_to(run_path.resolve()).as_posix() != "validation.json":
        raise CompletionVerificationError("validation_file must be 'validation.json'")
    validation = _read_json_mapping(validation_path, "validation report")
    if validation.get("status") != "ok":
        raise CompletionVerificationError("validation report status is not 'ok'")
    # run_path may be relative (the launchers pass --run-dir relative to the repo root) while
    # _safe_marker_member always returns a resolved absolute path, so this comparison must
    # resolve too -- otherwise relative_to() raises ValueError and the completion gate crashes
    # instead of returning a verdict.
    if (
        outputs["validation"].get("file")
        != validation_path.relative_to(run_path.resolve()).as_posix()
    ):
        raise CompletionVerificationError("validation output manifest names a different file")
    _read_json_mapping(
        run_path / str(outputs["run_results"]["file"]), "run results"
    )

    if bool(expected_arguments.get("cathode_backstop_enabled")):
        event_record = outputs.get("back_bombardment_events")
        if event_record is None:
            raise CompletionVerificationError(
                "backstop-enabled transport marker does not bind back_bombardment_events.h5"
            )
        if event_record.get("schema_version") != "back_bombardment_events_v2":
            raise CompletionVerificationError("transport event output has wrong recorded schema")
        if event_record.get("file") != "back_bombardment_events.h5":
            raise CompletionVerificationError(
                "transport event output must bind 'back_bombardment_events.h5'"
            )
    _verify_requested_transport_outputs(outputs, expected_arguments, expected_identity["z_max_m"])
    return {
        "status": "exact_match",
        "run_dir": str(run_path),
        "marker_sha256": sha256_file(marker_path),
        "run_config_sha256": marker_config_sha,
        "artifact": artifact,
    }


def _macropulse_source_identity(transport_dir: Path) -> dict[str, Any]:
    marker_path = transport_dir / TRANSPORT_MARKER_NAME
    marker = _read_json_mapping(marker_path, "source transport marker")
    if marker.get("schema_version") not in SUPPORTED_TRANSPORT_MARKER_SCHEMA_VERSIONS:
        raise CompletionVerificationError("source transport marker is legacy or unsupported")
    if marker.get("stage") != "transport" or marker.get("status") != "complete":
        raise CompletionVerificationError("source transport is not marked complete")
    if marker.get("validation_status") != "ok":
        raise CompletionVerificationError("source transport validation_status is not 'ok'")
    config_path = _safe_marker_member(
        transport_dir, marker.get("run_config_file"), "transport run_config_file"
    )
    if config_path.relative_to(transport_dir.resolve()).as_posix() != "run_config.json":
        raise CompletionVerificationError(
            "source transport run_config_file must be 'run_config.json'"
        )
    config_sha = sha256_file(config_path)
    if config_sha != marker.get("run_config_sha256"):
        raise CompletionVerificationError("source transport run_config digest mismatch")
    config = _read_json_mapping(config_path, "source transport run configuration")
    try:
        stored_identity = config["hardcoded_parameters"]["run_identity"]
        stored_arguments = stored_identity["args"]
    except (KeyError, TypeError) as exc:
        raise CompletionVerificationError(
            "source transport run_config lacks its parser-resolved run identity"
        ) from exc
    if not isinstance(stored_arguments, Mapping):
        raise CompletionVerificationError(
            "source transport run_identity.args is not a mapping"
        )
    if marker.get("run_family") != stored_identity.get("run_family"):
        raise CompletionVerificationError(
            "source transport marker/run_config run families disagree"
        )
    marker_tags = _split_scan_tags(
        marker.get("scan_tags"), label="source transport marker scan_tags"
    )
    stored_tags = _split_scan_tags(
        stored_arguments.get("scan_tags"), label="source transport run_config scan_tags"
    )
    if marker_tags != stored_tags:
        raise CompletionVerificationError(
            "source transport marker/run_config scan tags disagree"
        )
    missing_tags = sorted(_REQUIRED_TRANSPORT_TAG_KEYS - set(marker_tags))
    if missing_tags:
        raise CompletionVerificationError(
            f"source transport scan_tags are missing {missing_tags}"
        )
    _validate_provenance_tags(
        marker_tags,
        label="source transport scan_tags",
        expected_stage="transport",
        require_rftrack=True,
    )
    canonical_config = marker.get("canonical_config")
    if not isinstance(canonical_config, Mapping):
        raise CompletionVerificationError(
            "source transport marker lacks canonical_config"
        )
    canonical_sha = _require_sha256(
        marker.get("canonical_config_sha256"),
        "source transport canonical_config_sha256",
    )
    if canonical_json_sha256(canonical_config) != canonical_sha:
        raise CompletionVerificationError(
            "source transport canonical_config digest mismatch"
        )
    canonical_arguments = canonical_config.get("arguments")
    if not isinstance(canonical_arguments, Mapping):
        raise CompletionVerificationError(
            "source transport canonical_config.arguments is missing"
        )
    stored_without_output = {
        key: value for key, value in stored_arguments.items() if key != "output"
    }
    difference = _first_difference(
        canonical_arguments,
        stored_without_output,
        "source_transport.canonical_config.arguments",
    )
    if difference is not None:
        raise CompletionVerificationError(
            f"source transport canonical arguments disagree with run_config: {difference}"
        )
    canonical_resolved = canonical_config.get("resolved")
    if not isinstance(canonical_resolved, Mapping):
        raise CompletionVerificationError(
            "source transport canonical_config.resolved is missing"
        )
    difference = _first_difference(
        canonical_resolved,
        _resolved_transport_identity_from_run_config(config),
        "source_transport.canonical_config.resolved",
    )
    if difference is not None:
        raise CompletionVerificationError(
            f"source transport canonical resolved identity disagrees with run_config: {difference}"
        )
    field_identity = marker.get("field_identity")
    if not isinstance(field_identity, Mapping):
        raise CompletionVerificationError("source transport field_identity is missing")
    if marker.get("field_input_sha256") != canonical_json_sha256(field_identity):
        raise CompletionVerificationError("source transport field_input_sha256 mismatch")
    if marker.get("field_artifact_sha256") != field_identity.get("file_sha256"):
        raise CompletionVerificationError(
            "source transport top-level/artifact file digests disagree"
        )
    if marker.get("field_artifact_payload_sha256") != field_identity.get("payload_sha256"):
        raise CompletionVerificationError(
            "source transport top-level/artifact payload digests disagree"
        )

    outputs = _verify_output_manifest(transport_dir, marker)
    config_record = outputs.get("run_config")
    if not isinstance(config_record, Mapping) or config_record.get("file") != config_path.name:
        raise CompletionVerificationError(
            "source transport output manifest does not bind its run_config file"
        )
    if config_record.get("sha256") != config_sha:
        raise CompletionVerificationError(
            "source transport run_config digest disagrees inside its marker"
        )
    results_record = outputs.get("run_results")
    if not isinstance(results_record, Mapping) or results_record.get("file") != "run_results.json":
        raise CompletionVerificationError(
            "source transport output manifest does not bind 'run_results.json'"
        )
    _read_json_mapping(
        transport_dir / "run_results.json", "source transport run results"
    )
    validation_record = outputs.get("validation")
    if not isinstance(validation_record, Mapping):
        raise CompletionVerificationError("source transport marker does not bind validation.json")
    validation_path = _safe_marker_member(
        transport_dir, validation_record.get("file"), "transport validation file"
    )
    if validation_record.get("file") != "validation.json":
        raise CompletionVerificationError(
            "source transport validation output must bind 'validation.json'"
        )
    validation = _read_json_mapping(validation_path, "source transport validation report")
    if validation.get("status") != "ok":
        raise CompletionVerificationError("source transport validation report status is not 'ok'")
    event_record = outputs.get("back_bombardment_events")
    if not isinstance(event_record, Mapping):
        raise CompletionVerificationError("source transport marker does not bind its event HDF5")
    if event_record.get("schema_version") != "back_bombardment_events_v2":
        raise CompletionVerificationError("source transport event output is not v2")
    if event_record.get("file") != "back_bombardment_events.h5":
        raise CompletionVerificationError(
            "source transport event output must bind 'back_bombardment_events.h5'"
        )
    event_path = _safe_marker_member(
        transport_dir, event_record.get("file"), "transport event file"
    )
    return {
        "marker_file": marker_path.name,
        "marker_sha256": sha256_file(marker_path),
        "run_config_file": config_path.name,
        "run_config_sha256": config_sha,
        "event_file": event_path.name,
        "event_file_sha256": sha256_file(event_path),
        "event_schema_version": event_record.get("schema_version"),
    }


def macropulse_source_identity_sha256(transport_dir: str | Path) -> str:
    """Digest the complete validated transport source consumed by stage 2."""

    return canonical_json_sha256(_macropulse_source_identity(Path(transport_dir)))


def _normalize_identity_tags(tags: Sequence[str] | Mapping[str, Any]) -> dict[str, str]:
    if isinstance(tags, Mapping):
        result: dict[str, str] = {}
        for key, value in tags.items():
            if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
                raise CompletionVerificationError(
                    "macropulse identity-tag mapping must contain non-empty string keys/values"
                )
            result[key] = value
        return result
    return _split_scan_tags(list(tags), label="macropulse identity tags")


def _stable_identity_tags(tags: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in tags.items() if key not in _EPHEMERAL_SCAN_TAG_KEYS}


def _output_record(
    path: Path,
    *,
    schema_version: str | None = None,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    filename = path.name if relative_to is None else path.resolve().relative_to(
        relative_to.resolve()
    ).as_posix()
    record: dict[str, Any] = {"file": filename, "sha256": sha256_file(path)}
    if schema_version is not None:
        record["schema_version"] = schema_version
    return record


def _verify_macropulse_source_references(
    output_dir: Path, source_transport: Mapping[str, Any]
) -> None:
    """Cross-check the source-event hash embedded in every stage-2 product."""

    expected_event_sha = _require_sha256(
        source_transport.get("event_file_sha256"), "source transport event_file_sha256"
    )
    results = _read_json_mapping(output_dir / "study_results.json", "macropulse study results")
    if results.get("event_file_hash") != expected_event_sha:
        raise CompletionVerificationError(
            "study_results.json does not identify the bound transport event file"
        )

    try:
        import h5py

        references = (
            ("back_bombardment_heat_source.h5", "source_events_hash"),
            ("back_bombardment_macropulse.h5", "event_file_hash"),
        )
        for filename, attribute in references:
            with h5py.File(output_dir / filename, "r") as handle:
                actual = handle.attrs.get(attribute, "")
            if isinstance(actual, bytes):
                actual = actual.decode("utf-8")
            if actual != expected_event_sha:
                raise CompletionVerificationError(
                    f"{filename} {attribute} does not identify the bound transport event file"
                )
    except CompletionVerificationError:
        raise
    except Exception as exc:
        raise CompletionVerificationError(
            f"cannot inspect macropulse source-event references: {exc}"
        ) from exc


def write_macropulse_completion(
    output_dir: str | Path,
    transport_dir: str | Path,
    driver_arguments: Sequence[str],
    *,
    identity_tags: Sequence[str],
    expected_source_sha256: str | None = None,
) -> Path:
    output_path = Path(output_dir)
    marker_path = output_path / MACROPULSE_MARKER_NAME
    # A failed rewrite must never leave an older success claim in place.
    marker_path.unlink(missing_ok=True)
    expected_arguments = resolved_macropulse_arguments(driver_arguments)
    if Path(str(expected_arguments.get("output"))).resolve() != output_path.resolve():
        raise CompletionVerificationError("macropulse --output differs from marker output directory")
    if Path(str(expected_arguments.get("run_dir"))).resolve() != Path(transport_dir).resolve():
        raise CompletionVerificationError("macropulse --run-dir differs from source transport directory")
    source = _macropulse_source_identity(Path(transport_dir))
    source_sha256 = canonical_json_sha256(source)
    if expected_source_sha256 is not None:
        expected_source_sha256 = _require_sha256(
            expected_source_sha256, "expected macropulse source identity"
        )
        if source_sha256 != expected_source_sha256:
            raise CompletionVerificationError(
                "source transport changed while macropulse processing was running"
            )
    tags = _normalize_identity_tags(identity_tags)
    missing_tags = sorted(_REQUIRED_MACROPULSE_TAG_KEYS - set(tags))
    if missing_tags:
        raise CompletionVerificationError(
            f"macropulse identity tags are missing {missing_tags}"
        )
    _validate_provenance_tags(
        tags,
        label="macropulse identity tags",
        expected_stage="macropulse",
        require_rftrack=False,
    )
    for key in ("numpy_version", "scipy_version", "h5py_version"):
        if tags[key].strip().lower() in {"", "unknown", "unavailable", "unset", "none"}:
            raise CompletionVerificationError(
                f"macropulse identity tag {key} is unavailable"
            )
    tagged_source = tags.get("transport_source_sha256")
    if tagged_source is not None and _require_sha256(
        tagged_source, "macropulse transport_source_sha256"
    ) != source_sha256:
        raise CompletionVerificationError(
            "macropulse transport_source_sha256 tag disagrees with source transport"
        )

    outputs: dict[str, Any] = {}
    for filename in _MACROPULSE_REQUIRED_OUTPUTS:
        path = output_path / filename
        if not path.is_file():
            raise CompletionVerificationError(f"macropulse required output is missing: {path}")
        schema = None
        if filename == "back_bombardment_events.h5":
            schema = "back_bombardment_events_v2"
        elif filename == "back_bombardment_heat_source.h5":
            schema = "back_bombardment_heat_source_v1"
        elif filename == "back_bombardment_macropulse.h5":
            schema = "back_bombardment_macropulse_v1"
        outputs[Path(filename).stem] = _output_record(path, schema_version=schema)
    # Both JSON products must be readable before success is declared.
    _read_json_mapping(output_path / "study_config.json", "macropulse study config")
    _read_json_mapping(output_path / "study_results.json", "macropulse study results")
    _verify_macropulse_source_references(output_path, source)
    if bool(expected_arguments.get("save_figures")):
        expected_figures = (
            output_path / "figures" / "back_bombardment_source_qualification.png",
            output_path / "figures" / "back_bombardment_macropulse.png",
        )
        missing_figures = [str(path) for path in expected_figures if not path.is_file()]
        if missing_figures:
            raise CompletionVerificationError(
                f"requested macropulse figures are missing: {missing_figures}"
            )

    already_bound = {str(record["file"]) for record in outputs.values()}
    for path in sorted(output_path.rglob("*")):
        if not path.is_file() or path.name in {MACROPULSE_MARKER_NAME, "resource_usage.txt"}:
            continue
        relative = path.resolve().relative_to(output_path.resolve()).as_posix()
        if relative in already_bound:
            continue
        outputs[f"file:{relative}"] = _output_record(path, relative_to=output_path)

    payload = {
        "schema_version": MACROPULSE_MARKER_SCHEMA_VERSION,
        "stage": "macropulse",
        "status": "complete",
        "validation_status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request": {
            "arguments": expected_arguments,
            "arguments_sha256": canonical_json_sha256(expected_arguments),
        },
        "identity_tags": tags,
        "source_transport": source,
        "source_transport_sha256": source_sha256,
        "required_outputs": outputs,
    }
    _atomic_write_json(marker_path, payload)
    return marker_path


def verify_macropulse_completion(
    output_dir: str | Path,
    transport_dir: str | Path,
    driver_arguments: Sequence[str],
    *,
    identity_tags: Sequence[str],
) -> dict[str, Any]:
    output_path = Path(output_dir)
    marker_path = output_path / MACROPULSE_MARKER_NAME
    marker = _read_json_mapping(marker_path, "macropulse completion marker")
    if marker.get("schema_version") != MACROPULSE_MARKER_SCHEMA_VERSION:
        raise CompletionVerificationError(
            f"unsupported/legacy macropulse marker schema {marker.get('schema_version')!r}"
        )
    if marker.get("stage") != "macropulse" or marker.get("status") != "complete":
        raise CompletionVerificationError("macropulse marker stage/status is not macropulse/complete")
    if marker.get("validation_status") != "ok":
        raise CompletionVerificationError("macropulse marker validation_status is not 'ok'")

    expected_arguments = resolved_macropulse_arguments(driver_arguments)
    request = marker.get("request")
    if not isinstance(request, dict) or not isinstance(request.get("arguments"), dict):
        raise CompletionVerificationError("macropulse marker request identity is missing")
    difference = _first_difference(request["arguments"], expected_arguments, "macropulse_args")
    if difference is not None:
        raise CompletionVerificationError(f"macropulse request identity mismatch: {difference}")
    if request.get("arguments_sha256") != canonical_json_sha256(request["arguments"]):
        raise CompletionVerificationError("macropulse request digest mismatch")

    actual_tags = _normalize_identity_tags(marker.get("identity_tags", []))
    expected_tags = _normalize_identity_tags(identity_tags)
    for label, tags in (("stored", actual_tags), ("requested", expected_tags)):
        missing_tags = sorted(_REQUIRED_MACROPULSE_TAG_KEYS - set(tags))
        if missing_tags:
            raise CompletionVerificationError(
                f"{label} macropulse identity tags are missing {missing_tags}"
            )
        _validate_provenance_tags(
            tags,
            label=f"{label} macropulse identity tags",
            expected_stage="macropulse",
            require_rftrack=False,
        )
        for key in ("numpy_version", "scipy_version", "h5py_version"):
            if tags[key].strip().lower() in {"", "unknown", "unavailable", "unset", "none"}:
                raise CompletionVerificationError(
                    f"{label} macropulse identity tag {key} is unavailable"
                )
    if _stable_identity_tags(actual_tags) != _stable_identity_tags(expected_tags):
        difference = _first_difference(
            _stable_identity_tags(actual_tags),
            _stable_identity_tags(expected_tags),
            "macropulse_identity_tags",
        )
        raise CompletionVerificationError(f"macropulse identity mismatch: {difference}")
    for key in _EPHEMERAL_SCAN_TAG_KEYS:
        if key in expected_tags and key not in actual_tags:
            raise CompletionVerificationError(f"macropulse marker lacks scheduler tag {key!r}")

    expected_source = _macropulse_source_identity(Path(transport_dir))
    expected_source_sha256 = canonical_json_sha256(expected_source)
    for label, tags in (("stored", actual_tags), ("requested", expected_tags)):
        tagged_source = tags.get("transport_source_sha256")
        if tagged_source is not None:
            if _require_sha256(
                tagged_source, f"{label} transport_source_sha256"
            ) != expected_source_sha256:
                raise CompletionVerificationError(
                    f"{label} macropulse transport_source_sha256 disagrees with source transport"
                )
    if marker.get("source_transport_sha256") != expected_source_sha256:
        raise CompletionVerificationError("macropulse source transport digest mismatch")
    difference = _first_difference(marker.get("source_transport"), expected_source, "source_transport")
    if difference is not None:
        raise CompletionVerificationError(f"macropulse source transport changed: {difference}")
    checked = _verify_output_manifest(output_path, marker)
    for filename in _MACROPULSE_REQUIRED_OUTPUTS:
        key = Path(filename).stem
        if key not in checked:
            raise CompletionVerificationError(f"macropulse marker does not bind {filename}")
        if checked[key].get("file") != filename:
            raise CompletionVerificationError(
                f"macropulse output {key!r} must bind {filename!r}"
            )
    if bool(expected_arguments.get("save_figures")):
        bound_paths = {str(record.get("file", "")) for record in checked.values()}
        expected_figure_paths = {
            "figures/back_bombardment_source_qualification.png",
            "figures/back_bombardment_macropulse.png",
        }
        missing_figures = sorted(expected_figure_paths - bound_paths)
        if missing_figures:
            raise CompletionVerificationError(
                f"macropulse marker does not bind requested figures: {missing_figures}"
            )
    _read_json_mapping(output_path / "study_config.json", "macropulse study config")
    _read_json_mapping(output_path / "study_results.json", "macropulse study results")
    _verify_macropulse_source_references(output_path, expected_source)
    return {
        "status": "exact_match",
        "output_dir": str(output_path),
        "marker_sha256": sha256_file(marker_path),
        "source_transport": expected_source,
    }


def code_identity(repo: str | Path, *, launcher: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo).resolve()
    candidates: list[tuple[str, Path]] = []
    for filename in (
        "run_thermionic_tm010.py",
        "run_back_bombardment_macropulse.py",
        "verify_koa_run.py",
    ):
        path = root / filename
        if path.is_file():
            candidates.append((filename, path))
    package_dir = root / "rf_gun"
    if package_dir.is_dir():
        for path in sorted(package_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in _CODE_SUFFIXES:
                candidates.append((path.relative_to(root).as_posix(), path))
    if launcher is not None:
        launcher_path = Path(launcher)
        if not launcher_path.is_file():
            raise CompletionVerificationError(f"launcher file does not exist: {launcher_path}")
        candidates.append(("<slurm_launcher>", launcher_path))
    if not candidates:
        raise CompletionVerificationError(f"no simulation code files found below {root}")

    digest = hashlib.sha256()
    manifest: list[dict[str, Any]] = []
    for logical_name, path in candidates:
        file_digest = sha256_file(path)
        size = path.stat().st_size
        digest.update(logical_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
        manifest.append({"path": logical_name, "size_bytes": size, "sha256": file_digest})
    return {
        "schema_version": 1,
        "sha256": digest.hexdigest(),
        "file_count": len(manifest),
        "files": manifest,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    artifact_parser = subparsers.add_parser("verify-artifact")
    artifact_parser.add_argument("--artifact", type=Path, required=True)
    artifact_parser.add_argument(
        "--tail-policy", choices=("modal", "zero-control"), default="modal"
    )
    artifact_parser.add_argument("--expected-dr-um", type=float)
    artifact_parser.add_argument("--expected-dz-um", type=float)

    code_parser = subparsers.add_parser("code-identity")
    code_parser.add_argument("--repo", type=Path, required=True)
    code_parser.add_argument("--launcher", type=Path)
    code_parser.add_argument("--full-json", action="store_true")

    source_parser = subparsers.add_parser("source-transport-identity")
    source_parser.add_argument("--transport-dir", type=Path, required=True)

    stack_parser = subparsers.add_parser("python-stack-identity")
    stack_parser.add_argument("--full-json", action="store_true")

    transport_parser = subparsers.add_parser("verify-transport")
    transport_parser.add_argument("--run-dir", type=Path, required=True)
    transport_parser.add_argument("--artifact", type=Path, required=True)
    transport_parser.add_argument("driver_arguments", nargs=argparse.REMAINDER)

    for name in ("write-macropulse", "verify-macropulse"):
        macro_parser = subparsers.add_parser(name)
        macro_parser.add_argument("--output-dir", type=Path, required=True)
        macro_parser.add_argument("--transport-dir", type=Path, required=True)
        macro_parser.add_argument("--identity-tag", action="append", default=[])
        if name == "write-macropulse":
            macro_parser.add_argument("--expected-source-sha256")
        macro_parser.add_argument("driver_arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-artifact":
            result = inspect_artifact(
                args.artifact,
                tail_policy=args.tail_policy,
                expected_dr_um=args.expected_dr_um,
                expected_dz_um=args.expected_dz_um,
            )
            print(json.dumps(result, sort_keys=True))
        elif args.command == "code-identity":
            result = code_identity(args.repo, launcher=args.launcher)
            print(json.dumps(result, sort_keys=True) if args.full_json else result["sha256"])
        elif args.command == "verify-transport":
            result = verify_transport_completion(
                args.run_dir, args.artifact, args.driver_arguments
            )
            print(f"EXACT_MATCH: {result['run_dir']}")
        elif args.command == "source-transport-identity":
            print(macropulse_source_identity_sha256(args.transport_dir))
        elif args.command == "python-stack-identity":
            result = python_stack_identity()
            print(json.dumps(result, sort_keys=True) if args.full_json else result["sha256"])
        elif args.command == "write-macropulse":
            marker = write_macropulse_completion(
                args.output_dir,
                args.transport_dir,
                args.driver_arguments,
                identity_tags=args.identity_tag,
                expected_source_sha256=args.expected_source_sha256,
            )
            print(f"WROTE: {marker}")
        elif args.command == "verify-macropulse":
            result = verify_macropulse_completion(
                args.output_dir,
                args.transport_dir,
                args.driver_arguments,
                identity_tags=args.identity_tag,
            )
            print(f"EXACT_MATCH: {result['output_dir']}")
        else:  # pragma: no cover - argparse enforces the command set
            parser.error(f"unsupported command {args.command!r}")
    except CompletionVerificationError as exc:
        print(f"INVALID_OR_STALE: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
