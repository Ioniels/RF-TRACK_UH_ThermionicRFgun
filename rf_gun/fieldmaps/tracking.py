"""Validated field-artifact handoff shared by notebooks and tracking drivers.

Loading, geometry checks and power scaling do not require RF-Track itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np

from ..provenance import sha256_file


def load_qualified_artifact_for_tracking(
    path: Path,
    *,
    magnetic_field_mode: str,
    requested_z_min_m: float,
    requested_z_max_m: float,
    target_power_w: float,
    field_tail_policy: str = "modal",
    requested_aperture_delta_mm: float | None = None,
) -> dict[str, Any]:
    """Load and validate the immutable field input used by an artifact-backed run.

    This helper deliberately has no RF-Track dependency, so artifact integrity and domain
    compatibility can be tested before launching a costly simulation. The artifact owns its
    frequency, coordinates, spacing, and relative E/B phasors; all four components are scaled
    together from its recorded solver power to ``target_power_w``. The requested tracking
    interval is a separate window which must fit wholly inside that immutable map support.
    """

    from rf_gun.fieldmaps import (
        PRODUCTION_QUALITY_GATE_VERSION,
        REQUIRED_PRODUCTION_GATE_NAMES,
        read_axisymmetric_artifact,
    )

    mode = str(magnetic_field_mode).strip().lower()
    if mode not in {"artifact", "zero"}:
        raise ValueError("magnetic_field_mode must be 'artifact' or 'zero'")
    z_min = float(requested_z_min_m)
    z_max = float(requested_z_max_m)
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        raise ValueError(
            f"tracking domain must be finite with z_max > z_min, got [{z_min!r}, {z_max!r}] m"
        )

    artifact_path = Path(path)
    artifact = read_axisymmetric_artifact(
        artifact_path,
        verify="payload",
        require_qualified=True,
    )
    quality_gates = artifact.quality.get("gates")
    if not isinstance(quality_gates, dict):
        raise ValueError("qualified field artifact has no production quality-gate mapping")
    required_gates = set(REQUIRED_PRODUCTION_GATE_NAMES) | {"source_integrity"}
    missing_gates = sorted(required_gates - set(quality_gates))
    if missing_gates:
        raise ValueError(
            "qualified field artifact is missing current production gates: "
            + ", ".join(missing_gates)
        )
    stale_gates = sorted(
        name
        for name in required_gates
        if quality_gates[name].get("gate_version") != PRODUCTION_QUALITY_GATE_VERSION
    )
    if stale_gates:
        raise ValueError(
            f"field artifact was not qualified with {PRODUCTION_QUALITY_GATE_VERSION}: "
            + ", ".join(stale_gates)
        )
    requested_tail_policy = str(field_tail_policy).strip().lower()
    expected_variant = {
        "modal": "modal",
        "zero-control": "zero_tail_control",
    }.get(requested_tail_policy)
    if expected_variant is None:
        raise ValueError("field_tail_policy must be 'modal' or 'zero-control'")
    artifact_variant = str(artifact.quality.get("variant", ""))
    if artifact_variant != expected_variant:
        raise ValueError(
            "field artifact variant does not match the explicitly requested tail policy: "
            f"artifact={artifact_variant!r}, requested={requested_tail_policy!r} "
            f"(expected variant {expected_variant!r})"
        )
    field = artifact.field
    fill_record = field.metadata.get("fill_treatment", artifact.quality.get("fill_transient", {}))
    if fill_record and not fill_record.get("steady_state_verified", False):
        if field.metadata.get("fill_correction_applied", False):
            factor = field.metadata.get("fill_correction", {}).get("correction_factor", "unknown")
            detail = f"an unverified single-pole fill estimate (field factor {factor})"
        elif not fill_record.get("locally_stationary_window", fill_record.get("converged_export", False)):
            detail = "an uncorrected filling-transient control"
        else:
            detail = "a locally stationary saved window without independent steady-state calibration"
        warnings.warn(
            f"Field amplitude uses {detail}. Relative power scaling does not validate "
            "the steady-state amplitude or the supplier's port-power convention. "
            "See README.md, section Field maps from XFdtd.",
            RuntimeWarning, stacklevel=2,
        )
    source_power_raw = field.metadata.get("source_power_w")
    try:
        source_power_w = float(source_power_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "qualified field artifact metadata must contain a finite positive source_power_w"
        ) from exc
    requested_power_w = float(target_power_w)
    if not np.isfinite(source_power_w) or source_power_w <= 0.0:
        raise ValueError(
            "qualified field artifact metadata must contain a finite positive source_power_w"
        )
    if not np.isfinite(requested_power_w) or requested_power_w <= 0.0:
        raise ValueError("target_power_w must be finite and positive for RF tracking")
    source_manifest = artifact.provenance.get("source_manifest")
    if isinstance(source_manifest, dict) and "source_power_w" in source_manifest:
        manifest_power_w = float(source_manifest["source_power_w"])
        if not np.isclose(manifest_power_w, source_power_w, rtol=1.0e-12, atol=0.0):
            raise ValueError(
                "artifact field metadata and source manifest disagree on source_power_w: "
                f"{source_power_w:.12g} != {manifest_power_w:.12g} W"
            )
    support_min = float(field.z_m[0])
    support_max = float(field.z_m[-1])
    tolerance = max(1.0e-12, abs(field.hz_m) * 1.0e-8)
    if z_min < support_min - tolerance or z_max > support_max + tolerance:
        raise ValueError(
            "requested tracking domain is outside the qualified field artifact support: "
            f"requested=[{z_min:.12g}, {z_max:.12g}] m, "
            f"artifact=[{support_min:.12g}, {support_max:.12g}] m"
        )
    if requested_aperture_delta_mm is not None:
        if artifact.aperture_radius_m is None:
            raise ValueError(
                "qualified artifact has no stored aperture-radius profile, so its geometry "
                "cannot be checked against --delta_cathode_chamfer_mm"
            )
        from rf_gun.aperture import aperture_radius_profile_mm

        expected_aperture = aperture_radius_profile_mm(
            field.z_m * 1.0e3, float(requested_aperture_delta_mm)
        ) * 1.0e-3
        stored_aperture = np.asarray(artifact.aperture_radius_m, dtype=float)
        if stored_aperture.ndim == 2:
            # The artifact contract permits a full grid, although the project writer stores 1-D.
            stored_aperture = np.max(
                np.where(np.asarray(artifact.vacuum_mask, dtype=bool), field.r_m[None, :], 0.0),
                axis=1,
            )
        if stored_aperture.shape != expected_aperture.shape or not np.allclose(
            stored_aperture, expected_aperture, rtol=0.0, atol=2.0e-9
        ):
            raise ValueError(
                "artifact aperture geometry does not match the requested cathode/chamfer "
                f"offset ({float(requested_aperture_delta_mm):.6g} mm); prepare a matching "
                "field artifact rather than shifting only the tracking aperture"
            )

    # RF_FieldMap_2d and the dynamic aperture both infer their physical z grid from the supplied
    # array length and VolumeBuildParams interval.  Therefore a requested sub-domain must be an
    # exact contiguous slice of the artifact grid; passing the full array with a shorter z_max
    # would silently compress/misalign the aperture profile.
    start_index = int(np.argmin(np.abs(field.z_m - z_min)))
    stop_index = int(np.argmin(np.abs(field.z_m - z_max)))
    alignment_tolerance = max(1.0e-12, abs(field.hz_m) * 1.0e-7)
    if (
        abs(float(field.z_m[start_index]) - z_min) > alignment_tolerance
        or abs(float(field.z_m[stop_index]) - z_max) > alignment_tolerance
    ):
        raise ValueError(
            "requested tracking-domain endpoints must coincide with artifact z-grid planes: "
            f"requested=[{z_min:.12g}, {z_max:.12g}] m, nearest="
            f"[{float(field.z_m[start_index]):.12g}, {float(field.z_m[stop_index]):.12g}] m"
        )
    if stop_index <= start_index:
        raise ValueError("requested tracking domain selects fewer than two artifact z planes")

    from rf_gun.fieldmaps import AxisymmetricRFField

    selected = slice(start_index, stop_index + 1)
    selected_metadata = dict(field.metadata)
    selected_metadata["tracking_slice"] = {
        "source_z_start_index": start_index,
        "source_z_stop_index_inclusive": stop_index,
        "requested_z_min_m": z_min,
        "requested_z_max_m": z_max,
    }
    tracking_field = AxisymmetricRFField(
        frequency_hz=field.frequency_hz,
        r_m=field.r_m,
        z_m=field.z_m[selected],
        Er_Vpm=field.Er_Vpm[selected],
        Ez_Vpm=field.Ez_Vpm[selected],
        Btheta_T=field.Btheta_T[selected],
        Bz_T=field.Bz_T[selected],
        metadata=selected_metadata,
    ).scaled_to_power(requested_power_w, source_power_w)
    # Production artifacts must contain the cathode plane because phase and emission are tied to
    # its on-axis Ez phasor. Accessing the property performs the grid-point tolerance check.
    ez0_phasor_axis = tracking_field.Ez0_phasor_axis
    use_magnetic_field = mode == "artifact"
    return {
        "artifact": artifact,
        "field": tracking_field,
        "Er_grid": tracking_field.Er_Vpm,
        "Ez_grid": tracking_field.Ez_Vpm,
        "Bt_grid": tracking_field.Btheta_T if use_magnetic_field else None,
        "Bz_grid": tracking_field.Bz_T if use_magnetic_field else None,
        "Ez0_phasor_axis": ez0_phasor_axis,
        "magnetic_field_enabled": use_magnetic_field,
        "tracking_domain_m": [z_min, z_max],
        "artifact_support_m": [support_min, support_max],
        "artifact_z_slice": [start_index, stop_index],
        "payload_verification": "payload",
        "source_power_w": source_power_w,
        "target_power_w": requested_power_w,
        "common_field_scale": float(np.sqrt(requested_power_w / source_power_w)),
        "field_tail_policy": requested_tail_policy,
        "artifact_variant": artifact_variant,
        "quality_gate_version": PRODUCTION_QUALITY_GATE_VERSION,
    }


def artifact_tracking_provenance(path: Path, loaded: dict[str, Any]) -> dict[str, Any]:
    """Compact, explicit provenance for run_config.json and completion markers."""

    artifact = loaded["artifact"]
    provenance = dict(artifact.provenance)
    return {
        "path": str(Path(path).resolve()),
        "file_sha256": sha256_file(Path(path)),
        "payload_sha256": str(artifact.payload_sha256),
        "payload_verification": str(loaded["payload_verification"]),
        "quality_status": str(artifact.quality.get("status", "")),
        "source_sha256": provenance.get("source_sha256"),
        "processing_config_sha256": provenance.get("processing_config_sha256"),
        "processing_config": provenance.get("processing_config"),
        "artifact_provenance": provenance,
        "artifact_field_metadata": dict(artifact.field.metadata),
        "artifact_support_m": list(loaded["artifact_support_m"]),
        "tracking_domain_m": list(loaded["tracking_domain_m"]),
        "power_normalization": {
            "source_power_w": float(loaded["source_power_w"]),
            "target_power_w": float(loaded["target_power_w"]),
            "common_field_scale": float(loaded["common_field_scale"]),
            "policy": "sqrt(target/source) applied identically to E and B",
        },
        "field_tail_policy": str(loaded["field_tail_policy"]),
        "artifact_variant": str(loaded["artifact_variant"]),
    }
