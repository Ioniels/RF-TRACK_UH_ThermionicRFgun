#!/usr/bin/env python3
"""Prepare a compact, qualified XFdtd E/B artifact for RF-Track.

The 24 GB source is never loaded as one array.  Component-specific Yee grids
are sampled in bounded longitudinal slabs, all saved time frames are fitted at
one frequency/reference time, and Cartesian vectors are projected onto the
declared cylindrical m=0 production representation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

from rf_gun.fieldmaps.artifact import json_compatible

from rf_gun.aperture import L_END_MM, R_CAV_MM
from rf_gun.fieldmaps import (
    CAVITY_SOURCE_SHA256,
    HarmonicFitConfig,
    ReductionConfig,
    PRODUCTION_QUALITY_GATE_VERSION,
    apply_tail_policy,
    assess_reduced_field,
    check_xfdtd_cathode_origin,
    inspect_xfdtd_h5,
    plan_axisymmetric_grid,
    read_axisymmetric_artifact,
    reduce_xfdtd_to_axisymmetric,
    validate_xfdtd_h5,
    write_axisymmetric_artifact,
)


DEFAULT_SOURCE = Path("field_maps/cavity_E_H_full_volume.h5")
DEFAULT_ARTIFACT = Path("field_maps/cavity_axisymmetric_m0_rftrack.h5")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON summary path (default: OUTPUT with .report.json appended)",
    )
    parser.add_argument("--dr-um", type=float, default=100.0)
    parser.add_argument("--dz-um", type=float, default=50.0)
    parser.add_argument("--theta-count", type=int, default=32)
    parser.add_argument("--m-max", type=int, default=4)
    parser.add_argument("--longitudinal-block-points", type=int, default=64)
    parser.add_argument("--beam-core-radius-mm", type=float, default=2.0)
    parser.add_argument("--target-z-max-mm", type=float, default=L_END_MM)
    parser.add_argument("--r-max-mm", type=float, default=R_CAV_MM)
    parser.add_argument("--delta-cathode-chamfer-mm", type=float, default=0.0)
    parser.add_argument("--wall-margin-um", type=float, default=0.0)
    parser.add_argument("--wall-guard-cells", type=int, default=2)
    parser.add_argument("--axis-guard-cells", type=int, default=1)
    parser.add_argument("--tail-fit-start-mm", type=float, default=30.0)
    parser.add_argument(
        "--fill-correction",
        choices=("auto", "off", "single-pole"),
        default="auto",
        help=(
            "auto: preserve samples and mark a drifting export analysis-only; "
            "off: explicit raw-amplitude control; single-pole: provisional on-resonance "
            "step-drive extrapolation, requiring --fill-start-time-ns"
        ),
    )
    parser.add_argument(
        "--fill-start-time-ns", type=float, default=None,
        help="Assumed ideal step-drive start time, required only for --fill-correction single-pole",
    )
    parser.add_argument(
        "--tail-policy",
        choices=("modal", "zero", "none"),
        default="modal",
        help=(
            "modal: qualified below-cutoff exponential extension (production default); "
            "zero: explicit A/B control; none: stop at measured support"
        ),
    )
    parser.add_argument(
        "--also-write-zero-control",
        type=Path,
        default=None,
        help="Write a second explicit-zero-tail artifact without rereading the 24 GB source",
    )
    parser.add_argument(
        "--expected-source-sha256",
        default=CAVITY_SOURCE_SHA256,
        help="Expected whole-file SHA-256 from the supplier package",
    )
    parser.add_argument(
        "--verify-source-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream the 24 GB file once and verify the supplier checksum (default: true)",
    )
    parser.add_argument(
        "--source-digest-previously-confirmed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --no-verify-source-sha256, assert that this exact source digest was already "
            "stream-verified in a recorded prior run. Without this explicit assertion, an "
            "unhashed build cannot receive qualified status."
        ),
    )
    parser.add_argument(
        "--quality-status",
        choices=("auto", "qualified", "analysis_only"),
        default="auto",
        help=(
            "auto marks qualified only when all objective gates pass; qualified refuses to "
            "write if a gate fails; analysis_only always prevents production loading"
        ),
    )
    parser.add_argument("--max-logical-read-mib", type=float, default=512.0)
    parser.add_argument("--output-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect metadata and print the planned grid/read policy without field-data reads",
    )
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(json_compatible(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_manifest_record(manifest: Any) -> dict[str, Any]:
    components = {}
    for name, component in manifest.components.items():
        components[name] = {
            "shape": list(component.shape),
            "dtype": component.dtype,
            "chunks": None if component.chunks is None else list(component.chunks),
            "units": component.units,
        }
    return {
        "path": str(manifest.path),
        "size_bytes": int(manifest.file_size_bytes),
        "schema_version": int(manifest.schema_version),
        "solver": manifest.root_attributes.get("solver"),
        "source_run_id": manifest.root_attributes.get("source_run_id"),
        "frequency_hz": float(manifest.drive_frequency_hz),
        "source_power_w": float(manifest.source_power_w),
        "cathode_origin_native_m": manifest.cathode_origin_native_m.tolist(),
        "array_axes": list(manifest.array_axes),
        "electric_time_s": manifest.electric_time_s.tolist(),
        "magnetic_h_time_s": manifest.magnetic_h_time_s.tolist(),
        "components": components,
    }


def _progress(state: Mapping[str, Any]) -> None:
    stage = state.get("stage")
    if stage == "start":
        print(f"Reduction: {state['blocks']} longitudinal blocks", flush=True)
    elif stage == "component":
        print(
            f"  block {int(state['block']) + 1}: {state['component']} "
            f"z={float(state['z_start_m']) * 1e3:.3f}.."
            f"{float(state['z_end_m']) * 1e3:.3f} mm",
            flush=True,
        )
    elif stage == "block_complete":
        print(f"  block {int(state['block']) + 1} complete", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fill_correction == "single-pole" and args.fill_start_time_ns is None:
        raise ValueError("--fill-correction single-pole requires --fill-start-time-ns (an explicit model assumption)")
    if args.fill_start_time_ns is not None:
        if args.fill_correction != "single-pole":
            raise ValueError("--fill-start-time-ns is only valid with --fill-correction single-pole")
        if not np.isfinite(args.fill_start_time_ns) or args.fill_start_time_ns < 0.0:
            raise ValueError("--fill-start-time-ns must be finite and non-negative")
    started = time.time()
    source = args.source.resolve()
    output = args.output.resolve()
    report = (
        args.report.resolve()
        if args.report is not None
        else output.with_name(output.name + ".report.json")
    )
    zero_control = (
        args.also_write_zero_control.resolve()
        if args.also_write_zero_control is not None
        else None
    )
    paths = {
        "source": source,
        "output": output,
        "report": report,
    }
    if zero_control is not None:
        paths["zero_control"] = zero_control
    collisions = [
        (left_name, right_name, left_path)
        for index, (left_name, left_path) in enumerate(paths.items())
        for right_name, right_path in list(paths.items())[index + 1 :]
        if left_path == right_path
    ]
    if collisions:
        detail = ", ".join(
            f"{left} == {right} == {path}"
            for left, right, path in collisions
        )
        raise ValueError(
            "source, production artifact, report, and zero-tail control must use "
            f"distinct paths; refusing unsafe collision(s): {detail}"
        )
    # Keep every later write on the exact paths that were checked above.
    args.output = output
    args.report = report
    args.also_write_zero_control = zero_control
    if not source.is_file():
        raise FileNotFoundError(source)
    if not bool(args.dry_run) and not bool(args.overwrite):
        existing = [
            f"{name}={path}"
            for name, path in paths.items()
            if name != "source" and path.exists()
        ]
        if existing:
            raise FileExistsError(
                "refusing to spend time hashing/reducing before discovering existing "
                "destination(s); pass --overwrite deliberately: " + ", ".join(existing)
            )
    expected_sha = str(args.expected_source_sha256).strip().lower()
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("--expected-source-sha256 must be a 64-character hexadecimal digest")
    manifest = inspect_xfdtd_h5(source)
    if args.fill_start_time_ns is not None:
        reference = 0.5 * (
            min(float(manifest.electric_time_s[0]), float(manifest.magnetic_h_time_s[0]))
            + max(float(manifest.electric_time_s[-1]), float(manifest.magnetic_h_time_s[-1]))
        )
        if args.fill_start_time_ns * 1e-9 >= reference:
            raise ValueError("--fill-start-time-ns must precede the saved phasor reference")
    target_z_max_m = float(args.target_z_max_mm) * 1.0e-3
    plan = plan_axisymmetric_grid(
        str(source),
        target_z_max_m=target_z_max_m,
        requested_dr_m=float(args.dr_um) * 1.0e-6,
        requested_dz_m=float(args.dz_um) * 1.0e-6,
        r_max_m=float(args.r_max_mm) * 1.0e-3,
        delta_cathode_chamfer_mm=float(args.delta_cathode_chamfer_mm),
        wall_margin_m=float(args.wall_margin_um) * 1.0e-6,
    )
    theta_count = int(args.theta_count)
    if theta_count <= 0:
        raise ValueError("--theta-count must be positive")
    theta = 2.0 * np.pi * np.arange(theta_count) / theta_count
    reduction_config = ReductionConfig(
        r_m=plan.r_m,
        z_m=plan.measured_z_m,
        theta_rad=theta,
        vacuum_mask=plan.measured_vacuum_mask,
        harmonic=HarmonicFitConfig(
            manifest.drive_frequency_hz,
            include_offset=True,
            include_linear_envelope=True,
        ),
        longitudinal_block_points=int(args.longitudinal_block_points),
        wall_guard_cells=int(args.wall_guard_cells),
        beam_core_radius_m=float(args.beam_core_radius_mm) * 1.0e-3,
        m_max=int(args.m_max),
        output_dtype=str(args.output_dtype),
        max_logical_read_bytes=int(float(args.max_logical_read_mib) * 2**20),
    )
    if int(args.axis_guard_cells) < 0:
        raise ValueError("--axis-guard-cells must be non-negative")
    tail_fit_start_m = float(args.tail_fit_start_mm) * 1.0e-3
    if not np.isfinite(tail_fit_start_m):
        raise ValueError("--tail-fit-start-mm must be finite")
    tail_sample_count = int(np.count_nonzero(plan.measured_z_m >= tail_fit_start_m))
    if tail_sample_count < 3:
        raise ValueError(
            "--tail-fit-start-mm leaves fewer than three measured z planes for the tail fit"
        )
    extension_requested = str(args.tail_policy) != "none" or zero_control is not None
    if extension_requested and plan.missing_length_m <= 0.0:
        raise ValueError(
            "modal/zero tail extension requires the target endpoint to lie beyond measured support"
        )
    print(f"Source: {source} ({manifest.file_size_bytes / 1e9:.3f} GB)")
    print(
        f"Source RF: {manifest.drive_frequency_hz / 1e9:.9f} GHz at "
        f"{manifest.source_power_w / 1e6:.3f} MW; "
        f"{manifest.electric_time_s.size} E + {manifest.magnetic_h_time_s.size} H frames"
    )
    print(
        f"Grid: Nr={plan.r_m.size}, measured Nz={plan.measured_z_m.size}, "
        f"full Nz={plan.full_z_m.size}; dr={plan.dr_m * 1e6:.3f} um, "
        f"dz={plan.dz_m * 1e6:.3f} um"
    )
    print(
        f"Measured common support ends at {plan.source_support.z_max_m * 1e3:.6f} mm; "
        f"last aligned sample={plan.measured_z_max_m * 1e3:.6f} mm; "
        f"tracking end={plan.target_z_max_m * 1e3:.6f} mm; "
        f"modeled interval={plan.missing_length_m * 1e3:.6f} mm"
    )
    if args.dry_run:
        print(
            "Dry run: configuration validated; no 4-D field dataset was read and no "
            "artifact was written."
        )
        return 0

    validation = validate_xfdtd_h5(
        source,
        expected_sha256=expected_sha if bool(args.verify_source_sha256) else None,
    )
    validation.raise_for_errors()
    for warning in validation.warnings:
        print(f"Source warning [{warning.code}]: {warning.message}", file=sys.stderr)
    if bool(args.verify_source_sha256):
        print(f"Source SHA-256 verified: {expected_sha}")
        checksum_status = "verified_against_supplier_manifest"
    else:
        print("Warning: source SHA-256 was declared but not recomputed for this run.", file=sys.stderr)
        checksum_status = "declared_not_recomputed"

    cathode_origin = check_xfdtd_cathode_origin(source)
    cathode_origin_report = cathode_origin.as_dict()
    transition = cathode_origin_report["transition"]

    def format_optional(value: Any, scale: float, spec: str) -> str:
        return "n/a" if value is None else format(float(value) * scale, spec)

    print(
        "Cathode-origin physical check: "
        f"{cathode_origin_report['status']}; material z="
        f"{format_optional(transition['material_sample_beam_z_m'], 1e6, '+.3f')} um, "
        f"vacuum z={format_optional(transition['vacuum_sample_beam_z_m'], 1e6, '+.3f')} um, "
        f"midpoint={format_optional(transition['midpoint_beam_z_m'], 1e6, '+.3f')} um, "
        f"material/vacuum |E|="
        f"{format_optional(transition['field_suppression_ratio'], 1.0, '.3e')}"
    )

    reduction = reduce_xfdtd_to_axisymmetric(source, reduction_config, progress=_progress)

    assessment = assess_reduced_field(
        reduction, plan,
        simulated_duration_s=float(manifest.root_attributes["simulation_duration_s"]),
        cathode_origin_report=cathode_origin_report,
        source_integrity_gate={
            "passed": bool(args.verify_source_sha256 or args.source_digest_previously_confirmed),
            "metric": "schema valid and source digest verified now or explicitly confirmed previously",
            "verification_status": checksum_status,
            "previously_confirmed": bool(args.source_digest_previously_confirmed),
            "gate_version": PRODUCTION_QUALITY_GATE_VERSION,
            "name": "source_integrity",
        },
        fill_policy=str(args.fill_correction),
        drive_start_time_s=None if args.fill_start_time_ns is None else args.fill_start_time_ns * 1e-9,
        tail_policy=str(args.tail_policy),
        tail_fit_start_m=tail_fit_start_m,
        wall_guard_cells=int(args.wall_guard_cells),
        axis_guard_cells=int(args.axis_guard_cells),
        beam_core_radius_m=float(args.beam_core_radius_mm) * 1e-3,
        requested_quality_status=str(args.quality_status),
    )
    working_field, tail, quality = assessment.measured_field, assessment.tail, assessment.quality
    fill_record = quality["fill_transient"]
    status, failed = quality["status"], quality["failed_gates"]
    print(
        f"Fill treatment: {fill_record['status']}; policy={args.fill_correction}, "
        f"applied scale={fill_record['applied_correction_factor']:.6f}, "
        f"phasor reference={fill_record['reference_time_s'] * 1e9:.6f} ns. "
        "Steady-state amplitude and source-power convention remain unverified."
    )
    print(
        "Tail at measured endpoint: "
        f"E={tail.metrics['endpoint_e_fraction_of_map_peak']:.4%}, "
        f"B={tail.metrics['endpoint_b_fraction_of_map_peak']:.4%}, "
        f"energy/length={tail.metrics['endpoint_energy_per_length_fraction_of_peak']:.3e}; "
        f"decay length={tail.metrics['fitted_decay_length_m'] * 1e3:.6f} mm "
        f"(TM01={tail.metrics['analytic_tm01_decay_length_m'] * 1e3:.6f} mm), "
        f"missing voltage={tail.metrics['modal_tail_integrated_voltage_V'] / 1e3:.6f} kV"
    )
    print(
        "Symmetry diagnostic (discarded RMS amplitude): "
        f"E={reduction.report.symmetry['combined_e_nonaxisymmetric_fraction']:.3%}, "
        f"B={reduction.report.symmetry['combined_b_nonaxisymmetric_fraction']:.3%}. "
        "Production representation remains cylindrical vector m=0 by declared policy."
    )
    print(f"Artifact quality status: {status}; failed gates: {failed or 'none'}")

    selected_field = apply_tail_policy(
        working_field, plan, tail, policy=str(args.tail_policy)
    )
    selected_mask = (
        plan.measured_vacuum_mask if args.tail_policy == "none" else plan.full_vacuum_mask
    )
    selected_aperture = (
        plan.measured_aperture_radius_m
        if args.tail_policy == "none"
        else plan.full_aperture_radius_m
    )
    provenance = {
        "source_sha256": expected_sha,
        "source_sha256_status": checksum_status,
        "source_digest_previously_confirmed": bool(args.source_digest_previously_confirmed),
        "source_path": str(source),
        "source_file_size_bytes": int(manifest.file_size_bytes),
        "source_manifest": _source_manifest_record(manifest),
        "processing_config": {
            "reduction": reduction_config.as_metadata(),
            "grid_plan": plan.as_metadata(),
            "tail_policy": str(args.tail_policy),
            "tail_fit_start_m": tail_fit_start_m,
            "quality_gate_version": PRODUCTION_QUALITY_GATE_VERSION,
            "fill_treatment": fill_record,
        },
        "created_by": "prepare_xfdtd_fieldmap.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload_sha = write_axisymmetric_artifact(
        args.output,
        selected_field,
        transform=reduction_config.transform,
        quality=quality,
        provenance=provenance,
        vacuum_mask=selected_mask,
        aperture_radius_m=selected_aperture,
        overwrite=bool(args.overwrite),
    )
    verified = read_axisymmetric_artifact(
        args.output,
        verify="payload",
        require_qualified=status == "qualified",
    )
    if verified.payload_sha256 != payload_sha:
        raise RuntimeError("artifact digest changed between write and verification")
    print(f"Wrote and payload-verified: {args.output.resolve()}")
    print(f"Artifact payload SHA-256: {payload_sha}")

    zero_control_record = None
    if args.also_write_zero_control is not None:
        zero_field = apply_tail_policy(working_field, plan, tail, policy="zero")
        zero_quality = dict(quality)
        zero_quality["variant"] = "zero_tail_control"
        zero_provenance = dict(provenance)
        zero_provenance["processing_config"] = dict(provenance["processing_config"])
        zero_provenance["processing_config"]["tail_policy"] = "zero"
        zero_payload = write_axisymmetric_artifact(
            args.also_write_zero_control,
            zero_field,
            transform=reduction_config.transform,
            quality=zero_quality,
            provenance=zero_provenance,
            vacuum_mask=plan.full_vacuum_mask,
            aperture_radius_m=plan.full_aperture_radius_m,
            overwrite=bool(args.overwrite),
        )
        read_axisymmetric_artifact(
            args.also_write_zero_control,
            verify="payload",
            require_qualified=status == "qualified",
        )
        zero_control_record = {
            "path": str(args.also_write_zero_control.resolve()),
            "payload_sha256": zero_payload,
        }
        print(f"Wrote zero-tail control: {args.also_write_zero_control.resolve()}")

    report_path = args.report
    _atomic_json(
        report_path,
        {
            "status": status,
            "artifact_path": str(args.output.resolve()),
            "artifact_payload_sha256": payload_sha,
            "zero_control": zero_control_record,
            "source": _source_manifest_record(manifest),
            "grid_plan": plan.as_metadata(),
            "quality": quality,
            "elapsed_s": float(time.time() - started),
        },
    )
    print(f"Wrote report: {report_path}")
    return 0 if status == "qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
