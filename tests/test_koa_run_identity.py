from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import verify_koa_run as koa
from rf_gun.aperture import aperture_radius_profile_mm
from rf_gun.fieldmaps import (
    PRODUCTION_QUALITY_GATE_VERSION,
    REQUIRED_PRODUCTION_GATE_NAMES,
    AxisymmetricRFField,
    FrameTransform,
    write_axisymmetric_artifact,
)


def _write_artifact(
    path: Path,
    *,
    dr_m: float = 1.0e-3,
    dz_m: float = 1.0e-2,
    requested_dr_m: float | None = None,
    requested_dz_m: float | None = None,
) -> None:
    r_m = np.arange(3, dtype=float) * dr_m
    z_m = np.arange(3, dtype=float) * dz_m
    shape = (z_m.size, r_m.size)
    field = AxisymmetricRFField(
        frequency_hz=2.865e9,
        r_m=r_m,
        z_m=z_m,
        Er_Vpm=np.full(shape, 2.0e3 + 3.0e2j),
        Ez_Vpm=np.full(shape, 3.0e6 + 4.0e5j),
        Btheta_T=np.full(shape, 4.0e-3 + 2.0e-4j),
        Bz_T=np.full(shape, 5.0e-5 + 1.0e-5j),
        metadata={"source_solver": "Remcom XFdtd", "source_power_w": 1.8e6},
    )
    write_axisymmetric_artifact(
        path,
        field,
        transform=FrameTransform(np.zeros(3), np.eye(3)),
        quality={
            "status": "qualified",
            "variant": "modal",
            "gates": {
                name: {
                    "passed": True,
                    "gate_version": PRODUCTION_QUALITY_GATE_VERSION,
                }
                for name in set(REQUIRED_PRODUCTION_GATE_NAMES) | {"source_integrity"}
            },
        },
        provenance={
            "source_sha256": "a" * 64,
            "processing_config": {
                "projection": "m0",
                "tail_policy": "modal",
                **(
                    {
                        "grid_plan": {
                            "requested_dr_m": requested_dr_m,
                            "requested_dz_m": requested_dz_m,
                            "realized_dr_m": dr_m,
                            "realized_dz_m": dz_m,
                        }
                    }
                    if requested_dr_m is not None and requested_dz_m is not None
                    else {}
                ),
            },
            "source_manifest": {"source_power_w": 1.8e6},
        },
        aperture_radius_m=aperture_radius_profile_mm(z_m * 1.0e3, 0.0) * 1.0e-3,
    )


def _transport_args(
    artifact: Path,
    run_dir: Path,
    *,
    particles: int = 1234,
    job_id: str = "100",
) -> list[str]:
    return [
        "--field-artifact",
        str(artifact),
        "--rf-magnetic-field",
        "artifact",
        "--field-tail-policy",
        "modal",
        "--rf-forward-power-w",
        "1000000",
        "--z_max",
        "0.02",
        "--n_particles",
        str(particles),
        "--seed",
        "42",
        "--t_cathode_k",
        "1650",
        "--phi_eff_ev",
        "2.66",
        "--work-function-temperature-model",
        "linear_tcwf",
        "--finesse",
        "fine",
        "--sc_enabled",
        "--mirror-charges",
        "--beam_loading",
        "--cathode_backstop_enabled",
        "--emission-field-iteration",
        "--use-converged-iteration-source",
        "--n_screens",
        "3",
        "--save-screen-hdf5",
        "--save-openpmd-beam",
        "--save-figures",
        "--no-deflection_enabled",
        "--run-family",
        "identity_test",
        "--scan-tags",
        "campaign=test",
        "study=test",
        "stage=transport",
        "task_id=0",
        "git_commit=" + "1" * 40,
        "git_state=clean",
        "code_sha256=" + "2" * 64,
        "python_stack_sha256=" + "4" * 64,
        "rftrack_version=2.7.0",
        f"slurm_job_id={job_id}",
        "slurm_array_job_id=90",
        "slurm_array_task_id=0",
        "--output",
        str(run_dir),
    ]


def _write_h5_schema(
    path: Path, schema: str, *, attributes: dict[str, object] | None = None
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = schema
        for key, value in (attributes or {}).items():
            handle.attrs[key] = value
        handle.create_dataset("payload", data=np.arange(3))


def _write_fake_completed_transport(
    artifact_path: Path, run_dir: Path, driver_args: list[str]
) -> None:
    run_dir.mkdir(parents=True)
    resolved = koa.resolved_transport_arguments(driver_args)
    artifact = koa.inspect_artifact(artifact_path, tail_policy="modal")
    target_power = float(resolved["bl_p_fwd_w"])
    transport_phase_deg = 42.0
    canonical_resolved = {
        "frequency_hz": float(artifact["frequency_hz"]),
        "grid_shape": [int(artifact["nz"]), int(artifact["nr"])],
        "hr_m": float(artifact["dr_um"]) * 1.0e-6,
        "hz_m": float(artifact["dz_um"]) * 1.0e-6,
        "z_min_m": float(artifact["z_min_m"]),
        "z_max_m": float(artifact["z_max_m"]),
        "field_map_z0_m": float(artifact["z_min_m"]),
        "field_support_z_m": [
            float(artifact["z_min_m"]),
            float(artifact["z_max_m"]),
        ],
        "rf_magnetic_field_enabled": True,
        "rf_forward_power_w": target_power,
        "transport_phase_deg": transport_phase_deg,
        "phasor_mode_effective": "artifact",
    }
    run_config = {
        "schema_version": 1,
        "hardcoded_parameters": {
            "run_identity": {
                "run_family": resolved["run_family"],
                "scan_tags": resolved["scan_tags"],
                "args": resolved,
            },
            "cavity": {
                "f_hz": canonical_resolved["frequency_hz"],
                "rf_forward_power_w": target_power,
                "phasor_mode": "artifact",
            },
            "field_provenance": {
                "grid": {
                    "nz": canonical_resolved["grid_shape"][0],
                    "nr": canonical_resolved["grid_shape"][1],
                    "realized_hr_m": canonical_resolved["hr_m"],
                    "realized_hz_m": canonical_resolved["hz_m"],
                    "z_support_m": canonical_resolved["field_support_z_m"],
                },
                "magnetic_field_source": {"enabled": True},
            },
            "runtime_environment": {
                "rftrack": {
                    "version": "2.7.0",
                    "module_files": {
                        "compiled_extension": {
                            "path": "/synthetic/_RF_Track.so",
                            "sha256": "5" * 64,
                        }
                    },
                }
            },
        },
        "derived_parameters": {
            "z_min_m": canonical_resolved["z_min_m"],
            "z_max_m": canonical_resolved["z_max_m"],
            "transport_phase_deg": transport_phase_deg,
        },
    }
    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config), encoding="utf-8")
    results_path = run_dir / "run_results.json"
    results_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    validation_path = run_dir / "validation.json"
    validation_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    event_path = run_dir / "back_bombardment_events.h5"
    _write_h5_schema(event_path, "back_bombardment_events_v2")
    screens_dir = run_dir / "screen_distributions_hdf5"
    screens_dir.mkdir()
    required_figure_names = (
        "field_maps.png",
        "on_axis_field_profile.png",
        "screen_spectra.png",
        "emission_history.png",
        "initial_phase_space_x_px.png",
        "beam_moments_evolution.png",
        "beam_twiss_evolution.png",
        "emission_iteration_convergence.png",
        "emission_iteration_waveforms.png",
        "emission_iteration_near_cathode.png",
    )
    auxiliary_paths = [
        screens_dir / "B0_test.h5",
        screens_dir / "Bout_test.h5",
        *(screens_dir / f"screen_{index:03d}.h5" for index in range(3)),
        *(run_dir / "figures" / name for name in required_figure_names),
        run_dir / "lost_particle_diagnostics.json",
        run_dir / "emission_iteration.npz",
    ]
    for path in auxiliary_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic:{path.name}".encode("utf-8"))

    field_identity = {
        "kind": "qualified_field_artifact",
        "path": artifact["path"],
        "file_sha256": artifact["file_sha256"],
        "payload_sha256": artifact["payload_sha256"],
        "payload_verification": "payload",
        "source_sha256": artifact["source_sha256"],
        "processing_config_sha256": artifact["processing_config_sha256"],
        "rf_magnetic_field_mode": "artifact",
        "rf_magnetic_field_enabled": True,
        "source_power_w": artifact["source_power_w"],
        "target_power_w": target_power,
        "common_field_scale": np.sqrt(target_power / artifact["source_power_w"]),
        "field_tail_policy": "modal",
        "artifact_variant": "modal",
    }
    required_outputs = {
        "run_config": {
            "file": config_path.name,
            "sha256": koa.sha256_file(config_path),
        },
        "run_results": {
            "file": results_path.name,
            "sha256": koa.sha256_file(results_path),
        },
        "validation": {
            "file": validation_path.name,
            "sha256": koa.sha256_file(validation_path),
        },
        "back_bombardment_events": {
            "file": event_path.name,
            "sha256": koa.sha256_file(event_path),
            "schema_version": "back_bombardment_events_v2",
        },
    }
    for path in auxiliary_paths:
        relative = path.relative_to(run_dir).as_posix()
        required_outputs[f"file:{relative}"] = {
            "file": relative,
            "sha256": koa.sha256_file(path),
        }
    canonical_config = {
        "arguments": {key: value for key, value in resolved.items() if key != "output"},
        "resolved": canonical_resolved,
    }
    marker = {
        "schema_version": koa.TRANSPORT_MARKER_SCHEMA_VERSION,
        "stage": "transport",
        "status": "complete",
        "validation_status": "ok",
        "validation_file": validation_path.name,
        "run_config_file": config_path.name,
        "run_config_sha256": koa.sha256_file(config_path),
        "canonical_config": canonical_config,
        "canonical_config_sha256": koa.canonical_json_sha256(canonical_config),
        "run_family": resolved["run_family"],
        "scan_tags": resolved["scan_tags"],
        "field_artifact_sha256": artifact["file_sha256"],
        "field_artifact_payload_sha256": artifact["payload_sha256"],
        "field_input_sha256": koa.canonical_json_sha256(field_identity),
        "field_identity": field_identity,
        "required_outputs": required_outputs,
    }
    (run_dir / koa.TRANSPORT_MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")


def test_artifact_grid_verifier_uses_actual_spacing(tmp_path: Path):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact, dr_m=4.0e-6, dz_m=13.0e-6)
    info = koa.inspect_artifact(
        artifact, tail_policy="modal", expected_dr_um=4.0, expected_dz_um=13.0
    )
    assert info["dr_um"] == pytest.approx(4.0)
    assert info["dz_um"] == pytest.approx(13.0)
    with pytest.raises(koa.CompletionVerificationError, match="does not match"):
        koa.inspect_artifact(
            artifact, tail_policy="modal", expected_dr_um=5.0, expected_dz_um=13.0
        )


def test_artifact_grid_verifier_distinguishes_nominal_and_endpoint_fitted_spacing(
    tmp_path: Path,
):
    artifact = tmp_path / "endpoint_fitted.h5"
    _write_artifact(
        artifact,
        dr_m=3.99982e-6,
        dz_m=13.0003e-6,
        requested_dr_m=4.0e-6,
        requested_dz_m=13.0e-6,
    )
    info = koa.inspect_artifact(
        artifact, tail_policy="modal", expected_dr_um=4.0, expected_dz_um=13.0
    )
    assert info["dr_um"] == pytest.approx(3.99982)
    assert info["dz_um"] == pytest.approx(13.0003)


@pytest.mark.parametrize("marker_text", ["2026-09-11T00:00:00Z\n", "", "{broken"])
def test_transport_never_accepts_plain_empty_or_malformed_marker(
    tmp_path: Path, marker_text: str
):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / koa.TRANSPORT_MARKER_NAME).write_text(marker_text, encoding="utf-8")
    with pytest.raises(koa.CompletionVerificationError):
        koa.verify_transport_completion(run_dir, artifact, _transport_args(artifact, run_dir))


def test_transport_exact_match_allows_new_scheduler_job_id(tmp_path: Path):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    run_dir = tmp_path / "run"
    original_args = _transport_args(artifact, run_dir, job_id="100")
    _write_fake_completed_transport(artifact, run_dir, original_args)
    resumed_args = _transport_args(artifact, run_dir, job_id="200")
    result = koa.verify_transport_completion(run_dir, artifact, resumed_args)
    assert result["status"] == "exact_match"


def test_transport_rejects_changed_critical_argument(tmp_path: Path):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    run_dir = tmp_path / "run"
    args = _transport_args(artifact, run_dir)
    _write_fake_completed_transport(artifact, run_dir, args)
    with pytest.raises(koa.CompletionVerificationError, match="n_particles"):
        koa.verify_transport_completion(
            run_dir, artifact, _transport_args(artifact, run_dir, particles=4321)
        )


def test_transport_rejects_tampered_required_output(tmp_path: Path):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    run_dir = tmp_path / "run"
    args = _transport_args(artifact, run_dir)
    _write_fake_completed_transport(artifact, run_dir, args)
    (run_dir / "run_results.json").write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(koa.CompletionVerificationError, match="digest mismatch"):
        koa.verify_transport_completion(run_dir, artifact, args)


def test_transport_rejects_coherently_rehashed_false_resolved_grid_identity(
    tmp_path: Path,
):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    run_dir = tmp_path / "run"
    args = _transport_args(artifact, run_dir)
    _write_fake_completed_transport(artifact, run_dir, args)

    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["hardcoded_parameters"]["cavity"]["f_hz"] = 1.234e9
    config_path.write_text(json.dumps(config), encoding="utf-8")
    marker_path = run_dir / koa.TRANSPORT_MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["canonical_config"]["resolved"]["frequency_hz"] = 1.234e9
    marker["canonical_config_sha256"] = koa.canonical_json_sha256(
        marker["canonical_config"]
    )
    config_sha = koa.sha256_file(config_path)
    marker["run_config_sha256"] = config_sha
    marker["required_outputs"]["run_config"]["sha256"] = config_sha
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(koa.CompletionVerificationError, match="artifact/request"):
        koa.verify_transport_completion(run_dir, artifact, args)


def test_transport_rejects_artifact_aperture_for_another_cathode_offset(
    tmp_path: Path,
):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    run_dir = tmp_path / "run"
    args = [
        *_transport_args(artifact, run_dir),
        "--delta_cathode_chamfer_mm",
        "0.25",
    ]
    _write_fake_completed_transport(artifact, run_dir, args)
    with pytest.raises(koa.CompletionVerificationError, match="aperture geometry"):
        koa.verify_transport_completion(run_dir, artifact, args)


def test_transport_rejects_alternate_core_output_filename(tmp_path: Path):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    run_dir = tmp_path / "run"
    args = _transport_args(artifact, run_dir)
    _write_fake_completed_transport(artifact, run_dir, args)
    alternate = run_dir / "alternate_results.json"
    alternate.write_bytes((run_dir / "run_results.json").read_bytes())
    marker_path = run_dir / koa.TRANSPORT_MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["required_outputs"]["run_results"] = {
        "file": alternate.name,
        "sha256": koa.sha256_file(alternate),
    }
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(koa.CompletionVerificationError, match="manifest|must bind"):
        koa.verify_transport_completion(run_dir, artifact, args)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    (
        ("file_sha256", "f" * 64),
        ("payload_sha256", "e" * 64),
        ("source_sha256", "d" * 64),
        ("processing_config_sha256", "c" * 64),
        ("rf_magnetic_field_mode", "zero"),
        ("rf_magnetic_field_enabled", False),
        ("source_power_w", 1.7e6),
        ("target_power_w", 1.1e6),
        ("common_field_scale", 0.5),
        ("field_tail_policy", "zero-control"),
        ("artifact_variant", "zero_tail_control"),
    ),
)
def test_transport_rejects_rehashed_but_false_field_identity(
    tmp_path: Path, key: str, bad_value: object
):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    run_dir = tmp_path / "run"
    args = _transport_args(artifact, run_dir)
    _write_fake_completed_transport(artifact, run_dir, args)

    marker_path = run_dir / koa.TRANSPORT_MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["field_identity"][key] = bad_value
    # Rehash the internally consistent lie: verification must compare to the actual artifact and
    # requested run rather than merely checking the marker against its own digest.
    marker["field_input_sha256"] = koa.canonical_json_sha256(marker["field_identity"])
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(koa.CompletionVerificationError, match=key):
        koa.verify_transport_completion(run_dir, artifact, args)


def _macropulse_args(transport_dir: Path, output_dir: Path, *, bin_ns: float = 10.0):
    return [
        "--source-mode",
        "load_run",
        "--run-dir",
        str(transport_dir),
        "--thermal-backend",
        "python_xy_layered",
        "--material-set",
        "LaB6_UH_recommended_v1",
        "--deposition-model",
        "BB0_TIO",
        "--macropulse-duration-us",
        "10",
        "--thermal-bin-ns",
        str(bin_ns),
        "--initial-temperature-uniform-k",
        "1650",
        "--output",
        str(output_dir),
        "--save-figures",
    ]


def _write_fake_macropulse_outputs(output_dir: Path, *, event_file_hash: str) -> None:
    output_dir.mkdir()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    for name in (
        "back_bombardment_source_qualification.png",
        "back_bombardment_macropulse.png",
    ):
        (figures_dir / name).write_bytes(f"synthetic:{name}".encode("utf-8"))
    (output_dir / "study_config.json").write_text(json.dumps({"valid": True}))
    (output_dir / "study_results.json").write_text(
        json.dumps({"valid": True, "event_file_hash": event_file_hash})
    )
    _write_h5_schema(
        output_dir / "back_bombardment_events.h5", "back_bombardment_events_v2"
    )
    _write_h5_schema(
        output_dir / "back_bombardment_heat_source.h5",
        "back_bombardment_heat_source_v1",
        attributes={"source_events_hash": event_file_hash},
    )
    _write_h5_schema(
        output_dir / "back_bombardment_macropulse.h5",
        "back_bombardment_macropulse_v1",
        attributes={"event_file_hash": event_file_hash},
    )


def test_macropulse_marker_binds_request_transport_and_outputs(tmp_path: Path):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    transport_dir = tmp_path / "transport"
    _write_fake_completed_transport(
        artifact, transport_dir, _transport_args(artifact, transport_dir)
    )
    output_dir = tmp_path / "macropulse"
    _write_fake_macropulse_outputs(
        output_dir,
        event_file_hash=koa.sha256_file(transport_dir / "back_bombardment_events.h5"),
    )
    args = _macropulse_args(transport_dir, output_dir)
    tags = [
        "campaign=test",
        "study=iv",
        "stage=macropulse",
        "task_id=0",
        "git_commit=" + "1" * 40,
        "git_state=clean",
        "code_sha256=" + "3" * 64,
        "python_stack_sha256=" + "4" * 64,
        "numpy_version=2.0.0",
        "scipy_version=1.14.0",
        "h5py_version=3.12.0",
        "slurm_job_id=100",
        "slurm_array_job_id=90",
        "slurm_array_task_id=0",
    ]
    koa.write_macropulse_completion(
        output_dir, transport_dir, args, identity_tags=tags
    )
    resumed_tags = [
        *(tag for tag in tags if not tag.startswith(("slurm_job_id=", "slurm_array_job_id="))),
        "slurm_job_id=200",
        "slurm_array_job_id=190",
    ]
    assert (
        koa.verify_macropulse_completion(
            output_dir, transport_dir, args, identity_tags=resumed_tags
        )["status"]
        == "exact_match"
    )

    with pytest.raises(koa.CompletionVerificationError, match="request identity"):
        koa.verify_macropulse_completion(
            output_dir,
            transport_dir,
            _macropulse_args(transport_dir, output_dir, bin_ns=5.0),
            identity_tags=resumed_tags,
        )

    with h5py.File(transport_dir / "back_bombardment_events.h5", "a") as handle:
        handle.attrs["tampered"] = True
    with pytest.raises(koa.CompletionVerificationError, match="digest mismatch"):
        koa.verify_macropulse_completion(
            output_dir, transport_dir, args, identity_tags=resumed_tags
        )


def test_macropulse_rejects_plain_marker_and_failed_rewrite_clears_old_claim(
    tmp_path: Path,
):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    transport_dir = tmp_path / "transport"
    _write_fake_completed_transport(
        artifact, transport_dir, _transport_args(artifact, transport_dir)
    )
    output_dir = tmp_path / "macropulse"
    output_dir.mkdir()
    marker_path = output_dir / koa.MACROPULSE_MARKER_NAME
    marker_path.write_text("2026-09-11T00:00:00Z\n", encoding="utf-8")
    args = _macropulse_args(transport_dir, output_dir)
    tags = [
        "campaign=test",
        "study=iv",
        "stage=macropulse",
        "task_id=0",
        "git_commit=" + "1" * 40,
        "git_state=clean",
        "code_sha256=" + "3" * 64,
        "python_stack_sha256=" + "4" * 64,
        "numpy_version=2.0.0",
        "scipy_version=1.14.0",
        "h5py_version=3.12.0",
        "slurm_job_id=100",
        "slurm_array_job_id=90",
        "slurm_array_task_id=0",
    ]

    with pytest.raises(koa.CompletionVerificationError):
        koa.verify_macropulse_completion(
            output_dir, transport_dir, args, identity_tags=tags
        )
    with pytest.raises(koa.CompletionVerificationError, match="required output"):
        koa.write_macropulse_completion(
            output_dir, transport_dir, args, identity_tags=tags
        )
    assert not marker_path.exists()


def test_macropulse_refuses_source_transport_with_failed_validation_claim(
    tmp_path: Path,
):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    transport_dir = tmp_path / "transport"
    _write_fake_completed_transport(
        artifact, transport_dir, _transport_args(artifact, transport_dir)
    )
    transport_marker_path = transport_dir / koa.TRANSPORT_MARKER_NAME
    transport_marker = json.loads(transport_marker_path.read_text(encoding="utf-8"))
    transport_marker["validation_status"] = "failed"
    transport_marker_path.write_text(json.dumps(transport_marker), encoding="utf-8")

    output_dir = tmp_path / "macropulse"
    _write_fake_macropulse_outputs(
        output_dir,
        event_file_hash=koa.sha256_file(transport_dir / "back_bombardment_events.h5"),
    )
    tags = [
        "campaign=test",
        "study=iv",
        "stage=macropulse",
        "task_id=0",
        "git_commit=" + "1" * 40,
        "git_state=clean",
        "code_sha256=" + "3" * 64,
        "python_stack_sha256=" + "4" * 64,
        "numpy_version=2.0.0",
        "scipy_version=1.14.0",
        "h5py_version=3.12.0",
        "slurm_job_id=100",
        "slurm_array_job_id=90",
        "slurm_array_task_id=0",
    ]
    with pytest.raises(koa.CompletionVerificationError, match="validation_status"):
        koa.write_macropulse_completion(
            output_dir,
            transport_dir,
            _macropulse_args(transport_dir, output_dir),
            identity_tags=tags,
        )


def _macropulse_identity_tags() -> list[str]:
    return [
        "campaign=test",
        "study=iv",
        "stage=macropulse",
        "task_id=0",
        "git_commit=" + "1" * 40,
        "git_state=clean",
        "code_sha256=" + "3" * 64,
        "python_stack_sha256=" + "4" * 64,
        "numpy_version=2.0.0",
        "scipy_version=1.14.0",
        "h5py_version=3.12.0",
        "slurm_job_id=100",
        "slurm_array_job_id=90",
        "slurm_array_task_id=0",
    ]


def test_macropulse_completion_refuses_source_changed_during_processing(tmp_path: Path):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    transport_dir = tmp_path / "transport"
    _write_fake_completed_transport(
        artifact, transport_dir, _transport_args(artifact, transport_dir)
    )
    source_before = koa.macropulse_source_identity_sha256(transport_dir)
    transport_marker_path = transport_dir / koa.TRANSPORT_MARKER_NAME
    transport_marker = json.loads(transport_marker_path.read_text(encoding="utf-8"))
    transport_marker["timestamp_utc"] = "changed-after-stage-2-start"
    transport_marker_path.write_text(json.dumps(transport_marker), encoding="utf-8")

    output_dir = tmp_path / "macropulse"
    event_sha = koa.sha256_file(transport_dir / "back_bombardment_events.h5")
    _write_fake_macropulse_outputs(output_dir, event_file_hash=event_sha)
    with pytest.raises(koa.CompletionVerificationError, match="changed while"):
        koa.write_macropulse_completion(
            output_dir,
            transport_dir,
            _macropulse_args(transport_dir, output_dir),
            identity_tags=_macropulse_identity_tags(),
            expected_source_sha256=source_before,
        )
    assert not (output_dir / koa.MACROPULSE_MARKER_NAME).exists()


@pytest.mark.parametrize(
    ("filename", "attribute"),
    (
        ("study_results.json", None),
        ("back_bombardment_heat_source.h5", "source_events_hash"),
        ("back_bombardment_macropulse.h5", "event_file_hash"),
    ),
)
def test_macropulse_rejects_rehashed_false_internal_source_reference(
    tmp_path: Path, filename: str, attribute: str | None
):
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    transport_dir = tmp_path / "transport"
    _write_fake_completed_transport(
        artifact, transport_dir, _transport_args(artifact, transport_dir)
    )
    output_dir = tmp_path / "macropulse"
    event_sha = koa.sha256_file(transport_dir / "back_bombardment_events.h5")
    _write_fake_macropulse_outputs(output_dir, event_file_hash=event_sha)
    args = _macropulse_args(transport_dir, output_dir)
    tags = _macropulse_identity_tags()
    koa.write_macropulse_completion(output_dir, transport_dir, args, identity_tags=tags)

    product_path = output_dir / filename
    if attribute is None:
        result = json.loads(product_path.read_text(encoding="utf-8"))
        result["event_file_hash"] = "f" * 64
        product_path.write_text(json.dumps(result), encoding="utf-8")
    else:
        with h5py.File(product_path, "a") as handle:
            handle.attrs[attribute] = "f" * 64

    marker_path = output_dir / koa.MACROPULSE_MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["required_outputs"][product_path.stem]["sha256"] = koa.sha256_file(product_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(koa.CompletionVerificationError, match="bound transport event"):
        koa.verify_macropulse_completion(
            output_dir, transport_dir, args, identity_tags=tags
        )


def test_transport_verifier_accepts_a_relative_run_dir(tmp_path: Path, monkeypatch):
    """The launchers pass --run-dir relative to the repo root, so this must not crash.

    `_safe_marker_member` always returns a resolved absolute path. Comparing it against an
    unresolved relative `run_path` raised
    `ValueError: ... is not in the subpath of ...`, so the completion gate died with a
    traceback instead of returning a verdict -- taking out the SLURM resume/skip logic.
    """
    artifact = tmp_path / "field.h5"
    _write_artifact(artifact)
    run_dir = tmp_path / "run"
    args = _transport_args(artifact, run_dir)
    _write_fake_completed_transport(artifact, run_dir, args)

    # Re-express the run directory relative to the working directory, as the launchers do.
    monkeypatch.chdir(tmp_path)
    relative_run_dir = Path("run")
    assert not relative_run_dir.is_absolute()

    result = koa.verify_transport_completion(relative_run_dir, artifact, args)
    assert result["status"] == "exact_match"
