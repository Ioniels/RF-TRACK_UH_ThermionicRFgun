from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import run_thermionic_tm010 as cli
from rf_gun.fieldmaps import (
    ArtifactValidationError,
    AxisymmetricRFField,
    FrameTransform,
    PRODUCTION_QUALITY_GATE_VERSION,
    REQUIRED_PRODUCTION_GATE_NAMES,
    write_axisymmetric_artifact,
)


def _write_artifact(
    path: Path,
    *,
    status: str = "qualified",
    variant: str = "modal",
) -> AxisymmetricRFField:
    r_m = np.array([0.0, 1.0e-3, 2.0e-3])
    z_m = np.array([0.0, 1.0e-2, 2.0e-2])
    shape = (z_m.size, r_m.size)
    phase = np.exp(0.3j)
    field = AxisymmetricRFField(
        frequency_hz=2.865e9,
        r_m=r_m,
        z_m=z_m,
        Er_Vpm=np.full(shape, 2.0e3 * phase),
        Ez_Vpm=np.full(shape, 3.0e6 * phase),
        Btheta_T=np.full(shape, 4.0e-3 * phase),
        Bz_T=np.full(shape, 5.0e-5 * phase),
        metadata={"source_solver": "Remcom XFdtd", "source_power_w": 1.8e6},
    )
    production_gates = {
        name: {"passed": True, "gate_version": PRODUCTION_QUALITY_GATE_VERSION}
        for name in set(REQUIRED_PRODUCTION_GATE_NAMES) | {"source_integrity"}
    }
    write_axisymmetric_artifact(
        path,
        field,
        transform=FrameTransform(np.zeros(3), np.eye(3)),
        quality={
            "status": status,
            "variant": variant,
            **(
                {"gates": production_gates}
                if status == "qualified"
                else {}
            ),
        },
        provenance={
            "source_sha256": "a" * 64,
            "processing_config": {"theta_count": 32, "projection": "m0"},
            "source_manifest": {"source_power_w": 1.8e6},
        },
    )
    return field


def test_cli_exposes_artifact_and_magnetic_control(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_thermionic_tm010.py",
            "--field-artifact",
            "field.h5",
            "--rf-magnetic-field",
            "zero",
            "--field-tail-policy",
            "zero-control",
        ],
    )
    args = cli.parse_args()
    assert args.field_artifact == Path("field.h5")
    assert args.rf_magnetic_field == "zero"
    assert args.field_tail_policy == "zero-control"


def test_qualified_artifact_owns_fields_grid_frequency_and_b(tmp_path: Path):
    path = tmp_path / "qualified.h5"
    expected = _write_artifact(path)
    loaded = cli._load_qualified_artifact_for_tracking(
        path,
        magnetic_field_mode="artifact",
        requested_z_min_m=0.0,
        requested_z_max_m=0.02,
        target_power_w=1.8e6,
    )

    assert loaded["payload_verification"] == "payload"
    assert loaded["magnetic_field_enabled"] is True
    assert loaded["field"].frequency_hz == expected.frequency_hz
    np.testing.assert_array_equal(loaded["field"].r_m, expected.r_m)
    np.testing.assert_array_equal(loaded["field"].z_m, expected.z_m)
    np.testing.assert_allclose(loaded["Er_grid"], expected.Er_Vpm)
    np.testing.assert_allclose(loaded["Ez_grid"], expected.Ez_Vpm)
    np.testing.assert_allclose(loaded["Bt_grid"], expected.Btheta_T)
    np.testing.assert_allclose(loaded["Bz_grid"], expected.Bz_T)
    assert loaded["common_field_scale"] == pytest.approx(1.0)

    provenance = cli._artifact_provenance_record(path, loaded)
    assert provenance["payload_sha256"] == loaded["artifact"].payload_sha256
    assert provenance["source_sha256"] == "a" * 64
    assert len(provenance["processing_config_sha256"]) == 64
    assert provenance["processing_config"]["projection"] == "m0"


def test_zero_mode_is_explicit_e_only_control(tmp_path: Path):
    path = tmp_path / "qualified.h5"
    expected = _write_artifact(path)
    loaded = cli._load_qualified_artifact_for_tracking(
        path,
        magnetic_field_mode="zero",
        requested_z_min_m=0.0,
        requested_z_max_m=0.01,
        target_power_w=1.8e6,
    )
    assert loaded["magnetic_field_enabled"] is False
    assert loaded["Bt_grid"] is None
    assert loaded["Bz_grid"] is None
    np.testing.assert_allclose(loaded["Ez_grid"], expected.Ez_Vpm[:2])
    np.testing.assert_allclose(loaded["field"].z_m, expected.z_m[:2])


def test_artifact_tracking_domain_must_align_to_grid(tmp_path: Path):
    path = tmp_path / "qualified.h5"
    _write_artifact(path)
    with pytest.raises(ValueError, match="must coincide with artifact z-grid planes"):
        cli._load_qualified_artifact_for_tracking(
            path,
            magnetic_field_mode="artifact",
            requested_z_min_m=0.0,
            requested_z_max_m=0.0101,
            target_power_w=1.8e6,
        )


def test_artifact_tracking_domain_must_fit_support(tmp_path: Path):
    path = tmp_path / "qualified.h5"
    _write_artifact(path)
    with pytest.raises(ValueError, match="outside the qualified field artifact support"):
        cli._load_qualified_artifact_for_tracking(
            path,
            magnetic_field_mode="artifact",
            requested_z_min_m=0.0,
            requested_z_max_m=0.0201,
            target_power_w=1.8e6,
        )


def test_analysis_only_artifact_is_rejected_for_tracking(tmp_path: Path):
    path = tmp_path / "analysis_only.h5"
    _write_artifact(path, status="analysis_only")
    with pytest.raises(ArtifactValidationError, match="requires 'qualified'"):
        cli._load_qualified_artifact_for_tracking(
            path,
            magnetic_field_mode="artifact",
            requested_z_min_m=0.0,
            requested_z_max_m=0.02,
            target_power_w=1.8e6,
        )


def test_artifact_power_scaling_is_common_to_all_e_and_b_components(tmp_path: Path):
    path = tmp_path / "qualified.h5"
    expected = _write_artifact(path)
    loaded = cli._load_qualified_artifact_for_tracking(
        path,
        magnetic_field_mode="artifact",
        requested_z_min_m=0.0,
        requested_z_max_m=0.02,
        target_power_w=1.0e6,
    )
    scale = np.sqrt(1.0e6 / 1.8e6)
    assert loaded["common_field_scale"] == pytest.approx(scale)
    np.testing.assert_allclose(loaded["Er_grid"], expected.Er_Vpm * scale)
    np.testing.assert_allclose(loaded["Ez_grid"], expected.Ez_Vpm * scale)
    np.testing.assert_allclose(loaded["Bt_grid"], expected.Btheta_T * scale)
    np.testing.assert_allclose(loaded["Bz_grid"], expected.Bz_T * scale)


def test_zero_tail_artifact_requires_explicit_control_policy(tmp_path: Path):
    path = tmp_path / "zero_tail_control.h5"
    _write_artifact(path, variant="zero_tail_control")
    with pytest.raises(ValueError, match="variant does not match"):
        cli._load_qualified_artifact_for_tracking(
            path,
            magnetic_field_mode="artifact",
            requested_z_min_m=0.0,
            requested_z_max_m=0.02,
            target_power_w=1.0e6,
        )

    loaded = cli._load_qualified_artifact_for_tracking(
        path,
        magnetic_field_mode="artifact",
        requested_z_min_m=0.0,
        requested_z_max_m=0.02,
        target_power_w=1.0e6,
        field_tail_policy="zero-control",
    )
    assert loaded["artifact_variant"] == "zero_tail_control"


def test_omitting_the_field_artifact_fails_closed_instead_of_using_legacy_mat():
    """The retired .mat path must never be reached by accident.

    It carries no magnetic field, so a silent fallback would quietly downgrade a run to
    E-only against unqualified data while still reporting success.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "run_thermionic_tm010.py", "--preset", "quick", "--n_particles", "4"],
        cwd=repo, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode != 0, "a bare run must not silently succeed on legacy .mat data"
    combined = proc.stdout + proc.stderr
    assert "--field-artifact" in combined
    assert "--allow-legacy-mat-fieldmap" in combined


def test_legacy_mat_path_remains_reachable_behind_the_explicit_opt_in(monkeypatch):
    """The opt-in must still parse, so the escape hatch is real rather than decorative."""
    import sys

    import run_thermionic_tm010 as cli

    monkeypatch.setattr(
        sys, "argv", ["run_thermionic_tm010.py", "--allow-legacy-mat-fieldmap"]
    )
    args = cli.parse_args()
    assert args.allow_legacy_mat_fieldmap is True
    assert args.field_artifact is None
