from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "KOA_slurm_scripts"
LAUNCHERS = tuple(sorted(SLURM_DIR.glob("*.slurm")))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_expected_koa_launcher_set_is_covered() -> None:
    assert {path.name for path in LAUNCHERS} == {
        "study_i_medium_field_grid.slurm",
        "study_i_bis_fine_field_grid.slurm",
        "study_ii_finesse.slurm",
        "study_iii_temperature.slurm",
        "study_iii_bis_temperature_1500_1700K.slurm",
        "study_iv_deflection_macropulse.slurm",
    }


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda path: path.stem)
def test_koa_launchers_are_valid_shell_and_share_fail_closed_transport_contract(
    launcher: Path,
) -> None:
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    source = _text(launcher)

    assert ".mat" not in source.lower()
    assert '--field-artifact "$FIELD_ARTIFACT"' in source
    # The two field switches are overridable so the A/B controls are reachable from
    # automation, but production must still DEFAULT to E+B with the modal tail, and an
    # unrecognised override must fail closed rather than silently pick a default.
    assert '--rf-magnetic-field "$RF_MAGNETIC_FIELD"' in source
    assert '--field-tail-policy "$FIELD_TAIL_POLICY"' in source
    assert 'RF_MAGNETIC_FIELD="${RFTRACK_RF_MAGNETIC_FIELD:-artifact}"' in source
    assert 'FIELD_TAIL_POLICY="${RFTRACK_FIELD_TAIL_POLICY:-modal}"' in source
    assert 'case "$RF_MAGNETIC_FIELD" in artifact|zero' in source
    assert 'case "$FIELD_TAIL_POLICY" in modal|zero-control' in source
    assert "--rf-forward-power-w" in source
    assert "--run-family" in source
    assert "--scan-tags" in source
    assert "code-identity --repo \"$REPO_DIR\" --launcher \"$0\"" in source
    assert "verify-artifact" in source
    assert source.count("verify-transport") >= 2
    assert "INVALID_OR_STALE" not in source  # emitted centrally by the Python verifier

    for tag in (
        "campaign=",
        "study=",
        "task_id=",
        "git_commit=",
        "git_state=",
        "code_sha256=",
        "python_stack_sha256=",
        "rftrack_version=",
        "slurm_job_id=",
        "slurm_array_job_id=",
        "slurm_array_task_id=",
    ):
        assert tag in source

    # Shell never manufactures a success marker; only the validated Python writers may do so.
    assert "touch .run_complete" not in source
    assert 'date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/.run_complete"' not in source
    assert "Refusing to overwrite" in source or "refusing to overwrite" in source


@pytest.mark.parametrize(
    "name",
    ("study_i_medium_field_grid.slurm", "study_i_bis_fine_field_grid.slurm"),
)
def test_grid_studies_bind_task_labels_to_real_distinct_artifacts(name: str) -> None:
    source = _text(SLURM_DIR / name)
    assert (
        'FIELD_ARTIFACT="$FIELD_ARTIFACT_DIR/'
        'cavity_axisymmetric_m0_dr${DR_UM}um_dz${DZ_UM}um_rftrack.h5"'
    ) in source
    assert '--expected-dr-um "$DR_UM"' in source
    assert '--expected-dz-um "$DZ_UM"' in source
    assert "--dr_um" not in source
    assert "--dz_um" not in source
    assert 'run_thermionic_tm010.py "${RUN_ARGS[@]}"' in source
    assert '--artifact "$FIELD_ARTIFACT" -- "${RUN_ARGS[@]}"' in source


def test_study_iv_stage_two_is_bound_to_transport_request_and_outputs() -> None:
    source = _text(SLURM_DIR / "study_iv_deflection_macropulse.slurm")
    assert 'run_thermionic_tm010.py "${TRANSPORT_ARGS[@]}"' in source
    assert '--artifact "$FIELD_ARTIFACT" -- "${TRANSPORT_ARGS[@]}"' in source
    assert 'run_back_bombardment_macropulse.py "${MACROPULSE_ARGS[@]}"' in source
    assert "write-macropulse" in source
    assert source.count("verify-macropulse") == 2
    assert '--transport-dir "$TRANSPORT_DIR"' in source
    assert '"thermal_bin_ns=$THERMAL_BIN_NS"' in source
    assert '"numpy_version=$NUMPY_VERSION_STAGE2"' in source
    assert '"scipy_version=$SCIPY_VERSION_STAGE2"' in source
    assert '"h5py_version=$H5PY_VERSION_STAGE2"' in source
    assert "--expected-source-sha256 \"$TRANSPORT_SOURCE_SHA256\"" in source
    assert '"deflection_current_A=$CURRENT_A"' in source
    assert '"stage=transport"' in source
    assert '"stage=macropulse"' in source
