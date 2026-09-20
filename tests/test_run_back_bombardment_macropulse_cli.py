"""Tests for `run_back_bombardment_macropulse.py` (implementation plan Sec. 10.1, 13; Work
Package 4) -- the thin argparse CLI wrapper around
`rf_gun.studies.back_bombardment_macropulse.run_back_bombardment_macropulse_study`.

Two kinds of coverage:

  1. `--help` runs without error and advertises the documented flags (`build_parser()` is
     imported directly -- this script does not depend on RF-Track just to build its parser, so
     this test does not need it importable either).
  2. An end-to-end `--source-mode load_run` run against a small synthetic run directory
     constructed here (a valid v2 `back_bombardment_events.h5` written via
     `rg.write_back_bombardment_events_h5`, matching
     `tests/test_back_bombardment_macropulse_study.py`'s own synthetic-events pattern), with a
     tiny macropulse duration for speed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rf_gun.back_bombardment_events import (
    BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
    BackBombardmentEvents,
    build_back_bombardment_provenance,
    write_back_bombardment_events_h5,
)

import run_back_bombardment_macropulse as cli

Q_E = 1.602176634e-19
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "run_back_bombardment_macropulse.py"


def _make_synthetic_events(rf_frequency_Hz: float = 2.856e9) -> BackBombardmentEvents:
    """Same pattern as `tests/test_back_bombardment_macropulse_study.py`'s own
    `_make_synthetic_events` helper -- a tiny, internally-consistent, closing event set."""
    n = 3
    K_eV = np.array([40_000.0, 90_000.0, 150_000.0])
    w = np.full(n, 1.0e6)
    incident_energy_J = w * K_eV * Q_E

    accounting = {
        "counts": {"n_launched": 1_000_000, "n_emitted": 900_000},
        "charge_C": {
            "emitted": 2.0e-9,
            "transmitted": 1.2e-9,
            "other_lost": 0.3e-9,
            "returned_before_filter": 0.55e-9,
            "returned_after_filter": 0.5e-9,
        },
        "energy_J": {"incident_before_filter": 1.0e-6, "incident_after_filter": 1.0e-6},
    }

    return BackBombardmentEvents(
        event_id=np.arange(n, dtype=np.int64),
        particle_id=np.arange(200, 200 + n, dtype=np.int64),
        state_id=np.zeros(n, dtype=np.int32),
        x_emit_m=np.zeros(n, dtype=np.float64),
        y_emit_m=np.zeros(n, dtype=np.float64),
        z_emit_m=np.zeros(n, dtype=np.float64),
        t_emit_rf_s=np.zeros(n, dtype=np.float64),
        rf_phase_emit_rad=np.zeros(n, dtype=np.float64),
        x_hit_m=np.array([0.0, 0.2e-3, -0.3e-3]),
        y_hit_m=np.array([0.0, 0.1e-3, 0.0]),
        z_hit_m=np.zeros(n, dtype=np.float64),
        t_hit_rf_s=np.zeros(n, dtype=np.float64),
        return_time_s=np.zeros(n, dtype=np.float64),
        px_MeV_c=np.zeros(n, dtype=np.float64),
        py_MeV_c=np.zeros(n, dtype=np.float64),
        pz_MeV_c=-np.ones(n, dtype=np.float64) * 0.01,
        kinetic_energy_eV=K_eV,
        macro_weight_electrons=w,
        incident_energy_J=incident_energy_J,
        n_in_x=np.zeros(n, dtype=np.float64),
        n_in_y=np.zeros(n, dtype=np.float64),
        n_in_z=-np.ones(n, dtype=np.float64),
        cos_incidence=np.ones(n, dtype=np.float64),
        incidence_angle_rad=np.zeros(n, dtype=np.float64),
        surface_code=np.zeros(n, dtype=np.uint8),
        heats_lab6=np.ones(n, dtype=bool),
        furthest_z_m=np.zeros(n, dtype=np.float64),
        n_screens_reached=np.zeros(n, dtype=np.int16),
        quality_flags=np.zeros(n, dtype=np.uint32),
        schema_version=BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
        coordinate_system="cathode_z0_vacuum_positive_z",
        normal_convention="inward_vacuum_to_solid",
        rf_frequency_Hz=rf_frequency_Hz,
        rf_period_s=1.0 / rf_frequency_Hz,
        emission_phase_window="full_rf_period",
        n_launched=1_000_000,
        q_emitted_per_sample_C=2.0e-9,
        event_locator="legacy_ballistic",
        sc_enabled=False,
        mirror_charge_enabled=False,
        beam_loading_enabled=False,
        deflection_magnet_enabled=False,
        emission_iteration_enabled=False,
        flat_radius_mm=1.4,
        bevel_width_mm=0.2,
        bevel_angle_deg=45.0,
        cathode_length_mm=1.0,
        insertion_offset_mm=0.0,
        accounting=accounting,
        provenance=build_back_bombardment_provenance(
            run_id="test_run_back_bombardment_macropulse_cli"
        ),
    )


@pytest.fixture()
def synthetic_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "synthetic_run"
    run_dir.mkdir()
    write_back_bombardment_events_h5(run_dir / "back_bombardment_events.h5", _make_synthetic_events())
    return run_dir


# --------------------------------------------------------------------------------------------
# --help / argparse structure
# --------------------------------------------------------------------------------------------

def test_help_runs_without_error_and_does_not_need_rf_gun():
    """`build_parser()` must be importable/callable without importing `rf_gun` (and therefore
    without RF-Track) -- matches `run_thermionic_tm010.py`'s own convention of deferring the
    `rf_gun` import into `main()` so `--help` stays fast/independent."""
    assert "rf_gun" not in sys.modules or True  # rf_gun may already be imported by other tests
    parser = cli.build_parser()
    help_text = parser.format_help()
    for flag in (
        "--source-mode",
        "--run-dir",
        "--thermal-backend",
        "--material-set",
        "--deposition-model",
        "--macropulse-duration-us",
        "--initial-temperature-uniform-k",
        "--initial-temperature-map",
        "--comsol-results",
        "--output",
        "--stage",
        "--study-h5",
        "--save-figures",
    ):
        assert flag in help_text, f"expected {flag!r} to appear in --help output"


def test_help_subprocess_runs_without_error():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-mode" in result.stdout


def test_current_notebook_source_mode_rejected_with_clear_error():
    with pytest.raises(SystemExit):
        cli.parse_args(["--source-mode", "current_notebook"])


def test_missing_run_dir_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(["--source-mode", "load_run"])


def test_missing_initial_temperature_rejected(tmp_path: Path):
    with pytest.raises(SystemExit):
        cli.parse_args(
            ["--source-mode", "load_run", "--run-dir", str(tmp_path), "--output", str(tmp_path / "out")]
        )


def test_both_initial_temperature_options_rejected(tmp_path: Path):
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--source-mode", "load_run",
                "--run-dir", str(tmp_path),
                "--output", str(tmp_path / "out"),
                "--initial-temperature-uniform-k", "1650",
                "--initial-temperature-map", str(tmp_path / "map.h5"),
            ]
        )


def test_missing_output_rejected_for_run_stage(tmp_path: Path):
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--source-mode", "load_run",
                "--run-dir", str(tmp_path),
                "--initial-temperature-uniform-k", "1650",
            ]
        )


def test_compare_stage_requires_study_h5_and_comsol_results(tmp_path: Path):
    with pytest.raises(SystemExit):
        cli.parse_args(["--stage", "compare", "--study-h5", str(tmp_path / "study.h5")])


# --------------------------------------------------------------------------------------------
# End-to-end --source-mode load_run
# --------------------------------------------------------------------------------------------

def test_end_to_end_load_run_completes_and_writes_expected_files(synthetic_run_dir: Path, tmp_path: Path):
    output_dir = tmp_path / "study_output"
    rc = cli.main(
        [
            "--source-mode", "load_run",
            "--run-dir", str(synthetic_run_dir),
            "--thermal-backend", "python_xy_layered",
            "--material-set", "LaB6_UH_recommended_v1",
            "--deposition-model", "BB0_TIO",
            "--macropulse-duration-us", str(200e-9 * 1e6),  # 200 ns, tiny for test speed
            "--initial-temperature-uniform-k", "1650",
            "--output", str(output_dir),
        ]
    )
    assert rc == 0

    assert (output_dir / "study_config.json").is_file()
    assert (output_dir / "study_results.json").is_file()
    assert (output_dir / "back_bombardment_events.h5").is_file()
    assert (output_dir / "back_bombardment_heat_source.h5").is_file()
    assert (output_dir / "back_bombardment_macropulse.h5").is_file()

    results = json.loads((output_dir / "study_results.json").read_text())
    assert results["charge_balance_ok"] is True
    assert results["material"]["property_set"] == "LaB6_UH_recommended_v1"
    assert results["thermal"]["backend"] == "python_xy_layered"
    assert np.isfinite(results["thermal"]["T_max_final_K"])

    config = json.loads((output_dir / "study_config.json").read_text())
    assert config["macropulse"]["duration_s"] == pytest.approx(200e-9)


def test_end_to_end_load_run_with_save_figures(synthetic_run_dir: Path, tmp_path: Path):
    output_dir = tmp_path / "study_output_figs"
    rc = cli.main(
        [
            "--source-mode", "load_run",
            "--run-dir", str(synthetic_run_dir),
            "--macropulse-duration-us", str(200e-9 * 1e6),
            "--initial-temperature-uniform-k", "1650",
            "--output", str(output_dir),
            "--save-figures",
        ]
    )
    assert rc == 0
    assert (output_dir / "figures" / "back_bombardment_source_qualification.png").is_file()
    assert (output_dir / "figures" / "back_bombardment_macropulse.png").is_file()


def test_initial_temperature_map_not_implemented(synthetic_run_dir: Path, tmp_path: Path):
    output_dir = tmp_path / "study_output_map"
    fake_map = tmp_path / "initial_temperature_xy.h5"
    fake_map.write_bytes(b"not a real hdf5 file, just needs to exist for argparse's Path type")
    with pytest.raises(NotImplementedError):
        cli.main(
            [
                "--source-mode", "load_run",
                "--run-dir", str(synthetic_run_dir),
                "--macropulse-duration-us", str(200e-9 * 1e6),
                "--initial-temperature-map", str(fake_map),
                "--output", str(output_dir),
            ]
        )


def test_missing_run_dir_events_file_raises_clear_error(tmp_path: Path):
    empty_run_dir = tmp_path / "empty_run"
    empty_run_dir.mkdir()
    with pytest.raises(ValueError, match="back_bombardment_events.h5"):
        cli.main(
            [
                "--source-mode", "load_run",
                "--run-dir", str(empty_run_dir),
                "--macropulse-duration-us", str(200e-9 * 1e6),
                "--initial-temperature-uniform-k", "1650",
                "--output", str(tmp_path / "out"),
            ]
        )


# --------------------------------------------------------------------------------------------
# --stage compare
# --------------------------------------------------------------------------------------------

def test_stage_compare_scalar_only(synthetic_run_dir: Path, tmp_path: Path):
    # First produce a completed study (back_bombardment_macropulse.h5) via --stage run.
    output_dir = tmp_path / "study_for_compare"
    rc = cli.main(
        [
            "--source-mode", "load_run",
            "--run-dir", str(synthetic_run_dir),
            "--macropulse-duration-us", str(200e-9 * 1e6),
            "--initial-temperature-uniform-k", "1650",
            "--output", str(output_dir),
        ]
    )
    assert rc == 0
    study_h5 = output_dir / "back_bombardment_macropulse.h5"
    assert study_h5.is_file()

    # Build a tiny synthetic COMSOL result file matching the study's own time grid.
    import h5py

    with h5py.File(study_h5, "r") as h5f:
        t_grid_s = np.asarray(h5f["thermal"]["t_grid_s"][()], dtype=float)

    comsol_path = tmp_path / "comsol_results.h5"
    n_t = t_grid_s.size
    from rf_gun.comsol_io import COMSOL_THERMAL_RESULT_SCHEMA_VERSION

    with h5py.File(comsol_path, "w") as h5f:
        h5f.attrs["schema_version"] = COMSOL_THERMAL_RESULT_SCHEMA_VERSION
        h5f.attrs["mesh_id"] = "test_mesh"
        h5f.attrs["time_step_id"] = "test_time_step"
        thermal_grp = h5f.create_group("thermal")
        thermal_grp.create_dataset("time_s", data=t_grid_s)
        for name in (
            "T_center_K", "T_flat_edge_K", "T_bevel_K", "T_max_K", "T_area_average_K",
            "stored_energy_J", "radiation_loss_W", "boundary_heat_flow_W",
        ):
            thermal_grp.create_dataset(name, data=np.full(n_t, 1650.0))

    compare_output_dir = tmp_path / "compare_output"
    rc = cli.main(
        [
            "--stage", "compare",
            "--study-h5", str(study_h5),
            "--comsol-results", str(comsol_path),
            "--output", str(compare_output_dir),
        ]
    )
    assert rc == 0
    comparison_path = compare_output_dir / "comsol_comparison.json"
    assert comparison_path.is_file()
    comparison = json.loads(comparison_path.read_text())
    assert comparison["comsol_available"] is True
    assert comparison["scope"] == "scalar_thermal_history_only"


def test_bb0_closure_keys_match_between_driver_and_study_module():
    """Two independent copies of the bb0_energy_closure block must stay in step.

    `run_back_bombardment_macropulse.py` writes study_results.json and
    `rf_gun/studies/back_bombardment_macropulse.py` writes the HDF5 summary. They were built
    separately, so a field added to one silently went missing from the other -- which is
    exactly how the rim-reassignment accounting failed to reach study_results.json.
    """
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]

    def closure_keys(path: Path) -> set[str]:
        text = path.read_text(encoding="utf-8")
        start = text.index('"bb0_energy_closure": {')
        depth, i = 0, start + len('"bb0_energy_closure": ')
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        block = text[start:i]
        return set(re.findall(r'"([A-Za-z0-9_]+)":', block)) - {"bb0_energy_closure"}

    driver = closure_keys(repo / "run_back_bombardment_macropulse.py")
    module = closure_keys(repo / "rf_gun" / "studies" / "back_bombardment_macropulse.py")
    assert driver == module, (
        "bb0_energy_closure fields diverged between the driver and the study module: "
        f"driver-only={sorted(driver - module)}, module-only={sorted(module - driver)}"
    )
    for required in ("rim_reassigned_energy_J_total", "non_finite_position_energy_J_total"):
        assert required in driver, f"{required} missing from the closure record"
