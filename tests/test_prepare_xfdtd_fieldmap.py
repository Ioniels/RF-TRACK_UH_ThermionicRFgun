from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import prepare_xfdtd_fieldmap as cli
from rf_gun.fieldmaps import (
    production_quality_gates,
    resolve_production_quality_status,
)


def _quality_inputs() -> tuple[dict, dict, dict, dict]:
    reduction = {
        "temporal": {
            "production_metrics": {
                "max_family_combined_nrmse": 0.01,
                "max_significant_family_block_nrmse": 0.012,
                "max_significant_component_active_nrmse_p95_diagnostic": 0.75,
                "max_family_combined_fractional_drift_per_cycle": 0.005,
            },
        }
    }
    numerical = {
        "maxwell_vacuum": {
            "active_guarded_vacuum": {
                "combined": {
                    "faraday_family_normalized_rms": 0.03,
                    "ampere_family_normalized_rms": 0.04,
                }
            }
        },
        "maxwell_beam_core": {
            "active_guarded_vacuum": {
                "combined": {
                    "faraday_family_normalized_rms": 0.03,
                    "ampere_family_normalized_rms": 0.04,
                }
            }
        },
        "axis_regularity": {
            "odd_components": {
                "Er": {"fraction_of_combined_field_rms": 0.0},
                "Btheta": {"fraction_of_combined_field_rms": 0.0},
            }
        },
    }
    tail = {"status": "pass"}
    origin = {"status": "pass", "origin_physically_bracketed": True}
    return reduction, numerical, tail, origin


def test_shared_production_gates_control_quality_status():
    reduction, numerical, tail, origin = _quality_inputs()
    gates = production_quality_gates(reduction, numerical, tail, origin)
    assert all(gate["passed"] for gate in gates.values())
    assert resolve_production_quality_status("auto", gates) == "qualified"

    numerical["maxwell_beam_core"]["active_guarded_vacuum"]["combined"][
        "faraday_family_normalized_rms"
    ] = 0.21
    gates = production_quality_gates(reduction, numerical, tail, origin)
    assert not gates["maxwell_faraday_beam_core"]["passed"]
    assert resolve_production_quality_status("auto", gates) == "analysis_only"
    with pytest.raises(RuntimeError, match="maxwell_faraday_beam_core"):
        resolve_production_quality_status("qualified", gates)


def test_unselected_transient_treatment_prevents_production_qualification():
    reduction, numerical, tail, origin = _quality_inputs()
    gates = production_quality_gates(reduction, numerical, tail, origin)
    gates["fill_treatment"] = {"passed": False, "policy": "auto"}
    assert resolve_production_quality_status("auto", gates) == "analysis_only"


def test_step_assumption_is_required_before_any_source_read(monkeypatch):
    monkeypatch.setattr(cli, "inspect_xfdtd_h5", lambda _: pytest.fail("must fail before source read"))
    with pytest.raises(ValueError, match="requires --fill-start-time-ns"):
        cli.main(["--fill-correction", "single-pole"])
    with pytest.raises(ValueError, match="only valid"):
        cli.main(["--fill-correction", "off", "--fill-start-time-ns", "0"])


@pytest.mark.parametrize("collision", ["output", "report", "zero_control"])
def test_preparation_refuses_any_destination_equal_to_source(
    tmp_path: Path,
    collision: str,
):
    source = tmp_path / "raw.h5"
    source.write_bytes(b"not opened because path validation comes first")
    output = source if collision == "output" else tmp_path / "field.h5"
    arguments = ["--source", str(source), "--output", str(output), "--dry-run"]
    if collision == "report":
        arguments.extend(["--report", str(source)])
    if collision == "zero_control":
        arguments.extend(["--also-write-zero-control", str(source)])
    with pytest.raises(ValueError, match="distinct paths"):
        cli.main(arguments)


def test_preparation_refuses_production_control_and_report_collisions(tmp_path: Path):
    source = tmp_path / "raw.h5"
    source.write_bytes(b"not opened because path validation comes first")
    output = tmp_path / "field.h5"
    common = ["--source", str(source), "--output", str(output), "--dry-run"]

    with pytest.raises(ValueError, match="output == zero_control"):
        cli.main(common + ["--also-write-zero-control", str(output)])
    with pytest.raises(ValueError, match="output == report"):
        cli.main(common + ["--report", str(output)])


def test_existing_destination_is_rejected_before_source_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "raw.h5"
    output = tmp_path / "existing.h5"
    source.write_bytes(b"source")
    output.write_bytes(b"must not be replaced")
    monkeypatch.setattr(
        cli,
        "inspect_xfdtd_h5",
        lambda _path: pytest.fail("source inspection must not run after preflight failure"),
    )
    with pytest.raises(FileExistsError, match="before discovering existing"):
        cli.main(["--source", str(source), "--output", str(output)])
    assert output.read_bytes() == b"must not be replaced"


@pytest.mark.parametrize(
    ("extra_arguments", "message"),
    [
        (["--theta-count", "0"], "theta-count"),
        (["--tail-fit-start-mm", "1000"], "fewer than three"),
    ],
)
def test_invalid_expensive_run_settings_fail_during_read_free_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_arguments: list[str],
    message: str,
):
    source = tmp_path / "raw.h5"
    source.write_bytes(b"metadata is monkeypatched")
    manifest = SimpleNamespace(
        drive_frequency_hz=2.865e9,
        source_power_w=1.8e6,
        file_size_bytes=24_805_000_000,
        electric_time_s=np.arange(17),
        magnetic_h_time_s=np.arange(17),
    )
    r = np.linspace(0.0, 0.034, 341)
    measured_z = np.linspace(0.0, 0.033, 661)
    full_z = np.linspace(0.0, 0.040, 801)
    plan = SimpleNamespace(
        r_m=r,
        measured_z_m=measured_z,
        full_z_m=full_z,
        measured_vacuum_mask=np.ones((measured_z.size, r.size), dtype=bool),
        missing_length_m=float(full_z[-1] - measured_z[-1]),
    )
    monkeypatch.setattr(cli, "inspect_xfdtd_h5", lambda _path: manifest)
    monkeypatch.setattr(cli, "plan_axisymmetric_grid", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        cli,
        "validate_xfdtd_h5",
        lambda *_args, **_kwargs: pytest.fail(
            "schema/checksum validation must not run after configuration preflight failure"
        ),
    )
    arguments = [
        "--source",
        str(source),
        "--output",
        str(tmp_path / "output.h5"),
        "--dry-run",
        *extra_arguments,
    ]
    with pytest.raises(ValueError, match=message):
        cli.main(arguments)
