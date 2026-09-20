"""End-to-end tests for `rf_gun.studies.back_bombardment_macropulse` (implementation plan Sec. 4.2,
10.1, 10.2, 10.3; addendum Sec. 19.2) -- the high-level study orchestration wiring together material
resolution, BB0 deposition, the macropulse model, the `python_xy_layered` thermal solver, and the
(interface-only) COMSOL comparison.

Pure Python/numpy -- no RF-Track dependency. Uses a small synthetic `BackBombardmentEvents` set
(matching `tests/test_macropulse.py`'s/`tests/test_back_bombardment_deposition.py`'s own pattern)
wrapped in `source_mode="current_notebook"`.
"""
from __future__ import annotations

import dataclasses

import h5py
import numpy as np
import pytest

import rf_gun as rg
from rf_gun.back_bombardment_events import (
    BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
    BackBombardmentEvents,
    build_back_bombardment_provenance,
    resolve_back_bombardment_study_input,
)
from rf_gun.studies.back_bombardment_macropulse import BACK_BOMBARDMENT_MACROPULSE_SCHEMA_VERSION
from rf_gun.cathode_geometry import SURFACE_UNKNOWN

Q_E = 1.602176634e-19


def _make_synthetic_events(rf_frequency_Hz: float = 2.856e9) -> BackBombardmentEvents:
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
        provenance=build_back_bombardment_provenance(run_id="test_back_bombardment_macropulse_study"),
    )


@pytest.fixture()
def study_input():
    events = _make_synthetic_events()
    return resolve_back_bombardment_study_input(source_mode="current_notebook", current_events=events)


@pytest.fixture()
def tiny_config():
    """`default_uh_back_bombardment_study_config()` with the macropulse duration shrunk to 200ns
    (from the 8us production default) purely for test speed -- the `python_xy_layered` FV solver's
    cost scales with the number of thermal bins, and 200ns is enough to exercise every step of the
    pipeline (multiple bins, a genuine Picard-iterated implicit solve) without a slow test.
    """
    base = rg.default_uh_back_bombardment_study_config()
    return base.replace(macropulse=dataclasses.replace(base.macropulse, duration_s=200e-9))


def test_end_to_end_study_runs_without_error(study_input, tiny_config):
    study = rg.run_back_bombardment_macropulse_study(
        study_input=study_input,
        config=tiny_config,
        initial_temperature=rg.ConstantTemperatureMap(1650.0),
    )
    assert study.thermal_result.backend == "python_xy_layered"
    assert study.comparison.comsol_available is False
    assert study.charge_balance_ok is True
    assert study.charge_balance_error is None
    # No output_dir was supplied -> no files written.
    assert study.events_h5 is None
    assert study.macropulse_h5 is None


def test_end_to_end_study_validates(study_input, tiny_config):
    study = rg.run_back_bombardment_macropulse_study(
        study_input=study_input,
        config=tiny_config,
        initial_temperature=rg.ConstantTemperatureMap(1650.0),
    )
    rg.validate_back_bombardment_study(study)  # must not raise


def test_asymmetric_temperature_map_2d_initial_condition_runs_without_error(study_input, tiny_config):
    """An asymmetric `TemperatureMap2D` (as the notebook builds from a `CATHODE_TEMPERATURE_K`
    callable, see `UH_gun_tracking_demo.ipynb`'s back-bombardment macropulse cell) must run the
    same as `ConstantTemperatureMap`, not just a uniform value silently re-flattened to constant.
    """
    from rf_gun.cathode_geometry import CathodeGeometry

    half_width_mm = CathodeGeometry().bevel_outer_radius_mm * 1.05
    axis_mm = np.linspace(-half_width_mm, half_width_mm, 41)
    X_mm, Y_mm = np.meshgrid(axis_mm, axis_mm, indexing="ij")
    T_K = 1650.0 + 20.0 * (X_mm / half_width_mm)  # a genuinely asymmetric (x-dependent) profile

    initial_temperature = rg.TemperatureMap2D(
        x_m=axis_mm * 1e-3, y_m=axis_mm * 1e-3, T_K=T_K, mask=np.ones_like(X_mm, dtype=bool),
    )
    study = rg.run_back_bombardment_macropulse_study(
        study_input=study_input, config=tiny_config, initial_temperature=initial_temperature,
    )
    assert study.thermal_result.backend == "python_xy_layered"
    # The asymmetry must actually reach the solver's own initial condition, not get flattened to
    # a single mean value -- T_center (x=y=0, T0=1650K on this profile) must differ from a point
    # off-center where the profile was constructed to be genuinely hotter.
    assert np.isclose(study.thermal_result.T_center_t[0], 1650.0, atol=1.0)
    assert study.thermal_result.T_max_t[0] > 1650.0 + 5.0


def test_wrong_coupling_level_rejected_immediately(study_input, tiny_config):
    bad_config = tiny_config.replace(coupling=rg.CouplingConfig(level="L3_thermal_emission_feedback"))
    with pytest.raises(ValueError, match="L2_one_way"):
        rg.run_back_bombardment_macropulse_study(
            study_input=study_input,
            config=bad_config,
            initial_temperature=rg.ConstantTemperatureMap(1650.0),
        )


def test_charge_balance_failure_warns_but_does_not_abort(tiny_config):
    events = _make_synthetic_events()
    # Break the accounting: transmitted+other_lost+returned exceeds emitted.
    events.accounting["charge_C"]["emitted"] = 1.0e-9
    bad_input = resolve_back_bombardment_study_input(source_mode="current_notebook", current_events=events)

    with pytest.warns(RuntimeWarning, match="[Cc]harge balance"):
        study = rg.run_back_bombardment_macropulse_study(
            study_input=bad_input,
            config=tiny_config,
            initial_temperature=rg.ConstantTemperatureMap(1650.0),
        )
    assert study.charge_balance_ok is False
    assert study.charge_balance_error is not None

    # A hard post-hoc validation call must now raise (module docstring's documented contrast).
    with pytest.raises(ValueError):
        rg.validate_back_bombardment_study(study)


def test_electrons_that_simply_miss_the_cathode_are_not_a_validation_failure(tiny_config):
    """A backward electron that does not strike LaB6 is PHYSICS, not a geometry failure: the
    annulus outside the 3.2 mm LaB6 disk is open vacuum, so it crosses the cathode plane without
    hitting anything. The gate must not fire on that -- it used to, which is what made every
    production run breach a 1% threshold at 8-9%."""
    events = _make_synthetic_events()
    events = dataclasses.replace(
        events,
        surface_code=np.array([SURFACE_UNKNOWN, SURFACE_UNKNOWN, 0], dtype=np.uint8),
        heats_lab6=np.array([False, False, True]),
    )
    ok_input = resolve_back_bombardment_study_input(source_mode="current_notebook", current_events=events)
    study = rg.run_back_bombardment_macropulse_study(
        study_input=ok_input,
        config=tiny_config,
        initial_temperature=rg.ConstantTemperatureMap(1650.0),
    )
    rg.validate_back_bombardment_study(study)  # must not raise


def test_rays_that_failed_numerically_are_still_fatal(tiny_config):
    """What the gate IS for: rows whose classification cannot be trusted either way. A failed
    ID join means the event could not be tied back to an emitted particle at all."""
    from rf_gun.back_bombardment_events import QUALITY_FLAG_ID_JOIN_FAILED

    events = _make_synthetic_events()
    events = dataclasses.replace(
        events,
        quality_flags=np.array(
            [QUALITY_FLAG_ID_JOIN_FAILED, QUALITY_FLAG_ID_JOIN_FAILED, 0], dtype=np.uint32
        ),
    )
    bad_input = resolve_back_bombardment_study_input(source_mode="current_notebook", current_events=events)
    study = rg.run_back_bombardment_macropulse_study(
        study_input=bad_input,
        config=tiny_config,
        initial_temperature=rg.ConstantTemperatureMap(1650.0),
    )
    with pytest.raises(ValueError, match="numerical reasons|max_unknown_surface_fraction"):
        rg.validate_back_bombardment_study(study)


def test_unknown_surface_fraction_within_threshold_accepted(study_input, tiny_config):
    """The default fixture (0% unknown) must not trip the new gate -- guards against the check
    being miscalibrated to always fail."""
    study = rg.run_back_bombardment_macropulse_study(
        study_input=study_input,
        config=tiny_config,
        initial_temperature=rg.ConstantTemperatureMap(1650.0),
    )
    rg.validate_back_bombardment_study(study)  # must not raise


def test_output_dir_writes_macropulse_h5_with_schema_and_provenance(study_input, tiny_config, tmp_path):
    output_dir = tmp_path / "study_run"
    study = rg.run_back_bombardment_macropulse_study(
        study_input=study_input,
        config=tiny_config,
        initial_temperature=rg.ConstantTemperatureMap(1650.0),
        output_dir=output_dir,
    )
    assert study.macropulse_h5 is not None
    assert study.macropulse_h5.is_file()
    assert study.events_h5 is not None
    assert study.events_h5.is_file()
    assert study.heat_source_h5 is not None
    assert study.heat_source_h5.is_file()

    from rf_gun.back_bombardment_deposition import (
        BACK_BOMBARDMENT_HEAT_SOURCE_SCHEMA_VERSION,
        read_back_bombardment_heat_source_h5,
    )

    with h5py.File(str(study.heat_source_h5), "r") as h5f_source:
        assert h5f_source.attrs["schema_version"] == BACK_BOMBARDMENT_HEAT_SOURCE_SCHEMA_VERSION
        assert h5f_source.attrs["source_events_hash"] == study_input.event_file_hash

    reloaded_heat_source = read_back_bombardment_heat_source_h5(study.heat_source_h5)
    np.testing.assert_allclose(reloaded_heat_source.q_layer_J, study.heat_source.q_layer_J)

    with h5py.File(str(study.macropulse_h5), "r") as h5f:
        assert h5f.attrs["schema_version"] == BACK_BOMBARDMENT_MACROPULSE_SCHEMA_VERSION
        assert "provenance" in h5f
        import json

        provenance = json.loads(h5f["provenance"].attrs["json"])
        assert provenance["schema_version"] == BACK_BOMBARDMENT_MACROPULSE_SCHEMA_VERSION
        assert provenance["source_mode"] == "current_notebook"

        assert "current" in h5f
        assert "I_emit_A" in h5f["current"]
        assert "thermal" in h5f
        assert "T_max_K" in h5f["thermal"]
        assert "comsol" in h5f
        assert bool(h5f["comsol"].attrs["comsol_available"]) is False

        config_blob = json.loads(h5f["config"].attrs["json"])
        assert config_blob["macropulse"]["duration_s"] == pytest.approx(200e-9)

        benchmark = json.loads(h5f["benchmark_metrics"].attrs["json"])
        assert "charge_balance" in benchmark
        assert "bb0_energy_closure" in benchmark
