"""Tests for `rf_gun.back_bombardment_events` and `rf_gun.back_bombardment_study_config`
(implementation plan Sec. 2.3, 3.1, 4.1, 4.2, 10.2, 10.3; addendum Sec. 19).

Pure Python/HDF5 -- no RF-Track dependency, no particle tracking (matches the module's own
scope).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import rf_gun as rg
from rf_gun.back_bombardment_events import (
    BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
    EVENT_ARRAY_FIELDS,
    BackBombardmentEvents,
    build_back_bombardment_provenance,
    read_back_bombardment_events_h5,
    resolve_back_bombardment_study_input,
    write_back_bombardment_events_h5,
    display_back_bombardment_event_schema,
)

@pytest.fixture
def unversioned_event_file(tmp_path: Path) -> Path:
    """A readable HDF5 file with no schema tag, independent of local run history.

    The reader must reject it before inspecting event arrays; a valid HDF5
    container avoids confusing this schema check with a corrupt-file error.
    """
    path = tmp_path / "back_bombardment_events.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x_hit_mm", data=[0.0])
    return path


def _make_synthetic_events(n: int = 5) -> BackBombardmentEvents:
    rng = np.random.default_rng(1234)
    surface_code = np.array([0, 0, 1, 1, 10][:n], dtype=np.uint8)
    heats_lab6 = np.isin(surface_code, (0, 1, 2))
    return BackBombardmentEvents(
        event_id=np.arange(n, dtype=np.int64),
        particle_id=np.arange(100, 100 + n, dtype=np.int64),
        state_id=np.zeros(n, dtype=np.int32),
        x_emit_m=rng.uniform(-1e-3, 1e-3, n),
        y_emit_m=rng.uniform(-1e-3, 1e-3, n),
        z_emit_m=np.zeros(n, dtype=np.float64),
        t_emit_rf_s=rng.uniform(0, 3.5e-10, n),
        rf_phase_emit_rad=rng.uniform(0, 2 * np.pi, n),
        x_hit_m=rng.uniform(-1e-3, 1e-3, n),
        y_hit_m=rng.uniform(-1e-3, 1e-3, n),
        z_hit_m=np.zeros(n, dtype=np.float64),
        t_hit_rf_s=rng.uniform(0, 3.5e-10, n),
        return_time_s=rng.uniform(0, 1e-10, n),
        px_MeV_c=rng.uniform(-0.01, 0.01, n),
        py_MeV_c=rng.uniform(-0.01, 0.01, n),
        pz_MeV_c=-np.abs(rng.uniform(0.01, 0.1, n)),
        kinetic_energy_eV=rng.uniform(1e3, 5e5, n),
        macro_weight_electrons=np.full(n, 4476.386, dtype=np.float64),
        incident_energy_J=rng.uniform(1e-18, 1e-16, n),
        n_in_x=np.zeros(n, dtype=np.float64),
        n_in_y=np.zeros(n, dtype=np.float64),
        n_in_z=np.full(n, -1.0, dtype=np.float64),
        cos_incidence=rng.uniform(0.5, 1.0, n),
        incidence_angle_rad=rng.uniform(0.0, 1.0, n),
        surface_code=surface_code,
        heats_lab6=heats_lab6,
        furthest_z_m=rng.uniform(0.0, 0.05, n),
        n_screens_reached=rng.integers(0, 5, n).astype(np.int16),
        quality_flags=np.zeros(n, dtype=np.uint32),
        schema_version=BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
        coordinate_system="cathode_z0_vacuum_positive_z",
        normal_convention="inward_vacuum_to_solid",
        rf_frequency_Hz=2.856e9,
        rf_period_s=1.0 / 2.856e9,
        emission_phase_window="full_rf_period",
        n_launched=100_000,
        q_emitted_per_sample_C=1.0e-9,
        event_locator="legacy_ballistic",
        sc_enabled=True,
        mirror_charge_enabled=True,
        beam_loading_enabled=True,
        deflection_magnet_enabled=False,
        emission_iteration_enabled=False,
        flat_radius_mm=1.4,
        bevel_width_mm=0.2,
        bevel_angle_deg=45.0,
        cathode_length_mm=1.0,
        insertion_offset_mm=0.0,
        accounting={
            "counts": {"n_launched": 100_000, "n_emitted": 90_000, "n_returned_after_filter": n},
            "charge_C": {"returned_before_filter": 2.2e-11, "returned_after_filter": 2.18e-11},
            "energy_J": {"incident_before_filter": 1.9e-6, "incident_after_filter": 1.8e-6},
        },
        provenance=build_back_bombardment_provenance(run_id="test_run"),
    )


# ------------------------------------------------------------------------------------------
# Dataclass construction / validation
# ------------------------------------------------------------------------------------------

def test_synthetic_events_construct_and_validate():
    events = _make_synthetic_events()
    assert events.n_events == 5
    assert events.schema_version == BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION


def test_mismatched_array_lengths_raise_value_error():
    events = _make_synthetic_events()
    bad_event_id = np.arange(3, dtype=np.int64)  # wrong length vs. the other n=5 arrays
    with pytest.raises(ValueError):
        BackBombardmentEvents(
            **{**{name: getattr(events, name) for name in EVENT_ARRAY_FIELDS}, "event_id": bad_event_id},
            schema_version=events.schema_version,
            coordinate_system=events.coordinate_system,
            normal_convention=events.normal_convention,
            rf_frequency_Hz=events.rf_frequency_Hz,
            rf_period_s=events.rf_period_s,
            emission_phase_window=events.emission_phase_window,
            n_launched=events.n_launched,
            q_emitted_per_sample_C=events.q_emitted_per_sample_C,
            event_locator=events.event_locator,
            sc_enabled=events.sc_enabled,
            mirror_charge_enabled=events.mirror_charge_enabled,
            beam_loading_enabled=events.beam_loading_enabled,
            deflection_magnet_enabled=events.deflection_magnet_enabled,
            emission_iteration_enabled=events.emission_iteration_enabled,
            flat_radius_mm=events.flat_radius_mm,
            bevel_width_mm=events.bevel_width_mm,
            bevel_angle_deg=events.bevel_angle_deg,
            cathode_length_mm=events.cathode_length_mm,
            insertion_offset_mm=events.insertion_offset_mm,
            accounting=events.accounting,
            provenance=events.provenance,
        )


def test_wrong_schema_version_raises_value_error():
    events = _make_synthetic_events()
    fields = {name: getattr(events, name) for name in EVENT_ARRAY_FIELDS}
    with pytest.raises(ValueError):
        BackBombardmentEvents(
            **fields,
            schema_version="back_bombardment_events_v1",
            coordinate_system=events.coordinate_system,
            normal_convention=events.normal_convention,
            rf_frequency_Hz=events.rf_frequency_Hz,
            rf_period_s=events.rf_period_s,
            emission_phase_window=events.emission_phase_window,
            n_launched=events.n_launched,
            q_emitted_per_sample_C=events.q_emitted_per_sample_C,
            event_locator=events.event_locator,
            sc_enabled=events.sc_enabled,
            mirror_charge_enabled=events.mirror_charge_enabled,
            beam_loading_enabled=events.beam_loading_enabled,
            deflection_magnet_enabled=events.deflection_magnet_enabled,
            emission_iteration_enabled=events.emission_iteration_enabled,
            flat_radius_mm=events.flat_radius_mm,
            bevel_width_mm=events.bevel_width_mm,
            bevel_angle_deg=events.bevel_angle_deg,
            cathode_length_mm=events.cathode_length_mm,
            insertion_offset_mm=events.insertion_offset_mm,
            accounting=events.accounting,
            provenance=events.provenance,
        )


def test_heats_lab6_mask_matches_stored_field():
    events = _make_synthetic_events()
    np.testing.assert_array_equal(events.heats_lab6_mask, events.heats_lab6)


# ------------------------------------------------------------------------------------------
# HDF5 round trip
# ------------------------------------------------------------------------------------------

def test_hdf5_round_trip_reproduces_every_field(tmp_path):
    events = _make_synthetic_events()
    out_path = write_back_bombardment_events_h5(tmp_path / "back_bombardment_events.h5", events)
    assert out_path.is_file()

    loaded = read_back_bombardment_events_h5(out_path)

    for name in EVENT_ARRAY_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(loaded, name), dtype=float),
            np.asarray(getattr(events, name), dtype=float),
            rtol=1e-12, atol=1e-30,
            err_msg=f"field {name!r} did not round-trip exactly",
        )

    assert loaded.schema_version == events.schema_version
    assert loaded.coordinate_system == events.coordinate_system
    assert loaded.normal_convention == events.normal_convention
    assert loaded.rf_frequency_Hz == pytest.approx(events.rf_frequency_Hz)
    assert loaded.rf_period_s == pytest.approx(events.rf_period_s)
    assert loaded.sample_represents == events.sample_represents
    assert loaded.emission_phase_window == events.emission_phase_window
    assert loaded.n_launched == events.n_launched
    assert loaded.q_emitted_per_sample_C == pytest.approx(events.q_emitted_per_sample_C)
    assert loaded.event_locator == events.event_locator
    assert loaded.sc_enabled == events.sc_enabled
    assert loaded.mirror_charge_enabled == events.mirror_charge_enabled
    assert loaded.beam_loading_enabled == events.beam_loading_enabled
    assert loaded.deflection_magnet_enabled == events.deflection_magnet_enabled
    assert loaded.emission_iteration_enabled == events.emission_iteration_enabled
    assert loaded.flat_radius_mm == pytest.approx(events.flat_radius_mm)
    assert loaded.bevel_width_mm == pytest.approx(events.bevel_width_mm)
    assert loaded.bevel_angle_deg == pytest.approx(events.bevel_angle_deg)
    assert loaded.cathode_length_mm == pytest.approx(events.cathode_length_mm)
    assert loaded.insertion_offset_mm == pytest.approx(events.insertion_offset_mm)
    assert loaded.accounting == events.accounting
    assert loaded.provenance == events.provenance
    assert loaded.source_state == events.source_state


# ------------------------------------------------------------------------------------------
# Strict rejection of pre-v2 (legacy, unversioned) files
# ------------------------------------------------------------------------------------------

def test_reading_legacy_unversioned_file_raises_clear_pre_v2_error(unversioned_event_file):
    with pytest.raises(ValueError) as excinfo:
        read_back_bombardment_events_h5(unversioned_event_file)
    message = str(excinfo.value)
    assert "schema_version" in message
    assert "predates" in message or "not a supported" in message.lower() or "not" in message.lower()
    assert str(unversioned_event_file) in message


# ------------------------------------------------------------------------------------------
# resolve_back_bombardment_study_input
# ------------------------------------------------------------------------------------------

def test_resolve_current_notebook_mode():
    events = _make_synthetic_events()
    study_input = resolve_back_bombardment_study_input("current_notebook", current_events=events)
    assert study_input.source_mode == "current_notebook"
    assert study_input.events is events
    assert study_input.source_path is None
    assert study_input.origin_run_id is None
    assert isinstance(study_input.event_file_hash, str) and len(study_input.event_file_hash) == 64


def test_resolve_current_notebook_mode_requires_events():
    with pytest.raises(ValueError):
        resolve_back_bombardment_study_input("current_notebook", current_events=None)


def test_resolve_load_run_mode_on_legacy_run_dir_raises_same_clear_error(unversioned_event_file):
    with pytest.raises(ValueError) as excinfo:
        resolve_back_bombardment_study_input("load_run", run_dir=unversioned_event_file.parent)
    assert "schema_version" in str(excinfo.value)


def test_resolve_load_run_mode_round_trips_through_a_freshly_written_v2_file(tmp_path):
    events = _make_synthetic_events()
    run_dir = tmp_path / "a_new_v2_run"
    run_dir.mkdir()
    write_back_bombardment_events_h5(run_dir / "back_bombardment_events.h5", events)

    study_input = resolve_back_bombardment_study_input("load_run", run_dir=run_dir)
    assert study_input.source_mode == "load_run"
    assert study_input.origin_run_id == run_dir.name
    assert study_input.source_path == run_dir / "back_bombardment_events.h5"
    np.testing.assert_array_equal(study_input.events.event_id, events.event_id)


def test_resolve_bogus_mode_raises_value_error():
    with pytest.raises(ValueError):
        resolve_back_bombardment_study_input("bogus_mode")


# ------------------------------------------------------------------------------------------
# Study config
# ------------------------------------------------------------------------------------------

def test_default_config_has_8us_macropulse():
    cfg = rg.default_uh_back_bombardment_study_config()
    assert cfg.macropulse.duration_s == pytest.approx(8.0e-6)
    assert cfg.material.material_id == "LaB6"
    assert cfg.material.property_set == "LaB6_UH_recommended_v1"
    assert cfg.deposition.model == "BB0_TIO"
    assert cfg.thermal.backend == "python_xy_layered"
    assert cfg.coupling.level == "L2_one_way"


def test_replace_changes_only_the_targeted_nested_field():
    cfg = rg.default_uh_back_bombardment_study_config()
    cfg2 = cfg.replace(macropulse=rg.MacropulseConfig(duration_s=12.0e-6))

    assert cfg2.macropulse.duration_s == pytest.approx(12.0e-6)
    assert cfg2.macropulse.envelope == "top_hat"
    # Everything else stays at its default, unaffected by the macropulse override.
    assert cfg2.material == cfg.material
    assert cfg2.deposition == cfg.deposition
    assert cfg2.thermal == cfg.thermal
    assert cfg2.coupling == cfg.coupling
    assert cfg2.capture == cfg.capture
    assert cfg2.geometry == cfg.geometry
    # The original config is untouched (frozen dataclasses.replace never mutates in place).
    assert cfg.macropulse.duration_s == pytest.approx(8.0e-6)


# ------------------------------------------------------------------------------------------
# display_back_bombardment_event_schema
# ------------------------------------------------------------------------------------------

def test_display_schema_unsaved_in_memory_study(capsys):
    events = _make_synthetic_events()
    display_back_bombardment_event_schema(events, h5_path=None)
    captured = capsys.readouterr().out
    assert BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION in captured
    assert "No file written" in captured
    assert "event_id" in captured


def test_display_schema_with_a_real_file(tmp_path, capsys):
    events = _make_synthetic_events()
    out_path = write_back_bombardment_events_h5(tmp_path / "back_bombardment_events.h5", events)
    display_back_bombardment_event_schema(events, h5_path=out_path)
    captured = capsys.readouterr().out
    assert BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION in captured
    assert str(out_path) in captured
