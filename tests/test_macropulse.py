"""Tests for `rf_gun.macropulse` (implementation plan Sec. 8.1, 8.2, 8.5; addendum Sec. 19.2/19.4) --
Work Package 3/4's configurable macropulse model.

Pure Python/numpy -- no RF-Track dependency. Synthetic `BackBombardmentEvents`/
`BackBombardmentHeatSource` objects are built directly, matching the pattern already used by
`tests/test_back_bombardment_deposition.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

import rf_gun as rg
from rf_gun.back_bombardment_events import (
    BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
    BackBombardmentEvents,
    build_back_bombardment_provenance,
)
from rf_gun.cathode_geometry import CathodeGeometry
from rf_gun.macropulse import (
    build_macropulse_current_history,
    build_macropulse_heat_source,
    build_macropulse_time_grid,
    compute_n_rf_periods,
    evaluate_rf_envelope,
    validate_charge_balance,
)

Q_E = 1.602176634e-19  # elementary charge [C], matches rf_gun.constants.q_e


# ------------------------------------------------------------------------------------------------
# Synthetic fixtures
# ------------------------------------------------------------------------------------------------


def _make_synthetic_events(
    *,
    rf_frequency_Hz: float = 2.856e9,
    charge_C: dict | None = None,
) -> BackBombardmentEvents:
    """A minimal, self-consistent synthetic `BackBombardmentEvents` set with a populated
    `accounting["charge_C"]` dict (the exact shape documented on the class itself). Per-event
    arrays hold two trivial LaB6-flat events so `heats_lab6_mask`/energy-closure helpers elsewhere
    have something non-degenerate to work with, but this module's own functions only read
    `accounting`/`rf_frequency_Hz`/`rf_period_s`.
    """
    n = 2
    K_eV = np.array([50_000.0, 80_000.0])
    w = np.full(n, 1.0e6)
    incident_energy_J = w * K_eV * Q_E

    default_charge = {
        "emitted": 2.0e-9,
        "transmitted": 1.2e-9,
        "other_lost": 0.3e-9,
        "returned_before_filter": 0.55e-9,
        "returned_after_filter": 0.5e-9,
    }
    if charge_C is not None:
        default_charge.update(charge_C)

    accounting = {
        "counts": {"n_launched": 1_000_000, "n_emitted": 900_000},
        "charge_C": default_charge,
        "energy_J": {"incident_before_filter": 1.0e-6, "incident_after_filter": 1.0e-6},
    }

    return BackBombardmentEvents(
        event_id=np.arange(n, dtype=np.int64),
        particle_id=np.arange(100, 100 + n, dtype=np.int64),
        state_id=np.zeros(n, dtype=np.int32),
        x_emit_m=np.zeros(n, dtype=np.float64),
        y_emit_m=np.zeros(n, dtype=np.float64),
        z_emit_m=np.zeros(n, dtype=np.float64),
        t_emit_rf_s=np.zeros(n, dtype=np.float64),
        rf_phase_emit_rad=np.zeros(n, dtype=np.float64),
        x_hit_m=np.array([0.0, 0.2e-3]),
        y_hit_m=np.array([0.0, 0.0]),
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
        provenance=build_back_bombardment_provenance(run_id="test_macropulse"),
    )


@pytest.fixture(scope="module")
def geometry() -> CathodeGeometry:
    return CathodeGeometry()


@pytest.fixture(scope="module")
def material():
    return rg.load_cathode_material("LaB6", "LaB6_UH_recommended_v1")


@pytest.fixture()
def synthetic_events() -> BackBombardmentEvents:
    return _make_synthetic_events()


@pytest.fixture()
def small_heat_source(synthetic_events, geometry, material):
    deposition_config = rg.DepositionConfig(model="BB0_TIO")
    return rg.build_back_bombardment_heat_source(
        synthetic_events, geometry, material, deposition_config, xy_grid_n=9
    )


# ------------------------------------------------------------------------------------------------
# 1. evaluate_rf_envelope
# ------------------------------------------------------------------------------------------------


def test_top_hat_envelope_inside_and_boundaries():
    cfg = rg.MacropulseConfig(duration_s=8.0e-6, envelope="top_hat")
    t = np.array([0.0, 1.0e-6, 4.0e-6, 8.0e-6])
    env = evaluate_rf_envelope(t, cfg)
    assert np.array_equal(env, np.ones_like(t))


def test_top_hat_envelope_outside_is_exactly_zero():
    cfg = rg.MacropulseConfig(duration_s=8.0e-6, envelope="top_hat")
    t = np.array([-1.0e-9, -1.0e-6, 8.000001e-6, 20.0e-6])
    env = evaluate_rf_envelope(t, cfg)
    assert np.array_equal(env, np.zeros_like(t))


def test_unknown_envelope_raises_not_implemented_error():
    cfg = rg.MacropulseConfig(duration_s=8.0e-6, envelope="measured_fill_decay")
    with pytest.raises(NotImplementedError):
        evaluate_rf_envelope(np.array([1.0e-6]), cfg)


# ------------------------------------------------------------------------------------------------
# 2. build_macropulse_time_grid / compute_n_rf_periods
# ------------------------------------------------------------------------------------------------


def test_time_grid_bin_count_and_span():
    cfg = rg.MacropulseConfig(duration_s=1.0e-6, envelope="top_hat")
    t_grid = build_macropulse_time_grid(cfg, thermal_bin_s=100e-9)
    assert t_grid[0] == pytest.approx(0.0)
    assert t_grid[-1] == pytest.approx(1.0e-6)
    n_bins = t_grid.size - 1
    assert n_bins == pytest.approx(10, abs=0)  # 1us / 100ns = 10 bins exactly


def test_n_rf_periods_matches_plan_worked_example():
    cfg = rg.MacropulseConfig(duration_s=8.0e-6, envelope="top_hat")
    n_rf = compute_n_rf_periods(2.856e9, cfg)
    assert n_rf == pytest.approx(22848.0, rel=1.0e-6)


# ------------------------------------------------------------------------------------------------
# 3. build_macropulse_heat_source: exact energy-closure check
# ------------------------------------------------------------------------------------------------


def test_macropulse_heat_source_energy_closure(small_heat_source):
    rf_frequency_Hz = 2.856e9
    duration_s = 200e-9  # small, fast test pulse -- see module docstring for why this is fine
    cfg = rg.MacropulseConfig(duration_s=duration_s, envelope="top_hat")
    t_grid_edges_s = build_macropulse_time_grid(cfg, thermal_bin_s=20e-9)
    n_rf = compute_n_rf_periods(rf_frequency_Hz, cfg)

    macro_source = build_macropulse_heat_source(small_heat_source, rf_frequency_Hz, cfg, t_grid_edges_s)

    dt = np.diff(t_grid_edges_s)
    total_macropulse_energy_J = float(np.sum(macro_source.q_layer_W * dt[np.newaxis, np.newaxis, np.newaxis, :]))
    expected_J = float(np.sum(small_heat_source.q_layer_J)) * n_rf

    assert total_macropulse_energy_J == pytest.approx(expected_J, rel=1.0e-9)


def test_macropulse_heat_source_shape_matches_time_grid(small_heat_source):
    cfg = rg.MacropulseConfig(duration_s=200e-9, envelope="top_hat")
    t_grid_edges_s = build_macropulse_time_grid(cfg, thermal_bin_s=20e-9)
    macro_source = build_macropulse_heat_source(small_heat_source, 2.856e9, cfg, t_grid_edges_s)
    assert macro_source.q_layer_W.shape[-1] == t_grid_edges_s.size - 1
    assert np.array_equal(macro_source.t_grid_s, t_grid_edges_s)


# ------------------------------------------------------------------------------------------------
# 4. build_macropulse_current_history
# ------------------------------------------------------------------------------------------------


def test_current_history_matches_independent_f_rf_times_q(synthetic_events):
    cfg = rg.MacropulseConfig(duration_s=200e-9, envelope="top_hat")
    t_grid_edges_s = build_macropulse_time_grid(cfg, thermal_bin_s=20e-9)
    history = build_macropulse_current_history(synthetic_events, cfg, t_grid_edges_s)

    f_RF = synthetic_events.rf_frequency_Hz
    charge = synthetic_events.accounting["charge_C"]
    expected_I_emit = f_RF * charge["emitted"]
    expected_I_return = f_RF * charge["returned_after_filter"]
    expected_I_transmitted = f_RF * charge["transmitted"]
    expected_I_other_loss = f_RF * charge["other_lost"]

    # Every bin center lies inside [0, duration_s] for this grid, so the top-hat envelope is 1.0
    # everywhere here -- every sample should equal the bare per-period rate.
    assert np.allclose(history.I_emit_A, expected_I_emit)
    assert np.allclose(history.I_return_A, expected_I_return)
    assert np.allclose(history.I_transmitted_A, expected_I_transmitted)
    assert np.allclose(history.I_other_loss_A, expected_I_other_loss)
    # I_useful is explicitly defined as I_transmitted, NOT I_emit - I_return.
    assert np.allclose(history.I_useful_A, expected_I_transmitted)
    assert not np.allclose(history.I_useful_A, expected_I_emit - expected_I_return)


def test_current_history_is_zero_outside_duration_via_envelope(synthetic_events):
    """Direct check of the top-hat envelope's zero-outside behavior applied to a current: build a
    grid that deliberately extends past duration_s and confirm the current there is exactly zero.
    `build_macropulse_time_grid`'s own default grid never does this (documented choice, see its
    docstring), so this test constructs the extended grid explicitly.
    """
    cfg = rg.MacropulseConfig(duration_s=100e-9, envelope="top_hat")
    t_grid_edges_s = np.linspace(0.0, 200e-9, 11)  # extends to 2x duration_s
    history = build_macropulse_current_history(synthetic_events, cfg, t_grid_edges_s)
    beyond = history.t_s > cfg.duration_s
    assert np.any(beyond)
    assert np.all(history.I_emit_A[beyond] == 0.0)
    assert np.all(history.I_return_A[beyond] == 0.0)
    assert np.all(history.I_useful_A[beyond] == 0.0)


def test_current_history_missing_accounting_key_raises():
    events = _make_synthetic_events()
    del events.accounting["charge_C"]["other_lost"]
    cfg = rg.MacropulseConfig(duration_s=100e-9, envelope="top_hat")
    t_grid_edges_s = build_macropulse_time_grid(cfg, thermal_bin_s=20e-9)
    with pytest.raises(ValueError, match="other_lost"):
        build_macropulse_current_history(events, cfg, t_grid_edges_s)


# ------------------------------------------------------------------------------------------------
# 5. validate_charge_balance
# ------------------------------------------------------------------------------------------------


def test_charge_balance_passes_for_self_consistent_accounting():
    # emitted == return + transmitted + other_lost exactly -> Q_surviving == 0
    events = _make_synthetic_events(
        charge_C={
            "emitted": 2.0e-9,
            "returned_after_filter": 0.5e-9,
            "transmitted": 1.2e-9,
            "other_lost": 0.3e-9,
        }
    )
    validate_charge_balance(events)  # must not raise


def test_charge_balance_passes_with_nonzero_surviving_charge():
    events = _make_synthetic_events(
        charge_C={
            "emitted": 3.0e-9,
            "returned_after_filter": 0.5e-9,
            "transmitted": 1.2e-9,
            "other_lost": 0.3e-9,
        }
    )
    validate_charge_balance(events)  # 3.0 > 0.5+1.2+0.3=2.0 -> Q_surviving=1.0 >= 0, fine


def test_charge_balance_raises_for_broken_accounting():
    events = _make_synthetic_events(
        charge_C={
            "emitted": 1.0e-9,
            "returned_before_filter": 0.9e-9,
            "returned_after_filter": 0.8e-9,
            "transmitted": 0.8e-9,  # 0.8+0.8=1.6 > emitted=1.0 -- physically impossible
            "other_lost": 0.0,
        }
    )
    with pytest.raises(ValueError, match="[Cc]harge balance"):
        validate_charge_balance(events)


def test_charge_balance_missing_key_raises_distinct_error():
    events = _make_synthetic_events()
    del events.accounting["charge_C"]["emitted"]
    with pytest.raises(ValueError, match="emitted"):
        validate_charge_balance(events)
