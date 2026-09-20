"""Tests for `rf_gun.plotting.macropulse.plot_back_bombardment_macropulse` (implementation plan
Sec. 12, "Figure B") -- Work Package 4's plotting deliverable.

Pure Python/numpy/matplotlib -- no RF-Track dependency. Builds a small synthetic
`BackBombardmentMacropulseStudy` end to end (same pattern as
`tests/test_plot_back_bombardment_source_qualification.py`/
`tests/test_back_bombardment_macropulse_study.py`) and checks the figure this module produces.

The single most important test here (plan Sec. 8.2/12, addendum Sec. 19.2's explicit user
decision): when no COMSOL result is supplied, the figure must NOT draw any COMSOL curve, dashed
line, or placeholder for one -- it must instead clearly label the temperature panel as "Python
only". `test_no_comsol_result_draws_no_comsol_curves` checks this via the deterministic
`fig.macropulse_comsol_curves_plotted` flag plus the panel's own title text -- a real check, not
merely "did not raise".
"""
from __future__ import annotations

import dataclasses

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import rf_gun as rg
from rf_gun.back_bombardment_events import (
    BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
    BackBombardmentEvents,
    build_back_bombardment_provenance,
    resolve_back_bombardment_study_input,
)
from rf_gun.comsol_io import ComsolThermalResult

Q_E = 1.602176634e-19


def _make_synthetic_events(rf_frequency_Hz: float = 2.856e9) -> BackBombardmentEvents:
    n = 4
    K_eV = np.array([20_000.0, 60_000.0, 120_000.0, 200_000.0])
    w = np.full(n, 1.0e6)
    incident_energy_J = w * K_eV * Q_E
    surface_code = np.array([0, 0, 1, 1], dtype=np.uint8)

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
        particle_id=np.arange(300, 300 + n, dtype=np.int64),
        state_id=np.zeros(n, dtype=np.int32),
        x_emit_m=np.zeros(n, dtype=np.float64),
        y_emit_m=np.zeros(n, dtype=np.float64),
        z_emit_m=np.zeros(n, dtype=np.float64),
        t_emit_rf_s=np.zeros(n, dtype=np.float64),
        rf_phase_emit_rad=np.zeros(n, dtype=np.float64),
        x_hit_m=np.array([0.0, 0.2e-3, 1.45e-3, -1.5e-3]),
        y_hit_m=np.array([0.0, -0.1e-3, 0.0, 0.1e-3]),
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
        surface_code=surface_code,
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
        provenance=build_back_bombardment_provenance(run_id="test_plot_bb_macropulse"),
    )


def _build_tiny_study(comsol_results=None):
    events = _make_synthetic_events()
    study_input = resolve_back_bombardment_study_input(source_mode="current_notebook", current_events=events)
    base = rg.default_uh_back_bombardment_study_config()
    tiny_config = base.replace(macropulse=dataclasses.replace(base.macropulse, duration_s=200e-9))
    return rg.run_back_bombardment_macropulse_study(
        study_input=study_input,
        config=tiny_config,
        initial_temperature=rg.ConstantTemperatureMap(1650.0),
        comsol_results=comsol_results,
    )


@pytest.fixture()
def tiny_study_no_comsol():
    return _build_tiny_study(comsol_results=None)


def _write_synthetic_comsol_result(tmp_path, thermal_result) -> ComsolThermalResult:
    """A self-consistent synthetic `ComsolThermalResult` (mirroring
    `tests/test_comsol_io.py`'s own `write_synthetic_comsol_result_for_testing` stand-in pattern,
    duplicated locally so this test file has no cross-test-file dependency): a constant offset
    from the Python result's own time/temperature histories, so the two are trivially alignable.
    """
    time_s = thermal_result.t_grid_s
    offset_K = 5.0
    return ComsolThermalResult(
        time_s=time_s,
        T_center_K=thermal_result.T_center_t + offset_K,
        T_flat_edge_K=thermal_result.T_area_average_t + offset_K,
        T_bevel_K=thermal_result.T_bevel_mean_t + offset_K,
        T_max_K=thermal_result.T_max_t + offset_K,
        T_area_average_K=thermal_result.T_area_average_t + offset_K,
        stored_energy_J=np.zeros_like(time_s),
        radiation_loss_W=np.zeros_like(time_s),
        boundary_heat_flow_W=np.zeros_like(time_s),
        mesh_id="synthetic_test_mesh",
        time_step_id="synthetic_test_step",
    )


@pytest.fixture()
def tiny_study_with_comsol():
    """Build the Python-only study first (to get a real `ThermalResult` to derive a self-consistent
    synthetic COMSOL result from), then rebuild the study WITH that COMSOL result supplied."""
    pilot = _build_tiny_study(comsol_results=None)
    comsol_result = _write_synthetic_comsol_result(None, pilot.thermal_result)
    return _build_tiny_study(comsol_results=comsol_result)


def test_returns_figure_with_six_panels(tiny_study_no_comsol):
    fig = rg.plot_back_bombardment_macropulse(tiny_study_no_comsol)
    try:
        assert isinstance(fig, plt.Figure)
        assert hasattr(fig, "bb_macropulse_panels")
        assert len(fig.bb_macropulse_panels) == 6
        for ax in fig.bb_macropulse_panels:
            assert ax in fig.axes
    finally:
        plt.close(fig)


def test_no_comsol_result_draws_no_comsol_curves(tiny_study_no_comsol):
    """The single most important behavioral check for this figure (plan Sec. 8.2/12): with
    `comsol_results=None`, `study.comparison.comsol_available` must be False, and the figure must
    not draw any COMSOL curve/dashed line/placeholder -- checked via the deterministic
    `fig.macropulse_comsol_curves_plotted` flag AND the temperature panel's own title text (a real,
    substantive check, not merely "the function did not raise").
    """
    assert tiny_study_no_comsol.comparison.comsol_available is False
    assert tiny_study_no_comsol.comsol_result is None

    fig = rg.plot_back_bombardment_macropulse(tiny_study_no_comsol)
    try:
        assert hasattr(fig, "macropulse_comsol_curves_plotted")
        assert fig.macropulse_comsol_curves_plotted is False

        ax_temperature = fig.bb_macropulse_panels[2]
        title = ax_temperature.get_title().lower()
        assert "python only" in title
        assert "comsol unavailable" in title

        # No line in the temperature panel may be dashed (this figure's own dashed-line
        # convention is reserved exclusively for COMSOL curves -- see _panel_temperature).
        for line in ax_temperature.get_lines():
            assert line.get_linestyle() not in ("--", "dashed")

        # No legend entry may mention COMSOL.
        legend = ax_temperature.get_legend()
        if legend is not None:
            legend_texts = [t.get_text().lower() for t in legend.get_texts()]
            assert not any("comsol" in t for t in legend_texts)
    finally:
        plt.close(fig)


def test_comsol_result_supplied_draws_comsol_curves(tiny_study_with_comsol):
    """The mirror-image check: when a COMSOL result IS supplied and comparison succeeds, the
    figure DOES draw the dashed COMSOL curves and labels them accordingly -- confirming the
    no-COMSOL test above is testing a real conditional branch, not a function that never draws
    COMSOL curves at all.
    """
    assert tiny_study_with_comsol.comparison.comsol_available is True
    assert tiny_study_with_comsol.comsol_result is not None

    fig = rg.plot_back_bombardment_macropulse(tiny_study_with_comsol)
    try:
        assert fig.macropulse_comsol_curves_plotted is True
        ax_temperature = fig.bb_macropulse_panels[2]
        title = ax_temperature.get_title().lower()
        assert "comsol" in title
        assert "python" in title

        dashed_lines = [
            line for line in ax_temperature.get_lines() if line.get_linestyle() in ("--", "dashed")
        ]
        assert len(dashed_lines) >= 1

        legend = ax_temperature.get_legend()
        assert legend is not None
        legend_texts = [t.get_text().lower() for t in legend.get_texts()]
        assert any("comsol" in t for t in legend_texts)
    finally:
        plt.close(fig)


def test_does_not_raise_and_panels_have_titles(tiny_study_no_comsol):
    fig = rg.plot_back_bombardment_macropulse(tiny_study_no_comsol)
    try:
        ax1, ax2, ax3, snap0, snap1, snap2 = fig.bb_macropulse_panels
        assert "current" in ax1.get_title().lower()
        assert "power" in ax2.get_title().lower()
        assert "temperature" in ax3.get_title().lower()
        for ax in (snap0, snap1, snap2):
            assert "t/" in ax.get_title().lower() or "tau" in ax.get_title().lower()
    finally:
        plt.close(fig)


def test_print_summary_does_not_raise(tiny_study_no_comsol, capsys):
    rg.print_back_bombardment_macropulse_summary(tiny_study_no_comsol)
    out = capsys.readouterr().out
    assert "energy_residual_normalized" in out
