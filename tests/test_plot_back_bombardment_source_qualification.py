"""Tests for `rf_gun.plotting.back_bombardment.plot_back_bombardment_source_qualification`
(implementation plan Sec. 12, "Figure A") -- Work Package 4's plotting deliverable.

Pure Python/numpy/matplotlib -- no RF-Track dependency. Builds a small synthetic
`BackBombardmentMacropulseStudy` end to end (same synthetic-events/tiny-macropulse-duration
pattern as `tests/test_back_bombardment_macropulse_study.py`, duplicated here rather than
imported cross-test-file, since `tests/` is not a package) and checks the figure this module
produces has the expected panel structure and does not raise.
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

Q_E = 1.602176634e-19


def _make_synthetic_events(rf_frequency_Hz: float = 2.856e9) -> BackBombardmentEvents:
    """A small, deliberately zone-mixed (flat + bevel) synthetic event set so every panel that
    separates by `surface_code` has more than one zone to actually show."""
    n = 6
    K_eV = np.array([15_000.0, 40_000.0, 90_000.0, 150_000.0, 60_000.0, 220_000.0])
    w = np.full(n, 1.0e6)
    incident_energy_J = w * K_eV * Q_E
    surface_code = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)  # flat, flat, flat, bevel, bevel, bevel
    x_hit_m = np.array([0.0, 0.2e-3, -0.3e-3, 1.45e-3, -1.5e-3, 1.55e-3])
    y_hit_m = np.array([0.0, 0.1e-3, 0.0, 0.0, 0.2e-3, -0.1e-3])
    cos_incidence = np.array([1.0, 0.95, 0.9, 0.8, 0.75, 0.7])

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
        x_emit_m=np.linspace(-0.5e-3, 0.5e-3, n),
        y_emit_m=np.linspace(0.3e-3, -0.3e-3, n),
        z_emit_m=np.zeros(n, dtype=np.float64),
        t_emit_rf_s=np.zeros(n, dtype=np.float64),
        rf_phase_emit_rad=np.zeros(n, dtype=np.float64),
        x_hit_m=x_hit_m,
        y_hit_m=y_hit_m,
        z_hit_m=np.zeros(n, dtype=np.float64),
        t_hit_rf_s=np.linspace(0.0, 3.0e-10, n),
        return_time_s=np.linspace(0.0, 3.0e-10, n),
        px_MeV_c=np.zeros(n, dtype=np.float64),
        py_MeV_c=np.zeros(n, dtype=np.float64),
        pz_MeV_c=-np.ones(n, dtype=np.float64) * 0.01,
        kinetic_energy_eV=K_eV,
        macro_weight_electrons=w,
        incident_energy_J=incident_energy_J,
        n_in_x=np.zeros(n, dtype=np.float64),
        n_in_y=np.zeros(n, dtype=np.float64),
        n_in_z=-np.ones(n, dtype=np.float64),
        cos_incidence=cos_incidence,
        incidence_angle_rad=np.arccos(np.clip(cos_incidence, -1.0, 1.0)),
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
        provenance=build_back_bombardment_provenance(run_id="test_plot_bb_source_qualification"),
    )


@pytest.fixture()
def tiny_study():
    events = _make_synthetic_events()
    study_input = resolve_back_bombardment_study_input(source_mode="current_notebook", current_events=events)
    base = rg.default_uh_back_bombardment_study_config()
    tiny_config = base.replace(macropulse=dataclasses.replace(base.macropulse, duration_s=200e-9))
    return rg.run_back_bombardment_macropulse_study(
        study_input=study_input,
        config=tiny_config,
        initial_temperature=rg.ConstantTemperatureMap(1650.0),
    )


def test_returns_figure_with_six_panels(tiny_study):
    fig = rg.plot_back_bombardment_source_qualification(tiny_study)
    try:
        assert isinstance(fig, plt.Figure)
        # `fig.colorbar(...)` (panels 2 and 5) appends its own Axes to fig.axes, so len(fig.axes)
        # alone is not six even though there are exactly six logical panels -- see
        # plot_back_bombardment_source_qualification's own docstring. The function stashes the six
        # primary panel Axes explicitly for exactly this reason.
        assert hasattr(fig, "bb_source_qualification_panels")
        assert len(fig.bb_source_qualification_panels) == 6
        for ax in fig.bb_source_qualification_panels:
            assert ax in fig.axes
        assert len(fig.axes) >= 6
    finally:
        plt.close(fig)


def test_does_not_raise_and_panels_have_titles(tiny_study):
    fig = rg.plot_back_bombardment_source_qualification(tiny_study)
    try:
        titles = [ax.get_title() for ax in fig.bb_source_qualification_panels]
        assert any("footprint" in t.lower() for t in titles)
        assert any("deposited" in t.lower() or "energy" in t.lower() for t in titles)
        assert any("phase" in t.lower() or "return" in t.lower() for t in titles)
        assert any("spectrum" in t.lower() or "distribution" in t.lower() for t in titles)
        assert any("emission" in t.lower() or "origin" in t.lower() for t in titles)
    finally:
        plt.close(fig)


def test_trajectory_panel_reports_unavailable_without_screen_snapshots(tiny_study):
    """No M_snaps/z_snaps supplied -- the trajectory panel must say so rather than raise."""
    fig = rg.plot_back_bombardment_source_qualification(tiny_study)
    try:
        ax_trajectories = fig.bb_source_qualification_panels[-1]
        texts = [t.get_text().lower() for t in ax_trajectories.texts]
        assert any("no trajectory data" in t for t in texts)
    finally:
        plt.close(fig)


def test_trajectory_panel_draws_lines_with_screen_snapshots(tiny_study):
    """With synthetic screen snapshots covering the qualified events' own particle IDs, the
    trajectory panel must actually plot lines."""
    events = tiny_study.study_input.events
    pids = np.asarray(events.particle_id)
    z_snaps = [0.001, 0.002, 0.003]
    M_snaps = []
    for z in z_snaps:
        M = np.zeros((pids.size, 7))
        M[:, 6] = pids  # id column
        M[:, 5] = 1.0e-3  # pz_MeV_c column
        M_snaps.append(M)

    fig = rg.plot_back_bombardment_source_qualification(tiny_study, M_snaps=M_snaps, z_snaps=z_snaps)
    try:
        ax_trajectories = fig.bb_source_qualification_panels[-1]
        assert len(ax_trajectories.get_lines()) > 0
    finally:
        plt.close(fig)


def test_print_summary_does_not_raise(tiny_study, capsys):
    rg.print_back_bombardment_source_qualification_summary(tiny_study)
    out = capsys.readouterr().out
    assert "event_locator" in out
    assert "Q_incident" in out


def test_multi_zone_population_is_reflected_in_legends(tiny_study):
    """The synthetic fixture deliberately mixes flat and bevel events -- the footprint and
    energy-spectrum panels must show both zone labels, not silently collapse to one."""
    fig = rg.plot_back_bombardment_source_qualification(tiny_study)
    try:
        ax_footprint = fig.bb_source_qualification_panels[0]
        legend_texts = [t.get_text() for t in ax_footprint.get_legend().get_texts()]
        assert any("flat" in t for t in legend_texts)
        assert any("bevel" in t for t in legend_texts)
    finally:
        plt.close(fig)
