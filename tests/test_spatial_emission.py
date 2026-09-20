"""Tests for the spatially-resolved production emission path (implementation guide Sec. 6.4/13.6):
rf_gun.simulation.build_cathode_rf_source / build_bunch_thermionic_spatial.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rf_gun as rg
from rf_gun.simulation import EmissionParams, build_cathode_rf_source, build_bunch_thermionic_spatial

FIELD_MAPS_DIR = Path(__file__).resolve().parent.parent / "field_maps"
XY_PATH = FIELD_MAPS_DIR / "XYplanarSensorData.mat"
YZ_PATH = FIELD_MAPS_DIR / "YZplanarSensorData.mat"

pytestmark = pytest.mark.skipif(
    not (XY_PATH.exists() and YZ_PATH.exists()), reason="project field maps not available"
)


def _small_field_grids():
    """A synthetic Er/Ez grid with genuine radial variation, avoiding the ~90s cost of loading
    and interpolating the real .mat field maps for a test that only needs *some* spatial
    dependence (RF-Track's underlying field map is axisymmetric in (r,z), so a real (x,y) point
    still sees this radial profile through r=sqrt(x^2+y^2))."""
    nz, nr = 41, 8
    z_grid_m = np.linspace(0.0, 0.03, nz)
    r_grid_m = np.linspace(0.0, 0.005, nr)
    Ez = np.zeros((nz, nr), dtype=complex)
    for j, r in enumerate(r_grid_m):
        Ez[:, j] = -8.0e6 * (1.0 + 1.5 * (r / r_grid_m[-1]) ** 2) + 0j
    Er = np.zeros((nz, nr), dtype=complex)
    return Er, Ez, r_grid_m, z_grid_m


def _params(**overrides):
    defaults = dict(
        cathode_radius_mm=0.5, cathode_T_K=1700.0, work_function_eV=2.1,
        beta_field=1.0, emission_phase_range_deg=20.0, pz0_MeV_c=1e-4,
        pz_model="flux", emission_law="RD_schottky",
    )
    defaults.update(overrides)
    return EmissionParams(**defaults)


def _volume_params(r_grid_m, z_grid_m):
    return rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=0.0, z_max_m=0.03,
        hr_m=float(r_grid_m[1]), hz_m=float(z_grid_m[1]), dt_mm=0.5,
    )


def test_build_cathode_rf_source_reflects_spatial_field_variation():
    Er, Ez, r_grid_m, z_grid_m = _small_field_grids()
    source = build_cathode_rf_source(
        rg.rft, Er, Ez, phi_deg=0.0, volume_params=_volume_params(r_grid_m, z_grid_m),
        params=_params(), n_x_bins=6, n_y_bins=6, n_time_bins=10,
    )
    assert source["J_Apm2"].shape == (6, 6, 10)
    assert source["temperature_K"].shape == (6, 6)

    x_mm = source["x_grid_m"] * 1e3
    y_mm = source["y_grid_m"] * 1e3
    iy0 = int(np.argmin(np.abs(y_mm)))
    ix_center = int(np.argmin(np.abs(x_mm)))
    ix_edge = int(np.argmax(np.abs(x_mm)))
    J_center = source["J_Apm2"][ix_center, iy0, :]
    J_edge = source["J_Apm2"][ix_edge, iy0, :]
    # The field (and hence J) genuinely grows with radius in this synthetic map -- confirm the
    # sampler actually picked that up rather than falling back to a radius-independent value.
    assert np.all(J_edge > J_center), "J(x,y,t) did not reflect the synthetic field's radial growth"


def test_build_bunch_thermionic_spatial_matches_prescribed_charge():
    Er, Ez, r_grid_m, z_grid_m = _small_field_grids()
    params = _params()
    source = build_cathode_rf_source(
        rg.rft, Er, Ez, phi_deg=0.0, volume_params=_volume_params(r_grid_m, z_grid_m),
        params=params, n_x_bins=6, n_y_bins=6, n_time_bins=10,
    )
    dx_mm = source["x_grid_m"][1] * 1e3 - source["x_grid_m"][0] * 1e3
    dy_mm = source["y_grid_m"][1] * 1e3 - source["y_grid_m"][0] * 1e3
    cell_area_m2 = (dx_mm * 1e-3) * (dy_mm * 1e-3)
    dt_s = float(source["t_grid_s"][1] - source["t_grid_s"][0])
    # build_cathode_rf_source already zeroes J outside the cathode disk, so this exactly matches
    # the total charge build_bunch_thermionic_spatial itself integrates (no approximation needed).
    expected_Q_C = float(np.sum(source["J_Apm2"]) * cell_area_m2 * dt_s)

    B0, info = build_bunch_thermionic_spatial(rg.rft, 2000, source, params=params, rng=np.random.default_rng(1))
    assert B0.size() == 2000
    assert info["Q_total_C"] == pytest.approx(expected_Q_C, rel=1e-6)


def test_build_bunch_thermionic_spatial_samples_more_particles_where_field_is_stronger():
    """Confirms the joint (x,y,t) sampling is actually weighted by J(x,y,t), not just uniform."""
    Er, Ez, r_grid_m, z_grid_m = _small_field_grids()
    params = _params()
    source = build_cathode_rf_source(
        rg.rft, Er, Ez, phi_deg=0.0, volume_params=_volume_params(r_grid_m, z_grid_m),
        params=params, n_x_bins=6, n_y_bins=6, n_time_bins=10,
    )
    B0, info = build_bunch_thermionic_spatial(rg.rft, 20_000, source, params=params, rng=np.random.default_rng(2))

    r_sampled_mm = np.hypot(info["initial_phase_space"][:, 0], info["initial_phase_space"][:, 2])
    r_cathode_mm = params.cathode_radius_mm
    n_outer = np.sum(r_sampled_mm > 0.8 * r_cathode_mm)
    n_inner = np.sum(r_sampled_mm < 0.2 * r_cathode_mm)
    outer_annulus_area = np.pi * (r_cathode_mm ** 2 - (0.8 * r_cathode_mm) ** 2)
    inner_annulus_area = np.pi * (0.2 * r_cathode_mm) ** 2
    density_outer = n_outer / outer_annulus_area
    density_inner = n_inner / inner_annulus_area
    # Stronger field at the edge (guide's synthetic 1+1.5*(r/R)^2 profile) should give higher
    # areal particle density near the edge than near the center.
    assert density_outer > density_inner


def test_build_bunch_thermionic_spatial_honors_temperature_map():
    """An asymmetric T(x,y) should shift each sampled particle's own thermal momentum spread,
    not just the field-driven emission current (guide's asymmetric-cathode-temperature upgrade)."""
    Er, Ez, r_grid_m, z_grid_m = _small_field_grids()
    params = _params(cathode_T_K=1700.0)
    source = build_cathode_rf_source(
        rg.rft, Er, Ez, phi_deg=0.0, volume_params=_volume_params(r_grid_m, z_grid_m),
        params=params, n_x_bins=6, n_y_bins=6, n_time_bins=10,
        temperature_field_K=lambda x_mm, y_mm: 1500.0 + 800.0 * (x_mm > 0.0),
    )
    assert np.all(source["temperature_K"][source["x_grid_m"] > 0.0] == pytest.approx(2300.0))
    assert np.all(source["temperature_K"][source["x_grid_m"] < 0.0] == pytest.approx(1500.0))

    B0, info = build_bunch_thermionic_spatial(rg.rft, 20_000, source, params=params, rng=np.random.default_rng(4))
    x_sampled_mm = info["initial_phase_space"][:, 0]
    px_hot = info["initial_phase_space"][x_sampled_mm > 0.0, 1]
    px_cold = info["initial_phase_space"][x_sampled_mm < 0.0, 1]
    # Hotter side has a larger thermal transverse-momentum spread (sigma_p ~ sqrt(T)).
    assert np.std(px_hot) > np.std(px_cold)


def test_prescribed_source_from_emission_iteration_shape_is_accepted():
    """Guide Sec. 13.6: accept a converged EmissionFieldIterationResult's own J(x,y,t) directly."""
    Er, Ez, r_grid_m, z_grid_m = _small_field_grids()
    rf_axis_phasor = complex(-8.0e6, 0.0)
    vol_params = _volume_params(r_grid_m, z_grid_m)
    params = _params(cathode_T_K=1900.0, work_function_eV=1.8, emission_phase_range_deg=40.0)
    cfg = rg.EmissionFieldIterationConfig(
        max_iterations=2, n_x_bins=4, n_y_bins=4, n_time_bins=6,
        z_max_m=2.0e-3, sc_nx=16, sc_ny=16, sc_nz=16, include_mirror=True, mirror_z_m=0.0,
    )
    iteration_result = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, params, cfg, phi_deg=0.0)

    prescribed_source = {
        "x_grid_m": iteration_result.x_grid_m,
        "y_grid_m": iteration_result.y_grid_m,
        "t_grid_s": iteration_result.t_grid_s,
        "J_Apm2": iteration_result.J_history_Apm2[-1],
        "temperature_K": iteration_result.temperature_K,
    }
    B0, info = build_bunch_thermionic_spatial(rg.rft, 500, prescribed_source, params=params, rng=np.random.default_rng(3))
    assert B0.size() == 500
    assert info["Q_total_C"] > 0.0
