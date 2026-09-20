"""Binding tests for matched `Bt_grid`/`Bz_grid` input to `build_volume`.

The small electric grids are built from the historical planar fixture only to exercise the
RF-Track constructor cheaply; production E/B values come from the qualified volume artifact.
"""
from __future__ import annotations

from pathlib import Path
import inspect

import numpy as np
import pytest

import rf_gun as rg

FIELD_MAPS_DIR = Path(__file__).resolve().parent.parent / "field_maps"
XY_PATH = FIELD_MAPS_DIR / "XYplanarSensorData.mat"
YZ_PATH = FIELD_MAPS_DIR / "YZplanarSensorData.mat"

pytestmark = pytest.mark.skipif(
    not (XY_PATH.exists() and YZ_PATH.exists()), reason="project field maps not available"
)


def _build_coarse_grids_and_params():
    xy = rg.load_fieldmap_mat(str(XY_PATH), verbose=False)
    yz = rg.load_fieldmap_mat(str(YZ_PATH), verbose=False)

    t_ns = yz["time"].astype(np.float64)
    t_ns = t_ns - t_ns[0]
    ez_rms = np.sqrt(np.mean(yz["Ez"] ** 2, axis=0))

    f_hz = 2.856e9
    lambda_m = rg.c / f_hz
    z_max = lambda_m / 4.0 + 0.0075
    z_min = 0.0
    y_cathode_mm = 12.75
    r_max_m = rg.R_CAV_MM * 1e-3
    dr_um, dz_um = 25.0, 80.0  # coarse -- this test only needs build_volume() to accept B grids

    x_mm = xy["vertices"][:, 0]
    y_mm = xy["vertices"][:, 1]
    r_m = np.abs(x_mm) * 1e-3
    z_m = (y_cathode_mm - y_mm) * 1e-3

    i0, i90, _, _ = rg.select_iq_snapshots(t_ns, ez_rms, f_hz)
    ex_0, ex_90 = xy["Ex"][:, i0], xy["Ex"][:, i90]
    ey_0, ey_90 = xy["Ey"][:, i0], xy["Ey"][:, i90]
    ey_max_0, ey_max_90 = np.max(np.abs(ey_0)), np.max(np.abs(ey_90))
    ex_max_0, ex_max_90 = np.max(np.abs(ex_0)), np.max(np.abs(ex_90))
    e_ref = max(ey_max_0, ey_max_90)
    er_vertices = np.sign(x_mm) * rg.build_iq_phasor(ex_0, ex_90, ex_max_0, ex_max_90, e_ref)
    ez_vertices = rg.build_iq_phasor(ey_0, ey_90, ey_max_0, ey_max_90, e_ref)

    nr = int(r_max_m * 1e6 / dr_um) + 1
    nz = int((z_max - z_min) * 1e6 / dz_um) + 1
    r_grid = np.linspace(0.0, r_max_m, nr)
    z_grid = np.linspace(z_min, z_max, nz)
    z_grid[np.argmin(np.abs(z_grid))] = 0.0
    R, Z = np.meshgrid(r_grid, z_grid)
    pts = np.column_stack([r_m, z_m])

    ctx = rg.build_field_interpolation_context(pts, R, Z)
    er_grid = rg.interp_cfield(pts, R, Z, er_vertices, ctx=ctx)
    ez_grid = rg.interp_cfield(pts, R, Z, ez_vertices, ctx=ctx)

    hr = float(r_grid[1] - r_grid[0])
    hz = float(z_grid[1] - z_grid[0])
    vol_params = rg.VolumeBuildParams(
        f_hz=f_hz, map_z0_m=0.0, z_min_m=z_min, z_max_m=z_max,
        hr_m=hr, hz_m=hz, dt_mm=1.0, fm_nsteps=40, fm_tt_nsteps=40,
    )
    return er_grid, ez_grid, vol_params


def _track_one_cold_particle(er_grid, ez_grid, vol_params, *, Bt_grid=None, Bz_grid=None):
    B0 = rg.build_bunch_on_axis_cold(rg.rft, 1, pz0_MeV_c=1.0e-4, q_total_C=1.0e-12)
    V = rg.build_volume(rg.rft, er_grid, ez_grid, 0.0, vol_params, Bt_grid=Bt_grid, Bz_grid=Bz_grid)
    Bout = V.track(B0)
    return np.array(Bout.get_phase_space("%X %Px %Y %Py %Z %Pz", "all"), copy=True)


def test_build_volume_accepts_none_bfield_by_default():
    er_grid, ez_grid, vol_params = _build_coarse_grids_and_params()
    V = rg.build_volume(rg.rft, er_grid, ez_grid, 0.0, vol_params)
    assert V is not None


def test_build_volume_accepts_explicit_zero_bfield_arrays():
    er_grid, ez_grid, vol_params = _build_coarse_grids_and_params()
    zeros = np.zeros_like(er_grid)
    V = rg.build_volume(rg.rft, er_grid, ez_grid, 0.0, vol_params, Bt_grid=zeros, Bz_grid=zeros)
    assert V is not None


def test_zero_bfield_array_matches_literal_zero_default_bit_for_bit():
    """A zero-valued Bt_grid/Bz_grid must track identically to the current default (the literal
    0.0 RF-Track receives when neither is supplied) -- proving the new parameters are additive and
    do not change any existing run's physics."""
    er_grid, ez_grid, vol_params = _build_coarse_grids_and_params()
    zeros = np.zeros_like(er_grid)

    Mf_default = _track_one_cold_particle(er_grid, ez_grid, vol_params)
    Mf_explicit_zero = _track_one_cold_particle(er_grid, ez_grid, vol_params, Bt_grid=zeros, Bz_grid=zeros)

    assert np.array_equal(Mf_default, Mf_explicit_zero)


def test_only_one_bfield_component_raises_clear_error():
    """Despite the manual's prose suggesting each of Er/Ez/Bt/Bz is independently array-or-zero,
    the actual bound RF_FieldMap_2d overload set has no constructor for one array and one scalar
    within the same pair -- build_volume must reject this itself with a clear message rather than
    let a cryptic SWIG TypeError surface from deep inside RF-Track."""
    er_grid, ez_grid, vol_params = _build_coarse_grids_and_params()
    zeros = np.zeros_like(er_grid)
    with pytest.raises(ValueError, match="Bt_grid and Bz_grid must be supplied together"):
        rg.build_volume(rg.rft, er_grid, ez_grid, 0.0, vol_params, Bt_grid=zeros)


def test_bfield_shape_must_match_electric_field_grid():
    er_grid, ez_grid, vol_params = _build_coarse_grids_and_params()
    bad = np.zeros((er_grid.shape[0], er_grid.shape[1] - 1), dtype=er_grid.dtype)
    with pytest.raises(ValueError, match="must match the electric-field grid shape"):
        rg.build_volume(rg.rft, er_grid, ez_grid, 0.0, vol_params, Bt_grid=bad, Bz_grid=bad)


def test_every_high_level_field_consumer_exposes_keyword_bfield_pair():
    """Guard against a future layer silently dropping B after the artifact loader supplied it."""
    from rf_gun.emission_iteration import run_emission_field_iteration
    from rf_gun.frozen_source_attribution import run_frozen_source_attribution
    from rf_gun.rftrack_volume import track_volume_with_screens
    from rf_gun.simulation import build_cathode_rf_source, run_phase_scan, run_transport_with_progress

    for function in (
        track_volume_with_screens,
        build_cathode_rf_source,
        run_phase_scan,
        run_transport_with_progress,
        run_emission_field_iteration,
        run_frozen_source_attribution,
    ):
        parameters = inspect.signature(function).parameters
        assert "Bt_grid" in parameters, function.__qualname__
        assert "Bz_grid" in parameters, function.__qualname__
        assert parameters["Bt_grid"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["Bz_grid"].kind is inspect.Parameter.KEYWORD_ONLY


def test_track_volume_with_screens_forwards_bfield_pair(monkeypatch):
    import rf_gun.rftrack_volume as volume_module

    captured = {}

    class FakeVolume:
        def track(self, bunch):
            return ("tracked", bunch)

        def get_bunch_at_screens(self):
            return ["screen"]

    def fake_build_volume(*args, **kwargs):
        captured.update(kwargs)
        return FakeVolume()

    monkeypatch.setattr(volume_module, "build_volume", fake_build_volume)
    electric = np.zeros((3, 2), dtype=complex)
    magnetic_t = np.full((3, 2), 1.25e-3, dtype=complex)
    magnetic_z = np.full((3, 2), -2.5e-4, dtype=complex)

    result = volume_module.track_volume_with_screens(
        object(), electric, electric, 12.0, object(), "bunch", [0.01],
        Bt_grid=magnetic_t, Bz_grid=magnetic_z,
    )

    assert result == (("tracked", "bunch"), ["screen"])
    assert captured["Bt_grid"] is magnetic_t
    assert captured["Bz_grid"] is magnetic_z
