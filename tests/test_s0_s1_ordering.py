"""Regression for the RF-Track `Volume.set_s0()/set_s1()` ordering question in
`rf_gun.rftrack_volume.build_volume` (see the NOTE above `V.set_s0(...)` there).

The RF-Track 2.7 manual recommends setting s0/s1 only after every element has been added. A
previous attempt to follow that recommendation caused a real production run's phase scan to
converge on a spurious, all-particles-lost crest. This test is the "minimal experiment" that
attempt was missing: it builds two otherwise-identical Volumes from the real project field maps --
one with s0/s1 set before the dynamic aperture/backstop are added (this repository's current,
production order), one with s0/s1 set after -- and tracks a small on-axis-disk phase scan through
both, using the real RF-Track binding.

Result (recorded here, not just in a comment): for this base configuration -- RF field, dynamic
aperture, cathode backstop; space charge, mirror charges, beam loading, deflection, and screens all
off -- both orderings give bit-identical tracked outcomes. The historical divergence must involve
one of the elements this test does not exercise (most likely space charge/mirror, beam loading, or
the deflection UserField, alone or in combination, and/or production-scale particle counts). This
test guards the tested subset so a future change to `build_volume`'s element/ordering logic cannot
silently reintroduce the base-configuration divergence without anyone noticing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rf_gun as rg
from rf_gun.rftrack_volume import VolumeBuildParams, build_volume, _attach_beam_loading_sw

FIELD_MAPS_DIR = Path(__file__).resolve().parent.parent / "field_maps"
XY_PATH = FIELD_MAPS_DIR / "XYplanarSensorData.mat"
YZ_PATH = FIELD_MAPS_DIR / "YZplanarSensorData.mat"

pytestmark = pytest.mark.skipif(
    not (XY_PATH.exists() and YZ_PATH.exists()), reason="project field maps not available"
)


def _build_coarse_grids():
    xy = rg.load_fieldmap_mat(str(XY_PATH), verbose=False)
    yz = rg.load_fieldmap_mat(str(YZ_PATH), verbose=False)

    t_ns = yz["time"].astype(np.float64)
    t_ns = t_ns - t_ns[0]
    ez_rms = np.sqrt(np.mean(yz["Ez"] ** 2, axis=0))
    f_hz = 2.856e9

    x_mm = xy["vertices"][:, 0]
    y_mm = xy["vertices"][:, 1]
    r_m = np.abs(x_mm) * 1e-3
    y_cathode_mm = 12.75
    z_m = (y_cathode_mm - y_mm) * 1e-3

    i0, i90, _, _ = rg.select_iq_snapshots(t_ns, ez_rms, f_hz)
    ex_0, ex_90 = xy["Ex"][:, i0], xy["Ex"][:, i90]
    ey_0, ey_90 = xy["Ey"][:, i0], xy["Ey"][:, i90]
    ey_max_0, ey_max_90 = np.max(np.abs(ey_0)), np.max(np.abs(ey_90))
    ex_max_0, ex_max_90 = np.max(np.abs(ex_0)), np.max(np.abs(ex_90))
    e_ref = max(ey_max_0, ey_max_90)
    ex_phasor = rg.build_iq_phasor(ex_0, ex_90, ex_max_0, ex_max_90, e_ref)
    ey_phasor = rg.build_iq_phasor(ey_0, ey_90, ey_max_0, ey_max_90, e_ref)
    er_vertices = np.sign(x_mm) * ex_phasor
    ez_vertices = ey_phasor

    # Deliberately coarse (10x the 4/13um production grid) -- this test only needs a stable,
    # reproducible field, not production-grade resolution.
    dr_um, dz_um = 40.0, 130.0
    r_max_m = rg.R_CAV_MM * 1e-3
    lambda_m = rg.c / f_hz
    z_max = lambda_m / 4.0 + 0.0075
    z_min = 0.0
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
    return er_grid, ez_grid, hr, hz, z_min, z_max, f_hz


def _make_disk_bunch(n, rng, r_cathode_mm=1.4):
    r = r_cathode_mm * np.sqrt(rng.random(n))
    th = rng.random(n) * 2.0 * np.pi
    M = np.zeros((n, 6))
    M[:, 0] = r * np.cos(th)
    M[:, 2] = r * np.sin(th)
    M[:, 5] = 1e-3
    return rg.rft.Bunch6dT(rg.ME_MEV, 1.0e4, -1.0, M)


def _build_volume_s0_s1_after(rft, Er_grid, Ez_grid, phi_deg, p: VolumeBuildParams):
    """Same element set as `build_volume`, but s0/s1 set AFTER every element is added (the RF-Track
    manual's general recommendation) instead of before -- the ordering under test."""
    import rf_gun.rftrack_volume as rv

    FM = rft.RF_FieldMap_2d(
        Er_grid, Ez_grid, 0.0, 0.0, float(p.hr_m), float(p.hz_m), -1, float(p.f_hz), +1, 1.0, 1.0
    )
    if hasattr(FM, "set_nsteps"):
        FM.set_nsteps(int(p.fm_nsteps))
    if hasattr(FM, "set_tt_nsteps"):
        FM.set_tt_nsteps(int(p.fm_tt_nsteps))
    FM.set_phid(float(phi_deg))
    rv._call_first_available(FM, ("set_t0", "set_ref_time", "set_reference_time"), 0.0)
    _attach_beam_loading_sw(rft, FM, p)

    V = rft.Volume()
    V.add(FM, 0.0, 0.0, float(p.map_z0_m), "entrance")

    nz = int(np.asarray(Ez_grid).shape[0])
    z_grid_m = np.linspace(float(p.z_min_m), float(p.z_max_m), nz)
    dyn_aperture = rv.build_dynamic_aperture(rft, z_grid_m, float(p.aperture_delta_mm))
    V.add(dyn_aperture, 0.0, 0.0, float(z_grid_m[0]), "entrance")

    if p.cathode_backstop_enabled:
        backstop = rv.build_cathode_backstop(rft, thickness_mm=float(p.cathode_backstop_thickness_mm))
        V.add(
            backstop, 0.0, 0.0,
            float(z_grid_m[0]) - float(p.cathode_backstop_thickness_mm) * 1e-3, "entrance",
        )

    s0_m = float(p.z_min_m)
    if p.cathode_backstop_enabled:
        s0_m -= float(p.cathode_backstop_thickness_mm) * 1e-3
    V.set_s0(s0_m)
    V.set_s1(float(p.z_max_m))

    V.dt_mm = float(p.dt_mm)
    if hasattr(V, "cfx_dt_mm"):
        V.cfx_dt_mm = float(p.cfx_dt_mm)
    V.odeint_algorithm = p.ode_algorithm
    V.odeint_epsabs = float(p.ode_epsabs)
    V.t_max_mm = float(p.t_max_mm)
    return V


def test_s0_s1_ordering_matches_for_rf_and_aperture_only():
    er_grid, ez_grid, hr, hz, z_min, z_max, f_hz = _build_coarse_grids()
    params = VolumeBuildParams(
        f_hz=f_hz, map_z0_m=0.0, z_min_m=z_min, z_max_m=z_max, hr_m=hr, hz_m=hz,
        dt_mm=0.05, fm_nsteps=100, fm_tt_nsteps=100, cfx_dt_mm=0.1,
        cathode_backstop_enabled=True,
    )
    phases_deg = [0.0, 45.39, 90.0]

    for phi in phases_deg:
        rng_before = np.random.default_rng(42)
        B0_before = _make_disk_bunch(20, rng_before)
        V_before = build_volume(rg.rft, er_grid, ez_grid, phi, params)
        Bout_before = V_before.track(B0_before)
        M_before = Bout_before.get_phase_space()

        rng_after = np.random.default_rng(42)
        B0_after = _make_disk_bunch(20, rng_after)
        V_after = _build_volume_s0_s1_after(rg.rft, er_grid, ez_grid, phi, params)
        Bout_after = V_after.track(B0_after)
        M_after = Bout_after.get_phase_space()

        assert M_before.shape == M_after.shape, (
            f"phi={phi}: s0/s1-before kept {M_before.shape[0]} particles, "
            f"s0/s1-after kept {M_after.shape[0]} -- orderings now diverge for this configuration"
        )
        if M_before.shape[0]:
            np.testing.assert_allclose(
                M_before[:, 5], M_after[:, 5], atol=1e-9,
                err_msg=f"phi={phi}: pz differs between s0/s1 orderings",
            )
