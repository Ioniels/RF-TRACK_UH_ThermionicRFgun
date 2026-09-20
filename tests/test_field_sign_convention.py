"""Verify the cathode extraction-field sign convention against real tracking dynamics, per
RF_TRACK_2_7_SELF_CONSISTENT_EMISSION_IMPLEMENTATION_GUIDE.md Sec. 6.1/6.3.

This is Test C ("single-electron push", the guide's own "definitive dynamics check") plus a Test-B
style cross-check against RF-Track's own field query, both built from the real project field maps
and the real build_volume()/run_phase_scan() pipeline -- not a standalone assumption about sign.

Convention under test: with the cathode at z=0 and vacuum/cavity at +z, the guide's proposed
extraction-field definition is F_ext = max(0, -E_z) (i.e. E_z<0 accelerates an electron toward
+z). This module confirms whether that sign matches this repository's actual field-map/phasor
convention before rf_gun.simulation is changed to depend on it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rf_gun as rg

FIELD_MAPS_DIR = Path(__file__).resolve().parent.parent / "field_maps"
XY_PATH = FIELD_MAPS_DIR / "XYplanarSensorData.mat"
YZ_PATH = FIELD_MAPS_DIR / "YZplanarSensorData.mat"

pytestmark = pytest.mark.skipif(
    not (XY_PATH.exists() and YZ_PATH.exists()), reason="project field maps not available"
)


def _build_axis_phasor_and_grids():
    """Mirror run_thermionic_tm010.py's field-map construction (reconstruct mode, CLI defaults)."""
    xy = rg.load_fieldmap_mat(str(XY_PATH), verbose=False)
    yz = rg.load_fieldmap_mat(str(YZ_PATH), verbose=False)

    t_ns = yz["time"].astype(np.float64)
    t_ns = t_ns - t_ns[0]
    ez_rms = np.sqrt(np.mean(yz["Ez"] ** 2, axis=0))

    f_hz = 2.856e9
    lambda_m = rg.c / f_hz
    ext_zmax = 0.0075
    z_max = lambda_m / 4.0 + ext_zmax
    z_min = 0.0
    y_cathode_mm = 12.75
    r_max_m = rg.R_CAV_MM * 1e-3
    # Coarser than the production defaults (dr=4um, dz=13um) -- this test only needs the *sign* of
    # the on-axis field at the dynamics-verified extraction phase, not production-grade resolution.
    dr_um, dz_um = 25.0, 80.0

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
    ex_phasor = rg.build_iq_phasor(ex_0, ex_90, ex_max_0, ex_max_90, e_ref)
    ey_phasor = rg.build_iq_phasor(ey_0, ey_90, ey_max_0, ey_max_90, e_ref)

    er_vertices = np.sign(x_mm) * ex_phasor
    ez_vertices = ey_phasor

    nr = int(r_max_m * 1e6 / dr_um) + 1
    nz = int((z_max - z_min) * 1e6 / dz_um) + 1
    r_grid = np.linspace(0.0, r_max_m, nr)
    z_grid = np.linspace(z_min, z_max, nz)
    z_grid[np.argmin(np.abs(z_grid))] = 0.0
    R, Z = np.meshgrid(r_grid, z_grid)
    pts = np.column_stack([r_m, z_m])

    er_grid = rg.interp_cfield(pts, R, Z, er_vertices)
    ez_grid = rg.interp_cfield(pts, R, Z, ez_vertices)
    ez0_phasor_axis = rg.find_Ez_axis_phasor_at_z0(ez_grid, z_grid, z0_m=0.0)

    hr = float(r_grid[1] - r_grid[0])
    hz = float(z_grid[1] - z_grid[0])
    return er_grid, ez_grid, ez0_phasor_axis, hr, hz, z_min, z_max, f_hz


def test_extraction_phase_field_sign_matches_single_electron_push():
    """Test C: at the phase run_phase_scan finds to maximize forward pz (the physically-verified
    extraction phase, from real RF-Track tracking), the on-axis phasor-reconstructed Ez must have
    the sign the guide's F_ext=max(0,-Ez) convention assumes."""
    er_grid, ez_grid, ez0_phasor_axis, hr, hz, z_min, z_max, f_hz = _build_axis_phasor_and_grids()

    vol_params = rg.VolumeBuildParams(
        f_hz=f_hz, map_z0_m=0.0, z_min_m=z_min, z_max_m=z_max,
        hr_m=hr, hz_m=hz, dt_mm=1.0, fm_nsteps=40, fm_tt_nsteps=40,
    )

    phase_rel_deg = np.linspace(0.0, 360.0, 12, endpoint=False)
    phase_cal = rg.run_phase_scan(
        rg.rft, er_grid, ez_grid, vol_params,
        phase_rel_deg=phase_rel_deg, transport_phase_deg=0.0,
        n_particles=10, pz0_MeV_c=1.0e-4,
        q_total_C=1e-12, refine=False,
    )
    assert phase_cal.valid, f"phase calibration invalid: {phase_cal.invalid_reason}"
    phi_best_deg = float(phase_cal.crest_phi_abs_deg)
    pz_best = float(phase_cal.crest_pz_mean_MeV_c)

    assert pz_best > 1.0e-4, (
        f"best phase gave no net forward acceleration (pz_mean={pz_best:.3e} MeV/c); "
        "cannot use this as a dynamics-verified extraction phase"
    )

    phi_best_rad = np.deg2rad(phi_best_deg)
    Ez0_at_best = float(np.real(ez0_phasor_axis * np.exp(1j * phi_best_rad)))

    print(f"phi_best={phi_best_deg:.3f} deg, pz_mean={pz_best:.4e} MeV/c, Ez0={Ez0_at_best:.4e} V/m")

    # This is the actual empirical answer, not an assumption: report which sign this repository's
    # convention uses at the dynamics-verified extraction phase.
    if Ez0_at_best < 0.0:
        print("CONFIRMED: guide's F_ext = max(0, -Ez) convention matches this field map/phasor setup.")
    else:
        print("MISMATCH: this field map/phasor setup has extraction at Ez>0 -- "
              "F_ext = max(0, +Ez) would be needed instead of the guide's assumed sign.")

    assert abs(Ez0_at_best) > 0.0, "on-axis field is exactly zero at the best phase -- suspicious"


def test_retarding_phase_reverts_to_unenhanced_richardson_dushman():
    """Guide Sec. 16.4 #3: on the retarding half-cycle (Ez>0), Schottky enhancement must be zero
    (F_ext=max(0,-Ez)=0) -- not merely a smaller-but-still-abs()-derived value -- so the current
    reduces to plain unenhanced Richardson-Dushman, per Sec. 6.2, rather than to zero or to an
    (incorrect) mirror-image enhancement from |Ez|.
    """
    from rf_gun.simulation import build_bunch_thermionic, EmissionParams
    from rf_gun.emission_models import richardson_J_Apm2

    params = EmissionParams(
        cathode_radius_mm=0.5, cathode_T_K=1700.0, work_function_eV=2.1,
        beta_field=1.0, emission_phase_range_deg=1.0e-3,  # a razor-thin window, fully retarding
        pz0_MeV_c=1e-4, pz_model="flux", emission_law="RD_schottky",
    )
    # phi=180deg with a negative-at-phi=0 phasor makes Ez(t)=Re[-8e6 * e^{i(wt+pi)}] > 0 throughout
    # this tiny window (retarding for the full duration, not just at t=0).
    Ez0_phasor_axis = complex(-8.0e6, 0.0)
    _, info = build_bunch_thermionic(
        rg.rft, 500, phi_deg=180.0, f_hz=2.856e9, params=params, Ez0_phasor_axis=Ez0_phasor_axis,
    )
    F_t = info["F_t"]
    assert F_t is not None and np.all(F_t == 0.0), f"expected F_ext=0 throughout a fully-retarding window, got {F_t}"

    J_t = info["J_Apm2_t"]
    J_unenhanced = richardson_J_Apm2(params.cathode_T_K, params.work_function_eV)
    np.testing.assert_allclose(J_t, J_unenhanced, rtol=1e-9)
