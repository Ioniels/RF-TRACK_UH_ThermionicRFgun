"""Cross-validate rf_gun.beam_loading_envelope's causal modal-envelope model against RF-Track's
own production BeamLoadingSW collective effect, on a real (not synthetic) field map, per
UPGRADE_PLAN.md Sec. 6f/7's explicit "still open" item.

**Result of that cross-validation (2026-08-30): it could not be completed as originally planned.**
Real BeamLoadingSW, attached exactly as `rf_gun.rftrack_volume._attach_beam_loading_sw`/
`run_transport_with_progress` already do for the production `--beam_loading` CLI path, produces
**zero measurable effect** on the tracked bunch's dynamics for this project's gun geometry --
confirmed repeatedly, not just once:

- Two independent charge scales (~7 pC-scale vs. a ~1000x larger charge from a much hotter,
  physical-radius cathode) both gave an exit kinetic energy *bit-identical* to the BL-off case.
- Three `cfx_dt_mm` values spanning a 100x range (1.0, 0.1, 0.01 mm/c) all gave the same
  bit-identical result -- ruling out "too coarse a collective-effect step" as the cause.
- A full per-particle check (not just the mean): with the frozen source confirmed bit-identical
  between the BL-on/BL-off cases (same seed, same B0), *every* surviving particle's exit
  kinetic energy and arrival time was bit-identical too, not merely close.

This is not what `rf_gun.cathode_fields.extract_beam_loading_field`'s docstring already flagged
(that a *static* `get_field()` query doesn't reflect BeamLoadingSW's state) -- this is the actual
`V.track()` dynamics themselves showing no effect, for a collective effect the RF-Track 2.7
reference manual describes as exactly this project's own use case ("beam loading in non-
ultrarelativistic scenarios" via `BeamLoadingSW()`, Sec. 6.5.2). The causal-envelope model itself
predicted a small but clearly nonzero effect (~0.5% mean current-weighted chi(t)) for the same
configuration -- so this is not "both predict a negligible effect," it is "one predicts a small
nonzero effect and the other's production implementation shows literally none."

**Conclusion, not a dead end**: this is a real, reproducible, and currently unexplained limitation
of this RF-Track 2.7.0 binding (or this specific gun-geometry/BeamLoadingSW combination) that sits
outside what is fixable from the Python side of this codebase -- it needs escalation (e.g. to
RF-Track/CERN support, or testing against a different RF-Track version) rather than a code change
here. `rf_gun.beam_loading_envelope`'s causal model remains implemented and wired into
`rf_gun.emission_iteration`'s Picard loop exactly as before (validated against its own closed-form
textbook limits -- see `tests/test_beam_loading_envelope.py`); it is specifically the *cross-check
against RF-Track's own BeamLoadingSW* that this module documents as blocked, not the model itself.

This test is a fast (~5s), permanent regression/characterization check of that finding: if a future
RF-Track version (or a configuration fix found later) makes BeamLoadingSW actually change the
tracked dynamics, this test will start failing -- which is the point: it should be *revisited*,
not silently left describing a resolved limitation as if it still applied.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rf_gun as rg
from rf_gun.simulation import EmissionParams, TrackingParams, DiagnosticsParams, run_transport_with_progress

FIELD_MAPS_DIR = Path(__file__).resolve().parent.parent / "field_maps"
XY_PATH = FIELD_MAPS_DIR / "XYplanarSensorData.mat"
YZ_PATH = FIELD_MAPS_DIR / "YZplanarSensorData.mat"

pytestmark = pytest.mark.skipif(
    not (XY_PATH.exists() and YZ_PATH.exists()), reason="project field maps not available"
)

#: Hardcoded from a real (slower, separately-run) phase scan on this project's field maps at
#: crest -- avoids re-running the phase scan here (this test only needs the *presence/absence* of
#: a BeamLoadingSW effect, not a fresh calibration each run). See UPGRADE_PLAN.md Sec. 6f for the
#: full derivation (phase scan + r_over_q_per_m + effective_length_from_abs_ez).
_CREST_PHASE_DEG = 45.797
_Q_0 = 4000.0
_Q_EXT = 3500.0
_Q_L = 1.0 / (1.0 / _Q_EXT + 1.0 / _Q_0)
_R_OVER_Q_OHM_PER_M = 3.886e3


def _build_real_field_map_grids(f_hz: float, dr_um: float = 40.0, dz_um: float = 120.0):
    """Mirror test_field_sign_convention.py's real (non-synthetic) field-map construction, at a
    coarser resolution than that test (this one runs two full transports, not just a phase scan)."""
    xy = rg.load_fieldmap_mat(str(XY_PATH), verbose=False)
    yz = rg.load_fieldmap_mat(str(YZ_PATH), verbose=False)

    t_ns = yz["time"].astype(np.float64)
    t_ns = t_ns - t_ns[0]
    ez_rms = np.sqrt(np.mean(yz["Ez"] ** 2, axis=0))

    lambda_m = rg.c / f_hz
    z_max = lambda_m / 4.0 + 0.0075
    z_min = 0.0
    y_cathode_mm = 12.75
    r_max_m = rg.R_CAV_MM * 1e-3

    x_mm = xy["vertices"][:, 0]
    y_mm = xy["vertices"][:, 1]
    r_m = np.abs(x_mm) * 1e-3
    z_m = (y_cathode_mm - y_mm) * 1e-3

    i0, i90, _, _ = rg.select_iq_snapshots(t_ns, ez_rms, f_hz)
    ex_0, ex_90 = xy["Ex"][:, i0], xy["Ex"][:, i90]
    ey_0, ey_90 = xy["Ey"][:, i0], xy["Ey"][:, i90]
    e_ref = max(np.max(np.abs(ey_0)), np.max(np.abs(ey_90)))
    ex_phasor = rg.build_iq_phasor(ex_0, ex_90, np.max(np.abs(ex_0)), np.max(np.abs(ex_90)), e_ref)
    ey_phasor = rg.build_iq_phasor(ey_0, ey_90, np.max(np.abs(ey_0)), np.max(np.abs(ey_90)), e_ref)

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
    return er_grid, ez_grid, ez0_phasor_axis, hr, hz, z_min, z_max


def test_beam_loading_sw_shows_no_measurable_effect_on_frozen_source():
    """Characterizes (see module docstring) rather than hides: real BeamLoadingSW, attached with a
    physically reasonable Q_L/(R/Q) calibration, produces a bit-identical exit phase space to the
    BL-off case for this gun's frozen source -- confirmed per-particle, not just via the mean."""
    f_hz = 2.856e9
    er_grid, ez_grid, ez0_phasor_axis, hr, hz, z_min, z_max = _build_real_field_map_grids(f_hz)

    emission = EmissionParams(
        cathode_radius_mm=0.5, cathode_T_K=1900.0, work_function_eV=1.8,
        beta_field=1.0, emission_phase_range_deg=60.0, pz0_MeV_c=1.0e-4,
        pz_model="flux", emission_law="RDSchottky",
    )
    tracking = TrackingParams(phi_deg=_CREST_PHASE_DEG, n_particles=300, z_screens_m=None)
    diagnostics = DiagnosticsParams(store_screen_phase_space=False, save_lost_particles=False)

    results = {}
    b0_matrices = {}
    for bl_on in (False, True):
        vol_params = rg.VolumeBuildParams(
            f_hz=f_hz, map_z0_m=0.0, z_min_m=z_min, z_max_m=z_max,
            hr_m=hr, hz_m=hz, dt_mm=1.0, fm_nsteps=60, fm_tt_nsteps=60,
            sc_enabled=False, mirror_charge_enabled=False, beam_loading_enabled=bl_on,
            bl_Q_loaded=_Q_L, bl_r_over_q_ohm_per_m=_R_OVER_Q_OHM_PER_M, bl_ncells=1,
            bl_tinj_mode="auto_from_emission", cfx_dt_mm=1.0, beam_loading_verbose=False,
        )
        rng = np.random.default_rng(123)
        result, _stats = run_transport_with_progress(
            rg.rft, er_grid, ez_grid, ez0_phasor_axis, vol_params, emission, tracking,
            diagnostics=diagnostics, rng=rng,
        )
        results[bl_on] = result
        b0_matrices[bl_on] = np.asarray(result.B0.get_phase_space(tracking.phase_fmt, "all"))

    # Frozen-source sanity check first: if B0 itself already differs, the rest of this test's
    # premise (an apples-to-apples dynamics comparison) doesn't hold.
    np.testing.assert_array_equal(b0_matrices[False], b0_matrices[True])

    M_off = np.asarray(results[False].Bout.get_phase_space(tracking.phase_fmt, "all"))
    M_on = np.asarray(results[True].Bout.get_phase_space(tracking.phase_fmt, "all"))
    assert M_off.shape == M_on.shape

    # id column (%id, index 6 in EXTENDED_PHASE_FMT) may not preserve row order between the two
    # tracked cases even when it's the same particle set -- sort both by id before comparing.
    id_col = 6
    M_off_sorted = M_off[np.argsort(M_off[:, id_col])]
    M_on_sorted = M_on[np.argsort(M_on[:, id_col])]
    np.testing.assert_array_equal(M_off_sorted[:, id_col], M_on_sorted[:, id_col])

    np.testing.assert_array_equal(
        M_off_sorted, M_on_sorted,
        err_msg=(
            "BeamLoadingSW now produces a measurable effect on tracked dynamics for this gun "
            "geometry -- this is a change from the finding documented in this test's module "
            "docstring and in UPGRADE_PLAN.md Sec. 6f/7 (previously: bit-identical exit phase "
            "space). Re-run the cross-validation against rf_gun.beam_loading_envelope's causal "
            "model properly (see the module docstring here) rather than treating this failure as "
            "a simple regression to fix."
        ),
    )
