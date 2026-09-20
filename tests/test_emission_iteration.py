"""Tests for rf_gun.emission_iteration (implementation guide Sec. 16.6)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import rf_gun as rg


def _synthetic_setup(**cfg_overrides):
    nz, nr = 41, 8
    z_grid_m = np.linspace(0.0, 0.03, nz)
    r_grid_m = np.linspace(0.0, 0.005, nr)
    Ez = np.full((nz, nr), -8.0e6 + 0j)
    Er = np.zeros((nz, nr), dtype=complex)
    rf_axis_phasor = complex(-8.0e6, 0.0)

    vol_params = rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=0.0, z_max_m=0.03,
        hr_m=float(r_grid_m[1]), hz_m=float(z_grid_m[1]), dt_mm=0.5,
    )
    emission = rg.EmissionParams(
        cathode_radius_mm=0.5, cathode_T_K=1900.0, work_function_eV=1.8,
        beta_field=1.0, emission_phase_range_deg=40.0, pz0_MeV_c=1.0e-4,
        pz_model="flux", emission_law="RD_schottky",
    )
    defaults = dict(
        max_iterations=6, n_x_bins=4, n_y_bins=4, n_time_bins=6,
        z_max_m=2.0e-3, sc_nx=16, sc_ny=16, sc_nz=16,
        include_mirror=True, mirror_z_m=0.0,
    )
    defaults.update(cfg_overrides)
    cfg = rg.EmissionFieldIterationConfig(**defaults)
    return Er, Ez, rf_axis_phasor, vol_params, emission, cfg


def test_include_beam_loading_without_calibration_fails_fast():
    """include_beam_loading=True with no bl_Q_loaded/bl_r_over_q_ohm_per_m/bl_L_eff_m/bl_Veff_V
    calibration must raise immediately (before any RF-Track call -- rft=None is safe here), not
    silently produce a result identical to include_beam_loading=False. Guards against the exact
    confusion the user hit: a "SC+mirror+BL" figure/case that turned out to be numerically
    identical to "SC+mirror" with no indication BL had no effect."""
    Er, Ez, rf_axis_phasor, vol_params, emission, cfg = _synthetic_setup(
        max_iterations=2, include_beam_loading=True,
    )
    with pytest.raises(ValueError, match="include_beam_loading"):
        rg.run_emission_field_iteration(None, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0)


def test_include_beam_loading_produces_nonzero_finite_field_and_feeds_back():
    """With real calibration, include_beam_loading=True must produce a finite, nonzero E_BL history
    (not silently zero), and the resulting converged current must differ from the BL-off case --
    i.e. E_BL genuinely feeds back into the Picard loop's own emission-law evaluation, not just a
    cosmetic addition to the reported total field."""
    Q_L = 500.0
    r_over_q_ohm_per_m = 4000.0
    L_eff_m = 0.03
    Veff_V = 2.0e6
    Er, Ez, rf_axis_phasor, vol_params, emission, cfg_bl = _synthetic_setup(
        max_iterations=8, include_beam_loading=True,
        bl_Q_loaded=Q_L, bl_r_over_q_ohm_per_m=r_over_q_ohm_per_m, bl_L_eff_m=L_eff_m,
        bl_Veff_V=Veff_V,
    )
    result_bl = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg_bl, phi_deg=0.0)
    E_BL_final = result_bl.E_BL_history_Vpm[-1]
    assert np.all(np.isfinite(E_BL_final))
    assert np.max(np.abs(E_BL_final)) > 0.0

    cfg_no_bl = dataclasses.replace(cfg_bl, include_beam_loading=False)
    result_no_bl = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg_no_bl, phi_deg=0.0)
    np.testing.assert_allclose(result_no_bl.E_BL_history_Vpm[-1], 0.0)
    assert not np.allclose(result_bl.J_history_Apm2[-1], result_no_bl.J_history_Apm2[-1]), (
        "turning include_beam_loading on must change the converged current density -- otherwise "
        "E_BL is not actually feeding back into the self-consistency loop"
    )


def test_beam_loading_disabled_by_default_matches_pre_existing_zero_field():
    """The default config (include_beam_loading=False) must give E_BL_history_Vpm == 0 everywhere,
    a plain regression guard that the new history array doesn't change any BL-off result."""
    Er, Ez, rf_axis_phasor, vol_params, emission, cfg = _synthetic_setup(max_iterations=3)
    result = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0)
    for E_BL in result.E_BL_history_Vpm:
        np.testing.assert_allclose(E_BL, 0.0)


def test_sc_and_mirror_are_not_always_exactly_cancelling():
    """Guards the z=0 mirror-image degeneracy: placing an already-emitted particle exactly at
    z=0 would put its mirror image at the identical point, so source and image would cancel
    exactly and E_SC+E_mirror would always be zero regardless of the true charge state, making the
    whole iteration a no-op. The ballistic z-drift estimate
    (rf_gun.emission_iteration._ballistic_z_drift_mm) must keep E_SC and E_mirror from being
    identically opposite at full floating-point precision."""
    Er, Ez, rf_axis_phasor, vol_params, emission, cfg = _synthetic_setup(max_iterations=2)
    result = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0)
    E_sc = result.E_SC_history_Vpm[-1]
    E_mirror = result.E_mirror_history_Vpm[-1]
    assert np.max(np.abs(E_sc)) > 0.0
    assert not np.allclose(E_sc, -E_mirror, rtol=1e-9), "E_SC and E_mirror exactly cancel -- the z=0 degeneracy bug is back"


def test_iteration_reduces_current_density_residual_over_iterations():
    Er, Ez, rf_axis_phasor, vol_params, emission, cfg = _synthetic_setup(max_iterations=8)
    result = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0)
    eps = [e for e in result.eps_J_history if np.isfinite(e)]
    assert len(eps) >= 3
    # Not necessarily monotonic every step (relaxation adapts), but the tail should be well below
    # the initial residual.
    assert eps[-1] < 0.5 * eps[0]


def test_fixed_seed_gives_identical_residual_history():
    """Guide Sec. 16.6 #2: a fixed random seed must give identical residual histories."""
    Er, Ez, rf_axis_phasor, vol_params, emission, cfg = _synthetic_setup(max_iterations=4, random_seed=123)
    r1 = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0)
    r2 = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0)
    np.testing.assert_array_equal(
        np.nan_to_num(r1.eps_J_history), np.nan_to_num(r2.eps_J_history)
    )
    assert r1.Q_history_C == r2.Q_history_C


def test_zero_space_charge_and_mirror_reduces_to_rf_only_source():
    """Guide Sec. 16.6 #1: with SC disabled, the source should just be the RF-only emission law
    applied once (no iteration dynamics to converge)."""
    Er, Ez, rf_axis_phasor, vol_params, emission, cfg = _synthetic_setup(
        max_iterations=5, include_mirror=False,
    )
    cfg = dataclasses.replace(cfg, include_space_charge=False)
    result = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0)
    # Every iteration's field is RF-only (E_SC=E_mirror=0), so J should stop changing at once.
    np.testing.assert_allclose(result.J_history_Apm2[0], result.J_history_Apm2[-1])


def test_charge_conservation_between_source_weights_and_reported_Q():
    Er, Ez, rf_axis_phasor, vol_params, emission, cfg = _synthetic_setup(max_iterations=3)
    result = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0)
    N_i = result.source_bunch_matrix[:, 8]
    Q_from_weights = float(np.sum(N_i)) * 1.602176634e-19
    assert Q_from_weights == pytest.approx(result.Q_history_C[-1], rel=1e-6)


def test_capability_report_present_and_reflects_real_binding():
    Er, Ez, rf_axis_phasor, vol_params, emission, cfg = _synthetic_setup(max_iterations=1)
    result = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0)
    assert result.capability_report["SpaceCharge_PIC_FreeSpace_available"] is True


def test_asymmetric_temperature_profile_gives_asymmetric_emission():
    """A cathode T(x,y) profile (e.g. from a backbombardment/laser-heating map) must drive a
    genuinely asymmetric current density, not just a radius-dependent one -- the reason this
    module was generalized from an axisymmetric (r,t) grid to a Cartesian (x,y,t) grid."""
    def T_asym(x_mm, y_mm):
        return 1700.0 + 400.0 * (np.asarray(x_mm) / 0.5)

    Er, Ez, rf_axis_phasor, vol_params, emission, cfg = _synthetic_setup(
        max_iterations=1, cathode_temperature_K=T_asym,
    )
    result = rg.run_emission_field_iteration(rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0)
    assert result.temperature_K.shape == (cfg.n_x_bins, cfg.n_y_bins)
    assert not np.allclose(result.temperature_K, result.temperature_K[0, 0])

    J_final = result.J_history_Apm2[-1]
    J_hot_side = J_final[-1, :, :]  # x > 0, hottest per T_asym
    J_cold_side = J_final[0, :, :]  # x < 0, coldest per T_asym
    assert np.sum(J_hot_side) > np.sum(J_cold_side)


def test_nonfinite_iteration_aborts_immediately_with_reason(monkeypatch):
    """A NaN/inf appearing anywhere in J/E/Q mid-iteration must invalidate the run immediately --
    not get smoothed over into the convergence residuals, and not silently fill the remaining
    max_iterations with more corrupted history (RF_GUN_REPOSITORY_UPGRADE_INSTRUCTIONS.md B2)."""
    import rf_gun.emission_iteration as ei

    Er, Ez, rf_axis_phasor, vol_params, emission, cfg = _synthetic_setup(max_iterations=6)

    real_evaluate = ei.evaluate_emission_model
    call_count = {"n": 0}

    def _poison_second_call(*args, **kwargs):
        call_count["n"] += 1
        result = real_evaluate(*args, **kwargs)
        if call_count["n"] == 2:
            J = np.asarray(result.J_Apm2).copy()
            J.ravel()[0] = np.nan
            result = dataclasses.replace(result, J_Apm2=J)
        return result

    monkeypatch.setattr(ei, "evaluate_emission_model", _poison_second_call)

    result = ei.run_emission_field_iteration(
        rg.rft, Er, Ez, rf_axis_phasor, vol_params, emission, cfg, phi_deg=0.0,
    )

    assert result.converged is False
    assert result.failure_reason is not None and "non-finite" in result.failure_reason
    assert "J_new" in result.failure_reason
    # Stopped at the second call (iteration index 1), not run through all max_iterations=6.
    assert len(result.eps_J_history) == 2
    assert not np.all(np.isfinite(result.J_history_Apm2[-1]))
