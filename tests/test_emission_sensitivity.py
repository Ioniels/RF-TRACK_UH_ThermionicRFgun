"""Tests for rf_gun.emission_sensitivity (implementation guide Sec. 4.3/16.2 #6)."""
from __future__ import annotations

import numpy as np
import pytest

import rf_gun as rg


def test_finite_difference_matches_analytic_RD_schottky_sensitivities():
    F = np.geomspace(1e5, 1e7, 6)
    T_K, phi_eV = 1700.0, 2.1
    analytic = rg.rd_schottky_analytic_sensitivities(F, T_K, phi_eV)

    for i, Fi in enumerate(F):
        fd = rg.compute_log_sensitivities("RD_schottky", Fi, T_K, phi_eV)
        assert fd["S_F"][0] == pytest.approx(analytic["S_F"][i], rel=1e-3)
        assert fd["S_T"][0] == pytest.approx(analytic["S_T"][i], rel=1e-3)
        assert fd["S_phi"][0] == pytest.approx(analytic["S_phi"][i], rel=1e-3)


def test_sensitivities_stable_under_step_doubling():
    result = rg.compute_log_sensitivities("RD_schottky", np.array([1e6]), 1700.0, 2.1)
    assert not result["unstable"][0]


def test_sensitivity_masks_below_floor():
    # A field so weak J is essentially zero; the floor should mask the sensitivity to NaN.
    result = rg.compute_log_sensitivities("RD_schottky", np.array([1.0]), 300.0, 4.5, j_floor_Apm2=1e-10)
    assert np.isnan(result["S_F"][0])


def test_compare_emission_models_charge_weighted_error_zero_for_self_comparison():
    F = np.geomspace(1e5, 1e7, 10)
    result = rg.compare_emission_models(["RD_schottky", "RD_schottky"], F, 1700.0, 2.1)
    assert result["charge_weighted_error"]["RD_schottky"] == pytest.approx(0.0, abs=1e-12)


def test_compare_emission_models_handles_unimplemented_gracefully():
    F = np.geomspace(1e5, 1e7, 5)
    result = rg.compare_emission_models(["RD_schottky", "jensen_gtf_2007"], F, 1700.0, 2.1)
    assert np.all(np.isnan(result["J_by_model"]["jensen_gtf_2007"]))
    assert result["charge_weighted_error"]["jensen_gtf_2007"] is None


def test_select_operating_field_domain():
    F_hist = np.array([1e6, 3e6, 5e6, np.nan, -1.0])
    F_min, F_peak, F_max = rg.select_operating_field_domain(F_hist)
    assert F_peak == pytest.approx(5e6)
    assert F_min < 1e6
    assert F_max > 5e6
