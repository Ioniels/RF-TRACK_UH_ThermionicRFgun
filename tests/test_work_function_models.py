"""Tests for rf_gun.work_function_models, validated against
manual_references/LaB6_100_work_function_models.md's own worked table values."""
from __future__ import annotations

import pytest

import rf_gun as rg
from rf_gun.work_function_models import (
    phi_eff_constant,
    phi_eff_linear_tcwf,
    phi_eff_piecewise_surface_evolution,
    evaluate_work_function_eV,
)


def test_constant_model_ignores_temperature():
    assert phi_eff_constant(1000.0) == pytest.approx(2.66)
    assert phi_eff_constant(2000.0) == pytest.approx(2.66)
    assert phi_eff_constant(1500.0, phi_ref_eV=2.5) == pytest.approx(2.5)


@pytest.mark.parametrize("T_K,expected", [
    (1000.0, 2.521), (1200.0, 2.557), (1400.0, 2.593),
    (1600.0, 2.629), (1773.0, 2.660), (1800.0, 2.665), (2000.0, 2.701),
])
def test_linear_tcwf_matches_document_table(T_K, expected):
    assert phi_eff_linear_tcwf(T_K) == pytest.approx(expected, abs=1e-3)


def test_piecewise_matches_document_breakpoints():
    assert phi_eff_piecewise_surface_evolution(1573.0) == pytest.approx(2.525, abs=1e-3)
    assert phi_eff_piecewise_surface_evolution(1673.0) == pytest.approx(2.607, abs=1e-3)


def test_piecewise_continuous_at_breakpoints():
    eps = 1e-6
    lo = phi_eff_piecewise_surface_evolution(1573.0 - eps)
    hi = phi_eff_piecewise_surface_evolution(1573.0 + eps)
    assert lo == pytest.approx(hi, abs=1e-4)
    lo2 = phi_eff_piecewise_surface_evolution(1673.0 - eps)
    hi2 = phi_eff_piecewise_surface_evolution(1673.0 + eps)
    assert lo2 == pytest.approx(hi2, abs=1e-4)


def test_piecewise_matches_liu_high_T_branch():
    # Liu et al. 2017 regression anchor point.
    assert phi_eff_piecewise_surface_evolution(1773.0) == pytest.approx(2.668, abs=1e-3)


def test_evaluate_work_function_eV_registry():
    assert evaluate_work_function_eV("constant_phi_eff", 1700.0) == pytest.approx(2.66)
    assert evaluate_work_function_eV("linear_tcwf", 1773.0) == pytest.approx(2.66)
    assert evaluate_work_function_eV("piecewise_surface_evolution", 1773.0) == pytest.approx(2.668, abs=1e-3)


def test_evaluate_work_function_eV_unknown_model_raises():
    with pytest.raises(ValueError):
        evaluate_work_function_eV("not_a_model", 1700.0)


def test_exported_from_package():
    assert rg.evaluate_work_function_eV("constant_phi_eff", 1700.0) == pytest.approx(2.66)
    assert set(rg.WORK_FUNCTION_MODEL_NAMES) == {"constant_phi_eff", "linear_tcwf", "piecewise_surface_evolution"}
