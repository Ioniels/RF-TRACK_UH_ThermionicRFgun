"""Emission-model registry and direct-integral reference tests, per
RF_TRACK_2_7_SELF_CONSISTENT_EMISSION_IMPLEMENTATION_GUIDE.md Sec. 16.2.
"""
from __future__ import annotations

import numpy as np
import pytest

import rf_gun as rg
from rf_gun.emission_models import J_rld_schottky, J_mg0_sn


def test_unified_alias_resolves_to_jensen2014_additive():
    """`unified` and the pre-refactor `rld_schottky_plus_mg` name are both permanent
    backward-compatible aliases for the current canonical name (see EMISSION_MODEL_ALIASES)."""
    assert rg.canonical_emission_model_name("unified") == "jensen2014_RDSchottky_MurphyGood_additive"
    assert rg.canonical_emission_model_name("rld_schottky_plus_mg") == "jensen2014_RDSchottky_MurphyGood_additive"
    assert rg.canonical_emission_model_name("jensen2014_RDSchottky_MurphyGood_additive") == "jensen2014_RDSchottky_MurphyGood_additive"


def test_old_model_names_still_resolve_via_backward_compatible_aliases():
    assert rg.canonical_emission_model_name("RD_schottky") == "RDSchottky"
    assert rg.canonical_emission_model_name("rgtf_2019") == "jensen2019_RDSchottky_MurphyGood_transition"
    assert rg.canonical_emission_model_name("murphy_good_direct_reference") == "murphygood1956_SchottkyNordheim_integral"


def test_first_pass_renamed_names_still_resolve_via_backward_compatible_aliases():
    """This session's earlier (author_papercode_year) intermediate renaming, now itself
    superseded by the user-specified final names -- both must keep working."""
    assert rg.canonical_emission_model_name("richardson_dushman_schottky") == "RDSchottky"
    assert rg.canonical_emission_model_name("schottky_murphygood_additive_legacy") == "jensen2014_RDSchottky_MurphyGood_additive"
    assert rg.canonical_emission_model_name("jensen_rgtf_2019") == "jensen2019_RDSchottky_MurphyGood_transition"
    assert rg.canonical_emission_model_name("murphy_good_1956_integral") == "murphygood1956_SchottkyNordheim_integral"


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        rg.canonical_emission_model_name("not_a_model")


def test_RD_schottky_and_rld_schottky_plus_mg_registry_match_bare_functions():
    F = np.array([1e6, 5e6, 2e7])
    T_K, phi_eV = 1700.0, 2.1
    res = rg.evaluate_emission_model("RD_schottky", F, T_K, phi_eV)
    np.testing.assert_allclose(res.J_Apm2, J_rld_schottky(F, T_K, phi_eV))

    res_alias = rg.evaluate_emission_model("unified", F, T_K, phi_eV)
    res_canon = rg.evaluate_emission_model("rld_schottky_plus_mg", F, T_K, phi_eV)
    np.testing.assert_allclose(res_alias.J_Apm2, res_canon.J_Apm2)


def test_jensen_gtf_2007_not_implemented_by_design():
    """Kept as historical/mathematical comparison per the implementation guide --
    jensen2019_RDSchottky_MurphyGood_transition (the preferred production GTF model) is
    implemented and tested in tests/test_rgtf_2019.py."""
    with pytest.raises(NotImplementedError):
        rg.evaluate_emission_model("jensen_gtf_2007", np.array([1e6]), 1700.0, 2.1)


@pytest.mark.parametrize("F_Vpm", [1.0e5, 5.0e5])
def test_direct_integral_reduces_to_RD_schottky_at_weak_field_high_T(F_Vpm):
    """Guide Sec. 16.2 #3: the direct integral should tend toward the RLD-Schottky limit at
    weak field / high temperature (deep thermionic regime, negligible tunneling)."""
    T_K, phi_eV = 2000.0, 2.1
    J_direct = rg.J_murphy_good_direct_reference(F_Vpm, T_K, phi_eV)
    J_rd = float(J_rld_schottky(np.array([F_Vpm]), T_K, phi_eV)[0])
    ratio = J_direct / J_rd
    print(f"F={F_Vpm:.2e} V/m: J_direct={J_direct:.4e}, J_RD={J_rd:.4e}, ratio={ratio:.4f}")
    assert 0.8 < ratio < 1.25, f"direct integral off from RD-Schottky by more than 25% at weak field: ratio={ratio:.3f}"


def test_direct_integral_reduces_to_murphy_good_at_low_T_high_F():
    """Guide Sec. 16.2 #4: the direct integral should tend toward the Murphy-Good field-emission
    limit at low temperature / high field (deep tunneling regime, thermionic tail negligible).

    J_mg0_sn() is the standard closed-form Fowler-Nordheim/Murphy-Good result, itself derived by
    *linearizing* the Gamow exponent about En=mu (the Fermi level) before integrating -- this is
    the same simplification the Jensen 2007/2019 papers identify as causing a several-percent
    error in the exponent (and hence, since J~exp(-const/F), an O(1) multiplicative error in J)
    relative to a full non-linearized integral. This direct-integral reference does *not*
    linearize -- it evaluates the exact SN-elliptic Gamow factor at each En. A multiplicative gap
    of a few x in the deep-field regime is therefore an expected, reported difference between an
    exact and a linearized treatment, not evidence the two disagree in physical regime -- the
    weak-field RD-Schottky check above (which shares the same non-linearized-vs-not distinction
    but where the two treatments coincide) is the tight, exact-match test.
    """
    T_K, phi_eV = 300.0, 2.1
    F_Vpm = 6.0e9  # strong field, well into the field-emission-dominated regime
    J_direct = rg.J_murphy_good_direct_reference(F_Vpm, T_K, phi_eV)
    J_mg = float(J_mg0_sn(np.array([F_Vpm]), phi_eV)[0])
    ratio = J_direct / J_mg
    print(f"F={F_Vpm:.2e} V/m, T={T_K} K: J_direct={J_direct:.4e}, J_MG={J_mg:.4e}, ratio={ratio:.4f}")
    assert 0.2 < ratio < 6.0, f"direct integral off from Murphy-Good by more than 6x at strong field: ratio={ratio:.3f}"


def test_direct_integral_finite_and_nonnegative_over_domain():
    F_grid = np.geomspace(1e5, 1e10, 12)
    for F in F_grid:
        J = rg.J_murphy_good_direct_reference(float(F), 1700.0, 2.1)
        assert np.isfinite(J), f"non-finite J at F={F:.2e}"
        assert J >= 0.0, f"negative J at F={F:.2e}"
