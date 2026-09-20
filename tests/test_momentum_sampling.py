"""Momentum-distribution tests for rf_gun.emission_sampling (implementation guide Sec. 16.3).
This sampling code predates this upgrade and was not modified by it, but the guide explicitly
lists these checks and none existed before.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from rf_gun.constants import c, q_e
from rf_gun.emission_sampling import sample_pz_flux, sample_thermionic_momenta

KB_J_PER_K = 1.380649e-23
KB_EV_PER_K = 8.617333262e-5
ME_KG = 9.1093837015e-31
MEV_C_SI = (1.0e6 * q_e) / c


def test_pz_flux_nonnegative_at_creation():
    pz, _, _ = sample_pz_flux(20_000, 1700.0, rng=np.random.default_rng(1))
    assert np.all(pz >= 0.0)


def test_pz_flux_mean_normal_energy_matches_kT():
    """eps_z ~ Exp(kT): sampled mean should converge to k_B*T (guide's own analytic check)."""
    T_K = 1700.0
    rng = np.random.default_rng(2)
    _, mean_eps_eV, exp_eps_eV = sample_pz_flux(200_000, T_K, rng=rng)
    assert exp_eps_eV == pytest_approx_kT(T_K)
    assert mean_eps_eV == pytest_approx_kT(T_K, rel=0.02)


def pytest_approx_kT(T_K, rel=1e-9):
    import pytest
    return pytest.approx(KB_EV_PER_K * T_K, rel=rel)


def test_pz_flux_converges_with_particle_count():
    """Guide Sec. 16.3: verify convergence with particle count -- the sample mean's deviation
    from the analytic k_B*T should shrink as N grows (not merely happen to be close once)."""
    T_K = 1700.0
    exp_eps_eV = KB_EV_PER_K * T_K
    errors = []
    for n in (200, 20_000, 2_000_000):
        _, mean_eps_eV, _ = sample_pz_flux(n, T_K, rng=np.random.default_rng(42))
        errors.append(abs(mean_eps_eV - exp_eps_eV) / exp_eps_eV)
    assert errors[-1] < errors[0], f"relative error did not shrink with N: {errors}"


def test_pz_flux_distribution_matches_exponential_via_ks_test():
    """The normal energy eps_z = pz^2/(2m) should be Exp(k_B T)-distributed; a Kolmogorov-Smirnov
    test against that reference distribution should not reject at a generous threshold."""
    T_K = 1700.0
    rng = np.random.default_rng(3)
    pz, _, _ = sample_pz_flux(50_000, T_K, rng=rng)
    pz_SI = pz * MEV_C_SI
    eps_z_J = pz_SI ** 2 / (2.0 * ME_KG)
    scale = KB_J_PER_K * T_K
    ks_stat, p_value = stats.kstest(eps_z_J, "expon", args=(0.0, scale))
    assert p_value > 0.01, f"sampled eps_z distribution rejects Exp(kT) at p={p_value:.4f} (KS={ks_stat:.4f})"


def test_transverse_momenta_centered_with_thermal_variance():
    """px, py should be zero-mean Maxwellian with sigma_p = sqrt(m_e k_B T) (in MeV/c)."""
    T_K = 1700.0
    rng = np.random.default_rng(4)
    px, py, pz, _, _ = sample_thermionic_momenta(200_000, T_K, pz0_MeV_c=0.0, pz_model="flux", rng=rng)

    expected_sigma_p_MeV_c = np.sqrt(ME_KG * KB_J_PER_K * T_K) / MEV_C_SI

    assert np.mean(px) == pytest_approx_zero(np.std(px), n=200_000)
    assert np.mean(py) == pytest_approx_zero(np.std(py), n=200_000)
    assert np.std(px) == pytest_approx_rel(expected_sigma_p_MeV_c, rel=0.02)
    assert np.std(py) == pytest_approx_rel(expected_sigma_p_MeV_c, rel=0.02)
    assert np.all(pz >= 0.0)


def pytest_approx_zero(sigma, n):
    import pytest
    # Standard error of the mean for n zero-mean-Maxwellian samples.
    return pytest.approx(0.0, abs=5.0 * sigma / np.sqrt(n))


def pytest_approx_rel(value, rel):
    import pytest
    return pytest.approx(value, rel=rel)


def test_constant_pz_model_ignores_flux_sampling():
    px, py, pz, mean_eps_eV, exp_eps_eV = sample_thermionic_momenta(
        1000, 1700.0, pz0_MeV_c=5.0e-3, pz_model="constant", rng=np.random.default_rng(5)
    )
    assert np.all(pz == 5.0e-3)
    assert np.isnan(mean_eps_eV)
