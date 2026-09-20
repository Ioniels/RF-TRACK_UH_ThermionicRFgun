"""Regression test for rf_gun.emission_sampling.roughness_slope_rms's corrected derivation.

Found during a review requested after this session's broader upgrade work: the previous formula
(amp = sqrt(2)*Ra, then 2*pi*amp/lambda fed directly into a Gaussian as its standard deviation)
conflated Ra (ISO arithmetic-mean roughness) with Rq (RMS roughness), and returned the *peak*
slope of the assumed sinusoidal profile rather than its *RMS* slope despite the function's own
name and its direct use as a Gaussian sigma. Both numerically verified (see the module docstring
of roughness_slope_rms) against a fine-grained sampled sinusoid, not derived from memory.
"""
from __future__ import annotations

import numpy as np
import pytest

from rf_gun.emission_sampling import roughness_slope_rms, apply_roughness


def test_roughness_slope_rms_matches_iso_ra_derivation():
    Ra_um, Re_um = 0.5, 2.0
    expected = (np.pi ** 2 / np.sqrt(2.0)) * Ra_um / Re_um
    assert roughness_slope_rms(Ra_um, Re_um) == pytest.approx(expected, rel=1e-12)


def test_roughness_slope_rms_matches_numerically_sampled_sinusoid_rms_slope():
    """Independent cross-check: sample a fine sinusoid with amplitude a=(pi/2)*Ra directly and
    compare its numerically-measured RMS slope to the closed-form function's output."""
    Ra_um, Re_um = 0.3, 1.7
    a_um = (np.pi / 2.0) * Ra_um
    x = np.linspace(0.0, Re_um, 2_000_000, endpoint=False)
    z_um = a_um * np.sin(2.0 * np.pi * x / Re_um)
    dzdx = np.gradient(z_um, x)
    rms_slope_numeric = np.sqrt(np.mean(dzdx ** 2))
    assert roughness_slope_rms(Ra_um, Re_um) == pytest.approx(rms_slope_numeric, rel=1e-3)


def test_roughness_slope_rms_zero_when_disabled():
    assert roughness_slope_rms(0.0, 1.0) == 0.0
    assert roughness_slope_rms(1.0, 0.0) == 0.0
    assert roughness_slope_rms(-1.0, 1.0) == 0.0


def test_apply_roughness_uses_corrected_sigma():
    rng = np.random.default_rng(7)
    n = 500_000
    px = np.zeros(n)
    py = np.zeros(n)
    pz = np.ones(n)
    Ra_um, Re_um = 0.5, 2.0
    px_out, py_out, sigma_theta = apply_roughness(px, py, pz, Ra_um, Re_um, rng=rng)
    expected_sigma = roughness_slope_rms(Ra_um, Re_um)
    assert sigma_theta == pytest.approx(expected_sigma, rel=1e-12)
    # px_out = pz*theta_x with pz=1, so std(px_out) should match sigma_theta to sampling precision.
    assert np.std(px_out) == pytest.approx(expected_sigma, rel=0.01)
