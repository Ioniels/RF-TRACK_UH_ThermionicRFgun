"""Sampling of emission phase space (thermal, roughness)."""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from .constants import ME_KG, KB_EV_PER_K, KB_J_PER_K, c, q_e


def sample_pz_flux(
    n: int,
    T_K,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Flux model for normal energy: eps_z ~ Exp(mean=kT).

    `T_K` may be a scalar (uniform cathode temperature) or an array of length n (a per-particle
    temperature, e.g. each particle's own local value from a cathode temperature map T(x,y) --
    the emission site need not be at a uniform temperature once backbombardment/laser heating
    profiles are involved).

    Returns pz [MeV/c] (length n), the population's mean eps_z [eV] (a single diagnostic scalar,
    always -- the average over whatever mix of temperatures was sampled), and the expected mean
    eps_z [eV] (k_B*T_K, matching T_K's own shape: scalar in, scalar out; array in, array out).
    """
    rng = np.random.default_rng() if rng is None else rng

    MeV_c_SI = (1e6 * q_e) / c

    T_arr = np.asarray(T_K, dtype=float)
    eps_z_J = rng.exponential(scale=KB_J_PER_K * T_arr, size=n)
    pz_SI = np.sqrt(2.0 * ME_KG * eps_z_J)
    pz_MeV_c = pz_SI / MeV_c_SI

    mean_eps_eV = float(np.mean(eps_z_J) / q_e) if n > 0 else 0.0
    exp_eps_eV_arr = KB_EV_PER_K * T_arr
    exp_eps_eV = exp_eps_eV_arr if T_arr.ndim > 0 else float(exp_eps_eV_arr)
    return pz_MeV_c, mean_eps_eV, exp_eps_eV


def roughness_slope_rms(Ra_um: float, Re_um: float) -> float:
    """RMS surface slope from sinusoidal roughness, z(x) = a*sin(2*pi*x/lambda), with
    lambda ~ Re (the profile's correlation length, taken as its period) and `Ra_um` the ISO
    arithmetic-mean roughness (matches this project's own "arith. mean roughness height"
    convention -- see UH_gun_tracking_demo.ipynb's RA_UM comment).

    For this profile, Ra = (2/pi)*a exactly, so a = (pi/2)*Ra -- not sqrt(2)*Ra, which is instead
    the amplitude-from-RMS-roughness (Rq) relation, a = sqrt(2)*Rq, and would silently treat the
    input as Rq while labeling it Ra. The RMS slope of this profile is (2*pi/lambda)*a/sqrt(2) (an
    extra 1/sqrt(2) from the RMS of cos(.) over one period, distinct from the sinusoid's *peak*
    slope 2*pi*a/lambda) -- this is the standard deviation fed into apply_roughness()'s Gaussian.
    """
    Ra_um = float(Ra_um)
    Re_um = float(Re_um)
    if Ra_um <= 0.0 or Re_um <= 0.0:
        return 0.0
    amp_um = (np.pi / 2.0) * Ra_um  # ISO Ra (arithmetic mean) -> sinusoid amplitude
    slope_peak = (2.0 * np.pi * amp_um) / Re_um
    return float(slope_peak / np.sqrt(2.0))  # peak slope -> RMS slope


def apply_roughness(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    Ra_um: float,
    Re_um: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Small-angle rotation from local surface slopes.
    px <- px + pz*theta_x, py <- py + pz*theta_y.
    """
    rng = np.random.default_rng() if rng is None else rng
    sigma_theta = roughness_slope_rms(Ra_um, Re_um)
    if sigma_theta <= 0.0:
        return px, py, 0.0
    theta_x = rng.normal(0.0, sigma_theta, size=px.size)
    theta_y = rng.normal(0.0, sigma_theta, size=py.size)
    px = px + pz * theta_x
    py = py + pz * theta_y
    return px, py, float(sigma_theta)


def sample_thermionic_momenta(
    n: int,
    T_K,
    pz0_MeV_c: float,
    pz_model: Literal["constant", "flux"] = "flux",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Maxwellian transverse momenta with optional flux-normal pz.
    pz_model='flux' samples eps_z ~ Exp(kT); 'constant' uses pz0_MeV_c.

    `T_K` may be a scalar or a length-n array (one local temperature per particle -- see
    sample_pz_flux's docstring for why: a cathode temperature map T(x,y) need not be uniform).
    """
    rng = np.random.default_rng() if rng is None else rng

    T_arr = np.asarray(T_K, dtype=float)
    sigma_v = np.sqrt(KB_J_PER_K * T_arr / ME_KG)
    sigma_p_SI = ME_KG * sigma_v

    MeV_c_SI = (1e6 * q_e) / c
    sigma_p_MeV_c = sigma_p_SI / MeV_c_SI

    px = rng.normal(0.0, sigma_p_MeV_c, size=n)
    py = rng.normal(0.0, sigma_p_MeV_c, size=n)

    if pz_model == "flux":
        pz, mean_eps_eV, exp_eps_eV = sample_pz_flux(n, T_arr, rng=rng)
    elif pz_model == "constant":
        pz = np.full(n, float(pz0_MeV_c))
        mean_eps_eV = np.nan
        exp_eps_eV_arr = KB_EV_PER_K * T_arr
        exp_eps_eV = exp_eps_eV_arr if T_arr.ndim > 0 else float(exp_eps_eV_arr)
    else:
        raise ValueError(f"Unknown pz_model: {pz_model}")
    return px, py, pz, float(mean_eps_eV), exp_eps_eV
