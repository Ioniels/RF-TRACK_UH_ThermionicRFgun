"""Sampling of emission phase space (thermal, roughness)."""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from .constants import c, q_e


def sample_pz_flux(
    n: int,
    T_K: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Flux model for normal energy: eps_z ~ Exp(mean=kT).
    Returns pz [MeV/c], mean eps_z [eV], expected mean [eV].
    """
    rng = np.random.default_rng() if rng is None else rng

    kB_J_per_K = 1.380649e-23
    kB_eV_per_K = 8.617333262e-5
    me_kg = 9.1093837015e-31
    MeV_c_SI = (1e6 * q_e) / c

    eps_z_J = rng.exponential(scale=kB_J_per_K * T_K, size=n)
    pz_SI = np.sqrt(2.0 * me_kg * eps_z_J)
    pz_MeV_c = pz_SI / MeV_c_SI

    mean_eps_eV = float(np.mean(eps_z_J) / q_e) if n > 0 else 0.0
    exp_eps_eV = float(kB_eV_per_K * T_K)
    return pz_MeV_c, mean_eps_eV, exp_eps_eV


def roughness_slope_rms(Ra_um: float, Re_um: float) -> float:
    """
    RMS surface slope from sinusoidal roughness.
    Assume a ~ sqrt(2)*Ra and lambda ~ Re.
    """
    Ra_um = float(Ra_um)
    Re_um = float(Re_um)
    if Ra_um <= 0.0 or Re_um <= 0.0:
        return 0.0
    amp_um = np.sqrt(2.0) * Ra_um
    return float((2.0 * np.pi * amp_um) / Re_um)


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
    T_K: float,
    pz0_MeV_c: float,
    pz_model: Literal["constant", "flux"] = "flux",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Maxwellian transverse momenta with optional flux-normal pz.
    pz_model='flux' samples eps_z ~ Exp(kT); 'constant' uses pz0_MeV_c.
    """
    rng = np.random.default_rng() if rng is None else rng

    kB_J_per_K = 1.380649e-23
    me_kg = 9.1093837015e-31

    sigma_v = np.sqrt(kB_J_per_K * T_K / me_kg)
    sigma_p_SI = me_kg * sigma_v

    MeV_c_SI = (1e6 * q_e) / c
    sigma_p_MeV_c = sigma_p_SI / MeV_c_SI

    px = rng.normal(0.0, sigma_p_MeV_c, size=n)
    py = rng.normal(0.0, sigma_p_MeV_c, size=n)

    if pz_model == "flux":
        pz, mean_eps_eV, exp_eps_eV = sample_pz_flux(n, T_K, rng=rng)
    elif pz_model == "constant":
        pz = np.full(n, float(pz0_MeV_c))
        mean_eps_eV = np.nan
        exp_eps_eV = float(8.617333262e-5 * T_K)
    else:
        raise ValueError(f"Unknown pz_model: {pz_model}")
    return px, py, pz, float(mean_eps_eV), float(exp_eps_eV)
