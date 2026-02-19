"""Thermionic and field-emission models."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import A_FN, A_RICH, B_FN, EV, KB, epsilon_0, q_e


def delta_phi_schottky_eV(F_Vpm: np.ndarray) -> np.ndarray:
    """Schottky lowering for a local field magnitude [V/m]."""
    F = np.maximum(F_Vpm, 0.0)
    dphi_J = np.sqrt((q_e**3) * F / (4.0 * np.pi * epsilon_0))
    return dphi_J / EV


def schottky_delta_phi_eV(E_Vm: float, beta: float = 1.0) -> float:
    """Schottky lowering dphi [eV] for a local normal field magnitude |E| [V/m]."""
    E = abs(E_Vm) * beta
    dphi_J = np.sqrt((q_e**3) * E / (4.0 * np.pi * epsilon_0))
    return float(dphi_J / q_e)


def richardson_J_Apm2(T_K: float, phi_eff_eV: float) -> float:
    """Richardson-Dushman current density J [A/m^2]."""
    kB_eV_per_K = 8.617333262e-5
    return float(A_RICH * (T_K**2) * np.exp(-phi_eff_eV / (kB_eV_per_K * T_K)))


def emission_window_from_charge(Q_C: float, I_A: float) -> float:
    """Return emission duration tau [s] needed to emit charge Q at current I."""
    if I_A <= 0.0:
        return np.inf
    return float(Q_C / I_A)


def sn_y(F_Vpm: np.ndarray, phi_eV: float) -> np.ndarray:
    """Nordheim parameter y = Delta_phi/phi for SN barrier (0<y<1)."""
    phi = np.maximum(phi_eV, 1e-6)
    y = delta_phi_schottky_eV(np.maximum(F_Vpm, 0.0)) / phi
    return np.clip(y, 0.0, 0.999)


def sn_v(y: np.ndarray) -> np.ndarray:
    """SN barrier correction v(y) using a standard series approximation."""
    y = np.clip(y, 1e-6, 0.999)
    y2 = y * y
    return 1.0 - y2 + (y2 / 6.0) * np.log(y)


def sn_t(y: np.ndarray) -> np.ndarray:
    """SN slope correction t(y) using a standard series approximation."""
    y = np.clip(y, 1e-6, 0.999)
    y2 = y * y
    return 1.0 + (y2 / 9.0) - (y2 / 18.0) * np.log(y)


def J_rld_schottky(F_Vpm: np.ndarray, T_K: float, phi_eV: float, A_R: float = A_RICH) -> np.ndarray:
    """Richardson with Schottky lowering using local field magnitude."""
    dphi = delta_phi_schottky_eV(np.maximum(F_Vpm, 0.0))
    phi_eff = np.maximum(phi_eV - dphi, 1e-6)
    return A_R * (T_K**2) * np.exp(-(phi_eff * EV) / (KB * T_K))


def J_mg0_sn(F_Vpm: np.ndarray, phi_eV: float) -> np.ndarray:
    """Murphy-Good cold field emission with SN corrections."""
    F = np.maximum(F_Vpm, 1.0)
    phi = np.maximum(phi_eV, 1e-6)
    y = sn_y(F, phi)
    v = sn_v(y)
    t = sn_t(y)
    pre = A_FN * (F**2) / phi / np.maximum(t, 1e-12) ** 2
    expo = -B_FN * (phi**1.5) * v / F
    return pre * np.exp(expo)


def lambda_T(p: np.ndarray) -> np.ndarray:
    """Finite-temperature factor lambda_T = (pi p)/sin(pi p)."""
    x = np.pi * np.clip(p, 0.0, 0.999)
    small = x < 1e-3
    out = np.empty_like(x)
    out[small] = 1.0 + (x[small] ** 2) / 6.0
    out[~small] = x[~small] / np.sin(x[~small])
    return out


def beta_slope_eVinv(F_Vpm: np.ndarray, phi_eV: float) -> np.ndarray:
    """Barrier slope beta_slope = dG/dE at Fermi level [1/eV]."""
    F = np.maximum(F_Vpm, 1.0)
    phi = np.maximum(phi_eV, 1e-6)
    t = sn_t(sn_y(F, phi))
    return (B_FN * np.sqrt(phi) * t) / F


def n_regime(F_Vpm: np.ndarray, T_K: float, phi_eV: float) -> Tuple[np.ndarray, np.ndarray]:
    """Regime indicator n = 1/(kT * beta_slope)."""
    beta = beta_slope_eVinv(F_Vpm, phi_eV)
    kT_eV = (KB * T_K) / EV
    p = kT_eV * beta
    n = 1.0 / np.maximum(p, 1e-12)
    return n, p


def J_field_side_gtf(F_Vpm: np.ndarray, T_K: float, phi_eV: float) -> Tuple[np.ndarray, np.ndarray]:
    """Field-side GTF: MG0 * lambda_T with regime indicator n."""
    J0 = J_mg0_sn(F_Vpm, phi_eV)
    n, p = n_regime(F_Vpm, T_K, phi_eV)
    J = J0 * lambda_T(p)
    return J, n


def J_unified(
    F_Vpm: np.ndarray,
    T_K: float,
    phi_eV: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unified emission as additive thermionic + field-side channels."""
    J_th = J_rld_schottky(F_Vpm, T_K, phi_eV)
    J_fe, n = J_field_side_gtf(F_Vpm, T_K, phi_eV)
    J = J_th + J_fe
    return J, n, J_th, J_fe
