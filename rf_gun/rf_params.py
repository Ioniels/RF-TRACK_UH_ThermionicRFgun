"""RF parameter helpers for cavity calibration and beam-loading inputs."""

from __future__ import annotations

import numpy as np

from .constants import ME_MEV


def delivered_power_on_resonance(P_fwd_W: float, Q0: float, Qext: float) -> float:
    """On-resonance power delivered to the cavity [W], from the coupling factor beta=Q0/Qext:
    P_del = P_fwd * 4*beta/(1+beta)^2 (matched at beta=1, i.e. Q0=Qext)."""
    beta = Q0 / Qext
    return P_fwd_W * (4.0 * beta / (1.0 + beta) ** 2)


def effective_length_from_abs_ez(z_m: np.ndarray, Ez_complex: np.ndarray, tail_frac: float = 1e-3) -> float:
    """Effective cavity length [m]: the z-span between the `tail_frac` and `1-tail_frac` points of
    the cumulative integral of |Ez(z)|, i.e. the central span containing `1-2*tail_frac` of the
    total on-axis field-amplitude "mass" -- a robust alternative to a hard field-threshold cutoff."""
    Ez_abs = np.abs(Ez_complex)
    cum = np.cumsum((Ez_abs[:-1] + Ez_abs[1:]) * 0.5 * np.diff(z_m))
    cum = np.insert(cum, 0, 0.0)
    cum /= cum[-1]
    z_low = np.interp(tail_frac, cum, z_m)
    z_high = np.interp(1.0 - tail_frac, cum, z_m)
    return float(z_high - z_low)


def veff_from_phase_scan_pz(pz_mean_MeV_c: np.ndarray, pz0_MeV_c: float, me_MeV: float = ME_MEV) -> float:
    """Effective accelerating voltage [V]: the peak kinetic-energy gain (relativistic, from
    `pz0_MeV_c` to each phase-scan point's `pz_mean_MeV_c`) over the phase scan, in eV numerically
    (MeV gain * 1e6). `me_MeV` defaults to this project's own `constants.ME_MEV` so every call site
    agrees on the electron mass rather than each supplying its own literal."""
    Wk_mean = np.sqrt(pz_mean_MeV_c**2 + me_MeV**2) - me_MeV
    Wk0 = np.sqrt(pz0_MeV_c**2 + me_MeV**2) - me_MeV
    dW_max_MeV = float(np.max(Wk_mean - Wk0))
    return dW_max_MeV * 1e6


def r_over_q_per_m(Veff_V: float, P_del_W: float, Q_loaded: float, L_eff_m: float) -> float:
    """Normalized shunt impedance per unit length [Ohm/m]: (R/Q) = Veff^2/(P_del*Q_loaded),
    divided by the cavity's effective length so it can feed RF-Track's `BeamLoadingSW` directly
    (which wants r_over_q as an Ohm/m array, not a lumped per-cavity Ohm value)."""
    R_over_Q = (Veff_V**2) / (P_del_W * Q_loaded)
    return R_over_Q / L_eff_m
