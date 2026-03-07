"""RF parameter helpers for cavity calibration and beam-loading inputs."""

from __future__ import annotations

import numpy as np


def delivered_power_on_resonance(P_fwd_W: float, Q0: float, Qext: float) -> float:
    beta = Q0 / Qext
    return P_fwd_W * (4.0 * beta / (1.0 + beta) ** 2)


def effective_length_from_abs_ez(z_m: np.ndarray, Ez_complex: np.ndarray, tail_frac: float = 1e-3) -> float:
    Ez_abs = np.abs(Ez_complex)
    cum = np.cumsum((Ez_abs[:-1] + Ez_abs[1:]) * 0.5 * np.diff(z_m))
    cum = np.insert(cum, 0, 0.0)
    cum /= cum[-1]
    z_low = np.interp(tail_frac, cum, z_m)
    z_high = np.interp(1.0 - tail_frac, cum, z_m)
    return float(z_high - z_low)


def veff_from_phase_scan_pz(pz_mean_MeV_c: np.ndarray, pz0_MeV_c: float, me_MeV: float = 0.51099895) -> float:
    Wk_mean = np.sqrt(pz_mean_MeV_c**2 + me_MeV**2) - me_MeV
    Wk0 = np.sqrt(pz0_MeV_c**2 + me_MeV**2) - me_MeV
    dW_max_MeV = float(np.max(Wk_mean - Wk0))
    return dW_max_MeV * 1e6


def r_over_q_per_m(Veff_V: float, P_del_W: float, Q_loaded: float, L_eff_m: float) -> float:
    R_over_Q = (Veff_V**2) / (P_del_W * Q_loaded)
    return R_over_Q / L_eff_m
