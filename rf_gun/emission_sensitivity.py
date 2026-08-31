"""Emission-model comparison and log-sensitivity diagnostics (implementation guide Sec. 13.2/4.3).

Sensitivities are centered finite differences in log space:
    S_F   = d ln J / d ln F     (field sensitivity)
    S_T   = d ln J / d ln T     (temperature sensitivity)
    S_phi = -phi * d ln J / d phi   (work-function sensitivity, guide's sign convention so it
                                     comes out positive: J falls as phi rises)

For RDSchottky these have exact closed forms (guide Sec. 4.3), used here as
unit-test checks on the finite-difference implementation:
    S_F   = dphi(F) / (2 k_B T)
    S_T   = 2 + (phi - dphi(F)) / (k_B T)
    S_phi = phi / (k_B T)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .constants import KB_EV_PER_K
from .emission_models import EMISSION_MODEL_NAMES, delta_phi_schottky_eV, evaluate_emission_model


def rd_schottky_analytic_sensitivities(F_Vpm: np.ndarray, T_K: float, phi_eV: float) -> Dict[str, np.ndarray]:
    """Exact RDSchottky sensitivities (guide Sec. 4.3), for validating the
    finite-difference implementation below rather than trusting it blindly."""
    F = np.asarray(F_Vpm, dtype=float)
    dphi = delta_phi_schottky_eV(F)
    kT_eV = KB_EV_PER_K * T_K
    S_F = dphi / (2.0 * kT_eV)
    S_T = 2.0 + (phi_eV - dphi) / kT_eV
    S_phi = np.full_like(F, phi_eV / kT_eV)
    return {"S_F": S_F, "S_T": S_T, "S_phi": S_phi}


def _central_log_derivative(f, x0: float, rel_step: float) -> float:
    """d ln f / d ln x at x0, via a centered finite difference in log space."""
    dx = rel_step * abs(x0)
    if dx <= 0.0:
        return np.nan
    f_plus = f(x0 + dx)
    f_minus = f(x0 - dx)
    if not (np.isfinite(f_plus) and np.isfinite(f_minus)) or f_plus <= 0.0 or f_minus <= 0.0:
        return np.nan
    return (np.log(f_plus) - np.log(f_minus)) / (np.log(x0 + dx) - np.log(x0 - dx))


def compute_log_sensitivities(
    model: str,
    F_Vpm: np.ndarray,
    T_K: float,
    phi_eV: float,
    *,
    rel_step: float = 1.0e-3,
    j_floor_Apm2: float = 0.0,
    check_step_doubling: bool = True,
) -> Dict[str, Any]:
    """S_F, S_T, S_phi for `model` over the field array F_Vpm, via centered finite differences in
    log space (guide Sec. 4.3). Points where J is below `j_floor_Apm2` are masked to NaN
    ("numerically meaningless sensitivities", guide Sec. 4.4). When `check_step_doubling`, each of
    S_F/S_T/S_phi is independently recomputed at 2*rel_step and flagged unstable where the two
    disagree by more than 5% (guide's own stability check, extended here to all three
    sensitivities -- the original implementation only step-doubled S_F, which meant a model with a
    branched/discontinuous formula -- e.g. jensen2019_RDSchottky_MurphyGood_transition near a thermionic/field-emission regime
    crossing -- could show an unflagged spike in S_T or S_phi even when S_F's own check happened to
    fall on the same branch on both sides of the step).
    """
    F_arr = np.atleast_1d(np.asarray(F_Vpm, dtype=float))
    n = F_arr.size

    def J_of_F(F):
        return float(evaluate_emission_model(model, np.array([F]), T_K, phi_eV).J_Apm2[0])

    def J_of_T(T, F):
        return float(evaluate_emission_model(model, np.array([F]), T, phi_eV).J_Apm2[0])

    def J_of_phi(phi, F):
        return float(evaluate_emission_model(model, np.array([F]), T_K, phi).J_Apm2[0])

    def _unstable(val, val2):
        if not (np.isfinite(val) and np.isfinite(val2)) or abs(val) <= 1e-12:
            return False
        return abs(val2 - val) / abs(val) > 0.05

    S_F = np.full(n, np.nan)
    S_T = np.full(n, np.nan)
    S_phi = np.full(n, np.nan)
    unstable_F = np.zeros(n, dtype=bool)
    unstable_T = np.zeros(n, dtype=bool)
    unstable_phi = np.zeros(n, dtype=bool)
    J_at_F = np.full(n, np.nan)

    for i, F in enumerate(F_arr):
        J0 = J_of_F(F)
        J_at_F[i] = J0
        if not np.isfinite(J0) or J0 <= j_floor_Apm2:
            continue

        S_F[i] = _central_log_derivative(J_of_F, F, rel_step)
        S_T[i] = _central_log_derivative(lambda T: J_of_T(T, F), T_K, rel_step)
        # S_phi = -phi * dlnJ/dphi = -dlnJ/dln(phi) (since dln(phi)=dphi/phi)
        S_phi[i] = -_central_log_derivative(lambda phi: J_of_phi(phi, F), phi_eV, rel_step)

        if check_step_doubling:
            S_F2 = _central_log_derivative(J_of_F, F, 2.0 * rel_step)
            S_T2 = _central_log_derivative(lambda T: J_of_T(T, F), T_K, 2.0 * rel_step)
            S_phi2 = -_central_log_derivative(lambda phi: J_of_phi(phi, F), phi_eV, 2.0 * rel_step)
            unstable_F[i] = _unstable(S_F[i], S_F2)
            unstable_T[i] = _unstable(S_T[i], S_T2)
            unstable_phi[i] = _unstable(S_phi[i], S_phi2)

    unstable_any = unstable_F | unstable_T | unstable_phi

    return {
        "F_Vpm": F_arr, "J_Apm2": J_at_F,
        "S_F": S_F, "S_T": S_T, "S_phi": S_phi,
        "unstable": unstable_F,  # backward-compatible name/meaning: S_F instability only
        "unstable_F": unstable_F, "unstable_T": unstable_T, "unstable_phi": unstable_phi,
        "unstable_any": unstable_any,
        "model": model, "T_K": T_K, "phi_eV": phi_eV, "rel_step": rel_step,
    }


def compare_emission_models(
    models: Sequence[str],
    F_Vpm: np.ndarray,
    T_K: float,
    phi_eV: float,
) -> Dict[str, Any]:
    """J(F) for every model in `models`, plus a charge-weighted error metric (guide Sec. 3.4)
    against the first model in the list, used as the reference."""
    F_arr = np.atleast_1d(np.asarray(F_Vpm, dtype=float))
    J_by_model: Dict[str, np.ndarray] = {}
    errors: Dict[str, Optional[float]] = {}
    reference = models[0]

    for m in models:
        try:
            J_by_model[m] = np.asarray(evaluate_emission_model(m, F_arr, T_K, phi_eV).J_Apm2, dtype=float)
        except NotImplementedError:
            J_by_model[m] = np.full(F_arr.shape, np.nan)

    J_ref = J_by_model[reference]
    denom = float(np.trapezoid(np.abs(J_ref), F_arr)) + 1e-300
    for m in models:
        num = float(np.trapezoid(np.abs(J_by_model[m] - J_ref), F_arr))
        errors[m] = num / denom if np.all(np.isfinite(J_by_model[m])) else None

    return {"F_Vpm": F_arr, "J_by_model": J_by_model, "reference_model": reference, "charge_weighted_error": errors}


def select_operating_field_domain(F_history_Vpm: np.ndarray, pad_fraction: float = 0.1) -> Tuple[float, float, float]:
    """(F_min, F_peak, F_max) actually populated during a run's RF emission window (guide Sec.
    4.3: "mark the field range actually populated in the current run and the peak extraction
    field"), padded by `pad_fraction` on a log scale for plotting."""
    F = np.asarray(F_history_Vpm, dtype=float)
    F = F[np.isfinite(F) & (F > 0.0)]
    if F.size == 0:
        return (np.nan, np.nan, np.nan)
    F_min, F_max, F_peak = float(np.min(F)), float(np.max(F)), float(np.max(F))
    log_span = np.log10(F_max) - np.log10(F_min) if F_max > F_min else 1.0
    pad = pad_fraction * max(log_span, 1e-3)
    return (F_min * 10 ** (-pad), F_peak, F_max * 10 ** pad)
