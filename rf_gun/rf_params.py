"""RF parameter helpers for cavity calibration and beam-loading inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .constants import ME_MEV


@dataclass(frozen=True)
class PhaseCalibrationResult:
    """Typed result of an RF-only phase scan (`rf_gun.simulation.run_phase_scan`).

    `phi_rel_deg`/`phi_abs_deg`/`pz_mean_MeV_c`/`n_ok` are the full scan, sorted by relative phase
    (coarse grid plus the refined crest point, if any) -- `pz_mean_MeV_c` is NaN wherever the
    scan's test bunch was entirely lost or otherwise produced no finite mean forward momentum.
    `valid_intervals` lists contiguous finite-index runs (inclusive `(start, end)` index pairs into
    the sorted arrays), so a caller can plot/analyze coverage without recomputing the mask.

    `valid` requires a resolved crest: it must exist, have finite scan neighbors immediately on
    both sides (an isolated finite point surrounded by NaN is not a resolved local maximum -- see
    `crest_bracketed`), and a positive net momentum gain over `pz0_MeV_c`. `invalid_reason` explains
    a `valid=False` result. Callers must check `valid` before using `crest_*` for `Veff`/`R/Q` or
    any beam-loading-dependent path; an invalid result's `crest_*` fields may still be populated
    (e.g. an isolated, unbracketed finite crest) purely for diagnostic plotting, never for physics.
    """

    phi_rel_deg: np.ndarray
    phi_abs_deg: np.ndarray
    pz_mean_MeV_c: np.ndarray
    n_ok: np.ndarray
    pz0_MeV_c: float
    finite_mask: np.ndarray
    valid_intervals: List[Tuple[int, int]]
    crest_index: Optional[int]
    crest_phi_rel_deg: Optional[float]
    crest_phi_abs_deg: Optional[float]
    crest_pz_mean_MeV_c: Optional[float]
    crest_bracketed: bool
    refined: bool
    valid: bool
    invalid_reason: Optional[str]

    @property
    def valid_fraction(self) -> float:
        return float(np.mean(self.finite_mask)) if self.finite_mask.size else 0.0


def build_phase_calibration_result(
    phi_rel_deg: np.ndarray,
    phi_abs_deg: np.ndarray,
    pz_mean_MeV_c: np.ndarray,
    n_ok: np.ndarray,
    pz0_MeV_c: float,
    refined: bool,
) -> PhaseCalibrationResult:
    """Build and validate a `PhaseCalibrationResult` from a phase scan already sorted by
    `phi_rel_deg` (as `run_phase_scan` produces). Pure numpy -- no RF-Track dependency, so it is
    directly unit-testable against synthetic scans."""
    phi_rel_deg = np.asarray(phi_rel_deg, dtype=float)
    phi_abs_deg = np.asarray(phi_abs_deg, dtype=float)
    pz_mean_MeV_c = np.asarray(pz_mean_MeV_c, dtype=float)
    n_ok = np.asarray(n_ok, dtype=int)
    finite_mask = np.isfinite(pz_mean_MeV_c)

    valid_intervals: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, ok in enumerate(finite_mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            valid_intervals.append((start, i - 1))
            start = None
    if start is not None:
        valid_intervals.append((start, len(finite_mask) - 1))

    crest_index: Optional[int] = None
    crest_bracketed = False
    if np.any(finite_mask):
        crest_index = int(np.nanargmax(pz_mean_MeV_c))
        lo_i, hi_i = crest_index - 1, crest_index + 1
        crest_bracketed = (
            0 <= lo_i < pz_mean_MeV_c.size
            and 0 <= hi_i < pz_mean_MeV_c.size
            and bool(finite_mask[lo_i])
            and bool(finite_mask[hi_i])
        )

    crest_pz = float(pz_mean_MeV_c[crest_index]) if crest_index is not None else None
    invalid_reason: Optional[str] = None
    if crest_index is None:
        invalid_reason = "no finite phase-scan points; every test particle was lost or non-finite"
    elif not crest_bracketed:
        invalid_reason = (
            "crest has no finite bracketing neighbors -- an isolated finite point is not a "
            "resolved local maximum"
        )
    elif not (crest_pz is not None and crest_pz > float(pz0_MeV_c)):
        invalid_reason = f"crest mean pz ({crest_pz!r} MeV/c) is not a net gain over pz0={pz0_MeV_c!r} MeV/c"

    valid = invalid_reason is None

    return PhaseCalibrationResult(
        phi_rel_deg=phi_rel_deg,
        phi_abs_deg=phi_abs_deg,
        pz_mean_MeV_c=pz_mean_MeV_c,
        n_ok=n_ok,
        pz0_MeV_c=float(pz0_MeV_c),
        finite_mask=finite_mask,
        valid_intervals=valid_intervals,
        crest_index=crest_index,
        crest_phi_rel_deg=float(phi_rel_deg[crest_index]) if crest_index is not None else None,
        crest_phi_abs_deg=float(phi_abs_deg[crest_index]) if crest_index is not None else None,
        crest_pz_mean_MeV_c=crest_pz,
        crest_bracketed=crest_bracketed,
        refined=bool(refined),
        valid=valid,
        invalid_reason=invalid_reason,
    )


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
    agrees on the electron mass rather than each supplying its own literal.

    Raises `ValueError` if `pz_mean_MeV_c` is empty or entirely non-finite -- a phase scan where
    every test particle was lost cannot calibrate a voltage, and silently calling `np.max` on an
    all-NaN array (which numpy accepts, returning NaN with only a warning) is exactly how a failed
    scan used to propagate `Veff=NaN` into R/Q, beam loading, and saved run metadata unnoticed.
    Prefer `veff_from_phase_calibration` when a full `PhaseCalibrationResult` is available -- it
    additionally requires the crest to be a resolved (bracketed) local maximum, not merely finite.
    """
    pz_mean_MeV_c = np.asarray(pz_mean_MeV_c, dtype=float)
    finite = np.isfinite(pz_mean_MeV_c)
    if not np.any(finite):
        raise ValueError(
            "veff_from_phase_scan_pz: pz_mean_MeV_c has no finite points (every scan particle was "
            "lost or otherwise non-finite) -- cannot calibrate an effective voltage."
        )
    Wk_mean = np.sqrt(pz_mean_MeV_c[finite] ** 2 + me_MeV**2) - me_MeV
    Wk0 = np.sqrt(pz0_MeV_c**2 + me_MeV**2) - me_MeV
    dW_max_MeV = float(np.max(Wk_mean - Wk0))
    return dW_max_MeV * 1e6


def veff_from_phase_calibration(result: "PhaseCalibrationResult", me_MeV: float = ME_MEV) -> float:
    """Effective accelerating voltage [V] from a validated `PhaseCalibrationResult`'s crest.

    Raises `ValueError` if `result.valid` is False -- callers must not derive `Veff`/`R/Q`/any
    beam-loading input from an unresolved or lost-everywhere phase scan; see
    `PhaseCalibrationResult.invalid_reason` for why.
    """
    if not result.valid:
        raise ValueError(f"veff_from_phase_calibration: invalid phase calibration ({result.invalid_reason}).")
    Wk_crest = np.sqrt(float(result.crest_pz_mean_MeV_c) ** 2 + me_MeV**2) - me_MeV
    Wk0 = np.sqrt(float(result.pz0_MeV_c) ** 2 + me_MeV**2) - me_MeV
    return float((Wk_crest - Wk0) * 1e6)


def r_over_q_per_m(Veff_V: float, P_del_W: float, Q0: float, L_eff_m: float) -> float:
    """Normalized shunt impedance per unit length [Ohm/m]: (R/Q) = Veff^2/(P_del*Q0),
    divided by the cavity's effective length so it can feed RF-Track's `BeamLoadingSW` directly
    (which wants r_over_q as an Ohm/m array, not a lumped per-cavity Ohm value).

    `Q0` must be the cavity's unloaded/intrinsic quality factor, not the loaded `Q_L`, whenever
    `P_del_W` is the power dissipated in the cavity walls (as `delivered_power_on_resonance`
    returns on resonance). This follows directly from the definitions `Q0 = omega*U/P_wall` and
    `R/Q = V^2/(omega*U)` (a purely geometric quantity, independent of any Q): eliminating U gives
    `R/Q = V^2/(P_wall*Q0)` exactly, with no Q_L anywhere in that identity. Passing the loaded Q_L
    here instead of Q0 silently changes the calibrated (R/Q) by the factor Q0/Q_L (order ~2 for
    typical coupling factors), which then propagates directly into `BeamLoadingSW`'s predicted
    beam-induced gradient reduction.
    """
    R_over_Q = (Veff_V**2) / (P_del_W * Q0)
    return R_over_Q / L_eff_m
