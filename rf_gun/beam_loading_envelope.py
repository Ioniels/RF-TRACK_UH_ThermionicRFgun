"""Causal TM010 cavity-mode beam-loading envelope.

UPGRADE_PLAN.md Sec. 4/6d flagged this as the physically correct way to represent beam loading's
effect on the cathode field -- as a time-dependent amplitude of the cavity's own accelerating
mode, `E_BL(r,z,t) = -chi(t)*E_RF(r,z,t)`, evolved causally from the beam current, rather than a
spatially-uniform offset (which the user's own detailed physics review rejected outright: RF-Track's
real `BeamLoadingSW` model gives the beam-induced field the same spatial structure as the
accelerating mode itself, not a uniform correction). This module is a first, standalone
implementation of that causal envelope.

**This is NOT a reproduction of RF-Track's own internal `BeamLoadingSW` discretization** -- that
implementation is compiled/proprietary, and the RF-Track 2.7 manual documents its physical scope
(Sec. 6.5) but not its internal numerical algorithm. This is an independent, from-first-principles
implementation of the same well-established textbook physics, generally known as the "fundamental
theorem of beam loading" (P. B. Wilson, "Fundamentals of RF Superconductivity and Beam Loading",
SLAC-PUB-2884 (1982); see also T. Wangler, "RF Linear Accelerators", 2nd ed., Sec. 4.7), using
this repo's own already-validated R/Q (see UPGRADE_PLAN.md Sec. 6a's R/Q calibration fix), Q_L,
and f_hz conventions so its steady-state limit is directly comparable to this repo's existing
beam-loading calibration -- **not yet cross-checked against RF-Track's own production
`BeamLoadingSW` gradient reduction on a real run**; treat this as an order-of-magnitude estimate
until that comparison is done (see UPGRADE_PLAN.md).

Governing equation (on-resonance unless `detuning_rad_s` is given; SI units throughout, complex
phasor convention):

    dV_b/dt + V_b/tau = -(omega/2)*(R/Q)*I(t),      tau = 2*Q_L/omega

where `V_b(t)` is the complex beam-induced voltage phasor and `R/Q = Veff^2/(omega*U)` is this
repo's own convention (verified in `rf_gun.rf_params.r_over_q_per_m`'s docstring derivation --
*not* the "linac" convention `R/Q = 2*Veff^2/P` some texts use, which would introduce a spurious
factor of 2 here). The steady-state (I(t)=I0 constant, t >> tau) limit,
`V_b,ss = -(R/Q)*Q_L*I0`, is the standard "DC beam loading" formula and is used below as a
self-consistency check on the ODE integrator, not merely asserted.

`chi(t) = V_b(t) / V_eff` is the fractional reduction (complex, so it carries both amplitude and
phase relative to the unloaded RF) applied to the RF field at the cathode:
`E_BL(t) = -chi(t) * E_RF(t)` (real part taken where a real field is needed).

Scope: this solves a *single* RF bucket's causal response, starting from a given initial
condition (default: unloaded, `V_b(t=0)=0`). A thermionic cathode emitting over many RF periods in
a macropulse needs the cavity envelope's own periodic steady state (guide's own two-timescale
discussion: `tau/T_RF` is order 500-600 for this gun's own `Q_L`, so a single-bucket study
starting from zero very likely underestimates the beam loading present later in a real macropulse)
-- that multi-bucket recurrence is future work, not implemented here.

**Now fed back into the Picard iteration.** `rf_gun.emission_iteration.run_emission_field_iteration`
calls `estimate_beam_induced_cathode_field_from_current_density` each outer iteration (via its own
just-updated current-density weights), so `EmissionFieldIterationConfig.include_beam_loading=True`
gives a genuinely self-consistent SC+mirror+BL case (chi(t) responds to the emission current that
has itself already responded to E_BL from the previous iteration), not merely a post-hoc estimate
laid over an already-converged BL-off result. `estimate_beam_induced_cathode_field_map` remains for
forming that illustrative post-hoc estimate specifically when a run did *not* itself include beam
loading. Still not cross-checked against RF-Track's own production `BeamLoadingSW` gradient
reduction on a real run (see UPGRADE_PLAN.md) -- treat this as an order-of-magnitude estimate until
that comparison is done, e.g. via `rf_gun.frozen_source_attribution.run_frozen_source_attribution`'s
"RF only" vs. "RF + BL" cases.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def solve_causal_modal_envelope(
    t_grid_s: np.ndarray,
    I_beam_A: np.ndarray,
    f_hz: float,
    Q_L: float,
    r_over_q_ohm: float,
    detuning_rad_s: float = 0.0,
    V_b0: complex = 0.0j,
) -> np.ndarray:
    """Integrate the beam-induced voltage phasor V_b(t) [V] causally over `t_grid_s` (need not be
    uniformly spaced), given a real beam current history `I_beam_A(t)` [A] (e.g.
    `thermo_info["I_A_t"]`). Returns a complex array matching `t_grid_s`'s shape.

    Each step is solved *exactly* for the homogeneous (decay) part, using the forcing term's
    trapezoidal (endpoint) average as a piecewise-constant approximation over that step -- this is
    unconditionally stable regardless of how large a step is relative to `tau=2*Q_L/omega` (an
    ordinary explicit stepper would need `dt << tau`, which is impractical here since `tau` can be
    ~500-600 RF periods for this gun's own `Q_L`, per the module docstring), so the emission
    time grid's own (much finer) resolution can be reused directly without sub-stepping.
    """
    t = np.asarray(t_grid_s, dtype=float)
    I = np.asarray(I_beam_A, dtype=float)
    if t.shape != I.shape:
        raise ValueError(f"t_grid_s shape {t.shape} != I_beam_A shape {I.shape}")
    if t.size == 0:
        return np.zeros(0, dtype=complex)

    omega = 2.0 * np.pi * f_hz
    tau_s = 2.0 * Q_L / omega
    inv_tau = (1.0 / tau_s) - 1j * detuning_rad_s  # decay rate + rotation, complex

    Vb = np.empty(t.shape, dtype=complex)
    Vb[0] = V_b0
    for i in range(1, t.size):
        dt = float(t[i] - t[i - 1])
        forcing = -(omega / 2.0) * r_over_q_ohm * 0.5 * (I[i] + I[i - 1])
        if abs(inv_tau) > 0.0:
            V_particular = forcing / inv_tau
            Vb[i] = V_particular + (Vb[i - 1] - V_particular) * np.exp(-inv_tau * dt)
        else:
            Vb[i] = Vb[i - 1] + forcing * dt
    return Vb


def steady_state_beam_induced_voltage(I_beam_A: float, Q_L: float, r_over_q_ohm: float) -> float:
    """V_b,ss = -(R/Q)*Q_L*I0 -- the standard on-resonance long-pulse ("DC beam loading") limit,
    for cross-checking solve_causal_modal_envelope's own long-time behavior (see module
    docstring)."""
    return -r_over_q_ohm * Q_L * float(I_beam_A)


def _validate_veff(Veff_V: complex) -> complex:
    """`chi = V_b/Veff` must never divide by a non-finite or zero calibration voltage and continue
    with a silently invalid (inf/NaN) chi -- Veff must come from a validated
    `rf_gun.rf_params.PhaseCalibrationResult` (see `veff_from_phase_calibration`), which already
    guarantees this, but every direct caller of this module is checked again here since Veff is
    just a plain `complex`/`float` argument by the time it reaches this module."""
    v = complex(Veff_V)
    if not np.isfinite(v) or v == 0.0:
        raise ValueError(
            f"beam_loading_envelope: Veff_V={Veff_V!r} is not finite and nonzero -- refusing to "
            "compute chi=V_b/Veff from an invalid calibration voltage."
        )
    return v


def estimate_beam_induced_cathode_field(
    t_grid_s: np.ndarray,
    I_beam_A: np.ndarray,
    E_RF_Vpm: np.ndarray,
    f_hz: float,
    Q_L: float,
    r_over_q_ohm: float,
    Veff_V: complex,
    detuning_rad_s: float = 0.0,
) -> np.ndarray:
    """Convenience wrapper: E_BL(t) = -Re[chi(t)] * E_RF(t), chi(t) = V_b(t)/Veff_V, with V_b(t)
    from solve_causal_modal_envelope. `E_RF_Vpm` must already be the signed on-axis (or local)
    cathode RF field at each `t_grid_s` sample (e.g. `thermo_info["Ez_t"]`), matching the
    convention that E_BL shares the RF mode's own spatial structure (see module docstring) --
    this function only handles the temporal envelope, not a separate spatial solve.
    """
    E_RF = np.asarray(E_RF_Vpm, dtype=float)
    t = np.asarray(t_grid_s, dtype=float)
    if E_RF.shape != t.shape:
        raise ValueError(f"E_RF_Vpm shape {E_RF.shape} != t_grid_s shape {t.shape}")
    Veff_V = _validate_veff(Veff_V)
    V_b = solve_causal_modal_envelope(t, I_beam_A, f_hz, Q_L, r_over_q_ohm, detuning_rad_s=detuning_rad_s)
    chi = V_b / Veff_V
    return -np.real(chi) * E_RF


def estimate_beam_induced_cathode_field_from_current_density(
    J_Apm2: np.ndarray,
    area_m2: np.ndarray,
    t_grid_s: np.ndarray,
    E_RF_Vpm: np.ndarray,
    f_hz: float,
    Q_L: float,
    r_over_q_ohm: float,
    Veff_V: complex,
    detuning_rad_s: float = 0.0,
) -> np.ndarray:
    """Shared implementation behind `estimate_beam_induced_cathode_field_map` and
    `rf_gun.emission_iteration`'s own in-loop Picard feedback (see that module's
    `EmissionFieldIterationConfig.include_beam_loading`): given a cathode current-density array
    `J_Apm2` with shape `(*spatial_shape, n_t)` (2D `(n_x, n_y, n_t)` for a converged result, or
    flat `(n_points, n_t)` mid-iteration), and a matching-shaped-minus-time `area_m2`
    `(*spatial_shape,)`, forms the cathode-integrated beam current
    `I(t) = sum_spatial J(...,t)*area(...)`, solves the causal modal envelope for `V_b(t)`, and
    returns `E_BL(...,t) = -Re[V_b(t)/Veff_V] * E_RF_Vpm(...,t)` -- the single scalar chi(t)
    broadcast back across every spatial point, since the envelope ODE is a single-mode,
    spatially-integrated quantity (individual cathode cells don't each drive their own separate
    cavity mode) while `E_RF_Vpm` already carries the accelerating mode's real spatial structure.
    """
    J = np.asarray(J_Apm2, dtype=float)
    area = np.asarray(area_m2, dtype=float)
    if J.shape[:-1] != area.shape:
        raise ValueError(f"J_Apm2 spatial shape {J.shape[:-1]} != area_m2 shape {area.shape}")
    spatial_axes = tuple(range(J.ndim - 1))
    I_t = np.sum(J * area[..., None], axis=spatial_axes)  # (n_t,)

    Veff_V = _validate_veff(Veff_V)
    t_grid_s = np.asarray(t_grid_s, dtype=float)
    V_b = solve_causal_modal_envelope(t_grid_s, I_t, f_hz, Q_L, r_over_q_ohm, detuning_rad_s=detuning_rad_s)
    chi_t = V_b / Veff_V  # (n_t,)
    broadcast_shape = (1,) * (J.ndim - 1) + (chi_t.shape[0],)
    return -np.real(chi_t).reshape(broadcast_shape) * np.asarray(E_RF_Vpm, dtype=float)


def estimate_beam_induced_cathode_field_map(
    result,
    f_hz: float,
    Q_L: float,
    r_over_q_ohm: float,
    Veff_V: complex,
    detuning_rad_s: float = 0.0,
) -> np.ndarray:
    """Broadcasts the causal beam-loading envelope across a converged
    `rf_gun.emission_iteration.EmissionFieldIterationResult`'s own (x,y,t) grid, returning an
    `E_BL(x,y,t)` array shaped like `result.E_RF_history_Vpm[-1]`, using the result's own converged
    (final-iteration) current density -- see `estimate_beam_induced_cathode_field_from_current_density`
    for the shared implementation.

    If `result.config.include_beam_loading` was already True, `result.E_BL_history_Vpm[-1]` is the
    genuinely self-consistent field this iteration itself converged with (the Picard loop's own
    beam-induced-field feedback -- see `rf_gun.emission_iteration`) and calling this function again
    is redundant (though it will reproduce the same array to numerical precision, since both use
    the same final converged J(x,y,t)). This function remains useful specifically for the
    `include_beam_loading=False` case: forming an illustrative "SC+mirror+BL (estimated)" total
    field, `result.E_total_history_Vpm[-1] + E_BL`, from a run that did not itself include beam
    loading in its self-consistency loop (e.g. to compare the two).
    """
    J_final = np.asarray(result.J_history_Apm2[-1])  # (n_x, n_y, n_t)
    area_m2 = np.asarray(result.fixed_sample["grid"]["dA_mm2"]) * 1.0e-6  # (n_x, n_y)
    E_RF = np.asarray(result.E_RF_history_Vpm[-1])  # (n_x, n_y, n_t)
    return estimate_beam_induced_cathode_field_from_current_density(
        J_final, area_m2, result.t_grid_s, E_RF, f_hz, Q_L, r_over_q_ohm, Veff_V,
        detuning_rad_s=detuning_rad_s,
    )
