"""Tests for rf_gun.beam_loading_envelope -- pure math, no RF-Track binding needed at all.

Validates the causal modal-envelope ODE against known closed-form limits (steady state, zero
current, exponential decay after a current pulse ends) rather than just checking it runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from rf_gun.beam_loading_envelope import (
    solve_causal_modal_envelope,
    steady_state_beam_induced_voltage,
    estimate_beam_induced_cathode_field,
    estimate_beam_induced_cathode_field_from_current_density,
)

F_HZ = 2.856e9
Q_L = 1866.67
R_OVER_Q_OHM = 120.17  # a representative lumped (R/Q), matching this repo's own r_over_q_per_m scale


def test_zero_current_gives_zero_voltage():
    t = np.linspace(0.0, 1e-9, 200)
    I = np.zeros_like(t)
    V_b = solve_causal_modal_envelope(t, I, F_HZ, Q_L, R_OVER_Q_OHM)
    np.testing.assert_allclose(V_b, 0.0)


def test_reaches_the_standard_steady_state_dc_beam_loading_limit():
    """Held at constant current for many decay times (tau ~ 2*Q_L/omega), V_b must approach the
    textbook steady-state result V_b,ss = -(R/Q)*Q_L*I0 (P. B. Wilson's fundamental theorem of
    beam loading, DC/long-pulse limit) -- a real physics cross-check, not just "didn't crash"."""
    omega = 2.0 * np.pi * F_HZ
    tau_s = 2.0 * Q_L / omega
    I0 = 0.53  # A, order-of-magnitude representative peak current for this project

    t = np.linspace(0.0, 8.0 * tau_s, 20_000)
    I = np.full_like(t, I0)
    V_b = solve_causal_modal_envelope(t, I, F_HZ, Q_L, R_OVER_Q_OHM)

    V_ss_expected = steady_state_beam_induced_voltage(I0, Q_L, R_OVER_Q_OHM)
    assert V_b[-1].real == pytest.approx(V_ss_expected, rel=1e-3)
    assert V_b[-1].imag == pytest.approx(0.0, abs=abs(V_ss_expected) * 1e-6)


def test_decays_exponentially_after_current_stops():
    """After the beam current returns to zero, V_b must decay as exp(-t/tau) with the cavity's
    own tau=2*Q_L/omega -- a second, independent closed-form check on the same ODE."""
    omega = 2.0 * np.pi * F_HZ
    tau_s = 2.0 * Q_L / omega
    I0 = 0.53
    t_on = np.linspace(0.0, 3.0 * tau_s, 5000)
    I_on = np.full_like(t_on, I0)
    V_b_on = solve_causal_modal_envelope(t_on, I_on, F_HZ, Q_L, R_OVER_Q_OHM)

    t_off = t_on[-1] + np.linspace(0.0, 2.0 * tau_s, 5000)
    I_off = np.zeros_like(t_off)
    V_b_off = solve_causal_modal_envelope(t_off, I_off, F_HZ, Q_L, R_OVER_Q_OHM, V_b0=V_b_on[-1])

    expected_decay = V_b_on[-1] * np.exp(-(t_off - t_off[0]) / tau_s)
    np.testing.assert_allclose(V_b_off, expected_decay, rtol=1e-6)


def test_stable_for_a_step_much_larger_than_tau():
    """The exact-per-step integrator must not blow up or misbehave even when dt >> tau (this
    gun's own tau/T_RF ~ 594, so a naive explicit stepper sized to the RF period would need
    ~600x sub-stepping here; this integrator must not need that)."""
    omega = 2.0 * np.pi * F_HZ
    tau_s = 2.0 * Q_L / omega
    t = np.array([0.0, 50.0 * tau_s])  # a single giant step
    I = np.array([0.53, 0.53])
    V_b = solve_causal_modal_envelope(t, I, F_HZ, Q_L, R_OVER_Q_OHM)
    V_ss_expected = steady_state_beam_induced_voltage(0.53, Q_L, R_OVER_Q_OHM)
    assert np.isfinite(V_b[-1])
    assert V_b[-1].real == pytest.approx(V_ss_expected, rel=1e-6)


def test_estimate_beam_induced_cathode_field_scales_with_E_RF():
    """A crude but real end-to-end check: doubling E_RF (with everything else fixed) must exactly
    double the returned E_BL, since E_BL = -Re[chi(t)]*E_RF(t) is linear in E_RF by construction."""
    t = np.linspace(0.0, 1e-10, 50)
    I = np.full_like(t, 0.53)
    E_RF = np.full_like(t, -2.0e7)
    Veff_V = 6.9e5 + 0j

    E_BL_1 = estimate_beam_induced_cathode_field(t, I, E_RF, F_HZ, Q_L, R_OVER_Q_OHM, Veff_V)
    E_BL_2 = estimate_beam_induced_cathode_field(t, I, 2.0 * E_RF, F_HZ, Q_L, R_OVER_Q_OHM, Veff_V)
    np.testing.assert_allclose(E_BL_2, 2.0 * E_BL_1)
    assert np.all(np.isfinite(E_BL_1))


def test_estimate_beam_induced_cathode_field_from_current_density_matches_scalar_form():
    """The (spatial..., n_t) current-density form (used both by
    estimate_beam_induced_cathode_field_map and rf_gun.emission_iteration's own in-loop Picard
    feedback) must reduce to the same chi(t) and E_BL(t) as the scalar-current
    estimate_beam_induced_cathode_field, for a single-cell "spatial" grid whose one cell's
    current density times area equals the scalar current used in the other form."""
    t = np.linspace(0.0, 1e-10, 50)
    I0 = 0.53
    E_RF_scalar = np.full_like(t, -2.0e7)
    Veff_V = 6.9e5 + 0j

    E_BL_scalar = estimate_beam_induced_cathode_field(t, np.full_like(t, I0), E_RF_scalar, F_HZ, Q_L, R_OVER_Q_OHM, Veff_V)

    area_m2 = np.array([[2.0e-7]])  # a single "cell"
    J_Apm2 = (I0 / area_m2[0, 0]) * np.ones((1, 1, t.size))  # J*area = I0 at every t
    E_RF_map = np.broadcast_to(E_RF_scalar, (1, 1, t.size))
    E_BL_map = estimate_beam_induced_cathode_field_from_current_density(
        J_Apm2, area_m2, t, E_RF_map, F_HZ, Q_L, R_OVER_Q_OHM, Veff_V,
    )
    np.testing.assert_allclose(E_BL_map[0, 0, :], E_BL_scalar, rtol=1e-10)


def test_estimate_beam_induced_cathode_field_from_current_density_sums_over_all_spatial_cells():
    """Two cells with the same J must combine (I(t)=sum of both) rather than each independently
    driving its own separate cavity mode -- the envelope ODE is a single-mode, spatially-
    integrated quantity by construction (see the function's own docstring)."""
    t = np.linspace(0.0, 1e-10, 50)
    J0 = 1.0e6
    area_m2 = np.array([1.0e-7, 1.0e-7])
    J_one_cell = np.zeros((2, t.size))
    J_one_cell[0, :] = J0
    J_two_cells = np.zeros((2, t.size))
    J_two_cells[:, :] = J0
    E_RF = np.full((2, t.size), -2.0e7)
    Veff_V = 6.9e5 + 0j

    E_BL_one = estimate_beam_induced_cathode_field_from_current_density(
        J_one_cell, area_m2, t, E_RF, F_HZ, Q_L, R_OVER_Q_OHM, Veff_V,
    )
    E_BL_two = estimate_beam_induced_cathode_field_from_current_density(
        J_two_cells, area_m2, t, E_RF, F_HZ, Q_L, R_OVER_Q_OHM, Veff_V,
    )
    assert np.max(np.abs(E_BL_two)) > np.max(np.abs(E_BL_one)), (
        "doubling the cathode-integrated current (two emitting cells instead of one) must change "
        "the beam-induced field -- it should not be computed per cell independently"
    )
