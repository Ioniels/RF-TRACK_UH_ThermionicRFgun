"""Fill-transient extrapolation for the non-converged XFdtd export.

The shipped volume was terminated on a fixed step count with convergence detection
disabled, so its frames sample a still-filling cavity.  These tests pin the single-pole
inversion, the guard rails that stop an unqualified extrapolation reaching production, and
the invariant that the correction is a pure real amplitude scale.
"""

import math

import numpy as np
import pytest

from rf_gun.fieldmaps.models import AxisymmetricRFField
from rf_gun.fieldmaps.steady_state import (
    FILL_MODEL,
    apply_fill_correction,
    fit_fill_transient,
    prepare_fill_treatment,
)

# The real export's measured values.
SHIPPED_DRIFT = 0.0024986
F_HZ = 2.865417220e9
DURATION_S = 1.0000006242079978e-07


def _field(nz=9, nr=5, value=1.0 + 0.5j, freq=F_HZ):
    z = np.linspace(0.0, 4.0e-3, nz)
    r = np.linspace(0.0, 1.0e-3, nr)
    arr = np.full((nz, nr), value, dtype=complex)
    return AxisymmetricRFField(
        frequency_hz=freq, r_m=r, z_m=z,
        Er_Vpm=arr * 1.0e6, Ez_Vpm=arr * 2.0e6,
        Btheta_T=arr * 3.0e-3, Bz_T=arr * 4.0e-4,
        metadata={"source_power_w": 1.8e6},
    )


def test_recovers_a_known_fill_time_constant():
    """Round trip: synthesise the drift a chosen tau would produce, then recover tau."""
    for tau_ns in (40.0, 157.5, 600.0):
        tau = tau_ns * 1e-9
        u = DURATION_S / tau
        # g = (1/tau) e^-u / (1 - e^-u), expressed per RF cycle
        g = (1.0 / tau) * math.exp(-u) / (-math.expm1(-u))
        drift = g / F_HZ
        fit = fit_fill_transient(drift, F_HZ, DURATION_S)
        assert fit.tau_s == pytest.approx(tau, rel=1e-6), f"tau not recovered for {tau_ns} ns"
        assert fit.fill_fraction == pytest.approx(-math.expm1(-u), rel=1e-6)


def test_legacy_end_time_estimate_is_reproducible_under_the_step_assumptions():
    """Reproducing the archived calculation is not a calibration of the source."""
    fit = fit_fill_transient(SHIPPED_DRIFT, F_HZ, DURATION_S)
    assert fit.model == FILL_MODEL
    assert not fit.converged_export
    assert fit.passed
    assert fit.tau_s * 1e9 == pytest.approx(157.5, abs=0.5)
    assert fit.q_loaded == pytest.approx(1418.0, abs=5.0)
    assert fit.fill_fraction == pytest.approx(0.4700, abs=1e-3)
    assert fit.correction_factor == pytest.approx(2.128, abs=2e-3)
    # loaded Q and tau must agree with tau = 2 Q / omega
    omega = 2.0 * math.pi * F_HZ
    assert fit.tau_s == pytest.approx(2.0 * fit.q_loaded / omega, rel=1e-9)


def test_converged_export_is_left_alone():
    fit = fit_fill_transient(1.0e-9, F_HZ, DURATION_S)
    assert fit.converged_export
    assert fit.correction_factor == 1.0
    field = _field()
    assert apply_fill_correction(field, fit) is field


def test_correction_is_a_pure_real_amplitude_scale():
    """Phases, ratios and the grid must be untouched; only amplitude changes."""
    fit = fit_fill_transient(SHIPPED_DRIFT, F_HZ, DURATION_S)
    field = _field()
    out = apply_fill_correction(field, fit)
    k = fit.correction_factor
    for name in ("Er_Vpm", "Ez_Vpm", "Btheta_T", "Bz_T"):
        before = getattr(field, name)
        after = getattr(out, name)
        assert np.allclose(after, before * k), f"{name} not scaled by the common factor"
        assert np.allclose(np.angle(after), np.angle(before)), f"{name} phase changed"
    assert np.array_equal(out.r_m, field.r_m)
    assert np.array_equal(out.z_m, field.z_m)
    assert out.frequency_hz == field.frequency_hz
    # the E/B ratio is the physical invariant that must survive
    assert np.allclose(out.Ez_Vpm / out.Btheta_T, field.Ez_Vpm / field.Btheta_T)


def test_correction_is_recorded_in_metadata():
    fit = fit_fill_transient(SHIPPED_DRIFT, F_HZ, DURATION_S)
    out = apply_fill_correction(_field(), fit)
    assert out.metadata["fill_correction_applied"] is True
    assert out.metadata["amplitude_reference"] == "model_estimated_steady_state"
    assert out.metadata["steady_state_verified"] is False
    record = out.metadata["fill_correction"]
    assert record["model"] == FILL_MODEL
    assert record["drift_per_cycle"] == pytest.approx(SHIPPED_DRIFT)
    assert record["correction_factor"] == pytest.approx(fit.correction_factor)


def test_implausible_fits_are_refused_not_applied():
    """A drift implying an absurd Q or a huge boost must not silently become production."""
    # g*t >= 1 is unreachable for a single-pole fill at all.
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        fit_fill_transient(1.0, F_HZ, DURATION_S)

    # A drift still inside the single-pole range, but implying a cavity so far from steady
    # state that the one-parameter extrapolation would have to invent ~20x the amplitude.
    big = fit_fill_transient(0.0034, F_HZ, DURATION_S)
    assert not big.passed
    assert not big.checks["correction_within_bound"]
    assert big.correction_factor > 10.0
    with pytest.raises(ValueError, match="failed fit"):
        apply_fill_correction(_field(), big)

    # Just inside the bound must still be usable, so the guard is a real threshold.
    ok = fit_fill_transient(0.0033, F_HZ, DURATION_S)
    assert ok.passed and ok.correction_factor < 10.0


def test_correction_composes_with_power_scaling():
    """Correct-then-scale must equal one combined real factor, with B tracking E."""
    fit = fit_fill_transient(SHIPPED_DRIFT, F_HZ, DURATION_S)
    field = _field()
    out = apply_fill_correction(field, fit).scaled_to_power(1.0e6, 1.8e6)
    combined = fit.correction_factor * math.sqrt(1.0e6 / 1.8e6)
    assert np.allclose(out.Ez_Vpm, field.Ez_Vpm * combined)
    assert np.allclose(out.Btheta_T, field.Btheta_T * combined)
    assert combined == pytest.approx(1.586, abs=2e-3)


def test_reference_epoch_recovers_true_field_instead_of_using_run_end():
    from dataclasses import replace

    tau = 157.5e-9
    reference = 99.30136193505001e-9
    start = 5e-9
    fraction = -math.expm1(-(reference - start) / tau)
    drift = math.exp(-(reference - start) / tau) / (tau * fraction * F_HZ)
    fit = fit_fill_transient(
        drift, F_HZ, DURATION_S, reference_time_s=reference, drive_start_time_s=start
    )
    sampled = replace(_field(value=fraction), metadata={"reference_time_s": reference})
    corrected = apply_fill_correction(sampled, fit)
    np.testing.assert_allclose(corrected.Ez_Vpm, 2e6, rtol=1e-12)
    assert fit.tau_s == pytest.approx(tau)
    legacy = fit_fill_transient(drift, F_HZ, DURATION_S)
    assert abs(legacy.correction_factor * fraction - 1) > 0.1


def test_negative_drift_is_not_reported_as_converged():
    with pytest.raises(ValueError, match="decay"):
        fit_fill_transient(-0.002, F_HZ, DURATION_S)


def test_stationary_diagnostic_is_serializable_but_not_verified_steady_state():
    from rf_gun.fieldmaps.artifact import canonical_json

    fit = fit_fill_transient(0, F_HZ, DURATION_S)
    canonical_json(fit.as_dict())
    assert fit.as_dict()["steady_state_verified"] is False
    assert fit.as_dict()["normalized_time_t_over_tau"] is None


def test_double_application_and_mismatched_field_are_rejected():
    from dataclasses import replace

    fit = fit_fill_transient(SHIPPED_DRIFT, F_HZ, DURATION_S)
    with pytest.raises(ValueError, match="already"):
        apply_fill_correction(apply_fill_correction(_field(), fit), fit)
    with pytest.raises(ValueError, match="frequency"):
        apply_fill_correction(_field(freq=3e9), fit)
    with pytest.raises(ValueError, match="reference_time"):
        apply_fill_correction(replace(_field(), metadata={"reference_time_s": 99e-9}), fit)


def _treatment_input(amplitude_drift=0.0025, phase_drift=0.0, residual=0.0):
    from dataclasses import replace

    field = replace(_field(), metadata={"reference_time_s": 99.3e-9})
    norm = math.sqrt(amplitude_drift**2 + phase_drift**2 + residual**2)
    temporal = {
        "production_metrics": {"max_family_combined_fractional_drift_per_cycle": norm},
        "family_aggregate": {
            name: {
                "signed_amplitude_drift_per_cycle": amplitude_drift,
                "phase_drift_rad_per_cycle": phase_drift,
                "nonseparable_drift_per_cycle": residual,
            } for name in ("E", "H")
        },
    }
    return field, temporal


def test_auto_does_not_extrapolate_a_transient():
    field, temporal = _treatment_input()
    out, record = prepare_fill_treatment(field, temporal, simulated_duration_s=DURATION_S)
    np.testing.assert_array_equal(out.Ez_Vpm, field.Ez_Vpm)
    assert record["applied"] is False
    assert record["applied_correction_factor"] == 1
    assert record["production_eligible"] is False


def test_off_control_never_requires_an_admissible_fill_fit():
    field, temporal = _treatment_input(amplitude_drift=-0.1)
    out, record = prepare_fill_treatment(field, temporal, simulated_duration_s=DURATION_S, policy="off")
    np.testing.assert_array_equal(out.Ez_Vpm, field.Ez_Vpm)
    assert record["production_eligible"] is True
    assert record["applied"] is False
    assert out.metadata["fill_treatment"] == record


@pytest.mark.parametrize("phase,residual", [(0.0025, 0), (0, 0.0025)])
def test_phase_rotation_and_nonseparable_growth_cannot_be_treated_as_filling(phase, residual):
    field, temporal = _treatment_input(phase_drift=phase, residual=residual)
    with pytest.raises(ValueError, match="common-envelope"):
        prepare_fill_treatment(field, temporal, simulated_duration_s=DURATION_S,
                               policy="single-pole", drive_start_time_s=0)


def test_explicit_step_model_records_the_applied_factor_and_assumptions():
    field, temporal = _treatment_input()
    with pytest.raises(ValueError, match="explicit assumed"):
        prepare_fill_treatment(field, temporal, simulated_duration_s=DURATION_S, policy="single-pole")
    out, record = prepare_fill_treatment(field, temporal, simulated_duration_s=DURATION_S,
                                        policy="single-pole", drive_start_time_s=0)
    np.testing.assert_allclose(out.Ez_Vpm, field.Ez_Vpm * record["applied_correction_factor"])
    assert record["applied"] is True
    assert record["steady_state_verified"] is False
    assert record["reference_time_s"] == 99.3e-9


def test_temporal_projection_distinguishes_phase_rotation_and_decay():
    from rf_gun.fieldmaps.harmonic import HarmonicFitConfig, fit_fixed_frequency_phasor
    from rf_gun.fieldmaps.pipeline import _temporal_summary

    time = np.linspace(0, 4 / F_HZ, 65)
    ref = time.mean()
    phase = np.exp(2j * np.pi * F_HZ * (time - ref))
    phasor = np.array([1 + 2j, 3 - 4j])
    gamma = complex(-0.002, 0.003)  # decay plus phase rotation, per cycle
    values = np.real(phase[:, None] * phasor * (1 + (time - ref)[:, None] * F_HZ * gamma))
    fit = fit_fixed_frequency_phasor(time, values, HarmonicFitConfig(F_HZ, reference_time_s=ref, include_linear_envelope=True))
    summary = _temporal_summary(fit, 0, 1, values, np.array([1.0, 2.0]))
    measured = complex(summary["cylindrical_weighted_phasor_slope_overlap_real"],
                       summary["cylindrical_weighted_phasor_slope_overlap_imag"]) / summary["cylindrical_weighted_phasor_sumsq"]
    assert measured == pytest.approx(gamma, abs=1e-14)
    unsigned = math.sqrt(summary["cylindrical_weighted_slope_change_per_cycle_sumsq"] / summary["cylindrical_weighted_phasor_sumsq"])
    assert unsigned == pytest.approx(abs(gamma))
