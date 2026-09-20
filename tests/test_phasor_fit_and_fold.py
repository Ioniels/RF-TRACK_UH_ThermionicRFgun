from __future__ import annotations

import numpy as np
import pytest

from rf_gun.phasor import fit_harmonic_phasor, fold_axisymmetric_planar_fields


def test_all_time_harmonic_fit_recovers_complex_components_and_dc_offset():
    frequency_hz = 2.865417220e9
    time_s = np.array([0.0, 19.0, 73.0, 121.0, 212.0, 305.0, 441.0]) * 1.0e-12
    phasor = np.array([[2.0 + 3.0j, -4.5 + 0.25j], [0.2 - 0.7j, 8.0 - 1.0j]])
    offset = np.array([[0.1, -0.4], [1.2, 0.0]])
    theta = 2.0 * np.pi * frequency_hz * time_s
    samples = offset[..., None] + np.real(phasor[..., None] * np.exp(1j * theta))

    result = fit_harmonic_phasor(samples, time_s, frequency_hz)

    np.testing.assert_allclose(result.phasor, phasor, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(result.offset, offset, rtol=1.0e-12, atol=1.0e-12)
    assert result.normalized_rms_residual < 1.0e-12
    assert result.n_samples == time_s.size


def test_all_time_fit_preserves_relative_component_amplitudes():
    frequency_hz = 1.0e9
    time_s = np.linspace(0.0, 2.0 / frequency_hz, 17)
    base = np.real(np.exp(1j * 2.0 * np.pi * frequency_hz * time_s))
    ex = 0.125 * base[None, :]
    ez = 7.5 * base[None, :]

    ex_fit = fit_harmonic_phasor(ex, time_s, frequency_hz).phasor
    ez_fit = fit_harmonic_phasor(ez, time_s, frequency_hz).phasor

    assert abs(ex_fit[0] / ez_fit[0]) == pytest.approx(0.125 / 7.5)


def test_axisymmetric_fold_projects_odd_and_even_parity_without_duplicate_half_planes():
    # The small coordinate mismatch mimics harmless export rounding between reflected halves.
    x_m = np.array([-2.00001, -1.0, 0.0, 1.0, 2.0]) * 1.0e-3
    z_m = np.zeros_like(x_m)
    radial_cartesian = np.array([-4.0, -2.0, 99.0, 2.0, 4.0], dtype=complex)
    longitudinal = np.array([3.0, 5.0, 7.0, 5.0, 3.0], dtype=complex)

    r, z, radial, axial, diagnostics = fold_axisymmetric_planar_fields(
        x_m, z_m, radial_cartesian, longitudinal,
    )

    np.testing.assert_allclose(r, [0.0, 1.0e-3, 2.0e-3])
    np.testing.assert_allclose(z, 0.0)
    np.testing.assert_allclose(radial, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(axial, [7.0, 5.0, 3.0])
    assert diagnostics["n_output"] == 3
    assert diagnostics["radial_odd_residual"] == pytest.approx(0.0)
    assert diagnostics["longitudinal_even_residual"] == pytest.approx(0.0)


def test_axisymmetric_fold_rejects_unpaired_halves():
    with pytest.raises(ValueError, match="do not pair within tolerance"):
        fold_axisymmetric_planar_fields(
            np.array([-1.0, 0.0, 2.0]),
            np.zeros(3),
            np.array([-1.0, 0.0, 2.0]),
            np.ones(3),
            pair_tolerance_m=0.01,
        )
