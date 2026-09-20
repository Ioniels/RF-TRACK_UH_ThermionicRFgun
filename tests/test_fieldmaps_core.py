from __future__ import annotations

import numpy as np
import pytest

from rf_gun.fieldmaps.coordinates import (
    MU0_HPM,
    XFD_TD_CATHODE_TO_BEAM,
    cartesian_to_cylindrical_vectors,
    magnetic_flux_density_from_h,
    one_sided_surface_limit,
    pair_meridional_halves,
)
from rf_gun.fieldmaps.harmonic import HarmonicFitConfig, fit_eh_phasors, fit_fixed_frequency_phasor
from rf_gun.fieldmaps.models import AxisymmetricRFField, FieldMapValidationError
from rf_gun.fieldmaps.sections import RectilinearPhasor, colocate_rectilinear, linear_stencil
from rf_gun.fieldmaps.symmetry import (
    azimuthal_fourier_modes,
    cylindrical_samples_from_cartesian,
    project_cylindrical_m0,
)
from rf_gun.fieldmaps.tail import (
    circular_waveguide_tm01_decay_length_m,
    extend_axisymmetric_with_modal_tail,
    fit_evanescent_tail,
)


def _field() -> AxisymmetricRFField:
    r = np.linspace(0.0, 2.0e-3, 3)
    z = np.linspace(0.0, 3.0e-3, 4)
    shape = (z.size, r.size)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1j
    return AxisymmetricRFField(
        frequency_hz=2.865e9,
        r_m=r,
        z_m=z,
        Er_Vpm=base,
        Ez_Vpm=2.0 * base,
        Btheta_T=3.0e-3 * base,
        Bz_T=4.0e-3 * base,
        metadata={"representation": "m0"},
    )


def test_axisymmetric_field_validates_and_scales_all_components_together():
    field = _field()
    assert field.shape == (4, 3)
    assert field.hr_m == pytest.approx(1.0e-3)
    scaled = field.scaled_to_power(7.2e6, 1.8e6)
    np.testing.assert_allclose(scaled.Er_Vpm, 2.0 * field.Er_Vpm)
    np.testing.assert_allclose(scaled.Ez_Vpm, 2.0 * field.Ez_Vpm)
    np.testing.assert_allclose(scaled.Btheta_T, 2.0 * field.Btheta_T)
    np.testing.assert_allclose(scaled.Bz_T, 2.0 * field.Bz_T)
    zero_b = field.without_rf_magnetic_field()
    assert np.all(zero_b.Btheta_T == 0.0)
    assert np.all(zero_b.Bz_T == 0.0)
    np.testing.assert_array_equal(zero_b.Ez_Vpm, field.Ez_Vpm)


def test_axisymmetric_field_rejects_nonuniform_grid():
    field = _field()
    with pytest.raises(FieldMapValidationError, match="uniform"):
        AxisymmetricRFField(
            field.frequency_hz,
            np.array([0.0, 1.0e-3, 2.2e-3]),
            field.z_m,
            field.Er_Vpm,
            field.Ez_Vpm,
            field.Btheta_T,
            field.Bz_T,
        )


def test_repeated_power_scaling_sets_target_without_compounding_previous_scale():
    field = _field()
    repeated = field.scaled_to_power(7.2e6, 1.8e6).scaled_to_power(1.0e6, 1.8e6)
    direct = field.scaled_to_power(1.0e6, 1.8e6)
    for name in ("Er_Vpm", "Ez_Vpm", "Btheta_T", "Bz_T"):
        np.testing.assert_allclose(getattr(repeated, name), getattr(direct, name))
    assert repeated.metadata["common_field_scale"] == direct.metadata["common_field_scale"]
    same = repeated.scaled_to_power(1e6, 1.8e6)
    np.testing.assert_array_equal(same.Ez_Vpm, repeated.Ez_Vpm)
    with pytest.raises(ValueError, match="recorded normalization"):
        repeated.scaled_to_power(1e6, 2e6)
    with pytest.raises(ValueError, match="already scaled to zero"):
        field.scaled_to_power(0, 1.8e6).scaled_to_power(1e6, 1.8e6)


def test_axisymmetric_field_rejects_wrong_shape():
    field = _field()
    with pytest.raises(FieldMapValidationError, match="shape"):
        AxisymmetricRFField(
            field.frequency_hz,
            field.r_m,
            field.z_m,
            field.Er_Vpm[:, :-1],
            field.Ez_Vpm,
            field.Btheta_T,
            field.Bz_T,
        )


def test_frame_transform_is_proper_and_round_trips_points_and_vectors():
    transform = XFD_TD_CATHODE_TO_BEAM
    assert np.linalg.det(transform.R_target_from_native) == pytest.approx(1.0)
    native_point = transform.origin_native_m + np.array([1.0e-3, -2.0e-3, 3.0e-3])
    beam_point = transform.transform_points(native_point)
    np.testing.assert_allclose(beam_point, [1.0e-3, 3.0e-3, 2.0e-3])
    np.testing.assert_allclose(transform.inverse_points(beam_point), native_point)
    native_vector = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(transform.transform_vectors(native_vector), [1.0, 3.0, -2.0])
    np.testing.assert_allclose(
        transform.inverse_vectors(transform.transform_vectors(native_vector)), native_vector
    )


def test_h_to_b_cylindrical_conversion_and_parity_projection():
    np.testing.assert_allclose(
        magnetic_flux_density_from_h(np.array([1.0, 2.0])), MU0_HPM * np.array([1.0, 2.0])
    )
    theta = np.array([0.0, np.pi / 2.0])
    vectors = np.array([[2.0, 3.0, 4.0], [-3.0, 2.0, 4.0]])
    cylindrical = cartesian_to_cylindrical_vectors(vectors, theta)
    np.testing.assert_allclose(cylindrical, [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], atol=1e-14)
    np.testing.assert_allclose(pair_meridional_halves([2.0], [-2.0], parity="odd"), [2.0])
    np.testing.assert_allclose(pair_meridional_halves([2.0], [2.0], parity="even"), [2.0])


def test_one_sided_surface_limit_does_not_average_through_pec():
    coordinate = np.array([-0.1, 0.05, 0.15, 0.25])
    # Metal-side value is zero; the vacuum field has the line 20 + 4*z.
    values = np.array([0.0, 20.2, 20.6, 21.0])
    surface = one_sided_surface_limit(
        coordinate, values, surface_m=0.0, vacuum_side="positive", n_points=3, polynomial_order=1
    )
    assert float(surface) == pytest.approx(20.0)


def test_all_frame_phasor_fit_recovers_complex_amplitude_offset_and_drift():
    frequency = 2.865e9
    time = 98.6e-9 + np.arange(17) / (4.0 * frequency)
    reference = float(np.mean(time))
    tau = time - reference
    phasor = np.array([2.0 + 3.0j, -4.0 + 0.5j])
    slope = np.array([1.2e7 - 2.0e7j, -0.7e7 + 0.4e7j])
    offset = np.array([0.4, -0.2])
    values = offset + np.real(
        (phasor[None, :] + slope[None, :] * tau[:, None])
        * np.exp(1j * 2.0 * np.pi * frequency * tau)[:, None]
    )
    result = fit_fixed_frequency_phasor(
        time,
        values,
        HarmonicFitConfig(
            frequency_hz=frequency,
            reference_time_s=reference,
            include_offset=True,
            include_linear_envelope=True,
        ),
    )
    np.testing.assert_allclose(result.phasor, phasor, atol=1e-11)
    np.testing.assert_allclose(result.phasor_slope_per_s, slope, rtol=1e-10, atol=1e-3)
    np.testing.assert_allclose(result.offset, offset, atol=1e-12)
    np.testing.assert_allclose(result.evaluate(time), values, atol=1e-12)
    assert result.global_nrmse < 1e-12


def test_active_fit_metric_uses_only_the_declared_physical_mask():
    frequency = 1.0e9
    time = np.arange(9, dtype=float) / (4.0 * frequency)
    phase = np.exp(1j * 2.0 * np.pi * frequency * (time - np.mean(time)))
    # The excluded point is orders of magnitude larger.  It must not set the
    # active threshold for the two in-mask points and hide their residual.
    phasor = np.array([1.0, 0.8, 1.0e9])
    values = np.real(phase[:, None] * phasor[None, :])
    values[:, 1] += np.linspace(-0.2, 0.2, time.size)
    result = fit_fixed_frequency_phasor(
        time,
        values,
        HarmonicFitConfig(frequency, include_offset=True),
        metric_mask=np.array([True, True, False]),
    )
    assert result.active_nrmse_p95 > 0.01


def test_e_and_h_fits_use_separate_times_but_one_reference():
    frequency = 1.0e9
    e_time = np.arange(9) / (4.0 * frequency)
    h_time = e_time + 0.1e-12
    reference = 0.5 * (e_time[0] + h_time[-1])
    e_phasor = 3.0 - 2.0j
    h_phasor = -5.0 + 4.0j
    e = np.real(e_phasor * np.exp(1j * 2.0 * np.pi * frequency * (e_time - reference)))
    h = np.real(h_phasor * np.exp(1j * 2.0 * np.pi * frequency * (h_time - reference)))
    result = fit_eh_phasors(
        {"x": e}, e_time, {"z": h}, h_time, HarmonicFitConfig(frequency, include_offset=False)
    )
    assert result.reference_time_s == pytest.approx(reference)
    assert result.electric["x"].phasor == pytest.approx(e_phasor)
    assert result.magnetic_h["z"].phasor == pytest.approx(h_phasor)


def test_rectilinear_colocation_is_exact_for_linear_complex_field():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 2.0])
    z = np.array([0.0, 3.0])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    values = X + 2.0 * Y - Z + 1j * (3.0 * X + Z)
    field = RectilinearPhasor(x, y, z, values)
    point = np.array([[0.25, 0.5, 1.5]])
    expected = 0.25 + 1.0 - 1.5 + 1j * (0.75 + 1.5)
    assert colocate_rectilinear(field, point)[0] == pytest.approx(expected)
    stencil = linear_stencil(z, 1.5)
    np.testing.assert_array_equal(stencil.indices, [0, 1])
    np.testing.assert_allclose(stencil.weights, [0.5, 0.5])


def test_cylindrical_m0_projection_rotates_vectors_before_averaging():
    theta = 2.0 * np.pi * np.arange(32) / 32.0
    radial = 2.0 + 0.2 * np.cos(2.0 * theta)
    longitudinal = 5.0 + 0.1 * np.sin(theta)
    ex = radial * np.cos(theta)
    ey = radial * np.sin(theta)
    electric = np.stack((ex, ey, longitudinal), axis=-1)[:, None]
    btheta = 4.0 + 0.4 * np.cos(theta)
    bx = -btheta * np.sin(theta)
    by = btheta * np.cos(theta)
    magnetic = np.stack((bx, by, np.zeros_like(theta)), axis=-1)[:, None]
    samples = cylindrical_samples_from_cartesian(electric, magnetic, theta)
    projected = project_cylindrical_m0(samples, theta, m_max=4)
    assert projected.Er[0] == pytest.approx(2.0)
    assert projected.Ez[0] == pytest.approx(5.0)
    assert projected.Btheta[0] == pytest.approx(4.0)
    assert projected.component_nonaxisymmetric_fraction["Er"] > 0.0
    modes = azimuthal_fourier_modes(samples.Er, theta, m_max=3)
    assert modes.coefficient(0)[0] == pytest.approx(2.0)
    assert abs(modes.coefficient(2)[0]) == pytest.approx(0.1)


def test_evanescent_tail_fit_matches_tm01_scale_and_marks_extension():
    frequency = 2.86541722e9
    radius = 2.5275e-3
    decay_length = circular_waveguide_tm01_decay_length_m(radius, frequency)
    z = np.linspace(0.028, 0.033, 51)
    phasor = (2.0e5 - 1.0e5j) * np.exp(
        (-1.0 / decay_length + 1j * 2.0) * (z - z[-1])
    )
    model = fit_evanescent_tail(z, phasor, fit_start_z_m=0.030)
    assert model.decay_length_m == pytest.approx(decay_length, rel=1e-10)
    assert model.phase_constant_per_m == pytest.approx(2.0, rel=1e-9)
    assert model.log_amplitude_r_squared > 0.999999

    r = np.linspace(0.0, 2.0e-3, 3)
    base = np.outer(phasor, np.array([1.0, 0.8, 0.2]))
    field = AxisymmetricRFField(frequency, r, z, base, base, 1e-9 * base, 1e-10 * base)
    extended = extend_axisymmetric_with_modal_tail(field, 0.040589, model)
    assert extended.z_m[-1] >= 0.040589
    assert extended.metadata["spatial_support"]["extension"].startswith("extrapolated")
    expected_factor = model.factor(extended.z_m[-1] - field.z_m[-1])
    np.testing.assert_allclose(extended.Ez_Vpm[-1], field.Ez_Vpm[-1] * expected_factor)


def test_on_plane_sample_is_admitted_and_anchors_the_surface_limit():
    """A Yee node lying ON the conductor plane must anchor the fit, not be excluded.

    Tangential E and normal H have a node exactly on the plane carrying the boundary zero.
    Excluding it forces an extrapolation ~one cell past the data through curved field, which
    produced a *larger* tangential field on the emission plane than half a cell downstream.
    """
    import numpy as np

    from rf_gun.fieldmaps.coordinates import one_sided_surface_limit

    # Curved tangential profile that genuinely vanishes at the surface.
    cell = 100.0e-6
    coordinate = np.array([-1.216e-6, 98.784e-6, 198.784e-6, 298.784e-6])
    values = np.array([0.0, 1.0e4, 3.6e4, 7.8e4])  # ~quadratic, zero on the plane

    strict = one_sided_surface_limit(
        coordinate, values, surface_m=0.0, vacuum_side="positive",
        n_points=3, polynomial_order=1, surface_tolerance_m=0.0,
    )
    anchored = one_sided_surface_limit(
        coordinate, values, surface_m=0.0, vacuum_side="positive",
        n_points=3, polynomial_order=1, surface_tolerance_m=0.1 * cell,
    )

    # Strictly one-sided extrapolation overshoots badly (it even changes sign here).
    assert abs(anchored) < abs(strict), "admitting the on-plane node must reduce the surface value"
    # The physical requirement: a tangential field decaying to zero at the conductor must not
    # come out LARGER on the surface than at the first vacuum sample. That unphysical bump is
    # exactly what the strict fit produced on the real cathode plane.
    assert abs(anchored) < abs(values[1]), (
        f"surface value {anchored!r} exceeds the first vacuum sample {values[1]!r}"
    )
    assert abs(strict) > abs(values[1]), "test precondition: strict fit should overshoot"


def test_surface_tolerance_leaves_straddling_components_one_sided():
    """Normal E straddles the boundary: its nearest node is half a cell away and must stay out.

    With a tolerance well under half a cell the result must be identical to the strict
    one-sided fit, so the emission-driving normal field is never averaged across the PEC.
    """
    import numpy as np

    from rf_gun.fieldmaps.coordinates import one_sided_surface_limit

    cell = 100.0e-6
    # Straddling stencil: the metal-side node at -51.2 um must never be admitted.
    coordinate = np.array([-51.216e-6, 48.784e-6, 148.784e-6, 248.784e-6])
    values = np.array([0.0, 2.0e7, 1.9e7, 1.8e7])

    strict = one_sided_surface_limit(
        coordinate, values, surface_m=0.0, vacuum_side="positive",
        n_points=3, polynomial_order=1, surface_tolerance_m=0.0,
    )
    toleranced = one_sided_surface_limit(
        coordinate, values, surface_m=0.0, vacuum_side="positive",
        n_points=3, polynomial_order=1, surface_tolerance_m=0.1 * cell,
    )
    assert toleranced == strict
    # And it must extrapolate to the full vacuum-side value, not half of it.
    assert toleranced > 1.9e7
