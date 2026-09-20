"""Tests for rf_gun.cathode_fields: signed field conventions, RF field-map sampling (units
verified against the RF-Track reference manual, not assumed -- get_field takes mm/mm/c, not
m/seconds), and Tier B (zero-weight probe) space-charge/mirror field extraction.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import gaussian_bunch_matrix as _small_gaussian_bunch_matrix

import rf_gun as rg
from rf_gun.constants import ME_MEV


def test_extraction_field_zero_when_retarding():
    Ez = np.array([-5e6, 5e6, 0.0])
    F = rg.extraction_field(Ez)
    np.testing.assert_allclose(F, [5e6, 0.0, 0.0])


def test_extraction_field_applies_beta_enh():
    Ez = np.array([-2e6])
    F = rg.extraction_field(Ez, beta_enh=3.0)
    np.testing.assert_allclose(F, [6e6])


def test_sample_rf_field_on_cathode_matches_uniform_synthetic_field():
    """Confirms get_field's real unit convention (mm, mm/c) end to end -- a wrong conversion here
    would silently return the field at the wrong physical location instead of erroring."""
    nz, nr = 21, 5
    z_grid_m = np.linspace(0.0, 0.03, nz)
    r_grid_m = np.linspace(0.0, 0.005, nr)
    Ez = np.full((nz, nr), -1.0e6 + 0j)
    Er = np.zeros((nz, nr), dtype=complex)

    p = rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=0.0, z_max_m=0.03,
        hr_m=float(r_grid_m[1]), hz_m=float(z_grid_m[1]), dt_mm=0.5,
    )
    V = rg.build_volume(rg.rft, Er, Ez, 0.0, p)

    x_probe = np.array([0.0, 0.001, 0.002])
    y_probe = np.zeros_like(x_probe)
    t_probe = np.array([0.0, 1.0e-12])
    Ez_sampled = rg.sample_rf_field_on_cathode(rg.rft, V, x_probe, y_probe, t_probe, z_probe_m=0.0005)
    assert Ez_sampled.shape == (3, 2)

    # t=0: exact crest value (uniform synthetic field, real phasor at reference phase).
    np.testing.assert_allclose(Ez_sampled[:, 0], -1.0e6, rtol=1e-6)
    # t=1e-12 s: the field oscillates at f_hz -- an exact match to -1e6*cos(omega*t) (not just "close
    # to -1e6") confirms get_field is applying real RF time-dependence, not returning a static value.
    omega = 2.0 * np.pi * 2.856e9
    expected_t1 = -1.0e6 * np.cos(omega * 1.0e-12)
    np.testing.assert_allclose(Ez_sampled[:, 1], expected_t1, rtol=1e-6)


def test_zero_weight_probe_matches_tiny_weight_probe():
    """Tier B's core assumption (guide Sec. 9): a zero-weight probe does not perturb the charge
    mesh and measures the same ambient field a real (negligible-charge) particle would."""
    rng = np.random.default_rng(7)
    M_source = _small_gaussian_bunch_matrix(200, sigma_r_mm=0.4, z_mm=3.0, N_real_each=1.0e7, rng=rng)
    sc = rg.rft.SpaceCharge_PIC_FreeSpace(32, 32, 32)

    probe_x = np.array([0.2e-3])  # 0.2 mm, off-axis to avoid mesh-degenerate coincidence
    probe_y = np.zeros_like(probe_x)
    E_zero = rg.extract_sc_field_with_probes(rg.rft, sc, M_source, probe_x, probe_y, probe_z_m=3.5e-3)

    sc2 = rg.rft.SpaceCharge_PIC_FreeSpace(32, 32, 32)
    tiny_probe = np.array([[0.2, 0.0, 0.0, 0.0, 3.5, 0.0, ME_MEV, -1.0, 1.0, 0.0]])
    combined = np.vstack([M_source, tiny_probe])
    B = rg.rft.Bunch6dT(combined)
    F_tiny = np.asarray(sc2.compute_force(B))[-1, 2]
    E_tiny = F_tiny * -1.0e6

    assert E_zero[0] == pytest.approx(E_tiny, rel=1e-4)


def test_extract_sc_and_mirror_isolates_attractive_mirror_component():
    rng = np.random.default_rng(11)
    M_source = _small_gaussian_bunch_matrix(200, sigma_r_mm=0.4, z_mm=3.0, N_real_each=1.0e7, rng=rng)
    probe_x = np.array([0.0, 0.2e-3, 0.5e-3])
    probe_y = np.zeros_like(probe_x)

    E_free, E_plus_mirror = rg.extract_sc_and_mirror_from_snapshot(
        rg.rft, M_source, probe_x, probe_y, probe_z_m=3.0e-3,
        sc_nx=32, sc_ny=32, sc_nz=32, mirror_z_m=0.0,
    )
    E_mirror = E_plus_mirror - E_free
    # The bunch is above the z=0 mirror plane; its image is attractive, i.e. pulls the (real,
    # negative) charge back toward z=0 -- a positive E_z contribution accelerating an electron
    # toward -z would require E_z>0 (force=-eE; F_z<0 needs E_z>0 for an electron).
    assert np.all(E_mirror > 0.0)


def test_sampled_field_at_r0_matches_on_axis_phasor():
    """Guide Sec. 16.4 #4: radial field sampling at r=0 must match the on-axis phasor."""
    nz, nr = 21, 6
    z_grid_m = np.linspace(0.0, 0.03, nz)
    r_grid_m = np.linspace(0.0, 0.006, nr)
    # A field that genuinely varies with r (not the degenerate-uniform case used above), so this
    # test can't pass by accident if the sampling code silently ignored the r argument.
    Ez = np.zeros((nz, nr), dtype=complex)
    for j, r in enumerate(r_grid_m):
        Ez[:, j] = -1.0e6 * (1.0 + 2.0 * (r / r_grid_m[-1]) ** 2) + 0j
    Er = np.zeros((nz, nr), dtype=complex)

    p = rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=0.0, z_max_m=0.03,
        hr_m=float(r_grid_m[1]), hz_m=float(z_grid_m[1]), dt_mm=0.5,
    )
    V = rg.build_volume(rg.rft, Er, Ez, 0.0, p)

    on_axis_phasor = rg.find_Ez_axis_phasor_at_z0(Ez, z_grid_m, z0_m=0.0005)
    Ez_r0 = rg.sample_rf_field_on_cathode(rg.rft, V, np.array([0.0]), np.array([0.0]), np.array([0.0]), z_probe_m=0.0005)
    assert Ez_r0[0, 0] == pytest.approx(float(np.real(on_axis_phasor)), rel=1e-3)


def test_field_sampling_is_axisymmetric():
    """Guide Sec. 16.4 #5: field at a given radius must not depend on azimuth. The underlying
    RF_FieldMap_2d is an inherently axisymmetric (r,z) representation -- even though the sampling
    helper now takes genuine (x,y) point pairs (for the T(x,y)/E(x,y,t) asymmetric-cathode
    upgrade), the field itself must still depend only on r=sqrt(x^2+y^2), not on azimuth.
    """
    nz, nr = 21, 6
    z_grid_m = np.linspace(0.0, 0.03, nz)
    r_grid_m = np.linspace(0.0, 0.006, nr)
    Ez = np.zeros((nz, nr), dtype=complex)
    for j, r in enumerate(r_grid_m):
        Ez[:, j] = -1.0e6 * (1.0 + 2.0 * (r / r_grid_m[-1]) ** 2) + 0j
    Er = np.zeros((nz, nr), dtype=complex)

    p = rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=0.0, z_max_m=0.03,
        hr_m=float(r_grid_m[1]), hz_m=float(z_grid_m[1]), dt_mm=0.5,
    )
    V = rg.build_volume(rg.rft, Er, Ez, 0.0, p)

    r_test = 0.003
    azimuths_rad = np.array([0.0, np.pi / 3.0, np.pi / 2.0, np.pi, 1.7 * np.pi])
    x_test = r_test * np.cos(azimuths_rad)
    y_test = r_test * np.sin(azimuths_rad)
    E_by_azimuth = rg.sample_rf_field_on_cathode(rg.rft, V, x_test, y_test, np.array([0.0]), z_probe_m=0.0005)
    np.testing.assert_allclose(E_by_azimuth[:, 0], E_by_azimuth[0, 0], rtol=1e-6)


def test_inspect_rftrack_field_capabilities_reports_tier_B():
    report = rg.inspect_rftrack_field_capabilities(rg.rft)
    assert report["tier_B_sc_compute_force_available"] is True
    assert report["tier_A_sc_compute_field_available"] is False
    assert report["preferred_tier"] == "B"


# ---------------------------------------------------------------------------
# analytic_sc_and_mirror_surface_field: pure closed-form electrostatics, no RF-Track binding
# needed at all -- these tests exercise the formula directly against the classic point-charge-
# above-a-grounded-plane textbook result, independent of any RF-Track-dependent test above.
# ---------------------------------------------------------------------------

def test_analytic_surface_field_single_source_matches_closed_form_on_axis():
    """On-axis (rho=0) closed form: E_z(0,0,0) = -q*z/(2*pi*eps0*z^3) = -q/(2*pi*eps0*z^2) for
    the mirror-inclusive field, and exactly half that for the free (no-image) field."""
    from rf_gun.constants import epsilon_0

    q_C = 1.6e-9  # 1.6 nC, a representative macroparticle charge for this project
    z_m = 2.0e-4  # 0.2 mm above the cathode
    E_free, E_plus_mirror = rg.analytic_sc_and_mirror_surface_field(
        np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]), np.array([z_m]), np.array([q_C]),
    )
    expected_plus_mirror = -q_C / (2.0 * np.pi * epsilon_0 * z_m ** 2)
    assert E_plus_mirror[0] == pytest.approx(expected_plus_mirror, rel=1e-12)
    assert E_free[0] == pytest.approx(0.5 * expected_plus_mirror, rel=1e-12)


def test_analytic_surface_field_mirror_is_exactly_double_free():
    """The image-charge identity E_sc_plus_mirror = 2*E_sc_free at the conductor surface must hold
    off-axis and for multiple sources too, not just the on-axis single-source case above."""
    rng = np.random.default_rng(0)
    n_src = 12
    src_x = rng.uniform(-1.5e-3, 1.5e-3, n_src)
    src_y = rng.uniform(-1.5e-3, 1.5e-3, n_src)
    src_z = rng.uniform(1.0e-5, 5.0e-4, n_src)
    src_q = rng.uniform(0.1e-9, 2.0e-9, n_src)
    probe_x = np.linspace(-1.0e-3, 1.0e-3, 7)
    probe_y = np.linspace(-1.0e-3, 1.0e-3, 7)
    E_free, E_plus_mirror = rg.analytic_sc_and_mirror_surface_field(probe_x, probe_y, src_x, src_y, src_z, src_q)
    np.testing.assert_allclose(E_plus_mirror, 2.0 * E_free, rtol=1e-12)


def test_analytic_surface_field_source_at_z_zero_contributes_nothing():
    """A source that hasn't yet drifted from the cathode (z=0 -- e.g. charge from the *current*
    emission time bin, not yet an "already emitted" sheet) must contribute exactly zero, not a
    singular/NaN value: this is what lets the analytic kernel naturally exclude a bin's own
    self-field, unlike the old ballistic-z-probe approach's z=0 degeneracy (see
    rf_gun.emission_iteration module docstring)."""
    E_free, E_plus_mirror = rg.analytic_sc_and_mirror_surface_field(
        np.array([0.0, 5.0e-4]), np.array([0.0, 0.0]),
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0e-9]),
    )
    assert np.all(np.isfinite(E_free)) and np.all(np.isfinite(E_plus_mirror))
    np.testing.assert_allclose(E_free, 0.0)
    np.testing.assert_allclose(E_plus_mirror, 0.0)


def test_analytic_surface_field_superposition():
    """Two independent sources' fields must add linearly (Gauss's law / superposition)."""
    probe_x, probe_y = np.array([3.0e-4]), np.array([-2.0e-4])
    E1_free, E1_mirror = rg.analytic_sc_and_mirror_surface_field(
        probe_x, probe_y, np.array([0.0]), np.array([0.0]), np.array([1.0e-4]), np.array([1.0e-9]),
    )
    E2_free, E2_mirror = rg.analytic_sc_and_mirror_surface_field(
        probe_x, probe_y, np.array([2.0e-4]), np.array([1.0e-4]), np.array([3.0e-4]), np.array([0.5e-9]),
    )
    E12_free, E12_mirror = rg.analytic_sc_and_mirror_surface_field(
        probe_x, probe_y,
        np.array([0.0, 2.0e-4]), np.array([0.0, 1.0e-4]), np.array([1.0e-4, 3.0e-4]), np.array([1.0e-9, 0.5e-9]),
    )
    np.testing.assert_allclose(E12_free, E1_free + E2_free, rtol=1e-12)
    np.testing.assert_allclose(E12_mirror, E1_mirror + E2_mirror, rtol=1e-12)


def test_analytic_surface_field_induced_charge_integrates_to_minus_q():
    """sigma(rho) = eps0*E_z(rho,0) integrated over the whole plane must recover exactly -q (the
    standard grounded-plane induced-charge sum rule) -- a global cross-check independent of the
    on-axis closed form used in the first test above."""
    from rf_gun.constants import epsilon_0

    q_C = 1.0e-9
    z_m = 5.0e-4
    rho_max_m = 2.0  # far beyond z_m, so the tail contribution is negligible
    n = 200_000
    rho_edges = np.linspace(0.0, rho_max_m, n + 1)
    rho = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    d_rho = rho_edges[1] - rho_edges[0]
    _, E_plus_mirror = rg.analytic_sc_and_mirror_surface_field(
        rho, np.zeros_like(rho), np.array([0.0]), np.array([0.0]), np.array([z_m]), np.array([q_C]),
    )
    sigma = epsilon_0 * E_plus_mirror
    Q_induced = np.sum(sigma * 2.0 * np.pi * rho * d_rho)
    assert Q_induced == pytest.approx(-q_C, rel=2e-3)


def test_analytic_surface_field_same_cell_term_matches_exact_disk_closed_form():
    """The same-cell (rho=0) term with a radius given must match the exact on-axis
    uniformly-charged-disk formula E_z(z) = -q/(2*pi*eps0*a^2)*(1-z/sqrt(z^2+a^2)) exactly (this
    is now a closed-form physics result, not a tunable regularization -- see the function's
    docstring), and a different-cell (rho>0) term with the same radius must be *unaffected* by it
    (plain point-charge kernel), confirming the two branches are genuinely independent."""
    from rf_gun.constants import epsilon_0

    q_C = 1.0e-9
    a_m = 2.0e-4
    z_m = 5.0e-5

    _, E_plus_mirror_same_cell = rg.analytic_sc_and_mirror_surface_field(
        np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]), np.array([z_m]), np.array([q_C]),
        source_softening_m=np.array([a_m]),
    )
    expected_disk = -q_C / (2.0 * np.pi * epsilon_0 * a_m ** 2) * (1.0 - z_m / np.sqrt(z_m ** 2 + a_m ** 2))
    assert E_plus_mirror_same_cell[0] == pytest.approx(2.0 * expected_disk, rel=1e-12)

    rho_m = 3.0e-4  # a different cell -- same q, z, and softening radius, but rho>0 now
    _, E_plus_mirror_off_cell = rg.analytic_sc_and_mirror_surface_field(
        np.array([rho_m]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]), np.array([z_m]), np.array([q_C]),
        source_softening_m=np.array([a_m]),
    )
    expected_point = -q_C * z_m / (2.0 * np.pi * epsilon_0 * (rho_m ** 2 + z_m ** 2) ** 1.5)
    assert E_plus_mirror_off_cell[0] == pytest.approx(expected_point, rel=1e-12)


def test_analytic_surface_field_same_cell_disk_reduces_to_point_charge_far_from_cathode():
    """As z >> a, the exact on-axis disk formula must reduce to the ordinary on-axis point-charge
    field -- the physical justification for treating this as an "exact" replacement for the same-
    cell term rather than an approximation like the softening it replaces."""
    from rf_gun.constants import epsilon_0

    q_C = 1.0e-9
    a_m = 2.0e-4
    z_m = 200.0 * a_m
    _, E_plus_mirror = rg.analytic_sc_and_mirror_surface_field(
        np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]), np.array([z_m]), np.array([q_C]),
        source_softening_m=np.array([a_m]),
    )
    expected_point_charge = -q_C / (2.0 * np.pi * epsilon_0 * z_m ** 2)
    assert E_plus_mirror[0] == pytest.approx(expected_point_charge, rel=1e-4)


def test_analytic_surface_field_same_cell_at_z_zero_still_excluded_when_softening_given():
    """The current-emission-time-bin self-exclusion (z=0 -> zero contribution) must survive even
    when a same-cell source is given a nonzero softening/disk radius -- otherwise the exact disk
    formula's own (nonzero, physically real) z=0+ "half-sheet" limit would leak a bin's own just-
    created charge back into its own field evaluation, exactly the instability the z=0 convention
    guards against (see rf_gun.emission_iteration's `active = t_flat_s <= t_j` mask)."""
    E_free, E_plus_mirror = rg.analytic_sc_and_mirror_surface_field(
        np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0e-9]),
        source_softening_m=np.array([2.0e-4]),
    )
    assert np.all(np.isfinite(E_free)) and np.all(np.isfinite(E_plus_mirror))
    np.testing.assert_allclose(E_free, 0.0)
    np.testing.assert_allclose(E_plus_mirror, 0.0)


def test_analytic_surface_field_diverges_unsoftened_but_bounded_when_softened():
    """Characterizes (rather than hides) the point-charge kernel's own near-field sensitivity,
    the analytic-method counterpart of test_mirror_mesh_convergence.py's PIC-mesh finding.

    A source re-evaluated at its own cathode grid cell after only just drifting away from z=0 (a
    real, common case in rf_gun.emission_iteration's fixed-sample scheme -- see
    EmissionFieldIterationConfig.field_probe_method's docstring) sits at rho=0 relative to that
    cell's own probe point, so the unsoftened kernel's 1/z^3 dependence blows up as z shrinks --
    confirmed here directly, not just asserted. Softening by the cell's own natural scale (its
    disk-equivalent radius `a`) bounds this: scanning the softening scale by the same 16-64
    (here 0.5x-4x, since there is no natural "mesh size" to scan) ratio used for PIC mesh
    convergence keeps the field within a comparable, bounded factor -- not perfectly converged
    (this is an approximation, not exact physics -- see the docstring), but no longer diverging."""
    q_C = 1.0e-9
    a_m = 2.0e-4  # a representative cathode grid cell's disk-equivalent radius
    z_values_m = np.array([1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6])  # shrinking drift height
    zero1 = np.array([0.0])

    # One probe-source pair per z value (a (1,1) call each), not a single 5-probe/5-source call --
    # analytic_sc_and_mirror_surface_field cross-evaluates every probe against every source, so a
    # single multi-element call here would sum all 5 sources into every "probe", not pair them up.
    E_unsoftened = np.array([
        rg.analytic_sc_and_mirror_surface_field(zero1, zero1, zero1, zero1, np.array([z]), np.array([q_C]))[1][0]
        for z in z_values_m
    ])
    assert np.abs(E_unsoftened[-1]) > 100.0 * np.abs(E_unsoftened[0]), (
        "expected the unsoftened kernel to blow up as z shrinks toward the probe's own cell -- "
        "if it doesn't, the divergence this test guards against may already be fixed some other "
        "way and this test's premise should be revisited"
    )

    softening_scales = [0.5, 1.0, 2.0, 4.0]
    E_by_scale = []
    for scale in softening_scales:
        E_soft = np.array([
            rg.analytic_sc_and_mirror_surface_field(
                zero1, zero1, zero1, zero1, np.array([z]), np.array([q_C]),
                source_softening_m=np.array([scale * a_m]),
            )[1][0]
            for z in z_values_m
        ])
        E_by_scale.append(E_soft)
    E_by_scale = np.asarray(E_by_scale)  # (n_scale, n_z)

    assert np.all(E_by_scale < 0.0), "softened mirror-inclusive field changed sign across the scan"
    per_z_ratio = np.max(E_by_scale, axis=0) / np.min(E_by_scale, axis=0)
    assert np.all(per_z_ratio < 10.0), (
        f"softened field varies by more than 10x across a 0.5x-4x softening-scale scan "
        f"(ratios={per_z_ratio}); treat peak near-cathode fields from this method as "
        "order-of-magnitude, not precision, information (see UPGRADE_PLAN.md)"
    )
