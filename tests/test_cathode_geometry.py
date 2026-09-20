"""Tests for `rf_gun.cathode_geometry` (plan Sec. 3.2's required synthetic-ray test list plus the
true-bevel-area and legacy-cross-check requirements of Sec. 3.3/15.1).

No RF-Track dependency -- these are pure analytic-geometry tests against plain numpy arrays.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rf_gun.back_bombardment import (
    DEFAULT_CATHODE_CHAMFER_WIDTH_MM,
    SURFACE_CATHODE_CHAMFER,
    SURFACE_CATHODE_FACE,
    classify_impact_surface,
)
from rf_gun.cathode_geometry import (
    SURFACE_CATHODE_BEVEL,
    SURFACE_CATHODE_FLAT,
    SURFACE_HOLDER,
    SURFACE_UNKNOWN,
    CathodeGeometry,
)


@pytest.fixture
def geom() -> CathodeGeometry:
    """Default confirmed-hardware geometry (plan addendum Sec. 19.2): 1.4mm flat radius, 0.2mm x
    45deg bevel, so `bevel_outer_radius_mm = 1.6mm`; `holder_outer_radius_mm` defaults to
    `1.6 + 3.0 = 4.6mm`."""
    return CathodeGeometry()


# ------------------------------------------------------------------------------------------
# Geometry/area properties
# ------------------------------------------------------------------------------------------

def test_defaults_match_confirmed_hardware(geom: CathodeGeometry):
    assert geom.flat_radius_mm == pytest.approx(1.4)
    assert geom.bevel_width_mm == pytest.approx(DEFAULT_CATHODE_CHAMFER_WIDTH_MM)
    assert geom.bevel_angle_deg == pytest.approx(45.0)
    assert geom.bevel_outer_radius_mm == pytest.approx(1.6)
    assert geom.holder_outer_radius_mm == pytest.approx(4.6)


def test_bevel_true_area_exceeds_naive_projected_annulus_by_sqrt2(geom: CathodeGeometry):
    """Plan Sec. 3.3: heat flux on the bevel must use the true (slanted) area, not the projected
    annulus. At the confirmed 45deg bevel angle the true area exceeds the naive projected annulus
    area by exactly `1/cos(45deg) = sqrt(2)`."""
    r_in, r_out = geom.flat_radius_mm, geom.bevel_outer_radius_mm
    naive_projected_area_mm2 = np.pi * (r_out**2 - r_in**2)
    assert geom.bevel_true_area_mm2 == pytest.approx(naive_projected_area_mm2 * np.sqrt(2.0), rel=1e-12)
    assert geom.bevel_true_area_mm2 > naive_projected_area_mm2


def test_bevel_slant_width_at_45deg_is_sqrt2_times_radial_width(geom: CathodeGeometry):
    assert geom.bevel_slant_width_mm == pytest.approx(geom.bevel_width_mm * np.sqrt(2.0), rel=1e-12)


def test_z_of_bevel_boundary_values(geom: CathodeGeometry):
    """z=0 at r=flat_radius_mm (continuous with the flat face); recedes to negative z (into the
    solid) at the bevel's outer edge -- at 45deg, by exactly `bevel_width_mm` (tan(45)=1)."""
    z_at_flat_radius = geom.z_of_bevel(np.array([geom.flat_radius_mm]))
    z_at_outer = geom.z_of_bevel(np.array([geom.bevel_outer_radius_mm]))
    assert z_at_flat_radius[0] == pytest.approx(0.0, abs=1e-12)
    assert z_at_outer[0] == pytest.approx(-geom.bevel_width_mm, rel=1e-12)
    assert z_at_outer[0] < 0.0


# ------------------------------------------------------------------------------------------
# Cross-check against the existing legacy classifier (rf_gun.back_bombardment)
# ------------------------------------------------------------------------------------------

def test_classify_surface_by_radius_matches_legacy_face_chamfer_split(geom: CathodeGeometry):
    """`classify_surface_by_radius` must reproduce the same face/chamfer split as the existing
    `rf_gun.back_bombardment.classify_impact_surface` for the same default geometry -- checked only
    up to the bevel's outer radius, since the legacy function collapses everything beyond that into
    one `SURFACE_EXCLUDED` label while this module further distinguishes holder/unknown (documented
    difference, not a disagreement to reproduce)."""
    r_mm = np.linspace(0.0, geom.bevel_outer_radius_mm, 97)
    # Include the exact face/chamfer/outer-edge boundaries explicitly.
    r_mm = np.concatenate([r_mm, [0.0, geom.flat_radius_mm, geom.bevel_outer_radius_mm]])

    legacy = classify_impact_surface(r_mm, geom.flat_radius_mm, geom.bevel_width_mm)
    ours = geom.classify_surface_by_radius(r_mm)

    legacy_is_face = legacy == SURFACE_CATHODE_FACE
    legacy_is_chamfer = legacy == SURFACE_CATHODE_CHAMFER
    assert np.all(legacy_is_face | legacy_is_chamfer)  # every r in this range is face or chamfer
    assert np.array_equal(ours[legacy_is_face], np.full(legacy_is_face.sum(), SURFACE_CATHODE_FLAT))
    assert np.array_equal(ours[legacy_is_chamfer], np.full(legacy_is_chamfer.sum(), SURFACE_CATHODE_BEVEL))


def test_classify_surface_by_radius_beyond_bevel_and_nonfinite(geom: CathodeGeometry):
    r_mm = np.array([geom.bevel_outer_radius_mm + 1e-6, geom.holder_outer_radius_mm, geom.holder_outer_radius_mm + 1.0, np.nan, np.inf])
    code = geom.classify_surface_by_radius(r_mm)
    assert code[0] == SURFACE_HOLDER
    assert code[1] == SURFACE_HOLDER
    assert code[2] == SURFACE_UNKNOWN
    assert code[3] == SURFACE_UNKNOWN
    assert code[4] == SURFACE_UNKNOWN


# ------------------------------------------------------------------------------------------
# Synthetic straight rays (plan Sec. 3.2's required list): center, flat-face edge, bevel,
# bevel/holder boundary, outside holder.
# ------------------------------------------------------------------------------------------

def test_straight_ray_center_hits_flat_face(geom: CathodeGeometry):
    result = geom.intersect_ray(0.0, 0.0, 1.0, 0.0, 0.0, -1.0)
    assert bool(result.hit)
    assert result.surface_code == SURFACE_CATHODE_FLAT
    assert result.x_hit_mm == pytest.approx(0.0, abs=1e-12)
    assert result.y_hit_mm == pytest.approx(0.0, abs=1e-12)
    assert result.z_hit_mm == pytest.approx(0.0, abs=1e-12)
    assert result.n_in_x == pytest.approx(0.0)
    assert result.n_in_y == pytest.approx(0.0)
    assert result.n_in_z == pytest.approx(-1.0)
    assert result.cos_incidence == pytest.approx(1.0)
    assert result.incidence_angle_rad == pytest.approx(0.0, abs=1e-9)


def test_straight_ray_flat_face_edge(geom: CathodeGeometry):
    result = geom.intersect_ray(geom.flat_radius_mm, 0.0, 1.0, 0.0, 0.0, -1.0)
    assert bool(result.hit)
    assert result.surface_code == SURFACE_CATHODE_FLAT
    assert result.z_hit_mm == pytest.approx(0.0, abs=1e-12)
    assert result.x_hit_mm == pytest.approx(geom.flat_radius_mm)


def test_straight_ray_bevel(geom: CathodeGeometry):
    r0 = 0.5 * (geom.flat_radius_mm + geom.bevel_outer_radius_mm)  # 1.5mm, mid-bevel
    result = geom.intersect_ray(r0, 0.0, 0.5, 0.0, 0.0, -1.0)
    assert bool(result.hit)
    assert result.surface_code == SURFACE_CATHODE_BEVEL
    expected_z = geom.z_of_bevel(np.array([r0]))[0]
    assert result.z_hit_mm == pytest.approx(expected_z)
    assert result.x_hit_mm == pytest.approx(r0)
    assert result.cos_incidence > 0.0
    # Inward normal at 45deg: equal parts radially inward and -z, unit length.
    assert result.n_in_x == pytest.approx(-np.sin(geom.bevel_angle_rad))
    assert result.n_in_z == pytest.approx(-np.cos(geom.bevel_angle_rad))


def test_straight_ray_bevel_holder_boundary(geom: CathodeGeometry):
    """A straight-down ray at exactly `r=bevel_outer_radius_mm` (the disk's own outer edge) must
    land on the bevel cone's outer corner (`z=-bevel_width_mm` at 45deg), not on the z=0 holder
    placeholder plane -- the bevel band is inclusive of its outer radius (matching
    `classify_surface_by_radius`'s own convention)."""
    result = geom.intersect_ray(geom.bevel_outer_radius_mm, 0.0, 0.5, 0.0, 0.0, -1.0)
    assert bool(result.hit)
    assert result.surface_code == SURFACE_CATHODE_BEVEL
    assert result.z_hit_mm == pytest.approx(-geom.bevel_width_mm)


def test_ray_in_holder_band_misses_because_that_annulus_is_vacuum(geom: CathodeGeometry):
    """As-built, the annulus beyond the 3.2 mm LaB6 disk is open vacuum out to the cavity nose
    bore -- there is no holder FACE at z=0 for a returning electron to strike. It simply passes
    the cathode plane. Modelling it as solid invented impacts that never happen (19,698 rays
    carrying 25.4 uJ/RF period on the 1650 K / 0 A production run)."""
    r0 = geom.bevel_outer_radius_mm + 0.5 * (geom.holder_outer_radius_mm - geom.bevel_outer_radius_mm)
    result = geom.intersect_ray(r0, 0.0, 0.5, 0.0, 0.0, -1.0)
    assert not bool(result.hit)
    assert result.surface_code == SURFACE_UNKNOWN


def test_holder_band_can_be_restored_for_comparison(geom: CathodeGeometry):
    """`holder_face_is_solid=True` reproduces the pre-fix geometry, so the two can be compared."""
    solid = dataclasses.replace(geom, holder_face_is_solid=True)
    r0 = solid.bevel_outer_radius_mm + 0.5 * (solid.holder_outer_radius_mm - solid.bevel_outer_radius_mm)
    result = solid.intersect_ray(r0, 0.0, 0.5, 0.0, 0.0, -1.0)
    assert bool(result.hit)
    assert result.surface_code == SURFACE_HOLDER
    assert result.z_hit_mm == pytest.approx(0.0, abs=1e-9)


def test_lab6_disk_itself_is_unaffected_by_the_holder_change(geom: CathodeGeometry):
    """The whole point: removing the fictitious holder face must not change what lands on LaB6."""
    solid = dataclasses.replace(geom, holder_face_is_solid=True)
    for r0 in (0.0, 0.7, 1.39, 1.45, 1.55):
        vac = geom.intersect_ray(r0, 0.0, 0.5, 0.0, 0.0, -1.0)
        sol = solid.intersect_ray(r0, 0.0, 0.5, 0.0, 0.0, -1.0)
        assert bool(vac.hit) == bool(sol.hit)
        assert vac.surface_code == sol.surface_code


def test_straight_ray_outside_holder_misses(geom: CathodeGeometry):
    result = geom.intersect_ray(geom.holder_outer_radius_mm + 1.0, 0.0, 0.5, 0.0, 0.0, -1.0)
    assert not bool(result.hit)
    assert result.surface_code == SURFACE_UNKNOWN
    assert np.isnan(result.x_hit_mm)
    assert np.isnan(result.z_hit_mm)
    assert np.isnan(result.cos_incidence)


# ------------------------------------------------------------------------------------------
# Forward-emitted particles must never count as a hit (plan Sec. 3.1/3.2).
# ------------------------------------------------------------------------------------------

def test_forward_emitted_particles_never_counted(geom: CathodeGeometry):
    # Freshly emitted from the center of the flat face, moving straight out into vacuum (pz>0).
    r1 = geom.intersect_ray(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert not bool(r1.hit)

    # Already past the surface (small negative z, as if a backstop integration step overshot) but
    # still moving forward (+z) -- inward-momentum test must reject this regardless of position.
    r2 = geom.intersect_ray(0.0, 0.0, -0.01, 0.0, 0.0, 1.0)
    assert not bool(r2.hit)

    # An oblique forward-going ray from above the bevel.
    r3 = geom.intersect_ray(1.5, 0.0, 0.2, 0.3, 0.0, 0.9)
    assert not bool(r3.hit)

    # Purely transverse motion (pz=0): never reaches z=0 or the bevel cone's finite z range.
    r4 = geom.intersect_ray(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    assert not bool(r4.hit)


def test_forward_emitted_vectorized_mixed_with_real_hits(geom: CathodeGeometry):
    """A batch mixing genuine returning rays and freshly emitted ones -- only the returning ones
    should register a hit, each independently and without cross-contamination between rows."""
    x0 = np.array([0.0, 0.0, 1.5, 0.0])
    y0 = np.zeros(4)
    z0 = np.array([1.0, 0.0, 0.5, 1.0])
    px = np.zeros(4)
    py = np.zeros(4)
    pz = np.array([-1.0, 1.0, -1.0, 1.0])  # rows 0,2 return; rows 1,3 are forward-emitted

    result = geom.intersect_ray(x0, y0, z0, px, py, pz)
    assert result.hit.tolist() == [True, False, True, False]
    assert result.surface_code[0] == SURFACE_CATHODE_FLAT
    assert result.surface_code[2] == SURFACE_CATHODE_BEVEL
    assert result.surface_code[1] == SURFACE_UNKNOWN
    assert result.surface_code[3] == SURFACE_UNKNOWN


# ------------------------------------------------------------------------------------------
# Oblique ray that truly intersects the bevel, at a different (x,y,z) than a naive straight
# projection to the z=0 plane would suggest (plan Sec. 3.2's explicit required test).
# ------------------------------------------------------------------------------------------

def test_oblique_ray_hits_bevel_not_naive_z0_projection(geom: CathodeGeometry):
    """Construct a ray, well above the cathode, angled so that:

    - its true first intersection is the tilted 45deg bevel cone, at negative z and comfortably
      inside the physical bevel radius band;
    - the point where the SAME straight ray crosses the mathematical z=0 plane lands in the
      *unmodeled gap* `flat_radius_mm < r <= bevel_outer_radius_mm` -- empty space, not solid,
      because the true bevel surface at that radius is set back to negative z. A naive
      "project straight to z=0 and read off the radius" reconstruction (as the legacy
      `rf_gun.back_bombardment` ballistic method effectively does) would therefore either invent a
      surface where there is none, or at best report the wrong (x,y,z); the analytic ray-cast must
      instead find no candidate at z=0 and correctly resolve the true bevel intersection instead.

    Parameters solved directly from the on-axis cone/plane equations so the expected values below
    are independent of `CathodeGeometry.intersect_ray`'s own implementation.
    """
    z0 = 4.96
    a = np.deg2rad(16.64)
    ux, uz = np.sin(a), -np.cos(a)  # unit direction: outward in +x, forward (into -z) in z

    flat_r = geom.flat_radius_mm
    bevel_outer = geom.bevel_outer_radius_mm
    tan_theta = np.tan(geom.bevel_angle_rad)

    # True (on-axis, x0=y0=0) bevel-cone crossing: z0 + uz*t = flat_r - tan(theta)*ux*t.
    t_cone = (flat_r - z0) / (uz + tan_theta * ux)
    r_cone = ux * t_cone
    z_cone = z0 + uz * t_cone

    # Where the same ray crosses the mathematical z=0 plane.
    t_flat = -z0 / uz
    r_flat = ux * t_flat

    # Sanity-check the constructed scenario itself before trusting the module against it.
    assert flat_r < r_cone < bevel_outer, "test setup: cone crossing must be within the bevel band"
    assert flat_r < r_flat < bevel_outer, "test setup: naive z=0 projection must land in the unmodeled gap"
    assert t_cone > 0.0 and z_cone < 0.0

    result = geom.intersect_ray(0.0, 0.0, z0, ux, 0.0, uz)
    assert bool(result.hit)
    assert result.surface_code == SURFACE_CATHODE_BEVEL
    assert result.z_hit_mm == pytest.approx(z_cone, rel=1e-9)
    assert result.x_hit_mm == pytest.approx(r_cone, rel=1e-9)
    assert result.z_hit_mm < 0.0
    # The true hit is not at the naive z=0 projection point.
    assert result.z_hit_mm != pytest.approx(0.0, abs=1e-6)
    assert result.x_hit_mm != pytest.approx(r_flat, abs=1e-6)


def test_oblique_ray_off_axis_matches_rotated_on_axis_case(geom: CathodeGeometry):
    """The same oblique scenario as above, rotated 40deg about the z-axis into a genuinely 3D ray
    (nonzero x0,y0,px,py) -- azimuthal symmetry means the radius/z of the hit must be unchanged."""
    z0 = 4.96
    a = np.deg2rad(16.64)
    ux, uz = np.sin(a), -np.cos(a)
    phi = np.deg2rad(40.0)

    result_on_axis = geom.intersect_ray(0.0, 0.0, z0, ux, 0.0, uz)
    result_rotated = geom.intersect_ray(
        0.0, 0.0, z0, ux * np.cos(phi), ux * np.sin(phi), uz
    )
    assert bool(result_rotated.hit)
    assert result_rotated.surface_code == result_on_axis.surface_code
    assert result_rotated.z_hit_mm == pytest.approx(float(result_on_axis.z_hit_mm))
    r_rotated = np.hypot(result_rotated.x_hit_mm, result_rotated.y_hit_mm)
    assert r_rotated == pytest.approx(float(result_on_axis.x_hit_mm))
