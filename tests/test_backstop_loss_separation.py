"""Real RF-Track tests for `rf_gun.backstop_loss_separation` -- the Work Package 1 blocking
question from `BACK_BOMBARDMENT_MACROPULSE_IMPLEMENTATION_PLAN.md` Sec. 3.2: can backstop-loss
rows be reliably separated from dynamic-aperture-loss rows in `Volume.get_lost_particles()`'s one
combined table?

Every test here builds a real (but tiny/fast) `RF_Track.Volume` via `rf_gun.build_volume` with an
all-zero field map (so trajectories are exactly straight lines -- ground truth for "does this
particle cross R(z)" is then just comparing a linear `x(z)` against
`rf_gun.aperture.aperture_radius_profile_mm`, not a hand-derived approximation) and
`cathode_backstop_enabled=True` -- i.e. the real multi-element-Volume ambiguity the module
docstring warns about, not an artificially simplified single-element case. No mocking of RF-Track
itself anywhere in this file.
"""
from __future__ import annotations

import numpy as np
import pytest

rft = pytest.importorskip("RF_Track")

import rf_gun as rg
from rf_gun.constants import ME_MEV
from rf_gun.diagnostics import to_lost_table_array
from rf_gun.backstop_loss_separation import (
    DEFAULT_Z_SLACK_M,
    identify_backstop_loss_candidates,
)

# Shared tiny/fast Volume geometry -- z in [0, 35] mm covers the chamfer, cavity body, exit
# rounding, and a sliver of pipe 2 (L=28.854mm, see rf_gun.aperture), z_min_m=0.0 matches this
# project's actual production default (run_thermionic_tm010.py's --z_min defaults to 0.0), i.e.
# the cathode plane and the field map's own z-origin coincide -- the configuration this module's
# separation rule is validated for.
_NZ, _NR = 21, 5
_Z_MIN_M = 0.0
_Z_MAX_M = 0.035
_BACKSTOP_THICKNESS_MM = 2.0


def _zero_field():
    Er = np.zeros((_NZ, _NR), dtype=complex)
    Ez = np.zeros((_NZ, _NR), dtype=complex)
    return Er, Ez


def _build_volume(dt_mm: float = 0.5, thickness_mm: float = _BACKSTOP_THICKNESS_MM):
    Er, Ez = _zero_field()
    p = rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=_Z_MIN_M, z_min_m=_Z_MIN_M, z_max_m=_Z_MAX_M,
        hr_m=1e-3, hz_m=(_Z_MAX_M - _Z_MIN_M) / (_NZ - 1), dt_mm=dt_mm,
        cathode_backstop_enabled=True,
        cathode_backstop_thickness_mm=thickness_mm,
    )
    return rg.build_volume(rft, Er, Ez, 0.0, p)


def _track_bunch(Mext: np.ndarray, dt_mm: float = 0.5, thickness_mm: float = _BACKSTOP_THICKNESS_MM):
    """`Mext` columns: X, Px, Y, Py, Z, Pz, MASS, Q, N, T0 (mm/MeV-c convention)."""
    B0 = rft.Bunch6dT(np.asarray(Mext, dtype=float))
    V = _build_volume(dt_mm=dt_mm, thickness_mm=thickness_mm)
    Bout = V.track(B0)
    Bout_arr = np.asarray(Bout.get_phase_space("%X %Px %Y %Py %Z %Pz %id %t %E %K"), dtype=float)
    lost = to_lost_table_array(V.get_lost_particles())
    return Bout_arr, lost


def _make_row(x0, px0, z0, pz0, y0=0.0, py0=0.0):
    return [x0, px0, y0, py0, z0, pz0, ME_MEV, -1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------------------------
# Mixed-population regression test: reproduces the exact scenario the implementation plan (Sec.
# 3.2) and `rf_gun.aperture.build_cathode_backstop`'s docstring flag as unverified -- a Volume
# with BOTH the backstop and the dynamic aperture active, tracking a mix of particles constructed
# to hit each mechanism distinctly, plus the ambiguous cases the plan's own naive "Pz<0 and
# negative Z" heuristic would get wrong.
# ---------------------------------------------------------------------------------------------

def test_mixed_population_classified_correctly_with_zero_misclassification():
    rows = []
    expect_backstop = []
    label = []

    # Group A -- backstop, "already there": emitted (Z=0) already moving backward, Pz<0. RF-Track
    # flags these at their own initial state (Z==0 exactly, verified in exploratory testing) --
    # the easy case even the plan's literal Z<=0 wording gets right.
    for x0, pz0 in [(0.02, -0.05), (0.1, -0.1), (0.5, -0.2), (1.0, -0.3), (2.0, -0.5)]:
        rows.append(_make_row(x0, 0.0, 0.0, pz0))
        expect_backstop.append(True)
        label.append(f"A_already_there_x0={x0}_pz0={pz0}")

    # Group A2 -- backstop, "transit": emitted well downstream (Z0 in [2,20] mm, safely inside
    # R(z)>=R1_MM=2.5275mm for a 0.3mm transverse offset everywhere along the path), moving
    # backward. Ground truth: never violates the dynamic aperture anywhere between z0 and 0 (0.3mm
    # << R(z) for all z in [0, z0]), so must reach z=0 and be absorbed by the backstop. This is
    # the case where RF-Track's reported Z is a small POSITIVE residual, not <=0 -- exactly why
    # `z_slack_m` exists (see module docstring).
    for z0, pz0 in [(2.0, -0.05), (5.0, -0.5), (10.0, -1.0), (20.0, -3.0)]:
        rows.append(_make_row(0.3, 0.0, z0, pz0))
        expect_backstop.append(True)
        label.append(f"A2_transit_z0={z0}_pz0={pz0}")

    # Group B -- dynamic aperture, forward, well downstream: small initial offset (0.2mm, safely
    # inside R1_MM at z=0) but a divergence angle chosen so the straight-line trajectory crosses
    # the real R(z) profile in the cavity-body/exit-rounding region (z ~ 26-29mm), verified against
    # `aperture_radius_profile_mm` directly (independent of RF-Track) before tracking.
    from rf_gun.aperture import aperture_radius_profile_mm
    zz_mm = np.linspace(0.0, _Z_MAX_M * 1e3, 200_001)
    Rz_mm = aperture_radius_profile_mm(zz_mm, 0.0)
    for ratio in (0.08, 0.12, 0.15, 0.2):
        x0, pz0 = 0.2, 1.0
        px0 = ratio * pz0
        xz = x0 + (px0 / pz0) * zz_mm
        over = np.nonzero(np.abs(xz) > Rz_mm)[0]
        assert over.size, f"test construction error: ratio={ratio} never crosses R(z)"
        predicted_cross_mm = zz_mm[over[0]]
        assert predicted_cross_mm > 20.0, "test construction error: crossing not well downstream"
        rows.append(_make_row(x0, px0, 0.0, pz0))
        expect_backstop.append(False)
        label.append(f"B_dynamic_aperture_forward_ratio={ratio}_predicted_z_mm={predicted_cross_mm:.2f}")

    # Group C1 -- AMBIGUOUS: already transversely beyond the local R(z) at a positive z (27mm,
    # exit-rounding zone) while moving backward (Pz<0). A naive "Pz<0 -> backstop" rule would
    # misfile this; ground truth is dynamic-aperture (the transverse violation, not the cathode
    # plane, caused the loss), and it must be classified False.
    from rf_gun.aperture import aperture_radius_profile_mm as _prof
    R_at_27 = float(_prof(np.array([27.0]), 0.0)[0])
    for pz0 in (-0.5, -2.0):
        assert 6.0 > R_at_27, "test construction error: x0=6mm must exceed R(27mm)"
        rows.append(_make_row(6.0, 0.0, 27.0, pz0))
        expect_backstop.append(False)
        label.append(f"C1_ambiguous_pz_negative_dynamic_aperture_pz0={pz0}")

    # Group C2 -- boundary disambiguation: transversely already beyond R(0)=R1_MM=2.5275mm right
    # at the cathode plane (Z=0), but moving FORWARD (Pz>0). Ground truth is dynamic-aperture, not
    # backstop -- distinguished from Group A only by Pz's sign, since both report Z==0.
    rows.append(_make_row(3.0, 0.0, 0.0, 0.5))
    expect_backstop.append(False)
    label.append("C2_boundary_z0_forward_dynamic_aperture")

    Mext = np.array(rows, dtype=float)
    expect_backstop = np.array(expect_backstop, dtype=bool)

    Bout_arr, lost = _track_bunch(Mext, dt_mm=0.5)

    assert lost is not None and lost.shape[0] == len(rows), (
        f"expected every one of {len(rows)} particles to be lost (backstop or dynamic aperture), "
        f"got {0 if lost is None else lost.shape[0]} lost rows and "
        f"{Bout_arr.shape[0]} surviving in Bout"
    )

    mask = identify_backstop_loss_candidates(
        lost, backstop_z_min_m=-_BACKSTOP_THICKNESS_MM * 1e-3, backstop_z_max_m=0.0,
    )
    ids = lost[:, 10].astype(int)

    # Re-order the mask/expectation by particle id (bunch-construction order == id order, but
    # don't rely on the lost table preserving that order).
    order = np.argsort(ids)
    ids_sorted = ids[order]
    mask_sorted = mask[order]
    np.testing.assert_array_equal(ids_sorted, np.arange(len(rows)))

    mismatches = [
        (label[i], expect_backstop[i], bool(mask_sorted[i]), lost[order][i, 4], lost[order][i, 5])
        for i in range(len(rows))
        if expect_backstop[i] != mask_sorted[i]
    ]
    assert not mismatches, (
        "misclassified rows (label, expected_backstop, got_backstop, Z_mm, Pz_MeVc): "
        + repr(mismatches)
    )


def test_no_id_appears_in_both_bout_and_lost_table():
    """Plan Sec. 3.2 required test: backstop and dynamic-aperture loss populations are disjoint,
    and a truly-absorbed particle never also survives to `Bout`."""
    rows = [
        _make_row(0.02, 0.0, 0.0, -0.05),   # backstop, already-there
        _make_row(0.3, 0.0, 5.0, -0.5),     # backstop, transit
        _make_row(0.2, 0.08, 0.0, 1.0),     # dynamic aperture, well downstream
        _make_row(0.05, 0.0, 0.0, 0.05),    # survives (small forward pz, never lost)
    ]
    Mext = np.array(rows, dtype=float)
    Bout_arr, lost = _track_bunch(Mext, dt_mm=0.5)

    bout_ids = set(Bout_arr[:, 6].astype(int).tolist()) if Bout_arr.size else set()
    lost_ids = set(lost[:, 10].astype(int).tolist()) if lost is not None and lost.shape[0] else set()

    assert bout_ids & lost_ids == set(), f"IDs present in both Bout and lost table: {bout_ids & lost_ids}"
    assert bout_ids | lost_ids == set(range(len(rows))), "some particle id accounted for nowhere"
    assert bout_ids == {3}, f"expected only id=3 to survive to Bout, got {bout_ids}"


def test_no_id_appears_twice_in_lost_table():
    """Plan Sec. 3.2 required test: "a particle ID appears at most once in an absorbing-return
    model" -- checked here across a mixed backstop/dynamic-aperture population."""
    rows = [
        _make_row(0.02, 0.0, 0.0, -0.05),
        _make_row(1.0, 0.0, 0.0, -0.3),
        _make_row(0.3, 0.0, 10.0, -1.0),
        _make_row(0.2, 0.12, 0.0, 1.0),
        _make_row(6.0, 0.0, 27.0, -0.5),
    ]
    Mext = np.array(rows, dtype=float)
    _, lost = _track_bunch(Mext, dt_mm=0.5)
    assert lost is not None and lost.shape[0] == len(rows)
    ids = lost[:, 10].astype(int)
    assert len(set(ids.tolist())) == len(ids), f"duplicate IDs in lost table: {sorted(ids.tolist())}"


def test_ambiguous_pz_negative_dynamic_aperture_loss_is_not_misclassified():
    """Isolates Group C1 from the mixed test above: Pz<0 alone is not sufficient to identify a
    backstop candidate -- a genuine dynamic-aperture (transverse) loss can also carry Pz<0."""
    rows = [_make_row(6.0, 0.0, 27.0, -0.5), _make_row(6.0, 0.0, 27.0, -2.0)]
    Mext = np.array(rows, dtype=float)
    _, lost = _track_bunch(Mext, dt_mm=0.5)
    assert lost is not None and lost.shape[0] == 2
    assert np.all(lost[:, 5] < 0.0), "test construction error: expected Pz<0 for both rows"
    assert np.all(lost[:, 4] > 20.0), "expected Z well downstream (dynamic-aperture loss), not near 0"

    mask = identify_backstop_loss_candidates(
        lost, backstop_z_min_m=-_BACKSTOP_THICKNESS_MM * 1e-3, backstop_z_max_m=0.0,
    )
    assert not np.any(mask), "Pz<0 dynamic-aperture loss at z>>0 was misclassified as backstop"


def test_transit_backstop_hit_is_recorded_at_small_positive_z_not_negative():
    """Documents the key empirical finding driving `z_slack_m`'s existence: a backstop hit that
    requires actual transit (not starting already inside the band) is reported by
    `Volume.get_lost_particles()` at a small POSITIVE Z, not Z<=0 -- the plan's literal "negative-z
    backstop band" wording, taken alone, would systematically miss this (dominant, physically
    realistic) case."""
    rows = [_make_row(0.3, 0.0, 5.0, -0.5)]
    Mext = np.array(rows, dtype=float)
    _, lost = _track_bunch(Mext, dt_mm=0.02)
    assert lost is not None and lost.shape[0] == 1
    z_lost_mm = float(lost[0, 4])
    assert z_lost_mm > 0.0, (
        f"expected the classic positive-side residual (Z={z_lost_mm} mm) -- if RF-Track's "
        "behavior has changed and this now reports Z<=0, `identify_backstop_loss_candidates` "
        "with a smaller/zero z_slack_m may suffice and this module's docstring should be revised"
    )
    assert z_lost_mm < DEFAULT_Z_SLACK_M * 1e3, (
        f"residual {z_lost_mm} mm exceeds DEFAULT_Z_SLACK_M -- re-run the worst-case sweep in this "
        "module's docstring/PR discussion and raise DEFAULT_Z_SLACK_M if this is a genuine, "
        "reproducible new worst case"
    )


@pytest.mark.parametrize("dt_mm", [0.02, 0.1, 0.5, 2.0])
@pytest.mark.parametrize("pz0", [-0.1, -0.5, -1.0, -3.0, -10.0])
def test_backstop_transit_residual_stays_within_calibrated_slack(dt_mm, pz0):
    """Regression-pins the worst-case residual bound `DEFAULT_Z_SLACK_M` was calibrated against
    (module docstring: "up to ~0.63 mm ... across dt_mm in [0.001, 5] mm/c, Pz in [-50, -0.0001]
    MeV/c, transit distance in [2, 30] mm"), and confirms the resulting classification is correct
    at every sampled point despite the residual not shrinking monotonically with `dt_mm`."""
    rows = [_make_row(0.3, 0.0, 5.0, pz0)]
    Mext = np.array(rows, dtype=float)
    _, lost = _track_bunch(Mext, dt_mm=dt_mm)
    assert lost is not None and lost.shape[0] == 1
    z_lost_mm = float(lost[0, 4])
    assert -1e-9 <= z_lost_mm <= 1.0, (
        f"residual {z_lost_mm} mm at dt_mm={dt_mm}, pz0={pz0} exceeds this test's 1.0mm regression "
        "bound (still comfortably inside DEFAULT_Z_SLACK_M=1.5mm, but worth a closer look)"
    )
    mask = identify_backstop_loss_candidates(
        lost, backstop_z_min_m=-_BACKSTOP_THICKNESS_MM * 1e-3, backstop_z_max_m=0.0,
    )
    assert bool(mask[0]) is True


def test_empty_and_none_lost_table_return_empty_mask():
    assert identify_backstop_loss_candidates(None, backstop_z_min_m=-2e-3).shape == (0,)
    assert identify_backstop_loss_candidates(
        np.zeros((0, 11)), backstop_z_min_m=-2e-3
    ).shape == (0,)


def test_short_column_table_returns_all_false_not_raising():
    """A malformed/short table (fewer columns than `id_col` needs) returns an all-False mask
    rather than raising -- mirrors `rf_gun.particle_tags`'s shaped-not-raising convention."""
    short = np.zeros((3, 6))
    mask = identify_backstop_loss_candidates(short, backstop_z_min_m=-2e-3)
    assert mask.shape == (3,)
    assert not np.any(mask)


# ---------------------------------------------------------------------------------------------
# Aperture-wall separation (the fix for the 8.95% SURFACE_UNKNOWN rate in the KOA campaign).
#
# `identify_backstop_loss_candidates` is radius-blind and its +1.5 mm z-slack spans the whole
# cathode-side chamfer (R rises 2.5275 -> 5.76 mm over z in (0, 1.867) mm), so aperture-cone
# absorptions were being ray-cast onto the flat holder annulus at z=0 as if they had reached the
# cathode plane. Measured on the 1650 K / 0 A production run that covered 40.8% of captured events
# and 49.1% of the returned energy.
# ---------------------------------------------------------------------------------------------

from rf_gun.aperture import aperture_radius_profile_mm  # noqa: E402
from rf_gun.backstop_loss_separation import (  # noqa: E402
    DEFAULT_APERTURE_MATCH_TOLERANCE_MM,
    aperture_inward_normal,
    identify_aperture_wall_losses,
)


def test_points_on_the_chamfer_cone_are_flagged_as_aperture_losses():
    z = np.array([0.25, 0.75, 1.25, 1.75])
    R = aperture_radius_profile_mm(z, 0.0)
    assert np.all(np.diff(R) > 0), "chamfer must actually open up over this z band"
    on = identify_aperture_wall_losses(R, np.zeros_like(R), z, delta_mm=0.0)
    assert np.all(on)


def test_cathode_plane_returns_are_never_flagged_as_aperture_losses():
    """Rows that legitimately reach the LaB6 face sit >0.9 mm inside the wall on the production
    run; they must survive to the ray-cast untouched."""
    z = np.full(4, 0.02)
    r = np.array([0.0, 0.5, 1.0, 1.4])  # cathode flat, radius <= 1.4 mm
    assert not np.any(identify_aperture_wall_losses(r, np.zeros_like(r), z, delta_mm=0.0))


def test_rows_behind_the_cathode_plane_are_never_aperture_losses():
    z = np.array([-0.5, -0.1, 0.0])
    r = np.full(3, 10.0)  # absurdly large radius: the z<=0 guard must still win
    assert not np.any(identify_aperture_wall_losses(r, np.zeros_like(r), z, delta_mm=0.0))


def test_aperture_test_is_one_sided_and_catches_outward_overshoot():
    """RF-Track records the absorption at the last step, which may be inside OR outside the wall.
    Being outside is unambiguous and must be caught at any overshoot distance."""
    z = np.full(3, 1.0)
    R = float(aperture_radius_profile_mm(np.array([1.0]), 0.0)[0])
    r = np.array([R + 0.001, R + 0.5, R + 5.0])
    assert np.all(identify_aperture_wall_losses(r, np.zeros_like(r), z, delta_mm=0.0))
    # ... while a row well inside the wall is not an aperture loss.
    deep_inside = np.array([R - 10 * DEFAULT_APERTURE_MATCH_TOLERANCE_MM])
    assert not identify_aperture_wall_losses(deep_inside, np.zeros(1), np.array([1.0]))[0]


def test_aperture_inward_normal_is_unit_and_points_into_the_wall():
    z = np.array([0.5, 1.0, 1.5])
    R = aperture_radius_profile_mm(z, 0.0)
    nx, ny, nz = aperture_inward_normal(R, np.zeros_like(R), z, delta_mm=0.0)
    assert np.allclose(np.sqrt(nx**2 + ny**2 + nz**2), 1.0)
    # Solid lies at larger radius, so the inward normal has a positive radial component ...
    assert np.all(nx > 0)
    # ... and tilts backwards along -z because R(z) is increasing on the chamfer.
    assert np.all(nz < 0)


def test_outward_moving_particle_hits_the_wall_from_the_vacuum_side():
    """`p_hit . n_in > 0` is the project's heating-event test; an outward-going particle absorbed
    on the cone must satisfy it, otherwise the event would be silently rejected."""
    z = np.array([1.0])
    R = aperture_radius_profile_mm(z, 0.0)
    nx, ny, nz = aperture_inward_normal(R, np.zeros_like(R), z, delta_mm=0.0)
    px, py, pz = 0.05, 0.0, -0.7  # moving outward in +x and backwards in -z
    assert px * nx[0] + py * ny[0] + pz * nz[0] > 0
