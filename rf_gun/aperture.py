"""Dynamic radial aperture R(z): the cavity's real transverse channel, enforced by RF-Track
itself during tracking via `RF_Track.Aperture_1d`.

The channel is not a single cylinder: narrow near the cathode (a 60 deg cone, to the axis, opening
from the cathode bore to a rounded R.030 inch lip on the nose face), wide through the main cavity body, narrow again near the exit (a rounded transition
into pipe 2). A single scalar aperture cannot represent this shape, so this module builds a
tabulated R(z) profile (`aperture_radius_profile_mm`) sampled on the same z-grid as the RF field
map, and wraps it into an `Aperture_1d` element added to the same `Volume` as the field map
(`build_dynamic_aperture`), so a particle is removed the instant it crosses the real transverse
channel during tracking.

Dimensions follow the Duke FEL Laboratory cavity drawings: PF-32-103-68 R1 (cathode end plate),
PF-32-103-54 R3 (centre section) and PF-32-103-45 R0 (anode end plate).

`Aperture_1d` gotchas (manual Sec. 5.3.1: `Aperture_1d(Ra, hz, length=-1)` -- third arg is
*length*, not z0):
  1. `get_z0()` is always 0 after construction, regardless of the third argument -- the element's
     frame is always 0-based; place it globally via `V.add(A, 0, 0, global_z0, "entrance")`, never
     via `A.set_z0(...)` (confirmed to break particle-loss detection for a sub-range element).
  2. `get_z1()`/`get_length()` stay 0 (element silently inert) unless `.set_z1(length)` is called
     explicitly after construction. `build_dynamic_aperture` handles both below.

Coordinate convention: `s = z + delta_cathode_chamfer_mm`, where z=0 is the cathode emission
surface. `delta_cathode_chamfer_mm=0` means the cathode sits exactly at the start of the chamfer;
negative means the cathode is recessed upstream inside pipe 1; positive means it's already
inside the chamfer. This is a tunable knob (the user intends to try different cathode insertion
depths), so it is threaded through as `VolumeBuildParams.aperture_delta_mm`, not hardcoded.
"""
from __future__ import annotations


import numpy as np

INCH_MM = 25.4

#: Pipe 1 radius [mm]: cathode-side bore 0.199 +/- 0.001 inch, PF-32-103-68 R1 section B-B.
R1_MM = 0.5 * 0.199 * INCH_MM
#: Pipe 2 radius [mm] (= 5.055/2): exit bore 0.199 inch, PF-32-103-45 R0 section A-A.
R2_MM = 2.5275
#: Main cavity inner radius [mm] (= 68.029/2): 2.6783 +/- 0.0002 inch, PF-32-103-54 R3 section A-A.
R_CAV_MM = 34.0145
#: Cathode-cone mouth radius [mm] at the theoretical sharp corner: 0.493 inch, PF-32-103-68 R1
#: enlarged view of section B-B.
R_CONE_MOUTH_MM = 0.5 * 0.493 * INCH_MM
#: Cone angle to the beam axis [deg], PF-32-103-68 R1 section B-B.
CONE_ANGLE_TO_AXIS_DEG = 60.0
#: Axial extent of the cone from the bore to the theoretical sharp corner [mm] (2.1557), set by
#: R1_MM, R_CONE_MOUTH_MM and the 60 deg angle; drawing axial 0.085 +/- 0.001 inch = 2.159 +/- 0.025.
A_CHI_MM = float((R_CONE_MOUTH_MM - R1_MM) / np.tan(np.deg2rad(CONE_ANGLE_TO_AXIS_DEG)))
#: Cone angle from the radial plane [deg] and slant length bore to theoretical corner [mm].
CHAMFER_ANGLE_DEG = 90.0 - CONE_ANGLE_TO_AXIS_DEG
CHAMFER_LEN_MM = float(A_CHI_MM / np.sin(np.deg2rad(CHAMFER_ANGLE_DEG)))
#: Cathode-lip blend radius [mm]: R.030 inch, PF-32-103-68 R1 section B-B, tangent to the cone and
#: to the flat nose face (plane s = A_CHI_MM, perpendicular to the axis).
LIP_RADIUS_MM = 0.030 * INCH_MM
#: Lip tangent length from the theoretical corner [mm]: R tan(half the 30 deg turn).
_LIP_TANGENT_MM = float(LIP_RADIUS_MM * np.tan(0.5 * np.deg2rad(CHAMFER_ANGLE_DEG)))
#: Lip tangency on the cone (s [mm], R [mm]) and radius of the tangency on the nose face [mm].
S_LIP_CONE_MM = float(A_CHI_MM - _LIP_TANGENT_MM * np.cos(np.deg2rad(CONE_ANGLE_TO_AXIS_DEG)))
R_LIP_CONE_MM = float(R_CONE_MOUTH_MM - _LIP_TANGENT_MM * np.sin(np.deg2rad(CONE_ANGLE_TO_AXIS_DEG)))
R_LIP_FACE_MM = float(R_CONE_MOUTH_MM + _LIP_TANGENT_MM)
#: Exit rounding radius [mm]: R.122 inch = 3.0988, PF-32-103-45 R0 section A-A.
RHO_MM = 3.099
#: Start of pipe 2, measured from the start of the chamfer [mm].
L_MM = 28.854
#: End of pipe 2
L_END_MM = 40.589
#: Axial length of pipe 2 [mm].
L_PIPE2_MM = L_END_MM - L_MM

#: Default cathode-insertion offset -- cathode exactly at the start of the chamfer.
DEFAULT_DELTA_CATHODE_CHAMFER_MM = 0.0


def aperture_radius_profile_mm(z_mm: np.ndarray, delta_mm: float) -> np.ndarray:
    """Piecewise-analytic R(z) [mm] of the cavity's real transverse channel.

    `s = z_mm + delta_mm`:
      s <= 0                     -> R1_MM (still inside pipe 1)
      0 < s < S_LIP_CONE_MM      -> R1_MM + s*tan(60deg)   (cathode cone)
      S_LIP_CONE_MM <= s < A     -> R_c - sqrt(LIP_RADIUS^2 - (s - s_c)^2)   (R.030 lip,
                                    centre s_c = A - LIP_RADIUS, R_c = R_LIP_FACE_MM)
      A <= s < L-RHO             -> R_CAV_MM               (nose face, main cavity body)
      L-RHO <= s < L             -> R2_MM + RHO - sqrt(RHO^2 - (L-s)^2)   (rounded exit transition)
      s >= L                     -> R2_MM                  (pipe 2)
    with A = A_CHI_MM, the theoretical sharp corner of the cone mouth.
    """
    z_mm = np.asarray(z_mm, dtype=float)
    s = z_mm + float(delta_mm)
    slope = np.tan(np.deg2rad(CONE_ANGLE_TO_AXIS_DEG))

    R = np.full_like(s, R_CAV_MM, dtype=float)
    R = np.where(s <= 0.0, R1_MM, R)
    cone = (s > 0.0) & (s < S_LIP_CONE_MM)
    R = np.where(cone, R1_MM + s * slope, R)
    lip = (s >= S_LIP_CONE_MM) & (s < A_CHI_MM)
    ds_lip = s - (A_CHI_MM - LIP_RADIUS_MM)
    R = np.where(lip, R_LIP_FACE_MM - np.sqrt(np.maximum(LIP_RADIUS_MM**2 - ds_lip**2, 0.0)), R)
    round_zone = (s >= (L_MM - RHO_MM)) & (s < L_MM)
    dz = np.clip(L_MM - s, -RHO_MM, RHO_MM)
    R = np.where(round_zone, R2_MM + RHO_MM - np.sqrt(np.maximum(RHO_MM**2 - dz**2, 0.0)), R)
    R = np.where(s >= L_MM, R2_MM, R)
    return R


def exit_tube_end_z_m(delta_mm: float = DEFAULT_DELTA_CATHODE_CHAMFER_MM) -> float:
    """Cathode-referenced z [m] of the end of the last narrow tube section (`L_END_MM`).

    The tracking domain's natural `z_max`: the aperture profile is defined in `s = z + delta_mm`
    (see `aperture_radius_profile_mm`), so the tube end sits at `z = L_END_MM - delta_mm`. A
    recessed cathode (`delta_mm < 0`) pushes it further downstream in z, a cathode already inside
    the chamfer (`delta_mm > 0`) pulls it back, which is why this is derived rather than a bare
    constant.
    """
    return (L_END_MM - float(delta_mm)) * 1.0e-3


def build_dynamic_aperture(rft, z_grid_m: np.ndarray, delta_mm: float):
    """Build an `Aperture_1d` element spanning `z_grid_m`, for `V.add(A, 0.0, 0.0, z_grid_m[0],
    "entrance")` in the same Volume as the RF field map, at the field map's own placement offset
    (see module docstring gotcha #1).

    `z_grid_m` must be the same uniform grid used for the field map's `Er_grid`/`Ez_grid`, with at
    least 2 points.
    """
    z_grid_m = np.asarray(z_grid_m, dtype=float).reshape(-1)
    if z_grid_m.size < 2:
        raise ValueError("build_dynamic_aperture: z_grid_m needs at least 2 points")

    hz_m = float(z_grid_m[1] - z_grid_m[0])
    # A non-uniform z_grid_m would silently misalign R(z) against the field rather than raising.
    steps = np.diff(z_grid_m)
    if not np.allclose(steps, hz_m, rtol=1e-6, atol=1e-12):
        raise ValueError(
            "build_dynamic_aperture: z_grid_m must be uniformly spaced "
            f"(step varies from {float(np.min(steps)):.6g} to {float(np.max(steps)):.6g} m, "
            f"expected {hz_m:.6g} m)"
        )
    length_m = float(z_grid_m[-1] - z_grid_m[0])

    R_mm = aperture_radius_profile_mm(z_grid_m * 1e3, delta_mm)
    R_m = (R_mm * 1e-3).astype(float)

    # Third arg is length, not z0 (gotcha #1); caller places this at the field map's own offset.
    A = rft.Aperture_1d(R_m, hz_m, length_m)
    A.set_z1(length_m)  # required, else silently inert (gotcha #2)
    return A


#: Backstop thickness [mm]: generous vs. a tracking step (dt_mm<=0.05mm typically) so a backward
#: particle can't skip over the whole band; costs nothing physically since z<0 carries no field.
DEFAULT_CATHODE_BACKSTOP_THICKNESS_MM = 2.0

#: Backstop absorbing radius [m]: small vs. this project's ~mm transverse scale, but not exactly
#: 0 (untested degenerate r<=0 comparison).
DEFAULT_CATHODE_BACKSTOP_RADIUS_M = 1.0e-9


def build_cathode_backstop(
    rft,
    thickness_mm: float = DEFAULT_CATHODE_BACKSTOP_THICKNESS_MM,
    r_absorb_m: float = DEFAULT_CATHODE_BACKSTOP_RADIUS_M,
):
    """Thin absorbing `Aperture_1d` spanning `thickness_mm` immediately behind z=0, for
    `V.add(A, 0.0, 0.0, z0_global - thickness_mm*1e-3, "entrance")` at the same `z0_global` offset
    as the field map/dynamic aperture (global span `[z0_global - thickness_mm*1e-3, z0_global]`).

    Gives an exact z=0-crossing event for back-bombarding electrons via RF-Track's own
    `V.get_lost_particles()`, instead of `rf_gun.back_bombardment`'s field-free-drift
    extrapolation from `Bout` -- weakest for particles that turn around immediately, where
    RF/image fields vary fastest.

    Verified against a synthetic field (ending at z=0, like every real field map here) with
    backward-crossing test particles at a range of speeds/positions -- all correctly absorbed.

    Also verified against this project's real field map via a paired production-scale A/B run
    (identical seed/config, `--cathode_backstop_enabled` vs. `--no-cathode_backstop_enabled`,
    N=1000, coarse finesse): both runs agree on who turns around at all (`n_forward`=53 vs. 54 of
    1000 -- the 1-particle difference is the expected second-order space-charge feedback from
    absorbing returning charge earlier rather than a backstop defect), both close their own
    charge/count accounting exactly, and the backstop-off run's legacy Bout-drift reconstruction
    recovers a real but substantially *smaller* back-bombardment population (158 of 1000) than the
    backstop's real-time interception (311 of 1000) -- expected, not a discrepancy: a particle
    that turns around and is later pushed forward again before the tracking window ends is missed
    entirely by a method that only looks at `Bout`'s final state, which is exactly the failure
    mode this backstop was built to fix (see this function's own first paragraph). No NaN states,
    no ID mismatches, and no particles observed to "leak" past the absorbing element were found in
    either run.

    `V.get_lost_particles()` returns one combined table for every aperture-bearing element in the
    Volume (this backstop and the dynamic aperture alike) with no per-row element tag; separating
    backstop losses (back-bombardment) from dynamic-aperture losses (ordinary transverse loss) in
    that table is implemented in `rf_gun.backstop_loss_separation.identify_backstop_loss_candidates`
    (used by `rf_gun.back_bombardment_events.extract_back_bombardment_events`,
    `rf_gun.particle_tags.build_particle_tags`, and `rf_gun.diagnostics.classify_particle_outcomes`
    -- see that module's own docstring for the classification rule and why it isn't a naive
    `Z<=0` cut).
    """
    thickness_m = float(thickness_mm) * 1e-3
    if thickness_m <= 0.0:
        raise ValueError(f"build_cathode_backstop: thickness_mm must be positive, got {thickness_mm!r}")
    R_m = np.array([float(r_absorb_m), float(r_absorb_m)])
    A = rft.Aperture_1d(R_m, thickness_m, thickness_m)
    A.set_z1(thickness_m)
    return A
