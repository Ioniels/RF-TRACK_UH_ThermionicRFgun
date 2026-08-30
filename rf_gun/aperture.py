"""Dynamic radial aperture R(z): the cavity's real transverse channel, enforced by RF-Track
itself during tracking via `RF_Track.Aperture_1d`.

The channel is not a single cylinder: narrow near the cathode (a short conical chamfer feeding
into pipe 1), wide through the main cavity body, narrow again near the exit (a rounded transition
into pipe 2). A single scalar aperture cannot represent this shape, so this module builds a
tabulated R(z) profile (`aperture_radius_profile_mm`) sampled on the same z-grid as the RF field
map, and wraps it into an `Aperture_1d` element added to the same `Volume` as the field map
(`build_dynamic_aperture`), so a particle is removed the instant it crosses the real transverse
channel during tracking.

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

from typing import Dict

import numpy as np

#: Pipe 1 / pipe 2 radius [mm] (= 5.0546/2 -- the exit iris diameter measured for this cavity).
R1_MM = 2.5275
R2_MM = 2.5275
#: Main cavity inner radius [mm] (= 68.029/2).
R_CAV_MM = 34.0145
#: Chamfer slanted length [mm] and angle from the y-axis [deg].
CHAMFER_LEN_MM = 3.734
CHAMFER_ANGLE_DEG = 30.0
#: Axial extent of the chamfer [mm]: CHAMFER_LEN_MM * sin(CHAMFER_ANGLE_DEG).
A_CHI_MM = CHAMFER_LEN_MM * np.sin(np.deg2rad(CHAMFER_ANGLE_DEG))
#: Exit rounding radius [mm].
RHO_MM = 3.099
#: Start of pipe 2, measured from the start of the chamfer [mm].
L_MM = 28.854

#: Default cathode-insertion offset -- cathode exactly at the start of the chamfer.
DEFAULT_DELTA_CATHODE_CHAMFER_MM = 0.0


def aperture_radius_profile_mm(z_mm: np.ndarray, delta_mm: float) -> np.ndarray:
    """Piecewise-analytic R(z) [mm] of the cavity's real transverse channel.

    `s = z_mm + delta_mm`:
      s <= 0                  -> R1_MM (still inside pipe 1)
      0 < s < A_CHI_MM        -> R1_MM + s*cot(30deg)   (chamfer)
      A_CHI_MM <= s < L-RHO   -> R_CAV_MM               (main cavity body)
      L-RHO <= s < L          -> R2_MM + RHO - sqrt(RHO^2 - (L-s)^2)   (rounded exit transition)
      s >= L                  -> R2_MM                  (pipe 2)
    """
    z_mm = np.asarray(z_mm, dtype=float)
    s = z_mm + float(delta_mm)
    cot30 = 1.0 / np.tan(np.deg2rad(CHAMFER_ANGLE_DEG))

    R = np.full_like(s, R_CAV_MM, dtype=float)
    R = np.where(s <= 0.0, R1_MM, R)
    chamfer = (s > 0.0) & (s < A_CHI_MM)
    R = np.where(chamfer, R1_MM + s * cot30, R)
    round_zone = (s >= (L_MM - RHO_MM)) & (s < L_MM)
    dz = np.clip(L_MM - s, -RHO_MM, RHO_MM)
    R = np.where(round_zone, R2_MM + RHO_MM - np.sqrt(np.maximum(RHO_MM**2 - dz**2, 0.0)), R)
    R = np.where(s >= L_MM, R2_MM, R)
    return R


def important_locations_mm(delta_mm: float) -> Dict[str, float]:
    """z (mm, cathode-referenced) of each geometric transition, for plot annotations."""
    delta_mm = float(delta_mm)
    return {
        "z_ch_start": -delta_mm,
        "z_ch_end": A_CHI_MM - delta_mm,
        "z_round_start": (L_MM - RHO_MM) - delta_mm,
        "z_pipe2_start": L_MM - delta_mm,
    }


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
    Not yet verified against this project's real field map or a production run.

    Caveat: `V.get_lost_particles()` returns one combined table for every aperture-bearing element
    in the Volume (this backstop and the dynamic aperture alike) with no per-row element tag.
    Separating backstop losses (back-bombardment) from dynamic-aperture losses (ordinary transverse
    loss) in that table isn't implemented here -- the table's Z/T semantics for a multi-element
    Volume need their own verification (e.g. cross-referencing IDs against `Bout`'s z<0 tagging)
    first.
    """
    thickness_m = float(thickness_mm) * 1e-3
    if thickness_m <= 0.0:
        raise ValueError(f"build_cathode_backstop: thickness_mm must be positive, got {thickness_mm!r}")
    R_m = np.array([float(r_absorb_m), float(r_absorb_m)])
    A = rft.Aperture_1d(R_m, thickness_m, thickness_m)
    A.set_z1(thickness_m)
    return A
