"""Dynamic radial aperture R(z): the cavity's real transverse channel, enforced by RF-Track
itself during tracking via `RF_Track.Aperture_1d`.

The channel is not a single cylinder: narrow near the cathode (a short conical chamfer feeding
into pipe 1), wide through the main cavity body, narrow again near the exit (a rounded transition
into pipe 2). A single scalar aperture cannot represent this shape, so this module builds a
tabulated R(z) profile (`aperture_radius_profile_mm`) sampled on the same z-grid as the RF field
map, and wraps it into an `Aperture_1d` element added to the same `Volume` as the field map
(`build_dynamic_aperture`) -- replacing both RF-Track's own scalar `set_aperture` and the previous
post-hoc, Python-side radius cut applied to screen snapshots after the fact.

Critical, empirically-verified `Aperture_1d` gotcha (not documented anywhere, verified against
the installed RF-Track 2.6.3 binary with a real field-map + tracking test): the constructor
`Aperture_1d(R_array, hz, z0)` registers the mesh data correctly (`get_nz()` reports the right
count), but `get_z1()`/`get_length()` remain stuck at 0 unless `.set_z1(...)` (equivalently
`.set_length(...)`) is called *explicitly* after construction -- without it, the element is
silently inert (kills nothing). `build_dynamic_aperture` below does this for you.

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
    """Build an `Aperture_1d` element spanning `z_grid_m`, ready for `V.add(A, 0.0, 0.0, 0.0,
    "entrance")` in the same Volume as the RF field map (same placement convention as the field
    map itself -- both start at the Volume's own local z=0).

    `z_grid_m` must be the same uniform grid used to build the RF field map's `Er_grid`/`Ez_grid`
    (so the aperture and the field share one z-alignment). Requires at least 2 points.
    """
    z_grid_m = np.asarray(z_grid_m, dtype=float).reshape(-1)
    if z_grid_m.size < 2:
        raise ValueError("build_dynamic_aperture: z_grid_m needs at least 2 points")

    hz_m = float(z_grid_m[1] - z_grid_m[0])
    z0_m = float(z_grid_m[0])
    z1_m = float(z_grid_m[-1])

    R_mm = aperture_radius_profile_mm(z_grid_m * 1e3, delta_mm)
    R_m = (R_mm * 1e-3).astype(float)

    A = rft.Aperture_1d(R_m, hz_m, z0_m)
    # Required: without this, Aperture_1d's own z0/z1/length stay at 0 and the element is
    # silently inert (verified empirically -- see module docstring).
    A.set_z1(z1_m)
    return A
