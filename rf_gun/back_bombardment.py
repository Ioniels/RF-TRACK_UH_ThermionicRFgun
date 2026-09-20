"""Back-bombardment reconstruction: where and when do backward-turning particles re-hit the
cathode plane (z=0)?

The tracked field map only covers z>=0, so a particle that crosses back through z=0 feels no
further force and drifts in a straight line for the rest of the tracked time. Momentum (and
hence Px/Py/Pz/E/K) is therefore unchanged between the z=0 crossing and `Bout`; only (x, y) and
arrival time keep drifting. That lets the crossing be reconstructed analytically from `Bout` alone:

    x_hit = x_final - (Px/Pz) * z_final
    y_hit = y_final - (Py/Pz) * z_final
    t_hit = t_final - z_final / beta_z          (beta_z = Pz/E)

Cheaper than a Screen behind the cathode, which would keep backward particles under active
integration (~200x slower at production particle counts).

Caveat: assumes nothing else pushes the particle once behind the cathode -- space charge's
residual self-field there is not corrected for.
"""
from __future__ import annotations


import numpy as np


#: Column indices within `EXTENDED_PHASE_FMT` ("%X %Px %Y %Py %Z %Pz %id %t %E %K"), matching
#: `rf_gun.particle_tags`'s convention.
_T_COL = 7
_E_COL = 8
_K_COL = 9

#: Impact-surface classification: the cathode is a 3.2mm-diameter disk with a 45deg x 0.2mm
#: chamfer on its outer edge, leaving a 2.8mm-diameter flat emitting face (`cathode_radius_mm`).
#: Impacts on the face or the chamfer both deposit heat into the cathode; anything further out
#: hits the (unmodeled) holder or cavity wall and is excluded from heating accounting entirely.
SURFACE_CATHODE_FACE = "cathode_face"
SURFACE_CATHODE_CHAMFER = "cathode_chamfer"
SURFACE_EXCLUDED = "excluded"

#: Chamfer radial width [mm]: 3.2mm full diameter vs. 2.8mm emitting face -> 0.2mm annulus.
DEFAULT_CATHODE_CHAMFER_WIDTH_MM = 0.2


def classify_impact_surface(
    r_hit_mm: np.ndarray,
    cathode_radius_mm: float,
    chamfer_width_mm: float = DEFAULT_CATHODE_CHAMFER_WIDTH_MM,
) -> np.ndarray:
    """Per-row surface classification from reconstructed radial impact position `r_hit_mm`.
    Non-finite values classify as `SURFACE_EXCLUDED`."""
    r = np.asarray(r_hit_mm, dtype=float)
    cathode_radius_mm = float(cathode_radius_mm)
    chamfer_outer_mm = cathode_radius_mm + float(chamfer_width_mm)
    surface_id = np.full(r.shape, SURFACE_EXCLUDED, dtype="<U16")
    finite = np.isfinite(r)
    surface_id[finite & (r <= cathode_radius_mm)] = SURFACE_CATHODE_FACE
    surface_id[finite & (r > cathode_radius_mm) & (r <= chamfer_outer_mm)] = SURFACE_CATHODE_CHAMFER
    return surface_id
