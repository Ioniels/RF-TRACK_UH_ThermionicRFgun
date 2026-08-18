"""Post-hoc geometric aperture cut, applied to entrance/exit screen snapshots.

RF-Track's per-element aperture check acts on the whole Volume, not just the
element's own declared z-span (empirically verified against RF-Track 2.5.4), so a
physically localized aperture cannot be modeled with a single restrictive
element inside the existing continuous Volume. Instead, the beam is tracked
through the full domain as usual, screens are placed at the aperture entrance
and exit, and the radial cut r = sqrt(x^2+y^2) > radius is applied afterwards to
those two snapshots.

Because of the thermionic emission's phase/timing spread and the RF field near
the cathode, a substantial fraction of emitted particles never reach the
aperture at all: they are lost, or turn around and head back toward (or past)
the cathode, well before z reaches the aperture entrance. This is why the
aperture cut is only ever applied to screen snapshots rather than to the whole
initial bunch: a `Screen` element only ever records a particle at the moment it
actually crosses that screen's z-plane, so a particle that never reaches the
aperture entrance simply never appears in `M_entrance` (or `M_exit`) at all --
"N" below is already restricted to particles that got at least that far, and
"transmission_from_initial" makes that attrition explicit rather than hiding it
behind an aperture-only transmission number.

Note on an RF-Track quirk that shaped this module: a Screen's phase-space
`%Z` column is *not* a position at all. A screen's snapshot is an internal
`Bunch6d`, and per the RF-Track reference manual (Table 2.1) `%z` there is
"longitudinal coordinate w.r.t. the reference particle" -- confirmed
empirically (isolated test against this project's installed RF-Track binary)
to be the closed form `Z = z_screen_nominal_mm * (Vz_particle/c / Vz_ref/c -
1)`, i.e. each crossing particle's own velocity times its time offset from
whichever particle is currently the bunch's *reference* particle (row/id 0 by
default -- RF-Track silently substitutes the centroid instead if particle 0
has been lost, with no trace of which mode was active left in the returned
array). This is exactly 0 only for the reference particle itself, and only
small for particles whose velocity is close to the reference particle's --
which is why it reads as "~0 for every particle" for a narrow-velocity-spread
beam, but for this project's typical population (thermionic emission, wide
phase/energy spread, especially near the cathode) it can be large in either
sign for a genuinely slow or fast particle, growing with the screen's own
nominal z, and has no simple relationship to that particle's real lab-frame
position or to whether it is moving forward or backward. Every row RF-Track
puts into a screen's snapshot is still a genuine individual crossing (that
part of the mechanism was also confirmed directly, not just inferred) -- it
is specifically `%Z`'s magnitude and sign that must not be read as "position"
or "forward/backward," here or in any other per-screen phase-space panel.
So screen snapshots cannot be used to re-derive "did this particle really
reach z_start_m" from their own Z column -- that check is already
structurally guaranteed by the screen mechanism itself (see above), not
something this module re-verifies numerically. Absolute-z bookkeeping (e.g.
how many emitted particles even head in the aperture's direction at all)
belongs on `Bout`/`particle_classes`, not here -- see the notebook's aperture
diagnostics cell.

The radial cut is applied independently at the entrance and exit screens (the
screen phase-space format does not carry a particle-id column, so entrance and
exit rows are not joined per-particle). Reporting both bounds the aperture's
effect: if the two transmission fractions are close, particles that clear the
entrance are not developing large orbits inside the channel, so quoting either
one is representative; a large drop between entrance and exit signals particles
clipping the wall partway through and is reported rather than hidden.

Forward-filter-before-radius-cut: a screen records a particle whenever it crosses the screen's
z-plane, regardless of travel direction, so `M_entrance`/`M_exit` can (and, for a thermionic gun
with a wide emission-phase window, do) contain particles that are already turned around --
heading backward or sitting behind the cathode -- at the moment they cross that plane. Since a
screen's own z/pz cannot be trusted to detect this (see the note on `%Z` above), the caller must
apply the project's %id-based forward/backward tagging (`rf_gun.particle_tags.tag_mask`) before
calling `aperture_summary`, not this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .diagnostics import summarize_array


@dataclass(frozen=True)
class ApertureParams:
    enabled: bool = False
    z_start_m: float = 0.0
    z_end_m: float = 0.0
    radius_mm: float = 0.0


def aperture_survival_mask(M: np.ndarray, radius_mm: float) -> np.ndarray:
    """Boolean mask of particles with r = sqrt(x^2+y^2) <= radius_mm.

    Pure radial cut only -- does not check `z`/`pz`. Callers should already have applied the
    project's %id-based forward/backward tagging (`rf_gun.particle_tags.tag_mask`) before this,
    since a screen's own z/pz is not a reliable forward/backward indicator (see that module's
    docstring) -- `aperture_summary` expects its inputs pre-filtered this way.
    """
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return np.zeros((arr.shape[0] if arr.ndim == 2 else 0,), dtype=bool)
    r = np.sqrt(arr[:, 0] ** 2 + arr[:, 2] ** 2)
    return np.isfinite(r) & (r <= float(radius_mm))


def aperture_summary(
    M_entrance: np.ndarray,
    M_exit: np.ndarray,
    params: ApertureParams,
    n_initial: int,
) -> Dict[str, Any]:
    """Before/after aperture statistics from entrance and exit screen snapshots.

    `M_entrance`/`M_exit` are expected already forward-only (see `aperture_survival_mask`'s
    docstring). `N` at each plane counts particles that reached that plane;
    `transmission_from_initial` divides by the original emitted count so the fraction that never
    reaches the aperture at all (lost, or turned back toward the cathode) stays visible instead of
    being absorbed into a higher-looking "aperture transmission" number.
    """
    M_ent = np.asarray(M_entrance, dtype=float) if M_entrance is not None else np.zeros((0, 6))
    M_ext = np.asarray(M_exit, dtype=float) if M_exit is not None else np.zeros((0, 6))

    mask_ent = aperture_survival_mask(M_ent, params.radius_mm)
    mask_ext = aperture_survival_mask(M_ext, params.radius_mm)

    n_entrance = int(M_ent.shape[0])
    n_exit = int(M_ext.shape[0])
    n_clipped_entrance = int(np.sum(~mask_ent))
    n_surviving_entrance = int(np.sum(mask_ent))
    n_clipped_exit = int(np.sum(~mask_ext))
    n_surviving_exit = int(np.sum(mask_ext))

    def _r(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
            return np.asarray([], dtype=float)
        return np.sqrt(arr[:, 0] ** 2 + arr[:, 2] ** 2)

    def _frac(n: int, d: int) -> float | None:
        return float(n / d) if d > 0 else None

    return {
        "params": {
            "enabled": bool(params.enabled),
            "z_start_m": float(params.z_start_m),
            "z_end_m": float(params.z_end_m),
            "radius_mm": float(params.radius_mm),
        },
        "n_initial": int(n_initial),
        "entrance": {
            "N": n_entrance,
            "N_within_radius": n_surviving_entrance,
            "N_clipped": n_clipped_entrance,
            "transmission_from_initial": _frac(n_entrance, n_initial),
            "aperture_transmission": _frac(n_surviving_entrance, n_entrance),
            "radius_mm_summary": summarize_array(_r(M_ent)),
            "radius_mm_summary_within": summarize_array(_r(M_ent)[mask_ent] if mask_ent.size else np.asarray([])),
        },
        "exit": {
            "N": n_exit,
            "N_within_radius": n_surviving_exit,
            "N_clipped": n_clipped_exit,
            "transmission_from_initial": _frac(n_exit, n_initial),
            "aperture_transmission": _frac(n_surviving_exit, n_exit),
            "radius_mm_summary": summarize_array(_r(M_ext)),
            "radius_mm_summary_within": summarize_array(_r(M_ext)[mask_ext] if mask_ext.size else np.asarray([])),
        },
    }
