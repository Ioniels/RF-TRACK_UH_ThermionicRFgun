"""Shared particle-identity tagging: who is backward, who survives the aperture.

Computed once per run from `Bout`'s reliable absolute z/pz and the aperture entrance/exit
screens' radius cut, then reused everywhere else (the aperture cell, `compute_beam_properties`,
and every phase-space plot) via `%id` cross-referencing.

Why `%id`, not a screen's own (z, pz): a `Screen`'s recorded phase space comes from an internal
RF-Track `Bunch6d` object, not `Bunch6dT` -- confirmed empirically (see
`rf_gun.diagnostics.manual_twiss_and_emittance`'s docstring for the full writeup) that this loses
the true lab-frame sign of a backward-crossing particle's `Pz`. `Bout`'s own z/pz, in contrast,
are absolute and reliable, so tagging is done there and propagated to screens by particle
identity (`%id`, confirmed to survive intact through B0 -> Screen -> Bout) rather than trusted
from the screen's own columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .aperture import aperture_survival_mask

#: Column indices within `EXTENDED_PHASE_FMT` ("%X %Px %Y %Py %Z %Pz %id %t %E %K"), shared by
#: `beam_properties.py` and `plotting/phase_space.py` so the column layout is defined once.
ID_COL = 6
T_COL = 7
E_COL = 8
K_COL = 9


@dataclass(frozen=True)
class ParticleTags:
    backward_ids: frozenset
    aperture_lost_ids: frozenset  # empty frozenset when the aperture is disabled
    #: z (meters, absolute lab frame, matching `z_snaps`) of the aperture's entrance -- `None`
    #: when the aperture is disabled/unknown. Lets `tag_mask`/`surviving_mask` gate aperture-loss
    #: tagging by *where* a screen physically is: `aperture_lost_ids` is computed once from the
    #: two aperture screens (necessarily downstream), but is a property of particle identity, not
    #: of z -- applying it unconditionally to a screen far upstream of the aperture would tag a
    #: particle "aperture-lost" before the aperture even exists yet. See `tag_mask`'s docstring.
    aperture_z_start_m: Optional[float] = None


def _ids_of(M: np.ndarray, id_col: int = ID_COL) -> np.ndarray:
    """Row ids, or an all-`-1` sentinel of the correct length if `M` has no id column.

    The sentinel (rather than a zero-length array) matters: callers rely on the returned array
    being exactly `M.shape[0]` long so downstream boolean masks stay correctly shaped even when
    tagging is unavailable (no row's id, including `-1`, is ever a real particle id).
    """
    arr = np.asarray(M, dtype=float)
    n = arr.shape[0] if arr.ndim == 2 else 0
    if arr.ndim != 2 or n == 0:
        return np.zeros((0,), dtype=np.int64)
    if arr.shape[1] <= id_col:
        return np.full((n,), -1, dtype=np.int64)
    return arr[:, id_col].astype(np.int64)


def backward_ids_from_bout(
    Bout_M: np.ndarray,
    id_col: int = ID_COL,
    threshold_backward_mevc: float = 0.0,
) -> frozenset:
    """IDs of particles `Bout` classifies as backward (not: z>=0 and pz>threshold_backward_mevc).

    `Bout`'s own z/pz are absolute lab-frame and reliable (unlike a Screen's), so this is the one
    place z/pz are trusted directly for tagging; every other snapshot is tagged by id lookup
    against this set (see `tag_mask`).

    `threshold_backward_mevc` (default 0.0, the strict `pz>0` cut) can also catch a stagnant
    near-cathode beamlet that is nominally forward but too slow to ever join the transmitted
    beam, e.g. `threshold_backward_mevc=0.025` MeV/c. This is a different population from
    `rf_gun.back_bombardment` (particles that actually cross to z<0): a stagnant particle may
    never cross backward at all, so `compute_back_bombardment` is unaffected by this threshold.

    No other implicit filtering is applied here (e.g. no energy/radius plausibility cutoff) --
    any further narrowing of "backward" should stay visible and data-driven, which is what
    `rf_gun.acceptance_scan`'s trailing-particle removal (`extra_backward_ids` below) is for.
    """
    arr = np.asarray(Bout_M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return frozenset()
    good = (
        np.isfinite(arr[:, 4]) & np.isfinite(arr[:, 5])
        & (arr[:, 4] >= 0.0) & (arr[:, 5] > float(threshold_backward_mevc))
    )
    ids = _ids_of(arr, id_col)
    return frozenset(ids[~good].tolist())


def build_particle_tags(
    Bout_M: np.ndarray,
    M_ap_entrance: Optional[np.ndarray],
    M_ap_exit: Optional[np.ndarray],
    aperture_radius_mm: float,
    aperture_enabled: bool,
    id_col: int = ID_COL,
    aperture_z_start_m: Optional[float] = None,
    threshold_backward_mevc: float = 0.0,
    extra_backward_ids: Optional[frozenset] = None,
) -> ParticleTags:
    """Build the project-wide forward/aperture-survival tagging, once per run.

    A particle counts as "surviving the aperture" only if it is forward-going (not in
    `backward_ids`) AND within `aperture_radius_mm` at *both* the entrance and exit screens --
    matching `rf_gun.aperture`'s existing "must clear the radius at both planes" convention.
    `aperture_lost_ids` is the complement: forward particles that reached at least one aperture
    screen but did not make it into the surviving set, well-defined (via id) at *every* screen,
    not just the two aperture screens themselves.

    `aperture_z_start_m`, when given, is stored on the returned `ParticleTags` so `tag_mask`/
    `surviving_mask` can gate aperture-loss tagging to screens at or after that z -- see their
    docstrings for why (a particle's "aperture-lost" fate is otherwise applied retroactively to
    screens upstream of the aperture, which do not physically constrain it yet).

    `threshold_backward_mevc` is passed straight to `backward_ids_from_bout` -- see its docstring
    for the stagnant-near-cathode-beamlet rationale.

    `extra_backward_ids`, when given, is unioned into `backward_ids` after the threshold-based
    classification -- this is how `rf_gun.acceptance_scan`'s data-driven trailing-particle removal
    (`AcceptanceScanResult.trailing_ids`) plugs in, superseding `threshold_backward_mevc` as the
    project's mechanism for widening "backward" beyond the strict `z<0 or Pz<=0` definition.
    """
    backward_ids = backward_ids_from_bout(Bout_M, id_col, threshold_backward_mevc=threshold_backward_mevc)
    if extra_backward_ids:
        backward_ids = frozenset(backward_ids | extra_backward_ids)
    if not aperture_enabled or M_ap_entrance is None or M_ap_exit is None:
        return ParticleTags(backward_ids=backward_ids, aperture_lost_ids=frozenset(), aperture_z_start_m=None)

    ent = np.asarray(M_ap_entrance, dtype=float)
    ext = np.asarray(M_ap_exit, dtype=float)

    def _forward_ids(M: np.ndarray) -> set:
        ids = _ids_of(M, id_col)
        if ids.size == 0:
            return set()
        is_backward = np.isin(ids, list(backward_ids)) if backward_ids else np.zeros(ids.shape[0], dtype=bool)
        return set(ids[~is_backward].tolist())

    def _within_radius_ids(M: np.ndarray) -> set:
        if M.shape[0] == 0:
            return set()
        mask = aperture_survival_mask(M, aperture_radius_mm)
        return set(_ids_of(M[mask], id_col).tolist())

    forward_ent = _forward_ids(ent)
    forward_ext = _forward_ids(ext)
    surviving_ids = (forward_ent & _within_radius_ids(ent)) & (forward_ext & _within_radius_ids(ext))
    reached_forward_ids = forward_ent | forward_ext
    aperture_lost_ids = frozenset(reached_forward_ids - surviving_ids)
    return ParticleTags(
        backward_ids=backward_ids,
        aperture_lost_ids=aperture_lost_ids,
        aperture_z_start_m=float(aperture_z_start_m) if aperture_z_start_m is not None else None,
    )


def tag_mask(
    M: np.ndarray,
    tags: ParticleTags,
    id_col: int = ID_COL,
    screen_z_m: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns `(is_backward, is_aperture_lost)` boolean masks for `M`'s rows, via `%id` lookup.

    `is_aperture_lost` is always `False` where `is_backward` is `True` (a particle is tagged into
    at most one category).

    `screen_z_m`, when given together with `tags.aperture_z_start_m`, gates aperture-loss tagging
    to screens at or after the aperture's entrance: `aperture_lost_ids` is computed once from the
    (necessarily downstream) aperture entrance/exit screens and is otherwise applied to *every*
    snapshot regardless of z -- correct for "this particle's eventual fate," but wrong for "is this
    particle affected by the aperture *here*" at a screen upstream of the aperture, which the
    aperture cannot possibly have clipped yet (confirmed empirically: applying it unconditionally
    produces a spurious survivorship-bias artifact in per-screen statistics, e.g. a non-monotonic
    mean transverse position vs z that vanishes once this gating is applied). Omit `screen_z_m`
    (or leave `tags.aperture_z_start_m` unset) to keep the unconditional (pre-fix) behavior.
    """
    arr = np.asarray(M, dtype=float)
    n = arr.shape[0] if arr.ndim == 2 else 0
    if n == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)
    ids = _ids_of(arr, id_col)
    is_backward = np.isin(ids, list(tags.backward_ids)) if tags.backward_ids else np.zeros(n, dtype=bool)
    is_aperture_lost = np.isin(ids, list(tags.aperture_lost_ids)) if tags.aperture_lost_ids else np.zeros(n, dtype=bool)
    if screen_z_m is not None and tags.aperture_z_start_m is not None and float(screen_z_m) < tags.aperture_z_start_m:
        is_aperture_lost = np.zeros(n, dtype=bool)
    is_aperture_lost = is_aperture_lost & ~is_backward
    return is_backward, is_aperture_lost


def surviving_mask(
    M: np.ndarray,
    tags: ParticleTags,
    id_col: int = ID_COL,
    screen_z_m: Optional[float] = None,
) -> np.ndarray:
    """Boolean mask of rows that are neither backward nor aperture-lost -- the population used
    for `compute_beam_properties` ("forward-going AND aperture-surviving"). See `tag_mask` for
    `screen_z_m`."""
    is_backward, is_aperture_lost = tag_mask(M, tags, id_col, screen_z_m=screen_z_m)
    return ~is_backward & ~is_aperture_lost
