"""Shared particle-identity tagging: who is backward, who was lost by the dynamic aperture.

Computed once per run from `Bout`'s reliable absolute z/pz and RF-Track's own lost-particle
table (`V.get_lost_particles()`, populated automatically whenever the dynamic aperture,
`rf_gun.aperture.build_dynamic_aperture`, removes a particle during tracking), then reused
everywhere else (`compute_beam_properties`, every phase-space plot) via `%id` cross-referencing.

Why `%id`, not a screen's own (z, pz): a `Screen`'s recorded phase space comes from an internal
RF-Track `Bunch6d` object, not `Bunch6dT` -- confirmed empirically (see
`rf_gun.diagnostics.manual_twiss_and_emittance`'s docstring for the full writeup) that this loses
the true lab-frame sign of a backward-crossing particle's `Pz`. `Bout`'s own z/pz, in contrast,
are absolute and reliable, so tagging is done there and propagated to screens by particle
identity (`%id`, confirmed to survive intact through B0 -> Screen -> Bout) rather than trusted
from the screen's own columns.

Unlike the previous post-hoc, entrance/exit-screen-pair radius cut this replaces, a particle
removed by the dynamic aperture is physically gone from the moment it's removed onward: every
screen upstream of that point still records it (it was alive then), and every screen downstream
never records it at all. So, unlike before, no z-gating is needed here -- `lost_ids` is simply
"this id appears in RF-Track's own lost-particle table," true everywhere, always.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Column indices within `EXTENDED_PHASE_FMT` ("%X %Px %Y %Py %Z %Pz %id %t %E %K"), shared by
#: `beam_properties.py` and `plotting/phase_space.py` so the column layout is defined once.
ID_COL = 6
T_COL = 7
E_COL = 8
K_COL = 9

#: Column index of `%id` within RF-Track's own lost-particle table (`LOST_COLUMNS` in
#: `rf_gun.io`: x, px, y, py, z, pz, t, mass, q, N, id) -- verified empirically against the
#: installed RF-Track binary (a synthetic bunch with known per-row t0 values showed up in the
#: expected t-column position, id last).
LOST_TABLE_ID_COL = -1


@dataclass(frozen=True)
class ParticleTags:
    backward_ids: frozenset
    lost_ids: frozenset  # empty frozenset when nothing was removed by the dynamic aperture


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


def lost_ids_from_lost_table(lost_table: Optional[np.ndarray], id_col: int = LOST_TABLE_ID_COL) -> frozenset:
    """IDs of particles removed by the dynamic aperture during tracking, from RF-Track's own
    `V.get_lost_particles()` table (already normalized to an (n, 11) array by
    `rf_gun.diagnostics.to_lost_table_array` before it reaches here)."""
    if lost_table is None:
        return frozenset()
    arr = np.asarray(lost_table, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < abs(id_col):
        return frozenset()
    return frozenset(arr[:, id_col].astype(np.int64).tolist())


def build_particle_tags(
    Bout_M: np.ndarray,
    lost_table: Optional[np.ndarray],
    id_col: int = ID_COL,
    threshold_backward_mevc: float = 0.0,
    extra_backward_ids: Optional[frozenset] = None,
) -> ParticleTags:
    """Build the project-wide forward/backward/lost tagging, once per run.

    `lost_ids` comes straight from RF-Track's own lost-particle table (see
    `lost_ids_from_lost_table`) -- there is no longer a radius cut or entrance/exit screen pair to
    reconcile here, since the dynamic aperture already removed those particles during tracking.

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
    lost_ids = lost_ids_from_lost_table(lost_table)
    return ParticleTags(backward_ids=backward_ids, lost_ids=lost_ids)


def tag_mask(
    M: np.ndarray,
    tags: ParticleTags,
    id_col: int = ID_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns `(is_backward, is_lost)` boolean masks for `M`'s rows, via `%id` lookup.

    `is_lost` is always `False` where `is_backward` is `True` (a particle is tagged into at most
    one category) -- in practice the two sets are already disjoint (a particle removed by the
    dynamic aperture never reaches `Bout` to be classified backward), this just guards against
    an id colliding across both sets by construction error.
    """
    arr = np.asarray(M, dtype=float)
    n = arr.shape[0] if arr.ndim == 2 else 0
    if n == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)
    ids = _ids_of(arr, id_col)
    is_backward = np.isin(ids, list(tags.backward_ids)) if tags.backward_ids else np.zeros(n, dtype=bool)
    is_lost = np.isin(ids, list(tags.lost_ids)) if tags.lost_ids else np.zeros(n, dtype=bool)
    is_lost = is_lost & ~is_backward
    return is_backward, is_lost


def surviving_mask(
    M: np.ndarray,
    tags: ParticleTags,
    id_col: int = ID_COL,
) -> np.ndarray:
    """Boolean mask of rows that are neither backward nor lost -- the forward-transmitted
    population used for `compute_beam_properties`."""
    is_backward, is_lost = tag_mask(M, tags, id_col)
    return ~is_backward & ~is_lost
