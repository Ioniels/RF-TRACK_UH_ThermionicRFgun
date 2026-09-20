"""Shared particle-identity tagging: who is backward, who was lost to the dynamic aperture.

Computed once per run from `Bout`'s reliable absolute z/pz and RF-Track's own lost-particle
table (`V.get_lost_particles()`, populated automatically whenever the dynamic aperture,
`rf_gun.aperture.build_dynamic_aperture`, or the cathode backstop,
`rf_gun.aperture.build_cathode_backstop`, removes a particle during tracking), then reused
everywhere else (`compute_beam_properties`, every phase-space plot) via `%id` cross-referencing.

Why `%id`, not a screen's own (z, pz): a `Screen`'s recorded phase space comes from an internal
RF-Track `Bunch6d` object, not `Bunch6dT` -- confirmed empirically (see
`rf_gun.diagnostics.manual_twiss_and_emittance`'s docstring for the full writeup) that this loses
the true lab-frame sign of a backward-crossing particle's `Pz`. `Bout`'s own z/pz, in contrast,
are absolute and reliable, so tagging is done there and propagated to screens by particle
identity (`%id`, confirmed to survive intact through B0 -> Screen -> Bout) rather than trusted
from the screen's own columns.

A particle removed by the dynamic aperture or the backstop is physically gone from the moment
it's removed onward: every screen upstream of that point still records it (it was alive then),
and every screen downstream never records it at all. No z-gating is needed here -- `lost_ids` is
simply "this id appears in RF-Track's own lost-particle table," true everywhere, always.

The backstop and the dynamic aperture populate that same combined table with no per-row element
tag, but they are not the same category: a backstop capture is a returning, back-bombarding
particle -- physically backward, not a transverse aperture loss -- and `build_particle_tags`'s
`backstop_z_min_m` argument reclassifies it into `backward_ids` accordingly (see its docstring).
`lost_ids` alone is therefore the *dynamic-aperture-only* loss count whenever `backstop_z_min_m`
is supplied; use `backward_ids | lost_ids` for a simple forward-vs-everything-else dichotomy.
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

#: Generous upper bound (MeV) on any physically plausible kinetic energy a particle can carry in
#: this cavity: the delivered RF power gives an effective voltage well under 1 MV, so a real,
#: well-behaved transmitted or back-bombarding particle sits at most a few MeV. Set far above that,
#: and far below the values (K ~ 1e290+ MeV) empirically observed for a particle that grazed a
#: field singularity near the cathode chamfer and blew up numerically during ODE integration -- a
#: sanity backstop against that specific artifact, not a physics cut. The dynamic aperture only
#: bounds transverse position, so a particle like this can stay within it (and even reconstruct a
#: perfectly plausible x/y/t at Bout) without ever appearing in `lost_table`; it also isn't caught
#: by `backward_ids_from_bout` since it's still nominally forward (z>=0, pz>0). Shared with
#: `rf_gun.back_bombardment` so both "forward transmission" and "back-bombardment" tagging apply
#: the same physical sanity bound.
MAX_PHYSICAL_KINETIC_ENERGY_MEV = 10.0


@dataclass(frozen=True)
class ParticleTags:
    backward_ids: frozenset
    lost_ids: frozenset  # empty frozenset when nothing was removed by the dynamic aperture
    #: Subset of `backward_ids` whose back-bombardment ray-cast landed on LaB6 (the cathode flat
    #: or its bevel). This is the population that actually heats the cathode, and it is a strict
    #: subset: a backward particle can pass the cathode plane through the open annulus outside the
    #: 3.2 mm LaB6 disk, or be absorbed on the cavity nose, without ever striking LaB6.
    #: Empty when back-bombardment events were not captured for the run.
    cathode_hit_ids: frozenset = frozenset()



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


def unphysical_ids_from_bout(
    Bout_M: np.ndarray,
    id_col: int = ID_COL,
    k_col: int = K_COL,
    max_kinetic_energy_mev: float = MAX_PHYSICAL_KINETIC_ENERGY_MEV,
) -> frozenset:
    """IDs of particles whose `Bout` kinetic energy (`%K`, RF-Track's own column) is non-finite or
    exceeds `max_kinetic_energy_mev` -- see `MAX_PHYSICAL_KINETIC_ENERGY_MEV`'s docstring for why
    this exists and why nothing else in this module catches it."""
    arr = np.asarray(Bout_M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] <= k_col:
        return frozenset()
    k = arr[:, k_col]
    bad = ~np.isfinite(k) | (np.abs(k) > float(max_kinetic_energy_mev))
    ids = _ids_of(arr, id_col)
    return frozenset(ids[bad].tolist())


def backward_ids_from_lost_table(
    lost_table: Optional[np.ndarray], id_col: int = LOST_TABLE_ID_COL, pz_col: int = 5
) -> frozenset:
    """IDs of every lost-table row that was travelling BACKWARD when it was removed (`Pz < 0`).

    This is the physical definition of "went backward", and it is deliberately broader than
    `backstop_ids_from_lost_table`, which additionally requires the row to sit in a narrow z-band
    just behind the cathode. Those are two different questions:

      * which rows do we ray-cast for a cathode impact?  -> the z-band (they are near the cathode)
      * which particles actually turned around?          -> this one (`Pz < 0`, anywhere)

    On the 1650 K / 0 A production run the z-band captures 89,514 rows while 170,993 are genuinely
    backward: 81,479 turned around and were then absorbed on the cavity nose further out, at
    z ~ 1.9 mm, just past the 1.5 mm slack. Counting those as transverse *aperture* losses -- which
    is what happened before -- understates the backward population by nearly half.
    """
    arr = np.asarray(lost_table, dtype=float) if lost_table is not None else np.zeros((0, 0))
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] <= max(pz_col, abs(id_col)):
        return frozenset()
    pz = arr[:, pz_col]
    backward = np.isfinite(pz) & (pz < 0.0)
    return frozenset(arr[backward, id_col].astype(np.int64).tolist())


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


def backstop_ids_from_lost_table(lost_table: Optional[np.ndarray], backstop_z_min_m: float) -> frozenset:
    """IDs of `lost_table` rows the cathode backstop actually captured (a returning, back-
    bombarding particle), as opposed to an ordinary dynamic-aperture (transverse) loss -- both
    populate the same combined RF-Track table with no per-row element tag (see
    `rf_gun.aperture.build_cathode_backstop`'s docstring). Delegates the classification rule
    itself to `rf_gun.backstop_loss_separation.identify_backstop_loss_candidates`; see that
    function's docstring for why it's `Pz<0` and a Z-band, not a naive `Z<=0` cut."""
    if lost_table is None:
        return frozenset()
    arr = np.asarray(lost_table, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return frozenset()
    from .backstop_loss_separation import identify_backstop_loss_candidates

    is_backstop_row = identify_backstop_loss_candidates(arr, backstop_z_min_m=backstop_z_min_m)
    return frozenset(arr[is_backstop_row, LOST_TABLE_ID_COL].astype(np.int64).tolist())


def build_particle_tags(
    Bout_M: np.ndarray,
    lost_table: Optional[np.ndarray],
    id_col: int = ID_COL,
    threshold_backward_mevc: float = 0.0,
    extra_backward_ids: Optional[frozenset] = None,
    max_kinetic_energy_mev: Optional[float] = MAX_PHYSICAL_KINETIC_ENERGY_MEV,
    backstop_z_min_m: Optional[float] = None,
    cathode_hit_ids: Optional[frozenset] = None,
) -> ParticleTags:
    """Build the project-wide forward/backward/lost tagging, once per run.

    `lost_ids` comes straight from RF-Track's own lost-particle table (see
    `lost_ids_from_lost_table`), unioned with `unphysical_ids_from_bout` (any `Bout` particle whose
    kinetic energy is non-finite or exceeds `max_kinetic_energy_mev` -- pass `None` to disable this
    backstop). The dynamic aperture (`rf_gun.aperture`, an `Aperture_1d` element enforced during
    tracking) already removes particles outside the gun's real transverse channel R(z), so no
    separate radius cut is needed here.

    `threshold_backward_mevc` is passed straight to `backward_ids_from_bout` -- see its docstring
    for the stagnant-near-cathode-beamlet rationale.

    `extra_backward_ids`, when given, is unioned into `backward_ids` after the threshold-based
    classification -- this is how `rf_gun.acceptance_scan`'s data-driven trailing-particle removal
    (`AcceptanceScanResult.trailing_ids`) plugs in, superseding `threshold_backward_mevc` as the
    project's mechanism for widening "backward" beyond the strict `z<0 or Pz<=0` definition.

    `backstop_z_min_m`, when given (the cathode backstop's global z span's lower edge, i.e.
    `-thickness_mm*1e-3` in this project's `z0_global=0` convention), reclassifies every
    `lost_table` row the backstop actually captured (see `backstop_ids_from_lost_table`) as
    *backward*, not lost: a particle absorbed by the backstop physically turned around and
    returned toward the cathode -- it is the back-bombardment population, not an ordinary
    transverse aperture loss, and belongs in "backward" everywhere forward/backward/lost is
    reported. Omit (the default, `None`) to keep every `lost_table` row as `lost_ids` unchanged --
    the pre-backstop-aware behavior, kept for callers that have not supplied this yet. Use
    `backward_ids | lost_ids` wherever a simple forward-vs-"everything else" dichotomy (not the
    three-way breakdown) is wanted -- backward particles are still a form of total loss from the
    transmitted beam, just not a *dynamic-aperture* one.
    """
    backward_ids = backward_ids_from_bout(Bout_M, id_col, threshold_backward_mevc=threshold_backward_mevc)
    if extra_backward_ids:
        backward_ids = frozenset(backward_ids | extra_backward_ids)
    lost_ids = lost_ids_from_lost_table(lost_table)
    if backstop_z_min_m is not None:
        # "Backward" is a momentum-sign question, not a position one: every Pz<0 row turned
        # around, whether it was then absorbed on the backstop just behind the cathode or on the
        # cavity nose a millimetre further out. Using the narrow backstop z-band here instead
        # left ~half the backward population miscounted as transverse aperture losses.
        turned_around = backward_ids_from_lost_table(lost_table) & lost_ids
        backward_ids = frozenset(backward_ids | turned_around)
        lost_ids = frozenset(lost_ids - turned_around)
    if max_kinetic_energy_mev is not None:
        lost_ids = frozenset(
            lost_ids | unphysical_ids_from_bout(Bout_M, id_col, max_kinetic_energy_mev=max_kinetic_energy_mev)
        )
    cathode_hits = frozenset(cathode_hit_ids or frozenset()) & backward_ids
    return ParticleTags(
        backward_ids=backward_ids, lost_ids=lost_ids, cathode_hit_ids=cathode_hits
    )


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
