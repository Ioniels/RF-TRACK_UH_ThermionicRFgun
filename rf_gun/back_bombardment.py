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

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .constants import c, q_e
from .particle_tags import ID_COL, MAX_PHYSICAL_KINETIC_ENERGY_MEV

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


@dataclass(frozen=True)
class BackBombardmentData:
    """Per-particle reconstructed state at z=0, for every particle behind the cathode (z<0) at
    `Bout`. All arrays share one length; `valid` marks physically plausible reconstructions --
    index with it before using `x_hit_mm`/`y_hit_mm`/`t_hit_s`.

    `n_screens_reached`/`last_screen_z_mm`: how far into z>0 each particle got before turning
    around, from screen-crossing presence (see `_screen_reach`); populated regardless of `valid`.

    `ids`/`z_bout_mm`: the particle's `%id` and trusted `Bout` z, for `screen_trajectory` lookups
    and plotting.

    `surface_id`/`heating_relevant` (see `classify_impact_surface`): `heating_relevant = valid &
    (surface_id != SURFACE_EXCLUDED)` is the population that actually heats the cathode -- a
    plausible reconstruction landing on the holder/cavity wall is excluded from heating entirely,
    not just dropped from a plot.
    """

    x_hit_mm: np.ndarray
    y_hit_mm: np.ndarray
    t_hit_s: np.ndarray
    E_total_MeV: np.ndarray
    K_MeV: np.ndarray
    px_MeVc: np.ndarray
    py_MeVc: np.ndarray
    pz_MeVc: np.ndarray
    z_bout_mm: np.ndarray
    ids: np.ndarray
    n_screens_reached: np.ndarray
    last_screen_z_mm: np.ndarray
    valid: np.ndarray
    surface_id: np.ndarray
    heating_relevant: np.ndarray
    weight_per_macroparticle: float
    n_behind_cathode: int
    n_valid: int
    n_cathode_face: int
    n_cathode_chamfer: int
    n_excluded_geometry: int


def compute_back_bombardment(
    Bout_M: np.ndarray,
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    *,
    q_total_C: float,
    n_macroparticles: int,
    r_max_mm: float,
    cathode_radius_mm: float,
    cathode_chamfer_width_mm: float = DEFAULT_CATHODE_CHAMFER_WIDTH_MM,
    id_col: int = ID_COL,
    max_kinetic_energy_mev: Optional[float] = MAX_PHYSICAL_KINETIC_ENERGY_MEV,
) -> BackBombardmentData:
    """Reconstruct (x, y, t) at the z=0 crossing for every particle behind the cathode at Bout.

    `cathode_radius_mm`/`cathode_chamfer_width_mm`: surface classification, see
    `classify_impact_surface`.

    `q_total_C`/`n_macroparticles` give each macroparticle's real-electron weight
    (`|q_total_C| / q_e / n_macroparticles`), to convert macroparticle energy into real deposited
    energy (see `rf_gun.plotting.back_bombardment`).

    `r_max_mm` (the field map's transverse extent, mm) bounds `valid`: near-zero-Pz stragglers
    imply an enormous drift time and blow up the ballistic (x, y) reconstruction past where the
    simulated fields even exist.

    `max_kinetic_energy_mev` additionally excludes non-finite/excessive `%K` (a particle that
    grazed a field singularity can still reconstruct a plausible-looking x/y/t despite an
    astronomical K) -- see `rf_gun.particle_tags.MAX_PHYSICAL_KINETIC_ENERGY_MEV`. `None` disables it.

    `M_snaps`/`z_snaps`: screen snapshots used to trace how far forward each particle got before
    turning around -- see `_screen_reach`.
    """
    arr = np.asarray(Bout_M, dtype=float)
    z_all_mm = arr[:, 4] if arr.ndim == 2 and arr.shape[1] > 4 else np.zeros((0,))
    is_behind = np.isfinite(z_all_mm) & (z_all_mm < 0.0)
    M = arr[is_behind]

    n = int(M.shape[0])
    if n == 0:
        empty = np.zeros((0,), dtype=float)
        return BackBombardmentData(
            x_hit_mm=empty,
            y_hit_mm=empty,
            t_hit_s=empty,
            E_total_MeV=empty,
            K_MeV=empty,
            px_MeVc=empty,
            py_MeVc=empty,
            pz_MeVc=empty,
            z_bout_mm=empty,
            ids=np.zeros((0,), dtype=np.int64),
            n_screens_reached=np.zeros((0,), dtype=int),
            last_screen_z_mm=empty,
            valid=np.zeros((0,), dtype=bool),
            surface_id=np.zeros((0,), dtype="<U16"),
            heating_relevant=np.zeros((0,), dtype=bool),
            weight_per_macroparticle=_weight_per_macroparticle(q_total_C, n_macroparticles),
            n_behind_cathode=0,
            n_valid=0,
            n_cathode_face=0,
            n_cathode_chamfer=0,
            n_excluded_geometry=0,
        )

    x, px = M[:, 0], M[:, 1]
    y, py = M[:, 2], M[:, 3]
    z, pz = M[:, 4], M[:, 5]
    t_mm_c = M[:, _T_COL] if M.shape[1] > _T_COL else np.full(n, np.nan)
    E = M[:, _E_COL] if M.shape[1] > _E_COL else np.full(n, np.nan)
    K = M[:, _K_COL] if M.shape[1] > _K_COL else np.full(n, np.nan)
    ids = M[:, id_col].astype(np.int64) if M.shape[1] > id_col else np.full(n, -1, dtype=np.int64)

    valid_pz = np.isfinite(pz) & (pz != 0.0)
    safe_pz = np.where(valid_pz, pz, 1.0)
    x_hit = np.where(valid_pz, x - (px / safe_pz) * z, np.nan)
    y_hit = np.where(valid_pz, y - (py / safe_pz) * z, np.nan)

    beta_z = np.where(valid_pz & np.isfinite(E) & (E != 0.0), pz / np.where(E != 0.0, E, 1.0), np.nan)
    safe_beta_z = np.where(np.isfinite(beta_z) & (beta_z != 0.0), beta_z, 1.0)
    have_time = valid_pz & np.isfinite(t_mm_c) & np.isfinite(beta_z) & (beta_z != 0.0)
    t_hit_mm_c = np.where(have_time, t_mm_c - z / safe_beta_z, np.nan)
    t_hit_s = t_hit_mm_c * 1e-3 / c

    valid = (
        valid_pz
        & np.isfinite(x_hit) & np.isfinite(y_hit)
        & (np.abs(x_hit) <= r_max_mm) & (np.abs(y_hit) <= r_max_mm)
    )
    if max_kinetic_energy_mev is not None:
        valid = valid & np.isfinite(K) & (np.abs(K) <= float(max_kinetic_energy_mev))

    n_screens_reached, last_screen_z_mm = _screen_reach(ids, M_snaps, z_snaps, id_col=id_col)

    r_hit_mm = np.where(valid, np.hypot(x_hit, y_hit), np.nan)
    surface_id = classify_impact_surface(r_hit_mm, cathode_radius_mm, cathode_chamfer_width_mm)
    heating_relevant = valid & (surface_id != SURFACE_EXCLUDED)

    return BackBombardmentData(
        x_hit_mm=x_hit,
        y_hit_mm=y_hit,
        t_hit_s=t_hit_s,
        E_total_MeV=E,
        K_MeV=K,
        px_MeVc=px,
        py_MeVc=py,
        pz_MeVc=pz,
        z_bout_mm=z,
        ids=ids,
        n_screens_reached=n_screens_reached,
        last_screen_z_mm=last_screen_z_mm,
        valid=valid,
        surface_id=surface_id,
        heating_relevant=heating_relevant,
        weight_per_macroparticle=_weight_per_macroparticle(q_total_C, n_macroparticles),
        n_behind_cathode=n,
        n_valid=int(valid.sum()),
        n_cathode_face=int(np.sum(valid & (surface_id == SURFACE_CATHODE_FACE))),
        n_cathode_chamfer=int(np.sum(valid & (surface_id == SURFACE_CATHODE_CHAMFER))),
        n_excluded_geometry=int(np.sum(valid & (surface_id == SURFACE_EXCLUDED))),
    )


def _screen_reach(
    ids: np.ndarray,
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    id_col: int = ID_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """For each id, how many screens recorded it and the furthest (largest-z) one -- by presence
    in a screen's own array, using that screen's known z (not the particle's unreliable per-screen
    `%Z`). NaN/0 means never recorded (an immediate bounce-back)."""
    n = ids.size
    n_screens_reached = np.zeros(n, dtype=int)
    last_screen_z_mm = np.full(n, np.nan, dtype=float)
    if n == 0:
        return n_screens_reached, last_screen_z_mm

    id_to_row = {int(pid): row for row, pid in enumerate(ids)}
    z_mm = np.asarray(z_snaps, dtype=float) * 1e3
    for j, M in enumerate(M_snaps):
        screen = np.asarray(M, dtype=float)
        if screen.ndim != 2 or screen.shape[0] == 0 or screen.shape[1] <= id_col:
            continue
        for pid in screen[:, id_col].astype(np.int64):
            row = id_to_row.get(int(pid))
            if row is None:
                continue
            n_screens_reached[row] += 1
            zj = float(z_mm[j]) if j < z_mm.size else np.nan
            if np.isfinite(zj) and (not np.isfinite(last_screen_z_mm[row]) or zj > last_screen_z_mm[row]):
                last_screen_z_mm[row] = zj
    return n_screens_reached, last_screen_z_mm


def screen_trajectory(
    pid: int,
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    id_col: int = ID_COL,
    pz_col: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """One particle's `(screen z [mm], that screen's recorded Pz [MeV/c])`, sorted by z, over
    every screen where its `%id` was recorded.

    Caveat: a screen's own Pz can carry the wrong sign once a particle has turned around and
    re-crosses backward -- compare against the trusted `Bout` state rather than trusting a
    screen's Pz alone (see `plot_back_bombardment_screen_reach`).
    """
    zs, pzs = [], []
    for j, M in enumerate(M_snaps):
        screen = np.asarray(M, dtype=float)
        if screen.ndim != 2 or screen.shape[0] == 0 or screen.shape[1] <= max(id_col, pz_col):
            continue
        match = np.nonzero(screen[:, id_col].astype(np.int64) == int(pid))[0]
        if match.size:
            zs.append(float(z_snaps[j]) * 1e3)
            pzs.append(float(screen[match[0], pz_col]))
    order = np.argsort(zs)
    return np.asarray(zs, dtype=float)[order], np.asarray(pzs, dtype=float)[order]


def _weight_per_macroparticle(q_total_C: float, n_macroparticles: int) -> float:
    n_macro = max(int(n_macroparticles), 1)
    return abs(float(q_total_C)) / q_e / n_macro


def kinetic_energy_joules(data: BackBombardmentData) -> np.ndarray:
    """Kinetic energy deposited in Joules by the real electrons each row represents (scaled by
    `weight_per_macroparticle`) -- kinetic, not total: only kinetic energy converts to heat.
    Index with `data.valid` before using."""
    return data.K_MeV * 1e6 * q_e * data.weight_per_macroparticle
