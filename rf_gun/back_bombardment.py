"""Back-bombardment reconstruction: where and when do backward-turning particles re-hit the
cathode plane (z=0)?

The tracked field map is only defined for z>=0 (confirmed via the field-map grid construction and
`Volume` setup in `rf_gun/rftrack_volume.py`), so a particle that crosses back through z=0 feels no
further force and drifts in a perfectly straight line for whatever remains of the tracked time
(confirmed empirically: a real production run's `Bout` showed backward particles' z spread over
hundreds of mm, consistent with unbounded free drift, not a field-bounded oscillation). A
field-free straight line conserves momentum (and hence total/kinetic energy) exactly, so a
particle's Px/Py/Pz/E/K at `Bout` are identical to their values the instant it actually crossed
z=0 -- only its (x, y) position and its arrival time keep drifting after that. That means (x, y, t)
at the z=0 crossing can be reconstructed analytically from `Bout`'s already-tracked data alone:

    x_hit = x_final - (Px/Pz) * z_final
    y_hit = y_final - (Py/Pz) * z_final
    t_hit = t_final - z_final / beta_z          (beta_z = Pz/E, dimensionless in MeV/(MeV/c) units)

An earlier attempt used a dedicated Screen placed behind the cathode instead -- this required
extending the tracked domain there, which let backward particles keep being actively integrated
(rather than drifting force-free) and caused a ~200x tracking slowdown at production particle
counts. This module avoids that entirely: no extra screen, no domain change, no performance cost.

Caveat: the reconstruction assumes nothing else pushes the particle once behind the cathode. If
space charge is enabled, its self-field is a small residual effect not corrected for here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import c, q_e
from .particle_tags import ID_COL

#: Column indices within `EXTENDED_PHASE_FMT` ("%X %Px %Y %Py %Z %Pz %id %t %E %K"), matching
#: `rf_gun.particle_tags`'s convention.
_T_COL = 7
_E_COL = 8
_K_COL = 9


@dataclass(frozen=True)
class BackBombardmentData:
    """Per-particle reconstructed state at the cathode plane (z=0), for every particle behind the
    cathode (z<0) at `Bout`. All arrays are the same length; `valid` marks which entries have a
    physically plausible reconstruction (see `compute_back_bombardment`) -- callers should index
    with `valid` before using `x_hit_mm`/`y_hit_mm`/`t_hit_s`.

    `n_screens_reached`/`last_screen_z_mm` are the screen-crossing trace (see
    `compute_back_bombardment`): how far into z>0 each particle got, by presence of its `%id` in a
    screen's own recorded array (a screen's *known* z, not that particle's unreliable per-screen
    `%Z`), before it eventually turned around and crossed back through z=0. Populated regardless of
    `valid` (independent of the ballistic (x, y, t) reconstruction).

    `ids`/`z_bout_mm` are the particle's `%id` and its own (trusted, absolute) z at `Bout` -- used
    to look up a specific particle's screen trajectory (`screen_trajectory`) and to plot its
    trusted `Bout` point for comparison (`rf_gun.plotting.back_bombardment.plot_back_bombardment_screen_reach`).
    """

    x_hit_mm: np.ndarray
    y_hit_mm: np.ndarray
    t_hit_s: np.ndarray
    E_total_MeV: np.ndarray
    K_MeV: np.ndarray
    pz_MeVc: np.ndarray
    z_bout_mm: np.ndarray
    ids: np.ndarray
    n_screens_reached: np.ndarray
    last_screen_z_mm: np.ndarray
    valid: np.ndarray
    weight_per_macroparticle: float
    n_behind_cathode: int
    n_valid: int


def compute_back_bombardment(
    Bout_M: np.ndarray,
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    *,
    q_total_C: float,
    n_macroparticles: int,
    r_max_mm: float,
    id_col: int = ID_COL,
) -> BackBombardmentData:
    """Reconstruct (x, y, t) at the z=0 crossing for every particle behind the cathode at Bout.

    `q_total_C`/`n_macroparticles` give the real-electron weight of each (uniformly-weighted)
    macroparticle (`weight = |q_total_C| / q_e / n_macroparticles`), needed to convert a
    macroparticle's own energy into the energy actually deposited by the real electrons it
    represents (see `rf_gun.plotting.back_bombardment`'s figures).

    `r_max_mm` is the field map's own valid transverse extent (`R_MAX_M` in the notebook, in mm).
    A handful of near-zero-Pz stragglers make the ballistic reconstruction blow up to unphysical
    values (near-zero Pz implies an enormous drift time to reach even a small |z|, so any nonzero
    Px/Py also implies an enormous transverse drift) -- `valid` excludes any reconstructed (x, y)
    beyond this bound, since the simulated fields don't exist out there in the first place.

    `M_snaps`/`z_snaps` (the same screen snapshots and their known z, used everywhere else in the
    project) are used to trace how far forward each back-bombardment particle got before it turned
    around -- see `_screen_reach` and `BackBombardmentData`'s docstring.
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
            pz_MeVc=empty,
            z_bout_mm=empty,
            ids=np.zeros((0,), dtype=np.int64),
            n_screens_reached=np.zeros((0,), dtype=int),
            last_screen_z_mm=empty,
            valid=np.zeros((0,), dtype=bool),
            weight_per_macroparticle=_weight_per_macroparticle(q_total_C, n_macroparticles),
            n_behind_cathode=0,
            n_valid=0,
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

    n_screens_reached, last_screen_z_mm = _screen_reach(ids, M_snaps, z_snaps, id_col=id_col)

    return BackBombardmentData(
        x_hit_mm=x_hit,
        y_hit_mm=y_hit,
        t_hit_s=t_hit_s,
        E_total_MeV=E,
        K_MeV=K,
        pz_MeVc=pz,
        z_bout_mm=z,
        ids=ids,
        n_screens_reached=n_screens_reached,
        last_screen_z_mm=last_screen_z_mm,
        valid=valid,
        weight_per_macroparticle=_weight_per_macroparticle(q_total_C, n_macroparticles),
        n_behind_cathode=n,
        n_valid=int(valid.sum()),
    )


def _screen_reach(
    ids: np.ndarray,
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    id_col: int = ID_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """For each id in `ids`, how many screens (of `M_snaps`/`z_snaps`) recorded it, and the
    furthest (largest-z) one -- by presence of the id in that screen's own recorded array, using
    the screen's *known* z (not that particle's unreliable per-screen `%Z`). NaN/0 if never
    recorded at any screen (an immediate bounce-back, never far enough forward to cross one)."""
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
    """One particle's `(screen z [mm], that screen's own recorded Pz [MeV/c])`, sorted by z, over
    every screen where its `%id` was recorded.

    Caveat (see `rf_gun.particle_tags`'s module docstring): a screen's own Pz can carry the wrong
    sign once a particle has already turned around and re-crosses that plane heading backward. No
    screen in this project's snapshots has ever been observed to record the same id twice (checked
    against a real run's data), so exactly one crossing -- forward or backward, ambiguous from the
    screen's row alone -- is available per screen; callers should compare against the particle's
    trusted `Bout` state (`rf_gun.plotting.back_bombardment.plot_back_bombardment_screen_reach`
    does this) rather than trust a screen's Pz on its own.
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
    """Kinetic energy deposited in Joules by the REAL electrons each row represents (i.e. already
    scaled by `weight_per_macroparticle`) -- kinetic, not total, energy: only kinetic energy
    converts to heat on absorption, the rest-mass energy does not (a large fraction of the total
    here, since these energies are comparable to the electron rest mass). Same length as every
    other array on `data`; index with `data.valid` before using (matching `x_hit_mm`/`y_hit_mm`)."""
    return data.K_MeV * 1e6 * q_e * data.weight_per_macroparticle
