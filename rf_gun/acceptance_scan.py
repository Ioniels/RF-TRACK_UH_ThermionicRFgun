"""Threshold-free identification of trailing (stagnant/halo) particles via an acceptance scan.

Replaces the ad hoc `THRESHOLD_BACKWARD_MEVC` constant (a fixed Pz cutoff) with an automatic,
data-driven cut in the (z, Pz) phase space at `Bout`. `Bout` is used (not a screen) because it is
the one place a particle's `z`/`Pz` are absolute, reliable lab-frame quantities -- see
`rf_gun.particle_tags`'s module docstring; `Bout`'s own `%t` is, conversely, *not* usable here
(every particle shares the same value, since `Bout` is a fixed-time snapshot of a `Bunch6dT`
bunch -- confirmed empirically, std strictly 0.0 across an entire population), so this module is
the one place in the project that keeps `z` rather than switching to ToF.

Algorithm, run once per `k` in a scanned range:

1. Start from the population already excluding *true* backward particles (`z<0` or `Pz<=0` at
   `Bout` -- i.e. `backward_ids_from_bout(..., threshold_backward_mevc=0.0)`, the strict/original
   definition, passed in by the caller).
2. Iteratively re-fit the Courant-Snyder ellipse (`rf_gun.diagnostics._second_moment_twiss`, the
   project's existing second-moment Twiss machinery -- an "n-sigma ellipse cut" is mathematically
   identical to a Mahalanobis-distance cut against this same covariance) to the *currently kept*
   subset, re-test every particle's normalized action `J = gamma*u^2 + 2*alpha*u*pu + beta*pu^2`
   against `k` (`J/(2*eps) <= k`; `J = 2*eps` is the standard value on the RMS ellipse itself),
   and repeat to convergence. This is self-consistent sigma-clipping: a trailing particle inflates
   the covariance on the first pass, so simply comparing against the *unclipped* covariance would
   under-reject; iterating removes that bias.
3. Record the resulting transmission `T(k)` (fraction of the pre-filtered population kept) and
   re-derived emittance `eps(k)`.

`T(k)` is monotone non-decreasing by construction (a larger acceptance only ever keeps more
particles), which the two threshold finders below rely on:

- `k_trailing` (large k, *applied*): the smallest k beyond which `T(k)` has reached its final
  plateau -- i.e. essentially no further particles are found no matter how much larger the
  ellipse grows. This is the "remove the trailing/stagnant population in general" threshold.
- `k_core` (small k, *shown for reference only, never applied* -- the caller must not fold this
  into any tagging): the deepest local minimum of `dT/d(log k)` strictly between the initial rise
  and `k_trailing`. A population made of a well-behaved core plus a distinct trailing group shows
  up as two rises separated by a valley (a stretch of (z, Pz) space with few particles); the
  deepest such valley marks where the core beam's own smooth rise ends, before the trailing
  group's separate rise begins. Validated on real `Bout` data before being written here: a run
  with a clearly two-population structure gave `k_core` at T~93% (end of the smooth core rise) and
  `k_trailing` at T~99.9% (end of a distinct, narrow trailing bump), correctly distinguishing the
  two.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diagnostics import _second_moment_twiss
from .particle_tags import ID_COL

_Z_COL = 4
_PZ_COL = 5

#: Default k-grid: log-spaced, wide enough to cover from "just past the core" through
#: "everything, including extreme outliers" for a typical run.
DEFAULT_K_VALUES = np.geomspace(0.5, 100.0, 120)

#: Fraction of the peak (smoothed) slope below which `T(k)` is considered "flat" -- used both to
#: find where the final plateau begins (k_trailing) and to bound the search for k_core's valley.
_FLAT_SLOPE_FRACTION = 0.05
#: Odd window used to smooth `T(k)` before differentiating (raw per-k transmission is noisy at
#: low particle counts -- one crossing changes T by 1/n).
_SMOOTH_WINDOW = 7


@dataclass(frozen=True)
class AcceptanceScanResult:
    """Everything from the scan: the curves, both thresholds, and per-particle data at each."""

    k_values: np.ndarray
    transmission: np.ndarray  # T(k), fraction of `n_forward` kept
    emittance: np.ndarray  # eps(k), geometric (z_mm, Pz MeV/c) emittance [mm*MeV/c]
    n_forward: int  # size of the pre-filtered (strict-forward) population at Bout

    k_core: float  # main-beam threshold -- reference only
    k_trailing: float  # trailing-removal threshold -- the one actually applied

    ids_forward: np.ndarray  # %id of the pre-filtered population, aligned with the masks below
    z_mm: np.ndarray
    pz_MeVc: np.ndarray
    kept_mask_core: np.ndarray  # boolean mask at k_core, over `ids_forward` -- reference only
    kept_mask_trailing: np.ndarray  # boolean mask at k_trailing, over `ids_forward`

    @property
    def trailing_ids(self) -> frozenset:
        """IDs to add to `backward_ids` -- particles that fail the trailing-removal cut."""
        return frozenset(self.ids_forward[~self.kept_mask_trailing].tolist())


def _iterative_courant_snyder_mask(u: np.ndarray, pu: np.ndarray, k: float, max_iter: int = 20):
    """Self-consistent sigma-clipping at acceptance `k`. Returns `(kept_mask, eps)`."""
    n = u.size
    kept = np.ones(n, dtype=bool)
    eps = np.nan
    for _ in range(max_iter):
        if kept.sum() < 4:
            break
        alpha, beta, gamma, eps = _second_moment_twiss(u[kept], pu[kept])
        if not np.isfinite(eps):
            break
        u0 = u - np.mean(u[kept])
        pu0 = pu - np.mean(pu[kept])
        J = gamma * u0 * u0 + 2.0 * alpha * u0 * pu0 + beta * pu0 * pu0
        new_kept = (J / (2.0 * eps)) <= k
        if np.array_equal(new_kept, kept):
            break
        kept = new_kept
    return kept, eps


def _smooth(y: np.ndarray, win: int = _SMOOTH_WINDOW) -> np.ndarray:
    if y.size < win:
        return y.copy()
    pad = win // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, np.ones(win) / win, mode="valid")[: y.size]


def _find_thresholds(k_values: np.ndarray, T: np.ndarray) -> tuple[int, int]:
    """Returns `(idx_core, idx_trailing)` into `k_values`/`T`. See module docstring for the logic."""
    logk = np.log(k_values)
    T_s = _smooth(T)
    slope = np.gradient(T_s, logk)
    peak = slope.max()
    slope_n = slope / peak if peak > 0 else np.zeros_like(slope)

    # k_trailing: smallest k such that every point from there to the end is flat.
    idx_trailing = len(slope_n) - 1
    for i in range(len(slope_n) - 1, -1, -1):
        if slope_n[i] < _FLAT_SLOPE_FRACTION:
            idx_trailing = i
        else:
            break

    # k_core: deepest local minimum of slope strictly between the initial rise and k_trailing.
    rise_start = int(np.argmax(slope_n > _FLAT_SLOPE_FRACTION))
    minima = [
        i
        for i in range(rise_start + 1, idx_trailing - 1)
        if slope_n[i] <= slope_n[i - 1] and slope_n[i] <= slope_n[i + 1]
    ]
    idx_core = min(minima, key=lambda i: slope_n[i]) if minima else rise_start

    return idx_core, idx_trailing


def scan_acceptance(
    Bout_M: np.ndarray,
    backward_ids_strict: frozenset,
    k_values: Optional[np.ndarray] = None,
    id_col: int = ID_COL,
) -> AcceptanceScanResult:
    """Run the acceptance scan on `Bout`'s strict-forward population.

    `backward_ids_strict` must be the *strict* (`threshold_backward_mevc=0.0`) classification --
    i.e. `rf_gun.particle_tags.backward_ids_from_bout(Bout_M, threshold_backward_mevc=0.0)` -- this
    scan is what replaces any further widening of that threshold, so it must start from the
    unwidened population.
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    k_values = np.asarray(k_values, dtype=float)

    arr = np.asarray(Bout_M, dtype=float)
    ids_all = arr[:, id_col].astype(np.int64) if arr.shape[1] > id_col else np.full(arr.shape[0], -1, dtype=np.int64)
    is_backward = np.isin(ids_all, list(backward_ids_strict)) if backward_ids_strict else np.zeros(arr.shape[0], dtype=bool)
    forward = ~is_backward
    z_mm = arr[forward, _Z_COL]
    pz = arr[forward, _PZ_COL]
    ids_forward = ids_all[forward]
    n_forward = int(forward.sum())

    T = np.empty(k_values.size)
    eps_arr = np.empty(k_values.size)
    kept_masks = []
    for i, k in enumerate(k_values):
        kept, eps = _iterative_courant_snyder_mask(z_mm, pz, float(k))
        T[i] = kept.sum() / n_forward if n_forward else np.nan
        eps_arr[i] = eps
        kept_masks.append(kept)

    idx_core, idx_trailing = _find_thresholds(k_values, T)

    return AcceptanceScanResult(
        k_values=k_values,
        transmission=T,
        emittance=eps_arr,
        n_forward=n_forward,
        k_core=float(k_values[idx_core]),
        k_trailing=float(k_values[idx_trailing]),
        ids_forward=ids_forward,
        z_mm=z_mm,
        pz_MeVc=pz,
        kept_mask_core=kept_masks[idx_core],
        kept_mask_trailing=kept_masks[idx_trailing],
    )
