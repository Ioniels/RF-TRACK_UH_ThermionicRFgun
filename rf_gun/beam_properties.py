"""Consolidated beam-properties table: one function, one row per screen.

Computed only on the forward-going, dynamic-aperture-surviving population (via
`rf_gun.particle_tags.surviving_mask`), matching the project's requirement that beam properties
reflect only the beam that actually makes it through the gun's real transverse channel R(z).
Every quantity is manual (numpy second-moment), consistent with the rest of the project's
Twiss/emittance pipeline -- see `rf_gun.diagnostics.manual_twiss_and_emittance`'s docstring for
why RF-Track's native `get_info()` is not used here.

Column layout assumed for `M_snaps`: `rf_gun.simulation.EXTENDED_PHASE_FMT`
("%X %Px %Y %Py %Z %Pz %id %t %E %K"). The core 6 columns (position/momentum) are required;
`%t`/`%E`/`%K` are optional (missing -> NaN for the quantities that need them) so this still works
against older, un-extended phase-space arrays.

Longitudinal quantities use ToF (`%t`), not `%Z`, throughout -- a screen's own `%Z` is not a
lab-frame position at all (each crossing particle's velocity times its time offset from whichever
particle currently serves as the bunch's reference particle), whereas `%t` is a genuine, reliable
per-particle quantity at every screen. So `mean_t_ns`/`sigma_t_ns` (moments) and `alpha_t`/
`beta_t`/`gamma_t`/`emitt_t` (this project's own longitudinal-Twiss convention, (ToF,
Pz/mean(Pz))) replace what used to be `mean_z`/`sigma_z`/`alpha_z`/`beta_z`/`gamma_z`/`emitt_z`.
The longitudinal Twiss is still computed via `rf_gun.diagnostics.manual_twiss_and_emittance`
(which is agnostic to what its "z" column actually represents) by substituting ToF-in-ns into
that column before calling it -- see `_row_for_screen`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .constants import MM_C_TO_NS as _MM_C_TO_NS
from .diagnostics import manual_twiss_and_emittance, dispersion_from_moments
from .particle_tags import ParticleTags, surviving_mask, tag_mask, T_COL as _COL_T, E_COL as _COL_E, K_COL as _COL_K

TWISS_KEYS = [
    "alpha_x", "beta_x", "gamma_x", "emitt_x_norm", "emitt_x_geom",
    "alpha_y", "beta_y", "gamma_y", "emitt_y_norm", "emitt_y_geom",
    "alpha_t", "beta_t", "gamma_t", "emitt_t",
]
MOMENT_KEYS = [
    "mean_x", "mean_y", "mean_t_ns", "mean_px", "mean_py", "mean_pz",
    "sigma_x", "sigma_y", "sigma_t_ns", "sigma_px", "sigma_py", "sigma_pz",
]
DISPERSION_KEYS = ["disp_x", "disp_px", "disp_y", "disp_py"]
ENERGY_TIME_KEYS = ["sigma_E", "mean_K"]


def compute_beam_properties(
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    tags: ParticleTags,
    mass_MeV: float,
) -> List[Dict[str, Any]]:
    """One row per screen, on the forward-going + aperture-surviving subset of that screen.

    Returns a flat `list[dict]`, one per screen, directly usable as `pandas.DataFrame(rows)`.
    """
    rows: List[Dict[str, Any]] = []
    for z_m, M in zip(z_snaps, M_snaps):
        arr = np.asarray(M, dtype=float)
        mask = surviving_mask(arr, tags) if arr.shape[0] else np.zeros((0,), dtype=bool)
        rows.append(_row_for_screen(float(z_m), arr[mask], mass_MeV))
    return rows


def _row_for_screen(z_m: float, Mf: np.ndarray, mass_MeV: float) -> Dict[str, Any]:
    n = int(Mf.shape[0])
    row: Dict[str, Any] = {"z_m": z_m, "z_mm": z_m * 1e3, "N": n}

    if n < 4:
        for k in MOMENT_KEYS + TWISS_KEYS + DISPERSION_KEYS + ENERGY_TIME_KEYS:
            row[k] = np.nan
        return row

    x, px, y, py, _z_unreliable, pz = (Mf[:, i] for i in range(6))
    has_t_col = Mf.shape[1] > _COL_T
    t_ns = Mf[:, _COL_T] * _MM_C_TO_NS if has_t_col else np.full(n, np.nan)

    row["mean_x"], row["sigma_x"] = float(np.mean(x)), float(np.std(x))
    row["mean_y"], row["sigma_y"] = float(np.mean(y)), float(np.std(y))
    row["mean_t_ns"] = float(np.mean(t_ns)) if has_t_col else np.nan
    row["sigma_t_ns"] = float(np.std(t_ns)) if has_t_col else np.nan
    row["mean_px"], row["sigma_px"] = float(np.mean(px)), float(np.std(px))
    row["mean_py"], row["sigma_py"] = float(np.mean(py)), float(np.std(py))
    row["mean_pz"], row["sigma_pz"] = float(np.mean(pz)), float(np.std(pz))

    # Longitudinal Twiss (alpha_t/beta_t/gamma_t/emitt_t): manual_twiss_and_emittance is agnostic
    # to what its "z" column (index 4) represents, so substitute ToF-in-ns there instead of the
    # unreliable screen Z -- x/y Twiss (columns 0-3) and the pz-based betagamma scaling (column 5)
    # are untouched by this substitution.
    Mf_for_twiss = Mf.copy()
    Mf_for_twiss[:, 4] = t_ns
    twiss = manual_twiss_and_emittance(Mf_for_twiss, mass_MeV)
    row["alpha_t"] = twiss.get("alpha_z", np.nan)
    row["beta_t"] = twiss.get("beta_z", np.nan)
    row["gamma_t"] = twiss.get("gamma_z", np.nan)
    row["emitt_t"] = twiss.get("emitt_z", np.nan)
    for k in ("alpha_x", "beta_x", "gamma_x", "emitt_x_norm", "emitt_x_geom",
              "alpha_y", "beta_y", "gamma_y", "emitt_y_norm", "emitt_y_geom"):
        row[k] = twiss.get(k, np.nan)

    pz_mean = row["mean_pz"]
    if np.isfinite(pz_mean) and pz_mean != 0.0:
        delta = (pz - pz_mean) / pz_mean
        row["disp_x"] = dispersion_from_moments(x, delta)
        row["disp_px"] = dispersion_from_moments(px / pz, delta)
        row["disp_y"] = dispersion_from_moments(y, delta)
        row["disp_py"] = dispersion_from_moments(py / pz, delta)
    else:
        for k in DISPERSION_KEYS:
            row[k] = np.nan

    row["sigma_E"] = _std_of_column(Mf, _COL_E)
    row["mean_K"] = _mean_of_column(Mf, _COL_K)

    return row


def _std_of_column(M: np.ndarray, col: int) -> float:
    if M.shape[1] <= col:
        return np.nan
    v = M[:, col]
    v = v[np.isfinite(v)]
    return float(np.std(v)) if v.size else np.nan


def _mean_of_column(M: np.ndarray, col: int) -> float:
    if M.shape[1] <= col:
        return np.nan
    v = M[:, col]
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else np.nan


def transmission_curves(
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    tags: ParticleTags,
    n_initial: int,
) -> Dict[str, np.ndarray]:
    """Two transmission fractions vs z, both relative to `n_initial`:

    - `not_lost`: particles that survive the dynamic aperture (id not in `tags.lost_ids`),
      whether forward- or backward-going -- "did the aperture ever remove this particle," the
      same eventual-fate convention already used for backward tagging (computed once from the
      complete run, applied identically at every screen -- a screen upstream of where a particle
      is eventually removed still correctly shows it counted out here, since its fate is already
      sealed by the time this curve is read).
    - `forward_and_surviving`: also excludes backward-going particles -- forward-going AND
      surviving the aperture.
    """
    not_lost, fwd_surv = [], []
    for z_m, M in zip(z_snaps, M_snaps):
        arr = np.asarray(M, dtype=float)
        n = int(arr.shape[0])
        if n == 0:
            not_lost.append(0)
            fwd_surv.append(0)
            continue
        is_backward, is_lost = tag_mask(arr, tags)
        not_lost.append(int(np.sum(~is_lost)))
        fwd_surv.append(int(np.sum(~is_backward & ~is_lost)))

    n0 = max(int(n_initial), 1)
    return {
        "z_mm": 1e3 * np.asarray(z_snaps, dtype=float),
        "not_lost": np.asarray(not_lost, dtype=float) / n0,
        "forward_and_surviving": np.asarray(fwd_surv, dtype=float) / n0,
    }
