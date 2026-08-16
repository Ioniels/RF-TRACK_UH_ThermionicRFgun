"""Diagnostics and summary helpers shared across simulation and plotting."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def _second_moment_twiss(u: np.ndarray, pu: np.ndarray):
    """Shared second-moment computation: returns (alpha, beta, gamma, eps_geom).

    `eps_geom = sqrt(det(cov{u,pu}))` is the geometric emittance in whatever units
    `u`/`pu` are given in (e.g. mm and rad -> mm*rad). Kept private since
    `twiss_from_moments` (below) is the long-standing public 3-tuple return used
    throughout this codebase; `manual_twiss_and_emittance` (below) is the only other
    caller, added to expose `eps_geom` for the native-vs-manual comparison.
    """
    if u.size < 2 or pu.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    u0 = u - np.mean(u)
    pu0 = pu - np.mean(pu)
    s11 = float(np.mean(u0 * u0))
    s22 = float(np.mean(pu0 * pu0))
    s12 = float(np.mean(u0 * pu0))
    det = s11 * s22 - s12 * s12
    if not np.isfinite(det) or det <= 0.0:
        return np.nan, np.nan, np.nan, np.nan
    eps = np.sqrt(det)
    alpha = -s12 / eps
    beta = s11 / eps
    gamma = s22 / eps
    return float(alpha), float(beta), float(gamma), float(eps)


def twiss_from_moments(u: np.ndarray, pu: np.ndarray):
    """Twiss parameters (alpha, beta, gamma) from second moments."""
    alpha, beta, gamma, _eps_geom = _second_moment_twiss(u, pu)
    return alpha, beta, gamma


def dispersion_from_moments(u: np.ndarray, delta: np.ndarray) -> float:
    """D_u = Cov(u, delta) / Var(delta), mean-centered and population-normalized (ddof=0) to
    match `_second_moment_twiss`'s convention exactly. `u` can be a position (-> D_x [mm]) or a
    divergence `px/pz` (-> D_x' [rad]); `delta` is the relative momentum deviation
    `(pz - mean(pz)) / mean(pz)`.
    """
    if u.size < 2 or delta.size < 2:
        return np.nan
    u0 = u - np.mean(u)
    d0 = delta - np.mean(delta)
    var_d = float(np.mean(d0 * d0))
    if not np.isfinite(var_d) or var_d <= 0.0:
        return np.nan
    return float(np.mean(u0 * d0) / var_d)


def manual_twiss_and_emittance(M: np.ndarray, mass_MeV: float) -> Dict[str, float]:
    """Twiss/emittance for [x,px,y,py,z,pz] via plain numpy second moments.

    This is the project's sole Twiss/emittance computation. RF-Track's own native
    `Bunch6dT.get_info()` Twiss (`alpha_x`/`alpha_y`) was empirically found to be
    unreliable under real (non-negligible) x-x'/y-y' correlation -- reproduced
    against both RF-Track 2.5.4 and 2.6.3, so not a version-specific defect -- and
    a `Screen`'s own `get_info()` returns an internal `Bunch6d` (not `Bunch6dT`)
    object whose field names collide with this project's `Bunch6dT`-style lookups
    (e.g. its `sigma_px`/`sigma_py` are mrad angle spreads, not MeV/c momentum
    spreads, but a case-insensitive field lookup for "sigma_Px" matches them
    anyway). Neither RF-Track-native path is used here; see the notebook's
    beam-parameter cell and `UPGRADE_PLAN_notebook_and_architecture.md` for the
    full empirical writeup.

    `beta_x`/`beta_y` in mm, `emitt_x`/`emitt_y` (and `_norm` aliases) normalized in mm*mrad (via
    the paraxial beta*gamma = mean_pz / mass_MeV relation); `emitt_x_geom`/`emitt_y_geom` are the
    same quantity before that scaling (mm*rad). `gamma_x`/`gamma_y`/`gamma_z` = (1+alpha^2)/beta,
    from the same second-moment computation as alpha/beta (no separate formula/pass needed).

    Longitudinal (`alpha_z`/`beta_z`/`emitt_z`) uses this project's own convention
    -- `z` [mm] vs. `delta = pz / mean(pz)` [dimensionless] -- since RF-Track's own
    manual documents a different, incompletely-specified longitudinal convention
    (`emitt_z` in mm.permille, no closed-form alpha_z/beta_z formula given) that
    does not reduce to a simple unit rescaling of (z, delta). `emitt_z` is already the
    geometric value (z, delta) itself has no natural "normalized" counterpart here.
    """
    out_keys = [
        "alpha_x", "beta_x", "gamma_x", "emitt_x", "emitt_x_norm", "emitt_x_geom",
        "alpha_y", "beta_y", "gamma_y", "emitt_y", "emitt_y_norm", "emitt_y_geom",
        "alpha_z", "beta_z", "gamma_z", "emitt_z",
    ]
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] < 6:
        return {k: np.nan for k in out_keys}

    x, px, y, py, z, pz = (arr[:, i] for i in range(6))
    pz_mean = float(np.mean(pz))
    p0 = pz_mean if pz_mean > 0.0 else 1.0
    betagamma = pz_mean / float(mass_MeV) if mass_MeV > 0.0 else np.nan

    ax, bx, gx, ex = _second_moment_twiss(x, px / pz)
    ay, by, gy, ey = _second_moment_twiss(y, py / pz)
    az, bz, gz, ez = _second_moment_twiss(z, pz / p0)

    ex_norm = ex * betagamma * 1e3 if np.isfinite(ex) else np.nan
    ey_norm = ey * betagamma * 1e3 if np.isfinite(ey) else np.nan

    return {
        "alpha_x": ax, "beta_x": bx, "gamma_x": gx,
        "emitt_x": ex_norm, "emitt_x_norm": ex_norm, "emitt_x_geom": ex,
        "alpha_y": ay, "beta_y": by, "gamma_y": gy,
        "emitt_y": ey_norm, "emitt_y_norm": ey_norm, "emitt_y_geom": ey,
        "alpha_z": az, "beta_z": bz, "gamma_z": gz, "emitt_z": ez,
    }


def info_get(info: Any, key: str):
    if info is None:
        return np.nan
    if isinstance(info, dict):
        if key in info:
            return info[key]
        if key.lower() in info:
            return info[key.lower()]
        if key.upper() in info:
            return info[key.upper()]
        return np.nan
    if hasattr(info, key):
        val = getattr(info, key)
        return val() if callable(val) else val
    if hasattr(info, key.lower()):
        val = getattr(info, key.lower())
        return val() if callable(val) else val
    if hasattr(info, key.upper()):
        val = getattr(info, key.upper())
        return val() if callable(val) else val
    if hasattr(info, f"get_{key}"):
        getter = getattr(info, f"get_{key}")
        return getter() if callable(getter) else getter
    return np.nan


def info_get_first(info: Any, keys: Sequence[str]):
    for key in keys:
        val = info_get(info, key)
        try:
            fval = float(val)
        except Exception:
            continue
        if np.isfinite(fval):
            return fval
    return np.nan


def snapshot_stats(M: np.ndarray) -> Dict[str, float]:
    if M.size == 0:
        return {"N": 0.0, "mean_pz": np.nan, "sigma_pz": np.nan, "transmission": np.nan}
    out = {
        "N": float(M.shape[0]),
        "mean_pz": float(np.mean(M[:, 5])) if M.ndim == 2 and M.shape[1] > 5 else np.nan,
        "sigma_pz": float(np.std(M[:, 5])) if M.ndim == 2 and M.shape[1] > 5 else np.nan,
        "transmission": np.nan,
    }
    return out


def summarize_array(values: np.ndarray) -> Dict[str, Any]:
    """Return robust numeric summary, preserving count and finite_count."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    out: Dict[str, Any] = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if finite.size == 0:
        return out
    out["min"] = float(np.min(finite))
    out["max"] = float(np.max(finite))
    out["mean"] = float(np.mean(finite))
    out["std"] = float(np.std(finite))
    return out


def summarize_phase_space(M: np.ndarray, columns: Sequence[str] | None = None) -> Dict[str, Any]:
    """Column-wise summary for a 2D phase-space matrix."""
    arr = np.asarray(M, dtype=float)
    col_names = list(columns) if columns is not None else ["x_mm", "px_MeV_c", "y_mm", "py_MeV_c", "z_mm", "pz_MeV_c"]
    out: Dict[str, Any] = {
        "particle_count": int(arr.shape[0]) if arr.ndim == 2 else 0,
        "columns": col_names,
        "summary": {},
    }
    if arr.ndim != 2 or arr.shape[0] == 0:
        for i, name in enumerate(col_names):
            if i < (arr.shape[1] if arr.ndim == 2 else 0):
                out["summary"][name] = summarize_array(arr[:, i])
            else:
                out["summary"][name] = summarize_array(np.asarray([], dtype=float))
        return out

    n_cols = arr.shape[1]
    for i, name in enumerate(col_names):
        if i < n_cols:
            out["summary"][name] = summarize_array(arr[:, i])
        else:
            out["summary"][name] = summarize_array(np.asarray([], dtype=float))
    return out


def build_screen_summary_from_phase_space(
    M_screen: np.ndarray | None,
    screen_index: int,
    z_m: float,
    n_initial: int,
    n_previous: int | None = None,
) -> Dict[str, Any]:
    """Build robust per-screen summary from explicit phase-space array only."""
    arr = np.asarray(M_screen, dtype=float) if M_screen is not None else np.zeros((0, 6), dtype=float)
    if arr.ndim != 2:
        arr = np.zeros((0, 6), dtype=float)

    n_screen = int(arr.shape[0])
    n_prev = int(n_previous) if n_previous is not None else int(n_initial)

    pz = arr[:, 5] if arr.shape[1] > 5 else np.asarray([], dtype=float)
    z = arr[:, 4] if arr.shape[1] > 4 else np.asarray([], dtype=float)
    x = arr[:, 0] if arr.shape[1] > 0 else np.asarray([], dtype=float)
    y = arr[:, 2] if arr.shape[1] > 2 else np.asarray([], dtype=float)

    pz_f = pz[np.isfinite(pz)]
    z_f = z[np.isfinite(z)]
    x_f = x[np.isfinite(x)]
    y_f = y[np.isfinite(y)]

    def _mean_or_none(vals: np.ndarray, scale: float = 1.0):
        if vals.size == 0:
            return None
        return float(scale * np.mean(vals))

    def _std_or_none(vals: np.ndarray, scale: float = 1.0):
        if vals.size == 0:
            return None
        return float(scale * np.std(vals))

    tr_init = (float(n_screen) / float(n_initial)) if int(n_initial) > 0 else None
    tr_prev = (float(n_screen) / float(n_prev)) if int(n_prev) > 0 else None

    summary: Dict[str, Any] = {
        "screen_index": int(screen_index),
        "z_m": float(z_m),
        "N": int(n_screen),
        "transmission_from_initial": tr_init,
        "transmission_from_previous": tr_prev,
        "mean_pz_MeV_c": _mean_or_none(pz_f),
        "sigma_pz_MeV_c": _std_or_none(pz_f),
        "mean_x_mm": _mean_or_none(x_f),
        "sigma_x_mm": _std_or_none(x_f),
        "mean_y_mm": _mean_or_none(y_f),
        "sigma_y_mm": _std_or_none(y_f),
        "mean_z_mm": _mean_or_none(z_f, scale=1e3),
        "sigma_z_mm": _std_or_none(z_f, scale=1e3),
        # Backward-compatible aliases used by existing callers.
        "transmission": tr_init,
        "mean_pz": _mean_or_none(pz_f),
        "sigma_pz": _std_or_none(pz_f),
    }
    return summary


def build_screen_summaries(
    z_snaps: Sequence[float],
    info_snaps: Sequence[Any] | None,
    phase_spaces: Sequence[np.ndarray] | None,
) -> list[Dict[str, Any]]:
    """Build robust summary records from phase-space arrays, with optional RF-Track raw info."""
    n = int(len(z_snaps))
    out: list[Dict[str, Any]] = []
    n_initial = 0
    if phase_spaces is not None and len(phase_spaces) > 0:
        first = np.asarray(phase_spaces[0])
        if first.ndim == 2:
            n_initial = int(first.shape[0])
    n_previous = n_initial

    for i in range(n):
        rec: Dict[str, Any] = {"screen_index": int(i), "z_m": float(z_snaps[i])}
        info_i = info_snaps[i] if info_snaps is not None and i < len(info_snaps) else None
        rec["rftrack_raw_info"] = {
            "transmission": info_get_first(info_i, ["transmission", "Transmission"]),
            "mean_pz": info_get_first(info_i, ["mean_Pz", "mean_P", "mean_pz"]),
            "sigma_pz": info_get_first(info_i, ["sigma_Pz", "sigma_P", "sigma_pz"]),
        }

        m_i = phase_spaces[i] if phase_spaces is not None and i < len(phase_spaces) else None
        robust = build_screen_summary_from_phase_space(
            m_i,
            screen_index=i,
            z_m=float(z_snaps[i]),
            n_initial=n_initial,
            n_previous=n_previous,
        )
        n_previous = int(robust["N"])
        rec.update(robust)

        out.append(rec)
    return out


def _masked_mean(arr: np.ndarray, mask: np.ndarray):
    if arr is None or mask.size == 0 or not np.any(mask):
        return np.nan
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def classify_particle_outcomes(
    initial: np.ndarray,
    final: np.ndarray,
    t0_mm_c: np.ndarray | None = None,
    lost_table: np.ndarray | None = None,
):
    """Classify final particles into transmitted/backward and include lost table stats."""
    initial = np.asarray(initial)
    final = np.asarray(final)
    n0 = int(initial.shape[0]) if initial.ndim == 2 else 0

    if final.ndim != 2 or final.shape[0] == 0 or final.shape[1] < 6:
        transmitted_mask = np.zeros((0,), dtype=bool)
        backward_mask = np.zeros((0,), dtype=bool)
    else:
        zf = np.asarray(final[:, 4], dtype=float)
        pzf = np.asarray(final[:, 5], dtype=float)
        transmitted_mask = np.isfinite(zf) & np.isfinite(pzf) & (zf > 0.0) & (pzf > 0.0)
        backward_mask = np.isfinite(zf) & np.isfinite(pzf) & ((zf <= 0.0) | (pzf < 0.0))

    n_match = min(int(initial.shape[0]) if initial.ndim == 2 else 0, int(final.shape[0]) if final.ndim == 2 else 0)
    pz0 = np.asarray(initial[:n_match, 5], dtype=float) if initial.ndim == 2 and initial.shape[1] > 5 and n_match > 0 else None
    pzf = np.asarray(final[:n_match, 5], dtype=float) if final.ndim == 2 and final.shape[1] > 5 and n_match > 0 else None
    zf = np.asarray(final[:n_match, 4], dtype=float) if final.ndim == 2 and final.shape[1] > 4 and n_match > 0 else None
    t0 = np.asarray(t0_mm_c, dtype=float).reshape(-1)[:n_match] if t0_mm_c is not None and n_match > 0 else None
    transmitted_mask_match = transmitted_mask[:n_match] if transmitted_mask.size >= n_match else transmitted_mask
    backward_mask_match = backward_mask[:n_match] if backward_mask.size >= n_match else backward_mask

    n_trans = int(np.sum(transmitted_mask))
    n_back = int(np.sum(backward_mask))
    n_lost = int(np.asarray(lost_table).shape[0]) if lost_table is not None else max(0, n0 - int(final.shape[0]))

    def frac(n: int) -> float:
        return float(n / n0) if n0 > 0 else np.nan

    return {
        "n_initial": n0,
        "n_final": int(final.shape[0]) if final.ndim == 2 else 0,
        "transmitted": {
            "count": n_trans,
            "fraction": frac(n_trans),
            "initial_pz_mean": _masked_mean(pz0, transmitted_mask_match) if pz0 is not None else np.nan,
            "final_pz_mean": _masked_mean(pzf, transmitted_mask_match) if pzf is not None else np.nan,
            "initial_t0_mean_mm_c": _masked_mean(t0, transmitted_mask_match) if t0 is not None else np.nan,
            "final_z_mean_mm": 1e3 * _masked_mean(zf, transmitted_mask_match) if zf is not None else np.nan,
        },
        "backward_returned": {
            "count": n_back,
            "fraction": frac(n_back),
            "initial_pz_mean": _masked_mean(pz0, backward_mask_match) if pz0 is not None else np.nan,
            "final_pz_mean": _masked_mean(pzf, backward_mask_match) if pzf is not None else np.nan,
            "initial_t0_mean_mm_c": _masked_mean(t0, backward_mask_match) if t0 is not None else np.nan,
            "final_z_mean_mm": 1e3 * _masked_mean(zf, backward_mask_match) if zf is not None else np.nan,
        },
        "lost": {
            "count": n_lost,
            "fraction": frac(n_lost),
            "initial_pz_mean": np.nan,
            "final_pz_mean": np.nan,
            "initial_t0_mean_mm_c": np.nan,
            "final_z_mean_mm": np.nan,
        },
    }


def to_lost_table_array(raw_lost: Any):
    """Normalize RF-Track lost-particle table to ndarray with expected columns."""
    if raw_lost is None:
        return None
    arr = np.asarray(raw_lost)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return None
    if arr.shape[0] == 0:
        return np.zeros((0, 11), dtype=float)
    return np.asarray(arr, dtype=float)
