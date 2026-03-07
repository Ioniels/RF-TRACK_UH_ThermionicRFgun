"""Diagnostics and summary helpers shared across simulation and plotting."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def twiss_from_moments(u: np.ndarray, pu: np.ndarray):
    """Twiss parameters from second moments."""
    if u.size < 2 or pu.size < 2:
        return np.nan, np.nan, np.nan
    u0 = u - np.mean(u)
    pu0 = pu - np.mean(pu)
    s11 = float(np.mean(u0 * u0))
    s22 = float(np.mean(pu0 * pu0))
    s12 = float(np.mean(u0 * pu0))
    det = s11 * s22 - s12 * s12
    if not np.isfinite(det) or det <= 0.0:
        return np.nan, np.nan, np.nan
    eps = np.sqrt(det)
    alpha = -s12 / eps
    beta = s11 / eps
    gamma = s22 / eps
    return float(alpha), float(beta), float(gamma)


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


def build_screen_summaries(
    z_snaps: Sequence[float],
    info_snaps: Sequence[Any] | None,
    phase_spaces: Sequence[np.ndarray] | None,
) -> list[Dict[str, float]]:
    """Build one summary record per screen from info and/or phase space."""
    n = int(len(z_snaps))
    out: list[Dict[str, float]] = []
    for i in range(n):
        rec: Dict[str, float] = {"screen_index": float(i), "z_m": float(z_snaps[i])}

        info_i = info_snaps[i] if info_snaps is not None and i < len(info_snaps) else None
        rec["transmission"] = info_get_first(info_i, ["transmission", "Transmission"])
        rec["mean_pz_info"] = info_get_first(info_i, ["mean_Pz", "mean_P", "mean_pz"])
        rec["sigma_pz_info"] = info_get_first(info_i, ["sigma_Pz", "sigma_P", "sigma_pz"])

        m_i = phase_spaces[i] if phase_spaces is not None and i < len(phase_spaces) else None
        if m_i is not None and np.asarray(m_i).ndim == 2:
            stats = snapshot_stats(np.asarray(m_i))
            rec["N"] = stats["N"]
            rec["mean_pz"] = stats["mean_pz"]
            rec["sigma_pz"] = stats["sigma_pz"]
        else:
            rec["N"] = np.nan
            rec["mean_pz"] = np.nan
            rec["sigma_pz"] = np.nan

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
