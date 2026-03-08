"""I/O helpers for simulation outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .diagnostics import build_screen_summary_from_phase_space, info_get_first

SCREEN_COLUMNS = ["x_mm", "px_MeV_c", "y_mm", "py_MeV_c", "z_mm", "pz_MeV_c"]
LOST_COLUMNS = ["x", "px", "y", "py", "z", "pz", "t", "mass", "q", "N", "id"]


def to_json_safe(value: Any) -> Any:
    """Recursively sanitize objects for strict JSON (no NaN/Inf)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    return str(value)


def save_screen_distributions_json(
    output_dir: Path,
    z_snaps: Sequence[float],
    M_snaps: Sequence[np.ndarray] | None,
    I_snaps: Sequence[Any] | None,
    *,
    mode: str = "summary",
    n_initial: int | None = None,
    robust_summaries: Sequence[dict[str, Any]] | None = None,
) -> int:
    """Save per-screen JSON in summary or full mode."""
    if not z_snaps:
        return 0
    mode_norm = str(mode).strip().lower()
    if mode_norm not in ("summary", "full"):
        raise ValueError(f"Unknown screen JSON mode: {mode}")

    out_dir = Path(output_dir) / "screen_distributions_json"
    out_dir.mkdir(parents=True, exist_ok=True)

    precomputed = [dict(x) for x in robust_summaries] if robust_summaries is not None else []

    n0 = int(n_initial) if n_initial is not None else 0
    if n0 <= 0 and M_snaps is not None and len(M_snaps) > 0:
        first = np.asarray(M_snaps[0])
        if first.ndim == 2:
            n0 = int(first.shape[0])

    saved = 0
    n_prev = n0
    for i, z_m in enumerate(z_snaps):
        M_i = None
        if M_snaps is not None and i < len(M_snaps):
            M_i = np.asarray(M_snaps[i], dtype=float)

        if i < len(precomputed):
            summary = dict(precomputed[i])
        else:
            summary = build_screen_summary_from_phase_space(
                M_i,
                screen_index=i,
                z_m=float(z_m),
                n_initial=n0,
                n_previous=n_prev,
            )
        n_prev = int(summary["N"])

        info_i = I_snaps[i] if (I_snaps is not None and i < len(I_snaps)) else None
        raw_info = {
            "transmission": info_get_first(info_i, ["transmission", "Transmission"]),
            "mean_pz": info_get_first(info_i, ["mean_Pz", "mean_P", "mean_pz"]),
            "sigma_pz": info_get_first(info_i, ["sigma_Pz", "sigma_P", "sigma_pz"]),
        }

        payload: dict[str, Any] = {
            "screen_index": int(i),
            "z_m": float(z_m),
            "mode": mode_norm,
            "summary": summary,
            "rftrack_raw_info": raw_info,
        }

        if mode_norm == "full" and M_snaps is not None and i < len(M_snaps):
            arr = np.asarray(M_snaps[i])
            payload["columns"] = SCREEN_COLUMNS
            payload["n_particles"] = int(arr.shape[0]) if arr.ndim == 2 else 0
            payload["phase_space"] = arr.tolist() if arr.ndim == 2 and arr.size else []

        file_path = out_dir / f"screen_{i:04d}_z_{float(z_m):.6f}m.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(to_json_safe(payload), f, indent=2, sort_keys=True)
        saved += 1

    return saved


def save_lost_particles_json(output_dir: Path, lost_table: np.ndarray | None) -> Path | None:
    if lost_table is None:
        return None
    arr = np.asarray(lost_table)
    if arr.ndim != 2:
        return None

    payload = {
        "columns": LOST_COLUMNS,
        "n_particles": int(arr.shape[0]),
        "lost_particles": arr.tolist(),
    }
    out_path = Path(output_dir) / "lost_particle_diagnostics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2, sort_keys=True)
    return out_path
