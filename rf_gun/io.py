"""I/O helpers for simulation outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .diagnostics import build_screen_summaries

SCREEN_COLUMNS = ["x_mm", "px_MeV_c", "y_mm", "py_MeV_c", "z_mm", "pz_MeV_c"]
LOST_COLUMNS = ["x", "px", "y", "py", "z", "pz", "t", "mass", "q", "N", "id"]


def save_screen_distributions_json(
    output_dir: Path,
    z_snaps: Sequence[float],
    M_snaps: Sequence[np.ndarray] | None,
    I_snaps: Sequence[Any] | None,
    *,
    mode: str = "summary",
) -> int:
    """Save per-screen JSON in summary or full mode."""
    if not z_snaps:
        return 0
    mode_norm = str(mode).strip().lower()
    if mode_norm not in ("summary", "full"):
        raise ValueError(f"Unknown screen JSON mode: {mode}")

    out_dir = Path(output_dir) / "screen_distributions_json"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = build_screen_summaries(
        z_snaps,
        I_snaps,
        M_snaps if mode_norm == "full" else None,
    )

    saved = 0
    for i, z_m in enumerate(z_snaps):
        payload: dict[str, Any] = {
            "screen_index": int(i),
            "z_m": float(z_m),
            "mode": mode_norm,
            "summary": summaries[i] if i < len(summaries) else {},
        }

        if mode_norm == "full" and M_snaps is not None and i < len(M_snaps):
            arr = np.asarray(M_snaps[i])
            payload["columns"] = SCREEN_COLUMNS
            payload["n_particles"] = int(arr.shape[0]) if arr.ndim == 2 else 0
            payload["phase_space"] = arr.tolist() if arr.ndim == 2 and arr.size else []

        file_path = out_dir / f"screen_{i:04d}_z_{float(z_m):.6f}m.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
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
        json.dump(payload, f)
    return out_path
