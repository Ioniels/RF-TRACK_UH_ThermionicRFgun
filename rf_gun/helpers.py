"""Generic helpers used across modules."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def sample_disk(n: int, radius_mm: float, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random distribution over a disk of radius `radius_mm`."""
    if n <= 0 or radius_mm <= 0:
        return np.zeros(max(n, 0)), np.zeros(max(n, 0))
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(n)
    theta = 2.0 * np.pi * rng.random(n)
    r = radius_mm * np.sqrt(u)
    return r * np.cos(theta), r * np.sin(theta)


def min_step(vals: np.ndarray) -> float:
    """Min positive spacing."""
    u = np.unique(np.asarray(vals))
    if u.size < 2:
        return np.nan
    d = np.diff(np.sort(u))
    d = d[d > 0]
    return float(d.min()) if d.size else np.nan


def med_step(vals: np.ndarray) -> float:
    """Median positive spacing."""
    u = np.unique(np.asarray(vals))
    if u.size < 2:
        return np.nan
    d = np.diff(np.sort(u))
    d = d[d > 0]
    return float(np.median(d)) if d.size else np.nan


def fmt_bytes(n: float) -> str:
    """Byte size label."""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"
