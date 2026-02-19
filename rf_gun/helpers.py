"""Generic helpers used across modules."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .constants import ME_MEV, c, q_e


def kinetic_energy(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """Return kinetic energy [MeV] from momenta [MeV/c]."""
    p2 = px**2 + py**2 + pz**2
    gamma = np.sqrt(1.0 + p2 / ME_MEV**2)
    return (gamma - 1.0) * ME_MEV


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


def theoretical_energy_gain(Ez_axis_phasor: np.ndarray, z_m: np.ndarray, phi_rad: float) -> float:
    """Energy gain [MeV] from on-axis phasor: dW = -e * integral(Re(Ez*exp(i*phi))) dz."""
    Ez_real = np.real(Ez_axis_phasor * np.exp(1j * float(phi_rad)))
    dW_J = (-q_e) * np.trapezoid(Ez_real, z_m)
    return float(dW_J / (q_e * 1e6))


def cavity_wavelength(f_hz: float) -> Dict[str, float]:
    """lambda, lambda/2, lambda/4 for a given frequency."""
    lam = c / float(f_hz)
    return {"lambda": float(lam), "lambda/2": float(lam / 2.0), "lambda/4": float(lam / 4.0)}
