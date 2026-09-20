"""Small deterministic input builders shared by multiple regression tests."""
from __future__ import annotations

import numpy as np

from rf_gun.constants import ME_MEV


def gaussian_bunch_matrix(n, sigma_r_mm, z_mm, N_real_each, rng):
    """Stationary electron disk with Gaussian transverse coordinates."""
    r = sigma_r_mm * np.sqrt(-2.0 * np.log(rng.uniform(1e-12, 1.0, n)))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    x, y = r * np.cos(theta), r * np.sin(theta)
    zeros = np.zeros(n)
    return np.column_stack([
        x, zeros, y, zeros, np.full(n, z_mm), zeros,
        np.full(n, ME_MEV), np.full(n, -1.0), np.full(n, N_real_each), zeros,
    ])
