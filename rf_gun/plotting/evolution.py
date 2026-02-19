"""Evolution plots along z."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def plot_evolution(M_snaps: Sequence[np.ndarray], z_snaps: Sequence[float]):
    """Plot transverse RMS sizes and mean pz vs z."""
    import matplotlib.pyplot as plt

    if not len(M_snaps) or len(z_snaps) != len(M_snaps):
        print("No snapshots available (M_snaps empty or z_snaps mismatch).")
        return

    z_mm = 1e3 * np.asarray(z_snaps)
    sig_x = np.array([np.std(M[:, 0]) if M.shape[0] else np.nan for M in M_snaps])
    sig_y = np.array([np.std(M[:, 2]) if M.shape[0] else np.nan for M in M_snaps])
    pz_m = np.array([np.mean(M[:, 5]) if M.shape[0] else np.nan for M in M_snaps])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z_mm, sig_x, "o-", ms=3, label="sigma x")
    ax.plot(z_mm, sig_y, "o-", ms=3, label="sigma y")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("RMS size [mm]")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Transverse RMS size vs z")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z_mm, pz_m, "o-", ms=3)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Mean Pz [MeV/c]")
    ax.grid(alpha=0.3)
    ax.set_title("Mean longitudinal momentum vs z")
    plt.show()
