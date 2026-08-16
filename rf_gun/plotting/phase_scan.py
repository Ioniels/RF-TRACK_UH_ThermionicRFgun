"""Phase scan plots."""
from __future__ import annotations

import numpy as np


def phase_plot(phi_abs: np.ndarray, pz_mean: np.ndarray):
    """Plot fast phase scan pz vs phase."""
    import matplotlib.pyplot as plt

    mask = np.isfinite(pz_mean)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(phi_abs[mask], pz_mean[mask], "o-", ms=4, lw=1.6, color="tab:blue")
    ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.6)
    if np.any(mask):
        i_max = int(np.nanargmax(pz_mean))
        ax.axvline(phi_abs[i_max], color="tab:red", ls="--", lw=1.2, alpha=0.7)
        ax.plot(phi_abs[i_max], pz_mean[i_max], "o", ms=7, color="tab:red", zorder=5)
        ax.text(
            phi_abs[i_max],
            pz_mean[i_max],
            f"  peak {pz_mean[i_max]:.3f} MeV/c @ {phi_abs[i_max]:.1f} deg",
            va="bottom",
            ha="left",
            fontsize=9,
            color="tab:red",
        )
    ax.set_xlabel("RF phase [deg] (absolute)")
    ax.set_ylabel("Mean pz at exit [MeV/c]")
    ax.set_title("Fast phase scan: mean pz vs phase")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
