"""Phase scan plots."""
from __future__ import annotations

import numpy as np


def theory_plot(phi_deg: np.ndarray, dW_vals: np.ndarray, pz_theory: np.ndarray):
    """Plot theory phase scan for energy gain."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(phi_deg, dW_vals, "o-", ms=4, lw=1.6, color="tab:blue", label="Delta W")
    ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.6)

    ax2 = ax.twinx()
    ax2.plot(phi_deg, pz_theory, "s--", ms=3, lw=1.2, color="tab:orange", alpha=0.9, label="pz (end)")
    ax2.set_ylabel("pz [MeV/c]", color="tab:orange")
    ax2.tick_params(axis="y", colors="tab:orange")

    if np.any(np.isfinite(dW_vals)):
        i_max = int(np.nanargmax(dW_vals))
        ax.axvline(phi_deg[i_max], color="tab:red", ls="--", lw=1.2, alpha=0.7)
        ax.plot(phi_deg[i_max], dW_vals[i_max], "o", ms=7, color="tab:red", zorder=5)
        ax.text(
            phi_deg[i_max],
            dW_vals[i_max],
            f"  peak {dW_vals[i_max]:.3f} MeV @ {phi_deg[i_max]:.1f} deg",
            va="bottom",
            ha="left",
            fontsize=9,
            color="tab:red",
        )

    ax.set_xlabel("RF phase [deg]")
    ax.set_ylabel("Delta W [MeV]", color="tab:blue")
    ax.tick_params(axis="y", colors="tab:blue")
    ax.set_title("Theory: energy gain vs phase")
    ax.grid(alpha=0.3)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")

    plt.tight_layout()
    plt.show()


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
