"""Emission diagnostics plots."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def plot_emission_history(thermo_info: Dict[str, Any], show_components: bool = True):
    """Emission history at cathode: Ez/F, J, I, and n vs time."""
    import matplotlib.pyplot as plt

    t_s = thermo_info.get("t_s", None)
    if t_s is None:
        print("Emission history: no time samples available.")
        return

    Ez_t = thermo_info.get("Ez_t", None)
    F_t = thermo_info.get("F_t", None)
    J_t = thermo_info.get("J_Apm2_t", None)
    J_th_t = thermo_info.get("J_th_Apm2_t", None)
    J_fe_t = thermo_info.get("J_fe_Apm2_t", None)
    I_t = thermo_info.get("I_A_t", None)
    n_t = thermo_info.get("n_t", None)

    t_ns = np.asarray(t_s) * 1e9
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.0), sharex=True)

    if F_t is not None:
        axes[0].plot(t_ns, F_t, lw=1.6, label="F(t)")
        axes[0].set_ylabel("F [V/m]")
        axes[0].set_title("Cathode field")
    elif Ez_t is not None:
        axes[0].plot(t_ns, Ez_t, lw=1.6, label="Ez(t)")
        axes[0].set_ylabel("Ez [V/m]")
        axes[0].set_title("Cathode field")
    axes[0].grid(alpha=0.3)

    if J_t is not None:
        axes[1].plot(t_ns, J_t, lw=1.6, label="J")
        if show_components and J_th_t is not None and J_fe_t is not None:
            axes[1].plot(t_ns, J_th_t, lw=1.2, ls="--", label="J_th")
            axes[1].plot(t_ns, J_fe_t, lw=1.2, ls=":", label="J_fe")
        axes[1].set_ylabel("J [A/m^2]")
        axes[1].set_title("Emission current density")
        axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)

    if I_t is not None:
        axes[2].plot(t_ns, I_t, lw=1.6)
        axes[2].set_ylabel("I [A]")
        axes[2].set_title("Emission current")
    axes[2].grid(alpha=0.3)

    if n_t is not None:
        axes[3].plot(t_ns, n_t, lw=1.6, color="tab:orange")
        axes[3].set_ylabel("n")
        axes[3].set_title("Regime indicator")
    axes[3].set_xlabel("t [ns]")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_j_vs_n(thermo_info: Dict[str, Any]):
    """Scatter plot of J vs n over time (log-log)."""
    import matplotlib.pyplot as plt

    J_t = thermo_info.get("J_Apm2_t", None)
    n_t = thermo_info.get("n_t", None)
    if J_t is None or n_t is None:
        print("J vs n: missing J_t or n_t.")
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.scatter(n_t, J_t, s=14, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("J [A/m^2]")
    ax.set_title("J vs n")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
