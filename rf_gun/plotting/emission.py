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
    area_m2 = thermo_info.get("area_m2", None)
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
        J_cm2 = np.asarray(J_t) * 1e-4
        axes[1].plot(t_ns, J_cm2, lw=1.6, label="J")
        if show_components and J_th_t is not None and J_fe_t is not None:
            axes[1].plot(t_ns, np.asarray(J_th_t) * 1e-4, lw=1.2, ls="--", label="J_th")
            axes[1].plot(t_ns, np.asarray(J_fe_t) * 1e-4, lw=1.2, ls=":", label="J_fe")
        axes[1].set_ylabel("J [A/cm^2]")
        axes[1].set_title("Emission current density")
        axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)

    if I_t is not None or (J_t is not None and area_m2 is not None):
        if I_t is None:
            I_disp = np.asarray(J_t) * float(area_m2)
        else:
            I_disp = np.asarray(I_t)
        if area_m2 is not None and J_t is not None and I_t is not None:
            I_from_J = np.asarray(J_t) * float(area_m2)
            denom = np.maximum(np.abs(I_from_J), 1e-30)
            rel_err = np.nanmax(np.abs(I_disp - I_from_J) / denom)
            if np.isfinite(rel_err) and rel_err > 1e-3:
                print(f"Warning: I(t) and J(t)*area differ by up to {rel_err:.2e}")
        axes[2].plot(t_ns, I_disp, lw=1.6)
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
    ax.scatter(n_t, np.asarray(J_t) * 1e-4, s=14, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("J [A/cm^2]")
    ax.set_title("J vs n")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
