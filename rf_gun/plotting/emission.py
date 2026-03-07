"""Emission diagnostics plots."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _as_1d(arr):
    if arr is None:
        return None
    out = np.asarray(arr, dtype=float).reshape(-1)
    return out


def plot_emission_history(thermo_info: Dict[str, Any], show_components: bool = True):
    """Emission history at cathode with model and sampled emission PDF."""
    import matplotlib.pyplot as plt

    t_s = _as_1d(thermo_info.get("t_s", None))
    if t_s is None:
        print("Emission history: no time samples available.")
        return

    Ez_t = _as_1d(thermo_info.get("Ez_t", None))
    F_t = _as_1d(thermo_info.get("F_t", None))
    J_t = _as_1d(thermo_info.get("J_Apm2_t", None))
    J_th_t = _as_1d(thermo_info.get("J_th_Apm2_t", None))
    J_fe_t = _as_1d(thermo_info.get("J_fe_Apm2_t", None))
    I_t = _as_1d(thermo_info.get("I_A_t", None))
    area_m2 = thermo_info.get("area_m2", None)
    n_t = _as_1d(thermo_info.get("n_t", None))
    t_emit_s = _as_1d(thermo_info.get("t_emit_s", None))

    do_components = bool(show_components and J_th_t is not None and J_fe_t is not None)
    nrows = 3 if do_components else 2

    t_ns = np.asarray(t_s) * 1e9
    fig, axes = plt.subplots(nrows, 1, figsize=(8.8, 2.55 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    if F_t is not None:
        axes[0].plot(t_ns, F_t, lw=1.6, label="F(t)")
        axes[0].set_ylabel(r"$F\,(\mathrm{V\,m^{-1}})$")
        axes[0].set_title(r"Cathode extraction field")
    elif Ez_t is not None:
        axes[0].plot(t_ns, Ez_t, lw=1.6, label="Ez(t)")
        axes[0].set_ylabel(r"$E_z\,(\mathrm{V\,m^{-1}})$")
        axes[0].set_title(r"Cathode field")
    axes[0].grid(alpha=0.3)

    if J_t is not None:
        J_cm2 = np.asarray(J_t) * 1e-4
        ln1 = axes[1].plot(t_ns, J_cm2, lw=1.9, color="tab:blue", label=r"$J(t)$")
        axes[1].set_ylabel(r"$J\,(\mathrm{A\,cm^{-2}})$")

        j_finite = J_cm2[np.isfinite(J_cm2)]
        j_pos = j_finite[j_finite > 0.0]
        if j_pos.size > 0:
            j_min = float(np.min(j_pos))
            j_max = float(np.max(j_pos))
            if np.isfinite(j_min) and np.isfinite(j_max) and j_max > j_min:
                axes[1].set_ylim(0.8 * j_min, j_max)

        scale_I_per_J = None
        if area_m2 is not None:
            scale_I_per_J = float(area_m2) * 1e4  # I[A] = J[A/cm^2] * area[cm^2]
        elif I_t is not None and np.size(I_t) == np.size(J_cm2):
            denom = np.maximum(np.abs(J_cm2), 1e-30)
            ratio = np.asarray(I_t, dtype=float) / denom
            ratio = ratio[np.isfinite(ratio)]
            if ratio.size:
                scale_I_per_J = float(np.median(ratio))

        if scale_I_per_J is not None and np.isfinite(scale_I_per_J) and scale_I_per_J > 0.0:
            ax_right = axes[1].secondary_yaxis(
                "right",
                functions=(
                    lambda y: y * scale_I_per_J,
                    lambda y: y / scale_I_per_J,
                ),
            )
            ax_right.set_ylabel(r"$I\,(\mathrm{A})$")

        if t_emit_s is not None and t_emit_s.size > 0:
            t_emit_ns = t_emit_s * 1e9
            t0 = float(np.nanmin(t_ns))
            t1 = float(np.nanmax(t_ns))
            in_window = t_emit_ns[np.isfinite(t_emit_ns)]
            if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
                in_window = in_window[(in_window >= t0) & (in_window <= t1)]

            if in_window.size > 0:
                bins = int(np.clip(np.sqrt(in_window.size), 30, 140))
                p_hist, edges = np.histogram(in_window, bins=bins, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                widths = np.diff(edges)
                p_max = float(np.max(p_hist)) if p_hist.size else 0.0
                j_max = float(np.max(j_pos)) if j_pos.size else 0.0
                if p_max > 0.0 and j_max > 0.0:
                    scale_pdf = j_max / p_max
                    axes[1].bar(
                        centers,
                        p_hist * scale_pdf,
                        width=widths,
                        alpha=0.25,
                        color="tab:blue",
                        edgecolor="none",
                        label=f"PDF {int(in_window.size)} RF-Track macroparticles",
                        zorder=0,
                    )

        law = str(thermo_info.get("emission_law", "")).strip()
        if law == "RD_schottky":
            axes[1].set_title("Thermionic emission model: Richardson-Dushman with Schottky lowering")
        elif law == "unified":
            axes[1].set_title("Unified model: thermionic and field emission")
        else:
            axes[1].set_title(r"Emission model")

        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(handles, labels, frameon=False, loc="upper right")
    axes[1].grid(alpha=0.3)

    if do_components:
        comp_row = 2
        axes[comp_row].plot(t_ns, np.asarray(J_th_t) * 1e-4, lw=1.5, ls="--", label=r"$J_{\mathrm{th}}(t)$")
        axes[comp_row].plot(t_ns, np.asarray(J_fe_t) * 1e-4, lw=1.5, ls=":", label=r"$J_{\mathrm{fe}}(t)$")
        if n_t is not None and n_t.size == t_ns.size:
            axn = axes[comp_row].twinx()
            axn.plot(t_ns, n_t, lw=1.2, color="tab:purple", alpha=0.8, label=r"$n(t)$")
            axn.set_ylabel(r"$n$", color="tab:purple")
            axn.tick_params(axis="y", labelcolor="tab:purple")
        axes[comp_row].set_ylabel(r"$J\,(\mathrm{A\,cm^{-2}})$")
        axes[comp_row].set_title(r"Unified-law components")
        axes[comp_row].legend(frameon=False, loc="upper right")
        axes[comp_row].grid(alpha=0.3)

    axes[-1].set_xlabel(r"$t\,(\mathrm{ns})$")

    if area_m2 is not None and J_t is not None and I_t is not None:
        I_from_J = np.asarray(J_t) * float(area_m2)
        denom = np.maximum(np.abs(I_from_J), 1e-30)
        rel_err = np.nanmax(np.abs(np.asarray(I_t) - I_from_J) / denom)
        if np.isfinite(rel_err) and rel_err > 1e-3:
            print(f"Warning: I(t) and J(t)*area differ by up to {rel_err:.2e}")

    plt.tight_layout()
    plt.show()


def plot_j_vs_n(thermo_info: Dict[str, Any]):
    """Scatter plot of J vs n over time (log-log)."""
    import matplotlib.pyplot as plt

    J_t = _as_1d(thermo_info.get("J_Apm2_t", None))
    n_t = _as_1d(thermo_info.get("n_t", None))
    if J_t is None or n_t is None:
        print("J vs n: missing J_t or n_t.")
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.scatter(n_t, np.asarray(J_t) * 1e-4, s=14, alpha=0.7, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$J\,(\mathrm{A\,cm^{-2}})$")
    ax.set_title(r"Emission map: $J$ vs $n$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
