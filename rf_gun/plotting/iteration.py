"""Emission Fields Iteration figures (implementation guide Sec. 12)."""
from __future__ import annotations

from typing import Any

import numpy as np


def plot_emission_iteration_convergence(result: Any):
    """Figure 1, 2x2 (guide Sec. 12): residuals, charge/current, field measures, and relaxation
    history versus outer iteration."""
    import matplotlib.pyplot as plt

    n_iter = len(result.eps_J_history)
    it = np.arange(n_iter)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    axA.semilogy(it, np.abs(result.eps_J_history), "o-", label=r"$\epsilon_J$")
    axA.semilogy(it, np.abs(result.eps_E_history), "s-", label=r"$\epsilon_E$")
    axA.semilogy(it, np.abs(result.eps_Q_history), "^-", label=r"$\epsilon_Q$")
    cfg = result.config
    axA.axhline(cfg.current_tolerance, color="C0", ls="--", lw=0.9, alpha=0.6)
    axA.axhline(cfg.field_tolerance, color="C1", ls="--", lw=0.9, alpha=0.6)
    axA.axhline(cfg.charge_tolerance, color="C2", ls="--", lw=0.9, alpha=0.6)
    axA.set_ylabel("Residual")
    axA.set_title("Convergence residuals")
    axA.legend(frameon=False, fontsize=11)

    Q_nC = np.asarray(result.Q_history_C) * 1e9
    axB.plot(it, Q_nC, "o-", color="tab:blue", label="Q (nC)")
    axB.set_ylabel("Emitted charge (nC)", color="tab:blue")
    axB.tick_params(axis="y", labelcolor="tab:blue")
    axB2 = axB.twinx()
    axB2.plot(it, result.I_peak_history_A, "s-", color="tab:red", label=r"$I_{\rm peak}$")
    axB2.set_ylabel("Peak current (A)", color="tab:red")
    axB2.tick_params(axis="y", labelcolor="tab:red")
    axB.set_title("Charge and peak current")

    peak_sc = [float(np.max(np.abs(E))) if np.size(E) else 0.0 for E in result.E_SC_history_Vpm]
    peak_mirror = [float(np.max(np.abs(E))) if np.size(E) else 0.0 for E in result.E_mirror_history_Vpm]
    peak_total_minus_rf = [
        float(np.max(np.abs(Et - Erf)))
        for Et, Erf in zip(result.E_total_history_Vpm, result.E_RF_history_Vpm)
    ]
    axC.plot(it, peak_sc, "o-", label=r"peak $|E_{\rm SC}|$")
    axC.plot(it, peak_mirror, "s-", label=r"peak $|E_{\rm mirror}|$")
    axC.plot(it, peak_total_minus_rf, "^-", label=r"peak $|E_{\rm total}-E_{\rm RF}|$")
    axC.set_ylabel(r"$|E|\,(\mathrm{V\,m^{-1}})$")
    axC.set_title("Cathode field corrections")
    axC.legend(frameon=False, fontsize=11)

    axD.plot(it, result.relaxation_history, "o-", color="tab:purple", label=r"$\omega$")
    axD.set_ylabel(r"Relaxation $\omega$", color="tab:purple")
    axD.tick_params(axis="y", labelcolor="tab:purple")
    axD2 = axD.twinx()
    axD2.plot(it, result.runtime_history_s, "s-", color="tab:gray", alpha=0.7, label="runtime (s)")
    axD2.set_ylabel("Runtime per iteration (s)", color="tab:gray")
    axD2.tick_params(axis="y", labelcolor="tab:gray")
    axD.set_title("Relaxation and runtime")

    for ax in (axA, axB, axC, axD):
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3)

    status = "converged" if result.converged else f"not converged ({result.failure_reason})"
    fig.suptitle(f"Emission Fields Iteration convergence -- {status}")
    plt.show()


def plot_emission_iteration_waveforms(result: Any, x_index: int = 0, y_index: int = 0):
    """Figure 2, 2x2 (guide Sec. 12): initial vs converged fields/current at one representative
    cathode cell (`x_index`, `y_index` into result.x_grid_m / result.y_grid_m, default near a
    grid corner)."""
    import matplotlib.pyplot as plt

    t_ns = result.t_grid_s * 1e9
    x_mm = result.x_grid_m[x_index] * 1e3
    y_mm = result.y_grid_m[y_index] * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    E_RF = result.E_RF_history_Vpm[-1][x_index, y_index]
    E_SCf = result.E_SC_history_Vpm[-1][x_index, y_index]
    E_Mf = result.E_mirror_history_Vpm[-1][x_index, y_index]
    E_totf = result.E_total_history_Vpm[-1][x_index, y_index]

    axA.plot(t_ns, E_RF, "-", color="k", lw=1.6, label=r"$E_{\rm RF}$")
    axA.plot(t_ns, E_SCf, "--", color="tab:blue", label=r"$E_{\rm SC}$ (final)")
    axA.plot(t_ns, E_Mf, "--", color="tab:orange", label=r"$E_{\rm mirror}$ (final)")
    axA.plot(t_ns, E_totf, "-", color="tab:red", lw=1.8, label=r"$E_{\rm total}$ (final)")
    axA.set_ylabel(r"$E_z\,(\mathrm{V\,m^{-1}})$")
    axA.set_title(f"Signed fields at (x,y)=({x_mm:.3f}, {y_mm:.3f}) mm")
    axA.legend(frameon=False, fontsize=11)

    J0 = result.J_history_Apm2[0][x_index, y_index] * 1e-4
    Jf = result.J_history_Apm2[-1][x_index, y_index] * 1e-4
    axB.plot(t_ns, J0, "--", color="0.5", label=r"$J^{(0)}(t)$")
    axB.plot(t_ns, Jf, "-", color="tab:blue", label=r"$J^{(\rm final)}(t)$")
    axB.set_ylabel(r"$J\,(\mathrm{A\,cm^{-2}})$")
    axB.set_title("Initial vs converged current density")
    axB.legend(frameon=False, fontsize=11)

    dt_s = result.fixed_sample["dt_s"]
    area_m2 = result.fixed_sample["grid"]["dA_mm2"] * 1e-6  # (n_x, n_y)
    I0 = np.sum(result.J_history_Apm2[0] * area_m2[:, :, None], axis=(0, 1))
    If = np.sum(result.J_history_Apm2[-1] * area_m2[:, :, None], axis=(0, 1))
    Q0 = np.cumsum(I0) * dt_s
    Qf = np.cumsum(If) * dt_s
    axC.plot(t_ns, Q0 * 1e9, "--", color="0.5", label="initial")
    axC.plot(t_ns, Qf * 1e9, "-", color="tab:blue", label="final")
    axC.set_ylabel("Cumulative charge (nC)")
    axC.set_title("Cumulative emitted charge")
    axC.legend(frameon=False, fontsize=11)

    J_floor = 1e-6 * max(np.max(np.abs(J0)), 1e-300)
    denom = np.maximum(np.abs(J0), J_floor)
    rel_corr = (Jf - J0) / denom
    axD.plot(t_ns, rel_corr, "-", color="tab:red")
    axD.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axD.set_ylabel(r"$(J^{(\rm final)}-J^{(0)})/J^{(0)}$")
    axD.set_title("Relative correction")

    for ax in (axA, axB, axC, axD):
        ax.set_xlabel(r"$t\,(\mathrm{ns})$")
        ax.grid(alpha=0.3)

    plt.show()


def plot_emission_iteration_near_cathode(result: Any):
    """Figure 3, 1x3 (guide Sec. 12): near-cathode (x,y)-resolved cathode state after convergence
    -- areal emitted charge, temperature, and peak surface field.

    Genuine 2D (x,y) maps (not radial profiles): the cathode temperature need not be azimuthally
    symmetric (backbombardment/laser heating profiles), so the areal emitted charge and surface
    field it produces need not be either. Cells outside the cathode disk are masked to NaN
    (transparent): `J_history_Apm2`/`temperature_K`/`E_total_history_Vpm` are evaluated on the
    full bounding square, but only the disk (`fixed_sample["grid"]["inside_disk"]`) is physically
    the cathode -- the emission law is still evaluated at every square cell (only each cell's
    *area* weight is zeroed outside the disk, in `run_emission_field_iteration`), so plotting the
    raw arrays unmasked would show a nonphysical current density/field in the four corners of the
    square. Every panel also draws the cathode's own boundary circle
    (`rf_gun.plotting.style.add_cathode_boundary_circle`) so the disk mask's edge is unambiguous.

    The middle (temperature) panel is drawn as a single filled disk, not a heatmap, whenever
    `result.config.cathode_temperature_K` is a plain uniform value (`None` or a scalar -- the
    default, matching every prior version of this code): a pcolormesh with its own colorbar would
    otherwise suggest spatial structure that isn't actually there. A spatially resolved profile
    (a callable `T_K(x_mm, y_mm)`) still renders as a heatmap.
    """
    import matplotlib.pyplot as plt

    from .style import add_cathode_boundary_circle

    x_mm_grid = result.x_grid_m * 1e3
    y_mm_grid = result.y_grid_m * 1e3
    inside_disk = result.fixed_sample["grid"]["inside_disk"]
    cathode_radius_mm = float(result.cathode_radius_mm)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)

    Q_areal_nC_mm2 = np.trapezoid(result.J_history_Apm2[-1], result.t_grid_s, axis=-1) * 1.0e3
    Q_areal_nC_mm2 = np.where(inside_disk, Q_areal_nC_mm2, np.nan)
    im0 = axA.pcolormesh(x_mm_grid, y_mm_grid, Q_areal_nC_mm2.T, cmap="viridis", shading="nearest")
    fig.colorbar(im0, ax=axA, label=r"$\int J\,dt\,(\mathrm{nC\,mm^{-2}})$")
    axA.set_title("Areal emitted charge (converged)")

    uniform_T = not callable(result.config.cathode_temperature_K)
    if uniform_T:
        T_grid_K_masked = np.where(inside_disk, result.temperature_K, np.nan)
        T_val_K = float(np.nanmean(T_grid_K_masked)) if np.any(inside_disk) else float("nan")
        axB.add_patch(plt.Circle(
            (0.0, 0.0), cathode_radius_mm, facecolor=plt.get_cmap("inferno")(0.55),
            edgecolor="none", zorder=1,
        ))
        axB.set_title(rf"Cathode temperature profile (uniform, $T={T_val_K:.0f}\,$K)")
    else:
        T_grid_K = np.where(inside_disk, result.temperature_K, np.nan)
        im1 = axB.pcolormesh(x_mm_grid, y_mm_grid, T_grid_K.T, cmap="inferno", shading="nearest")
        fig.colorbar(im1, ax=axB, label="T (K)")
        axB.set_title("Cathode temperature profile")

    E_peak_MVpm = np.max(np.abs(result.E_total_history_Vpm[-1]), axis=-1) * 1.0e-6
    E_peak_MVpm = np.where(inside_disk, E_peak_MVpm, np.nan)
    im2 = axC.pcolormesh(x_mm_grid, y_mm_grid, E_peak_MVpm.T, cmap="magma", shading="nearest")
    fig.colorbar(im2, ax=axC, label=r"peak $|E_{\mathrm{total}}|\,(\mathrm{MV\,m^{-1}})$")
    axC.set_title("Cathode field profile (converged)")

    view_lim = cathode_radius_mm * 1.12
    for ax in (axA, axB, axC):
        add_cathode_boundary_circle(ax, cathode_radius_mm)
        ax.set_xlim(-view_lim, view_lim)
        ax.set_ylim(-view_lim, view_lim)
        ax.set_xlabel(r"$x\,(\mathrm{mm})$")
        ax.set_ylabel(r"$y\,(\mathrm{mm})$")
        ax.set_aspect("equal")

    plt.show()
