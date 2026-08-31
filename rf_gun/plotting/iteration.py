"""Emission Fields Iteration figures (implementation guide Sec. 12)."""
from __future__ import annotations

from typing import Any

import numpy as np


def plot_emission_iteration_convergence(result: Any):
    """Figure 1, 2x2 (guide Sec. 12): residuals, charge/current, field measures, and relaxation
    history versus outer iteration.

    Units chosen for readability at this project's typical scale: charge in pC (not nC), peak
    current in mA (not A), field corrections in kV/m (not V/m) -- e.g. a ~72 pC / ~530 mA /
    ~240 kV/m run reads far more legibly than 0.072 nC / 0.530 A / 2.4e5 V/m. The field-correction
    panel's title/legend state explicitly whether mirror charges were enabled for this run
    (`result.config.include_mirror`), since a mirror-off run's peak |E_mirror| is exactly zero by
    construction (extract_sc_and_mirror_from_snapshot returns E_sc_free unchanged when no mirror
    plane is configured) -- not a bug, but easy to misread as one without this label.
    """
    import matplotlib.pyplot as plt

    n_iter = len(result.eps_J_history)
    it = np.arange(n_iter)
    cfg = result.config
    mirror_on = bool(getattr(cfg, "include_mirror", False))
    bl_on = bool(getattr(cfg, "include_beam_loading", False))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    axA.semilogy(it, np.abs(result.eps_J_history), "o-", label=r"$\epsilon_J$")
    axA.semilogy(it, np.abs(result.eps_E_history), "s-", label=r"$\epsilon_E$")
    axA.semilogy(it, np.abs(result.eps_Q_history), "^-", label=r"$\epsilon_Q$")
    axA.axhline(cfg.current_tolerance, color="C0", ls="--", lw=0.9, alpha=0.6)
    axA.axhline(cfg.field_tolerance, color="C1", ls="--", lw=0.9, alpha=0.6)
    axA.axhline(cfg.charge_tolerance, color="C2", ls="--", lw=0.9, alpha=0.6)
    axA.set_ylabel("Residual")
    axA.set_title("Convergence residuals")
    axA.legend(frameon=False, fontsize=11)

    Q_pC = np.asarray(result.Q_history_C) * 1e12
    I_mA = np.asarray(result.I_peak_history_A) * 1e3
    axB.plot(it, Q_pC, "o-", color="tab:blue", label="Q (pC)")
    axB.set_ylabel("Emitted charge (pC)", color="tab:blue")
    axB.tick_params(axis="y", labelcolor="tab:blue")
    axB2 = axB.twinx()
    axB2.plot(it, I_mA, "s-", color="tab:red", label=r"$I_{\rm peak}$")
    axB2.set_ylabel("Peak current (mA)", color="tab:red")
    axB2.tick_params(axis="y", labelcolor="tab:red")
    axB.set_title("Charge and peak current")

    peak_sc_kVpm = [1e-3 * float(np.max(np.abs(E))) if np.size(E) else 0.0 for E in result.E_SC_history_Vpm]
    peak_mirror_kVpm = [1e-3 * float(np.max(np.abs(E))) if np.size(E) else 0.0 for E in result.E_mirror_history_Vpm]
    peak_bl_kVpm = [1e-3 * float(np.max(np.abs(E))) if np.size(E) else 0.0 for E in result.E_BL_history_Vpm]
    peak_total_minus_rf_kVpm = [
        1e-3 * float(np.max(np.abs(Et - Erf)))
        for Et, Erf in zip(result.E_total_history_Vpm, result.E_RF_history_Vpm)
    ]
    axC.plot(it, peak_sc_kVpm, "o-", label=r"peak $|E_{\rm SC,\,free\,space}|$")
    axC.plot(
        it, peak_mirror_kVpm, "s-",
        label=r"peak $|E_{\rm mirror}|$" + (" (mirror OFF -- always 0)" if not mirror_on else ""),
    )
    axC.plot(
        it, peak_bl_kVpm, "d-",
        label=r"peak $|E_{\rm BL}|$ (causal envelope)" + (" (BL OFF -- always 0)" if not bl_on else ""),
    )
    axC.plot(it, peak_total_minus_rf_kVpm, "^-", label=r"peak $|E_{\rm total}-E_{\rm RF}|$ (net correction)")
    axC.set_ylabel(r"$|E|\,(\mathrm{kV\,m^{-1}})$")
    axC.set_title(
        f"Cathode field corrections (mirror {'ON' if mirror_on else 'OFF'}, BL {'ON' if bl_on else 'OFF'})"
    )
    axC.legend(frameon=False, fontsize=9)

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

    cfg = result.config
    mirror_on = bool(getattr(cfg, "include_mirror", False))
    bl_on = bool(getattr(cfg, "include_beam_loading", False))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    E_RF = result.E_RF_history_Vpm[-1][x_index, y_index] * 1e-3
    E_SCf = result.E_SC_history_Vpm[-1][x_index, y_index] * 1e-3
    E_Mf = result.E_mirror_history_Vpm[-1][x_index, y_index] * 1e-3
    E_BLf = result.E_BL_history_Vpm[-1][x_index, y_index] * 1e-3
    E_totf = result.E_total_history_Vpm[-1][x_index, y_index] * 1e-3

    axA.plot(t_ns, E_RF, "-", color="k", lw=1.6, label=r"$E_{\rm RF}$")
    axA.plot(t_ns, E_SCf, "--", color="tab:blue", label=r"$E_{\rm SC}$ (final, free space)")
    mirror_label = r"$E_{\rm mirror}$ (final)" + (" -- mirror OFF, always 0" if not mirror_on else "")
    axA.plot(t_ns, E_Mf, "--", color="tab:orange", label=mirror_label)
    bl_label = r"$E_{\rm BL}$ (final)" + (" -- BL OFF, always 0" if not bl_on else "")
    axA.plot(t_ns, E_BLf, "--", color="tab:green", label=bl_label)
    axA.plot(t_ns, E_totf, "-", color="tab:red", lw=1.8, label=r"$E_{\rm total}$ (final)")
    axA.set_ylabel(r"$E_z\,(\mathrm{kV\,m^{-1}})$")
    axA.set_title(f"Signed fields at (x,y)=({x_mm:.3f}, {y_mm:.3f}) mm -- mirror {'ON' if mirror_on else 'OFF'}")
    axA.legend(frameon=False, fontsize=9)

    J0 = result.J_history_Apm2[0][x_index, y_index] * 1e-4
    Jf = result.J_history_Apm2[-1][x_index, y_index] * 1e-4
    axB.plot(t_ns, J0, "--", color="0.5", label=r"$J^{(0)}(t)$")
    axB.plot(t_ns, Jf, "-", color="tab:blue", label=r"$J^{(\rm final)}(t)$")
    axB.set_ylabel(r"$J\,(\mathrm{A\,cm^{-2}})$")
    axB.set_title("Initial vs final current density")
    axB.legend(frameon=False, fontsize=11)

    dt_s = result.fixed_sample["dt_s"]
    area_m2 = result.fixed_sample["grid"]["dA_mm2"] * 1e-6  # (n_x, n_y)
    I0 = np.sum(result.J_history_Apm2[0] * area_m2[:, :, None], axis=(0, 1))
    If = np.sum(result.J_history_Apm2[-1] * area_m2[:, :, None], axis=(0, 1))
    Q0 = np.cumsum(I0) * dt_s
    Qf = np.cumsum(If) * dt_s
    axC.plot(t_ns, Q0 * 1e12, "--", color="0.5", label="initial")
    axC.plot(t_ns, Qf * 1e12, "-", color="tab:blue", label="final")
    axC.set_ylabel("Cumulative charge (pC)")
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

    status = "converged" if result.converged else f"not converged ({result.failure_reason})"
    fig.suptitle(f"Emission Fields Iteration waveforms -- {status}")
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
    status_word = "converged" if result.converged else "final, NOT converged"

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)

    Q_areal_nC_mm2 = np.trapezoid(result.J_history_Apm2[-1], result.t_grid_s, axis=-1) * 1.0e3
    Q_areal_nC_mm2 = np.where(inside_disk, Q_areal_nC_mm2, np.nan)
    if np.any(inside_disk) and not np.any(np.isfinite(Q_areal_nC_mm2[inside_disk])):
        raise ValueError(
            "plot_emission_iteration_near_cathode: areal emitted charge is non-finite over the "
            "entire cathode disk -- this is an invalid iteration result, not a plottable (blank) "
            f"panel (result.converged={result.converged}, failure_reason={result.failure_reason!r})."
        )
    im0 = axA.pcolormesh(x_mm_grid, y_mm_grid, Q_areal_nC_mm2.T, cmap="viridis", shading="nearest")
    fig.colorbar(im0, ax=axA, label=r"$\int J\,dt\,(\mathrm{nC\,mm^{-2}})$")
    axA.set_title(f"Areal emitted charge ({status_word})")

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
    if np.any(inside_disk) and not np.any(np.isfinite(E_peak_MVpm[inside_disk])):
        raise ValueError(
            "plot_emission_iteration_near_cathode: peak cathode field is non-finite over the "
            "entire cathode disk -- this is an invalid iteration result, not a plottable (blank) "
            f"panel (result.converged={result.converged}, failure_reason={result.failure_reason!r})."
        )
    im2 = axC.pcolormesh(x_mm_grid, y_mm_grid, E_peak_MVpm.T, cmap="magma", shading="nearest")
    fig.colorbar(im2, ax=axC, label=r"peak $|E_{\mathrm{total}}|\,(\mathrm{MV\,m^{-1}})$")
    axC.set_title(f"Cathode field profile ({status_word})")

    view_lim = cathode_radius_mm * 1.12
    for ax in (axA, axB, axC):
        add_cathode_boundary_circle(ax, cathode_radius_mm)
        ax.set_xlim(-view_lim, view_lim)
        ax.set_ylim(-view_lim, view_lim)
        ax.set_xlabel(r"$x\,(\mathrm{mm})$")
        ax.set_ylabel(r"$y\,(\mathrm{mm})$")
        ax.set_aspect("equal")

    plt.show()


def plot_emission_iteration_submodel_comparison(cases: Any):
    """(1 + n_cases) x 2 panel comparing independently-converged Emission Fields Iteration runs
    of the *same* fixed source/cathode geometry under different self-consistency physics -- e.g.
    space-charge only vs. space-charge + cathode mirror -- rather than showing a single run's
    converged state (that's plot_emission_iteration_near_cathode, still available for the
    single-run case).

    `cases`: an ordered sequence of `(label, EmissionFieldIterationResult)` pairs, e.g.::

        plot_emission_iteration_submodel_comparison([
            ("Space charge only", result_sc_only),
            ("Space charge + mirror", result_sc_mirror),
        ])

    Row 0 (shared baseline, taken from `cases[0]`'s result -- identical across cases since the
    external RF field and the cathode temperature do not depend on which SC/mirror physics is
    switched on): the external RF field at the cathode at t=0 (iteration 0 is the RF-only
    baseline, before any SC/mirror update -- see run_emission_field_iteration's docstring), and
    the temperature profile.

    Row k>=1 (one per case): a genuinely spatially-resolved field-correction map --
    max_t |E_total(x,y,t) - E_RF(x,y,t)| across the whole emission window, not just the single
    scalar "peak" number in plot_emission_iteration_convergence's panel C -- and that case's
    converged areal emitted-charge map (same quantity as
    plot_emission_iteration_near_cathode's panel A).

    Deliberately does *not* include a "space charge + mirror + beam loading" row: this reduced-
    cost cathode iteration cannot include beam loading at all right now (see
    EmissionFieldIterationConfig.field_probe_method's module docstring, and
    run_emission_field_iteration, which raises rather than silently ignoring
    include_beam_loading=True) -- showing such a row here would misleadingly look identical to
    the SC+mirror row while claiming to be a different physics case. See UPGRADE_PLAN.md for the
    planned causal beam-loading model that would eventually justify a third row.
    """
    import matplotlib.pyplot as plt

    from .style import add_cathode_boundary_circle

    cases = list(cases)
    if not cases:
        raise ValueError("plot_emission_iteration_submodel_comparison needs at least one case")

    label0, result0 = cases[0]
    x_mm_grid = result0.x_grid_m * 1e3
    y_mm_grid = result0.y_grid_m * 1e3
    inside_disk = result0.fixed_sample["grid"]["inside_disk"]
    cathode_radius_mm = float(result0.cathode_radius_mm)
    view_lim = cathode_radius_mm * 1.12

    n_rows = 1 + len(cases)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10.5, 4.0 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    # Row 0: shared external-field/temperature baseline.
    E_rf_t0_kVpm = result0.E_RF_history_Vpm[0][:, :, 0] * 1e-3
    E_rf_t0_kVpm = np.where(inside_disk, E_rf_t0_kVpm, np.nan)
    ax_rf = axes[0, 0]
    im_rf = ax_rf.pcolormesh(x_mm_grid, y_mm_grid, E_rf_t0_kVpm.T, cmap="RdBu_r", shading="nearest")
    fig.colorbar(im_rf, ax=ax_rf, label=r"$E_{\rm RF,z}(t{=}0)\,(\mathrm{kV\,m^{-1}})$")
    ax_rf.set_title("External RF field on cathode (t=0)")

    ax_T = axes[0, 1]
    uniform_T = not callable(result0.config.cathode_temperature_K)
    if uniform_T:
        T_grid_K_masked = np.where(inside_disk, result0.temperature_K, np.nan)
        T_val_K = float(np.nanmean(T_grid_K_masked)) if np.any(inside_disk) else float("nan")
        ax_T.add_patch(plt.Circle(
            (0.0, 0.0), cathode_radius_mm, facecolor=plt.get_cmap("inferno")(0.55),
            edgecolor="none", zorder=1,
        ))
        ax_T.set_title(rf"Cathode temperature (uniform, $T={T_val_K:.0f}\,$K)")
    else:
        T_grid_K = np.where(inside_disk, result0.temperature_K, np.nan)
        im_T = ax_T.pcolormesh(x_mm_grid, y_mm_grid, T_grid_K.T, cmap="inferno", shading="nearest")
        fig.colorbar(im_T, ax=ax_T, label="T (K)")
        ax_T.set_title("Cathode temperature profile")

    # Rows 1..n: one per case, a spatial field-correction map + converged areal emitted charge.
    for row, (label, result) in enumerate(cases, start=1):
        case_status = "converged" if result.converged else "NOT converged"
        E_total = np.asarray(result.E_total_history_Vpm[-1])
        E_rf = np.asarray(result.E_RF_history_Vpm[-1])
        correction_peak_kVpm = np.max(np.abs(E_total - E_rf), axis=-1) * 1e-3
        correction_peak_kVpm = np.where(inside_disk, correction_peak_kVpm, np.nan)
        if np.any(inside_disk) and not np.any(np.isfinite(correction_peak_kVpm[inside_disk])):
            raise ValueError(
                f"plot_emission_iteration_submodel_comparison: case {label!r}'s field correction "
                f"is non-finite over the entire cathode disk (converged={result.converged}, "
                f"failure_reason={result.failure_reason!r}) -- refusing to plot a blank panel."
            )

        ax_corr = axes[row, 0]
        im_corr = ax_corr.pcolormesh(x_mm_grid, y_mm_grid, correction_peak_kVpm.T, cmap="magma", shading="nearest")
        fig.colorbar(im_corr, ax=ax_corr, label=r"$\max_t|E_{\rm total}-E_{\rm RF}|\,(\mathrm{kV\,m^{-1}})$")
        ax_corr.set_title(f"{label} ({case_status}): field correction vs. RF-only")

        Q_areal_pC_mm2 = np.trapezoid(result.J_history_Apm2[-1], result.t_grid_s, axis=-1) * 1.0e6
        Q_areal_pC_mm2 = np.where(inside_disk, Q_areal_pC_mm2, np.nan)
        if np.any(inside_disk) and not np.any(np.isfinite(Q_areal_pC_mm2[inside_disk])):
            raise ValueError(
                f"plot_emission_iteration_submodel_comparison: case {label!r}'s areal emitted "
                f"charge is non-finite over the entire cathode disk (converged={result.converged}, "
                f"failure_reason={result.failure_reason!r}) -- refusing to plot a blank panel."
            )
        ax_emit = axes[row, 1]
        im_emit = ax_emit.pcolormesh(x_mm_grid, y_mm_grid, Q_areal_pC_mm2.T, cmap="viridis", shading="nearest")
        fig.colorbar(im_emit, ax=ax_emit, label=r"$\int J\,dt\,(\mathrm{pC\,mm^{-2}})$")
        ax_emit.set_title(f"{label} ({case_status}): emission profile")

    for ax in axes.ravel():
        add_cathode_boundary_circle(ax, cathode_radius_mm)
        ax.set_xlim(-view_lim, view_lim)
        ax.set_ylim(-view_lim, view_lim)
        ax.set_xlabel(r"$x\,(\mathrm{mm})$")
        ax.set_ylabel(r"$y\,(\mathrm{mm})$")
        ax.set_aspect("equal")

    fig.suptitle("Emission Fields Iteration -- submodel comparison")

    plt.show()
