"""Figure B -- configured macropulse current and heating evolution (implementation plan Sec. 8,
12, addendum Sec. 19.2 -- "one-way" `coupling_level=L2_one_way` macropulse study plotting).

Scope: pure Python/matplotlib, NO RF-Track dependency. Operates on a
`rf_gun.studies.back_bombardment_macropulse.BackBombardmentMacropulseStudy` (`study.current_history`,
`study.thermal_result`, `study.macropulse_heat_source`, `study.comparison`, `study.comsol_result`).

The single most important behavioral requirement in this module (plan Sec. 8.2/12, addendum
Sec. 19.2's explicit user decision, restated in `rf_gun.comsol_io`'s own module docstring): when
`study.comparison.comsol_available is False` (i.e. `study.comsol_result is None`), NO COMSOL curve,
dashed line, or placeholder for one is ever drawn -- the temperature panel's title/legend say
"Python only, COMSOL unavailable" instead. `tests/test_plot_back_bombardment_macropulse.py` tests
this explicitly via the `fig.macropulse_comsol_curves_plotted` flag this function sets (see
`plot_back_bombardment_macropulse`'s own docstring).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..macropulse import evaluate_rf_envelope
from .style import (
    DEFAULT_PLOT_STYLE,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_NEUTRAL,
    add_cathode_boundary_circle,
    get_default_density_cmap,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import-order/circularity concerns
    from ..studies.back_bombardment_macropulse import BackBombardmentMacropulseStudy

__all__ = ["plot_back_bombardment_macropulse"]


def _panel_currents(ax, current_history) -> None:
    """Figure B, panel 1: `I_emit`, `I_return`, `I_transmitted`, `I_useful` vs. macro time (plan
    Sec. 12, item 1)."""
    t_us = np.asarray(current_history.t_s, dtype=float) * 1.0e6
    ax.plot(t_us, current_history.I_emit_A * 1.0e3, color=COLOR_PRIMARY, label=r"$I_{\rm emit}$")
    ax.plot(t_us, current_history.I_return_A * 1.0e3, color=COLOR_SECONDARY, label=r"$I_{\rm return}$")
    ax.plot(t_us, current_history.I_transmitted_A * 1.0e3, color="tab:green", label=r"$I_{\rm transmitted}$")
    ax.plot(t_us, current_history.I_useful_A * 1.0e3, color="black", ls="--", lw=1.2, label=r"$I_{\rm useful}$")
    ax.set_xlabel(r"$t\,(\mathrm{\mu s})$")
    ax.set_ylabel(r"$I\,(\mathrm{mA})$")
    ax.set_title("Macropulse current components")
    ax.legend(fontsize=6.5, loc="best")
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.02, r"$I_{\rm useful} \equiv I_{\rm transmitted}$ (explicit definition, plan Sec. 8.5)",
        transform=ax.transAxes, fontsize=5.3, va="bottom",
    )


def _panel_current_density(ax, current_history, config) -> None:
    """Figure B, panel 2: `J_emit_mean_A_m2` vs. macro time (plan Sec. 12, item 2).

    `MacropulseCurrentHistory` (`rf_gun.macropulse`) does not itself carry a current-density field
    -- it is derived here from `I_emit_A / flat_area_m2`, the same coarse flat-face-average
    convention `rf_gun.studies.back_bombardment_macropulse.write_back_bombardment_macropulse_h5`
    already uses for its own `/current/J_emit_mean_A_m2` dataset (reused verbatim, not
    reinvented). This is a single scalar spatial average, NOT a `J(x,y,t)` map.
    """
    flat_area_m2 = float(config.geometry.flat_area_mm2) * 1.0e-6
    if flat_area_m2 > 0.0:
        J_emit_mean_A_m2 = current_history.I_emit_A / flat_area_m2
    else:
        J_emit_mean_A_m2 = np.full_like(current_history.I_emit_A, np.nan)

    t_us = np.asarray(current_history.t_s, dtype=float) * 1.0e6
    ax.plot(t_us, J_emit_mean_A_m2, color=COLOR_PRIMARY)
    ax.set_xlabel(r"$t\,(\mathrm{\mu s})$")
    ax.set_ylabel(r"$J_{\rm emit,mean}\,(\mathrm{A/m^2})$")
    ax.set_title("Flat-face-average emitted current density")
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.02, r"$J_{\rm emit,mean} = I_{\rm emit}/A_{\rm flat}$ (coarse spatial average, NOT a $J(x,y,t)$ map)",
        transform=ax.transAxes, fontsize=5.3, va="bottom",
    )


def _panel_bb_power(ax, study, config) -> None:
    """Figure B, panel 3: incident and deposited BB power (split into flat/bevel; holder is
    always zero in BB0), overlaid with the RF/cavity envelope (plan Sec. 12, item 3).

    Deposited power is `study.macropulse_heat_source.q_layer_W` (already the macropulse-scaled,
    envelope-applied power series, plan Sec. 8.1/8.2) summed over `(x, y, layer)` at each time
    bin. Incident power has no macropulse-scaled spatial tensor of its own (BB0 only tracks
    per-cell DEPOSITED energy exactly -- see `rf_gun.back_bombardment_deposition`'s module
    docstring); it is reconstructed here the same way `rf_gun.macropulse.build_macropulse_heat_source`
    scales deposited energy: `study.heat_source.total_incident_energy_J * rf_frequency_Hz *
    envelope(t)`.
    """
    mhs = study.macropulse_heat_source
    geometry = config.geometry
    t_grid_edges_s = np.asarray(mhs.t_grid_s, dtype=float)
    t_centers_s = 0.5 * (t_grid_edges_s[:-1] + t_grid_edges_s[1:])
    envelope = evaluate_rf_envelope(t_centers_s, config.macropulse)

    q_layer_W = np.asarray(mhs.q_layer_W, dtype=float)
    P_dep_total_W = np.sum(q_layer_W, axis=(0, 1, 2))

    x_mm = np.asarray(mhs.x_centers_m, dtype=float) * 1.0e3
    y_mm = np.asarray(mhs.y_centers_m, dtype=float) * 1.0e3
    r_mm = np.hypot(x_mm[:, np.newaxis], y_mm[np.newaxis, :])
    footprint = np.asarray(mhs.cathode_footprint_mask, dtype=bool)
    flat_mask = footprint & (r_mm <= geometry.flat_radius_mm)
    bevel_mask = footprint & (r_mm > geometry.flat_radius_mm)
    P_dep_flat_W = np.sum(q_layer_W[flat_mask, :, :], axis=(0, 1)) if np.any(flat_mask) else np.zeros_like(P_dep_total_W)
    P_dep_bevel_W = np.sum(q_layer_W[bevel_mask, :, :], axis=(0, 1)) if np.any(bevel_mask) else np.zeros_like(P_dep_total_W)

    rf_frequency_Hz = float(study.study_input.events.rf_frequency_Hz)
    P_inc_total_W = float(study.heat_source.total_incident_energy_J) * rf_frequency_Hz * envelope

    t_us = t_centers_s * 1.0e6
    ax.plot(t_us, P_inc_total_W, color=COLOR_NEUTRAL, label=r"$P_{\rm incident}$ (all LaB6)")
    ax.plot(t_us, P_dep_total_W, color=COLOR_SECONDARY, label=r"$P_{\rm deposited}$ (total)")
    ax.plot(t_us, P_dep_flat_W, color=COLOR_PRIMARY, ls="--", lw=1.0, label=r"$P_{\rm deposited}$ (flat)")
    ax.plot(t_us, P_dep_bevel_W, color="tab:orange", ls="--", lw=1.0, label=r"$P_{\rm deposited}$ (bevel)")

    ax_env = ax.twinx()
    ax_env.plot(t_us, envelope, color="black", ls=":", lw=1.0, label="RF envelope")
    ax_env.set_ylabel("envelope (0-1)")
    ax_env.set_ylim(-0.05, 1.15)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_env.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=5.5, loc="upper right")
    ax.set_xlabel(r"$t\,(\mathrm{\mu s})$")
    ax.set_ylabel(r"$P\,(\mathrm{W})$")
    ax.set_title("BB power (incident vs. deposited) and RF envelope")
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.02, "Holder power is always 0 here: BB0 excludes non-LaB6 events entirely.",
        transform=ax.transAxes, fontsize=5.0, va="bottom",
    )


def _panel_temperature(ax, study) -> bool:
    """Figure B, panel 4: `T_center`, area average, bevel, `T_max` -- solid Python curves; dashed
    COMSOL curves ONLY if `study.comparison.comsol_available` (plan Sec. 12, item 4; Sec. 8.2's
    non-negotiable "the figure omits those curves and labels the backend as Python only" rule).

    Returns `True` if COMSOL curves were actually drawn, `False` otherwise -- the caller records
    this on the returned `Figure` (`fig.macropulse_comsol_curves_plotted`) so a test can check the
    no-COMSOL behavior deterministically without having to parse rendered pixels.
    """
    tr = study.thermal_result
    t_us = np.asarray(tr.t_grid_s, dtype=float) * 1.0e6
    ax.plot(t_us, tr.T_center_t, color=COLOR_PRIMARY, label=r"$T_{\rm center}$ (Python)")
    ax.plot(t_us, tr.T_area_average_t, color=COLOR_SECONDARY, label=r"$T_{\rm avg}$ (Python)")
    ax.plot(t_us, tr.T_bevel_mean_t, color="tab:green", label=r"$T_{\rm bevel}$ (Python)")
    ax.plot(t_us, tr.T_max_t, color="black", label=r"$T_{\rm max}$ (Python)")

    comsol_curves_plotted = False
    comparison = study.comparison
    comsol_result = study.comsol_result
    if comparison.comsol_available and comsol_result is not None:
        t_c_us = np.asarray(comsol_result.time_s, dtype=float) * 1.0e6
        ax.plot(t_c_us, comsol_result.T_center_K, color=COLOR_PRIMARY, ls="--", label=r"$T_{\rm center}$ (COMSOL)")
        ax.plot(t_c_us, comsol_result.T_area_average_K, color=COLOR_SECONDARY, ls="--", label=r"$T_{\rm avg}$ (COMSOL)")
        ax.plot(t_c_us, comsol_result.T_bevel_K, color="tab:green", ls="--", label=r"$T_{\rm bevel}$ (COMSOL)")
        ax.plot(t_c_us, comsol_result.T_max_K, color="black", ls="--", label=r"$T_{\rm max}$ (COMSOL)")
        comsol_curves_plotted = True
        ax.set_title("Cathode temperature: Python (solid) vs. COMSOL (dashed)")
    else:
        # Plan Sec. 8.2's non-negotiable rule: no COMSOL curve, dashed line, or placeholder for
        # one is ever drawn here -- the title/legend say so explicitly instead.
        ax.set_title("Cathode temperature -- Python only, COMSOL unavailable")

    ax.set_xlabel(r"$t\,(\mathrm{\mu s})$")
    ax.set_ylabel(r"$T\,(\mathrm{K})$")
    ax.legend(fontsize=6.0, loc="best")
    ax.grid(alpha=0.3)
    return comsol_curves_plotted


def _panel_surface_snapshots(fig, gs_cell, study) -> list:
    """Figure B, panel 5: asymmetric `T_surface(x, y)` maps at `t/tau_macro = 0, 0.5, 1` (plan
    Sec. 12, item 5), as a 1x3 nested sub-grid within this one subplot slot -- documented choice
    (the plan leaves the exact rendering to the implementer): three small side-by-side panels with
    a common color scale, cathode-zone outlines, and hotspot markers, sharing one colorbar.

    Returns the list of the three snapshot `Axes` (for `fig.bb_macropulse_panels`'s bookkeeping).
    """
    tr = study.thermal_result
    geometry = study.config.geometry
    duration_s = float(study.config.macropulse.duration_s)
    target_fracs = (0.0, 0.5, 1.0)
    t_grid_s = np.asarray(tr.t_grid_s, dtype=float)
    idxs = [int(np.argmin(np.abs(t_grid_s - frac * duration_s))) for frac in target_fracs]

    T_snapshots = tr.T_surface_xyt[:, :, idxs]
    finite = T_snapshots[np.isfinite(T_snapshots)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0

    sub_gs = gs_cell.subgridspec(1, 3, wspace=0.15)
    x_mm = np.asarray(tr.x_centers_m, dtype=float) * 1.0e3
    y_mm = np.asarray(tr.y_centers_m, dtype=float) * 1.0e3
    mask = np.asarray(tr.cathode_footprint_mask, dtype=bool)
    cmap = get_default_density_cmap()

    snapshot_axes = []
    im = None
    for j, (frac, idx) in enumerate(zip(target_fracs, idxs)):
        axj = fig.add_subplot(sub_gs[0, j])
        T_map = np.where(mask, tr.T_surface_xyt[:, :, idx], np.nan)
        im = axj.pcolormesh(x_mm, y_mm, T_map.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
        add_cathode_boundary_circle(axj, geometry.flat_radius_mm, color="black", ls="--")
        add_cathode_boundary_circle(axj, geometry.bevel_outer_radius_mm, color="black", ls="-")
        hx, hy = tr.hotspot_centroid_xy_t[idx]
        if np.isfinite(hx) and np.isfinite(hy):
            axj.plot(hx * 1.0e3, hy * 1.0e3, marker="*", ms=10, color="cyan", mec="black", mew=0.8)
        axj.set_title(f"$t/\\tau={frac:.2g}$\n($t$={t_grid_s[idx] * 1.0e6:.2f} $\\mu$s)", fontsize=7)
        axj.set_aspect("equal")
        axj.set_xticks([])
        axj.set_yticks([])
        snapshot_axes.append(axj)

    if im is not None:
        fig.colorbar(im, ax=snapshot_axes, label=r"$T_{\rm surface}\,(\mathrm{K})$", fraction=0.05, pad=0.03)
    snapshot_axes[0].set_ylabel(r"$T_{\rm surface}(x,y)$")
    return snapshot_axes


def _panel_validation_metrics(ax, study) -> None:
    """Figure B, panel 6: energy residual and validation metrics as a text panel (plan Sec. 12,
    item 6) -- `Delta_T`, heating slope, current rise, and an explicit note that keyframe/hotspot-
    displacement metrics do not apply to this `coupling_level=L2_one_way` pass (no adaptive
    keyframes, no COMSOL comparison unless supplied). No placeholder values are fabricated for
    metrics that genuinely do not apply.
    """
    ax.axis("off")
    tr = study.thermal_result
    ch = study.current_history
    config = study.config
    comparison = study.comparison
    duration_s = float(config.macropulse.duration_s)

    Delta_T_max_K = float(tr.T_max_t[-1] - tr.T_max_t[0])
    Delta_T_center_K = float(tr.T_center_t[-1] - tr.T_center_t[0])
    heating_slope_K_per_us = Delta_T_max_K / (duration_s * 1.0e6) if duration_s > 0.0 else float("nan")
    current_rise_mA = float(ch.I_emit_A[-1] - ch.I_emit_A[0]) * 1.0e3

    if comparison.comsol_available and comparison.hotspot_displacement_m is not None:
        hotspot_line = f"hotspot displacement   = {comparison.hotspot_displacement_m:.4g} m (Python vs. COMSOL)"
    elif comparison.comsol_available:
        hotspot_line = "hotspot displacement   = not available (no matching full surface field on both sides)"
    else:
        hotspot_line = "hotspot displacement   = not applicable (COMSOL unavailable in this pass)"

    lines = [
        f"energy_residual_normalized = {tr.energy_residual_normalized:.3e}",
        f"Delta_T (T_max, end-start)  = {Delta_T_max_K:.4g} K",
        f"Delta_T (T_center, end-start) = {Delta_T_center_K:.4g} K",
        f"heating slope (T_max)       = {heating_slope_K_per_us:.4g} K/us",
        f"current rise (I_emit)       = {current_rise_mA:.4g} mA",
        "",
        f"coupling_level = {config.coupling.level!r}",
        "keyframe locations     = not applicable (single representative-period source; "
        "no adaptive keyframes in this L2_one_way pass)",
        hotspot_line,
        "UH benchmark bands     = not evaluated here (separate 5 us benchmark configuration, "
        "plan Sec. 14)",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=6.6, va="top", family="monospace")
    ax.set_title("Energy residual and validation metrics")


def plot_back_bombardment_macropulse(study: "BackBombardmentMacropulseStudy", *, style=None):
    """Figure B (plan Sec. 12): configured macropulse current and heating evolution -- a 2x3
    layout on a common macro-time axis, built from `study.current_history`, `study.thermal_result`,
    `study.macropulse_heat_source`, and `study.comparison`/`study.comsol_result`.

    See each `_panel_*` helper above for exact panel content and documented design decisions.

    `fig.bb_macropulse_panels` is set to the list of the six primary panel objects (panel 5 is
    itself a list of the three snapshot `Axes`, since it renders as a nested 1x3 sub-grid) -- for
    the same panel-count-testability reason as
    `rf_gun.plotting.back_bombardment.plot_back_bombardment_source_qualification`'s
    `fig.bb_source_qualification_panels`.

    `fig.macropulse_comsol_curves_plotted` is `True` iff panel 4 actually drew COMSOL curves
    (`study.comparison.comsol_available and study.comsol_result is not None`), `False` otherwise --
    the deterministic hook `tests/test_plot_back_bombardment_macropulse.py` uses to verify plan
    Sec. 8.2's non-negotiable "no COMSOL, no fabricated curves" requirement without having to
    inspect rendered pixels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    style = DEFAULT_PLOT_STYLE if style is None else style  # noqa: F841 - kept for API symmetry
    config = study.config

    fig = plt.figure(figsize=(19.0, 11.5))
    gs = GridSpec(2, 3, figure=fig, wspace=0.55, hspace=0.6)

    ax1 = fig.add_subplot(gs[0, 0])
    _panel_currents(ax1, study.current_history)

    ax2 = fig.add_subplot(gs[0, 1])
    _panel_current_density(ax2, study.current_history, config)

    ax3 = fig.add_subplot(gs[0, 2])
    _panel_bb_power(ax3, study, config)

    ax4 = fig.add_subplot(gs[1, 0])
    comsol_curves_plotted = _panel_temperature(ax4, study)

    snapshot_axes = _panel_surface_snapshots(fig, gs[1, 1], study)

    ax6 = fig.add_subplot(gs[1, 2])
    _panel_validation_metrics(ax6, study)

    fig.bb_macropulse_panels = [ax1, ax2, ax3, ax4, snapshot_axes, ax6]
    fig.macropulse_comsol_curves_plotted = comsol_curves_plotted
    fig.suptitle(
        f"Back-bombardment macropulse study -- duration={config.macropulse.duration_s * 1.0e6:.3g} "
        f"$\\mu$s, coupling_level={config.coupling.level!r}, "
        f"comsol_available={study.comparison.comsol_available}",
        y=0.995,
    )
    return fig
