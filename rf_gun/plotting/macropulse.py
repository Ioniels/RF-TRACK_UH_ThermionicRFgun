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
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import-order/circularity concerns
    from ..studies.back_bombardment_macropulse import BackBombardmentMacropulseStudy

__all__ = ["plot_back_bombardment_macropulse", "print_back_bombardment_macropulse_summary"]


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
    ax.set_title("Current components")
    ax.legend(fontsize=6.5, loc="best")
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.02, r"$I_{\rm useful} \equiv I_{\rm transmitted}$ (explicit definition, plan Sec. 8.5)",
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
    ax.set_title("BB power vs. RF envelope")
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
        ax.set_title("Temperature: Python vs. COMSOL")
    else:
        # Plan Sec. 8.2's non-negotiable rule: no COMSOL curve, dashed line, or placeholder for
        # one is ever drawn here -- the title/legend say so explicitly instead.
        ax.set_title("Temperature -- Python only, COMSOL unavailable")

    ax.set_xlabel(r"$t\,(\mathrm{\mu s})$")
    ax.set_ylabel(r"$T\,(\mathrm{K})$")
    ax.legend(fontsize=6.0, loc="best")
    ax.grid(alpha=0.3)
    return comsol_curves_plotted


def _surface_snapshot_targets(study):
    """`(target_fracs, idxs, vmin, vmax)` shared by every snapshot panel: the three `t/tau_macro`
    sample indices and one common color scale across all three."""
    tr = study.thermal_result
    duration_s = float(study.config.macropulse.duration_s)
    target_fracs = (0.0, 0.5, 1.0)
    t_grid_s = np.asarray(tr.t_grid_s, dtype=float)
    idxs = [int(np.argmin(np.abs(t_grid_s - frac * duration_s))) for frac in target_fracs]
    T_snapshots = tr.T_surface_xyt[:, :, idxs]
    finite = T_snapshots[np.isfinite(T_snapshots)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    return target_fracs, idxs, vmin, vmax


def _panel_surface_snapshot(ax, study, frac: float, idx: int, vmin: float, vmax: float, show_ylabel: bool):
    """One `T_surface(x, y)` snapshot at `t/tau_macro=frac`, with cathode-zone outlines and a
    hotspot marker. Returns the `pcolormesh` handle (for a shared colorbar)."""
    tr = study.thermal_result
    geometry = study.config.geometry
    x_mm = np.asarray(tr.x_centers_m, dtype=float) * 1.0e3
    y_mm = np.asarray(tr.y_centers_m, dtype=float) * 1.0e3
    mask = np.asarray(tr.cathode_footprint_mask, dtype=bool)
    t_grid_s = np.asarray(tr.t_grid_s, dtype=float)

    T_map = np.where(mask, tr.T_surface_xyt[:, :, idx], np.nan)
    # Plain plasma, not get_default_density_cmap(): that one renders vmin as transparent white
    # (correct for "zero particles here", wrong for a legitimate temperature -- the t=0 snapshot
    # is uniformly at vmin and would otherwise vanish entirely).
    im = ax.pcolormesh(x_mm, y_mm, T_map.T, cmap="plasma", vmin=vmin, vmax=vmax, shading="nearest")
    add_cathode_boundary_circle(ax, geometry.flat_radius_mm, color="black", ls="--")
    add_cathode_boundary_circle(ax, geometry.bevel_outer_radius_mm, color="black", ls="-")
    hx, hy = tr.hotspot_centroid_xy_t[idx]
    if np.isfinite(hx) and np.isfinite(hy):
        ax.plot(hx * 1.0e3, hy * 1.0e3, marker="*", ms=12, color="cyan", mec="black", mew=0.8)
    ax.set_title(rf"$t/\tau={frac:.2g}$ ($t$={t_grid_s[idx] * 1.0e6:.2f} $\mu$s)")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x\,(\mathrm{mm})$")
    if show_ylabel:
        ax.set_ylabel(r"$y\,(\mathrm{mm})$")
    return im


def print_back_bombardment_macropulse_summary(study) -> None:
    """Printed counterpart to `plot_back_bombardment_macropulse`'s figure: energy residual,
    temperature rise/slope, current rise, and coupling/benchmark applicability notes."""
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
        hotspot_line = f"hotspot displacement={comparison.hotspot_displacement_m:.4g} m (Python vs. COMSOL)"
    elif comparison.comsol_available:
        hotspot_line = "hotspot displacement: not available (no matching full surface field on both sides)"
    else:
        hotspot_line = "hotspot displacement: not applicable (COMSOL unavailable in this pass)"

    print(
        f"Back-bombardment macropulse study -- duration={duration_s * 1.0e6:.3g} us, "
        f"coupling_level={config.coupling.level!r}, comsol_available={comparison.comsol_available}"
    )
    print(f"  energy_residual_normalized={tr.energy_residual_normalized:.3e}")
    print(
        f"  Delta_T: T_max={Delta_T_max_K:.4g} K, T_center={Delta_T_center_K:.4g} K "
        f"(end-start) | heating slope (T_max)={heating_slope_K_per_us:.4g} K/us"
    )
    print(f"  current rise (I_emit)={current_rise_mA:.4g} mA")
    print(f"  {hotspot_line}")
    print("  keyframe locations: not applicable (single representative-period source, no adaptive keyframes)")


def plot_back_bombardment_macropulse(study: "BackBombardmentMacropulseStudy", *, style=None):
    """Configured macropulse current and heating evolution, 2x3: row 1 is current components, BB
    power, and cathode temperature vs. macro time; row 2 is the surface-temperature snapshot at
    `t/tau_macro = 0, 0.5, 1`, each a full-size panel sharing one color scale. Energy-residual and
    validation numbers live in `print_back_bombardment_macropulse_summary`, not in the axes.

    `fig.macropulse_comsol_curves_plotted` is `True` iff the temperature panel actually drew COMSOL
    curves (`study.comparison.comsol_available and study.comsol_result is not None`) -- the
    deterministic hook `tests/test_plot_back_bombardment_macropulse.py` uses to verify no COMSOL
    curve is ever fabricated when no COMSOL result was supplied.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    style = DEFAULT_PLOT_STYLE if style is None else style  # noqa: F841 - kept for API symmetry
    config = study.config

    fig = plt.figure(figsize=(15.0, 9.0))
    gs = GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    _panel_currents(ax1, study.current_history)

    ax2 = fig.add_subplot(gs[0, 1])
    _panel_bb_power(ax2, study, config)

    ax3 = fig.add_subplot(gs[0, 2])
    comsol_curves_plotted = _panel_temperature(ax3, study)

    target_fracs, idxs, vmin, vmax = _surface_snapshot_targets(study)
    snapshot_axes = [fig.add_subplot(gs[1, j]) for j in range(3)]
    im = None
    for j, (ax, frac, idx) in enumerate(zip(snapshot_axes, target_fracs, idxs)):
        im = _panel_surface_snapshot(ax, study, frac, idx, vmin, vmax, show_ylabel=(j == 0))
    if im is not None:
        fig.colorbar(im, ax=snapshot_axes, label=r"$T_{\rm surface}\,(\mathrm{K})$", fraction=0.035, pad=0.02)

    fig.bb_macropulse_panels = [ax1, ax2, ax3] + snapshot_axes
    fig.macropulse_comsol_curves_plotted = comsol_curves_plotted
    return fig
