"""Back-bombardment figures: where, how energetically, and when do backward-turning particles
re-hit the cathode plane (z=0)? See `rf_gun.back_bombardment` for the underlying reconstruction
and the physics assumptions behind it (field-free drift behind the cathode, momentum conservation).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np

from ..cathode_geometry import (
    SURFACE_CATHODE_BEVEL,
    SURFACE_CATHODE_FLAT,
    SURFACE_CATHODE_SIDE,
    SURFACE_CAVITY_WALL,
    SURFACE_HOLDER,
    SURFACE_UNKNOWN,
)
from ..back_bombardment_events import screen_trajectory
from ..constants import q_e
from .style import DEFAULT_PLOT_STYLE, add_cathode_boundary_circle, get_default_density_cmap

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import-order/circularity concerns
    from ..studies.back_bombardment_macropulse import BackBombardmentMacropulseStudy


def _robust_range(*arrays: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0, pad_frac: float = 0.08):
    """`(lo, hi)` axis range from the `lo_pct`-`hi_pct` percentile of the concatenated, finite
    values in `arrays`, padded by `pad_frac` of the span -- robust to the handful of extreme
    ballistic-reconstruction outliers (near-zero-Pz stragglers, see `rf_gun.back_bombardment`'s
    module docstring) that would otherwise single-handedly dictate a plain min/max axis limit and
    squash every other point into an unreadable sliver. Returns `None` if there's no finite data
    (caller should skip setting limits in that case, falling back to matplotlib's own autoscale).
    """
    vals = np.concatenate([np.asarray(a, dtype=float).reshape(-1) for a in arrays]) if arrays else np.asarray([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if hi <= lo:
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi <= lo:
            pad = max(abs(lo), 1.0) * pad_frac
            return lo - pad, hi + pad
    span = hi - lo
    pad = span * pad_frac
    return float(lo - pad), float(hi + pad)


def _weighted_range(values: np.ndarray, weights: np.ndarray, *, lo_pct: float, hi_pct: float, pad_frac: float):
    """Like `_robust_range`, but the percentile is taken over `values` weighted by `weights`
    (rather than treating every sample as equally important) -- used for the heat-flux-vs-time
    figure, where a near-zero-Pz straggler's reconstructed re-hit time can be extreme even though
    it carries almost none of the deposited energy (see `rf_gun.back_bombardment`'s module
    docstring); weighting by energy keeps the range tied to where the energy actually lands rather
    than to the mere presence of an outlier particle.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    finite = np.isfinite(v) & np.isfinite(w) & (w >= 0.0)
    v, w = v[finite], w[finite]
    if v.size == 0 or float(np.sum(w)) <= 0.0:
        return _robust_range(values, lo_pct=lo_pct, hi_pct=hi_pct, pad_frac=pad_frac)
    order = np.argsort(v)
    v_sorted, w_sorted = v[order], w[order]
    cum_w = np.cumsum(w_sorted) / np.sum(w_sorted)
    lo = float(np.interp(lo_pct / 100.0, cum_w, v_sorted))
    hi = float(np.interp(hi_pct / 100.0, cum_w, v_sorted))
    if hi <= lo:
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi <= lo:
            pad = max(abs(lo), 1.0) * pad_frac
            return lo - pad, hi + pad
    span = hi - lo
    pad = span * pad_frac
    return lo - pad, hi + pad


def _sci(value: float, precision: int = 2) -> str:
    """`1.08e-04` -> `1.08\\times10^{-4}` -- proper mathtext scientific notation for titles.

    `value` non-finite (NaN/Inf) renders as a plain "n/a" -- `f"{nan:.2e}"` formats to the bare
    string `"nan"` (no `"e"` to split on), which would otherwise raise here.
    """
    if not np.isfinite(value):
        return r"\mathrm{n/a}"
    mantissa, exp = f"{value:.{precision}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exp)}}}"


_ZONE_STYLE: Dict[int, Dict[str, str]] = {
    int(SURFACE_CATHODE_FLAT): {"color": "tab:blue", "marker": "o", "label": "flat"},
    int(SURFACE_CATHODE_BEVEL): {"color": "tab:orange", "marker": "^", "label": "bevel"},
    int(SURFACE_CATHODE_SIDE): {"color": "tab:green", "marker": "s", "label": "side"},
    int(SURFACE_HOLDER): {"color": "tab:red", "marker": "x", "label": "holder"},
    int(SURFACE_CAVITY_WALL): {"color": "tab:purple", "marker": "d", "label": "cavity wall"},
    int(SURFACE_UNKNOWN): {"color": "gray", "marker": ".", "label": "missed cathode"},
}


def _zone_style(code: int) -> Dict[str, str]:
    return _ZONE_STYLE.get(int(code), {"color": "black", "marker": ".", "label": f"code {int(code)}"})


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentiles: Sequence[float]) -> np.ndarray:
    """Weighted percentile(s) of `values` (weighted by `weights`), matching
    `_weighted_range`'s own cumulative-weight interpolation method above but returning the
    percentile value(s) directly rather than a padded axis range. Used for Figure A's "simple
    statistical bands" (plan Sec. 12, item 4) -- a 16th/50th/84th-percentile weighted spread, not a
    full bootstrap (deliberately out of scope for this pass, per the task description).

    Returns `nan` for every requested percentile if there is no finite, non-negative-weight data.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    finite = np.isfinite(v) & np.isfinite(w) & (w >= 0.0)
    v, w = v[finite], w[finite]
    pcts = np.asarray(percentiles, dtype=float)
    if v.size == 0 or float(np.sum(w)) <= 0.0:
        return np.full(pcts.shape, np.nan)
    order = np.argsort(v)
    v_sorted, w_sorted = v[order], w[order]
    cum_w = np.cumsum(w_sorted) / np.sum(w_sorted)
    return np.interp(pcts / 100.0, cum_w, v_sorted)


def _fmt_charge_energy(value: Optional[float]) -> str:
    """`None` -> `"n/a"` (an accounting key genuinely absent from `events.accounting`); a finite
    float -> scientific notation via `_sci`; anything else stringified as a defensive fallback."""
    if value is None:
        return "n/a"
    try:
        return _sci(float(value))
    except (TypeError, ValueError):
        return str(value)


def _panel_impact_footprint(ax, events, geometry) -> None:
    """Figure A, panel 1: weighted impacts with flat-face circle, bevel outer edge, holder
    boundary, and a center-of-energy marker, colored/shaped by surface zone (plan Sec. 12, item 1).
    """
    x_mm = np.asarray(events.x_hit_m, dtype=float) * 1.0e3
    y_mm = np.asarray(events.y_hit_m, dtype=float) * 1.0e3
    w = np.asarray(events.macro_weight_electrons, dtype=float)
    codes = np.asarray(events.surface_code)
    w_max = float(np.max(w)) if w.size else 0.0

    for code in sorted(int(c) for c in np.unique(codes)):
        style = _zone_style(code)
        m = codes == code
        if not np.any(m):
            continue
        sizes = 8.0 + 40.0 * (w[m] / w_max if w_max > 0.0 else 0.0)
        ax.scatter(
            x_mm[m], y_mm[m], s=sizes, c=style["color"], marker=style["marker"], alpha=0.7,
            edgecolors="none", rasterized=True, label=f"{style['label']} (N={int(np.sum(m)):,})",
        )

    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    ax.plot(
        geometry.flat_radius_mm * np.cos(theta), geometry.flat_radius_mm * np.sin(theta),
        "k--", lw=1.2, label="flat edge",
    )
    ax.plot(
        geometry.bevel_outer_radius_mm * np.cos(theta), geometry.bevel_outer_radius_mm * np.sin(theta),
        "k-", lw=1.2, label="bevel outer edge",
    )
    ax.plot(
        geometry.holder_outer_radius_mm * np.cos(theta), geometry.holder_outer_radius_mm * np.sin(theta),
        color="gray", ls=":", lw=1.2, label="holder boundary",
    )

    E = np.asarray(events.incident_energy_J, dtype=float)
    if np.sum(E) > 0.0:
        cx = float(np.sum(x_mm * E) / np.sum(E))
        cy = float(np.sum(y_mm * E) / np.sum(E))
        ax.plot(cx, cy, marker="*", ms=16, color="gold", mec="black", mew=1.0, ls="none",
                 label="center of energy", zorder=10)

    ax.set_xlabel(r"$x\,(\mathrm{mm})$")
    ax.set_ylabel(r"$y\,(\mathrm{mm})$")
    ax.set_aspect("equal")
    ax.set_title("Impact footprint and zones")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False, columnspacing=1.0)


def _panel_deposited_energy_density(ax, fig, heat_source, geometry) -> None:
    """Figure A, panel 2: deposited-energy density map (plan Sec. 12, item 2).

    DESIGN DECISION (the plan explicitly leaves this to the implementer -- see this figure's
    top-level docstring): rather than a separate true-area-corrected bevel panel or an azimuth-
    unwrapped `(phi, arc length)` view, this is ONE combined `(x, y)` map spanning both the flat
    face and the bevel annulus on the SAME uniform Cartesian grid `heat_source` already uses -- the
    plan's own explicitly sanctioned "simpler-but-correct first implementation" of a "companion
    inset/unwrapped bevel map". The bevel's true surface area exceeds its projected in-plane cell
    area by `1/cos(bevel_angle_deg)` (`CathodeGeometry.bevel_true_area_mm2`); the color scale here
    is per PROJECTED cell area, NOT area-corrected for the bevel -- stated explicitly in the
    panel's own annotation below so it is never mistaken for an area-corrected map.
    """
    q_xy = np.sum(np.asarray(heat_source.q_layer_J, dtype=float), axis=2)  # (nx, ny), all layers
    x_mm = np.asarray(heat_source.x_centers_m, dtype=float) * 1.0e3
    y_mm = np.asarray(heat_source.y_centers_m, dtype=float) * 1.0e3
    mask = np.asarray(heat_source.cathode_footprint_mask, dtype=bool)
    q_masked = np.where(mask, q_xy, np.nan)

    cmap = get_default_density_cmap()
    im = ax.pcolormesh(x_mm, y_mm, q_masked.T * 1e9, cmap=cmap, shading="nearest", rasterized=True)
    fig.colorbar(im, ax=ax, label=r"$E_{\rm dep}\,(\mathrm{nJ/cell})$", fraction=0.046)
    add_cathode_boundary_circle(ax, geometry.flat_radius_mm, color="black", ls="--", label="flat edge")
    add_cathode_boundary_circle(
        ax, geometry.bevel_outer_radius_mm, color="black", ls="-", label="bevel outer edge"
    )
    ax.set_xlabel(r"$x\,(\mathrm{mm})$")
    ax.set_ylabel(r"$y\,(\mathrm{mm})$")
    ax.set_aspect("equal")
    ax.set_title("Deposited energy per cell")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False)


def _panel_return_phase_energy(ax, events) -> None:
    """Figure A, panel 3: `K_hit` vs. `t_hit_rf`, weighted by physical charge/energy and colored
    by zone (plan Sec. 12, item 3). Uses the true impact time `t_hit_rf_s` (rather than the
    emission RF phase) since it is the more directly informative "return phase" quantity for this
    panel's purpose -- documented here per the task's "your call, document it" allowance.
    """
    t_ns = np.asarray(events.t_hit_rf_s, dtype=float) * 1.0e9
    K_keV = np.asarray(events.kinetic_energy_eV, dtype=float) / 1.0e3
    w = np.asarray(events.macro_weight_electrons, dtype=float)
    codes = np.asarray(events.surface_code)
    w_max = float(np.max(w)) if w.size else 0.0

    for code in sorted(int(c) for c in np.unique(codes)):
        style = _zone_style(code)
        m = codes == code
        if not np.any(m):
            continue
        sizes = 8.0 + 40.0 * (w[m] / w_max if w_max > 0.0 else 0.0)
        ax.scatter(
            t_ns[m], K_keV[m], s=sizes, c=style["color"], marker=style["marker"], alpha=0.7,
            edgecolors="none", rasterized=True, label=style["label"],
        )

    ax.set_xlabel(r"$t_{\rm hit,RF}\,(\mathrm{ns})$")
    ax.set_ylabel(r"$K_{\rm hit}\,(\mathrm{keV})$")
    ax.set_title("Return phase vs. impact energy")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)


def _panel_energy_incidence_distributions(ax, events) -> None:
    """Figure A, panel 4: zone-separated weighted energy spectrum with simple statistical bands
    (plan Sec. 12, item 4).

    DESIGN DECISION: the incidence-angle distribution is folded into a compact per-zone weighted
    16th/50th/84th-percentile text summary on this SAME axes, rather than a second full histogram
    panel -- this keeps Figure A a literal 2x3 = six-panel grid (plan Sec. 12's own "2x3 layout"
    framing) instead of nesting sub-grids that would multiply the number of visible panels beyond
    six. The energy spectrum is the primary plotted content; both quantities are still
    zone-separated and weighted, and both carry a statistical-band summary (shaded 16-84th
    percentile for energy, numeric 16/50/84th percentile for angle).
    """
    codes = np.asarray(events.surface_code)
    w = np.asarray(events.macro_weight_electrons, dtype=float)
    K_keV = np.asarray(events.kinetic_energy_eV, dtype=float) / 1.0e3

    for code in sorted(int(c) for c in np.unique(codes)):
        style = _zone_style(code)
        m = codes == code
        if not np.any(m):
            continue
        counts, edges = np.histogram(K_keV[m], bins=20, weights=w[m])
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centers, counts, where="mid", color=style["color"], label=style["label"])

        p16, p50, p84 = _weighted_percentile(K_keV[m], w[m], [16.0, 50.0, 84.0])
        if np.isfinite(p16) and np.isfinite(p84):
            ax.axvspan(p16, p84, color=style["color"], alpha=0.12)
        if np.isfinite(p50):
            ax.axvline(p50, color=style["color"], ls=":", lw=1)

    ax.set_xlabel(r"$K_{\rm hit}\,(\mathrm{keV})$")
    ax.set_ylabel("weighted counts (electrons)")
    ax.set_title("Energy spectrum by zone\n(shaded: 16–84th percentile)")
    ax.legend(loc="upper right")


def _panel_origin_to_impact(ax, events) -> None:
    """Figure A, panel 5: emission `(x, y)` colored by a per-event energy proxy for deposited
    energy (plan Sec. 12, item 5).

    DESIGN DECISION: colored by each event's own `incident_energy_J` -- the exact per-event
    deposited energy exists only aggregated onto the spatial/depth grid
    (`heat_source.q_layer_J`), not per event (a single event's energy spreads across several depth
    layers along its own CSDA path, see `rf_gun.back_bombardment_deposition`'s module docstring),
    so `incident_energy_J` is used here as the best available per-event proxy and is labeled as
    such, per the task's "your call, document it" allowance.
    """
    x_emit_mm = np.asarray(events.x_emit_m, dtype=float) * 1.0e3
    y_emit_mm = np.asarray(events.y_emit_m, dtype=float) * 1.0e3
    color_val = np.asarray(events.incident_energy_J, dtype=float)

    sc = ax.scatter(x_emit_mm, y_emit_mm, c=color_val * 1e12, cmap=get_default_density_cmap(), s=16, edgecolors="none", rasterized=True)
    fig = ax.figure
    fig.colorbar(sc, ax=ax, label=r"$E_{\rm incident}\,(\mathrm{pJ/event})$", fraction=0.046)
    ax.set_xlabel(r"$x_{\rm emit}\,(\mathrm{mm})$")
    ax.set_ylabel(r"$y_{\rm emit}\,(\mathrm{mm})$")
    ax.set_title("Emission origin")
    ax.set_aspect("equal")


def _panel_backward_trajectories(ax, events, M_snaps, z_snaps, n_samples: int) -> None:
    """Panel 6: z/pz trajectory across screens for a handful of returning particles, sampled to
    span the range of how far each got (`furthest_z_m`) before turning back -- not a random or
    first-N sample, so the panel shows both near-immediate returns and particles that nearly
    transmitted before reversing.
    """
    pids = np.asarray(events.particle_id, dtype=np.int64)
    if not M_snaps or not z_snaps or pids.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No trajectory data available", ha="center", va="center", transform=ax.transAxes)
        return

    reach_m = np.asarray(events.furthest_z_m, dtype=float)
    order = np.argsort(reach_m)
    n_pick = min(int(n_samples), pids.size)
    pick = np.unique(np.linspace(0, pids.size - 1, n_pick).round().astype(int))
    sample_pids = pids[order][pick]

    for pid in sample_pids:
        z_mm, pz_MeVc = screen_trajectory(int(pid), M_snaps, z_snaps)
        if z_mm.size:
            ax.plot(z_mm, pz_MeVc, "-o", ms=3, lw=1.2, alpha=0.85)

    ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel(r"screen $z\,(\mathrm{mm})$")
    ax.set_ylabel(r"$p_z\,(\mathrm{MeV/c})$")
    ax.set_title(f"Sample trajectories (N={len(sample_pids)})")
    ax.grid(alpha=0.3)


def print_back_bombardment_source_qualification_summary(study: "BackBombardmentMacropulseStudy") -> None:
    """Printed counterpart to `plot_back_bombardment_source_qualification`'s figure: event
    locator, surface counts, charge/energy accounting, incidence-angle quantiles, and the
    deposited-energy area-correction caveat -- everything the figure keeps out of its axes."""
    events = study.study_input.events
    hs = study.heat_source
    geometry = study.config.geometry

    codes = np.asarray(events.surface_code)
    w = np.asarray(events.macro_weight_electrons, dtype=float)
    theta_deg = np.degrees(np.asarray(events.incidence_angle_rad, dtype=float))
    Q_C = float(np.sum(w) * q_e)
    E_inc_total = float(np.sum(np.asarray(events.incident_energy_J, dtype=float)))

    print(f"Back-bombardment source qualification -- event_locator={events.event_locator!r}, N={events.n_events}")
    print(f"  Q_incident={_sci(Q_C)} C | E_incident={_sci(E_inc_total)} J")
    for code in sorted(int(c) for c in np.unique(codes)):
        style = _zone_style(code)
        m = codes == code
        if not np.any(m):
            continue
        t16, t50, t84 = _weighted_percentile(theta_deg[m], w[m], [16.0, 50.0, 84.0])
        print(f"  {style['label']}: N={int(np.sum(m))}, incidence angle 50th[16,84]pct = {t50:.1f}[{t16:.1f},{t84:.1f}] deg")

    bevel_annulus_mm2 = float(np.pi * (geometry.bevel_outer_radius_mm**2 - geometry.flat_radius_mm**2))
    area_factor = 1.0 / float(np.cos(geometry.bevel_angle_rad))
    print(
        f"  Deposited-energy panel is J/cell, NOT area-corrected: true bevel area "
        f"({geometry.bevel_true_area_mm2:.4f} mm^2) = {area_factor:.3f}x its projected annulus "
        f"({bevel_annulus_mm2:.4f} mm^2)."
    )

    accounting = events.accounting if isinstance(events.accounting, dict) else {}
    charge = accounting.get("charge_C", {}) if isinstance(accounting.get("charge_C", {}), dict) else {}
    energy = accounting.get("energy_J", {}) if isinstance(accounting.get("energy_J", {}), dict) else {}
    n = events.n_events
    print(
        f"  Q: emitted={_fmt_charge_energy(charge.get('emitted'))} returned={_fmt_charge_energy(charge.get('returned_after_filter'))} "
        f"transmitted={_fmt_charge_energy(charge.get('transmitted'))} other_loss={_fmt_charge_energy(charge.get('other_lost'))} C"
    )
    print(
        f"  E: incident={_fmt_charge_energy(energy.get('incident_after_filter'))} J | BB0 incident={_sci(float(hs.total_incident_energy_J))} "
        f"deposited={_sci(float(hs.total_deposited_energy_J))} escape_geom={_sci(float(hs.escaping_energy_geometric_J_total))} "
        f"escape_floor={_sci(float(hs.escaping_energy_below_tio_validity_J_total))} excluded_non_lab6={_sci(float(hs.excluded_non_lab6_energy_J_total))} J"
    )
    print(
        f"  N_events={n} (BB0 included={hs.n_events_included}, excluded={hs.n_events_excluded}) | "
        f"statistical uncertainty ~1/sqrt(N)={(1.0 / np.sqrt(max(n, 1))):.2%}"
    )


def plot_back_bombardment_source_qualification(
    study: "BackBombardmentMacropulseStudy",
    *,
    M_snaps: Optional[Sequence[np.ndarray]] = None,
    z_snaps: Optional[Sequence[float]] = None,
    n_trajectory_samples: int = 10,
    style=None,
):
    """Representative-cycle back-bombardment source qualification, 2x3: row 1 is the three spatial
    maps (impact footprint, deposited energy, emission origin); row 2 is return phase vs. energy,
    the energy spectrum by zone, and a sample of return trajectories.

    `M_snaps`/`z_snaps` (the production transport's own screen snapshots) are needed for the
    trajectory panel; without them that panel reports itself unavailable rather than the figure
    failing. Accounting/quantiles/units live in
    `print_back_bombardment_source_qualification_summary`, not in the axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    style = DEFAULT_PLOT_STYLE if style is None else style  # noqa: F841 - kept for API symmetry
    events = study.study_input.events
    heat_source = study.heat_source
    geometry = study.config.geometry

    fig = plt.figure(figsize=(16.0, 11.5), layout="constrained")
    gs = GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    _panel_impact_footprint(ax1, events, geometry)

    ax2 = fig.add_subplot(gs[0, 1])
    _panel_deposited_energy_density(ax2, fig, heat_source, geometry)

    ax3 = fig.add_subplot(gs[0, 2])
    _panel_origin_to_impact(ax3, events)

    ax4 = fig.add_subplot(gs[1, 0])
    _panel_return_phase_energy(ax4, events)

    ax5 = fig.add_subplot(gs[1, 1])
    _panel_energy_incidence_distributions(ax5, events)

    ax6 = fig.add_subplot(gs[1, 2])
    _panel_backward_trajectories(ax6, events, M_snaps, z_snaps, n_trajectory_samples)

    fig.bb_source_qualification_panels = [ax1, ax2, ax3, ax4, ax5, ax6]
    return fig
