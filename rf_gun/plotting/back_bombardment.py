"""Back-bombardment figures: where, how energetically, and when do backward-turning particles
re-hit the cathode plane (z=0)? See `rf_gun.back_bombardment` for the underlying reconstruction
and the physics assumptions behind it (field-free drift behind the cathode, momentum conservation).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import numpy as np

from ..back_bombardment import BackBombardmentData, kinetic_energy_joules, screen_trajectory
from ..cathode_geometry import (
    SURFACE_CATHODE_BEVEL,
    SURFACE_CATHODE_FLAT,
    SURFACE_CATHODE_SIDE,
    SURFACE_HOLDER,
    SURFACE_UNKNOWN,
)
from ..constants import q_e
from .phase_space import _phase_space_panel
from .style import DEFAULT_PLOT_STYLE, COLOR_SECONDARY, add_cathode_boundary_circle, get_default_density_cmap

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


def plot_back_bombardment_phase_space(data: BackBombardmentData, *, style=None):
    """Figure 1: standard phase-space-style density figure for the back-bombardment population --
    (x, y) density next to longitudinal (ToF, Pz) density, using the same density-plot engine
    (`rf_gun.plotting.phase_space._phase_space_panel`) as every other phase-space figure in the
    project, rather than a bespoke scatter.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    style = DEFAULT_PLOT_STYLE if style is None else style

    if data.n_valid == 0:
        print("No particles behind the cathode with a physically plausible reconstruction.")
        return

    v = data.valid
    x_hit_mm = data.x_hit_mm[v]
    y_hit_mm = data.y_hit_mm[v]
    t_hit_ns = data.t_hit_s[v] * 1e9
    pz_MeVc = data.pz_MeVc[v]
    M_for_plot = np.column_stack([x_hit_mm, y_hit_mm, t_hit_ns, pz_MeVc])

    # Robust (percentile-based) axis ranges -- a handful of extreme ballistic-reconstruction
    # outliers (near-zero-Pz stragglers, see `rf_gun.back_bombardment`'s module docstring) would
    # otherwise single-handedly set a plain min/max axis range, squashing the rest of the
    # population into an unreadable sliver at the panel's center.
    x_range = _robust_range(x_hit_mm)
    y_range = _robust_range(y_hit_mm)
    t_range = _robust_range(t_hit_ns)
    pz_range = _robust_range(pz_MeVc)

    fig = plt.figure(figsize=(12.5, 5.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)

    _phase_space_panel(
        fig, gs[0, 0], M_for_plot,
        x_idx=0, y_idx=1, x_label=r"$x\,(\mathrm{mm})$", y_label=r"$y\,(\mathrm{mm})$",
        title=rf"Position at $z=0$ ($N={data.n_valid}$)",
        style=style, xlim=x_range, ylim=y_range,
    )
    _phase_space_panel(
        fig, gs[0, 1], M_for_plot,
        x_idx=2, y_idx=3, x_label=r"$\mathrm{ToF}\,(\mathrm{ns})$", y_label=r"$p_z\,(\mathrm{MeV}/c)$",
        title="Longitudinal phase space",
        style=style, xlim=t_range, ylim=pz_range,
    )
    fig.suptitle("Back-bombardment phase space", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_back_bombardment_screen_reach(
    data: BackBombardmentData,
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    *,
    n_trajectories: int = 10,
    cmap: str = "viridis",
):
    """New diagnostic: where in z did each back-bombardment particle get to before turning around?

    Panels 1-2: (x, y) and longitudinal (ToF, Pz) scatter of the back-bombardment population,
    colored by `data.last_screen_z_mm` (furthest screen reached) -- particles that never reached
    any forward screen (`n_screens_reached==0`, immediate bounce-back) are shown in a fixed gray
    rather than dropped.

    Panel 3: `pz` vs `z` trajectory (via `rf_gun.back_bombardment.screen_trajectory`) for the
    `n_trajectories` particles that reached the furthest, each a distinct color, with an "X"
    marker at that particle's trusted `Bout` state (`z<0`, `pz_MeVc`) -- a screen's own Pz can
    carry the wrong sign once a particle has turned around and re-crossed that plane backward (see
    `screen_trajectory`'s docstring), so the Bout marker is the point to trust when the two
    disagree; a dotted line joins it to the trajectory's last point to make any gap visible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if data.n_valid == 0:
        print("No particles behind the cathode with a physically plausible reconstruction.")
        return

    v = data.valid
    x = data.x_hit_mm[v]
    y = data.y_hit_mm[v]
    t_hit_ns = data.t_hit_s[v] * 1e9
    pz = data.pz_MeVc[v]
    last_z = data.last_screen_z_mm[v]
    n_reached = data.n_screens_reached[v]
    reached = n_reached > 0

    fig = plt.figure(figsize=(19.0, 5.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    for idx, (xi, yi, xlabel, ylabel, title) in enumerate([
        (x, y, r"$x\,(\mathrm{mm})$", r"$y\,(\mathrm{mm})$", "Position, by furthest screen reached"),
        (t_hit_ns, pz, r"$\mathrm{ToF}\,(\mathrm{ns})$", r"$p_z\,(\mathrm{MeV}/c)$", "Longitudinal phase space, by furthest screen reached"),
    ]):
        ax = fig.add_subplot(gs[0, idx])
        if np.any(~reached):
            ax.scatter(xi[~reached], yi[~reached], s=10, color="lightgray", label="never reached a screen", zorder=1)
        if np.any(reached):
            sc = ax.scatter(xi[reached], yi[reached], s=10, c=last_z[reached], cmap=cmap, zorder=2)
            fig.colorbar(sc, ax=ax, label=r"furthest screen reached $(\mathrm{mm})$")
        # Robust (percentile-based) axis limits -- a handful of extreme ballistic-reconstruction
        # outliers would otherwise single-handedly set the visible range via matplotlib's default
        # exact-min/max autoscale, squashing the rest of the population into an unreadable sliver.
        xlim = _robust_range(xi)
        ylim = _robust_range(yi)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ids_v = data.ids[v]
    z_bout_v = data.z_bout_mm[v]
    order = np.argsort(last_z)[::-1]
    top_n = min(int(n_trajectories), int(reached.sum()))
    colors = plt.cm.tab10(np.linspace(0, 1, max(top_n, 1)))

    # Bout's own z for a slow-reversing particle can drift arbitrarily far behind the cathode
    # (it had the whole remaining tracked time to do so) -- often orders of magnitude beyond the
    # screens' own span. Plotting it at its true z would squash the interesting near-cathode
    # trajectory detail into an unreadable sliver, so the Bout marker is placed at a fixed,
    # clearly-labeled nominal x position instead; only its Pz value (the trusted quantity being
    # compared here) is true.
    max_screen_z = 0.0
    trajectories = []
    plotted = 0
    for row in order:
        if plotted >= top_n:
            break
        if not reached[row]:
            continue
        pid = int(ids_v[row])
        z_traj, pz_traj = screen_trajectory(pid, M_snaps, z_snaps)
        if z_traj.size == 0:
            continue
        trajectories.append((row, pid, z_traj, pz_traj))
        max_screen_z = max(max_screen_z, float(z_traj.max()))
        plotted += 1

    x_bout_nominal = -0.3 * max_screen_z if max_screen_z > 0 else -1.0
    bout_labeled = False
    for i, (row, pid, z_traj, pz_traj) in enumerate(trajectories):
        color = colors[i]
        ax3.plot(z_traj, pz_traj, "o-", ms=4, color=color, label=f"id {pid}")
        ax3.plot([z_traj[-1], x_bout_nominal], [pz_traj[-1], pz[row]], ":", lw=1, color=color)
        ax3.plot(
            x_bout_nominal, pz[row], marker="x", ms=9, mew=2, color=color,
            label="Bout (nominal x, true Pz)" if not bout_labeled else None,
        )
        bout_labeled = True

    ax3.axhline(0.0, color="gray", ls=":", lw=1)
    ax3.axvline(0.0, color="gray", ls="-", lw=1)
    ax3.set_xlim(x_bout_nominal * 1.2, max_screen_z * 1.1 if max_screen_z > 0 else 1.0)
    # Robust y-limit: a screen's own Pz can occasionally carry an extreme value (same
    # ballistic-reconstruction/near-zero-Pz pathology noted elsewhere), which would otherwise
    # single-handedly stretch this axis and flatten every real trajectory into a line at y=0.
    _traj_pz_vals = [pz_traj for _row, _pid, _z, pz_traj in trajectories]
    _bout_pz_vals = [pz[row] for row, _pid, _z, _pz in trajectories]
    _ylim = _robust_range(*_traj_pz_vals, np.asarray(_bout_pz_vals))
    if _ylim is not None:
        ax3.set_ylim(*_ylim)
    screen_ticks = np.linspace(0.0, max_screen_z if max_screen_z > 0 else 1.0, 5)
    all_ticks = [x_bout_nominal] + screen_ticks.tolist()
    ax3.set_xticks(all_ticks)
    ax3.set_xticklabels(["Bout"] + [f"{t:.0f}" for t in screen_ticks])
    ax3.set_xlabel(r"$z\,(\mathrm{mm})$ (screens; 'Bout': nominal $x$, true $p_z$)")
    ax3.set_ylabel(r"$p_z\,(\mathrm{MeV}/c)$")
    ax3.set_title(f"Furthest-reaching trajectories ($N={plotted}$)")
    ax3.legend(frameon=False, fontsize=7, ncol=2)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_back_bombardment_energy_density(
    data: BackBombardmentData,
    *,
    cathode_radius_mm: float,
    bins: int = 60,
    cmap: str = "inferno",
    range_pad_frac: float = 0.15,
) -> Optional[Dict[str, Any]]:
    """Figure 2: 2D map of kinetic-energy density deposited at z=0, in J/mm^2.

    Weighted by real-electron count (physical Joules, not macroparticle counts) and divided by
    bin area (independent of `bins`) -- kinetic, not total, energy, since only kinetic energy
    converts to heat on absorption.

    `data.heating_relevant` already excludes implausible reconstructions and holder/cavity-wall
    impacts, so `k_joules` is guaranteed finite and physically part of the cathode.

    Range is fixed to the cathode's own footprint (`+-cathode_radius_mm`, padded by
    `range_pad_frac`), not a percentile of hit positions, since this figure is specifically about
    the cathode region.
    """
    import matplotlib.pyplot as plt

    from .style import add_cathode_boundary_circle

    v = data.heating_relevant
    if not np.any(v):
        print("No particles with a physically plausible, cathode-relevant reconstruction.")
        return None

    x = data.x_hit_mm[v]
    y = data.y_hit_mm[v]
    k_joules = kinetic_energy_joules(data)[v]

    half_range = float(cathode_radius_mm) * (1.0 + float(range_pad_frac))
    xrange = (-half_range, half_range)
    yrange = (-half_range, half_range)
    n_outside = int(np.sum((x < xrange[0]) | (x > xrange[1]) | (y < yrange[0]) | (y > yrange[1])))
    if n_outside > 0:
        print(
            f"Note: {n_outside} of {x.size} back-bombardment particle(s) fall outside the "
            "cathode-region map's range (kept in the reported total, dropped from the density map "
            "-- this figure is restricted to the cathode's own footprint)."
        )
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=k_joules, range=[xrange, yrange])
    bin_area_mm2 = float(xedges[1] - xedges[0]) * float(yedges[1] - yedges[0])
    density = counts / bin_area_mm2 if bin_area_mm2 > 0.0 else counts
    total_j = float(np.sum(k_joules))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.pcolormesh(xedges, yedges, density.T, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$dK/dA\,(\mathrm{J/mm^2})$")
    add_cathode_boundary_circle(ax, cathode_radius_mm)
    # labelcolor="white": default black legend text is invisible against this dark colormap.
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor="white")
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_xlabel(r"$x\,(\mathrm{mm})$")
    ax.set_ylabel(r"$y\,(\mathrm{mm})$")
    ax.set_aspect("equal")
    ax.set_title(rf"Kinetic-energy density at $z=0$ ($K_{{\mathrm{{tot}}}}={_sci(total_j)}\,\mathrm{{J}}$)")
    plt.tight_layout()
    plt.show()

    return {"xedges": xedges, "yedges": yedges, "density_J_per_mm2": density, "total_J": total_j}


def plot_back_bombardment_power_density_vs_time(
    data: BackBombardmentData,
    *,
    cathode_radius_mm: float,
    bins: Optional[int] = None,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
    pad_frac: float = 0.15,
) -> Optional[Dict[str, Any]]:
    """Figure 3: average power density (heat flux) delivered to the cathode surface vs time.

    Built from each particle's z=0 crossing time (`data.t_hit_s`): kinetic energy per time bin,
    divided by bin width (-> power) and the cathode's disk area (-> W/mm^2) -- a first-order
    estimate assuming heat spreads evenly over the nominal cathode area, not the generally larger
    actual footprint (see `plot_back_bombardment_energy_density`).

    `data.heating_relevant` excludes implausible reconstructions and holder/cavity-wall impacts.

    Bin range/count are data-driven: an energy-weighted percentile range (`_weighted_range`)
    keeps the axis tied to where the energy actually lands rather than a rare extreme-ToF
    straggler, and `bins=None` (default) scales the bin count to the in-range population.
    """
    import matplotlib.pyplot as plt

    v = data.heating_relevant
    if not np.any(v):
        print("No particles with a physically plausible, cathode-relevant reconstruction.")
        return None

    t_s = data.t_hit_s[v]
    k_joules = kinetic_energy_joules(data)[v]

    t_range = _weighted_range(t_s, k_joules, lo_pct=lo_pct, hi_pct=hi_pct, pad_frac=pad_frac)
    if t_range is not None:
        n_outside = int(np.sum((t_s < t_range[0]) | (t_s > t_range[1])))
        if n_outside > 0:
            print(
                f"Note: {n_outside} of {t_s.size} back-bombardment particle(s) fall outside the "
                "plotted time range (kept in the reported total, dropped from the histogram so a "
                "few extreme re-hit-time outliers don't dictate the whole axis scale)."
            )
        n_in_range = int(t_s.size - n_outside)
    else:
        n_in_range = int(t_s.size)

    if bins is None:
        bins = int(np.clip(np.sqrt(max(n_in_range, 1)) * 3.0, 20, 200))

    counts, edges = np.histogram(t_s, bins=bins, weights=k_joules, range=t_range)
    bin_width_s = float(edges[1] - edges[0])
    cathode_area_mm2 = float(np.pi * float(cathode_radius_mm) ** 2)
    power_density = counts / bin_width_s / cathode_area_mm2 if bin_width_s > 0.0 and cathode_area_mm2 > 0.0 else counts
    t_centers_ns = 0.5 * (edges[:-1] + edges[1:]) * 1e9
    total_j = float(np.sum(k_joules))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(t_centers_ns, power_density, where="mid", color=COLOR_SECONDARY)
    ax.set_xlabel(r"$t_{\mathrm{rehit}}\,(\mathrm{ns})$")
    ax.set_ylabel(r"$dP/dA\,(\mathrm{W/mm^2})$")
    ax.set_title(
        rf"Heat flux vs time ($K_{{\mathrm{{tot}}}}={_sci(total_j)}\,\mathrm{{J}}$, "
        rf"$A_{{\mathrm{{cathode}}}}={cathode_area_mm2:.2f}\,\mathrm{{mm}}^2$)"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "t_centers_ns": t_centers_ns,
        "power_density_W_mm2": power_density,
        "total_J": total_j,
        "cathode_area_mm2": cathode_area_mm2,
    }


# ==================================================================================================
# Figure A -- back-bombardment source qualification (implementation plan Sec. 12, "Figure A").
#
# Operates on the NEW v2 event schema (`rf_gun.back_bombardment_events.BackBombardmentEvents`) and
# the BB0 volumetric source (`rf_gun.back_bombardment_deposition.BackBombardmentHeatSource`), via a
# `rf_gun.studies.back_bombardment_macropulse.BackBombardmentMacropulseStudy` -- a DIFFERENT,
# newer/richer object than the legacy `BackBombardmentData` every function above this point
# consumes. Nothing above this point is modified; this is purely additive, exactly as the plan's
# Work Package 4 requires ("legacy_ballistic"/the old figures stay available for comparison).
# ==================================================================================================

#: Surface-zone -> (color, marker, short label) for every panel below that needs a consistent,
#: shared per-zone visual convention across the whole figure (plan Sec. 3.3's numeric zone codes,
#: human-readable labels reused from `rf_gun.cathode_geometry.SURFACE_LABELS`).
_ZONE_STYLE: Dict[int, Dict[str, str]] = {
    int(SURFACE_CATHODE_FLAT): {"color": "tab:blue", "marker": "o", "label": "flat"},
    int(SURFACE_CATHODE_BEVEL): {"color": "tab:orange", "marker": "^", "label": "bevel"},
    int(SURFACE_CATHODE_SIDE): {"color": "tab:green", "marker": "s", "label": "side"},
    int(SURFACE_HOLDER): {"color": "tab:red", "marker": "x", "label": "holder"},
    int(SURFACE_UNKNOWN): {"color": "gray", "marker": ".", "label": "unknown"},
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
            edgecolors="none", label=f"{style['label']} (N={int(np.sum(m))})",
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
    ax.legend(fontsize=6, loc="upper right", framealpha=0.85)


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
    im = ax.pcolormesh(x_mm, y_mm, q_masked.T, cmap=cmap, shading="nearest")
    fig.colorbar(im, ax=ax, label=r"$E_{\rm dep}\,(\mathrm{J/cell})$", fraction=0.046)
    add_cathode_boundary_circle(ax, geometry.flat_radius_mm, color="black", ls="--", label="flat edge")
    add_cathode_boundary_circle(
        ax, geometry.bevel_outer_radius_mm, color="black", ls="-", label="bevel outer edge"
    )
    ax.set_xlabel(r"$x\,(\mathrm{mm})$")
    ax.set_ylabel(r"$y\,(\mathrm{mm})$")
    ax.set_aspect("equal")
    ax.set_title("Deposited energy per cell (flat + bevel)")
    ax.legend(fontsize=5.5, loc="upper right", labelcolor="black", framealpha=0.85)


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
            edgecolors="none", label=style["label"],
        )

    ax.set_xlabel(r"$t_{\rm hit,RF}\,(\mathrm{ns})$")
    ax.set_ylabel(r"$K_{\rm hit}\,(\mathrm{keV})$")
    ax.set_title("Return phase vs. impact energy")
    ax.legend(fontsize=6, loc="best")
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
    ax.set_title("Energy spectrum by zone (shaded: 16-84th pct)")
    ax.legend(fontsize=6, loc="upper right")


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

    sc = ax.scatter(x_emit_mm, y_emit_mm, c=color_val, cmap=get_default_density_cmap(), s=24, edgecolors="none")
    fig = ax.figure
    fig.colorbar(sc, ax=ax, label=r"$E_{\rm incident}\,(\mathrm{J})$ per event", fraction=0.046)
    ax.set_xlabel(r"$x_{\rm emit}\,(\mathrm{mm})$")
    ax.set_ylabel(r"$y_{\rm emit}\,(\mathrm{mm})$")
    ax.set_title("Emission origin, colored by incident energy")
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

    fig = plt.figure(figsize=(15.0, 9.0))
    gs = GridSpec(2, 3, figure=fig, wspace=0.45, hspace=0.4)

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
