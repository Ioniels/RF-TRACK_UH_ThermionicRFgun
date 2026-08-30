"""Back-bombardment figures: where, how energetically, and when do backward-turning particles
re-hit the cathode plane (z=0)? See `rf_gun.back_bombardment` for the underlying reconstruction
and the physics assumptions behind it (field-free drift behind the cathode, momentum conservation).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..back_bombardment import BackBombardmentData, kinetic_energy_joules, screen_trajectory
from .phase_space import _phase_space_panel
from .style import DEFAULT_PLOT_STYLE, COLOR_SECONDARY


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
