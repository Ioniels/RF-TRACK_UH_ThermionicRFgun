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
    t_hit_ns = data.t_hit_s[v] * 1e9
    M_for_plot = np.column_stack([data.x_hit_mm[v], data.y_hit_mm[v], t_hit_ns, data.pz_MeVc[v]])

    fig = plt.figure(figsize=(12.5, 5.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)

    _phase_space_panel(
        fig, gs[0, 0], M_for_plot,
        x_idx=0, y_idx=1, x_label=r"$x\,(\mathrm{mm})$", y_label=r"$y\,(\mathrm{mm})$",
        title=rf"Position at $z=0$ ($N={data.n_valid}$)",
        style=style,
    )
    _phase_space_panel(
        fig, gs[0, 1], M_for_plot,
        x_idx=2, y_idx=3, x_label=r"$\mathrm{ToF}\,(\mathrm{ns})$", y_label=r"$p_z\,(\mathrm{MeV}/c)$",
        title="Longitudinal phase space",
        style=style,
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
    bins: int = 60,
    cmap: str = "inferno",
) -> Optional[Dict[str, Any]]:
    """Figure 2: 2D map of kinetic-energy density deposited at z=0, in J/mm^2.

    Uses each macroparticle's real-electron weight (`rf_gun.back_bombardment`'s
    `weight_per_macroparticle`) so the values are physical (Joules), not macroparticle counts, and
    divides by each bin's area so the map is independent of the `bins` choice -- both needed to
    make this usable directly for a cathode temperature-rise estimate. Kinetic, not total, energy:
    only kinetic energy converts to heat on absorption.
    """
    import matplotlib.pyplot as plt

    if data.n_valid == 0:
        print("No particles behind the cathode with a physically plausible reconstruction.")
        return None

    v = data.valid
    x = data.x_hit_mm[v]
    y = data.y_hit_mm[v]
    k_joules = kinetic_energy_joules(data)[v]

    # `valid` (rf_gun.back_bombardment.compute_back_bombardment) only checks the ballistic
    # (x, y) reconstruction's finiteness, not %E/%K -- a non-finite kinetic energy here (observed
    # for a small number of particles that pick up an extreme kick right at a dynamic-aperture
    # loss point) would otherwise silently turn a histogram bin, and the total, into NaN.
    finite_k = np.isfinite(k_joules)
    n_excluded = int((~finite_k).sum())
    if n_excluded > 0:
        print(
            f"Warning: excluding {n_excluded} of {finite_k.size} back-bombardment particle(s) "
            "with non-finite kinetic energy from the energy-density map and total."
        )
    x, y, k_joules = x[finite_k], y[finite_k], k_joules[finite_k]
    if x.size == 0:
        print("No back-bombardment particles with a finite kinetic energy.")
        return None

    counts, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=k_joules)
    bin_area_mm2 = float(xedges[1] - xedges[0]) * float(yedges[1] - yedges[0])
    density = counts / bin_area_mm2 if bin_area_mm2 > 0.0 else counts
    total_j = float(np.sum(k_joules))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.pcolormesh(xedges, yedges, density.T, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$dK/dA\,(\mathrm{J/mm^2})$")
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
    bins: int = 60,
) -> Optional[Dict[str, Any]]:
    """Figure 3: average power density (heat flux) delivered to the cathode surface vs time.

    Reconstructed from each particle's time-of-flight at the moment it crossed back through z=0
    (`data.t_hit_s`). Kinetic energy per time bin (real electrons, via each macroparticle's
    weight) is divided by the bin width (-> power) and by the cathode's own disk area (-> average
    heat flux, W/mm^2) -- a first-order estimate that assumes the deposited heat spreads evenly
    over the nominal cathode area, not the (generally larger) actual bombarded footprint shown by
    `plot_back_bombardment_energy_density`.
    """
    import matplotlib.pyplot as plt

    if data.n_valid == 0:
        print("No particles behind the cathode with a physically plausible reconstruction.")
        return None

    v = data.valid
    t_s = data.t_hit_s[v]
    k_joules = kinetic_energy_joules(data)[v]

    counts, edges = np.histogram(t_s, bins=bins, weights=k_joules)
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
