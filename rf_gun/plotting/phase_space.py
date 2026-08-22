"""Phase space and spectrum plots.

Two independent visualization knobs replace the project's former `clean_e`/`clean_except_zpz`/
`show_zle0`/`highlight_mode`/`highlight_zlt0`/`highlight_pzlt0`/`highlight_mask`/`highlight_cmap`
parameter set: `exclude_backward_losses` and `exclude_aperture_losses`. Each independently either
drops the corresponding population entirely or keeps it, highlighted in a distinct color
(grayscale for backward, green for aperture-loss) -- see `_prepare_plot_population`. Tagging is
`%id`-based via `rf_gun.particle_tags.ParticleTags` (not a screen's own z/pz, which does not
reliably carry the true lab-frame sign for a backward-crossing particle -- see
`rf_gun.diagnostics.manual_twiss_and_emittance`'s docstring for the full empirical finding).

`Bout` is intentionally not plotted here: unlike a Screen, it is a fixed-*time* snapshot with a
spread of z among its particles (forward-transmitted ones have traveled further than
backward-turned ones), so it has no single z to display and is not shown as a phase-space panel.
The last screen (e.g. the aperture-exit screen) serves as the "exit" view instead.

The third panel in every triplet below shows ToF-pz, not z-pz (confirmed empirically -- see
`rf_gun.aperture`'s module docstring for the full writeup): a screen's own `%Z` column is not a
lab-frame position at all -- it's each crossing particle's velocity times its time offset from
whichever particle is currently the bunch's *reference* particle, so it can be large in either
sign for a genuinely slow or fast particle without that particle being anywhere near backward.
`%t` (arrival time), by contrast, is a genuine, reliable per-particle quantity at every screen, so
it's used instead throughout -- for `Bout`/`B0` too, for consistency between panels/rows in the
same figure (their own `%Z` is reliable, but mixing conventions within one figure would be worse).
`exclude_backward_losses`/`exclude_aperture_losses` still filter by particle identity (via `%id`
against `Bout`'s reliable absolute z/pz), independent of whichever longitudinal quantity is shown.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..constants import ME_MEV, MM_C_TO_NS as _MM_C_TO_NS
from ..particle_tags import ParticleTags, tag_mask, T_COL
from .style import (
    DEFAULT_PLOT_STYLE,
    PlotStyleConfig,
    get_default_density_cmap,
    get_aperture_loss_cmap,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
)

#: Matches `rf_gun.simulation.EXTENDED_PHASE_FMT` (not imported directly to avoid a
#: plotting -> simulation dependency; kept identical by convention).
EXTENDED_PHASE_FMT_DEFAULT = "%X %Px %Y %Py %Z %Pz %id %t %E %K"


def _safe_get_phase_space(bunch, selection: str, phase_fmt: str) -> np.ndarray:
    return np.array(bunch.get_phase_space(phase_fmt, selection), copy=True)


def _prepare_plot_population(
    M: np.ndarray,
    tags: Optional[ParticleTags],
    *,
    exclude_backward_losses: bool,
    exclude_aperture_losses: bool,
    screen_z_m: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one phase-space snapshot into `(M_plot, backward_highlight, aperture_highlight)`.

    - `tags is None` (tagging unavailable, e.g. no `%id` column): returns `M` unchanged with both
      highlight masks all-`False`.
    - `exclude_*=True`: that population is dropped from `M_plot` entirely.
    - `exclude_*=False`: that population stays in `M_plot`; its highlight mask marks which rows
      they are so the caller can render them in a distinct color.
    - `screen_z_m`: this snapshot's own z (meters, absolute lab frame) -- passed through to
      `tag_mask` so aperture-loss tagging doesn't apply upstream of the aperture (see
      `rf_gun.particle_tags.tag_mask`'s docstring). `None` (e.g. Launch/B0) keeps the
      unconditional behavior, which is correct there since z=0 is always upstream of any aperture.
    """
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or tags is None:
        empty = np.zeros((arr.shape[0] if arr.ndim == 2 else 0,), dtype=bool)
        return arr, empty, empty

    is_backward, is_aperture_lost = tag_mask(arr, tags, screen_z_m=screen_z_m)
    keep = np.ones(arr.shape[0], dtype=bool)
    if exclude_backward_losses:
        keep &= ~is_backward
    if exclude_aperture_losses:
        keep &= ~is_aperture_lost

    M_plot = arr[keep]
    bw_highlight = is_backward[keep] if not exclude_backward_losses else np.zeros(M_plot.shape[0], dtype=bool)
    ap_highlight = is_aperture_lost[keep] if not exclude_aperture_losses else np.zeros(M_plot.shape[0], dtype=bool)
    return M_plot, bw_highlight, ap_highlight


def phase_space_density(
    ax,
    x,
    y,
    *,
    scatter: bool = DEFAULT_PLOT_STYLE.scatter,
    cmap=None,
    zorder: int = 1,
    extent=None,
    bins: int = DEFAULT_PLOT_STYLE.bins,
    scatter_size: int = DEFAULT_PLOT_STYLE.scatter_size,
    scatter_alpha: float = DEFAULT_PLOT_STYLE.scatter_alpha,
):
    """Render a 2D phase-space density map.

    - ``scatter=False``: use ``ax.hexbin(...)``
    - ``scatter=True``: KDE-colored ``ax.scatter(...)`` (with sparse-data safeguards)

    Callers pass an explicit `cmap` for whichever population layer they're drawing (normal,
    backward-highlighted, aperture-highlighted) -- see `_phase_space_panel`.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size == 0:
        return None

    cmap_eff = cmap if cmap is not None else get_default_density_cmap()
    bins_eff = max(4, int(bins))

    if not bool(scatter):
        return ax.hexbin(
            x,
            y,
            gridsize=bins_eff,
            cmap=cmap_eff,
            mincnt=1,
            extent=extent,
            linewidths=0.0,
            alpha=float(scatter_alpha),
            zorder=int(zorder),
        )

    use_kde = x.size >= 12
    if use_kde:
        try:
            from scipy.stats import gaussian_kde

            if np.std(x) <= 0.0 or np.std(y) <= 0.0:
                use_kde = False
            else:
                dens = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
                order = np.argsort(dens)
                x_plot = x[order]
                y_plot = y[order]
                dens_plot = dens[order]
                return ax.scatter(
                    x_plot,
                    y_plot,
                    c=dens_plot,
                    cmap=cmap_eff,
                    s=float(scatter_size),
                    alpha=float(scatter_alpha),
                    edgecolors="none",
                    zorder=int(zorder),
                )
        except Exception:
            use_kde = False

    c_val = np.linspace(0.0, 1.0, x.size)
    return ax.scatter(
        x,
        y,
        c=c_val,
        cmap=cmap_eff,
        s=float(scatter_size),
        alpha=float(scatter_alpha),
        edgecolors="none",
        zorder=int(zorder),
    )


def _phase_space_panel(
    fig,
    sub_spec,
    M: np.ndarray,
    *,
    x_idx: int,
    y_idx: int,
    x_label: str,
    y_label: str,
    title: str,
    style: PlotStyleConfig,
    backward_mask: np.ndarray | None = None,
    aperture_mask: np.ndarray | None = None,
    show_colorbar: bool = False,
    x_scale: float = 1.0,
):
    """Draw one canonical phase-space panel with mirrored inset marginals.

    Up to three density layers, drawn in order: normal (plasma, default), backward-highlighted
    (grayscale, `PlotStyleConfig.lost_cmap`), aperture-highlighted (green,
    `get_aperture_loss_cmap()`). A row is normal iff it is in neither highlight mask.

    `x_scale` rescales the raw `M[:, x_idx]` column for display -- used for the ToF panel, whose
    column (`%t`, mm/c) needs converting to ns; 1.0 (no-op) for every other panel.
    """
    ax_main = fig.add_subplot(sub_spec)

    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] == 0 or M.shape[1] <= max(x_idx, y_idx):
        ax_main.text(0.5, 0.5, "No particles", ha="center", va="center", transform=ax_main.transAxes)
        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel(y_label)
        ax_main.set_title(title)
        ax_main.grid(alpha=0.3)
        return ax_main

    x = np.asarray(M[:, x_idx], dtype=float) * float(x_scale)
    y = np.asarray(M[:, y_idx], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)

    n = M.shape[0]
    bw = np.asarray(backward_mask, dtype=bool).reshape(-1) if backward_mask is not None and np.asarray(backward_mask).size == n else np.zeros(n, dtype=bool)
    ap = np.asarray(aperture_mask, dtype=bool).reshape(-1) if aperture_mask is not None and np.asarray(aperture_mask).size == n else np.zeros(n, dtype=bool)
    normal = finite & ~bw & ~ap

    x_main = x[finite]
    y_main = y[finite]

    artist = phase_space_density(
        ax_main,
        x[normal],
        y[normal],
        scatter=bool(style.scatter),
        bins=int(style.bins),
        scatter_size=int(style.scatter_size),
        scatter_alpha=float(style.scatter_alpha),
        cmap=get_default_density_cmap(),
        zorder=1,
    )
    if np.any(bw & finite):
        phase_space_density(
            ax_main,
            x[bw & finite],
            y[bw & finite],
            scatter=bool(style.scatter),
            bins=int(style.bins),
            scatter_size=int(style.scatter_size),
            scatter_alpha=float(style.scatter_alpha),
            cmap=str(style.lost_cmap),
            zorder=2,
        )
    if np.any(ap & finite):
        phase_space_density(
            ax_main,
            x[ap & finite],
            y[ap & finite],
            scatter=bool(style.scatter),
            bins=int(style.bins),
            scatter_size=int(style.scatter_size),
            scatter_alpha=float(style.scatter_alpha),
            cmap=get_aperture_loss_cmap(),
            zorder=3,
        )

    dezoom_frac = float(getattr(style, "dezoom_frac", 0.05))
    if np.isfinite(dezoom_frac) and dezoom_frac > 0.0 and x_main.size and y_main.size:
        x_min, x_max = float(np.min(x_main)), float(np.max(x_main))
        y_min, y_max = float(np.min(y_main)), float(np.max(y_main))

        x_span = x_max - x_min
        y_span = y_max - y_min
        x_pad = dezoom_frac * x_span if x_span > 0.0 else max(1.0, abs(x_max)) * dezoom_frac
        y_pad = dezoom_frac * y_span if y_span > 0.0 else max(1.0, abs(y_max)) * dezoom_frac

        ax_main.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_main.set_ylim(y_min - y_pad, y_max + y_pad)

    if bool(style.show_histograms):
        ax_top = ax_main.inset_axes([0.0, 0.72, 1.0, 0.28], transform=ax_main.transAxes)
        ax_right = ax_main.inset_axes([0.72, 0.0, 0.28, 1.0], transform=ax_main.transAxes)
        ax_top.patch.set_alpha(0.0)
        ax_right.patch.set_alpha(0.0)

        if x_main.size:
            ax_top.hist(
                x_main,
                bins=int(style.bins),
                histtype="step",
                linewidth=float(getattr(style, "hist_linewidth", 1.2)),
                color=str(style.hist_color),
                alpha=float(style.hist_alpha),
                fill=False,
            )
        if y_main.size:
            ax_right.hist(
                y_main,
                bins=int(style.bins),
                histtype="step",
                linewidth=float(getattr(style, "hist_linewidth", 1.2)),
                color=str(style.hist_color),
                alpha=float(style.hist_alpha),
                orientation="horizontal",
                fill=False,
            )

        ax_top.set_xlim(ax_main.get_xlim())
        ax_right.set_ylim(ax_main.get_ylim())
        ax_top.invert_yaxis()
        ax_right.invert_xaxis()

        ax_top.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_right.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_top.spines.values():
            spine.set_visible(False)
        for spine in ax_right.spines.values():
            spine.set_visible(False)

    if bool(show_colorbar) and artist is not None:
        fig.colorbar(artist, ax=ax_main, fraction=0.046, pad=0.02)

    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    ax_main.set_title(title)
    ax_main.grid(alpha=0.3)
    return ax_main


def _initial_pz_hist_panel(
    fig,
    sub_spec,
    M_launch: np.ndarray,
    backward_mask: np.ndarray | None,
    aperture_mask: np.ndarray | None,
    *,
    style: PlotStyleConfig,
    title: str,
):
    """Special launch panel: histogram of initial pz split by final tag (surviving/backward/
    aperture-lost), in place of a z-pz scatter (which is not meaningful at launch, where all
    particles sit at z=0)."""
    ax = fig.add_subplot(sub_spec)
    M = np.asarray(M_launch, dtype=float)
    if M.ndim != 2 or M.shape[0] == 0 or M.shape[1] < 6:
        ax.text(0.5, 0.5, "No particles", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        return ax

    pz0_keV = 1e3 * np.asarray(M[:, 5], dtype=float)
    finite = np.isfinite(pz0_keV)
    pz0_keV_f = pz0_keV[finite]
    if pz0_keV_f.size == 0:
        ax.text(0.5, 0.5, "No finite particles", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        return ax

    bins = np.histogram_bin_edges(pz0_keV_f, bins=int(style.bins))
    n = M.shape[0]
    bw = np.asarray(backward_mask, dtype=bool)[finite] if backward_mask is not None and np.asarray(backward_mask).size == n else np.zeros(pz0_keV_f.shape[0], dtype=bool)
    ap = np.asarray(aperture_mask, dtype=bool)[finite] if aperture_mask is not None and np.asarray(aperture_mask).size == n else np.zeros(pz0_keV_f.shape[0], dtype=bool)
    normal = ~bw & ~ap
    any_split = bool(np.any(bw) or np.any(ap))

    if any_split:
        if np.any(normal):
            ax.hist(
                pz0_keV_f[normal], bins=bins, histtype="step",
                linewidth=float(getattr(style, "hist_linewidth", 1.2)),
                color=COLOR_PRIMARY, alpha=float(style.hist_alpha), label="surviving",
            )
        if np.any(bw):
            ax.hist(
                pz0_keV_f[bw], bins=bins, histtype="step",
                linewidth=float(getattr(style, "hist_linewidth", 1.2)),
                color=COLOR_SECONDARY, alpha=float(style.hist_alpha), label="backward",
            )
        if np.any(ap):
            ax.hist(
                pz0_keV_f[ap], bins=bins, histtype="step",
                linewidth=float(getattr(style, "hist_linewidth", 1.2)),
                color="tab:green", alpha=float(style.hist_alpha), label="aperture-lost",
            )
        ax.legend(frameon=False, loc="best")
    else:
        ax.hist(
            pz0_keV_f, bins=bins, histtype="step",
            linewidth=float(getattr(style, "hist_linewidth", 1.2)),
            color=str(style.hist_color), alpha=float(style.hist_alpha),
        )

    ax.set_xlabel(r"$p_z\,(\mathrm{keV}/c)$")
    ax.set_ylabel(r"$N_e$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def _phase_space_triplet(
    fig,
    row_spec,
    M: np.ndarray,
    *,
    style: PlotStyleConfig,
    backward_mask: np.ndarray | None = None,
    aperture_mask: np.ndarray | None = None,
    show_colorbar: bool = False,
    prefix: str = "",
    launch_pz_hist_masks: tuple[np.ndarray | None, np.ndarray | None] | None = None,
):
    """Render x-px, y-py, ToF-pz using one unified panel layout and density engine.

    The third panel uses ToF (`%t`), not z (`%Z`) -- see the module docstring for why.

    `launch_pz_hist_masks`, when given, is `(backward_mask, aperture_mask)` for the special
    ToF-pz-slot launch histogram (see `_initial_pz_hist_panel`) in place of a scatter panel --
    used only for the Launch row (ToF-pz is not physically meaningful there).
    """
    cfgs = [
        (0, 1, r"$x\,(\mathrm{mm})$", r"$p_x\,(\mathrm{MeV}/c)$", rf"${prefix}x\!\!-\!p_x$", 1.0),
        (2, 3, r"$y\,(\mathrm{mm})$", r"$p_y\,(\mathrm{MeV}/c)$", rf"${prefix}y\!\!-\!p_y$", 1.0),
        (T_COL, 5, r"$\mathrm{ToF}\,(\mathrm{ns})$", r"$p_z\,(\mathrm{MeV}/c)$", rf"${prefix}\mathrm{{ToF}}\!\!-\!p_z$", _MM_C_TO_NS),
    ]
    for j, (ix, iy, xl, yl, ttl, xs) in enumerate(cfgs):
        if j == 2 and launch_pz_hist_masks is not None:
            bw_launch, ap_launch = launch_pz_hist_masks
            _initial_pz_hist_panel(fig, row_spec[j], M, bw_launch, ap_launch, style=style, title=ttl)
            continue
        _phase_space_panel(
            fig,
            row_spec[j],
            M,
            x_idx=ix,
            y_idx=iy,
            x_label=xl,
            y_label=yl,
            x_scale=xs,
            title=ttl,
            style=style,
            backward_mask=backward_mask,
            aperture_mask=aperture_mask,
            show_colorbar=show_colorbar,
        )


def render_screen_phase_space_figure(
    M: np.ndarray,
    *,
    label: str,
    z_mm: float | None = None,
    tags: ParticleTags | None = None,
    exclude_backward_losses: bool = True,
    exclude_aperture_losses: bool = True,
    style: PlotStyleConfig | None = None,
    show_colorbar: bool = False,
    n_macroparticles: int | None = None,
    n_ref: int | None = None,
    frame_position: tuple[int, int] | None = None,
):
    """Render one screen-style phase-space figure without widgets/display side effects.

    Transmission-percentage text is always row-count-based (`M_plot.shape[0]` vs. whichever of
    `n_macroparticles`/`n_ref` is available) -- this no longer reads an RF-Track
    `Screen.get_info()` object for anything (mean-arrival-time, when shown, comes from the `%t`
    column already present in `M` itself via the extended phase-space format).

    There used to be an `n_real_ref` option here (the *real*, charge-weighted electron count,
    `Q_total_C / q_e`) -- removed because it's a category error against a row count: a
    thermionic bunch's real electron count is many orders of magnitude larger than its
    macroparticle count (e.g. ~1e10 real electrons vs. ~1e4 macroparticles), so dividing a row
    count by it silently produced a "transmission" smaller than reality by that same many orders
    of magnitude. Since every macroparticle in this project represents an equal share of the real
    bunch (no per-particle weight column anywhere in the phase-space format), a macroparticle-count
    ratio *is* the real transmission fraction; there is nothing `n_real_ref` could have added.
    """
    import matplotlib.pyplot as plt

    style = DEFAULT_PLOT_STYLE if style is None else style

    M_raw = np.asarray(M, dtype=float)
    if M_raw.ndim != 2 or M_raw.shape[1] < 6:
        M_raw = np.zeros((0, 6), dtype=float)

    M_plot, bw_mask, ap_mask = _prepare_plot_population(
        M_raw, tags,
        exclude_backward_losses=exclude_backward_losses,
        exclude_aperture_losses=exclude_aperture_losses,
        screen_z_m=(float(z_mm) / 1e3) if z_mm is not None else None,
    )

    if n_macroparticles is not None and int(n_macroparticles) > 0:
        transmission_txt = f"{100.0 * float(M_plot.shape[0]) / float(int(n_macroparticles)):.2f}%"
    elif n_ref is not None and int(n_ref) > 0:
        transmission_txt = f"{100.0 * float(M_plot.shape[0]) / float(int(n_ref)):.2f}%"
    else:
        transmission_txt = "n/a"

    prefix = f"{label}"
    if frame_position is not None and len(frame_position) == 2:
        prefix = f"{label} {int(frame_position[0])}/{int(frame_position[1])}"

    t_txt = ""
    if M_raw.shape[1] > T_COL:
        t_col = M_raw[:, T_COL]
        t_col = t_col[np.isfinite(t_col)]
        if t_col.size:
            t_ns = float(np.mean(t_col)) * _MM_C_TO_NS
            t_txt = f" | t={t_ns:.4f} ns"

    if z_mm is not None and np.isfinite(float(z_mm)):
        title = f"{prefix} | z={float(z_mm):.3f} mm{t_txt} | N={int(M_plot.shape[0])} | transmission={transmission_txt}"
    else:
        title = f"{prefix}{t_txt} | N={int(M_plot.shape[0])} | transmission={transmission_txt}"

    fig = plt.figure(figsize=(16.5, 5.4))
    row = fig.add_gridspec(1, 3, wspace=0.18)
    if M_plot.shape[0] == 0:
        axes = [fig.add_subplot(row[j]) for j in range(3)]
        for ax in axes:
            ax.text(0.5, 0.5, "No particles", ha="center", va="center")
            ax.grid(alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    _phase_space_triplet(
        fig,
        row,
        M_plot,
        style=style,
        backward_mask=bw_mask,
        aperture_mask=ap_mask,
        show_colorbar=bool(show_colorbar),
        prefix="",
    )
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_spectra(
    Bout,
    transport_phase_deg: float,
    B0=None,
    thermo_info: dict | None = None,
    *,
    tags: ParticleTags | None = None,
    exclude_backward_losses: bool = False,
    phase_fmt: str = EXTENDED_PHASE_FMT_DEFAULT,
):
    """Plot emission-time, final kinetic-energy, and time-of-flight histograms.

    `exclude_backward_losses` controls only whether backward-tagged particles are removed before
    histogramming (via `%id` lookup against `tags`, not `Bout`'s own z/pz directly) -- there is no
    aperture-loss split here since these are 1D distributions of the final bunch as a whole, not
    phase-space scatter panels.
    """
    import matplotlib.pyplot as plt

    try:
        Mf_f_all = _safe_get_phase_space(Bout, "all", phase_fmt)
    except Exception:
        Mf_f_all = _safe_get_phase_space(Bout, "good", phase_fmt)

    finite_z = np.isfinite(Mf_f_all[:, 4])
    is_backward = np.zeros(Mf_f_all.shape[0], dtype=bool)
    if tags is not None and Mf_f_all.shape[0]:
        is_backward, _ = tag_mask(Mf_f_all, tags)

    Mf_f = Mf_f_all[finite_z & ~is_backward] if exclude_backward_losses else Mf_f_all[finite_z]
    if Mf_f.shape[0] == 0:
        print("No particles in output bunch.")
        return

    pz_f = Mf_f[:, 5]
    tof_ns = Mf_f[:, 4] * _MM_C_TO_NS
    tof_ns = tof_ns[np.isfinite(tof_ns)]

    t_emit_ns_good = np.array([])
    t_emit_ns_bad = np.array([])
    if thermo_info is not None:
        t_emit_s = thermo_info.get("t_emit_s", None)
        if t_emit_s is not None:
            t_emit_s = np.asarray(t_emit_s, dtype=float).reshape(-1)
            ids_exit = _try_get_ids(Mf_f_all)
            ids_launch = None
            if B0 is not None:
                try:
                    M0 = _safe_get_phase_space(B0, "all", phase_fmt)
                    ids_launch = _try_get_ids(M0)
                except Exception:
                    ids_launch = None

            mask_good = finite_z & ~is_backward
            mask_bad = finite_z & is_backward
            if ids_exit is not None and ids_launch is not None and ids_launch.size == t_emit_s.size:
                t_by_id = {int(pid): tval for pid, tval in zip(ids_launch, t_emit_s)}
                t_emit_exit = np.array([t_by_id.get(int(pid), np.nan) for pid in ids_exit], dtype=float)
                t_emit_ns_good = t_emit_exit[mask_good] * 1e9
                t_emit_ns_bad = t_emit_exit[mask_bad] * 1e9
                ids_exit_set = set(ids_exit.tolist())
                lost_t = np.array(
                    [t_by_id.get(int(pid), np.nan) for pid in ids_launch.tolist() if int(pid) not in ids_exit_set],
                    dtype=float,
                )
                lost_t_ns = lost_t[np.isfinite(lost_t)] * 1e9
                if lost_t_ns.size:
                    t_emit_ns_bad = np.concatenate([t_emit_ns_bad, lost_t_ns])
            elif t_emit_s.size == Mf_f_all.shape[0]:
                t_emit_ns_good = t_emit_s[mask_good] * 1e9
                t_emit_ns_bad = t_emit_s[mask_bad] * 1e9

    t_emit_ns_good = t_emit_ns_good[np.isfinite(t_emit_ns_good)] if t_emit_ns_good.size else t_emit_ns_good
    t_emit_ns_bad = t_emit_ns_bad[np.isfinite(t_emit_ns_bad)] if t_emit_ns_bad.size else t_emit_ns_bad

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    if t_emit_ns_good.size > 0 or t_emit_ns_bad.size > 0:
        t_all = np.concatenate([arr for arr in (t_emit_ns_good, t_emit_ns_bad) if arr.size > 0])
        bins_t = np.histogram_bin_edges(t_all, bins=60)
        bin_width_s = float(bins_t[1] - bins_t[0]) * 1e-9 if bins_t.size > 1 else np.nan

        # Convert the per-macroparticle emission-time histogram into a current density: each
        # macroparticle represents Q_total_C/N_total of real charge, so summing that charge over
        # a bin's width gives the current in that bin, and dividing by the cathode area gives J
        # -- same convention as `plot_emission_history`'s J(t) panel (left: J in A/cm^2, right: I
        # in A via a linear secondary axis).
        _area_cm2 = None
        _J_weight = None
        if thermo_info is not None and np.isfinite(bin_width_s) and bin_width_s > 0.0:
            _Q_tot = float(thermo_info.get("Q_total_C", np.nan))
            _area_m2 = thermo_info.get("area_m2", None)
            _N_tot = int(Mf_f_all.shape[0])
            if np.isfinite(_Q_tot) and _area_m2 is not None and _N_tot > 0:
                _q_macro = abs(_Q_tot) / _N_tot
                _area_cm2 = float(_area_m2) * 1e4
                if _area_cm2 > 0.0:
                    _J_weight = _q_macro / bin_width_s / _area_cm2

        _stk_data = [arr for arr in (t_emit_ns_bad, t_emit_ns_good) if arr.size > 0]
        _stk_colors = ([COLOR_SECONDARY] if t_emit_ns_bad.size > 0 else []) + ([COLOR_PRIMARY] if t_emit_ns_good.size > 0 else [])
        _stk_labels = (["backward/lost"] if t_emit_ns_bad.size > 0 else []) + (["surviving"] if t_emit_ns_good.size > 0 else [])
        _stk_weights = [np.full(arr.shape, _J_weight) for arr in _stk_data] if _J_weight is not None else None
        if _stk_data:
            axes[0].hist(
                _stk_data, bins=bins_t, stacked=True, color=_stk_colors, label=_stk_labels,
                weights=_stk_weights, alpha=0.8, edgecolor="black", lw=0.4,
            )
        axes[0].set_xlabel(r"$t_{\mathrm{emit}}\,(\mathrm{ns})$")
        if _J_weight is not None:
            axes[0].set_ylabel(r"$J\,(\mathrm{A\,cm^{-2}})$")
            ax0_right = axes[0].secondary_yaxis(
                "right",
                functions=(
                    lambda y: y * _area_cm2,
                    lambda y: y / _area_cm2,
                ),
            )
            ax0_right.set_ylabel(r"$I\,(\mathrm{A})$")
        else:
            axes[0].set_ylabel(r"$N_e$")
            axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axes[0].grid(alpha=0.3)
        axes[0].set_title("Emission-time distribution")
        axes[0].legend(frameon=False)
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "Emission-time distribution not available", ha="center", va="center")

    # Convert pz (MeV/c) to relativistic kinetic energy (keV).
    ke_keV = (np.sqrt(pz_f**2 + ME_MEV**2) - ME_MEV) * 1e3

    _ma_kev_factor = None
    if thermo_info is not None:
        _Q = float(thermo_info.get("Q_total_C", np.nan))
        _tau_ns = float(thermo_info.get("tau_ns", np.nan))
        _N_total = int(Mf_f_all.shape[0])
        if np.isfinite(_Q) and np.isfinite(_tau_ns) and _tau_ns > 0 and _N_total > 0:
            _q_macro = abs(_Q) / _N_total
            _tau_s = _tau_ns * 1e-9
            _ma_kev_factor = _q_macro / _tau_s * 1e3

    if _ma_kev_factor is not None and ke_keV.size > 1:
        _counts, _edges = np.histogram(ke_keV, bins=60)
        _widths = np.diff(_edges)
        _centers = 0.5 * (_edges[:-1] + _edges[1:])
        _hist_ma_kev = _counts * _ma_kev_factor / _widths
        axes[1].bar(_centers, _hist_ma_kev, width=_widths, alpha=0.8, edgecolor="black", linewidth=0.5, color=COLOR_PRIMARY)
        axes[1].set_xlabel(r"$K\,(\mathrm{keV})$")
        axes[1].set_ylabel(r"$dI/dK\,(\mathrm{mA/keV})$")
        axes[1].grid(alpha=0.3)
        axes[1].set_title(rf"Output spectrum, $\phi={transport_phase_deg:.1f}^\circ$")
    else:
        axes[1].hist(ke_keV, bins=60, alpha=0.8, edgecolor="black", lw=0.5, color=COLOR_PRIMARY)
        axes[1].set_xlabel(r"$K\,(\mathrm{keV})$")
        axes[1].set_ylabel(r"$N_e$")
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axes[1].grid(alpha=0.3)
        axes[1].set_title(rf"Output spectrum, $\phi={transport_phase_deg:.1f}^\circ$")

    if tof_ns.size > 0:
        axes[2].hist(tof_ns, bins=60, alpha=0.8, edgecolor="black", lw=0.5, color=COLOR_PRIMARY)
        axes[2].set_xlabel(r"$\mathrm{ToF}\,(\mathrm{ns})$")
        axes[2].set_ylabel(r"$N_e$")
        axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axes[2].grid(alpha=0.3)
        axes[2].set_title("Time-of-flight distribution")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "ToF not available", ha="center", va="center")

    plt.tight_layout()
    plt.show()


def _try_get_ids(M: np.ndarray) -> np.ndarray | None:
    """Extract the `%id` column from an already-fetched phase-space array (column 6, per
    `EXTENDED_PHASE_FMT`), or `None` if that column isn't present."""
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[1] <= 6 or arr.shape[0] == 0:
        return None
    return arr[:, 6].astype(np.int64)


def plot_phase_space(
    B0,
    M_exit: np.ndarray,
    transport_phase_deg: float,
    *,
    tags: ParticleTags | None = None,
    exclude_backward_losses: bool = True,
    exclude_aperture_losses: bool = True,
    phase_fmt: str = EXTENDED_PHASE_FMT_DEFAULT,
    style: PlotStyleConfig | None = None,
    show_colorbar: bool = False,
    exit_label: str = r"\mathrm{Exit}:\ ",
    exit_z_m: float | None = None,
):
    """Plot Launch (B0) vs. Exit phase-space using one shared panel engine.

    `M_exit` is an already-extracted phase-space array (e.g. the last/aperture-exit screen) --
    *not* `Bout`, which is a fixed-time snapshot with a spread of z among its particles and so has
    no single z to plot against (see the module docstring). `exit_z_m` is that screen's own z
    (meters, absolute lab frame) -- pass it so aperture-loss tagging doesn't apply if `M_exit`
    happens to be upstream of the aperture (see `rf_gun.particle_tags.tag_mask`).
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    style = DEFAULT_PLOT_STYLE if style is None else style

    M_launch_all = _safe_get_phase_space(B0, "all", phase_fmt)
    M_exit_all = np.asarray(M_exit, dtype=float)

    M_launch, bw_launch, ap_launch = _prepare_plot_population(
        M_launch_all, tags,
        exclude_backward_losses=exclude_backward_losses, exclude_aperture_losses=exclude_aperture_losses,
        screen_z_m=0.0,
    )
    M_ex, bw_ex, ap_ex = _prepare_plot_population(
        M_exit_all, tags,
        exclude_backward_losses=exclude_backward_losses, exclude_aperture_losses=exclude_aperture_losses,
        screen_z_m=exit_z_m,
    )

    if tags is not None and M_exit_all.shape[0]:
        is_bw_all, is_ap_all = tag_mask(M_exit_all, tags, screen_z_m=exit_z_m)
        print(
            f"Exit snapshot: N_raw={M_exit_all.shape[0]} | backward-tagged={int(np.sum(is_bw_all))} "
            f"| aperture-lost-tagged={int(np.sum(is_ap_all))} | plotted={M_ex.shape[0]}"
        )

    if M_ex.shape[0] == 0 or M_launch.shape[0] == 0:
        print("No particles to plot in phase space.")
        return

    fig = plt.figure(figsize=(18.0, 10.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.28)

    _phase_space_triplet(
        fig,
        gs[0, :].subgridspec(1, 3, wspace=0.26),
        M_launch,
        style=style,
        backward_mask=bw_launch,
        aperture_mask=ap_launch,
        show_colorbar=bool(show_colorbar),
        prefix=r"\mathrm{Launch}:\ ",
        launch_pz_hist_masks=(bw_launch, ap_launch),
    )
    _phase_space_triplet(
        fig,
        gs[1, :].subgridspec(1, 3, wspace=0.26),
        M_ex,
        style=style,
        backward_mask=bw_ex,
        aperture_mask=ap_ex,
        show_colorbar=bool(show_colorbar),
        prefix=exit_label,
    )

    fig.suptitle(
        rf"$\mathrm{{Phase\ space\ diagnostics}}\;\left(\phi={transport_phase_deg:.1f}^\circ\right)$",
        y=0.995,
    )

    fig.subplots_adjust(top=0.93, wspace=0.28, hspace=0.28)
    plt.show()


def plot_screen_phase_space_slider(
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    *,
    tags: ParticleTags | None = None,
    exclude_backward_losses: bool = True,
    exclude_aperture_losses: bool = True,
    bins: int | None = None,
    n_macroparticles: int | None = None,
    style: PlotStyleConfig | None = None,
    show_colorbar: bool = False,
    B0=None,
    phase_fmt: str = EXTENDED_PHASE_FMT_DEFAULT,
):
    """Interactive screen phase-space slider: Launch (B0) plus every screen.

    `Bout` is intentionally not included as a frame -- see the module docstring; the last screen
    (e.g. the aperture exit) serves as the final "exit-like" frame instead, with a real, single z.
    """
    import matplotlib.pyplot as plt

    style = DEFAULT_PLOT_STYLE if style is None else style
    if bins is not None:
        style = PlotStyleConfig(
            scatter_size=style.scatter_size,
            scatter_alpha=style.scatter_alpha,
            bins=int(bins),
            hist_alpha=style.hist_alpha,
            hist_color=style.hist_color,
            hist_linewidth=style.hist_linewidth,
            lost_cmap=style.lost_cmap,
            show_histograms=style.show_histograms,
            dezoom_frac=style.dezoom_frac,
            scatter=style.scatter,
        )

    z_mm = 1e3 * np.asarray(z_snaps, dtype=float) if z_snaps is not None else np.asarray([], dtype=float)

    datasets: list[dict] = []
    if B0 is not None:
        try:
            datasets.append({"label": "Launch (B0)", "M": _safe_get_phase_space(B0, "all", phase_fmt), "z_mm": np.nan})
        except Exception:
            pass

    if M_snaps is not None and len(M_snaps) > 0 and z_mm.size > 0:
        n_screens = min(len(M_snaps), z_mm.size)
        for i in range(n_screens):
            datasets.append({"label": f"Screen {i + 1}", "M": np.asarray(M_snaps[i], dtype=float), "z_mm": float(z_mm[i])})

    if len(datasets) == 0:
        print("No phase-space datasets available (screens and B0 missing).")
        return

    n = len(datasets)

    def _plotted_count(i: int) -> int:
        z_mm_i = datasets[i]["z_mm"]
        M_plot, _, _ = _prepare_plot_population(
            datasets[i]["M"], tags,
            exclude_backward_losses=exclude_backward_losses,
            exclude_aperture_losses=exclude_aperture_losses,
            screen_z_m=(float(z_mm_i) / 1e3) if np.isfinite(z_mm_i) else 0.0,
        )
        return int(M_plot.shape[0])

    screen_counts = np.array([_plotted_count(i) for i in range(n)], dtype=int)
    n_ref = int(np.max(screen_counts)) if screen_counts.size else 0

    def _draw_figure(i: int):
        rec = datasets[i]
        render_screen_phase_space_figure(
            rec["M"],
            label=rec["label"],
            z_mm=rec["z_mm"] if np.isfinite(rec["z_mm"]) else None,
            tags=tags,
            exclude_backward_losses=exclude_backward_losses,
            exclude_aperture_losses=exclude_aperture_losses,
            style=style,
            show_colorbar=bool(show_colorbar),
            n_macroparticles=n_macroparticles,
            n_ref=n_ref,
            frame_position=(i + 1, n),
        )
        plt.show()

    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display

        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=n - 1,
            step=1,
            description="Screen",
            continuous_update=False,
        )
        output = widgets.Output()

        def _on_value_change(change):
            i = int(change["new"])
            with output:
                clear_output(wait=True)
                _draw_figure(i)

        slider.observe(_on_value_change, names="value")
        with output:
            _draw_figure(0)
        display(widgets.VBox([slider, output]))
        return
    except Exception:
        pass

    print("ipywidgets not available: showing first screen only.")
    _draw_figure(0)
