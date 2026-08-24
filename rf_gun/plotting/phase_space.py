"""Phase space and spectrum plots.

Two independent visualization knobs replace the project's former `clean_e`/`clean_except_zpz`/
`show_zle0`/`highlight_mode`/`highlight_zlt0`/`highlight_pzlt0`/`highlight_mask`/`highlight_cmap`
parameter set: `exclude_backward_losses` and `exclude_lost`. Each independently either drops the
corresponding population entirely or keeps it, highlighted in a distinct color (grayscale for
backward, green for lost) -- see `_prepare_plot_population`. Tagging is `%id`-based via
`rf_gun.particle_tags.ParticleTags` (not a screen's own z/pz, which does not reliably carry the
true lab-frame sign for a backward-crossing particle -- see
`rf_gun.diagnostics.manual_twiss_and_emittance`'s docstring for the full empirical finding).
"Lost" means removed by the dynamic aperture (`rf_gun.aperture`) during tracking; unlike the
project's former post-hoc radius cut, this tagging needs no per-screen z-gating -- a particle
that was lost is simply and correctly absent from every screen from that point onward, and
present (untagged) at every screen upstream of it, by construction of the physical removal.

`Bout` is intentionally not plotted here: unlike a Screen, it is a fixed-*time* snapshot with a
spread of z among its particles (forward-transmitted ones have traveled further than
backward-turned ones), so it has no single z to display and is not shown as a phase-space panel.
The last screen serves as the "exit" view instead.

The third panel in every triplet below shows ToF-pz, not z-pz (confirmed empirically): a screen's
own `%Z` column is not a lab-frame position at all -- it's each crossing particle's velocity
times its time offset from whichever particle is currently the bunch's *reference* particle, so
it can be large in either sign for a genuinely slow or fast particle without that particle being
anywhere near backward. `%t` (arrival time), by contrast, is a genuine, reliable per-particle
quantity at every screen, so it's used instead throughout -- for `Bout`/`B0` too, for consistency
between panels/rows in the same figure (their own `%Z` is reliable, but mixing conventions within
one figure would be worse). `exclude_backward_losses`/`exclude_lost` still filter by particle
identity (via `%id` against `Bout`'s reliable absolute z/pz and RF-Track's own lost-particle
table), independent of whichever longitudinal quantity is shown.
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
    get_lost_cmap,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_LOST,
)

#: Matches `rf_gun.simulation.EXTENDED_PHASE_FMT` (not imported directly to avoid a
#: plotting -> simulation dependency; kept identical by convention).
EXTENDED_PHASE_FMT_DEFAULT = "%X %Px %Y %Py %Z %Pz %id %t %E %K"

#: Larger-than-default axis-label/tick/legend sizes for `plot_spectra`'s 2x3 grid, where the
#: default rcParams sizes read as cramped next to 6 panels' worth of dual axes and legends.
_LABEL_FONTSIZE = 13
_TICK_FONTSIZE = 11
_LEGEND_FONTSIZE = 12


def _safe_get_phase_space(bunch, selection: str, phase_fmt: str) -> np.ndarray:
    return np.array(bunch.get_phase_space(phase_fmt, selection), copy=True)


def _prepare_plot_population(
    M: np.ndarray,
    tags: Optional[ParticleTags],
    *,
    exclude_backward_losses: bool,
    exclude_lost: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one phase-space snapshot into `(M_plot, backward_highlight, lost_highlight)`.

    - `tags is None` (tagging unavailable, e.g. no `%id` column): returns `M` unchanged with both
      highlight masks all-`False`.
    - `exclude_*=True`: that population is dropped from `M_plot` entirely.
    - `exclude_*=False`: that population stays in `M_plot`; its highlight mask marks which rows
      they are so the caller can render them in a distinct color.
    """
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or tags is None:
        empty = np.zeros((arr.shape[0] if arr.ndim == 2 else 0,), dtype=bool)
        return arr, empty, empty

    is_backward, is_lost = tag_mask(arr, tags)
    keep = np.ones(arr.shape[0], dtype=bool)
    if exclude_backward_losses:
        keep &= ~is_backward
    if exclude_lost:
        keep &= ~is_lost

    M_plot = arr[keep]
    bw_highlight = is_backward[keep] if not exclude_backward_losses else np.zeros(M_plot.shape[0], dtype=bool)
    lost_highlight = is_lost[keep] if not exclude_lost else np.zeros(M_plot.shape[0], dtype=bool)
    return M_plot, bw_highlight, lost_highlight


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
    backward-highlighted, lost-highlighted) -- see `_phase_space_panel`.
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
    lost_mask: np.ndarray | None = None,
    show_colorbar: bool = False,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
):
    """Draw one canonical phase-space panel with mirrored inset marginals.

    Up to three density layers, drawn in order: normal (plasma, default), backward-highlighted
    (grayscale, `PlotStyleConfig.backward_cmap`), lost-highlighted (green, `get_lost_cmap()`). A
    row is normal iff it is in neither highlight mask.

    `x_scale`/`y_scale` rescale the raw `M[:, x_idx]`/`M[:, y_idx]` columns for display -- `x_scale`
    is used for the ToF panel, whose column (`%t`, mm/c) needs converting to ns; `y_scale` converts
    the momentum columns (`%Px`/`%Py`/`%Pz`, MeV/c) to keV/c. 1.0 (no-op) where not needed.
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
    y = np.asarray(M[:, y_idx], dtype=float) * float(y_scale)
    finite = np.isfinite(x) & np.isfinite(y)

    n = M.shape[0]
    bw = np.asarray(backward_mask, dtype=bool).reshape(-1) if backward_mask is not None and np.asarray(backward_mask).size == n else np.zeros(n, dtype=bool)
    lost = np.asarray(lost_mask, dtype=bool).reshape(-1) if lost_mask is not None and np.asarray(lost_mask).size == n else np.zeros(n, dtype=bool)
    normal = finite & ~bw & ~lost

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
            cmap=str(style.backward_cmap),
            zorder=2,
        )
    if np.any(lost & finite):
        phase_space_density(
            ax_main,
            x[lost & finite],
            y[lost & finite],
            scatter=bool(style.scatter),
            bins=int(style.bins),
            scatter_size=int(style.scatter_size),
            scatter_alpha=float(style.scatter_alpha),
            cmap=get_lost_cmap(),
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
    lost_mask: np.ndarray | None,
    *,
    style: PlotStyleConfig,
    title: str,
):
    """Special launch panel: histogram of initial pz split by final tag (surviving/backward/
    lost), in place of a z-pz scatter (which is not meaningful at launch, where all particles sit
    at z=0)."""
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
    lost = np.asarray(lost_mask, dtype=bool)[finite] if lost_mask is not None and np.asarray(lost_mask).size == n else np.zeros(pz0_keV_f.shape[0], dtype=bool)
    normal = ~bw & ~lost
    any_split = bool(np.any(bw) or np.any(lost))

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
        if np.any(lost):
            ax.hist(
                pz0_keV_f[lost], bins=bins, histtype="step",
                linewidth=float(getattr(style, "hist_linewidth", 1.2)),
                color=COLOR_LOST, alpha=float(style.hist_alpha), label="lost",
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


def _patch_launch_t_emit(M_launch_all: np.ndarray, thermo_info: dict | None) -> np.ndarray | None:
    """A copy of `M_launch_all` with its `%t` column (`T_COL`) overwritten by each particle's
    real emission time (`thermo_info["initial_t0_mm_c"]`), so the Initial row's third panel can
    show a real (t_emit, pz) distribution instead of RF-Track's own `%t`.

    `initial_t0_mm_c` is already in mm/c -- the same units `_phase_space_triplet` already scales
    by `_MM_C_TO_NS` for the ToF slot -- and, crucially, in the *same row order* as `M_launch_all`:
    both come from the same never-tracked, never-filtered `B0` (confirmed empirically against the
    installed RF-Track binding -- a freshly built `Bunch6dT`'s `%id` is 0..n-1 in construction
    order), so no `%id` re-matching is needed here, unlike every post-tracking population in this
    project. Returns `None` (meaning: no substitution, caller should fall back to the pz-only
    histogram) when `thermo_info` doesn't have a matching-length `initial_t0_mm_c`.
    """
    if thermo_info is None or M_launch_all.ndim != 2 or M_launch_all.shape[1] <= T_COL:
        return None
    t0_mm_c = thermo_info.get("initial_t0_mm_c", None)
    if t0_mm_c is None:
        return None
    t0_mm_c = np.asarray(t0_mm_c, dtype=float).reshape(-1)
    if t0_mm_c.size != M_launch_all.shape[0]:
        return None
    M_patched = np.array(M_launch_all, dtype=float, copy=True)
    M_patched[:, T_COL] = t0_mm_c
    return M_patched


def _phase_space_triplet(
    fig,
    row_spec,
    M: np.ndarray,
    *,
    style: PlotStyleConfig,
    backward_mask: np.ndarray | None = None,
    lost_mask: np.ndarray | None = None,
    show_colorbar: bool = False,
    prefix: str = "",
    launch_pz_hist_masks: tuple[np.ndarray | None, np.ndarray | None] | None = None,
    third_col_is_t_emit: bool = False,
):
    """Render x-px, y-py, (ToF or t_emit)-pz using one unified panel layout and density engine.

    The third panel uses `%t`/ToF, not z (`%Z`) -- see the module docstring for why -- *except*
    for the Initial row, where `%t` is RF-Track's own elapsed-tracking-time field and is identically
    0 for every particle in a bunch that hasn't been tracked yet (confirmed empirically against
    the installed RF-Track binding), so a real physical quantity is substituted in: see
    `_patch_launch_t_emit` and its callers, which overwrite that column with each particle's known
    emission time before this function ever sees it. `third_col_is_t_emit=True` only changes this
    panel's label/title to say so; the column read is still `T_COL` either way.

    `launch_pz_hist_masks`, when given, is `(backward_mask, lost_mask)` for the fallback ToF-pz-
    slot launch histogram (see `_initial_pz_hist_panel`), used in place of a scatter panel only
    when the real per-particle emission time isn't available (no `thermo_info`) -- a 1D pz
    histogram is still more honest than a scatter against a column that is identically zero.
    """
    third_label = r"$t_{\mathrm{emit}}\,(\mathrm{ns})$" if third_col_is_t_emit else r"$\mathrm{ToF}\,(\mathrm{ns})$"
    third_title_sym = r"t_{\mathrm{emit}}\!\!-\!p_z" if third_col_is_t_emit else r"\mathrm{ToF}\!\!-\!p_z"
    # Momentum columns (Px/Py/Pz) are stored in MeV/c but displayed in keV/c (y_scale=1e3) --
    # more readable given this project's typical thermal/near-cathode momentum scale (~0.1-1 keV/c).
    cfgs = [
        (0, 1, r"$x\,(\mathrm{mm})$", r"$p_x\,(\mathrm{keV}/c)$", rf"${prefix}x\!\!-\!p_x$", 1.0, 1e3),
        (2, 3, r"$y\,(\mathrm{mm})$", r"$p_y\,(\mathrm{keV}/c)$", rf"${prefix}y\!\!-\!p_y$", 1.0, 1e3),
        (T_COL, 5, third_label, r"$p_z\,(\mathrm{keV}/c)$", rf"${prefix}{third_title_sym}$", _MM_C_TO_NS, 1e3),
    ]
    for j, (ix, iy, xl, yl, ttl, xs, ys) in enumerate(cfgs):
        if j == 2 and launch_pz_hist_masks is not None:
            bw_launch, lost_launch = launch_pz_hist_masks
            _initial_pz_hist_panel(fig, row_spec[j], M, bw_launch, lost_launch, style=style, title=ttl)
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
            y_scale=ys,
            title=ttl,
            style=style,
            backward_mask=backward_mask,
            lost_mask=lost_mask,
            show_colorbar=show_colorbar,
        )


def render_screen_phase_space_figure(
    M: np.ndarray,
    *,
    label: str,
    z_mm: float | None = None,
    tags: ParticleTags | None = None,
    exclude_backward_losses: bool = True,
    exclude_lost: bool = True,
    style: PlotStyleConfig | None = None,
    show_colorbar: bool = False,
    n_macroparticles: int | None = None,
    n_ref: int | None = None,
    frame_position: tuple[int, int] | None = None,
    thermo_info: dict | None = None,
):
    """Render one screen-style phase-space figure without widgets/display side effects.

    Transmission-percentage text is always row-count-based (`M_plot.shape[0]` vs. whichever of
    `n_macroparticles`/`n_ref` is available) -- this no longer reads an RF-Track
    `Screen.get_info()` object for anything (mean-arrival-time, when shown, comes from the `%t`
    column already present in `M` itself via the extended phase-space format) -- *except* for the
    Initial (B0) frame, identified by `z_mm is None`: `%t` there is RF-Track's own elapsed-tracking-
    time field, identically 0 for every particle before any tracking happens, so `thermo_info`
    (when given, with a matching-length `initial_t0_mm_c`) substitutes each particle's real
    emission time instead -- see `_patch_launch_t_emit`.

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

    is_launch_frame = z_mm is None
    M_launch_patched = _patch_launch_t_emit(M_raw, thermo_info) if is_launch_frame else None
    if M_launch_patched is not None:
        M_raw = M_launch_patched

    M_plot, bw_mask, lost_mask = _prepare_plot_population(
        M_raw, tags,
        exclude_backward_losses=exclude_backward_losses,
        exclude_lost=exclude_lost,
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
        lost_mask=lost_mask,
        show_colorbar=bool(show_colorbar),
        prefix="",
        launch_pz_hist_masks=(bw_mask, lost_mask) if (is_launch_frame and M_launch_patched is None) else None,
        third_col_is_t_emit=M_launch_patched is not None,
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
    phase_fmt: str = EXTENDED_PHASE_FMT_DEFAULT,
):
    """2x3 spectra figure, no figure-level title: Initial longitudinal kinetic energy, Initial-time,
    and Initial radial distributions (top row, split by `%id` into three eventual-fate categories --
    forward-transmitted, backward-transmitted, and removed by the dynamic aperture -- with a shared
    legend above the row); Output longitudinal kinetic energy, Output time-of-flight, and Output
    radial distributions (bottom row, `Bout`'s own population split into forward vs. backward, with
    its own shared legend positioned in the gap above that row). There is no "lost" category shown
    in the bottom row, but a `Bout` row tagged lost (see below) is dropped entirely rather than
    folded into forward or backward -- it belongs to neither.

    A particle removed by the dynamic aperture during tracking is gone from the live bunch from
    that point on, so it can never be a `Bout` row -- but `tags.lost_ids` (see
    `rf_gun.particle_tags.build_particle_tags`) also includes any `Bout` row whose kinetic energy is
    non-finite or exceeds `rf_gun.particle_tags.MAX_PHYSICAL_KINETIC_ENERGY_MEV`, a numerical
    artifact (e.g. a particle that grazed a field singularity) that *does* still reach `Bout`. Such
    a row is excluded from both bottom-row categories here -- left in, it would otherwise silently
    land in "forward" (its z/pz still nominally look forward-going) and dominate the Output K_z
    histogram with an absurd value.

    Color convention *specific to this figure* (deliberately different from the rest of this
    project's backward=red/lost=green convention, at the user's request): blue (`COLOR_PRIMARY`)
    is always forward; green (`COLOR_LOST`) is always backward (in either row); red
    (`COLOR_SECONDARY`) is lost, top row only -- the bottom row never shows it as a third category,
    dropping any lost-tagged row instead (see above).

    Every time-axis panel (emission-time, output ToF) reports the same two physical quantities as
    `plot_emission_history`'s J(t) panel: current density J (A/cm^2, left) and current I (A,
    right), since each histogram bin there is literally dQ/dt, a rate in time. Every energy-axis
    panel (emission K_z, output K_z) reports the energy-domain analogue instead: current *spectral*
    density dJ/dK_z (mA/(cm^2 keV), left) and dI/dK_z (mA/keV, right), since each bin there is a
    slice of the total emitted charge by energy, not by time -- dividing that charge by the emission
    window `tau` (not a local time step) turns it into the average current attributable to that
    energy slice, the energy-domain counterpart of a current. Both pairs share the same relation,
    (absolute) = (density) x (cathode area) -- only the denominator (time vs. energy) differs.

    The radial panels (initial/output r) are areal particle densities, dN_e/dA in mm^-2 -- *not*
    a plain per-bin count vs. r: an equal-width bin in r covers an annulus of area
    pi*(r_hi^2-r_lo^2), which grows with r, so a naive count-per-bin histogram makes even a
    spatially uniform beam look like its density ramps up toward larger r, an artifact of the
    growing bin area rather than the actual radial profile. Dividing each bin's count by its own
    annulus area (see `_radial_density_hist`) removes that artifact. There is no equally natural
    "current-like" absolute quantity to pair on a secondary axis here, unlike time or energy, so
    these two panels are single-axis.

    The energy-axis panels plot K_z = sqrt(p_z^2 + m_e^2) - m_e, the *longitudinal* kinetic energy
    from p_z alone (not the total 3-momentum p = sqrt(p_x^2+p_y^2+p_z^2)) -- deliberately, since
    p_x/p_y are the beam's transverse degrees of freedom (already shown in the phase-space scatter
    panels elsewhere) and mixing them into this energy axis would conflate a longitudinal spectrum
    with transverse spread.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    try:
        Mf_f_all = _safe_get_phase_space(Bout, "all", phase_fmt)
    except Exception:
        Mf_f_all = _safe_get_phase_space(Bout, "good", phase_fmt)

    finite_z = np.isfinite(Mf_f_all[:, 4])
    is_backward = np.zeros(Mf_f_all.shape[0], dtype=bool)
    is_lost = np.zeros(Mf_f_all.shape[0], dtype=bool)
    if tags is not None and Mf_f_all.shape[0]:
        is_backward, is_lost = tag_mask(Mf_f_all, tags)

    # Exclude a lost-tagged `Bout` row (see this function's docstring) from both bottom-row
    # categories -- it is neither a real forward nor a real backward output particle.
    keep = finite_z & ~is_lost
    Mf_f = Mf_f_all[keep]
    if Mf_f.shape[0] == 0:
        print("No particles in output bunch.")
        return
    is_backward_f = is_backward[keep]

    pz_f = Mf_f[:, 5]
    tof_ns_all = Mf_f[:, 4] * _MM_C_TO_NS
    kz_out_keV_all = (np.sqrt(pz_f**2 + ME_MEV**2) - ME_MEV) * 1e3
    r_out_mm_all = np.sqrt(Mf_f[:, 0] ** 2 + Mf_f[:, 2] ** 2)

    finite_tof = np.isfinite(tof_ns_all)
    finite_kz_out = np.isfinite(kz_out_keV_all)
    finite_r_out = np.isfinite(r_out_mm_all)
    tof_ns_fwd = tof_ns_all[finite_tof & ~is_backward_f]
    tof_ns_bwd = tof_ns_all[finite_tof & is_backward_f]
    kz_out_keV_fwd = kz_out_keV_all[finite_kz_out & ~is_backward_f]
    kz_out_keV_bwd = kz_out_keV_all[finite_kz_out & is_backward_f]
    r_out_mm_fwd = r_out_mm_all[finite_r_out & ~is_backward_f]
    r_out_mm_bwd = r_out_mm_all[finite_r_out & is_backward_f]

    mask_good = finite_z & ~is_backward & ~is_lost
    mask_bad = finite_z & is_backward & ~is_lost

    ids_exit = _try_get_ids(Mf_f_all)
    ids_launch = None
    M0 = None
    if B0 is not None:
        try:
            M0 = _safe_get_phase_space(B0, "all", phase_fmt)
            ids_launch = _try_get_ids(M0)
        except Exception:
            M0, ids_launch = None, None

    lost_ids = tags.lost_ids if tags is not None else frozenset()

    def _split_by_id(values_at_launch: np.ndarray):
        """(forward, backward, lost) split of a per-launch-particle quantity, by each particle's
        eventual fate: forward-transmitted and backward-transmitted come from the exit (`Bout`)
        population's own forward/backward split by `%id`; lost is matched directly against
        `tags.lost_ids` (a removed particle can never reach `Bout`, so this is exhaustive with the
        other two). Returns three empty arrays if id-matching isn't possible."""
        if ids_launch is None or ids_exit is None:
            return np.array([]), np.array([]), np.array([])
        vals = np.asarray(values_at_launch, dtype=float)
        by_id = {int(pid): v for pid, v in zip(ids_launch, vals)}
        v_exit = np.array([by_id.get(int(pid), np.nan) for pid in ids_exit], dtype=float)
        fwd = v_exit[mask_good]
        bwd = v_exit[mask_bad]
        lost_mask = np.array([int(pid) in lost_ids for pid in ids_launch.tolist()])
        lost_vals = vals[lost_mask]
        fwd = fwd[np.isfinite(fwd)]
        bwd = bwd[np.isfinite(bwd)]
        lost_vals = lost_vals[np.isfinite(lost_vals)]
        return fwd, bwd, lost_vals

    t_emit_ns_fwd = t_emit_ns_bwd = t_emit_ns_lost = np.array([])
    if thermo_info is not None:
        t_emit_s = thermo_info.get("t_emit_s", None)
        if t_emit_s is not None:
            t_emit_s = np.asarray(t_emit_s, dtype=float).reshape(-1)
            if ids_launch is not None and ids_launch.size == t_emit_s.size:
                f, b, l = _split_by_id(t_emit_s)
                t_emit_ns_fwd, t_emit_ns_bwd, t_emit_ns_lost = f * 1e9, b * 1e9, l * 1e9
            elif t_emit_s.size == Mf_f_all.shape[0]:
                t_emit_ns_fwd = t_emit_s[mask_good] * 1e9
                t_emit_ns_bwd = t_emit_s[mask_bad] * 1e9

    kz0_keV_fwd = kz0_keV_bwd = kz0_keV_lost = np.array([])
    if M0 is not None and M0.ndim == 2 and M0.shape[1] > 5 and ids_launch is not None:
        pz0 = np.asarray(M0[:, 5], dtype=float)
        f, b, l = _split_by_id(pz0)
        kz0_keV_fwd = (np.sqrt(f**2 + ME_MEV**2) - ME_MEV) * 1e3 if f.size else f
        kz0_keV_bwd = (np.sqrt(b**2 + ME_MEV**2) - ME_MEV) * 1e3 if b.size else b
        kz0_keV_lost = (np.sqrt(l**2 + ME_MEV**2) - ME_MEV) * 1e3 if l.size else l

    r0_mm_fwd = r0_mm_bwd = r0_mm_lost = np.array([])
    if M0 is not None and M0.ndim == 2 and M0.shape[1] > 5 and ids_launch is not None:
        r0_mm_all = np.sqrt(np.asarray(M0[:, 0], dtype=float) ** 2 + np.asarray(M0[:, 2], dtype=float) ** 2)
        r0_mm_fwd, r0_mm_bwd, r0_mm_lost = _split_by_id(r0_mm_all)

    # Real-electron weighting, shared by every panel below: every histogram here is built from a
    # subset of the same N_total macroparticles launched from B0, and each macroparticle
    # represents an equal share (Q_total_C/N_total) of the real emitted charge, regardless of
    # which quantity (time or energy) or which subset (forward/backward-and-lost/output) is being
    # histogrammed -- so this one q_macro/tau/area normalization serves all four panels.
    q_macro = tau_s = area_cm2 = None
    if thermo_info is not None:
        _Q = float(thermo_info.get("Q_total_C", np.nan))
        _tau_ns = float(thermo_info.get("tau_ns", np.nan))
        _N_total = int(Mf_f_all.shape[0])
        _area_m2 = thermo_info.get("area_m2", None)
        if np.isfinite(_Q) and _N_total > 0:
            q_macro = abs(_Q) / _N_total
        if np.isfinite(_tau_ns) and _tau_ns > 0:
            tau_s = _tau_ns * 1e-9
        if _area_m2 is not None and np.isfinite(float(_area_m2)) and float(_area_m2) > 0.0:
            area_cm2 = float(_area_m2) * 1e4

    def _dual_axis_hist(ax, groups, *, weight, xlabel, y_density, y_absolute, title):
        """`groups`: list of (data, color) pairs to stack (1 pair = a single, unsplit series).
        `weight`: the per-event scalar height contribution for the *density* (left) axis --
        already normalized by bin width and by area/tau as appropriate -- or `None` to fall back
        to a plain counts histogram (`N_e`, no secondary axis, used when `thermo_info` lacks the
        charge/area/timing needed to convert counts into a physical rate).
        """
        data_list = [d for d, _c in groups if d.size > 0]
        color_list = [c for d, c in groups if d.size > 0]
        if not data_list:
            ax.axis("off")
            ax.text(0.5, 0.5, "not available", ha="center", va="center")
            return
        bins = np.histogram_bin_edges(np.concatenate(data_list), bins=60)
        weights_list = [np.full(d.shape, weight) for d in data_list] if weight is not None else None
        ax.hist(
            data_list, bins=bins, stacked=True, color=color_list,
            weights=weights_list, alpha=0.8, edgecolor="black", lw=0.4,
        )
        ax.set_xlabel(xlabel, fontsize=_LABEL_FONTSIZE)
        if weight is not None and area_cm2 is not None:
            ax.set_ylabel(y_density, fontsize=_LABEL_FONTSIZE)
            ax_right = ax.secondary_yaxis(
                "right", functions=(lambda y: y * area_cm2, lambda y: y / area_cm2)
            )
            ax_right.set_ylabel(y_absolute, fontsize=_LABEL_FONTSIZE)
            ax_right.tick_params(labelsize=_TICK_FONTSIZE)
        else:
            ax.set_ylabel(r"$N_e$", fontsize=_LABEL_FONTSIZE)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.tick_params(labelsize=_TICK_FONTSIZE)
        ax.grid(alpha=0.3)
        ax.set_title(title)

    def _radial_density_hist(ax, groups, *, title, bins=60):
        """Particles per unit transverse *area* in each radial bin, not a raw per-bin count: an
        equal-width bin in r covers an annulus of area pi*(r_hi^2 - r_lo^2), which grows with r,
        so a plain `ax.hist(r, ...)` makes even a spatially uniform (or centrally peaked) beam
        look like it ramps up toward larger r -- an artifact of growing bin area, not of the
        actual density. Dividing each bin's count by its own annulus area removes that artifact
        and gives the physical quantity a radial profile is supposed to show: density at r, not
        count in [r, r+dr). `groups`: list of (data, color) pairs to stack (matches
        `_dual_axis_hist`'s convention).
        """
        data_list = [d for d, _c in groups if d.size > 0]
        color_list = [c for d, c in groups if d.size > 0]
        if not data_list:
            ax.axis("off")
            ax.text(0.5, 0.5, "not available", ha="center", va="center")
            return
        edges = np.histogram_bin_edges(np.concatenate(data_list), bins=bins)
        annulus_area_mm2 = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        widths = np.diff(edges)
        bottom = np.zeros(edges.shape[0] - 1)
        for d, color in zip(data_list, color_list):
            density = np.histogram(d, bins=edges)[0].astype(float) / annulus_area_mm2
            ax.bar(
                edges[:-1], density, width=widths, bottom=bottom, align="edge",
                color=color, alpha=0.8, edgecolor="black", linewidth=0.4,
            )
            bottom = bottom + density
        ax.set_xlabel(r"$r\,(\mathrm{mm})$", fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel(r"$dN_e/dA\,(\mathrm{mm^{-2}})$", fontsize=_LABEL_FONTSIZE)
        ax.tick_params(labelsize=_TICK_FONTSIZE)
        ax.grid(alpha=0.3)
        ax.set_title(title)

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 9), gridspec_kw={"hspace": 0.5, "wspace": 0.45})

    # -- Initial longitudinal kinetic energy, K_z (top row, col 1) -- 3-way: forward/backward/lost.
    kz0_bins = np.concatenate([arr for arr in (kz0_keV_fwd, kz0_keV_bwd, kz0_keV_lost) if arr.size > 0])
    kz0_bin_width_keV = float(np.diff(np.histogram_bin_edges(kz0_bins, bins=60))[0]) if kz0_bins.size > 1 else np.nan
    kz_weight = None
    if q_macro is not None and tau_s is not None and area_cm2 is not None and np.isfinite(kz0_bin_width_keV) and kz0_bin_width_keV > 0.0:
        kz_weight = (q_macro / tau_s * 1e3) / kz0_bin_width_keV / area_cm2
    _dual_axis_hist(
        axes[0, 0],
        [(kz0_keV_lost, COLOR_SECONDARY), (kz0_keV_bwd, COLOR_LOST), (kz0_keV_fwd, COLOR_PRIMARY)],
        weight=kz_weight,
        xlabel=r"$K_{z,\mathrm{emit}}\,(\mathrm{keV})$",
        y_density=r"$dJ/dK_z\,(\mathrm{mA\,cm^{-2}\,keV^{-1}})$",
        y_absolute=r"$dI/dK_z\,(\mathrm{mA/keV})$",
        title="Initial Longitudinal Kinetic Energy distribution",
    )

    # -- Initial-time (top row, col 2) -- 3-way: forward/backward/lost.
    t_emit_bins = np.concatenate([arr for arr in (t_emit_ns_fwd, t_emit_ns_bwd, t_emit_ns_lost) if arr.size > 0])
    t_emit_bin_width_s = float(np.diff(np.histogram_bin_edges(t_emit_bins, bins=60))[0]) * 1e-9 if t_emit_bins.size > 1 else np.nan
    t_weight = None
    if q_macro is not None and area_cm2 is not None and np.isfinite(t_emit_bin_width_s) and t_emit_bin_width_s > 0.0:
        t_weight = q_macro / t_emit_bin_width_s / area_cm2
    _dual_axis_hist(
        axes[0, 1],
        [(t_emit_ns_lost, COLOR_SECONDARY), (t_emit_ns_bwd, COLOR_LOST), (t_emit_ns_fwd, COLOR_PRIMARY)],
        weight=t_weight,
        xlabel=r"$t_{\mathrm{emit}}\,(\mathrm{ns})$",
        y_density=r"$J\,(\mathrm{A\,cm^{-2}})$",
        y_absolute=r"$I\,(\mathrm{A})$",
        title="Initial-time distribution",
    )

    # -- Initial radial distribution (top row, col 3) -- 3-way: forward/backward/lost.
    _radial_density_hist(
        axes[0, 2],
        [(r0_mm_lost, COLOR_SECONDARY), (r0_mm_bwd, COLOR_LOST), (r0_mm_fwd, COLOR_PRIMARY)],
        title="Initial radial distribution",
    )

    # -- Output longitudinal kinetic energy, K_z (bottom row, col 1) -- forward vs. backward; no
    # "lost" category here (any lost-tagged row was already dropped from `Mf_f`/`is_backward_f`
    # above, see this function's docstring).
    kz_out_bins = np.concatenate([arr for arr in (kz_out_keV_fwd, kz_out_keV_bwd) if arr.size > 0])
    kz_out_bin_width_keV = float(np.diff(np.histogram_bin_edges(kz_out_bins, bins=60))[0]) if kz_out_bins.size > 1 else np.nan
    kz_out_weight = None
    if q_macro is not None and tau_s is not None and area_cm2 is not None and np.isfinite(kz_out_bin_width_keV) and kz_out_bin_width_keV > 0.0:
        kz_out_weight = (q_macro / tau_s * 1e3) / kz_out_bin_width_keV / area_cm2
    _dual_axis_hist(
        axes[1, 0],
        [(kz_out_keV_bwd, COLOR_LOST), (kz_out_keV_fwd, COLOR_PRIMARY)],
        weight=kz_out_weight,
        xlabel=r"$K_z\,(\mathrm{keV})$",
        y_density=r"$dJ/dK_z\,(\mathrm{mA\,cm^{-2}\,keV^{-1}})$",
        y_absolute=r"$dI/dK_z\,(\mathrm{mA/keV})$",
        title="Output Longitudinal Kinetic Energy distribution",
    )

    # -- Output time-of-flight (bottom row, col 2) -- same forward/backward split --
    tof_bins = np.concatenate([arr for arr in (tof_ns_fwd, tof_ns_bwd) if arr.size > 0])
    tof_bin_width_s = float(np.diff(np.histogram_bin_edges(tof_bins, bins=60))[0]) * 1e-9 if tof_bins.size > 1 else np.nan
    tof_weight = None
    if q_macro is not None and area_cm2 is not None and np.isfinite(tof_bin_width_s) and tof_bin_width_s > 0.0:
        tof_weight = q_macro / tof_bin_width_s / area_cm2
    _dual_axis_hist(
        axes[1, 1],
        [(tof_ns_bwd, COLOR_LOST), (tof_ns_fwd, COLOR_PRIMARY)],
        weight=tof_weight,
        xlabel=r"$\mathrm{ToF}\,(\mathrm{ns})$",
        y_density=r"$J\,(\mathrm{A\,cm^{-2}})$",
        y_absolute=r"$I\,(\mathrm{A})$",
        title="Output Time-of-flight distribution",
    )

    # -- Output radial distribution (bottom row, col 3) -- same forward/backward split --
    _radial_density_hist(
        axes[1, 2],
        [(r_out_mm_bwd, COLOR_LOST), (r_out_mm_fwd, COLOR_PRIMARY)],
        title="Output radial distribution",
    )

    # Two shared legends, one per row: the top row has three categories (forward/backward/lost),
    # the bottom only two (forward/backward -- `Bout` can't contain a lost particle). No figure
    # suptitle: the per-panel titles plus these two legends already say everything a title would.
    top_legend_handles = [
        Patch(facecolor=COLOR_PRIMARY, edgecolor="black", label="forward, passed aperture"),
        Patch(facecolor=COLOR_LOST, edgecolor="black", label="backward, passed aperture"),
        Patch(facecolor=COLOR_SECONDARY, edgecolor="black", label="lost (removed by aperture)"),
    ]
    fig.legend(
        handles=top_legend_handles, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.02), fontsize=_LEGEND_FONTSIZE,
    )

    # Positioned in the actual gap between the two rows (not a guessed constant), biased toward
    # the bottom row's top edge so it clears the top row's x-tick labels, which encroach downward
    # into that gap.
    row0_bottom = min(ax.get_position().y0 for ax in axes[0, :])
    row1_top = max(ax.get_position().y1 for ax in axes[1, :])
    bottom_legend_y = row1_top + 0.35 * (row0_bottom - row1_top)
    bottom_legend_handles = [
        Patch(facecolor=COLOR_PRIMARY, edgecolor="black", label="forward, transmitted"),
        Patch(facecolor=COLOR_LOST, edgecolor="black", label="backward, transmitted"),
    ]
    fig.legend(
        handles=bottom_legend_handles, loc="center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, bottom_legend_y), fontsize=_LEGEND_FONTSIZE,
    )

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
    exclude_lost: bool = True,
    phase_fmt: str = EXTENDED_PHASE_FMT_DEFAULT,
    style: PlotStyleConfig | None = None,
    show_colorbar: bool = False,
    output_label: str = r"\mathrm{Output}:\ ",
    thermo_info: dict | None = None,
):
    """Plot Initial (B0) vs. Output phase-space using one shared panel engine.

    `M_exit` is an already-extracted phase-space array (e.g. the last screen) -- *not* `Bout`,
    which is a fixed-time snapshot with a spread of z among its particles and so has no single z
    to plot against (see the module docstring).

    `thermo_info`, when given (with a matching-length `initial_t0_mm_c`), lets the Initial row's
    third panel show the real (t_emit, pz) distribution -- see `_patch_launch_t_emit` -- instead
    of the pz-only histogram fallback used when it's unavailable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    style = DEFAULT_PLOT_STYLE if style is None else style

    M_launch_all = _safe_get_phase_space(B0, "all", phase_fmt)
    M_exit_all = np.asarray(M_exit, dtype=float)

    M_launch_patched = _patch_launch_t_emit(M_launch_all, thermo_info)
    M_launch, bw_launch, lost_launch = _prepare_plot_population(
        M_launch_patched if M_launch_patched is not None else M_launch_all, tags,
        exclude_backward_losses=exclude_backward_losses, exclude_lost=exclude_lost,
    )
    M_ex, bw_ex, lost_ex = _prepare_plot_population(
        M_exit_all, tags,
        exclude_backward_losses=exclude_backward_losses, exclude_lost=exclude_lost,
    )

    if tags is not None and M_exit_all.shape[0]:
        is_bw_all, is_lost_all = tag_mask(M_exit_all, tags)
        print(
            f"Output snapshot: N_raw={M_exit_all.shape[0]} | backward-tagged={int(np.sum(is_bw_all))} "
            f"| lost-tagged={int(np.sum(is_lost_all))} | plotted={M_ex.shape[0]}"
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
        lost_mask=lost_launch,
        show_colorbar=bool(show_colorbar),
        prefix=r"\mathrm{Initial}:\ ",
        launch_pz_hist_masks=None if M_launch_patched is not None else (bw_launch, lost_launch),
        third_col_is_t_emit=M_launch_patched is not None,
    )
    _phase_space_triplet(
        fig,
        gs[1, :].subgridspec(1, 3, wspace=0.26),
        M_ex,
        style=style,
        backward_mask=bw_ex,
        lost_mask=lost_ex,
        show_colorbar=bool(show_colorbar),
        prefix=output_label,
    )

    fig.subplots_adjust(top=0.96, wspace=0.28, hspace=0.28)
    plt.show()


def plot_screen_phase_space_slider(
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    *,
    tags: ParticleTags | None = None,
    exclude_backward_losses: bool = True,
    exclude_lost: bool = True,
    bins: int | None = None,
    n_macroparticles: int | None = None,
    style: PlotStyleConfig | None = None,
    show_colorbar: bool = False,
    B0=None,
    phase_fmt: str = EXTENDED_PHASE_FMT_DEFAULT,
    thermo_info: dict | None = None,
):
    """Interactive screen phase-space slider: Initial (B0) plus every screen.

    `Bout` is intentionally not included as a frame -- see the module docstring; the last screen
    serves as the final "exit-like" frame instead, with a real, single z.

    `thermo_info`, when given, lets the Initial (B0) frame's third panel show the real (t_emit, pz)
    distribution instead of a pz-only histogram -- see `render_screen_phase_space_figure`.
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
            backward_cmap=style.backward_cmap,
            show_histograms=style.show_histograms,
            dezoom_frac=style.dezoom_frac,
            scatter=style.scatter,
        )

    z_mm = 1e3 * np.asarray(z_snaps, dtype=float) if z_snaps is not None else np.asarray([], dtype=float)

    datasets: list[dict] = []
    if B0 is not None:
        try:
            datasets.append({"label": "Initial (B0)", "M": _safe_get_phase_space(B0, "all", phase_fmt), "z_mm": np.nan})
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
        M_plot, _, _ = _prepare_plot_population(
            datasets[i]["M"], tags,
            exclude_backward_losses=exclude_backward_losses,
            exclude_lost=exclude_lost,
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
            exclude_lost=exclude_lost,
            style=style,
            show_colorbar=bool(show_colorbar),
            n_macroparticles=n_macroparticles,
            n_ref=n_ref,
            frame_position=(i + 1, n),
            thermo_info=thermo_info,
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
