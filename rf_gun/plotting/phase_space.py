"""Phase space and spectrum plots."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..constants import c
from ..diagnostics import info_get_first
from .style import DEFAULT_PLOT_STYLE, PlotStyleConfig, get_default_density_cmap


def _safe_get_phase_space(bunch, selection: str, phase_fmt: str) -> np.ndarray:
    return np.array(bunch.get_phase_space(phase_fmt, selection), copy=True)


def _try_get_ids(bunch, selection: str):
    for fmt in ("%id",):
        try:
            ids = np.array(bunch.get_phase_space(fmt, selection), copy=True).reshape(-1)
            if ids.size:
                return ids
        except Exception:
            continue
    return None


def phase_space_density(
    ax,
    x,
    y,
    *,
    scatter: bool = DEFAULT_PLOT_STYLE.scatter,
    lost: bool = False,
    zorder: int = 1,
    extent=None,
    bins: int = DEFAULT_PLOT_STYLE.bins,
    scatter_size: int = DEFAULT_PLOT_STYLE.scatter_size,
    scatter_alpha: float = DEFAULT_PLOT_STYLE.scatter_alpha,
    cmap=None,
    lost_cmap: str = DEFAULT_PLOT_STYLE.lost_cmap,
):
    """Render a 2D phase-space density map.

    - ``scatter=False``: use ``ax.hexbin(...)``
    - ``scatter=True``: KDE-colored ``ax.scatter(...)`` (with sparse-data safeguards)
    - ``lost=False``: uses default plasma-with-white colormap
    - ``lost=True``: uses ``lost_cmap``
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size == 0:
        return None

    cmap_eff = cmap if cmap is not None else (lost_cmap if lost else get_default_density_cmap())
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
    highlight_mask: np.ndarray | None = None,
    highlight_cmap: str | None = None,
    show_colorbar: bool = False,
):
    """Draw one canonical phase-space panel with mirrored inset marginals.

    Marginals are embedded *inside* the phase-space axis using inset axes and
    mirrored toward the plot interior.
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

    x = np.asarray(M[:, x_idx], dtype=float)
    y = np.asarray(M[:, y_idx], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x_main = x[finite]
    y_main = y[finite]

    artist = phase_space_density(
        ax_main,
        x_main,
        y_main,
        scatter=bool(style.scatter),
        lost=False,
        bins=int(style.bins),
        scatter_size=int(style.scatter_size),
        scatter_alpha=float(style.scatter_alpha),
        cmap=get_default_density_cmap(),
        lost_cmap=str(style.lost_cmap),
        zorder=1,
    )

    if highlight_mask is not None:
        hm = np.asarray(highlight_mask, dtype=bool).reshape(-1)
        if hm.size == x.size:
            hm = hm & finite
            if np.any(hm):
                phase_space_density(
                    ax_main,
                    x[hm],
                    y[hm],
                    scatter=bool(style.scatter),
                    lost=True,
                    bins=int(style.bins),
                    scatter_size=int(style.scatter_size),
                    scatter_alpha=float(style.scatter_alpha),
                    cmap=highlight_cmap,
                    lost_cmap=str(style.lost_cmap),
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
    launch_transmitted_mask: np.ndarray | None,
    *,
    style: PlotStyleConfig,
    title: str,
):
    """Special launch panel: histogram of initial pz split by final transport outcome."""
    ax = fig.add_subplot(sub_spec)
    M = np.asarray(M_launch, dtype=float)
    if M.ndim != 2 or M.shape[0] == 0 or M.shape[1] < 6:
        ax.text(0.5, 0.5, "No particles", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        return ax

    pz0_keV = 1e3 * np.asarray(M[:, 5], dtype=float)
    finite = np.isfinite(pz0_keV)
    pz0_keV = pz0_keV[finite]
    if pz0_keV.size == 0:
        ax.text(0.5, 0.5, "No finite particles", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        return ax

    bins = np.histogram_bin_edges(pz0_keV, bins=int(style.bins))

    if launch_transmitted_mask is not None and launch_transmitted_mask.size == M.shape[0]:
        mask_t = np.asarray(launch_transmitted_mask, dtype=bool)[finite]
        mask_nt = ~mask_t
        ax.hist(
            pz0_keV[mask_t],
            bins=bins,
            histtype="step",
            linewidth=float(getattr(style, "hist_linewidth", 1.2)),
            color="tab:green",
            alpha=float(style.hist_alpha),
            label=r"transmitted ($z\geq 0,\ p_z>0$)",
        )
        ax.hist(
            pz0_keV[mask_nt],
            bins=bins,
            histtype="step",
            linewidth=float(getattr(style, "hist_linewidth", 1.2)),
            color="tab:red",
            alpha=float(style.hist_alpha),
            label=r"non-transmitted ($z<0$ or $p_z\leq 0$)",
        )
        ax.legend(frameon=False, loc="best")
    else:
        ax.hist(
            pz0_keV,
            bins=bins,
            histtype="step",
            linewidth=float(getattr(style, "hist_linewidth", 1.2)),
            color=str(style.hist_color),
            alpha=float(style.hist_alpha),
        )

    ax.set_xlabel(r"$p_z\,(\mathrm{keV}/c)$")
    ax.set_ylabel("N Electrons")
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
    highlight_mask: np.ndarray | None = None,
    highlight_cmap: str | None = None,
    show_colorbar: bool = False,
    prefix: str = "",
    launch_pz_hist_masks: tuple[np.ndarray | None, np.ndarray | None] | None = None,
    zpz_override_M: np.ndarray | None = None,
    zpz_override_highlight_mask: np.ndarray | None = None,
):
    """Render x-px, y-py, z-pz using one unified panel layout and density engine."""
    cfgs = [
        (0, 1, r"$x\,(\mathrm{mm})$", r"$p_x\,(\mathrm{MeV}/c)$", rf"${prefix}x\!\!-\!p_x$"),
        (2, 3, r"$y\,(\mathrm{mm})$", r"$p_y\,(\mathrm{MeV}/c)$", rf"${prefix}y\!\!-\!p_y$"),
        (4, 5, r"$z\,(\mathrm{mm})$", r"$p_z\,(\mathrm{MeV}/c)$", rf"${prefix}z\!\!-\!p_z$"),
    ]
    for j, (ix, iy, xl, yl, ttl) in enumerate(cfgs):
        panel_M = zpz_override_M if (j == 2 and zpz_override_M is not None) else M
        panel_hm = zpz_override_highlight_mask if (j == 2 and zpz_override_M is not None) else highlight_mask
        if j == 2 and launch_pz_hist_masks is not None:
            launch_mask, _ = launch_pz_hist_masks
            _initial_pz_hist_panel(
                fig,
                row_spec[j],
                panel_M,
                launch_mask,
                style=style,
                title=ttl,
            )
            continue
        _phase_space_panel(
            fig,
            row_spec[j],
            panel_M,
            x_idx=ix,
            y_idx=iy,
            x_label=xl,
            y_label=yl,
            title=ttl,
            style=style,
            highlight_mask=panel_hm,
            highlight_cmap=highlight_cmap,
            show_colorbar=show_colorbar,
        )


def _resolve_highlight_mask(
    M: np.ndarray,
    *,
    highlight_mode: str | None,
    highlight_zlt0: bool,
    highlight_pzlt0: bool,
    highlight_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Resolve highlight mask from explicit mode and optional additive switches."""
    arr = np.asarray(M)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None

    mode = None if highlight_mode is None else str(highlight_mode).strip().lower()
    allowed = {None, "zlt0", "pzlt0", "lost", "mask"}
    if mode not in allowed:
        raise ValueError("highlight_mode must be one of: None, 'zlt0', 'pzlt0', 'lost', 'mask'")

    out = np.zeros(arr.shape[0], dtype=bool)

    if mode == "zlt0":
        out |= np.isfinite(arr[:, 4]) & (arr[:, 4] < 0.0)
    elif mode == "pzlt0":
        out |= np.isfinite(arr[:, 5]) & (arr[:, 5] < 0.0)
    elif mode == "lost":
        z_bad = np.isfinite(arr[:, 4]) & (arr[:, 4] < 0.0)
        pz_bad = np.isfinite(arr[:, 5]) & (arr[:, 5] < 0.0)
        out |= z_bad | pz_bad
    elif mode == "mask":
        if highlight_mask is None:
            return None
        hm = np.asarray(highlight_mask, dtype=bool).reshape(-1)
        if hm.size != arr.shape[0]:
            raise ValueError("highlight_mask size must match number of particles")
        out |= hm

    if bool(highlight_zlt0):
        out |= np.isfinite(arr[:, 4]) & (arr[:, 4] < 0.0)
    if bool(highlight_pzlt0):
        out |= np.isfinite(arr[:, 5]) & (arr[:, 5] < 0.0)

    if highlight_mask is not None and mode != "mask":
        hm = np.asarray(highlight_mask, dtype=bool).reshape(-1)
        if hm.size == arr.shape[0]:
            out |= hm

    return out if np.any(out) else None


def _extract_time_ns(info) -> float:
    t_mm_c = info_get_first(info, ["t", "mean_t", "mean_T"])
    if not np.isfinite(t_mm_c):
        return np.nan
    return float((t_mm_c * 1e-3 / c) * 1e9)


def _extract_transmission(info) -> float:
    return info_get_first(info, ["transmission", "Transmission"])


def plot_spectra(
    Bout,
    transport_phase_deg: float,
    B0=None,
    thermo_info: dict | None = None,
    clean_e: bool = False,
    show_zle0: bool = True,
    phase_fmt: str = "%X %Px %Y %Py %Z %Pz",
):
    """Plot emission-time, final pz, and ToF histograms."""
    import matplotlib.pyplot as plt

    try:
        Mf_f_all = _safe_get_phase_space(Bout, "all", phase_fmt)
    except Exception:
        Mf_f_all = _safe_get_phase_space(Bout, "good", phase_fmt)

    finite_z = np.isfinite(Mf_f_all[:, 4])
    if clean_e:
        Mf_f = Mf_f_all[finite_z & (Mf_f_all[:, 4] > 0.0)]
    else:
        Mf_f = Mf_f_all[finite_z]
    if Mf_f.shape[0] == 0:
        print("No particles in output bunch.")
        return

    pz_f = Mf_f[:, 5]
    tof_ns = (Mf_f[:, 4] * 1e-3 / c) * 1e9
    tof_ns = tof_ns[np.isfinite(tof_ns)]

    t_emit_ns_good = np.array([])
    t_emit_ns_bad = np.array([])
    if thermo_info is not None:
        t_emit_s = thermo_info.get("t_emit_s", None)
        if t_emit_s is not None:
            t_emit_s = np.asarray(t_emit_s, dtype=float).reshape(-1)
            ids_exit = _try_get_ids(Bout, "all")
            ids_launch = _try_get_ids(B0, "all") if B0 is not None else None

            if (
                ids_exit is not None
                and ids_launch is not None
                and ids_exit.size == Mf_f_all.shape[0]
                and ids_launch.size == t_emit_s.size
            ):
                t_by_id = {pid: tval for pid, tval in zip(ids_launch, t_emit_s)}
                t_emit_exit = np.array([t_by_id.get(pid, np.nan) for pid in ids_exit], dtype=float)
                mask_good = finite_z & (Mf_f_all[:, 4] > 0.0)
                t_emit_ns_good = t_emit_exit[mask_good] * 1e9
                mask_bad = finite_z & (Mf_f_all[:, 4] <= 0.0)
                t_emit_ns_bad = t_emit_exit[mask_bad] * 1e9
            elif t_emit_s.size == Mf_f_all.shape[0]:
                mask_good = finite_z & (Mf_f_all[:, 4] > 0.0)
                mask_bad = finite_z & (Mf_f_all[:, 4] <= 0.0)
                t_emit_ns_good = t_emit_s[mask_good] * 1e9
                t_emit_ns_bad = t_emit_s[mask_bad] * 1e9

    t_emit_ns_good = t_emit_ns_good[np.isfinite(t_emit_ns_good)] if t_emit_ns_good.size else t_emit_ns_good
    t_emit_ns_bad = t_emit_ns_bad[np.isfinite(t_emit_ns_bad)] if t_emit_ns_bad.size else t_emit_ns_bad

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    t_emit_ns_bad_plot = t_emit_ns_bad if show_zle0 else np.array([])

    if t_emit_ns_good.size > 0 or t_emit_ns_bad_plot.size > 0:
        t_all = np.concatenate([arr for arr in (t_emit_ns_good, t_emit_ns_bad_plot) if arr.size > 0])
        bins_t = np.histogram_bin_edges(t_all, bins=60)
        if t_emit_ns_good.size > 0:
            axes[0].hist(
                t_emit_ns_good,
                bins=bins_t,
                alpha=0.8,
                edgecolor="black",
                lw=0.4,
                color="tab:blue",
                label="final z > 0",
            )
        if t_emit_ns_bad_plot.size > 0:
            axes[0].hist(
                t_emit_ns_bad_plot,
                bins=bins_t,
                alpha=0.8,
                edgecolor="black",
                lw=0.4,
                color="tab:red",
                label="final z <= 0",
            )
        axes[0].set_xlabel("Emission time [ns]")
        axes[0].set_ylabel("N Electrons")
        axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axes[0].grid(alpha=0.3)
        axes[0].set_title("Initial emission distribution vs time")
        if t_emit_ns_bad_plot.size > 0:
            axes[0].legend(frameon=False)
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "Emission-time distribution not available", ha="center", va="center")

    axes[1].hist(pz_f, bins=60, alpha=0.8, edgecolor="black", lw=0.5, color="tab:blue")
    axes[1].set_xlabel("Pz [MeV/c]")
    axes[1].set_ylabel("N Electrons")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[1].grid(alpha=0.3)
    axes[1].set_title(f"Final longitudinal momentum @ phi = {transport_phase_deg:.1f} deg")

    if tof_ns.size > 0:
        axes[2].hist(tof_ns, bins=60, alpha=0.8, edgecolor="black", lw=0.5, color="tab:green")
        axes[2].set_xlabel("ToF [ns]")
        axes[2].set_ylabel("N Electrons")
        axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axes[2].grid(alpha=0.3)
        axes[2].set_title("Time of flight histogram")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "ToF not available", ha="center", va="center")

    plt.tight_layout()
    plt.show()


def plot_phase_space(
    B0,
    Bout,
    transport_phase_deg: float,
    clean_e: bool = False,
    show_zle0: bool = True,
    phase_fmt: str = "%X %Px %Y %Py %Z %Pz",
    *,
    style: PlotStyleConfig | None = None,
    highlight_mode: str | None = None,
    highlight_zlt0: bool = False,
    highlight_pzlt0: bool = False,
    highlight_mask: np.ndarray | None = None,
    highlight_cmap: str | None = None,
    show_colorbar: bool = False,
    clean_except_zpz: bool = False,
):
    """Plot launch/final phase-space using one shared panel engine.

    Examples
    --------
    Normal plot:
        ``plot_phase_space(B0, Bout, phase_deg)``
    Highlight backward particles with z < 0:
        ``plot_phase_space(B0, Bout, phase_deg, highlight_mode='zlt0')``
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    style = DEFAULT_PLOT_STYLE if style is None else style

    try:
        Mf_f_all = _safe_get_phase_space(Bout, "all", phase_fmt)
        Mf_launch_all = _safe_get_phase_space(B0, "all", phase_fmt)
        has_all = True
    except Exception:
        Mf_f_all = _safe_get_phase_space(Bout, "good", phase_fmt)
        Mf_launch_all = _safe_get_phase_space(B0, "good", phase_fmt)
        has_all = False

    finite_z = np.isfinite(Mf_f_all[:, 4])
    finite_pz = np.isfinite(Mf_f_all[:, 5])
    finite_phase = finite_z & finite_pz
    mask_good = finite_phase & (Mf_f_all[:, 4] >= 0.0) & (Mf_f_all[:, 5] > 0.0)
    mask_bad = finite_phase & ~mask_good

    Mf_f_full = Mf_f_all[finite_phase]
    Mf_f_good = Mf_f_all[mask_good]
    Mf_f = Mf_f_good if clean_e else Mf_f_full

    ids_exit = _try_get_ids(Bout, "all") if has_all else None
    ids_launch = _try_get_ids(B0, "all") if has_all else None
    lost_ids = None
    if ids_exit is not None and ids_exit.size == Mf_f_all.shape[0]:
        lost_ids = ids_exit[mask_bad]
        print(f"Non-transmitted particles (z < 0 or pz <= 0): {lost_ids.size} of {ids_exit.size}")
        if lost_ids.size:
            preview = np.array2string(lost_ids[:20], separator=", ")
            print(f"Non-transmitted particle IDs (first 20): {preview}")

    total_tracked = int(np.sum(finite_phase))
    lost_count = int(np.sum(mask_bad))
    good_count = int(np.sum(mask_good))
    n_initial = int(Mf_launch_all.shape[0]) if Mf_launch_all.ndim == 2 else 0
    transmission_pct = 100.0 * good_count / n_initial if n_initial > 0 else 0.0
    print(f"Non-transmitted (z < 0 or pz <= 0): {lost_count} / {total_tracked} (among final finite-phase bunch)")
    print(f"Transmission (z >= 0 and pz > 0): {good_count} / {n_initial} = {transmission_pct:.2f}% (vs initial)")

    Mf_launch = Mf_launch_all
    launch_mask_bad = None
    if has_all and lost_ids is not None and ids_launch is not None:
        if ids_launch.size == Mf_launch_all.shape[0]:
            launch_mask_bad = np.isin(ids_launch, lost_ids)
        else:
            print("Warning: launch ID array length does not match launch phase space.")
    elif has_all and Mf_launch_all.shape[0] == Mf_f_all.shape[0]:
        launch_mask_bad = mask_bad
    elif has_all and Mf_launch_all.shape[0] != Mf_f_all.shape[0]:
        print(
            "Warning: 'all' selections differ in size; cannot map launch to exit one-to-one. "
            "Skipping red overlay on launch plots."
        )

    launch_mask_bad_full = None
    if launch_mask_bad is not None and launch_mask_bad.size == Mf_launch_all.shape[0]:
        launch_mask_bad_full = np.asarray(launch_mask_bad, dtype=bool)

    if clean_e and launch_mask_bad_full is not None:
        keep_launch = ~launch_mask_bad_full
        Mf_launch = Mf_launch_all[keep_launch]
    else:
        Mf_launch = Mf_launch_all

    if Mf_f.shape[0] == 0 or Mf_launch.shape[0] == 0:
        print("No particles to plot in phase space.")
        return

    # Preserve historical quick switch only when not cleaning non-transmitted particles.
    if (not clean_e) and show_zle0 and not any(
        (
            highlight_mode is not None,
            bool(highlight_zlt0),
            bool(highlight_pzlt0),
            highlight_mask is not None,
        )
    ):
        highlight_zlt0 = True

    final_hm_all = _resolve_highlight_mask(
        Mf_f_all,
        highlight_mode=highlight_mode,
        highlight_zlt0=bool(highlight_zlt0),
        highlight_pzlt0=bool(highlight_pzlt0),
        highlight_mask=highlight_mask,
    )

    final_indices = np.flatnonzero(finite_phase & (mask_good if clean_e else True))
    final_hm = None
    if final_hm_all is not None and final_indices.size:
        final_hm = np.asarray(final_hm_all[final_indices], dtype=bool)

    launch_hm_xy = None
    launch_hm_zpz = None
    launch_transmitted_mask_xy = None
    launch_transmitted_mask_zpz = None
    if launch_mask_bad_full is not None:
        launch_hm_zpz = np.asarray(launch_mask_bad_full, dtype=bool)
        launch_transmitted_mask_zpz = ~launch_hm_zpz
        if not clean_e:
            launch_hm_xy = launch_hm_zpz
            launch_transmitted_mask_xy = launch_transmitted_mask_zpz

    final_hm_full = None
    final_indices_full = np.flatnonzero(finite_phase)
    if final_hm_all is not None and final_indices_full.size:
        final_hm_full = np.asarray(final_hm_all[final_indices_full], dtype=bool)

    launch_zpz_M = Mf_launch_all if (clean_e and clean_except_zpz) else None
    final_zpz_M = Mf_f_full if (clean_e and clean_except_zpz) else None
    launch_hm_for_zpz = launch_hm_zpz if (clean_e and clean_except_zpz) else None
    final_hm_for_zpz = final_hm_full if (clean_e and clean_except_zpz) else None
    launch_masks_for_hist = (
        launch_transmitted_mask_zpz,
        launch_hm_zpz,
    ) if (clean_e and clean_except_zpz) else (
        launch_transmitted_mask_xy,
        launch_hm_xy,
    )

    fig = plt.figure(figsize=(18.0, 10.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.28)

    _phase_space_triplet(
        fig,
        gs[0, :].subgridspec(1, 3, wspace=0.26),
        Mf_launch,
        style=style,
        highlight_mask=launch_hm_xy,
        highlight_cmap=highlight_cmap,
        show_colorbar=bool(show_colorbar),
        prefix=r"\mathrm{Launch}:\ ",
        launch_pz_hist_masks=launch_masks_for_hist,
        zpz_override_M=launch_zpz_M,
        zpz_override_highlight_mask=launch_hm_for_zpz,
    )
    _phase_space_triplet(
        fig,
        gs[1, :].subgridspec(1, 3, wspace=0.26),
        Mf_f,
        style=style,
        highlight_mask=final_hm,
        highlight_cmap=highlight_cmap,
        show_colorbar=bool(show_colorbar),
        prefix=r"\mathrm{Exit}:\ ",
        zpz_override_M=final_zpz_M,
        zpz_override_highlight_mask=final_hm_for_zpz,
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
    info_snaps: Sequence[object] | None = None,
    clean_e: bool = True,
    bins: int | None = None,
    n_real_ref: float | None = None,
    n_macroparticles: int | None = None,
    *,
    style: PlotStyleConfig | None = None,
    highlight_mode: str | None = None,
    highlight_zlt0: bool = False,
    highlight_pzlt0: bool = False,
    highlight_mask: np.ndarray | Sequence[np.ndarray] | None = None,
    highlight_cmap: str | None = None,
    show_colorbar: bool = False,
    B0=None,
    Bout=None,
    phase_fmt: str = "%X %Px %Y %Py %Z %Pz",
    clean_except_zpz: bool = False,
):
    """Interactive screen phase-space with shared panel rendering core.

    Examples
    --------
    Normal screen view:
        ``plot_screen_phase_space_slider(M_snaps, z_snaps)``
    Highlight z < 0 particles on each screen:
        ``plot_screen_phase_space_slider(M_snaps, z_snaps, highlight_mode='zlt0')``
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
            datasets.append({"kind": "launch", "label": "Launch (B0)", "M": _safe_get_phase_space(B0, "all", phase_fmt), "z_mm": np.nan, "info": None, "hmask": None})
        except Exception:
            pass

    if M_snaps is not None and len(M_snaps) > 0 and z_mm.size > 0:
        n_screens = min(len(M_snaps), z_mm.size)
        for i in range(n_screens):
            info_i = info_snaps[i] if (info_snaps is not None and len(info_snaps) > i) else None
            hm_i = highlight_mask[i] if isinstance(highlight_mask, (list, tuple)) and i < len(highlight_mask) else highlight_mask
            datasets.append({
                "kind": "screen",
                "label": f"Screen {i+1}",
                "M": np.asarray(M_snaps[i], dtype=float),
                "z_mm": float(z_mm[i]),
                "info": info_i,
                "hmask": hm_i,
            })

    if Bout is not None:
        try:
            datasets.append({"kind": "exit", "label": "Exit (Bout)", "M": _safe_get_phase_space(Bout, "all", phase_fmt), "z_mm": np.nan, "info": None, "hmask": None})
        except Exception:
            pass

    if len(datasets) == 0:
        print("No phase-space datasets available (screens and B0/Bout missing).")
        return

    n = len(datasets)

    def _matrix_for_logic(rec: dict) -> np.ndarray:
        M_logic = np.asarray(rec["M"], dtype=float)
        if M_logic.ndim != 2 or M_logic.shape[1] < 6:
            return M_logic
        if rec.get("kind") == "screen":
            z0_mm = rec.get("z_mm", np.nan)
            if np.isfinite(z0_mm):
                M_logic = np.array(M_logic, copy=True)
                M_logic[:, 4] = np.asarray(M_logic[:, 4], dtype=float) + float(z0_mm)
        return M_logic

    def _pick_data(i: int):
        rec = datasets[i]
        M_raw = np.asarray(rec["M"], dtype=float)
        if M_raw.ndim != 2 or M_raw.shape[1] < 6:
            return np.empty((0, 6), dtype=float), np.zeros((0,), dtype=bool)
        M_logic = _matrix_for_logic(rec)
        hm = _resolve_highlight_mask(
            M_logic,
            highlight_mode=highlight_mode,
            highlight_zlt0=bool(highlight_zlt0),
            highlight_pzlt0=bool(highlight_pzlt0),
            highlight_mask=rec["hmask"],
        )
        if clean_e:
            good = np.isfinite(M_logic[:, 4]) & np.isfinite(M_logic[:, 5]) & (M_logic[:, 4] >= 0.0) & (M_logic[:, 5] > 0.0)
            hm = hm[good] if hm is not None and hm.size == M_raw.shape[0] else hm
            M_raw = M_raw[good]
        else:
            good = np.isfinite(M_logic[:, 4]) & np.isfinite(M_logic[:, 5])
            hm = hm[good] if hm is not None and hm.size == M_raw.shape[0] else hm
            M_raw = M_raw[good]
        return M_raw, (hm if hm is not None else np.zeros(M_raw.shape[0], dtype=bool))

    screen_counts = np.array([_pick_data(i)[0].shape[0] for i in range(n)], dtype=int)
    transmission_info = np.asarray([
        _extract_transmission(datasets[i]["info"]) if datasets[i]["kind"] == "screen" and datasets[i]["info"] is not None else np.nan
        for i in range(n)
    ], dtype=float)
    use_real_ref = n_real_ref is not None and np.isfinite(n_real_ref) and float(n_real_ref) > 0.0 and np.any(np.isfinite(transmission_info))
    use_macro_ref = n_macroparticles is not None and int(n_macroparticles) > 0
    n_ref = int(np.max(screen_counts)) if screen_counts.size else 0

    def _title_for(i: int) -> str:
        if use_real_ref and np.isfinite(transmission_info[i]):
            transmission_pct = 100.0 * float(transmission_info[i]) / float(n_real_ref)
            transmission_txt = f"{transmission_pct:.2f}%"
        elif use_macro_ref:
            transmission_pct = 100.0 * float(screen_counts[i]) / float(int(n_macroparticles))
            transmission_txt = f"{transmission_pct:.2f}%"
        elif n_ref > 0:
            transmission_pct = 100.0 * float(screen_counts[i]) / float(n_ref)
            transmission_txt = f"{transmission_pct:.2f}%"
        else:
            transmission_txt = "n/a"

        z_val = datasets[i]["z_mm"]
        info_i = datasets[i]["info"]
        t_val = _extract_time_ns(info_i) if info_i is not None else np.nan
        if np.isfinite(z_val) and np.isfinite(t_val):
            return (
                f"{datasets[i]['label']} {i+1}/{n} | z={z_val:.3f} mm | t={t_val:.4f} ns"
                f" | N={int(screen_counts[i])} | transmission={transmission_txt}"
            )
        if np.isfinite(z_val):
            return (
                f"{datasets[i]['label']} {i+1}/{n} | z={z_val:.3f} mm"
                f" | N={int(screen_counts[i])} | transmission={transmission_txt}"
            )
        return (
            f"{datasets[i]['label']} {i+1}/{n}"
            f" | N={int(screen_counts[i])} | transmission={transmission_txt}"
        )

    def _draw_figure(i: int):
        fig = plt.figure(figsize=(16.5, 5.4))
        row = fig.add_gridspec(1, 3, wspace=0.18)
        M, hm = _pick_data(i)
        if M.shape[0] == 0:
            axes = [fig.add_subplot(row[j]) for j in range(3)]
            for ax in axes:
                ax.text(0.5, 0.5, "No particles", ha="center", va="center")
                ax.grid(alpha=0.3)
            fig.suptitle(_title_for(i))
            plt.tight_layout()
            plt.show()
            return

        _phase_space_triplet(
            fig,
            row,
            M,
            style=style,
            highlight_mask=hm,
            highlight_cmap=highlight_cmap,
            show_colorbar=bool(show_colorbar),
            prefix="",
            zpz_override_M=(datasets[i]["M"] if (clean_e and clean_except_zpz) else None),
            zpz_override_highlight_mask=(
                _resolve_highlight_mask(
                    _matrix_for_logic(datasets[i]),
                    highlight_mode=highlight_mode,
                    highlight_zlt0=bool(highlight_zlt0),
                    highlight_pzlt0=bool(highlight_pzlt0),
                    highlight_mask=datasets[i]["hmask"],
                )
                if (clean_e and clean_except_zpz) else None
            ),
        )

        fig.suptitle(_title_for(i))
        plt.tight_layout()
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
