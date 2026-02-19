"""Phase space and spectrum plots."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..constants import c


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


def _scatter_density(ax, x: np.ndarray, y: np.ndarray, bins: int = 80, cmap: str = "hot_r"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return

    x_bins = np.histogram_bin_edges(x, bins=bins)
    y_bins = np.histogram_bin_edges(y, bins=bins)
    hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])

    ix = np.clip(np.digitize(x, x_bins) - 1, 0, hist.shape[0] - 1)
    iy = np.clip(np.digitize(y, y_bins) - 1, 0, hist.shape[1] - 1)
    density = hist[ix, iy]

    order = np.argsort(density)
    ax.scatter(x[order], y[order], c=density[order], cmap=cmap, s=10, alpha=0.7, edgecolors="none")


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
                t_emit_ns_good = t_emit_exit[mask_good := (finite_z & (Mf_f_all[:, 4] > 0.0))] * 1e9
                t_emit_ns_bad = t_emit_exit[mask_bad := (finite_z & (Mf_f_all[:, 4] <= 0.0))] * 1e9
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
        axes[0].set_ylabel("Counts")
        axes[0].grid(alpha=0.3)
        axes[0].set_title("Initial emission distribution vs time")
        if t_emit_ns_bad_plot.size > 0:
            axes[0].legend(frameon=False)
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "Emission-time distribution not available", ha="center", va="center")

    axes[1].hist(pz_f, bins=60, alpha=0.8, edgecolor="black", lw=0.5, color="tab:blue")
    axes[1].set_xlabel("Pz [MeV/c]")
    axes[1].set_ylabel("Counts")
    axes[1].grid(alpha=0.3)
    axes[1].set_title(f"Final longitudinal momentum @ phi = {transport_phase_deg:.1f} deg")

    if tof_ns.size > 0:
        axes[2].hist(tof_ns, bins=60, alpha=0.8, edgecolor="black", lw=0.5, color="tab:green")
        axes[2].set_xlabel("ToF [ns]")
        axes[2].set_ylabel("Counts")
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
):
    """Plot launch and exit phase space, with launch pz histogram colored by loss."""
    import matplotlib.pyplot as plt

    try:
        Mf_f_all = _safe_get_phase_space(Bout, "all", phase_fmt)
        Mf_launch_all = _safe_get_phase_space(B0, "all", phase_fmt)
        has_all = True
    except Exception:
        Mf_f_all = _safe_get_phase_space(Bout, "good", phase_fmt)
        Mf_launch_all = _safe_get_phase_space(B0, "good", phase_fmt)
        has_all = False

    finite_z = np.isfinite(Mf_f_all[:, 4])
    mask_bad = finite_z & (Mf_f_all[:, 4] <= 0.0)
    mask_good = finite_z & (Mf_f_all[:, 4] > 0.0)

    Mf_f_good = Mf_f_all[mask_good]
    Mf_f = Mf_f_good if clean_e else Mf_f_all[finite_z]

    ids_exit = _try_get_ids(Bout, "all") if has_all else None
    ids_launch = _try_get_ids(B0, "all") if has_all else None
    lost_ids = None
    if ids_exit is not None and ids_exit.size == Mf_f_all.shape[0]:
        lost_ids = ids_exit[mask_bad]
        print(f"Lost particles (z <= 0): {lost_ids.size} of {ids_exit.size}")
        if lost_ids.size:
            preview = np.array2string(lost_ids[:20], separator=", ")
            print(f"Lost particle IDs (first 20): {preview}")

    total_tracked = int(np.sum(finite_z))
    lost_count = int(np.sum(mask_bad))
    good_count = int(np.sum(mask_good))
    transmission_pct = 100.0 * good_count / total_tracked if total_tracked > 0 else 0.0
    print(f"Lost particles (z <= 0): {lost_count} / {total_tracked}")
    print(f"Transmission (z > 0): {transmission_pct:.2f}% (tracked)")

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

    Mf_launch_bad = Mf_launch_all[launch_mask_bad] if launch_mask_bad is not None else np.empty((0, 6))

    if Mf_f.shape[0] == 0 or Mf_launch.shape[0] == 0:
        print("No particles to plot in phase space.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    _scatter_density(axes[0, 0], Mf_launch[:, 0], Mf_launch[:, 1], bins=80, cmap="hot")
    if show_zle0 and Mf_launch_bad.shape[0] > 0:
        axes[0, 0].scatter(Mf_launch_bad[:, 0], Mf_launch_bad[:, 1], s=8, alpha=0.55, c="red", edgecolors="none")
    axes[0, 0].set_xlabel("x [mm]")
    axes[0, 0].set_ylabel("px [MeV/c]")
    axes[0, 0].set_title("Launch: x-px")
    axes[0, 0].grid(alpha=0.3)

    _scatter_density(axes[0, 1], Mf_launch[:, 2], Mf_launch[:, 3], bins=80, cmap="hot")
    if show_zle0 and Mf_launch_bad.shape[0] > 0:
        axes[0, 1].scatter(Mf_launch_bad[:, 2], Mf_launch_bad[:, 3], s=8, alpha=0.55, c="red", edgecolors="none")
    axes[0, 1].set_xlabel("y [mm]")
    axes[0, 1].set_ylabel("py [MeV/c]")
    axes[0, 1].set_title("Launch: y-py")
    axes[0, 1].grid(alpha=0.3)

    pz_bins = 60
    pz_good = Mf_launch[:, 5]
    bins = np.histogram_bin_edges(pz_good, bins=pz_bins)
    axes[0, 2].hist(pz_good, bins=bins, alpha=0.8, color="tab:blue", label="launch")
    if show_zle0 and Mf_launch_bad.shape[0] > 0:
        axes[0, 2].hist(
            Mf_launch_bad[:, 5],
            bins=bins,
            alpha=0.7,
            color="tab:red",
            label="launch z<=0",
        )
    axes[0, 2].set_xlabel("pz [MeV/c]")
    axes[0, 2].set_ylabel("Counts")
    axes[0, 2].set_title("Launch: pz histogram")
    axes[0, 2].legend(frameon=False)
    axes[0, 2].grid(alpha=0.3)

    _scatter_density(axes[1, 0], Mf_f[:, 0], Mf_f[:, 1], bins=80, cmap="hot")
    axes[1, 0].set_xlabel("x [mm]")
    axes[1, 0].set_ylabel("px [MeV/c]")
    axes[1, 0].set_title("Exit: x-px")
    axes[1, 0].grid(alpha=0.3)

    _scatter_density(axes[1, 1], Mf_f[:, 2], Mf_f[:, 3], bins=80, cmap="hot")
    axes[1, 1].set_xlabel("y [mm]")
    axes[1, 1].set_ylabel("py [MeV/c]")
    axes[1, 1].set_title("Exit: y-py")
    axes[1, 1].grid(alpha=0.3)

    _scatter_density(axes[1, 2], Mf_f[:, 4], Mf_f[:, 5], bins=80, cmap="hot")
    axes[1, 2].set_xlabel("z [mm]")
    axes[1, 2].set_ylabel("pz [MeV/c]")
    axes[1, 2].set_title("Exit: z-pz")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
