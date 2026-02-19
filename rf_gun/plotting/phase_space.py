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


def plot_spectra(
    Bout,
    transport_phase_deg: float,
    phase_fmt: str = "%X %Px %Y %Py %Z %Pz",
):
    """Plot final pz histogram and ToF histogram."""
    import matplotlib.pyplot as plt

    try:
        Mf_f_all = _safe_get_phase_space(Bout, "all", phase_fmt)
    except Exception:
        Mf_f_all = _safe_get_phase_space(Bout, "good", phase_fmt)

    finite_z = np.isfinite(Mf_f_all[:, 4])
    Mf_f = Mf_f_all[finite_z]
    if Mf_f.shape[0] == 0:
        print("No particles in output bunch.")
        return

    pz_f = Mf_f[:, 5]
    tof_ns = (Mf_f[:, 4] * 1e-3 / c) * 1e9
    tof_ns = tof_ns[np.isfinite(tof_ns)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pz_f, bins=60, alpha=0.8, edgecolor="black", lw=0.5, color="tab:blue")
    axes[0].set_xlabel("Pz [MeV/c]")
    axes[0].set_ylabel("Counts")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Final longitudinal momentum @ phi = {transport_phase_deg:.1f} deg")

    if tof_ns.size > 0:
        axes[1].hist(tof_ns, bins=60, alpha=0.8, edgecolor="black", lw=0.5, color="tab:green")
        axes[1].set_xlabel("ToF [ns]")
        axes[1].set_ylabel("Counts")
        axes[1].grid(alpha=0.3)
        axes[1].set_title("Time of flight histogram")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "ToF not available", ha="center", va="center")

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

    axes[0, 0].plot(Mf_launch[:, 0], Mf_launch[:, 1], ".", ms=2, alpha=0.5)
    if show_zle0 and Mf_launch_bad.shape[0] > 0:
        axes[0, 0].plot(Mf_launch_bad[:, 0], Mf_launch_bad[:, 1], ".", ms=2, alpha=0.7, color="red")
    axes[0, 0].set_xlabel("x [mm]")
    axes[0, 0].set_ylabel("px [MeV/c]")
    axes[0, 0].set_title("Launch: x-px")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(Mf_launch[:, 2], Mf_launch[:, 3], ".", ms=2, alpha=0.5)
    if show_zle0 and Mf_launch_bad.shape[0] > 0:
        axes[0, 1].plot(Mf_launch_bad[:, 2], Mf_launch_bad[:, 3], ".", ms=2, alpha=0.7, color="red")
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

    axes[1, 0].plot(Mf_f[:, 0], Mf_f[:, 1], ".", ms=2, alpha=0.5)
    axes[1, 0].set_xlabel("x [mm]")
    axes[1, 0].set_ylabel("px [MeV/c]")
    axes[1, 0].set_title("Exit: x-px")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(Mf_f[:, 2], Mf_f[:, 3], ".", ms=2, alpha=0.5)
    axes[1, 1].set_xlabel("y [mm]")
    axes[1, 1].set_ylabel("py [MeV/c]")
    axes[1, 1].set_title("Exit: y-py")
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(Mf_f[:, 4], Mf_f[:, 5], ".", ms=2, alpha=0.5)
    axes[1, 2].set_xlabel("z [mm]")
    axes[1, 2].set_ylabel("pz [MeV/c]")
    axes[1, 2].set_title("Exit: z-pz")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
