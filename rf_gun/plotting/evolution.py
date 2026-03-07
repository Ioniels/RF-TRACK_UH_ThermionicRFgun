"""Evolution plots along z."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..constants import c
from ..diagnostics import twiss_from_moments, info_get, info_get_first


def _extract_time_ns_from_info(info):
    t_mm_c = info_get_first(info, ["t", "mean_t", "mean_T"])
    if not np.isfinite(t_mm_c):
        return np.nan
    return float((t_mm_c * 1e-3 / c) * 1e9)


def plot_evolution(
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    info_snaps: Sequence[object] | None = None,
    clean_e: bool = False,
):
    """Plot beam evolution diagnostics in a 2x2 figure vs z.

    Uses RF-Track native `get_info()` outputs when provided via `info_snaps`.
    Falls back to phase-space moments from `M_snaps` if needed.
    """
    import matplotlib.pyplot as plt

    use_info = info_snaps is not None and len(info_snaps) == len(z_snaps)
    has_m = len(M_snaps) == len(z_snaps) and len(M_snaps) > 0
    if not use_info and not has_m:
        print("No snapshots available (provide info_snaps or M_snaps matching z_snaps).")
        return

    z_mm = 1e3 * np.asarray(z_snaps)

    if use_info and clean_e:
        print("Note: clean_e is ignored when using RF-Track get_info() summaries.")

    if use_info:
        sig_x = np.asarray([info_get_first(info, ["sigma_X", "sigma_x"]) for info in info_snaps])
        sig_y = np.asarray([info_get_first(info, ["sigma_Y", "sigma_y"]) for info in info_snaps])
        sig_px = np.asarray([info_get_first(info, ["sigma_Px", "sigma_px", "sigma_xp"]) for info in info_snaps])
        sig_py = np.asarray([info_get_first(info, ["sigma_Py", "sigma_py", "sigma_yp"]) for info in info_snaps])
        pz_m = np.asarray([info_get_first(info, ["mean_Pz", "mean_P", "mean_pz"]) for info in info_snaps])
        sig_pz = np.asarray([info_get_first(info, ["sigma_Pz", "sigma_P", "sigma_pz"]) for info in info_snaps])
    else:
        if info_snaps is not None and len(info_snaps) != len(z_snaps):
            print("Warning: info_snaps length mismatch; using moment-based evolution fallback.")

        cleaned = []
        for M in M_snaps:
            if clean_e and M.shape[0]:
                mask = M[:, 4] > 0.0
                cleaned.append(M[mask])
            else:
                cleaned.append(M)

        sig_x = np.array([np.std(M[:, 0]) if M.shape[0] else np.nan for M in cleaned])
        sig_y = np.array([np.std(M[:, 2]) if M.shape[0] else np.nan for M in cleaned])
        sig_px = np.array([np.std(M[:, 1]) if M.shape[0] else np.nan for M in cleaned])
        sig_py = np.array([np.std(M[:, 3]) if M.shape[0] else np.nan for M in cleaned])
        pz_m = np.array([np.mean(M[:, 5]) if M.shape[0] else np.nan for M in cleaned])
        sig_pz = np.array([np.std(M[:, 5]) if M.shape[0] else np.nan for M in cleaned])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    axes[0, 0].plot(z_mm, sig_x, "o-", ms=3, label="sigma x")
    axes[0, 0].plot(z_mm, sig_y, "o-", ms=3, label="sigma y")
    axes[0, 0].set_ylabel("RMS size [mm]")
    axes[0, 0].set_title("Transverse size vs z")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(z_mm, sig_px, "o-", ms=3, label="sigma px")
    axes[0, 1].plot(z_mm, sig_py, "o-", ms=3, label="sigma py")
    axes[0, 1].set_ylabel("RMS momentum [MeV/c]")
    axes[0, 1].set_title("Transverse momentum spread vs z")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(z_mm, pz_m, "o-", ms=3, color="tab:blue")
    axes[1, 0].set_xlabel("z [mm]")
    axes[1, 0].set_ylabel("Mean pz [MeV/c]")
    axes[1, 0].set_title("Mean longitudinal momentum vs z")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(z_mm, sig_pz, "o-", ms=3, color="tab:orange")
    axes[1, 1].set_xlabel("z [mm]")
    axes[1, 1].set_ylabel("Sigma pz [MeV/c]")
    axes[1, 1].set_title("Longitudinal momentum spread vs z")
    axes[1, 1].grid(alpha=0.3)

    src = "RF-Track get_info()" if use_info else "screen moments"
    fig.suptitle(f"Beam evolution diagnostics ({src})", y=0.98)
    plt.tight_layout()
    plt.show()


def plot_twiss_evolution(
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    info_snaps: Sequence[object] | None = None,
    clean_e: bool = False,
):
    """Plot Twiss parameter evolution along z.

    Uses RF-Track native `get_info()` outputs when provided via `info_snaps`.
    Falls back to moment reconstruction from `M_snaps` if needed.
    """
    import matplotlib.pyplot as plt

    use_info = info_snaps is not None and len(info_snaps) == len(z_snaps)
    has_m = len(M_snaps) == len(z_snaps) and len(M_snaps) > 0
    if not use_info and not has_m:
        print("No snapshots available (provide info_snaps or M_snaps matching z_snaps).")
        return

    z_mm = 1e3 * np.asarray(z_snaps)
    cleaned = []
    for M in M_snaps:
        if clean_e and M.shape[0]:
            mask = M[:, 4] > 0.0
            cleaned.append(M[mask])
        else:
            cleaned.append(M)

    alpha_x = []
    beta_x = []
    alpha_y = []
    beta_y = []
    alpha_z = []
    beta_z = []

    if use_info:
        for info in info_snaps:
            alpha_x.append(float(info_get(info, "alpha_x")))
            beta_x.append(float(info_get(info, "beta_x")))
            alpha_y.append(float(info_get(info, "alpha_y")))
            beta_y.append(float(info_get(info, "beta_y")))
            alpha_z.append(float(info_get(info, "alpha_z")))
            beta_z.append(float(info_get(info, "beta_z")))
    else:
        if info_snaps is not None and len(info_snaps) != len(z_snaps):
            print("Warning: info_snaps length mismatch; using moment-based Twiss fallback.")
        for M in cleaned:
            if M.shape[0] < 2:
                alpha_x.append(np.nan)
                beta_x.append(np.nan)
                alpha_y.append(np.nan)
                beta_y.append(np.nan)
                alpha_z.append(np.nan)
                beta_z.append(np.nan)
                continue

            ax, bx, _ = twiss_from_moments(M[:, 0], M[:, 1])
            ay, by, _ = twiss_from_moments(M[:, 2], M[:, 3])
            az, bz, _ = twiss_from_moments(M[:, 4], M[:, 5])
            alpha_x.append(ax)
            beta_x.append(bx)
            alpha_y.append(ay)
            beta_y.append(by)
            alpha_z.append(az)
            beta_z.append(bz)

    alpha_x = np.asarray(alpha_x)
    beta_x = np.asarray(beta_x)
    alpha_y = np.asarray(alpha_y)
    beta_y = np.asarray(beta_y)
    alpha_z = np.asarray(alpha_z)
    beta_z = np.asarray(beta_z)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    axes[0, 0].plot(z_mm, alpha_x, "o-", ms=3, label="alpha_x")
    axes[0, 0].plot(z_mm, alpha_y, "o-", ms=3, label="alpha_y")
    axes[0, 0].set_ylabel("alpha [-]")
    axes[0, 0].set_title("Transverse alpha vs z")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(z_mm, beta_x, "o-", ms=3, label="beta_x")
    axes[0, 1].plot(z_mm, beta_y, "o-", ms=3, label="beta_y")
    axes[0, 1].set_ylabel("beta")
    axes[0, 1].set_title("Transverse beta vs z")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(z_mm, alpha_z, "o-", ms=3, color="tab:blue")
    axes[1, 0].set_xlabel("z [mm]")
    axes[1, 0].set_ylabel("alpha_z [-]")
    axes[1, 0].set_title("Longitudinal alpha vs z")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(z_mm, beta_z, "o-", ms=3, color="tab:orange")
    axes[1, 1].set_xlabel("z [mm]")
    axes[1, 1].set_ylabel("beta_z")
    axes[1, 1].set_title("Longitudinal beta vs z")
    axes[1, 1].grid(alpha=0.3)

    src = "RF-Track get_info()" if use_info else "screen moments"
    fig.suptitle(f"Twiss evolution ({src})", y=0.98)
    plt.tight_layout()
    plt.show()


def plot_emittance_evolution(
    z_snaps: Sequence[float],
    info_snaps: Sequence[object] | None,
):
    """Plot geometric and normalized emittance evolution from RF-Track get_info().

    Notes
    -----
    - This function reads emittance values directly from ``info_snaps`` only.
    - It does **not** compute emittance from ``M_snaps`` moments/covariance.
        - If RF-Track emittance keys are not present in ``get_info()``, it returns.
    """
    import matplotlib.pyplot as plt

    if not len(z_snaps) or info_snaps is None or len(info_snaps) != len(z_snaps):
        print("No emittance data available (info_snaps missing or length mismatch).")
        return

    z_mm = 1e3 * np.asarray(z_snaps, dtype=float)

    geom_keys = {
        "x": ["emit_x", "emittance_x", "eps_x", "epsilon_x", "ex"],
        "y": ["emit_y", "emittance_y", "eps_y", "epsilon_y", "ey"],
        "z": ["emit_z", "emittance_z", "eps_z", "epsilon_z", "ez"],
    }
    norm_keys = {
        "x": ["emitt_x", "emit_nx", "emitnx", "norm_emit_x", "normalized_emittance_x", "eps_nx", "epsilon_nx"],
        "y": ["emitt_y", "emit_ny", "emitny", "norm_emit_y", "normalized_emittance_y", "eps_ny", "epsilon_ny"],
        "z": ["emitt_z", "emit_nz", "emitnz", "norm_emit_z", "normalized_emittance_z", "eps_nz", "epsilon_nz"],
    }

    eps_x = np.asarray([info_get_first(info, geom_keys["x"]) for info in info_snaps], dtype=float)
    eps_y = np.asarray([info_get_first(info, geom_keys["y"]) for info in info_snaps], dtype=float)
    eps_z = np.asarray([info_get_first(info, geom_keys["z"]) for info in info_snaps], dtype=float)

    eps_nx = np.asarray([info_get_first(info, norm_keys["x"]) for info in info_snaps], dtype=float)
    eps_ny = np.asarray([info_get_first(info, norm_keys["y"]) for info in info_snaps], dtype=float)
    eps_nz = np.asarray([info_get_first(info, norm_keys["z"]) for info in info_snaps], dtype=float)

    has_geom_x = np.any(np.isfinite(eps_x))
    has_geom_y = np.any(np.isfinite(eps_y))
    has_geom_z = np.any(np.isfinite(eps_z))
    has_norm_x = np.any(np.isfinite(eps_nx))
    has_norm_y = np.any(np.isfinite(eps_ny))
    has_norm_z = np.any(np.isfinite(eps_nz))

    has_any_transverse = has_geom_x or has_geom_y or has_norm_x or has_norm_y
    has_any_longitudinal = has_geom_z or has_norm_z
    if not (has_any_transverse or has_any_longitudinal):
        return

    if has_any_transverse:
        fig_xy, ax_xy = plt.subplots(figsize=(8.5, 4.2))
        if has_geom_x:
            ax_xy.plot(z_mm, eps_x, "o-", ms=3, color="tab:blue", label=r"$\varepsilon_{x}$ (geom)")
        if has_geom_y:
            ax_xy.plot(z_mm, eps_y, "o-", ms=3, color="tab:orange", label=r"$\varepsilon_{y}$ (geom)")
        if has_norm_x:
            ax_xy.plot(z_mm, eps_nx, "--", lw=1.4, color="tab:blue", alpha=0.8, label=r"$\varepsilon_{n,x}$")
        if has_norm_y:
            ax_xy.plot(z_mm, eps_ny, "--", lw=1.4, color="tab:orange", alpha=0.8, label=r"$\varepsilon_{n,y}$")
        ax_xy.set_xlabel(r"$z\,(\mathrm{mm})$")
        ax_xy.set_ylabel(r"$\varepsilon$")
        ax_xy.set_title(r"$\mathrm{Transverse\ emittance\ evolution}$")
        ax_xy.grid(alpha=0.3)
        ax_xy.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.show()

    if has_any_longitudinal:
        fig_z, ax_z = plt.subplots(figsize=(8.5, 4.2))
        if has_geom_z:
            ax_z.plot(z_mm, eps_z, "o-", ms=3, color="tab:green", label=r"$\varepsilon_{z}$ (geom)")
        if has_norm_z:
            ax_z.plot(z_mm, eps_nz, "--", lw=1.4, color="tab:green", alpha=0.8, label=r"$\varepsilon_{n,z}$")
        ax_z.set_xlabel(r"$z\,(\mathrm{mm})$")
        ax_z.set_ylabel(r"$\varepsilon$")
        ax_z.set_title(r"$\mathrm{Longitudinal\ emittance\ evolution}$")
        ax_z.grid(alpha=0.3)
        ax_z.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.show()

def plot_transmission_evolution(
    z_snaps: Sequence[float],
    info_snaps: Sequence[object],
    n_real_ref: float | None = None,
    n_macroparticles: int | None = None,
):
    """Plot transmission vs z from RF-Track get_info().

    `info_snaps` transmission is RF-Track's *real-particle equivalent* count, not
    the number of simulated macroparticles.
    """
    import matplotlib.pyplot as plt

    if not len(z_snaps) or info_snaps is None or len(info_snaps) != len(z_snaps):
        print("No transmission data available (info_snaps missing or length mismatch).")
        return

    z_mm = 1e3 * np.asarray(z_snaps)
    transmission = np.asarray([float(info_get(info, "transmission")) for info in info_snaps])
    t_ns = np.asarray([_extract_time_ns_from_info(info) for info in info_snaps])

    fig, ax = plt.subplots(figsize=(8, 4))
    if n_real_ref is not None and np.isfinite(n_real_ref) and n_real_ref > 0:
        frac = 100.0 * transmission / float(n_real_ref)
        ax.plot(z_mm, frac, "o-", ms=3, color="tab:blue", label=r"fraction of emitted charge")
        ax.set_ylabel(r"$\mathrm{Transmission}\,(\%\ \mathrm{of\ emitted\ charge})$")

        if n_macroparticles is not None and int(n_macroparticles) > 0:
            macro_eq = transmission / float(n_real_ref) * int(n_macroparticles)
            ax2 = ax.twinx()
            ax2.plot(z_mm, macro_eq, "--", lw=1.3, color="tab:green", label=r"macro-equivalent")
            ax2.set_ylabel(r"$N_{\mathrm{macro,eq}}$")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, loc="best")
    else:
        ax.plot(z_mm, transmission, "o-", ms=3, color="tab:green", label=r"real-particle equivalent")
        ax.set_ylabel(r"$N_{\mathrm{real,eq}}$")
        ax.legend(frameon=False)

    ax.set_xlabel(r"$z\,(\mathrm{mm})$")
    ax.set_title(r"$\mathrm{Transmission\ vs\ }z\ (\mathrm{RF\!\!\!-Track\ get\_info})$")
    if np.any(np.isfinite(t_ns)):
        i0 = int(np.nanargmin(np.abs(z_mm - z_mm[0])))
        i1 = int(np.nanargmin(np.abs(z_mm - z_mm[-1])))
        ax.text(
            0.02,
            0.97,
            f"t span ≈ {t_ns[i0]:.3f} to {t_ns[i1]:.3f} ns",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
