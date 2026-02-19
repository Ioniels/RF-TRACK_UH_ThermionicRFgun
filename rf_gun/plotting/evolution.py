"""Evolution plots along z."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _twiss_from_moments(u: np.ndarray, pu: np.ndarray):
    if u.size < 2 or pu.size < 2:
        return np.nan, np.nan, np.nan
    u0 = u - np.mean(u)
    pu0 = pu - np.mean(pu)
    s11 = float(np.mean(u0 * u0))
    s22 = float(np.mean(pu0 * pu0))
    s12 = float(np.mean(u0 * pu0))
    det = s11 * s22 - s12 * s12
    if not np.isfinite(det) or det <= 0.0:
        return np.nan, np.nan, np.nan
    eps = np.sqrt(det)
    alpha = -s12 / eps
    beta = s11 / eps
    gamma = s22 / eps
    return alpha, beta, gamma


def _info_get(info, key: str):
    if info is None:
        return np.nan
    if isinstance(info, dict):
        if key in info:
            return info[key]
        if key.lower() in info:
            return info[key.lower()]
        if key.upper() in info:
            return info[key.upper()]
        return np.nan
    if hasattr(info, key):
        val = getattr(info, key)
        return val() if callable(val) else val
    if hasattr(info, key.lower()):
        val = getattr(info, key.lower())
        return val() if callable(val) else val
    if hasattr(info, key.upper()):
        val = getattr(info, key.upper())
        return val() if callable(val) else val
    if hasattr(info, f"get_{key}"):
        getter = getattr(info, f"get_{key}")
        return getter() if callable(getter) else getter
    return np.nan


def _info_get_first(info, keys: Sequence[str]):
    for key in keys:
        val = _info_get(info, key)
        try:
            fval = float(val)
        except Exception:
            continue
        if np.isfinite(fval):
            return fval
    return np.nan


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

    if not len(M_snaps) or len(z_snaps) != len(M_snaps):
        print("No snapshots available (M_snaps empty or z_snaps mismatch).")
        return

    z_mm = 1e3 * np.asarray(z_snaps)
    use_info = info_snaps is not None and len(info_snaps) == len(z_snaps)

    if use_info and clean_e:
        print("Note: clean_e is ignored when using RF-Track get_info() summaries.")

    if use_info:
        sig_x = np.asarray([_info_get_first(info, ["sigma_X", "sigma_x"]) for info in info_snaps])
        sig_y = np.asarray([_info_get_first(info, ["sigma_Y", "sigma_y"]) for info in info_snaps])
        sig_px = np.asarray([_info_get_first(info, ["sigma_Px", "sigma_px", "sigma_xp"]) for info in info_snaps])
        sig_py = np.asarray([_info_get_first(info, ["sigma_Py", "sigma_py", "sigma_yp"]) for info in info_snaps])
        pz_m = np.asarray([_info_get_first(info, ["mean_Pz", "mean_P", "mean_pz"]) for info in info_snaps])
        sig_pz = np.asarray([_info_get_first(info, ["sigma_Pz", "sigma_P", "sigma_pz"]) for info in info_snaps])
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

    if not len(M_snaps) or len(z_snaps) != len(M_snaps):
        print("No snapshots available (M_snaps empty or z_snaps mismatch).")
        return

    z_mm = 1e3 * np.asarray(z_snaps)
    cleaned = []
    for M in M_snaps:
        if clean_e and M.shape[0]:
            mask = M[:, 4] > 0.0
            cleaned.append(M[mask])
        else:
            cleaned.append(M)

    use_info = info_snaps is not None and len(info_snaps) == len(z_snaps)

    alpha_x = []
    beta_x = []
    alpha_y = []
    beta_y = []
    alpha_z = []
    beta_z = []

    if use_info:
        for info in info_snaps:
            alpha_x.append(float(_info_get(info, "alpha_x")))
            beta_x.append(float(_info_get(info, "beta_x")))
            alpha_y.append(float(_info_get(info, "alpha_y")))
            beta_y.append(float(_info_get(info, "beta_y")))
            alpha_z.append(float(_info_get(info, "alpha_z")))
            beta_z.append(float(_info_get(info, "beta_z")))
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

            ax, bx, _ = _twiss_from_moments(M[:, 0], M[:, 1])
            ay, by, _ = _twiss_from_moments(M[:, 2], M[:, 3])
            az, bz, _ = _twiss_from_moments(M[:, 4], M[:, 5])
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


def plot_transmission_evolution(
    z_snaps: Sequence[float],
    info_snaps: Sequence[object],
):
    """Plot transmission (number of particles in bunch) vs z from RF-Track get_info()."""
    import matplotlib.pyplot as plt

    if not len(z_snaps) or info_snaps is None or len(info_snaps) != len(z_snaps):
        print("No transmission data available (info_snaps missing or length mismatch).")
        return

    z_mm = 1e3 * np.asarray(z_snaps)
    transmission = np.asarray([float(_info_get(info, "transmission")) for info in info_snaps])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z_mm, transmission, "o-", ms=3, color="tab:green")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Transmission [particles]")
    ax.set_title("Transmission vs z (RF-Track get_info)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
