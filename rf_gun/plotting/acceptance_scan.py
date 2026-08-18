"""3-panel summary figure for `rf_gun.acceptance_scan.scan_acceptance`."""
from __future__ import annotations

import numpy as np

from ..acceptance_scan import AcceptanceScanResult, _smooth
from .style import COLOR_PRIMARY


def plot_acceptance_scan(result: AcceptanceScanResult):
    """Panel 1: transmission vs acceptance. Panel 2: emittance vs acceptance. Panel 3: the
    (smoothed) transmission slope used to locate both thresholds. `k_core` (dashed, main-beam
    selection) is shown for reference only and is never applied; `k_trailing` (solid, trailing-
    particle removal) is the threshold actually used for tagging.
    """
    import matplotlib.pyplot as plt

    k = result.k_values
    T = result.transmission
    eps = result.emittance
    logk = np.log(k)
    slope = np.gradient(_smooth(T), logk)
    peak = slope.max()
    slope_n = slope / peak if peak > 0 else np.zeros_like(slope)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(k, 100.0 * T, "o-", ms=3, color=COLOR_PRIMARY)
    ax.set_xscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$T\,(\%)$")
    ax.set_title("Transmission vs acceptance")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(k, eps, "o-", ms=3, color=COLOR_PRIMARY)
    ax.set_xscale("log")
    # eps is all-NaN when too few particles are forward for a Courant-Snyder fit; a log scale on
    # non-positive data raises ValueError, so fall back to linear and annotate why.
    eps_finite_positive = eps[np.isfinite(eps) & (eps > 0)]
    if eps_finite_positive.size >= 2:
        ax.set_yscale("log")
    else:
        ax.text(
            0.5, 0.5, "insufficient forward particles\nfor an emittance scan",
            transform=ax.transAxes, ha="center", va="center", fontsize=9, color="gray",
        )
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\varepsilon(k)\,(\mathrm{mm}\cdot\mathrm{MeV}/c)$")
    ax.set_title("Emittance vs acceptance")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(k, slope_n, "o-", ms=3, color=COLOR_PRIMARY)
    ax.axhline(0.05, color="gray", ls=":", lw=1, label="flat threshold")
    ax.set_xscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$dT/d(\log k)$, normalized")
    ax.set_title("Transmission slope")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    n = result.n_forward
    n_core = int(result.kept_mask_core.sum())
    n_trailing = int(result.kept_mask_trailing.sum())
    pct_core = 100.0 * n_core / n if n else float("nan")
    pct_trailing = 100.0 * n_trailing / n if n else float("nan")
    for ax in axes:
        ax.axvline(
            result.k_core, color="tab:purple", ls="--", lw=1.5,
            label=rf"$k_{{\mathrm{{core}}}}={result.k_core:.2f}$ ($T={pct_core:.1f}\%$, reference only)",
        )
        ax.axvline(
            result.k_trailing, color="tab:red", ls="-", lw=1.5,
            label=rf"$k_{{\mathrm{{trailing}}}}={result.k_trailing:.2f}$ ($T={pct_trailing:.1f}\%$, applied)",
        )
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle(
        rf"Acceptance scan at Bout ($N_{{\mathrm{{fwd}}}}={n}$): "
        rf"{n - n_trailing} tagged trailing, {n_trailing - n_core} shown but not removed",
        y=1.03,
    )
    plt.tight_layout()
    plt.show()
