"""Phase scan plots."""
from __future__ import annotations

import numpy as np

from .style import COLOR_PRIMARY, COLOR_SECONDARY


def phase_plot(
    phi_abs: np.ndarray,
    pz_mean: np.ndarray,
    crest_phi_abs_deg: float | None = None,
    crest_pz_mean_MeV_c: float | None = None,
):
    """Plot the RF-only phase scan: mean pz vs. absolute phase.

    Valid and lost/invalid samples are shown distinctly (lost points are marked, not silently
    dropped), the line is broken across the circular wrap and across any invalid interval (never
    bridging a gap or connecting the physically-identical 0/360 endpoints), and the crest is a
    single marker -- callers should print the crest value/quality report themselves (e.g. from a
    `PhaseCalibrationResult`) rather than relying on the in-plot annotation this used to draw.
    Pass `crest_phi_abs_deg`/`crest_pz_mean_MeV_c` explicitly (e.g. from a validated
    `PhaseCalibrationResult`) to mark the actual accepted crest rather than a naive `nanargmax`.
    """
    import matplotlib.pyplot as plt

    phi_abs = np.asarray(phi_abs, dtype=float)
    pz_mean = np.asarray(pz_mean, dtype=float)
    order = np.argsort(phi_abs)
    phi_sorted = phi_abs[order]
    pz_sorted = pz_mean[order]
    mask = np.isfinite(pz_sorted)

    fig, ax = plt.subplots(figsize=(8, 4.2))

    # Plot only contiguous finite runs -- never bridges a gap or the circular 0/360 wrap.
    run_start = None
    first_run = True
    for i, ok in enumerate(mask):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            ax.plot(
                phi_sorted[run_start:i], pz_sorted[run_start:i], "o-", ms=4, lw=1.6,
                color=COLOR_PRIMARY, label=("valid" if first_run else None),
            )
            first_run = False
            run_start = None
    if run_start is not None:
        ax.plot(
            phi_sorted[run_start:], pz_sorted[run_start:], "o-", ms=4, lw=1.6,
            color=COLOR_PRIMARY, label=("valid" if first_run else None),
        )

    if np.any(~mask):
        ax.plot(
            phi_sorted[~mask], np.zeros(int(np.sum(~mask))), "x", ms=6, color="0.6",
            label="lost/invalid",
        )

    ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.6)

    if crest_phi_abs_deg is None and np.any(mask):
        i_max = int(np.nanargmax(pz_sorted))
        crest_phi_abs_deg = float(phi_sorted[i_max])
        crest_pz_mean_MeV_c = float(pz_sorted[i_max])
    if crest_phi_abs_deg is not None and crest_pz_mean_MeV_c is not None and np.isfinite(crest_pz_mean_MeV_c):
        ax.plot(
            [crest_phi_abs_deg], [crest_pz_mean_MeV_c], "*", ms=14, color=COLOR_SECONDARY,
            zorder=5, label="crest",
        )

    ax.set_xlabel(r"RF phase (deg) (absolute)")
    ax.set_ylabel(r"Mean $p_z$ at exit (MeV/c)")
    ax.set_title("Fast phase scan: mean pz vs phase")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
