"""Where the dynamic aperture (`rf_gun.aperture`) actually removed particles during tracking.

Reads directly from RF-Track's own lost-particle table (`SimulationResult.lost_table`) -- a real,
physically meaningful loss location, since a particle is removed the instant it crosses the
channel R(z) during tracking.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..aperture import aperture_radius_profile_mm
from .phase_space import phase_space_density
from .style import DEFAULT_PLOT_STYLE, PlotStyleConfig, COLOR_PRIMARY, get_default_density_cmap


def _fixed_width_bin_edges(x: np.ndarray, bin_width: float) -> np.ndarray:
    """Bin edges of exactly `bin_width` spanning `x`'s own range -- a fixed physical bin size
    (rather than a fixed bin *count*) so the resolution stays meaningful regardless of how few or
    many loss points there are, matching the ~mm-scale features of the aperture geometry itself."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    bin_width = float(bin_width)
    if x.size == 0:
        return np.array([-0.5 * bin_width, 0.5 * bin_width])
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        lo, hi = lo - 0.5 * bin_width, hi + 0.5 * bin_width
    n_bins = max(1, int(np.ceil((hi - lo) / bin_width)))
    return lo + bin_width * np.arange(n_bins + 1)


def plot_dynamic_aperture_losses(
    lost_table: Optional[np.ndarray],
    z_grid_m: np.ndarray,
    delta_cathode_chamfer_mm: float,
    *,
    bin_width_mm: float = 0.5,
    style: PlotStyleConfig | None = None,
):
    """Two panels: (left) where each lost particle was removed, r vs z, as a density-colored
    scatter (same KDE-density engine and colormap as the phase-space plots) with R(z) overlaid;
    (right) a histogram of loss z-location, fixed `bin_width_mm`-wide bins.

    Returns the figure, or `None` if `lost_table` is empty/unavailable.
    """
    import matplotlib.pyplot as plt

    style = DEFAULT_PLOT_STYLE if style is None else style

    if lost_table is None or np.asarray(lost_table).shape[0] == 0:
        print("No particles were removed by the dynamic aperture.")
        return None

    arr = np.asarray(lost_table, dtype=float)
    z_mm = arr[:, 4]
    r_mm = np.hypot(arr[:, 0], arr[:, 2])

    z_curve_mm = np.asarray(z_grid_m, dtype=float) * 1e3
    r_curve_mm = aperture_radius_profile_mm(z_curve_mm, float(delta_cathode_chamfer_mm))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(z_curve_mm, r_curve_mm, "k--", lw=1.4, label="R(z)")
    phase_space_density(
        axes[0], z_mm, r_mm,
        scatter=bool(style.scatter),
        bins=int(style.bins),
        scatter_size=int(style.scatter_size),
        scatter_alpha=float(style.scatter_alpha),
        cmap=get_default_density_cmap(),
        zorder=2,
    )
    axes[0].set_xlabel(r"$z\,(\mathrm{mm})$")
    axes[0].set_ylabel(r"$r\,(\mathrm{mm})$")
    axes[0].set_title("Where particles were removed vs. R(z)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.3)

    edges = _fixed_width_bin_edges(z_mm, bin_width_mm)
    axes[1].hist(z_mm, bins=edges, color=COLOR_PRIMARY, alpha=0.85, edgecolor="black", lw=0.3)
    axes[1].set_xlabel(r"$z\,(\mathrm{mm})$ of loss")
    axes[1].set_ylabel("Counts")
    axes[1].set_title(f"Loss location ({int(arr.shape[0])} particles, {edges.size - 1} bins)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig
