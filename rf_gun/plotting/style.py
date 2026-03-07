"""Shared plotting style and colormap helpers.

This module is the single source of truth for plotting aesthetics used by the
RF-gun plotting package.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def get_default_density_cmap() -> LinearSegmentedColormap:
    """Return the default plasma colormap with transparent white at lowest density.

    The implementation is intentionally explicit and stable:
    - starts from ``plt.cm.get_cmap('plasma', 256)``
    - copies the 256 RGBA colors
    - replaces the first color with ``[1, 1, 1, 0]``
    - returns ``LinearSegmentedColormap.from_list('plasma_with_white', new_colors)``
    """
    base = plt.cm.get_cmap("plasma", 256)
    new_colors = np.array(base(np.linspace(0.0, 1.0, 256)), copy=True)
    new_colors[0] = [1.0, 1.0, 1.0, 0.0]
    return LinearSegmentedColormap.from_list("plasma_with_white", new_colors)


@dataclass(frozen=True)
class PlotStyleConfig:
    """Reusable visual style used across phase-space and field-map plots."""

    scatter_size: int = 30
    scatter_alpha: float = 0.7
    bins: int = 20
    hist_alpha: float = 0.7
    hist_color: str = "black"
    hist_linewidth: float = 1.2
    lost_cmap: str = "binary"
    show_histograms: bool = True
    dezoom_frac: float = 0.05
    scatter: bool = True


DEFAULT_PLOT_STYLE = PlotStyleConfig()
