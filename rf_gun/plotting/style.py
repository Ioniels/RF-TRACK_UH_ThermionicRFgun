"""Shared plotting style and colormap helpers.

This module is the single source of truth for plotting aesthetics used by the
RF-gun plotting package.

No system LaTeX is installed in this environment, so `text.usetex=True` is not viable (it would
error at render time) -- `mathtext.fontset='cm'` gives the same Computer-Modern "LaTeX look" for
anything inside `$...$` using matplotlib's own bundled glyphs, with a serif `font.family` for
regular text. Applied once, at import time, project-wide.

`COLOR_PRIMARY`/`COLOR_SECONDARY`/`COLOR_NEUTRAL` are the project's standard curve palette for
figures with one or two curves (a nice blue/red/gray), replacing what used to be an ad hoc
"tab:purple"/"tab:brown"/etc. per call site. Convention: x-plane vs y-plane pairs -> blue/red;
mean-type vs sigma/spread-type single curves -> blue/red; a de-emphasized third "baseline" curve
(e.g. an unfiltered reference) -> gray. Density/colormap-based panels (KDE scatter, colorbars,
per-particle categorical colors) are a different visual channel and are not part of this
convention -- units in labels use parentheses, not brackets (`x\\,(\\mathrm{mm})`, not `x [mm]`).
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
plt.rcParams["axes.formatter.use_mathtext"] = True

#: Standard curve palette -- see module docstring for the convention.
COLOR_PRIMARY = "#0072B2"
COLOR_SECONDARY = "#C0392B"
COLOR_NEUTRAL = "#7F7F7F"


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


def get_aperture_loss_cmap() -> LinearSegmentedColormap:
    """Green-gradient colormap for aperture-clipped particles, transparent at lowest density.

    Same construction pattern as `get_default_density_cmap` (used for the "normal" population)
    and `PlotStyleConfig.lost_cmap`/`"binary"` (used for backward-loss highlighting) -- this is
    the third, independent color channel for `FIG_EXCLUDE_APERTURE_LOSSES=False` highlighting.
    """
    base = plt.cm.get_cmap("Greens", 256)
    new_colors = np.array(base(np.linspace(0.0, 1.0, 256)), copy=True)
    new_colors[0] = [1.0, 1.0, 1.0, 0.0]
    return LinearSegmentedColormap.from_list("greens_with_white", new_colors)


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
