"""Shared plotting style and colormap helpers.

This module is the single source of truth for plotting aesthetics used by the
RF-gun plotting package.

No system LaTeX is installed in this environment, so `text.usetex=True` is not viable (it would
error at render time) -- `mathtext.fontset='cm'` gives the same Computer-Modern "LaTeX look" for
anything inside `$...$` using matplotlib's own bundled glyphs, with a serif `font.family` for
regular text. Applied once, at import time, project-wide.

`COLOR_PRIMARY`/`COLOR_SECONDARY`/`COLOR_NEUTRAL` are the project's standard curve palette for
figures with one or two curves (a nice blue/red/gray). Convention: x-plane vs y-plane pairs -> blue/red;
mean-type vs sigma/spread-type single curves -> blue/red; a de-emphasized third "baseline" curve
(e.g. an unfiltered reference) -> gray. `COLOR_LOST` is the one fixed exception to that
two-color convention: green specifically and only means "removed by the dynamic aperture,"
everywhere that category appears (`_initial_pz_hist_panel`'s launch pz histogram,
`plot_spectra`'s output panels), so a reader learns the color once and can rely on it in every
figure. Density/colormap-based panels (KDE scatter, colorbars, per-particle categorical colors)
are a different visual channel and are not part of this convention -- units in labels use
parentheses, not brackets (`x\\,(\\mathrm{mm})`, not `x [mm]`).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import matplotlib as mpl
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
plt.rcParams["axes.formatter.use_mathtext"] = True

# Phase-space panels draw mirrored marginal histograms as inset axes (see
# phase_space._phase_space_panel), which tight_layout() cannot lay out and warns about on every
# such figure. The layout is correct in practice (checked visually) -- benign, so silenced here.
warnings.filterwarnings(
    "ignore",
    message="This figure includes Axes that are not compatible with tight_layout",
    category=UserWarning,
)

#: Standard curve palette -- see module docstring for the convention.
COLOR_PRIMARY = "#0072B2"
COLOR_SECONDARY = "#C0392B"
COLOR_NEUTRAL = "#7F7F7F"
COLOR_LOST = "tab:green"

#: Per-model color convention (implementation guide Sec. 4.1), used consistently across every
#: emission-model figure (history overlay, sensitivity panels, comparison curves) so a reader
#: learns a model's color once. Solid lines for fast production models; the direct reference uses
#: a dark dashed line (guide's own convention) rather than a distinct color. Keyed by canonical
#: model name (rf_gun.emission_models.EMISSION_MODEL_NAMES); old pre-refactor names still resolve
#: to the same colors via EMISSION_MODEL_ALIASES, so callers should look up
#: `EMISSION_MODEL_COLORS[canonical_emission_model_name(m)]` rather than indexing by a possibly-old
#: name directly.
EMISSION_MODEL_COLORS = {
    "RDSchottky": "tab:blue",
    "jensen2014_RDSchottky_MurphyGood_additive": "tab:orange",
    "jensen_gtf_2007": "tab:purple",
    "jensen2019_RDSchottky_MurphyGood_transition": "tab:green",
    "murphygood1956_SchottkyNordheim_integral": "tab:red",
}


@lru_cache(maxsize=1)
def get_default_density_cmap() -> LinearSegmentedColormap:
    """Return the default plasma colormap with transparent white at lowest density.

    The implementation is intentionally explicit and stable:
    - starts from ``matplotlib.colormaps['plasma'].resampled(256)``
    - copies the 256 RGBA colors
    - replaces the first color with ``[1, 1, 1, 0]``
    - returns ``LinearSegmentedColormap.from_list('plasma_with_white', new_colors)``

    Uses `matplotlib.colormaps[name]` (the API since Matplotlib 3.7) rather than the removed
    `plt.cm.get_cmap(name, N)` -- that call was fully removed (not just deprecated) in Matplotlib
    3.9, so this function raised AttributeError on any modern Matplotlib install.
    """
    base = mpl.colormaps["plasma"].resampled(256)
    new_colors = np.array(base(np.linspace(0.0, 1.0, 256)), copy=True)
    new_colors[0] = [1.0, 1.0, 1.0, 0.0]
    return LinearSegmentedColormap.from_list("plasma_with_white", new_colors)


@lru_cache(maxsize=1)
def get_lost_cmap() -> LinearSegmentedColormap:
    """Green-gradient colormap for particles removed by the dynamic aperture, transparent at
    lowest density.

    Same construction pattern as `get_default_density_cmap` (used for the "normal" population)
    and `PlotStyleConfig.backward_cmap`/`"binary"` (used for backward-loss highlighting) -- this
    is the third, independent color channel for `exclude_lost=False` highlighting.
    """
    base = mpl.colormaps["Greens"].resampled(256)
    new_colors = np.array(base(np.linspace(0.0, 1.0, 256)), copy=True)
    new_colors[0] = [1.0, 1.0, 1.0, 0.0]
    return LinearSegmentedColormap.from_list("greens_with_white", new_colors)


def get_recentered_diverging_cmap(
    base: str = "RdBu_r",
    over_color: str = "#3f0000",
    under_color: str = "#001433",
) -> LinearSegmentedColormap:
    """Diverging colormap for use with a `Normalize(vmin=-v, vmax=v)` tighter than the data's own
    range, so lower-magnitude field structure gets more color resolution; values beyond `+-v` are
    clipped to `over_color`/`under_color` instead of saturating to the colormap's own endpoint.
    """
    cmap = plt.get_cmap(base).copy()
    cmap.set_over(over_color)
    cmap.set_under(under_color)
    return cmap


def add_reference_lines(
    ax,
    *,
    cathode_z_mm: Optional[float] = 0.0,
    z_end_mm: Optional[float] = None,
    lambda_quarter_mm: Optional[float] = None,
    halo: bool = False,
    lw: float = 1.2,
    alpha: float = 0.9,
    zorder: float = 5,
) -> None:
    """Shared reference lines for `fields.field_maps`/`fields.axis_phase`: cathode (z=0) and
    z_end solid, lambda/4 dotted. `halo=True` adds a white stroke so black stays legible over a
    dark colormap panel. Any position left `None` skips that line. See `add_aperture_curve` for
    the dynamic aperture's R(z) profile.
    """
    effects = [patheffects.withStroke(linewidth=lw + 1.8, foreground="white", alpha=0.9)] if halo else None

    def _line(x: Optional[float], ls: str, label: str) -> None:
        if x is None:
            return
        ax.axvline(float(x), color="black", ls=ls, lw=lw, alpha=alpha, label=label, path_effects=effects, zorder=zorder)

    _line(cathode_z_mm, "-", "Cathode (z=0)")
    _line(z_end_mm, "-", r"$z_{\mathrm{end}}$")
    _line(lambda_quarter_mm, ":", r"$\lambda/4$")


def add_aperture_curve(
    ax,
    z_mm: np.ndarray,
    r_mm: np.ndarray,
    *,
    color: str = "black",
    lw: float = 1.4,
    alpha: float = 0.9,
    halo: bool = True,
    zorder: float = 6,
    label: str = "Dynamic aperture R(z)",
) -> None:
    """Draw the dynamic aperture's R(z) profile as +/-r_mm curves on top of an (r, z) panel
    (`fields.field_maps`'s two bottom heatmaps). Mirrors `add_reference_lines`'s styling (white
    halo so it stays legible over a dark colormap)."""
    effects = [patheffects.withStroke(linewidth=lw + 1.8, foreground="white", alpha=0.9)] if halo else None
    z_mm = np.asarray(z_mm, dtype=float)
    r_mm = np.asarray(r_mm, dtype=float)
    ax.plot(z_mm, r_mm, color=color, lw=lw, alpha=alpha, ls="--", path_effects=effects, zorder=zorder, label=label)
    ax.plot(z_mm, -r_mm, color=color, lw=lw, alpha=alpha, ls="--", path_effects=effects, zorder=zorder)


def add_cathode_boundary_circle(
    ax,
    radius_mm: float,
    *,
    color: str = "white",
    lw: float = 1.4,
    alpha: float = 0.95,
    ls: str = "--",
    zorder: float = 6,
    label: str = "Cathode boundary",
) -> None:
    """Draw the cathode disk's boundary (radius `radius_mm`, centered on the emission axis) as a
    circle on an (x, y) cathode-plane panel -- shared across every such figure (back-bombardment
    energy density, near-cathode emission-iteration state) so the boundary looks identical
    wherever it appears. A black halo (mirroring `add_reference_lines`/`add_aperture_curve`'s own
    styling) keeps the default white line legible over both a dark heatmap background and a
    bright, near-white hot spot.
    """
    effects = [patheffects.withStroke(linewidth=lw + 1.6, foreground="black", alpha=0.85)]
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    r = float(radius_mm)
    ax.plot(
        r * np.cos(theta), r * np.sin(theta),
        color=color, lw=lw, alpha=alpha, ls=ls, path_effects=effects, zorder=zorder, label=label,
    )


@dataclass(frozen=True)
class PlotStyleConfig:
    """Reusable visual style used across phase-space and field-map plots."""

    scatter_size: int = 30
    scatter_alpha: float = 0.7
    bins: int = 20
    hist_alpha: float = 0.7
    hist_color: str = "black"
    hist_linewidth: float = 1.2
    backward_cmap: str = "binary"
    show_histograms: bool = True
    dezoom_frac: float = 0.05
    scatter: bool = True


DEFAULT_PLOT_STYLE = PlotStyleConfig()
