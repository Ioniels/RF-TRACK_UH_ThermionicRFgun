"""Plotting helpers for RF gun simulations."""

from .fields import field_maps, axis_phase
from .emission import plot_emission_history, plot_j_vs_n
from .phase_space import plot_phase_space, plot_spectra, plot_screen_phase_space_slider
from .evolution import (
    plot_beam_moments_evolution,
    plot_beam_twiss_evolution,
)
from .phase_scan import phase_plot
from .back_bombardment import (
    plot_back_bombardment_phase_space,
    plot_back_bombardment_screen_reach,
    plot_back_bombardment_energy_density,
    plot_back_bombardment_power_density_vs_time,
)
from .acceptance_scan import plot_acceptance_scan
from .save_run import (
    save_run_figures,
    plot_class_conditioned_histograms,
    save_beam_phase_space_json,
    save_screen_phase_space_batch,
    capture_figures,
    FigureCapture,
)
from .style import (
    PlotStyleConfig,
    DEFAULT_PLOT_STYLE,
    get_default_density_cmap,
    get_aperture_loss_cmap,
    get_recentered_diverging_cmap,
    add_reference_lines,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_NEUTRAL,
)

__all__ = [
    "field_maps",
    "axis_phase",
    "plot_emission_history",
    "plot_j_vs_n",
    "plot_phase_space",
    "plot_spectra",
    "plot_screen_phase_space_slider",
    "plot_beam_moments_evolution",
    "plot_beam_twiss_evolution",
    "phase_plot",
    "plot_back_bombardment_phase_space",
    "plot_back_bombardment_screen_reach",
    "plot_back_bombardment_energy_density",
    "plot_back_bombardment_power_density_vs_time",
    "plot_acceptance_scan",
    "save_run_figures",
    "plot_class_conditioned_histograms",
    "save_beam_phase_space_json",
    "save_screen_phase_space_batch",
    "capture_figures",
    "FigureCapture",
    "PlotStyleConfig",
    "DEFAULT_PLOT_STYLE",
    "get_default_density_cmap",
    "get_aperture_loss_cmap",
    "get_recentered_diverging_cmap",
    "add_reference_lines",
    "COLOR_PRIMARY",
    "COLOR_SECONDARY",
    "COLOR_NEUTRAL",
]
