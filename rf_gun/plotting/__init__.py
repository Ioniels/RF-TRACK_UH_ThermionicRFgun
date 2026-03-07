"""Plotting helpers for RF gun simulations."""

from .fields import field_maps, axis_phase
from .emission import plot_emission_history, plot_j_vs_n
from .phase_space import plot_phase_space, plot_spectra, plot_screen_phase_space_slider
from .evolution import plot_evolution, plot_twiss_evolution, plot_emittance_evolution, plot_transmission_evolution
from .phase_scan import theory_plot, phase_plot
from .save_run import (
    save_run_figures,
    plot_class_conditioned_histograms,
    save_beam_phase_space_json,
    save_screen_phase_space_batch,
)
from .style import PlotStyleConfig, DEFAULT_PLOT_STYLE, get_default_density_cmap

__all__ = [
    "field_maps",
    "axis_phase",
    "plot_emission_history",
    "plot_j_vs_n",
    "plot_phase_space",
    "plot_spectra",
    "plot_screen_phase_space_slider",
    "plot_evolution",
    "plot_twiss_evolution",
    "plot_emittance_evolution",
    "plot_transmission_evolution",
    "theory_plot",
    "phase_plot",
    "save_run_figures",
    "plot_class_conditioned_histograms",
    "save_beam_phase_space_json",
    "save_screen_phase_space_batch",
    "PlotStyleConfig",
    "DEFAULT_PLOT_STYLE",
    "get_default_density_cmap",
]
