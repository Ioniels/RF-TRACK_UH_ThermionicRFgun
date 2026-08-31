"""Plotting helpers for RF gun simulations."""

from .fields import field_maps, axis_phase, plot_deflection_field_profile
from .emission import plot_emission_history, plot_j_vs_n, plot_emission_model_sensitivities
from .attribution import plot_frozen_source_attribution
from .iteration import (
    plot_emission_iteration_convergence,
    plot_emission_iteration_waveforms,
    plot_emission_iteration_near_cathode,
    plot_emission_iteration_submodel_comparison,
)
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
    plot_back_bombardment_source_qualification,
    print_back_bombardment_source_qualification_summary,
)
from .macropulse import plot_back_bombardment_macropulse, print_back_bombardment_macropulse_summary
from .aperture import plot_dynamic_aperture_losses
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
    get_lost_cmap,
    get_recentered_diverging_cmap,
    add_reference_lines,
    add_aperture_curve,
    add_cathode_boundary_circle,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_NEUTRAL,
    COLOR_LOST,
    EMISSION_MODEL_COLORS,
)

__all__ = [
    "field_maps",
    "axis_phase",
    "plot_deflection_field_profile",
    "plot_emission_history",
    "plot_j_vs_n",
    "plot_emission_model_sensitivities",
    "plot_frozen_source_attribution",
    "plot_emission_iteration_convergence",
    "plot_emission_iteration_waveforms",
    "plot_emission_iteration_near_cathode",
    "plot_emission_iteration_submodel_comparison",
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
    "plot_back_bombardment_source_qualification",
    "print_back_bombardment_source_qualification_summary",
    "plot_back_bombardment_macropulse",
    "print_back_bombardment_macropulse_summary",
    "plot_dynamic_aperture_losses",
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
    "get_lost_cmap",
    "get_recentered_diverging_cmap",
    "add_reference_lines",
    "add_aperture_curve",
    "add_cathode_boundary_circle",
    "COLOR_PRIMARY",
    "COLOR_SECONDARY",
    "COLOR_NEUTRAL",
    "COLOR_LOST",
    "EMISSION_MODEL_COLORS",
]
