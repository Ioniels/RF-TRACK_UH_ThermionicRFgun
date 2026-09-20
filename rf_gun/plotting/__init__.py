"""Plotting helpers for RF-gun simulation and field-map analysis.

Exports are resolved lazily so the XFdtd preprocessing plots remain usable on
analysis hosts that do not have the RF-Track Python binding installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_MODULES = (
    "field_analysis",
    "figure_io",
    "fields",
    "emission",
    "attribution",
    "iteration",
    "phase_space",
    "evolution",
    "phase_scan",
    "back_bombardment",
    "macropulse",
    "aperture",
    "acceptance_scan",
    "save_run",
    "style",
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for module_name in _LAZY_MODULES:
        module = import_module(f".{module_name}", __name__)
        try:
            value = getattr(module, name)
        except AttributeError:
            continue
        globals()[name] = value
        return value
    raise AttributeError(
        f"public symbol {name!r} is listed by {__name__!r} but no provider module exports it"
    )


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "DEFAULT_FIGURE_FORMATS",
    "EPS_MAX_BYTES",
    "SIZE_CAPPED_FORMATS",
    "normalize_formats",
    "save_figure_formats",
    "field_maps",
    "axis_phase",
    "plot_deflection_field_profile",
    "plot_mesh_time_metadata",
    "plot_temporal_fit_drift",
    "plot_symmetry_mode_spectra",
    "plot_meridional_fields",
    "plot_on_axis_tail_decay",
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
