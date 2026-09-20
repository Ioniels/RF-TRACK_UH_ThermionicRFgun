from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import jv

from rf_gun.fieldmaps.models import AxisymmetricRFField
from rf_gun.fieldmaps.quality import (
    SPEED_OF_LIGHT_MPS,
    axis_regularity_diagnostics,
    maxwell_residual_diagnostics,
    qualify_axisymmetric_field,
    tail_profile_diagnostics,
)
from rf_gun.plotting.field_analysis import (
    plot_meridional_fields,
    plot_mesh_time_metadata,
    plot_on_axis_tail_decay,
    plot_symmetry_mode_spectra,
    plot_temporal_fit_drift,
)


def _analytic_tm_field(*, magnetic_sign: float = 1.0, scale: float = 1.0) -> AxisymmetricRFField:
    """z-independent cylindrical TM solution for the exp(+i*omega*t) convention."""

    frequency_hz = 1.0e9
    omega = 2.0 * np.pi * frequency_hz
    k0 = omega / SPEED_OF_LIGHT_MPS
    r_m = np.linspace(0.0, 0.030, 401)
    z_m = np.linspace(0.0, 0.020, 31)
    amplitude = scale * 3.0e6
    ez_line = amplitude * jv(0, k0 * r_m)
    btheta_line = magnetic_sign * 1j * amplitude * k0 / omega * jv(1, k0 * r_m)
    Ez = np.broadcast_to(ez_line, (z_m.size, r_m.size)).copy().astype(np.complex128)
    Btheta = np.broadcast_to(btheta_line, Ez.shape).copy()
    zeros = np.zeros_like(Ez)
    return AxisymmetricRFField(
        frequency_hz,
        r_m,
        z_m,
        zeros,
        Ez,
        Btheta,
        zeros,
        metadata={"representation": "analytic_tm"},
    )


def _decaying_field() -> AxisymmetricRFField:
    r_m = np.linspace(0.0, 2.0e-3, 21)
    z_m = np.linspace(0.0, 0.040, 81)
    radial = np.exp(-(r_m / 2.5e-3) ** 2)
    decay = np.exp(-180.0 * z_m)
    carrier = np.exp(1j * 7.0 * z_m)
    base = 5.0e6 * decay[:, None] * carrier[:, None] * radial[None, :]
    return AxisymmetricRFField(
        2.86541722e9,
        r_m,
        z_m,
        0.05 * (r_m[None, :] / r_m[-1]) * base,
        base,
        2.0e-9 * (r_m[None, :] / r_m[-1]) * base,
        1.0e-10 * base,
        metadata={"spatial_support": {"measured_z_max_m": 0.033}},
    )


def test_maxwell_residuals_recover_exp_plus_iwt_tm_sign_and_scale():
    field = _analytic_tm_field()
    report = maxwell_residual_diagnostics(
        field, wall_guard_cells=2, axis_guard_cells=1, active_field_fraction=1.0e-4
    )
    active = report["active_guarded_vacuum"]
    assert active["primary"]["faraday_theta"]["normalized_rms"] < 1.0e-5
    assert active["primary"]["ampere_z"]["normalized_rms"] < 1.0e-5
    assert active["combined"]["faraday_family_normalized_rms"] < 1.0e-5
    assert active["combined"]["ampere_family_normalized_rms"] < 1.0e-5

    scaled = maxwell_residual_diagnostics(
        _analytic_tm_field(scale=13.0),
        wall_guard_cells=2,
        axis_guard_cells=1,
        active_field_fraction=1.0e-4,
    )
    assert scaled["active_guarded_vacuum"]["primary"]["faraday_theta"][
        "normalized_rms"
    ] == pytest.approx(active["primary"]["faraday_theta"]["normalized_rms"], rel=1.0e-8)


def test_wrong_magnetic_phasor_sign_is_detected_by_both_curl_equations():
    report = maxwell_residual_diagnostics(
        _analytic_tm_field(magnetic_sign=-1.0), wall_guard_cells=2, axis_guard_cells=1
    )
    active = report["active_guarded_vacuum"]["primary"]
    assert active["faraday_theta"]["normalized_rms"] == pytest.approx(np.sqrt(2.0), rel=1.0e-5)
    assert active["ampere_z"]["normalized_rms"] == pytest.approx(np.sqrt(2.0), rel=1.0e-5)


def test_wall_guard_rejects_boundary_corruption_and_axis_check_catches_odd_offset():
    clean = _analytic_tm_field()
    corrupted_ez = np.array(clean.Ez_Vpm, copy=True)
    corrupted_ez[:, -1] += 1.0e12
    corrupted_ez[[0, -1], :] += 1.0e12
    corrupted = AxisymmetricRFField(
        clean.frequency_hz,
        clean.r_m,
        clean.z_m,
        clean.Er_Vpm,
        corrupted_ez,
        clean.Btheta_T,
        clean.Bz_T,
    )
    report = maxwell_residual_diagnostics(corrupted, wall_guard_cells=2, axis_guard_cells=1)
    # Two guarded layers keep central derivative stencils away from the deliberately bad boundary.
    assert report["active_guarded_vacuum"]["primary"]["faraday_theta"]["normalized_rms"] < 1.0e-5

    er = np.array(clean.Er_Vpm, copy=True)
    er[:, 0] = 1.0e4
    irregular = AxisymmetricRFField(
        clean.frequency_hz,
        clean.r_m,
        clean.z_m,
        er,
        clean.Ez_Vpm,
        clean.Btheta_T,
        clean.Bz_T,
    )
    axis = axis_regularity_diagnostics(irregular)
    assert axis["odd_components"]["Er"]["axis_rms"] == pytest.approx(1.0e4)
    assert axis["odd_components"]["Er"]["fraction_of_combined_field_rms"] > 1.0e-3


def test_tail_and_aggregate_quality_are_json_serializable_and_quantify_endpoint():
    field = _decaying_field()
    tail = tail_profile_diagnostics(field, tail_start_z_m=0.030, near_zero_fraction=1.0e-2)
    ez_endpoint = tail["endpoint"]["Ez_Vpm"]
    assert ez_endpoint["end_to_global_max_fraction"] == pytest.approx(np.exp(-7.2), rel=2.0e-3)
    assert ez_endpoint["end_below_near_zero_threshold"] is True
    assert tail["decay_fit_E_combined"]["decay_constant_per_m"] == pytest.approx(180.0, rel=2.0e-3)
    aggregate = qualify_axisymmetric_field(
        field,
        wall_guard_cells=2,
        axis_guard_cells=1,
        tail_start_z_m=0.030,
        beam_core_radius_m=0.002,
    )
    serialized = json.dumps(aggregate, allow_nan=False)
    assert "axisymmetric_vacuum_quality_v1" in serialized
    assert aggregate["norms"]["components"]["Btheta"]["units"] == "T"
    assert aggregate["maxwell_beam_core"]["radius_m"] == pytest.approx(0.002)
    assert aggregate["maxwell_beam_core"]["active_guarded_cell_count"] > 0


def test_all_field_analysis_plots_smoke():
    field = _decaying_field()
    times_e = 98.6e-9 + np.arange(17) / (4.0 * field.frequency_hz)
    times_h = times_e + 0.096e-12
    figures = []
    figures.append(
        plot_mesh_time_metadata(
            {
                "Ex:x": np.linspace(-0.035, 0.035, 31),
                "Ey:y": np.linspace(-0.033, 0.007, 27),
                "Ez:z": np.linspace(-0.036, 0.036, 35),
            },
            times_e,
            times_h,
            field.frequency_hz,
        )
    )
    figures.append(
        plot_temporal_fit_drift(
            {
                "components": {
                    "Ex": [{"global_nrmse": 0.004, "active_nrmse_p95": 0.01, "peak_fractional_drift_per_cycle": 0.002}],
                    "Hy": [{"global_nrmse": 0.006, "active_nrmse_p95": 0.015, "peak_fractional_drift_per_cycle": 0.003}],
                }
            }
        )
    )
    figures.append(
        plot_symmetry_mode_spectra(
            {
                "component_nonaxisymmetric_fraction": {"Er": 0.12, "Ez": 0.02, "Btheta": 0.15, "Bz": 0.08},
                "combined_e_nonaxisymmetric_fraction": 0.021,
                "combined_b_nonaxisymmetric_fraction": 0.145,
                "mode_power_fraction": {
                    "Ez": {"-1": 0.001, "0": 0.997, "1": 0.002},
                    "Btheta": {"-1": 0.011, "0": 0.976, "1": 0.013},
                },
            }
        )
    )
    measured = np.broadcast_to(field.z_m[:, None] <= 0.033, field.shape)
    figures.append(
        plot_meridional_fields(
            field,
            phase_deg=27.0,
            measured_mask=measured,
            aperture_radius_m=np.full(field.z_m.shape, 1.8e-3),
        )
    )
    figures.append(plot_on_axis_tail_decay(field, tail_start_z_m=0.030))
    try:
        assert all(isinstance(figure, plt.Figure) for figure in figures)
        assert len(figures[0].axes) == 4
        assert len(figures[1].axes) == 3
        assert len(figures[2].axes) == 2
        assert len(figures[3].axes) == 8  # four field panels plus four colorbars
        assert len(figures[4].axes) == 4
        titles = " ".join(axis.get_title() for axis in figures[3].axes)
        assert "E_z" in titles and "B_\\theta" in titles
    finally:
        for figure in figures:
            plt.close(figure)


def test_quality_status_requires_the_full_gate_set_not_merely_no_failures():
    """An empty or truncated gate mapping must never resolve to 'qualified'.

    Resolving on "nothing failed" alone means a build that never evaluated the production
    checks publishes as production -- absence of evidence read as a pass.
    """
    import pytest

    from rf_gun.fieldmaps.workflow import (
        PRODUCTION_QUALITY_GATE_VERSION,
        REQUIRED_PRODUCTION_GATE_NAMES,
        resolve_production_quality_status,
    )

    assert resolve_production_quality_status("auto", {}) == "analysis_only"

    one_gate = {
        "cathode_origin": {
            "passed": True, "gate_version": PRODUCTION_QUALITY_GATE_VERSION,
        }
    }
    assert resolve_production_quality_status("auto", one_gate) == "analysis_only"
    with pytest.raises(RuntimeError, match="incomplete"):
        resolve_production_quality_status("qualified", one_gate)

    full = {
        name: {"passed": True, "gate_version": PRODUCTION_QUALITY_GATE_VERSION}
        for name in REQUIRED_PRODUCTION_GATE_NAMES
    }
    assert resolve_production_quality_status("auto", full) == "qualified"

    stale = dict(full)
    stale["downstream_tail"] = {"passed": True, "gate_version": "xfdtd_m0_gate_v1"}
    assert resolve_production_quality_status("auto", stale) == "analysis_only"

    one_failed = dict(full)
    one_failed["axis_Er_zero"] = {
        "passed": False, "gate_version": PRODUCTION_QUALITY_GATE_VERSION,
    }
    assert resolve_production_quality_status("auto", one_failed) == "analysis_only"
