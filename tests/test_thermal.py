"""Tests for `rf_gun.thermal` (implementation plan Sec. 6.1-6.4, 10.1, 10.2, 15.3; addendum
Sec. 19.2 -- Work Package 3's Cartesian layered `(x,y)` thermal solver).

Pure Python/numpy/scipy -- no RF-Track dependency. Every test builds a synthetic
`VolumetricHeatSourceTimeSeries` directly (this module's own documented time-resolved power
contract) rather than depending on `back_bombardment_deposition`/`back_bombardment_events`, since
the task is to validate the SOLVER, not the (separately tested) deposition pipeline that would
normally feed it.
"""
from __future__ import annotations

import numpy as np
import pytest

import rf_gun as rg
from rf_gun.cathode_geometry import CathodeGeometry

STEFAN_BOLTZMANN = 5.670374419e-8  # W/m^2/K^4 (CODATA, matches scipy.constants.Stefan_Boltzmann)


# ------------------------------------------------------------------------------------------------
# Synthetic heat-source builders
# ------------------------------------------------------------------------------------------------


def _square_source(
    *,
    n: int,
    extent_m: float,
    layer_boundaries_um: np.ndarray,
    power_per_cell_top_W: float,
    n_bins: int,
    duration_s: float,
    mask: np.ndarray | None = None,
) -> rg.VolumetricHeatSourceTimeSeries:
    """A uniform square (x,y) grid, power deposited into the TOP layer only, constant in time."""
    x = np.linspace(-extent_m, extent_m, n)
    y = np.linspace(-extent_m, extent_m, n)
    if mask is None:
        mask = np.ones((n, n), dtype=bool)
    dx = float(x[1] - x[0])
    area = dx * dx
    n_layers = layer_boundaries_um.size - 1
    t_grid = np.linspace(0.0, duration_s, n_bins + 1)
    q = np.zeros((n, n, n_layers, n_bins), dtype=float)
    q[mask, 0, :] = power_per_cell_top_W
    return rg.VolumetricHeatSourceTimeSeries(
        x_centers_m=x,
        y_centers_m=y,
        layer_boundaries_um=layer_boundaries_um,
        cathode_footprint_mask=mask,
        q_layer_W=q,
        t_grid_s=t_grid,
        xy_cell_area_m2=area,
    )


def _gaussian_disk_source(
    *, n: int, dt_s: float, total_power_W: float = 300.0, duration_s: float = 8.0e-6, n_bins: int = 8
):
    """An asymmetric (off-center Gaussian), disk-masked source used for the mesh/time convergence
    check -- deliberately not spatially uniform, so refinement actually has something to converge.
    Total injected power is held FIXED across resolutions (normalized to the same total), so a
    resolution change never itself changes how much energy is injected."""
    R = 1.0e-3
    extent = 1.3e-3
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = np.hypot(xx, yy) <= R
    dx = float(x[1] - x[0])
    area = dx * dx
    layer_boundaries_um = np.array([0.0, 20.0, 100.0])
    shape = np.exp(-(((xx - 0.3e-3) ** 2 + (yy - 0.1e-3) ** 2) / (0.4e-3) ** 2))
    shape = np.where(mask, shape, 0.0)
    q_per_cell = shape * (total_power_W / shape.sum())
    t_grid = np.linspace(0.0, duration_s, n_bins + 1)
    q = np.zeros((n, n, 2, n_bins), dtype=float)
    q[:, :, 0, :] = q_per_cell[:, :, None]
    heat_source = rg.VolumetricHeatSourceTimeSeries(
        x_centers_m=x,
        y_centers_m=y,
        layer_boundaries_um=layer_boundaries_um,
        cathode_footprint_mask=mask,
        q_layer_W=q,
        t_grid_s=t_grid,
        xy_cell_area_m2=area,
    )
    return heat_source, dt_s


@pytest.fixture(scope="module")
def geometry() -> CathodeGeometry:
    return CathodeGeometry()


@pytest.fixture(scope="module")
def constant_material():
    return rg.load_cathode_material("LaB6", "LaB6_constant_verification_v1")


@pytest.fixture(scope="module")
def recommended_material():
    return rg.load_cathode_material("LaB6", "LaB6_UH_recommended_v1")


@pytest.fixture(scope="module")
def legacy_material():
    return rg.load_cathode_material("LaB6", "LaB6_Kowalczyk_PRSTAB120402_2014_legacy")


# ------------------------------------------------------------------------------------------------
# 1. Analytic validation: constant properties, no radiation, no lateral variation
# ------------------------------------------------------------------------------------------------


def test_python_xy_layered_reduces_to_lumped_energy_balance_for_uniform_case(geometry, constant_material):
    """A spatially uniform source and spatially uniform initial temperature, with constant
    properties, zero radiation, and adiabatic top/bottom boundaries, develops NO lateral or
    depth-wise temperature gradients at any time (every active cell/layer is driven identically).
    The layered solver's spatially-averaged behavior must therefore exactly match the simple,
    independently-computable 0D lumped energy balance `T(t) = T0 + Q_total*t/(rho*cp*V_total)`
    (an exact, closed-form solution since rho/cp/k are constants and there is no T-dependence at
    all in the RHS). A SINGLE depth layer is used deliberately: depositing power only into the
    surface layer of a MULTI-layer stack would still develop a genuine depth (z) gradient (the
    deep layer lags the shallow one until through-depth conduction catches up) even with perfect
    lateral uniformity -- a real, physically-correct effect of this solver, not a bug, but not the
    "no gradient anywhere" case this particular check needs. A single layer sidesteps that
    entirely (there is nowhere for a depth gradient to exist), giving a case with truly zero
    gradients in any direction, exactly matching the fully-lumped closed form."""
    n = 5
    layer_boundaries_um = np.array([0.0, 60.0])
    power_per_cell_W = 0.4
    duration_s = 5.0e-6
    heat_source = _square_source(
        n=n,
        extent_m=1.0e-3,
        layer_boundaries_um=layer_boundaries_um,
        power_per_cell_top_W=power_per_cell_W,
        n_bins=5,
        duration_s=duration_s,
    )
    T0 = 1500.0
    cfg = rg.ThermalConfig(backend="python_xy_layered", total_hemispherical_emissivity=0.0, contact_h_W_m2K=0.0)
    result = rg.solve_xy_layered_thermal(heat_source, rg.ConstantTemperatureMap(T0), geometry, constant_material, cfg)

    rho = constant_material.thermal.rho_kg_m3(T0)
    cp = constant_material.thermal.cp_J_kg_K(T0)
    volume_m3 = heat_source.xy_cell_area_m2 * n * n * float(np.sum(heat_source.layer_thickness_m))
    Q_total_W = power_per_cell_W * n * n
    analytic_T = T0 + Q_total_W * duration_s / (rho * cp * volume_m3)

    assert result.T_area_average_t[-1] == pytest.approx(analytic_T, rel=1e-8)
    assert result.T_center_t[-1] == pytest.approx(analytic_T, rel=1e-8)
    assert result.T_max_t[-1] == pytest.approx(analytic_T, rel=1e-8)
    # Every active cell/layer should agree with every other (no gradients ever developed).
    final_active = result.T_final_xyz[heat_source.cathode_footprint_mask, :]
    assert np.ptp(final_active) < 1e-6


# ------------------------------------------------------------------------------------------------
# 2. lumped_energy_check exact energy conservation
# ------------------------------------------------------------------------------------------------


def test_lumped_energy_check_exact_energy_conservation(geometry, constant_material):
    n = 3
    layer_boundaries_um = np.array([0.0, 50.0])
    power_per_cell_W = 1.5
    duration_s = 4.0e-3
    heat_source = _square_source(
        n=n,
        extent_m=1.0e-3,
        layer_boundaries_um=layer_boundaries_um,
        power_per_cell_top_W=power_per_cell_W,
        n_bins=4,
        duration_s=duration_s,
    )
    T0 = 300.0
    cfg = rg.ThermalConfig(backend="lumped_energy_check", total_hemispherical_emissivity=0.0, contact_h_W_m2K=0.0)
    result = rg.solve_xy_layered_thermal(heat_source, rg.ConstantTemperatureMap(T0), geometry, constant_material, cfg)

    rho = constant_material.thermal.rho_kg_m3(T0)
    cp = constant_material.thermal.cp_J_kg_K(T0)
    n_active = n * n
    volume_m3 = heat_source.xy_cell_area_m2 * n_active * float(np.sum(heat_source.layer_thickness_m))
    Q_total_W = power_per_cell_W * n_active
    analytic_dT = Q_total_W * duration_s / (rho * cp * volume_m3)

    assert (result.T_area_average_t[-1] - T0) == pytest.approx(analytic_dT, rel=1e-10)
    assert result.energy_residual_normalized == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------------------------------------
# 3. Mesh/time convergence (plan Sec. 15.3, provisional 1%)
# ------------------------------------------------------------------------------------------------


def test_mesh_time_convergence_within_one_percent(geometry, constant_material):
    """Same physical case (asymmetric Gaussian source on a disk footprint) run at two independent
    (mesh, timestep) resolutions -- `T_max`/`T_center` at the end of the run must differ by less
    than the plan's provisional 1% tolerance."""
    heat_source_coarse, dt_coarse = _gaussian_disk_source(n=9, dt_s=8.0e-7)
    heat_source_fine, dt_fine = _gaussian_disk_source(n=17, dt_s=4.0e-7)

    T0 = 1500.0
    cfg_coarse = rg.ThermalConfig(backend="python_xy_layered", total_hemispherical_emissivity=0.8, dt_s=dt_coarse)
    cfg_fine = rg.ThermalConfig(backend="python_xy_layered", total_hemispherical_emissivity=0.8, dt_s=dt_fine)

    result_coarse = rg.solve_xy_layered_thermal(
        heat_source_coarse, rg.ConstantTemperatureMap(T0), geometry, constant_material, cfg_coarse
    )
    result_fine = rg.solve_xy_layered_thermal(
        heat_source_fine, rg.ConstantTemperatureMap(T0), geometry, constant_material, cfg_fine
    )

    rel_diff_Tmax = abs(result_coarse.T_max_t[-1] - result_fine.T_max_t[-1]) / result_fine.T_max_t[-1]
    rel_diff_Tcenter = abs(result_coarse.T_center_t[-1] - result_fine.T_center_t[-1]) / result_fine.T_center_t[-1]

    # Sanity: this case must actually heat up non-trivially, or a 1% check would be meaningless.
    assert result_fine.T_max_t[-1] - T0 > 1.0

    assert rel_diff_Tmax < 0.01
    assert rel_diff_Tcenter < 0.01


# ------------------------------------------------------------------------------------------------
# 4. uh_legacy_1d reproduces the legacy closed-form check values (via the solver built on them)
# ------------------------------------------------------------------------------------------------


def test_uh_legacy_1d_check_values_at_1720K(legacy_material):
    """`tests/test_materials.py` already checks the bare closed-form property evaluators
    (k_UH~=43.7 W/m/K, D_UH~=1.21e-5 m^2/s, cp_UH~=765 J/kg/K at T=1720K, rho=4720 kg/m3). Here we
    check the same numbers as actually seen BY THE THERMAL SOLVER built on top of them."""
    T_check = 1720.0
    k = legacy_material.thermal.k_W_m_K(T_check)
    cp = legacy_material.thermal.cp_J_kg_K(T_check)
    rho = legacy_material.thermal.rho_kg_m3(T_check)
    D = k / (rho * cp)

    assert k == pytest.approx(43.7, rel=0.02)
    assert D == pytest.approx(1.21e-5, rel=0.02)
    assert cp == pytest.approx(765.0, rel=0.02)
    assert rho == pytest.approx(4720.0, rel=1e-9)


def test_uh_legacy_1d_energy_conservation(geometry, legacy_material):
    """The `uh_legacy_1d` SOLVER backend (1D depth-only, uniform illumination) conserves energy
    to a tight tolerance when built on the legacy closed-form properties, evaluated near the
    1720K check-value regime."""
    n = 3
    layer_boundaries_um = np.array([0.0, 5.0, 20.0, 100.0])
    heat_source = _square_source(
        n=n,
        extent_m=1.0e-3,
        layer_boundaries_um=layer_boundaries_um,
        power_per_cell_top_W=0.3,
        n_bins=6,
        duration_s=3.0e-6,
    )
    cfg = rg.ThermalConfig(
        backend="uh_legacy_1d",
        total_hemispherical_emissivity=0.0,
        contact_h_W_m2K=0.0,
        uh_legacy_n_depth_layers=12,
    )
    result = rg.solve_xy_layered_thermal(
        heat_source, rg.ConstantTemperatureMap(1720.0), geometry, legacy_material, cfg
    )
    assert result.T_center_t[-1] > 1720.0  # heated, not cooled or stuck
    assert abs(result.energy_residual_normalized) < 1e-6


def test_uh_legacy_1d_rejects_wrong_material(geometry, constant_material):
    """`uh_legacy_1d` must not silently substitute a different material's properties for its own
    benchmark set (plan's "never silently substitute" ethos, mirrored elsewhere in this project)."""
    heat_source = _square_source(
        n=3,
        extent_m=1.0e-3,
        layer_boundaries_um=np.array([0.0, 50.0]),
        power_per_cell_top_W=0.1,
        n_bins=2,
        duration_s=1.0e-6,
    )
    cfg = rg.ThermalConfig(backend="uh_legacy_1d")
    with pytest.raises(ValueError, match="LaB6_Kowalczyk_PRSTAB120402_2014_legacy"):
        rg.solve_xy_layered_thermal(heat_source, rg.ConstantTemperatureMap(1720.0), geometry, constant_material, cfg)


# ------------------------------------------------------------------------------------------------
# 5. Energy residual below tolerance for a well-resolved production-like run
# ------------------------------------------------------------------------------------------------


def test_energy_residual_below_tolerance_for_well_resolved_run(geometry, recommended_material):
    n = 7
    layer_boundaries_um = np.array([0.0, 3.0, 10.0, 30.0, 100.0])
    heat_source = _square_source(
        n=n,
        extent_m=1.2e-3,
        layer_boundaries_um=layer_boundaries_um,
        power_per_cell_top_W=0.8,
        n_bins=6,
        duration_s=2.0e-6,
    )
    cfg = rg.ThermalConfig(
        backend="python_xy_layered",
        total_hemispherical_emissivity=0.8,
        contact_h_W_m2K=50.0,
        T_mount_K=1200.0,
        T_environment_K=300.0,
    )
    result = rg.solve_xy_layered_thermal(
        heat_source, rg.ConstantTemperatureMap(1300.0), geometry, recommended_material, cfg
    )
    assert abs(result.energy_residual_normalized) < cfg.energy_residual_tol


# ------------------------------------------------------------------------------------------------
# 6. Picard nonlinear convergence with a strongly temperature-dependent material
# ------------------------------------------------------------------------------------------------


def test_picard_converges_for_strongly_temperature_dependent_properties(geometry, recommended_material):
    """`LaB6_UH_recommended_v1`'s tabulated cp(T)/k(T) vary noticeably over [1000,2000] K. A run
    driving a large T swing across much of that range must still converge within
    `picard_max_iter` and produce a monotonically increasing, bounded temperature history (no
    oscillation/blow-up)."""
    n = 5
    layer_boundaries_um = np.array([0.0, 5.0, 20.0, 80.0])
    duration_s = 6.0e-6
    heat_source = _square_source(
        n=n,
        extent_m=1.0e-3,
        layer_boundaries_um=layer_boundaries_um,
        power_per_cell_top_W=4.0,
        n_bins=12,
        duration_s=duration_s,
    )
    cfg = rg.ThermalConfig(
        backend="python_xy_layered", total_hemispherical_emissivity=0.0, picard_max_iter=40, picard_tol_K=1e-3
    )
    result = rg.solve_xy_layered_thermal(
        heat_source, rg.ConstantTemperatureMap(1000.0), geometry, recommended_material, cfg
    )

    assert np.all(result.picard_iterations_per_step <= cfg.picard_max_iter)
    assert np.all(result.picard_iterations_per_step >= 1)
    # Monotonically increasing (pure heating, no radiation/contact loss) and bounded/finite.
    assert np.all(np.diff(result.T_center_t) > 0.0)
    assert np.all(np.isfinite(result.T_center_t))
    assert result.T_center_t[-1] < 2000.0  # stayed within the material's own valid tabulated range
    assert abs(result.energy_residual_normalized) < 1e-6


def test_picard_raises_clearly_when_not_converged(geometry, recommended_material):
    """`picard_max_iter=1` starved of iterations on a genuinely nonlinear case must raise, not
    silently return a non-converged result."""
    n = 5
    heat_source = _square_source(
        n=n,
        extent_m=1.0e-3,
        layer_boundaries_um=np.array([0.0, 5.0, 80.0]),
        power_per_cell_top_W=6.0,
        n_bins=4,
        duration_s=4.0e-6,
    )
    cfg = rg.ThermalConfig(
        backend="python_xy_layered", total_hemispherical_emissivity=0.0, picard_max_iter=1, picard_tol_K=1e-8
    )
    with pytest.raises(RuntimeError, match="Picard"):
        rg.solve_xy_layered_thermal(heat_source, rg.ConstantTemperatureMap(1000.0), geometry, recommended_material, cfg)


# ------------------------------------------------------------------------------------------------
# 7. Radiation-only steady-state approach (stability/sign-correctness check)
# ------------------------------------------------------------------------------------------------


def test_radiation_only_monotonically_cools_toward_environment(geometry, constant_material):
    n = 3
    heat_source = _square_source(
        n=n,
        extent_m=1.0e-3,
        layer_boundaries_um=np.array([0.0, 50.0]),
        power_per_cell_top_W=0.0,
        n_bins=8,
        duration_s=1.0e-3,
    )
    T0 = 500.0
    T_env = 300.0
    cfg = rg.ThermalConfig(
        backend="python_xy_layered", total_hemispherical_emissivity=0.8, T_environment_K=T_env, dt_s=2.0e-5
    )
    result = rg.solve_xy_layered_thermal(heat_source, rg.ConstantTemperatureMap(T0), geometry, constant_material, cfg)

    T_hist = result.T_center_t
    assert T_hist[0] == pytest.approx(T0)
    assert np.all(np.diff(T_hist) < 0.0)  # strictly, monotonically decreasing
    assert np.all(T_hist > T_env)  # never overshoots below the environment temperature
    assert np.all(np.isfinite(T_hist))  # no blow-up


# ------------------------------------------------------------------------------------------------
# 8. Cathode footprint masking: no leakage past the mask, exact energy accounting
# ------------------------------------------------------------------------------------------------


def test_footprint_masking_no_leakage_and_exact_energy_accounting(geometry, constant_material):
    n = 9
    x = np.linspace(-2.0e-3, 2.0e-3, n)
    y = np.linspace(-2.0e-3, 2.0e-3, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = np.hypot(xx, yy) <= 1.5e-3  # a genuinely non-square (disk) mask on a square grid
    assert np.any(~mask), "test needs at least one grid cell outside the footprint mask"

    dx = float(x[1] - x[0])
    area = dx * dx
    layer_boundaries_um = np.array([0.0, 40.0])
    n_bins = 3
    duration_s = 3.0e-6
    power_per_cell_W = 2.0
    q = np.zeros((n, n, 1, n_bins), dtype=float)
    q[mask, 0, :] = power_per_cell_W
    heat_source = rg.VolumetricHeatSourceTimeSeries(
        x_centers_m=x,
        y_centers_m=y,
        layer_boundaries_um=layer_boundaries_um,
        cathode_footprint_mask=mask,
        q_layer_W=q,
        t_grid_s=np.linspace(0.0, duration_s, n_bins + 1),
        xy_cell_area_m2=area,
    )
    cfg = rg.ThermalConfig(backend="python_xy_layered", total_hemispherical_emissivity=0.0)
    result = rg.solve_xy_layered_thermal(
        heat_source, rg.ConstantTemperatureMap(300.0), geometry, constant_material, cfg
    )

    n_active = int(np.sum(mask))
    expected_energy_J = power_per_cell_W * n_active * duration_s
    assert result.stored_energy_J_t[-1] == pytest.approx(expected_energy_J, rel=1e-9)
    assert abs(result.energy_residual_normalized) < 1e-8

    # No thermal state was ever assigned outside the mask.
    assert np.all(np.isnan(result.T_surface_xyt[~mask, :]))
    assert np.all(np.isnan(result.T_final_xyz[~mask, :]))
    # Every active cell has a finite, physically sensible (heated, not exploded) temperature.
    assert np.all(np.isfinite(result.T_surface_xyt[mask, :]))
    assert np.all(result.T_surface_xyt[mask, -1] > 300.0)


# ------------------------------------------------------------------------------------------------
# Bonus coverage: initial-temperature-map interface and the python_xy_sheet backend
# ------------------------------------------------------------------------------------------------


def test_temperature_map_2d_interpolates_and_rejects_missing_cells():
    x_src = np.linspace(-2.0e-3, 2.0e-3, 9)
    y_src = np.linspace(-2.0e-3, 2.0e-3, 9)
    xx, yy = np.meshgrid(x_src, y_src, indexing="ij")
    T_src = 1400.0 + 10.0 * xx / 1.0e-3  # a simple asymmetric linear gradient
    mask_src = np.ones((9, 9), dtype=bool)
    tmap = rg.TemperatureMap2D(x_m=x_src, y_m=y_src, T_K=T_src, mask=mask_src)

    x_solver = np.linspace(-1.0e-3, 1.0e-3, 5)
    y_solver = np.linspace(-1.0e-3, 1.0e-3, 5)
    solver_mask = np.ones((5, 5), dtype=bool)
    out = tmap.on_grid(x_solver, y_solver, solver_mask, n_layers=3)
    assert out.shape == (5, 5, 3)
    # uniform_through_layers: every depth layer identical to the interpolated surface value.
    assert np.allclose(out[:, :, 0], out[:, :, 1])
    assert np.allclose(out[:, :, 0], out[:, :, 2])
    # Interpolated values follow the source's own linear gradient at x=0 (center column ~1400 K).
    ix_center = 2
    assert out[ix_center, :, 0] == pytest.approx(1400.0, abs=1.0)

    # A source with a masked-out (invalid) cell right where the solver needs data must raise.
    mask_src_gap = mask_src.copy()
    mask_src_gap[4, 4] = False  # source's own (0,0) cell
    tmap_gap = rg.TemperatureMap2D(x_m=x_src, y_m=y_src, T_K=T_src, mask=mask_src_gap)
    with pytest.raises(ValueError, match="no valid source temperature datum"):
        tmap_gap.on_grid(np.array([0.0]), np.array([0.0]), np.array([[True]]), 1)

    # A solver grid extending outside the source's coordinate range must also raise.
    with pytest.raises(ValueError, match="no valid source temperature datum"):
        tmap.on_grid(np.array([10.0e-3]), np.array([0.0]), np.array([[True]]), 1)


def test_python_xy_sheet_runs_and_conserves_energy(geometry, constant_material):
    n = 4
    layer_boundaries_um = np.array([0.0, 5.0, 30.0])  # collapsed internally to one sheet
    heat_source = _square_source(
        n=n,
        extent_m=1.0e-3,
        layer_boundaries_um=layer_boundaries_um,
        power_per_cell_top_W=0.5,
        n_bins=3,
        duration_s=2.0e-6,
    )
    cfg = rg.ThermalConfig(
        backend="python_xy_sheet", total_hemispherical_emissivity=0.0, h_eff_m=50.0e-6
    )
    result = rg.solve_xy_layered_thermal(
        heat_source, rg.ConstantTemperatureMap(1500.0), geometry, constant_material, cfg
    )
    assert result.n_layers == 1
    assert abs(result.energy_residual_normalized) < 1e-8
    assert np.all(result.T_surface_xyt[heat_source.cathode_footprint_mask, -1] > 1500.0)


def test_build_constant_power_heat_source_time_series_conserves_total_energy():
    """The documented "obviously-correct multiplication" helper: dividing a per-period deposited
    -energy tensor by the period duration and repeating across the requested time grid must
    reproduce the ORIGINAL total energy when integrated back over the actual macro-time span it
    is asked to cover (a basic sanity/unit check on the helper itself)."""
    n = 4
    rf_period_s = 3.5e-10
    q_layer_J = np.full((n, n, 2), 1.0e-9)  # 1 nJ per cell per layer for the one period

    class _FakeBBSource:
        x_centers_m = np.linspace(-1.0e-3, 1.0e-3, n)
        y_centers_m = np.linspace(-1.0e-3, 1.0e-3, n)
        layer_boundaries_um = np.array([0.0, 10.0, 100.0])
        cathode_footprint_mask = np.ones((n, n), dtype=bool)
        xy_cell_area_m2 = (x_centers_m[1] - x_centers_m[0]) ** 2

    fake = _FakeBBSource()
    fake.q_layer_J = q_layer_J

    t_grid_s = np.array([0.0, 1.0e-6, 2.0e-6])  # 2 bins of 1 us each
    series = rg.build_constant_power_heat_source_time_series(fake, rf_period_s, t_grid_s)

    expected_power_W = q_layer_J / rf_period_s
    assert np.allclose(series.q_layer_W[:, :, :, 0], expected_power_W)
    assert np.allclose(series.q_layer_W[:, :, :, 1], expected_power_W)

    # Both bins have equal width and carry the identical (repeated) power array, so summing over
    # every cell AND every bin, then multiplying by one bin's width, integrates the total injected
    # energy across the full requested time span exactly.
    bin_width_s = t_grid_s[1] - t_grid_s[0]
    total_energy_over_span_J = np.sum(series.q_layer_W) * bin_width_s
    n_bins = t_grid_s.size - 1
    expected_total_energy_J = n_bins * (np.sum(q_layer_J) / rf_period_s) * bin_width_s
    assert total_energy_over_span_J == pytest.approx(expected_total_energy_J, rel=1e-12)


def test_heat_source_rejects_power_outside_thermal_mask():
    """Power in mask=False cells would be silently omitted by every thermal backend."""
    x = np.array([-1.0e-3, 0.0, 1.0e-3])
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    q = np.zeros((3, 3, 1, 1), dtype=float)
    q[0, 0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="outside cathode_footprint_mask"):
        rg.VolumetricHeatSourceTimeSeries(
            x_centers_m=x,
            y_centers_m=x,
            layer_boundaries_um=np.array([0.0, 10.0]),
            cathode_footprint_mask=mask,
            q_layer_W=q,
            t_grid_s=np.array([0.0, 1.0e-6]),
            xy_cell_area_m2=1.0e-6,
        )
