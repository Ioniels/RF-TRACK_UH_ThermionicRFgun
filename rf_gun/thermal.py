"""Cartesian layered `(x,y)` thermal solver for back-bombardment heating (implementation plan
Sec. 6.1-6.4, 10.1, 10.2, 15.3; addendum Sec. 19.2, Work Package 3).

Scope: pure Python/numpy/scipy, NO RF-Track dependency. This module solves the ALREADY-SCALED,
already-time-resolved volumetric power source that plan Sec. 6.2 describes -- it has no notion of
"one RF period" or RF frequency at all; that conversion (RF envelope, keyframe interpolation,
current histories) is `rf_gun/macropulse.py`'s job (out of this module's scope, plan addendum
Sec. 19.2's Work Package 6 vs. Work Package 3).

Contents:
  * `ConstantTemperatureMap` / `TemperatureMap2D`: the initial-temperature-map interface (plan
    Sec. 6.1).
  * `VolumetricHeatSourceTimeSeries`: this module's own explicit, documented convention for a
    time-resolved deposited-power tensor (see its docstring for the exact shape/units contract),
    plus `build_constant_power_heat_source_time_series`, the small "obviously-correct
    multiplication" helper that turns a `back_bombardment_deposition.BackBombardmentHeatSource`
    (one representative RF period's deposited ENERGY [J]) into one of these (a POWER [W] time
    series) -- NOT the real macropulse/RF-envelope scaling, just a documented unit conversion and
    time-grid replication (see that function's docstring).
  * `ThermalResult`: the reported-output container (plan Sec. 6.2's explicit list).
  * `solve_xy_layered_thermal`: the public dispatch entry point over the four backends
    (`python_xy_layered`, `python_xy_sheet`, `lumped_energy_check`, `uh_legacy_1d`), each
    implemented in its own `_solve_<backend>` function below.

Numerical method (the two "hard" backends, `python_xy_layered`/`python_xy_sheet`, both routed
through the shared `_solve_fv_core`):
  * Implicit (backward) Euler time-stepping (plan Sec. 6.2: "Use implicit Euler initially").
  * Nonlinear `rho(T)`, `cp(T)`, `k(T)` (and the radiative-loss term) are handled by Picard
    (fixed-point) iteration within each timestep: assemble the linear system with every
    T-dependent coefficient evaluated at the CURRENT Picard iterate (initialized to the previous
    timestep's converged state), solve, check `max(abs(delta_T)) < picard_tol_K`, repeat up to
    `picard_max_iter` (raises `RuntimeError` if it does not converge -- never silently returns a
    non-converged state).
  * Radiation's `T^4` loss term is NOT Newton-linearized -- it is simply re-evaluated at each
    Picard iterate's current temperature and folded into that iteration's right-hand side as a
    fixed source, exactly like `rho(T)/cp(T)/k(T)` are already handled; convergence to
    self-consistency comes from the Picard loop itself. This is the simpler of the two options the
    task description offers, chosen because it needs no separate derivative bookkeeping and is
    already consistent with how every other nonlinear term here is treated.
  * In-plane conduction uses a standard 5-point (fewer at the cathode footprint mask's own edges)
    finite-volume stencil with harmonic-mean face conductance between neighboring in-footprint
    cells; a cell with no in-footprint neighbor on some side simply has no flux term on that side
    (exact zero-flux/Neumann realization -- no special-cased boundary equation is needed). This
    solver ASSUMES (and validates) a uniform, square-pixel lateral grid (`dx == dy`, constant
    spacing) -- true of every grid `back_bombardment_deposition.build_back_bombardment_heat_source`
    actually builds; a caller supplying a non-uniform/non-square grid gets a clear `ValueError`
    rather than a silently wrong stencil.
  * Through-depth conductance between neighboring layers uses the standard series-resistance form
    `G = k_eff/(0.5*dz_ell + 0.5*dz_(ell+1))` with `k_eff` the HARMONIC mean of the two layers'
    `k(T)` (the standard finite-volume choice for a conductivity at a cell face, since it is exact
    for a steady 1D two-segment conduction path in series -- an arithmetic mean would not
    reproduce that exact series-resistance limit).
  * The full sparse linear system is assembled over ALL `(x,y,layer)` unknowns jointly (in-plane
    and depth coupling in one solve per Picard iteration) via `scipy.sparse`/`scipy.sparse.linalg.
    spsolve` -- simplicity/correctness over speed, per the task's explicit instruction; no
    operator-split scheme is attempted.
  * Energy accounting (`ThermalResult.energy_residual_normalized`) is built to telescope exactly
    with the assembled equations themselves (see `_step_backward_euler`'s docstring): at Picard
    convergence, the SAME per-cell heat-capacity coefficients (`C_ell` evaluated at the converged
    iterate) and the SAME per-cell loss terms used in the last linear solve are summed over the
    whole domain to get the step's stored-energy change and injected/lost power, so the only
    sources of residual are genuine Picard slack (bounded by `picard_tol_K`) and floating-point
    round-off -- not a second, independently-computed energy functional that could disagree with
    what the solver actually did.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.constants import Stefan_Boltzmann
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve

from .back_bombardment_study_config import DEFAULT_TOTAL_HEMISPHERICAL_EMISSIVITY, ThermalConfig
from .cathode_geometry import CathodeGeometry
from .materials.base import CathodeMaterialSet, ThermalComponent
from .materials.registry import validate_material_for

__all__ = [
    "DEFAULT_TOTAL_HEMISPHERICAL_EMISSIVITY",
    "ThermalConfig",
    "ConstantTemperatureMap",
    "TemperatureMap2D",
    "InitialTemperatureMap",
    "VolumetricHeatSourceTimeSeries",
    "build_constant_power_heat_source_time_series",
    "ThermalResult",
    "solve_xy_layered_thermal",
]


# ================================================================================================
# 1. Initial temperature map interface (plan Sec. 6.1)
# ================================================================================================


@dataclass(frozen=True)
class ConstantTemperatureMap:
    """`T_s(x,y,0) = T0_K` for every point in the cathode footprint mask (flat + bevel projection),
    no thermal state outside it (plan Sec. 6.1, item 1). The current default/only-used initial
    condition in every existing back-bombardment study; a trivial, always-unambiguous case for
    every depth-extension mode (plan Sec. 6.1: "For the present constant map all three [depth
    -extension modes] reduce to a constant initial cathode temperature").
    """

    T0_K: float

    def __post_init__(self) -> None:
        if not (self.T0_K > 0.0):
            raise ValueError(f"T0_K must be positive, got {self.T0_K!r}")

    def on_grid(
        self, x_centers_m: np.ndarray, y_centers_m: np.ndarray, mask: np.ndarray, n_layers: int
    ) -> np.ndarray:
        """`(n_x, n_y, n_layers)` initial-temperature array: `T0_K` inside `mask`, `nan` outside
        (every layer identical -- `uniform_through_layers` is trivial for a spatially constant
        surface value, plan Sec. 6.1's closing sentence)."""
        nx, ny = mask.shape
        out = np.full((nx, ny, n_layers), np.nan, dtype=float)
        out[mask, :] = self.T0_K
        return out


@dataclass(frozen=True)
class TemperatureMap2D:
    """An imported/asymmetric `T(x,y,t=0,z=0)` map (plan Sec. 6.1, item 2; Sec. 4's
    `initial_temperature_xy.h5` contract) -- e.g. a future COMSOL/heater-derived baseline.

    `x_m`/`y_m`: strictly increasing 1D coordinate axes of the SOURCE grid (not necessarily the
    solver's own grid) -- a regular Cartesian grid, since `on_grid` below interpolates with
    `scipy.interpolate.RegularGridInterpolator` (chosen because plan Sec. 6.1 only requires the
    imported map be interpolated "conservatively where possible" onto the solver grid, and a
    regular-grid bilinear interpolator is the simplest correct choice for a regular source grid;
    it is documented here explicitly per the task's instruction rather than silently assumed).
    `T_K`: `(x_m.size, y_m.size)` source temperatures. `mask`: `(x_m.size, y_m.size)` bool, `True`
    where the source datum is valid (e.g. within the source's own cathode footprint or sensor
    field of view) -- `False`/missing cells are never interpolated across.
    `metadata`: free-form provenance dict (coordinate frame, source-file hash, uncertainty, etc.,
    per plan Sec. 6.1's "coordinates, units, cathode origin/orientation, mask, interpolation
    method, source-file hash").
    `depth_extension`: `"uniform_through_layers"` (implemented -- copies the interpolated surface
    value into every depth layer) is the only mode implemented in this pass.
    `"reference_depth_profile"`/`"comsol_volume"` (plan Sec. 6.1, items 2-3) are explicit
    `NotImplementedError` stubs: no named reference depth profile or COMSOL volumetric baseline
    exists yet in this project.
    """

    x_m: np.ndarray
    y_m: np.ndarray
    T_K: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    depth_extension: str = "uniform_through_layers"

    def __post_init__(self) -> None:
        x = np.asarray(self.x_m, dtype=float)
        y = np.asarray(self.y_m, dtype=float)
        T = np.asarray(self.T_K, dtype=float)
        mask = np.asarray(self.mask, dtype=bool)
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "y_m", y)
        object.__setattr__(self, "T_K", T)
        object.__setattr__(self, "mask", mask)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("TemperatureMap2D.x_m/y_m must be 1D coordinate axes")
        if not (np.all(np.diff(x) > 0.0) and np.all(np.diff(y) > 0.0)):
            raise ValueError("TemperatureMap2D.x_m/y_m must be strictly increasing")
        if T.shape != (x.size, y.size) or mask.shape != (x.size, y.size):
            raise ValueError(
                f"TemperatureMap2D.T_K/mask must have shape (x_m.size, y_m.size)="
                f"{(x.size, y.size)}, got T_K={T.shape}, mask={mask.shape}"
            )
        if self.depth_extension not in ("uniform_through_layers", "reference_depth_profile", "comsol_volume"):
            raise ValueError(
                f"Unknown depth_extension {self.depth_extension!r}; valid values: "
                "'uniform_through_layers' (implemented), 'reference_depth_profile' "
                "(NotImplementedError, plan Sec. 6.1), 'comsol_volume' (NotImplementedError, "
                "plan Sec. 6.1)"
            )

    def on_grid(
        self, x_centers_m: np.ndarray, y_centers_m: np.ndarray, mask: np.ndarray, n_layers: int
    ) -> np.ndarray:
        """Interpolate this map onto the solver's own `(x_centers_m, y_centers_m)` grid and extend
        it through `n_layers` depth layers per `self.depth_extension`.

        Raises `ValueError` (not a silent mean-fill) if any solver-grid cathode cell (`mask=True`)
        has no corresponding valid source datum -- either because it falls outside
        `[self.x_m.min(), self.x_m.max()] x [self.y_m.min(), self.y_m.max()]` (a coordinate-frame
        mismatch) or because the nearest source cells are themselves masked out (a genuine data
        gap) -- per plan Sec. 6.1: "Missing cathode cells or coordinate mismatches are errors, not
        filled with a hidden mean temperature."
        """
        if self.depth_extension != "uniform_through_layers":
            raise NotImplementedError(
                f"TemperatureMap2D.depth_extension={self.depth_extension!r} is not implemented "
                "in this pass (plan Sec. 6.1): no named reference depth profile or COMSOL "
                "volumetric baseline exists yet in this project. Only 'uniform_through_layers' "
                "is implemented."
            )

        # T_K with masked-out source cells replaced by nan, so the interpolator can never blend a
        # valid neighbor with an invalid one without at least one of the four corners being nan
        # (which RegularGridInterpolator then propagates to nan in the query, caught below --
        # never silently averaged in).
        T_masked = np.where(self.mask, self.T_K, np.nan)
        interp = RegularGridInterpolator(
            (self.x_m, self.y_m), T_masked, method="linear", bounds_error=False, fill_value=np.nan
        )
        xx, yy = np.meshgrid(x_centers_m, y_centers_m, indexing="ij")
        query = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        T_query = interp(query).reshape(xx.shape)

        missing = mask & ~np.isfinite(T_query)
        if np.any(missing):
            n_missing = int(np.sum(missing))
            ix0, iy0 = (int(v[0]) for v in np.nonzero(missing))
            raise ValueError(
                f"TemperatureMap2D.on_grid: {n_missing} solver-grid cathode cell(s) have no valid "
                "source temperature datum (either outside the supplied map's coordinate range "
                f"[{self.x_m.min():.6g},{self.x_m.max():.6g}]x"
                f"[{self.y_m.min():.6g},{self.y_m.max():.6g}] m, or interpolated from masked-out "
                f"source cells) -- first offending cell at solver-grid index (ix={ix0}, iy={iy0}), "
                f"(x={x_centers_m[ix0]:.6g}, y={y_centers_m[iy0]:.6g}) m. Per plan Sec. 6.1 this is "
                "an error, not filled with a hidden mean/default temperature -- check the source "
                "map's coordinate frame/mask against the solver's own cathode footprint."
            )

        out = np.full((mask.shape[0], mask.shape[1], n_layers), np.nan, dtype=float)
        for ell in range(n_layers):
            out[:, :, ell] = np.where(mask, T_query, np.nan)
        return out


InitialTemperatureMap = "ConstantTemperatureMap | TemperatureMap2D"


# ================================================================================================
# Time-resolved volumetric heat source (public input contract)
# ================================================================================================


@dataclass(frozen=True)
class VolumetricHeatSourceTimeSeries:
    """This module's explicit, documented convention for an ALREADY-SCALED, already-time-resolved
    volumetric power source (the task's required "unambiguous and consistent" contract).

    Units/shape decision (documented here, not guessed downstream):
      * `q_layer_W` has shape `(n_x, n_y, n_layers, n_t_bins)` and is deposited POWER in WATTS
        already integrated over each cell's volume (i.e. `q_layer_W[ix,iy,ell,ib]` is the total
        power [W] flowing into lateral cell `(ix,iy)`, depth layer `ell`, during time bin `ib` --
        NOT a volumetric power density [W/m^3] requiring a further multiplication by cell volume).
        This mirrors `back_bombardment_deposition.BackBombardmentHeatSource.q_layer_J`'s own
        per-cell-energy (not per-volume) convention exactly, so the conversion from one to the
        other (`build_constant_power_heat_source_time_series` below) is a single division by a
        time, not a division by volume as well.
      * `t_grid_s` has shape `(n_t_bins + 1,)` -- these are BIN EDGES (strictly increasing, in
        seconds), not per-sample time-stamps. Bin `ib` covers `[t_grid_s[ib], t_grid_s[ib+1]]` and
        `q_layer_W[...,ib]` is held CONSTANT (piecewise-constant-in-time) across that entire bin.
        This bin-edge convention (rather than "one array entry per time-stamp, with an awkward
        last unused sample") is chosen so every entry of `q_layer_W`'s time axis is unambiguously
        used by exactly one step of the solver.
      * `cathode_footprint_mask` is `(n_x, n_y)` bool -- the exact set of lateral cells the solver
        is responsible for; a cell with `mask=False` is never assembled into the linear system
        (no flux crosses it, per plan Sec. 6.2's "no flux crosses cells outside the LaB6
        geometry"), regardless of what `q_layer_W` happens to hold there (should be zero, but the
        solver does not rely on that -- it only ever reads masked-in cells).
      * `x_centers_m`/`y_centers_m`: 1D cell-center coordinate axes. The `python_xy_layered`/
        `python_xy_sheet` finite-volume stencil below REQUIRES this to be a uniform, square-pixel
        grid (`dx == dy`, constant spacing) -- true of every grid
        `back_bombardment_deposition.build_back_bombardment_heat_source` builds -- and raises a
        clear `ValueError` otherwise rather than silently using a wrong stencil.
      * `layer_boundaries_um`: `(n_layers + 1,)` strictly increasing normal-depth layer boundaries
        [um], `layer_boundaries_um[0]=0` at the vacuum-facing surface -- identical convention to
        `back_bombardment_deposition.DEFAULT_DEPTH_LAYER_BOUNDARIES_UM`/
        `BackBombardmentHeatSource.layer_boundaries_um`. Layer index `0` is the shallowest
        (vacuum-facing, radiates) layer; layer index `n_layers-1` is the deepest (mount-facing,
        contact-conductance) layer.
      * `xy_cell_area_m2`: uniform lateral cell area [m^2] (`dx*dy`), matching
        `BackBombardmentHeatSource.xy_cell_area_m2` exactly.
    """

    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    layer_boundaries_um: np.ndarray
    cathode_footprint_mask: np.ndarray
    q_layer_W: np.ndarray
    t_grid_s: np.ndarray
    xy_cell_area_m2: float

    def __post_init__(self) -> None:
        x = np.asarray(self.x_centers_m, dtype=float)
        y = np.asarray(self.y_centers_m, dtype=float)
        lb = np.asarray(self.layer_boundaries_um, dtype=float)
        mask = np.asarray(self.cathode_footprint_mask, dtype=bool)
        q = np.asarray(self.q_layer_W, dtype=float)
        t = np.asarray(self.t_grid_s, dtype=float)
        object.__setattr__(self, "x_centers_m", x)
        object.__setattr__(self, "y_centers_m", y)
        object.__setattr__(self, "layer_boundaries_um", lb)
        object.__setattr__(self, "cathode_footprint_mask", mask)
        object.__setattr__(self, "q_layer_W", q)
        object.__setattr__(self, "t_grid_s", t)

        n_x, n_y = x.size, y.size
        n_layers = lb.size - 1
        n_bins = t.size - 1
        if lb.ndim != 1 or lb.size < 2 or not np.all(np.diff(lb) > 0.0):
            raise ValueError(f"layer_boundaries_um must be 1D strictly increasing, got {lb}")
        if mask.shape != (n_x, n_y):
            raise ValueError(f"cathode_footprint_mask shape {mask.shape} != (n_x,n_y)={(n_x, n_y)}")
        if t.ndim != 1 or t.size < 2 or not np.all(np.diff(t) > 0.0):
            raise ValueError("t_grid_s must be 1D strictly increasing with at least 2 entries (bin edges)")
        if q.shape != (n_x, n_y, n_layers, n_bins):
            raise ValueError(
                f"q_layer_W shape {q.shape} != (n_x,n_y,n_layers,n_t_bins)="
                f"{(n_x, n_y, n_layers, n_bins)} (n_t_bins = t_grid_s.size-1)"
            )
        if not (self.xy_cell_area_m2 > 0.0):
            raise ValueError(f"xy_cell_area_m2 must be positive, got {self.xy_cell_area_m2!r}")

    @property
    def n_layers(self) -> int:
        return int(self.layer_boundaries_um.size - 1)

    @property
    def layer_thickness_m(self) -> np.ndarray:
        return np.diff(self.layer_boundaries_um) * 1.0e-6


def build_constant_power_heat_source_time_series(
    bb_source: "Any",
    rf_period_s: float,
    t_grid_s: np.ndarray,
) -> VolumetricHeatSourceTimeSeries:
    """Minimal, "obviously-correct" conversion from one representative RF period's deposited
    ENERGY tensor (`back_bombardment_deposition.BackBombardmentHeatSource.q_layer_J`, joules per
    cell for that one period) into this module's time-resolved POWER contract
    (`VolumetricHeatSourceTimeSeries.q_layer_W`, watts per cell per time bin).

    THIS IS NOT THE MACROPULSE/RF-ENVELOPE SCALING (`rf_gun/macropulse.py`, explicitly out of this
    module's scope) -- it is nothing more than:
      1. an "obviously-correct multiplication" (per the task description) converting per-period
         energy to average power: `q_layer_W_bin = q_layer_J / rf_period_s` (dividing an energy
         deposited over one period by that period's duration gives the average power sustaining
         that same deposition rate if the period repeated indefinitely);
      2. holding that single constant power array across every bin of the caller-supplied
         `t_grid_s` (a trivial top-hat repetition, not a modeled RF fill/decay envelope).

    This exists purely so `back_bombardment_deposition`'s output has SOME obviously-correct,
    documented path into this module's own input contract for testing/development -- e.g. a
    caller building a synthetic constant-illumination macropulse case. A real macropulse study
    needs the actual RF envelope, keyframe interpolation, and current/temperature feedback loop
    (`rf_gun/macropulse.py`), none of which this helper attempts.

    `rf_period_s`: the duration of the one RF period `bb_source.q_layer_J` represents (e.g.
    `1/rf_frequency_Hz`) -- the caller's responsibility to supply correctly; this function does
    not infer it from anywhere (this project's HDF5 metadata carries an explicit
    `rf_period_s`/`rf_frequency_Hz`, plan Sec. 4.1, but reading that file is a different module's
    job).
    """
    if not (rf_period_s > 0.0):
        raise ValueError(f"rf_period_s must be positive, got {rf_period_s!r}")
    t_grid_s = np.asarray(t_grid_s, dtype=float)
    n_bins = t_grid_s.size - 1
    if n_bins < 1:
        raise ValueError("t_grid_s must have at least 2 entries (bin edges)")

    q_layer_W_one_bin = bb_source.q_layer_J / rf_period_s  # (n_x, n_y, n_layers)
    q_layer_W = np.repeat(q_layer_W_one_bin[:, :, :, np.newaxis], n_bins, axis=3)

    return VolumetricHeatSourceTimeSeries(
        x_centers_m=bb_source.x_centers_m,
        y_centers_m=bb_source.y_centers_m,
        layer_boundaries_um=bb_source.layer_boundaries_um,
        cathode_footprint_mask=bb_source.cathode_footprint_mask,
        q_layer_W=q_layer_W,
        t_grid_s=t_grid_s,
        xy_cell_area_m2=bb_source.xy_cell_area_m2,
    )


def _rebin_depth_profile(orig_boundaries_um: np.ndarray, orig_values: np.ndarray, new_boundaries_um: np.ndarray) -> np.ndarray:
    """Redistribute an extensive per-layer quantity (e.g. total power [W] summed over x,y) from
    `orig_boundaries_um`'s depth bins onto `new_boundaries_um`'s depth bins by exact interval
    overlap, assuming each original bin's value is spread UNIFORMLY across its own depth extent.
    Conserves the total (`sum(orig_values) == sum(result)`) exactly by construction -- used by
    `uh_legacy_1d`/`lumped_energy_check` to move a source built on the layered solver's own
    (possibly geometric) depth grid onto their own, differently-resolved depth grids without
    losing or fabricating energy.
    """
    orig_boundaries_um = np.asarray(orig_boundaries_um, dtype=float)
    new_boundaries_um = np.asarray(new_boundaries_um, dtype=float)
    orig_values = np.asarray(orig_values, dtype=float)
    n_new = new_boundaries_um.size - 1
    out = np.zeros(n_new, dtype=float)
    orig_widths = np.diff(orig_boundaries_um)
    for j in range(orig_values.size):
        if orig_widths[j] <= 0.0 or orig_values[j] == 0.0:
            continue
        lo, hi = orig_boundaries_um[j], orig_boundaries_um[j + 1]
        density = orig_values[j] / orig_widths[j]  # value per um, uniform within bin j
        overlap_lo = np.maximum(new_boundaries_um[:-1], lo)
        overlap_hi = np.minimum(new_boundaries_um[1:], hi)
        overlap = np.clip(overlap_hi - overlap_lo, 0.0, None)
        out += density * overlap
    return out


# ================================================================================================
# Reported-output container (plan Sec. 6.2's explicit list)
# ================================================================================================


@dataclass(frozen=True)
class ThermalResult:
    """Reported outputs (plan Sec. 6.2), common to every backend. Fields that are meaningless for
    a given backend (e.g. spatial maps for `lumped_energy_check`) are still populated with
    minimal/degenerate (typically single-cell) arrays rather than `None`, so downstream code (e.g.
    plotting) can treat every backend uniformly; `backend` tells a caller what it is actually
    looking at.
    """

    backend: str
    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    layer_boundaries_um: np.ndarray
    cathode_footprint_mask: np.ndarray
    t_grid_s: np.ndarray  # (n_t,) = bin edges, same convention as the input heat source

    T_surface_xyt: np.ndarray  # (n_x, n_y, n_t) -- top (vacuum-facing) layer, always stored
    T_layer_xyzt: np.ndarray | None  # (n_x, n_y, n_layers, n_t) if store_layer_history else None
    T_final_xyz: np.ndarray  # (n_x, n_y, n_layers) final full-depth state, always stored

    T_center_t: np.ndarray  # (n_t,)
    T_area_average_t: np.ndarray  # (n_t,)
    T_max_t: np.ndarray  # (n_t,)
    T_max_location_xy_t: np.ndarray  # (n_t, 2)
    T_flat_mean_t: np.ndarray  # (n_t,)
    T_bevel_mean_t: np.ndarray  # (n_t,)
    hotspot_centroid_xy_t: np.ndarray  # (n_t, 2)
    lineout_x_final: tuple  # (x_coords_m, T_K) through the final-time hotspot row
    lineout_y_final: tuple  # (y_coords_m, T_K) through the final-time hotspot column

    stored_energy_J_t: np.ndarray  # (n_t,), cumulative, relative to t_grid_s[0] (=0.0 there)
    radiation_loss_power_W_t: np.ndarray  # (n_t,)
    contact_loss_power_W_t: np.ndarray  # (n_t,)
    total_input_energy_J: float
    energy_residual_normalized: float

    picard_iterations_per_step: np.ndarray
    n_layers: int
    material_property_set: str
    thermal_config: ThermalConfig


# ================================================================================================
# Shared finite-volume core (python_xy_layered / python_xy_sheet / uh_legacy_1d all route here)
# ================================================================================================


def _validate_uniform_square_grid(x_centers_m: np.ndarray, y_centers_m: np.ndarray) -> tuple[float, float]:
    """Returns `(dx, dy)`, raising `ValueError` unless both axes have constant spacing and
    `dx == dy` (to a small relative tolerance) -- the assumption the in-plane FV stencil below
    relies on (module docstring)."""
    if x_centers_m.size < 2:
        dx = float(y_centers_m[1] - y_centers_m[0]) if y_centers_m.size >= 2 else 1.0
        return dx, dx
    if y_centers_m.size < 2:
        dx = float(x_centers_m[1] - x_centers_m[0])
        return dx, dx
    dxs = np.diff(x_centers_m)
    dys = np.diff(y_centers_m)
    dx = float(dxs[0])
    dy = float(dys[0])
    if not np.allclose(dxs, dx, rtol=1e-9) or not np.allclose(dys, dy, rtol=1e-9):
        raise ValueError(
            "python_xy_layered/python_xy_sheet require a uniform-spacing (x,y) grid; got "
            f"non-constant spacing (x diffs range [{dxs.min():.6g},{dxs.max():.6g}], y diffs "
            f"range [{dys.min():.6g},{dys.max():.6g}])"
        )
    if not np.isclose(dx, dy, rtol=1e-6):
        raise ValueError(
            f"python_xy_layered/python_xy_sheet require a square-pixel (x,y) grid (dx==dy); got "
            f"dx={dx:.6g}, dy={dy:.6g}"
        )
    return dx, dy


def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Harmonic mean, safe against a<=0 or b<=0 (returns 0 there -- a conductivity should never be
    non-positive; this only guards against a defensive edge case, e.g. an unphysical Picard
    overshoot, rather than raising mid-solve)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = a + b
    safe = np.where(denom > 0.0, denom, 1.0)
    result = np.where(denom > 0.0, 2.0 * a * b / safe, 0.0)
    return result


@dataclass
class _FVGrid:
    """Precomputed active-cell bookkeeping for the shared FV core: which `(ix,iy)` lateral cells
    are active (`cathode_footprint_mask`), their linear index, and their in-plane neighbor pairs."""

    nx: int
    ny: int
    n_layers: int
    active_ix: np.ndarray  # (n_active,)
    active_iy: np.ndarray  # (n_active,)
    active_index_of: np.ndarray  # (nx, ny) -> linear active index, or -1 if not active
    neighbor_pairs: list  # list of (a_index, b_index) active-index pairs, each unordered pair once
    dx: float
    n_active: int

    def dof(self, active_index: np.ndarray, ell: int) -> np.ndarray:
        return active_index * self.n_layers + ell


def _build_fv_grid(mask: np.ndarray, dx: float, n_layers: int) -> _FVGrid:
    nx, ny = mask.shape
    active_index_of = np.full((nx, ny), -1, dtype=int)
    active_ix, active_iy = np.nonzero(mask)
    active_index_of[active_ix, active_iy] = np.arange(active_ix.size)
    n_active = active_ix.size

    pairs = []
    for a in range(n_active):
        ix, iy = active_ix[a], active_iy[a]
        if ix + 1 < nx and active_index_of[ix + 1, iy] >= 0:
            pairs.append((a, active_index_of[ix + 1, iy]))
        if iy + 1 < ny and active_index_of[ix, iy + 1] >= 0:
            pairs.append((a, active_index_of[ix, iy + 1]))
    return _FVGrid(
        nx=nx,
        ny=ny,
        n_layers=n_layers,
        active_ix=active_ix,
        active_iy=active_iy,
        active_index_of=active_index_of,
        neighbor_pairs=pairs,
        dx=dx,
        n_active=n_active,
    )


def _assemble_and_solve_step(
    grid: _FVGrid,
    T_iter: np.ndarray,  # (n_active, n_layers) current Picard iterate
    T_prev: np.ndarray,  # (n_active, n_layers) converged state at the start of this step
    q_bin_W: np.ndarray,  # (n_active, n_layers) power this step (constant over dt)
    dt: float,
    xy_cell_area_m2: float,
    layer_thickness_m: np.ndarray,
    thermal: ThermalComponent,
    emissivity: float,
    T_env_K: float,
    contact_h: float,
    T_mount_K: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One Picard linear solve: returns `(T_new, C_used, q_rad_used_W, q_contact_used_W)` --
    `C_used`/`q_rad_used_W`/`q_contact_used_W` are exactly the coefficients this call assembled
    with (evaluated at `T_iter`, except `q_contact_used_W`, which is linear in `T_new` and is
    evaluated at the returned `T_new` since the contact term is exact, not Picard-approximated;
    see module docstring's "Energy accounting" paragraph) -- returned so the caller can build an
    energy-consistent diagnostic from the SAME numbers the assembly used.
    """
    n_active, n_layers = T_iter.shape
    n_dof = n_active * n_layers
    A = grid.dx  # cell area factor for in-plane flux (dx==dy, so face-area/distance ratio is 1)

    rho0 = float(thermal.rho_kg_m3(300.0, apply_expansion=False))  # constant regardless of T (see ThermalComponent)
    cp_iter = np.asarray(thermal.cp_J_kg_K(T_iter), dtype=float)  # (n_active, n_layers)
    k_iter = np.asarray(thermal.k_W_m_K(T_iter), dtype=float)  # (n_active, n_layers)

    C_used = rho0 * cp_iter * layer_thickness_m[np.newaxis, :]  # (n_active, n_layers), J/m^2/K

    rows: list = []
    cols: list = []
    data: list = []
    b = np.zeros(n_dof, dtype=float)

    # Transient (accumulation) term + RHS anchor to T_prev.
    diag_accum = C_used * xy_cell_area_m2 / dt  # (n_active, n_layers), W/K
    for ell in range(n_layers):
        dofs = grid.dof(np.arange(n_active), ell)
        rows.append(dofs)
        cols.append(dofs)
        data.append(diag_accum[:, ell])
        b[dofs] += diag_accum[:, ell] * T_prev[:, ell]
        # Deposited power this step (already Watts per cell, per input contract).
        b[dofs] += q_bin_W[:, ell]

    # In-plane conduction (harmonic-mean face conductance), each unordered neighbor pair once.
    if grid.neighbor_pairs:
        pair_a = np.array([p[0] for p in grid.neighbor_pairs], dtype=int)
        pair_b = np.array([p[1] for p in grid.neighbor_pairs], dtype=int)
        for ell in range(n_layers):
            k_face = _harmonic_mean(k_iter[pair_a, ell], k_iter[pair_b, ell])
            g_inplane = k_face * layer_thickness_m[ell] / A  # W/K per face, dx==dy assumed
            dof_a = grid.dof(pair_a, ell)
            dof_b = grid.dof(pair_b, ell)
            # a's equation: -g*(T_b - T_a) moved to LHS as +g*T_a - g*T_b
            rows.append(dof_a); cols.append(dof_a); data.append(g_inplane)
            rows.append(dof_a); cols.append(dof_b); data.append(-g_inplane)
            # b's equation (symmetric):
            rows.append(dof_b); cols.append(dof_b); data.append(g_inplane)
            rows.append(dof_b); cols.append(dof_a); data.append(-g_inplane)

    # Through-depth conductance between adjacent layers of the same active cell.
    for ell in range(n_layers - 1):
        k_face = _harmonic_mean(k_iter[:, ell], k_iter[:, ell + 1])
        dz_avg = 0.5 * (layer_thickness_m[ell] + layer_thickness_m[ell + 1])
        G = np.where(dz_avg > 0.0, k_face / np.where(dz_avg > 0.0, dz_avg, 1.0), 0.0)  # W/m^2/K
        g_depth = G * xy_cell_area_m2  # W/K
        dof_top = grid.dof(np.arange(n_active), ell)
        dof_bot = grid.dof(np.arange(n_active), ell + 1)
        rows.append(dof_top); cols.append(dof_top); data.append(g_depth)
        rows.append(dof_top); cols.append(dof_bot); data.append(-g_depth)
        rows.append(dof_bot); cols.append(dof_bot); data.append(g_depth)
        rows.append(dof_bot); cols.append(dof_top); data.append(-g_depth)

    # Top-layer radiative loss (ell=0): re-evaluated at T_iter, folded into RHS as a fixed source
    # for this Picard iteration (see module docstring).
    T_top_iter = T_iter[:, 0]
    q_rad_used_W = emissivity * Stefan_Boltzmann * (T_top_iter**4 - T_env_K**4) * xy_cell_area_m2
    dof_top0 = grid.dof(np.arange(n_active), 0)
    b[dof_top0] -= q_rad_used_W

    # Bottom-layer contact conductance (ell=n_layers-1): LINEAR in the unknown, so included
    # directly in the matrix (exact, not Picard-approximated).
    ell_bot = n_layers - 1
    dof_bot0 = grid.dof(np.arange(n_active), ell_bot)
    g_contact = contact_h * xy_cell_area_m2  # W/K
    if g_contact > 0.0:
        rows.append(dof_bot0); cols.append(dof_bot0); data.append(np.full(n_active, g_contact))
        b[dof_bot0] += g_contact * T_mount_K

    row_arr = np.concatenate(rows) if rows else np.array([], dtype=int)
    col_arr = np.concatenate(cols) if cols else np.array([], dtype=int)
    data_arr = np.concatenate(data) if data else np.array([], dtype=float)
    M = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(n_dof, n_dof)).tocsr()

    T_new_flat = spsolve(M, b)
    T_new = T_new_flat.reshape(n_active, n_layers)

    # Contact loss evaluated at the (exact, not iterate-approximate) solved T_new.
    q_contact_used_W = np.zeros(n_active, dtype=float)
    if g_contact > 0.0:
        q_contact_used_W = g_contact * (T_new[:, ell_bot] - T_mount_K)

    return T_new, C_used, q_rad_used_W, q_contact_used_W


def _solve_fv_core(
    x_centers_m: np.ndarray,
    y_centers_m: np.ndarray,
    layer_boundaries_um: np.ndarray,
    mask: np.ndarray,
    xy_cell_area_m2: float,
    q_layer_W: np.ndarray,  # (nx, ny, n_layers, n_bins)
    t_grid_s: np.ndarray,  # (n_bins+1,)
    T0_xyz: np.ndarray,  # (nx, ny, n_layers), nan outside mask
    thermal: ThermalComponent,
    config: ThermalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Shared implicit-Euler + Picard time-marching loop. Returns
    `(T_history_xyzt, stored_energy_J_t, radiation_loss_W_t, contact_loss_W_t, picard_iters,
    total_input_energy_J, energy_residual_normalized, dx)` where `T_history_xyzt` has shape
    `(nx, ny, n_layers, n_t)` with `n_t = t_grid_s.size` (one slice per BIN EDGE, i.e. exactly the
    macro-time points the caller asked for, regardless of any internal `dt_s` sub-stepping).
    """
    dx, _dy = _validate_uniform_square_grid(x_centers_m, y_centers_m)
    n_layers = layer_boundaries_um.size - 1
    layer_thickness_m = np.diff(layer_boundaries_um) * 1.0e-6
    grid = _build_fv_grid(mask, dx, n_layers)
    n_active = grid.n_active
    n_t = t_grid_s.size
    n_bins = n_t - 1

    T_active = T0_xyz[grid.active_ix, grid.active_iy, :].copy()  # (n_active, n_layers)

    T_history = np.full((grid.nx, grid.ny, n_layers, n_t), np.nan, dtype=float)
    T_history[grid.active_ix, grid.active_iy, :, 0] = T_active

    stored_energy_J_t = np.zeros(n_t, dtype=float)
    radiation_loss_W_t = np.zeros(n_t, dtype=float)
    contact_loss_W_t = np.zeros(n_t, dtype=float)
    picard_iters_per_bin = np.zeros(n_bins, dtype=int)

    cumulative_lhs_J = 0.0  # sum of C_used*A*(T_new-T_prev), the exact stored-energy tracker
    cumulative_qin_J = 0.0
    cumulative_qloss_J = 0.0

    # Radiation/contact loss reported at t_grid_s[0] uses the initial state.
    T_top0 = T_active[:, 0]
    radiation_loss_W_t[0] = float(
        np.sum(config.total_hemispherical_emissivity * Stefan_Boltzmann * (T_top0**4 - config.T_environment_K**4))
        * xy_cell_area_m2
    )
    if config.contact_h_W_m2K > 0.0:
        T_bot0 = T_active[:, -1]
        contact_loss_W_t[0] = float(np.sum(config.contact_h_W_m2K * (T_bot0 - config.T_mount_K)) * xy_cell_area_m2)

    hist_idx = 1
    for ib in range(n_bins):
        t_lo, t_hi = float(t_grid_s[ib]), float(t_grid_s[ib + 1])
        bin_width = t_hi - t_lo
        q_bin_active = q_layer_W[grid.active_ix, grid.active_iy, :, ib]  # (n_active, n_layers)

        if config.dt_s is None:
            substep_dts = [bin_width]
        else:
            n_sub = max(1, int(np.ceil(bin_width / config.dt_s)))
            substep_dts = [bin_width / n_sub] * n_sub

        max_iters_this_bin = 0
        for dt in substep_dts:
            T_prev = T_active.copy()
            T_iter = T_prev.copy()
            converged = False
            for it in range(config.picard_max_iter):
                T_new, C_used, q_rad_used_W, q_contact_used_W = _assemble_and_solve_step(
                    grid,
                    T_iter,
                    T_prev,
                    q_bin_active,
                    dt,
                    xy_cell_area_m2,
                    layer_thickness_m,
                    thermal,
                    config.total_hemispherical_emissivity,
                    config.T_environment_K,
                    config.contact_h_W_m2K,
                    config.T_mount_K,
                )
                delta = float(np.max(np.abs(T_new - T_iter))) if n_active else 0.0
                T_iter = T_new
                max_iters_this_bin = max(max_iters_this_bin, it + 1)
                if delta < config.picard_tol_K:
                    converged = True
                    break
            if not converged:
                raise RuntimeError(
                    f"Picard iteration failed to converge within picard_max_iter="
                    f"{config.picard_max_iter} (last delta={delta:.6g} K >= "
                    f"picard_tol_K={config.picard_tol_K:.6g} K) at macro-time bin [{t_lo:.6g}, "
                    f"{t_hi:.6g}] s. Consider a smaller dt_s, a larger picard_max_iter, or a "
                    "larger picard_tol_K if this case is known to be numerically difficult."
                )

            T_active = T_iter
            cumulative_lhs_J += float(np.sum(C_used * xy_cell_area_m2 * (T_active - T_prev)))
            cumulative_qin_J += float(np.sum(q_bin_active)) * dt
            cumulative_qloss_J += float(np.sum(q_rad_used_W) + np.sum(q_contact_used_W)) * dt

        picard_iters_per_bin[ib] = max_iters_this_bin
        T_history[grid.active_ix, grid.active_iy, :, hist_idx] = T_active
        stored_energy_J_t[hist_idx] = cumulative_lhs_J

        T_top = T_active[:, 0]
        radiation_loss_W_t[hist_idx] = float(
            np.sum(
                config.total_hemispherical_emissivity * Stefan_Boltzmann * (T_top**4 - config.T_environment_K**4)
            )
            * xy_cell_area_m2
        )
        if config.contact_h_W_m2K > 0.0:
            T_bot = T_active[:, -1]
            contact_loss_W_t[hist_idx] = float(
                np.sum(config.contact_h_W_m2K * (T_bot - config.T_mount_K)) * xy_cell_area_m2
            )
        hist_idx += 1

    total_input_energy_J = cumulative_qin_J
    residual_J = cumulative_lhs_J - (cumulative_qin_J - cumulative_qloss_J)
    # Normalize by the largest of input/loss/stored energy magnitudes, not input energy alone --
    # a run with zero (or near-zero) net input energy (e.g. the radiation-only decay test) would
    # otherwise divide a tiny floating-point residual by a near-zero denominator and report a
    # meaningless, enormous "residual". Falls back to input energy alone (plan Sec. 15.3's own
    # normalization) whenever it is not itself negligible.
    denom = max(abs(total_input_energy_J), abs(cumulative_qloss_J), abs(cumulative_lhs_J), 1e-30)
    energy_residual_normalized = residual_J / denom

    return (
        T_history,
        stored_energy_J_t,
        radiation_loss_W_t,
        contact_loss_W_t,
        picard_iters_per_bin,
        total_input_energy_J,
        energy_residual_normalized,
        dx,
    )


# ================================================================================================
# Diagnostic/reported-scalar helpers (shared by every spatially-resolved backend)
# ================================================================================================


def _scalar_diagnostics(
    T_history_xyzt: np.ndarray,
    x_centers_m: np.ndarray,
    y_centers_m: np.ndarray,
    mask: np.ndarray,
    geometry: CathodeGeometry | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(T_center_t, T_area_avg_t, T_max_t, T_max_loc_xy_t, T_flat_mean_t, T_bevel_mean_t,
    hotspot_xy_t)` -- all evaluated on the TOP layer (`ell=0`), matching the always-reported
    `T_surface(x,y,t)` (plan Sec. 6.2)."""
    nx, ny, n_layers, n_t = T_history_xyzt.shape
    active_ix, active_iy = np.nonzero(mask)
    xx, yy = np.meshgrid(x_centers_m, y_centers_m, indexing="ij")

    # Center cell: nearest (0,0).
    r2 = xx[active_ix, active_iy] ** 2 + yy[active_ix, active_iy] ** 2
    center_a = int(np.argmin(r2))
    ix_c, iy_c = active_ix[center_a], active_iy[center_a]

    if geometry is not None:
        r_mm = np.hypot(xx[active_ix, active_iy], yy[active_ix, active_iy]) * 1.0e3
        is_flat = r_mm <= geometry.flat_radius_mm
    else:
        is_flat = np.ones(active_ix.size, dtype=bool)

    T_center_t = np.full(n_t, np.nan)
    T_area_avg_t = np.full(n_t, np.nan)
    T_max_t = np.full(n_t, np.nan)
    T_max_loc_xy_t = np.full((n_t, 2), np.nan)
    T_flat_mean_t = np.full(n_t, np.nan)
    T_bevel_mean_t = np.full(n_t, np.nan)
    hotspot_xy_t = np.full((n_t, 2), np.nan)

    for it in range(n_t):
        T_top_active = T_history_xyzt[active_ix, active_iy, 0, it]
        T_center_t[it] = T_history_xyzt[ix_c, iy_c, 0, it]
        T_area_avg_t[it] = float(np.mean(T_top_active))
        argmax_a = int(np.argmax(T_top_active))
        T_max_t[it] = T_top_active[argmax_a]
        T_max_loc_xy_t[it] = (x_centers_m[active_ix[argmax_a]], y_centers_m[active_iy[argmax_a]])
        if np.any(is_flat):
            T_flat_mean_t[it] = float(np.mean(T_top_active[is_flat]))
        if np.any(~is_flat):
            T_bevel_mean_t[it] = float(np.mean(T_top_active[~is_flat]))
        weight = np.clip(T_top_active - np.min(T_top_active), 0.0, None)
        wsum = float(np.sum(weight))
        if wsum > 0.0:
            hx = float(np.sum(weight * x_centers_m[active_ix]) / wsum)
            hy = float(np.sum(weight * y_centers_m[active_iy]) / wsum)
        else:
            hx, hy = float(x_centers_m[ix_c]), float(y_centers_m[iy_c])
        hotspot_xy_t[it] = (hx, hy)

    return T_center_t, T_area_avg_t, T_max_t, T_max_loc_xy_t, T_flat_mean_t, T_bevel_mean_t, hotspot_xy_t


def _lineouts_through_hotspot(
    T_history_xyzt: np.ndarray, x_centers_m: np.ndarray, y_centers_m: np.ndarray, mask: np.ndarray
) -> tuple[tuple, tuple]:
    """x/y lineouts through the FINAL-time hotspot (plan Sec. 6.2), masked-out cells as `nan`."""
    T_final_top = T_history_xyzt[:, :, 0, -1]
    masked = np.where(mask, T_final_top, -np.inf)
    ix_h, iy_h = np.unravel_index(np.argmax(masked), masked.shape)
    line_x = np.where(mask[:, iy_h], T_final_top[:, iy_h], np.nan)
    line_y = np.where(mask[ix_h, :], T_final_top[ix_h, :], np.nan)
    return (x_centers_m.copy(), line_x), (y_centers_m.copy(), line_y)


# ================================================================================================
# Backend 1/2: python_xy_layered / python_xy_sheet (sheet = layered with n_layers=1)
# ================================================================================================


def _initial_T_array(
    initial_temperature: Any, x_centers_m: np.ndarray, y_centers_m: np.ndarray, mask: np.ndarray, n_layers: int
) -> np.ndarray:
    if not hasattr(initial_temperature, "on_grid"):
        raise TypeError(
            f"initial_temperature must be a ConstantTemperatureMap or TemperatureMap2D (got "
            f"{type(initial_temperature)!r})"
        )
    return initial_temperature.on_grid(x_centers_m, y_centers_m, mask, n_layers)


def _solve_python_xy_layered_or_sheet(
    heat_source: VolumetricHeatSourceTimeSeries,
    initial_temperature: Any,
    geometry: CathodeGeometry,
    material: CathodeMaterialSet,
    config: ThermalConfig,
    *,
    as_sheet: bool,
) -> ThermalResult:
    validate_material_for(material, "python_xy_layered")
    thermal = material.thermal

    if as_sheet:
        # Collapse to one depth-integrated layer of thickness h_eff_m (plan Sec. 6.3): sum the
        # deposited power over all original depth layers into a single layer.
        layer_boundaries_um = np.array([0.0, config.h_eff_m * 1.0e6])
        q_layer_W = np.sum(heat_source.q_layer_W, axis=2, keepdims=True)
    else:
        layer_boundaries_um = heat_source.layer_boundaries_um
        q_layer_W = heat_source.q_layer_W

    n_layers = layer_boundaries_um.size - 1
    mask = heat_source.cathode_footprint_mask
    T0_xyz = _initial_T_array(initial_temperature, heat_source.x_centers_m, heat_source.y_centers_m, mask, n_layers)

    (
        T_history,
        stored_energy_J_t,
        radiation_loss_W_t,
        contact_loss_W_t,
        picard_iters,
        total_input_energy_J,
        energy_residual_normalized,
        _dx,
    ) = _solve_fv_core(
        heat_source.x_centers_m,
        heat_source.y_centers_m,
        layer_boundaries_um,
        mask,
        heat_source.xy_cell_area_m2,
        q_layer_W,
        heat_source.t_grid_s,
        T0_xyz,
        thermal,
        config,
    )

    (
        T_center_t,
        T_area_avg_t,
        T_max_t,
        T_max_loc_xy_t,
        T_flat_mean_t,
        T_bevel_mean_t,
        hotspot_xy_t,
    ) = _scalar_diagnostics(T_history, heat_source.x_centers_m, heat_source.y_centers_m, mask, geometry)
    lineout_x, lineout_y = _lineouts_through_hotspot(T_history, heat_source.x_centers_m, heat_source.y_centers_m, mask)

    backend_name = "python_xy_sheet" if as_sheet else "python_xy_layered"
    return ThermalResult(
        backend=backend_name,
        x_centers_m=heat_source.x_centers_m,
        y_centers_m=heat_source.y_centers_m,
        layer_boundaries_um=layer_boundaries_um,
        cathode_footprint_mask=mask,
        t_grid_s=heat_source.t_grid_s,
        T_surface_xyt=T_history[:, :, 0, :],
        T_layer_xyzt=T_history if config.store_layer_history else None,
        T_final_xyz=T_history[:, :, :, -1],
        T_center_t=T_center_t,
        T_area_average_t=T_area_avg_t,
        T_max_t=T_max_t,
        T_max_location_xy_t=T_max_loc_xy_t,
        T_flat_mean_t=T_flat_mean_t,
        T_bevel_mean_t=T_bevel_mean_t,
        hotspot_centroid_xy_t=hotspot_xy_t,
        lineout_x_final=lineout_x,
        lineout_y_final=lineout_y,
        stored_energy_J_t=stored_energy_J_t,
        radiation_loss_power_W_t=radiation_loss_W_t,
        contact_loss_power_W_t=contact_loss_W_t,
        total_input_energy_J=total_input_energy_J,
        energy_residual_normalized=energy_residual_normalized,
        picard_iterations_per_step=picard_iters,
        n_layers=n_layers,
        material_property_set=material.property_set,
        thermal_config=config,
    )


# ================================================================================================
# Backend 3: lumped_energy_check (0D energy/units sanity check, plan Sec. 6.3)
# ================================================================================================


def _solve_lumped_energy_check(
    heat_source: VolumetricHeatSourceTimeSeries,
    initial_temperature: Any,
    geometry: CathodeGeometry,
    material: CathodeMaterialSet,
    config: ThermalConfig,
) -> ThermalResult:
    """Zero-dimensional energy/units check ONLY (plan Sec. 6.3): total energy in /
    `(rho*cp*Volume)` gives a single `dT_avg/dt`, no spatial resolution at all. Forward-Euler
    explicit stepping is used deliberately (not implicit/Picard) -- for the constant-property,
    zero-loss case this is a red herring anyway (the ODE is exactly linear, so any consistent
    scheme is exact to floating-point precision); explicit stepping keeps this backend maximally
    transparent as "just a sanity check", per its own name and plan Sec. 6.3's description.
    """
    validate_material_for(material, "python_xy_layered")
    thermal = material.thermal
    mask = heat_source.cathode_footprint_mask
    n_active = int(np.sum(mask))
    volume_m3 = heat_source.xy_cell_area_m2 * n_active * float(np.sum(heat_source.layer_thickness_m))
    if not (volume_m3 > 0.0):
        raise ValueError("lumped_energy_check: zero active cathode volume (empty footprint mask)")

    if isinstance(initial_temperature, ConstantTemperatureMap):
        T0 = float(initial_temperature.T0_K)
    else:
        T0_xyz = _initial_T_array(
            initial_temperature, heat_source.x_centers_m, heat_source.y_centers_m, mask, heat_source.n_layers
        )
        T0 = float(np.nanmean(T0_xyz))

    rho0 = float(thermal.rho_kg_m3(T0, apply_expansion=False))
    t_grid_s = heat_source.t_grid_s
    n_t = t_grid_s.size
    n_bins = n_t - 1

    q_total_W_per_bin = np.array(
        [float(np.sum(heat_source.q_layer_W[:, :, :, ib])) for ib in range(n_bins)]
    )

    T_avg_t = np.zeros(n_t, dtype=float)
    T_avg_t[0] = T0
    stored_energy_J_t = np.zeros(n_t, dtype=float)
    radiation_loss_W_t = np.zeros(n_t, dtype=float)
    contact_loss_W_t = np.zeros(n_t, dtype=float)

    A_top_total = heat_source.xy_cell_area_m2 * n_active  # top-surface area of the whole footprint

    def _rad_loss_W(T: float) -> float:
        return config.total_hemispherical_emissivity * Stefan_Boltzmann * (T**4 - config.T_environment_K**4) * A_top_total

    def _contact_loss_W(T: float) -> float:
        return config.contact_h_W_m2K * (T - config.T_mount_K) * A_top_total

    radiation_loss_W_t[0] = _rad_loss_W(T0)
    contact_loss_W_t[0] = _contact_loss_W(T0)

    cumulative_qin_J = 0.0
    cumulative_qloss_J = 0.0
    T_prev = T0
    for ib in range(n_bins):
        dt = float(t_grid_s[ib + 1] - t_grid_s[ib])
        cp_prev = float(thermal.cp_J_kg_K(T_prev))
        heat_capacity_J_K = rho0 * cp_prev * volume_m3
        q_loss_W = _rad_loss_W(T_prev) + _contact_loss_W(T_prev)
        dT = dt * (q_total_W_per_bin[ib] - q_loss_W) / heat_capacity_J_K
        T_new = T_prev + dT
        cumulative_qin_J += q_total_W_per_bin[ib] * dt
        cumulative_qloss_J += q_loss_W * dt
        stored_energy_J_t[ib + 1] = stored_energy_J_t[ib] + heat_capacity_J_K * dT
        T_avg_t[ib + 1] = T_new
        radiation_loss_W_t[ib + 1] = _rad_loss_W(T_new)
        contact_loss_W_t[ib + 1] = _contact_loss_W(T_new)
        T_prev = T_new

    total_input_energy_J = cumulative_qin_J
    residual_J = stored_energy_J_t[-1] - (cumulative_qin_J - cumulative_qloss_J)
    denom = max(abs(total_input_energy_J), abs(cumulative_qloss_J), abs(stored_energy_J_t[-1]), 1e-30)
    energy_residual_normalized = residual_J / denom

    # Degenerate single-cell spatial maps (this backend has no spatial resolution at all).
    x1 = np.array([0.0])
    y1 = np.array([0.0])
    mask1 = np.array([[True]])
    T_surface_1t = T_avg_t.reshape(1, 1, n_t)
    T_final_1z = np.array([[[T_avg_t[-1]]]])

    return ThermalResult(
        backend="lumped_energy_check",
        x_centers_m=x1,
        y_centers_m=y1,
        layer_boundaries_um=np.array([0.0, 1.0]),
        cathode_footprint_mask=mask1,
        t_grid_s=t_grid_s,
        T_surface_xyt=T_surface_1t,
        T_layer_xyzt=T_surface_1t.reshape(1, 1, 1, n_t) if config.store_layer_history else None,
        T_final_xyz=T_final_1z,
        T_center_t=T_avg_t.copy(),
        T_area_average_t=T_avg_t.copy(),
        T_max_t=T_avg_t.copy(),
        T_max_location_xy_t=np.zeros((n_t, 2)),
        T_flat_mean_t=T_avg_t.copy(),
        T_bevel_mean_t=np.full(n_t, np.nan),
        hotspot_centroid_xy_t=np.zeros((n_t, 2)),
        lineout_x_final=(x1, np.array([T_avg_t[-1]])),
        lineout_y_final=(y1, np.array([T_avg_t[-1]])),
        stored_energy_J_t=stored_energy_J_t,
        radiation_loss_power_W_t=radiation_loss_W_t,
        contact_loss_power_W_t=contact_loss_W_t,
        total_input_energy_J=total_input_energy_J,
        energy_residual_normalized=energy_residual_normalized,
        picard_iterations_per_step=np.zeros(n_bins, dtype=int),
        n_layers=1,
        material_property_set=material.property_set,
        thermal_config=config,
    )


# ================================================================================================
# Backend 4: uh_legacy_1d (historical depth-only benchmark, plan Sec. 6.3)
# ================================================================================================


def _solve_uh_legacy_1d(
    heat_source: VolumetricHeatSourceTimeSeries,
    initial_temperature: Any,
    geometry: CathodeGeometry,
    material: CathodeMaterialSet,
    config: ThermalConfig,
) -> ThermalResult:
    """1D depth-only (no x,y resolution -- single point, uniform illumination assumed) diffusion
    solve using the ALREADY-IMPLEMENTED legacy closed-form properties
    (`LaB6_Kowalczyk_PRSTAB120402_2014_legacy`'s `k_UH(T)`, `D_UH(T)`/`cp_UH(T)`), reusing the same
    `_solve_fv_core` machinery with a single lateral cell (`n_x=n_y=1`, whose area is the ENTIRE
    cathode footprint area, i.e. "uniform illumination") and its own independent uniform depth
    grid (`config.uh_legacy_n_depth_layers` layers spanning `[0, geometry.cathode_length_mm]`,
    NOT the layered solver's own `layer_boundaries_um` convergence grid).

    Reproduces the historical UH gun's own reduced 1D thermal model -- NOT the actual Kowalczyk
    2013 experimental comparison (synthetic radiometric temperature, measured RF envelope), which
    is explicitly out of scope here (plan Sec. 14; this task only builds the SOLVER backend).
    """
    if material.property_set != "LaB6_Kowalczyk_PRSTAB120402_2014_legacy":
        raise ValueError(
            f"uh_legacy_1d requires material.property_set == "
            f"'LaB6_Kowalczyk_PRSTAB120402_2014_legacy' (this benchmark exists specifically to "
            f"reproduce that legacy closed-form set); got {material.property_set!r}. Load it "
            "explicitly via rf_gun.load_cathode_material('LaB6', "
            "'LaB6_Kowalczyk_PRSTAB120402_2014_legacy') rather than silently substituting it."
        )
    thermal = material.thermal

    n_depth = config.uh_legacy_n_depth_layers
    length_um = geometry.cathode_length_mm * 1000.0
    layer_boundaries_um = np.linspace(0.0, length_um, n_depth + 1)

    A_total = heat_source.xy_cell_area_m2 * int(np.sum(heat_source.cathode_footprint_mask))
    if not (A_total > 0.0):
        raise ValueError("uh_legacy_1d: zero active cathode footprint area in the supplied heat_source")

    x1 = np.array([0.0])
    y1 = np.array([0.0])
    mask1 = np.array([[True]])

    n_bins = heat_source.t_grid_s.size - 1
    orig_boundaries_um = heat_source.layer_boundaries_um
    q_layer_W_1d = np.zeros((1, 1, n_depth, n_bins), dtype=float)
    for ib in range(n_bins):
        # Total power vs. ORIGINAL depth layer, summed over x,y (uniform-illumination collapse).
        orig_totals_W = np.sum(heat_source.q_layer_W[:, :, :, ib], axis=(0, 1))
        q_layer_W_1d[0, 0, :, ib] = _rebin_depth_profile(orig_boundaries_um, orig_totals_W, layer_boundaries_um)

    if isinstance(initial_temperature, ConstantTemperatureMap):
        T0_xyz = np.full((1, 1, n_depth), initial_temperature.T0_K)
    else:
        T0_full = _initial_T_array(
            initial_temperature, heat_source.x_centers_m, heat_source.y_centers_m,
            heat_source.cathode_footprint_mask, 1,
        )
        T0_uniform = float(np.nanmean(T0_full))
        T0_xyz = np.full((1, 1, n_depth), T0_uniform)

    (
        T_history,
        stored_energy_J_t,
        radiation_loss_W_t,
        contact_loss_W_t,
        picard_iters,
        total_input_energy_J,
        energy_residual_normalized,
        _dx,
    ) = _solve_fv_core(
        x1, y1, layer_boundaries_um, mask1, A_total, q_layer_W_1d, heat_source.t_grid_s, T0_xyz, thermal, config,
    )

    n_t = heat_source.t_grid_s.size
    T_center_t = T_history[0, 0, 0, :]
    depth_um = 0.5 * (layer_boundaries_um[:-1] + layer_boundaries_um[1:])

    return ThermalResult(
        backend="uh_legacy_1d",
        x_centers_m=x1,
        y_centers_m=y1,
        layer_boundaries_um=layer_boundaries_um,
        cathode_footprint_mask=mask1,
        t_grid_s=heat_source.t_grid_s,
        T_surface_xyt=T_history[:, :, 0, :],
        T_layer_xyzt=T_history if config.store_layer_history else None,
        T_final_xyz=T_history[:, :, :, -1],
        T_center_t=T_center_t.copy(),
        T_area_average_t=T_center_t.copy(),
        T_max_t=T_center_t.copy(),
        T_max_location_xy_t=np.zeros((n_t, 2)),
        T_flat_mean_t=T_center_t.copy(),
        T_bevel_mean_t=np.full(n_t, np.nan),
        hotspot_centroid_xy_t=np.zeros((n_t, 2)),
        lineout_x_final=(x1, np.array([T_history[0, 0, 0, -1]])),
        lineout_y_final=(depth_um, T_history[0, 0, :, -1]),
        stored_energy_J_t=stored_energy_J_t,
        radiation_loss_power_W_t=radiation_loss_W_t,
        contact_loss_power_W_t=contact_loss_W_t,
        total_input_energy_J=total_input_energy_J,
        energy_residual_normalized=energy_residual_normalized,
        picard_iterations_per_step=picard_iters,
        n_layers=n_depth,
        material_property_set=material.property_set,
        thermal_config=config,
    )


# ================================================================================================
# Public dispatch entry point
# ================================================================================================


def solve_xy_layered_thermal(
    heat_source: VolumetricHeatSourceTimeSeries,
    initial_temperature: Any,
    geometry: CathodeGeometry,
    material: CathodeMaterialSet,
    thermal_config: ThermalConfig,
) -> ThermalResult:
    """Dispatch on `thermal_config.backend` (plan Sec. 10.2's suggested public function; addendum
    Sec. 19.2, Work Package 3 scope): `"python_xy_layered"` (default), `"python_xy_sheet"`,
    `"lumped_energy_check"`, or `"uh_legacy_1d"`.

    `heat_source` must be a `VolumetricHeatSourceTimeSeries` (this module's own documented
    time-resolved power contract -- see that class's docstring for the exact shape/units
    convention); build one from a `back_bombardment_deposition.BackBombardmentHeatSource` via
    `build_constant_power_heat_source_time_series` if needed (NOT the real macropulse scaling --
    see that helper's docstring).
    """
    backend = thermal_config.backend
    if backend == "python_xy_layered":
        return _solve_python_xy_layered_or_sheet(
            heat_source, initial_temperature, geometry, material, thermal_config, as_sheet=False
        )
    if backend == "python_xy_sheet":
        return _solve_python_xy_layered_or_sheet(
            heat_source, initial_temperature, geometry, material, thermal_config, as_sheet=True
        )
    if backend == "lumped_energy_check":
        return _solve_lumped_energy_check(heat_source, initial_temperature, geometry, material, thermal_config)
    if backend == "uh_legacy_1d":
        return _solve_uh_legacy_1d(heat_source, initial_temperature, geometry, material, thermal_config)
    raise ValueError(
        f"Unknown thermal_config.backend {backend!r}; valid values: 'python_xy_layered' "
        "(default), 'python_xy_sheet', 'lumped_energy_check', 'uh_legacy_1d' (comsol_3d is a "
        "separate, not-yet-implemented interface per addendum Sec. 19.2's Work Package 5 scope)."
    )
