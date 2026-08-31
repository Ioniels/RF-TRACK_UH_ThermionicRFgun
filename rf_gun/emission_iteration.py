"""Emission Fields Iteration (implementation guide Secs. 7-8): a reduced-cost, near-cathode,
under-relaxed Picard iteration coupling the emission current to the space-charge and cathode
mirror fields it generates, run once between the fast phase scan and the full production
transport.

Cathode grid: a regular Cartesian (x,y) grid over the cathode's bounding square, clipped to the
disk by zeroing any cell whose center falls outside cathode_radius_mm (area and charge weight
both zero there, so those cells never emit). This is deliberately *not* an axisymmetric r-only
grid: a heater/backbombardment-driven temperature profile T(x,y), and the resulting emission
profile, need not be azimuthally symmetric, so every array here is genuinely 2D in space
(shape (n_x, n_y, n_t)) rather than collapsing to a radial coordinate. A regular grid (rather than
a flat, disk-only point list) keeps every field a plain rectangular array, which is what makes a
plain imshow/pcolormesh of the cathode state possible without extra bookkeeping.

Near-cathode field approximation (documented explicitly, not hidden): RF-Track's Python bindings
expose no mid-tracking callback returning intermediate bunch snapshots (guide Sec. 9.1 flags this
as needing investigation; none of the segmented-tracking/watchpoint mechanisms it lists are wired
up here). This iteration instead evaluates, for each emission-time bin t_j, the space-charge and
mirror field from all *already-emitted* fixed-sample macroparticles (t_i <= t_j), each placed at a
ballistically-estimated z (guide's own framing: "the charge is near the cathode and still at low
energy... restrict the iteration to the relevant early region") rather than exactly at z=0 --
placing a particle exactly at z=0 would put its own mirror image at the identical point (opposite
sign, same location), so source and image would exactly cancel for any external probe regardless
of the true charge state (see _ballistic_z_drift_mm). It is a first-order approximation, not a
claim of full dynamical fidelity -- a natural future refinement is segmented tracking with real
per-bin snapshots (guide Sec. 9.1, mechanism 3).

Fixed source samples (guide Sec. 8.5): positions, azimuths, and momentum variates are drawn once
and held fixed across outer iterations; only each macroparticle's weight (the Bunch6dT extended
matrix's "N" column, already used for per-particle real-particle counts throughout this
repository -- see rf_gun.simulation.build_bunch_thermionic) is updated each iteration. This avoids
Monte Carlo resampling noise from masquerading as iteration non-convergence.

Cathode temperature: `cathode_temperature_K` accepts a scalar (uniform temperature, the default,
matching every prior run of this code) or a callable `T_K(x_mm, y_mm) -> T_K` for a spatially
resolved profile -- e.g. fit to a steady-state heater simulation or a measured backbombardment
power map. The temperature used here is the profile applicable *during this run's emission
window*; slower (thermal-timescale) evolution of T(x,y) between shots is the caller's
responsibility (run this iteration again with an updated profile), not something this single
self-consistency study models internally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .constants import ME_MEV, c, q_e
from .emission_models import evaluate_emission_model
from .emission_sampling import sample_thermionic_momenta
from .work_function_models import evaluate_work_function_eV
from .cathode_fields import (
    extraction_field,
    sample_rf_field_on_cathode,
    extract_sc_and_mirror_from_snapshot,
    analytic_sc_and_mirror_surface_field,
)
from .beam_loading_envelope import estimate_beam_induced_cathode_field_from_current_density
from .rftrack_volume import inspect_rftrack_capabilities

#: A uniform temperature (matches every prior version of this code) or a callable T(x_mm,y_mm).
TemperatureField = Union[float, Callable[[np.ndarray, np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class EmissionFieldIterationConfig:
    enabled: bool = True
    finesse: str = "medium"
    max_iterations: int = 12
    relaxation: float = 0.30
    relaxation_min: float = 0.05
    relaxation_max: float = 0.50
    current_tolerance: float = 1.0e-2
    field_tolerance: float = 1.0e-2
    charge_tolerance: float = 1.0e-2
    z_probe_m: Optional[float] = None  # derived from z_max_m/4 when None
    z_max_m: float = 2.0e-3
    include_space_charge: bool = True
    include_mirror: bool = True
    include_beam_loading: bool = False
    #: Causal TM010 modal-envelope beam-loading calibration (rf_gun.beam_loading_envelope),
    #: required (all must be > 0, Veff nonzero) whenever include_beam_loading=True -- mirrors
    #: VolumeBuildParams' own bl_Q_loaded/bl_r_over_q_ohm_per_m naming and validation style
    #: (rftrack_volume._attach_beam_loading_sw) so the same calibration numbers used for the real
    #: production BeamLoadingSW can be passed here unchanged; bl_L_eff_m converts the per-metre
    #: (R/Q) into the lumped Ohm value the envelope ODE uses (r_over_q_ohm = bl_r_over_q_ohm_per_m
    #: * bl_L_eff_m). bl_Veff_V is the same effective on-axis voltage magnitude [V] used in
    #: rf_params.r_over_q_per_m's own (R/Q) derivation, so chi(t)=V_b(t)/bl_Veff_V is on the same
    #: convention. See rf_gun.beam_loading_envelope module docstring for the physics and
    #: UPGRADE_PLAN.md Sec. 6e/6f for what has (steady-state/decay closed-form limits) and has not
    #: (cross-check against real RF-Track BeamLoadingSW; multi-bucket periodic steady state) been
    #: validated yet.
    bl_Q_loaded: float = 0.0
    bl_r_over_q_ohm_per_m: float = 0.0
    bl_L_eff_m: float = 0.0
    bl_Veff_V: float = 0.0
    bl_detuning_rad_s: float = 0.0
    #: "pic_probe" (default, preserves every previously-validated result): zero-weight probe
    #: particles into the real SpaceCharge_PIC_FreeSpace engine at z=z_probe_m (see
    #: cathode_fields.extract_sc_and_mirror_from_snapshot). "analytic_point_charge_image": the
    #: closed-form conductor-image kernel (cathode_fields.analytic_sc_and_mirror_surface_field) --
    #: no PIC mesh, no z_probe_m distance to choose (evaluated exactly at the true cathode surface
    #: z=0), and naturally excludes a time bin's own not-yet-drifted charge.
    #:
    #: **Cross-checked against pic_probe on a real RF-Track run (this was not possible until this
    #: session's environment repair) -- the result is a genuinely important, not just a
    #: reassuring, finding.** Neither method is actually mesh/scale-converged for the *peak*
    #: near-cathode field: pic_probe's own peak |E_SC| grows monotonically and by >10x as
    #: sc_nx/ny/nz refines from 16 to 64 (3.4e6 -> 8.1e6 -> 22.7e6 -> 42.7e6 -> 61.8e6 V/m in one
    #: tested configuration), and stops converging (residuals no longer settle) at the finer
    #: meshes -- a real PIC "self-field" sensitivity as a macroparticle's own recent deposit gets
    #: resolved by an ever-finer mesh, not a bug specific to analytic_point_charge_image's own
    #: point-charge idealization (which shows the same qualitative sensitivity, parameterized by
    #: `analytic_softening_scale` below instead of mesh size). This matches
    #: test_mirror_mesh_convergence.py's own pre-existing, deliberately loose convergence
    #: criterion ("bounded (<4x) spread ... not strict monotonic convergence") -- this session's
    #: finding is a sharper quantification of an issue that test was already guarding against, not
    #: a new one. **Conclusion: treat the *peak* near-cathode field from either method as
    #: order-of-magnitude, not precision, information** until a properly self-force-free emission
    #: model (e.g. segmented real-trajectory tracking, or the sheet-based analytic surface-charge
    #: approach in UPGRADE_PLAN.md Sec. 4) replaces this fixed-sample/ballistic-z approximation.
    #: Kept opt-in (not the default) given this.
    field_probe_method: str = "pic_probe"
    #: Multiplier on each source macroparticle's natural disk-equivalent radius (sqrt(its own
    #: cathode grid cell area / pi)) used only by analytic_point_charge_image's same-cell term --
    #: see analytic_sc_and_mirror_surface_field's docstring: that term is now the *exact* on-axis
    #: finite-disk field (not a generic softening), with this radius as the disk's actual radius,
    #: so 1.0 (default) is the physically motivated value, not a numerical-convenience choice.
    #: Different-cell terms are unaffected by this (they use the plain point-charge kernel, no
    #: regularization needed). Scan this (e.g. 0.5, 2.0) the same way sc_nx/ny/nz is scanned for
    #: pic_probe, to characterize how sensitive the *same-cell* term's own cell-size choice is --
    #: see test_analytic_surface_field_diverges_unsoftened_but_bounded_when_softened in
    #: tests/test_cathode_fields.py.
    analytic_softening_scale: float = 1.0
    sc_nx: int = 32
    sc_ny: int = 32
    sc_nz: int = 32
    mirror_z_m: float = 0.0
    n_x_bins: int = 12
    n_y_bins: int = 12
    n_time_bins: int = 24
    min_consecutive_converged: int = 2
    random_seed: int = 42
    #: Uniform T_K (float, backward-compatible default) or a callable T_K(x_mm, y_mm) for a
    #: spatially resolved cathode temperature profile (see module docstring).
    cathode_temperature_K: Optional[TemperatureField] = None


@dataclass
class EmissionFieldIterationResult:
    x_grid_m: np.ndarray
    y_grid_m: np.ndarray
    t_grid_s: np.ndarray
    cathode_radius_mm: float
    temperature_K: np.ndarray
    J_history_Apm2: List[np.ndarray]
    E_RF_history_Vpm: List[np.ndarray]
    E_SC_history_Vpm: List[np.ndarray]
    E_mirror_history_Vpm: List[np.ndarray]
    E_BL_history_Vpm: List[np.ndarray]
    E_total_history_Vpm: List[np.ndarray]
    Q_history_C: List[float]
    I_peak_history_A: List[float]
    eps_J_history: List[float]
    eps_E_history: List[float]
    eps_E_inf_history: List[float]
    eps_Q_history: List[float]
    relaxation_history: List[float]
    runtime_history_s: List[float]
    converged: bool
    failure_reason: Optional[str]
    fixed_sample: Dict[str, np.ndarray]
    source_bunch_matrix: Optional[np.ndarray]
    capability_report: Dict[str, Any]
    config: EmissionFieldIterationConfig


def _cathode_xy_grid(cathode_radius_mm: float, n_x: int, n_y: int) -> Dict[str, np.ndarray]:
    """Regular Cartesian grid over the cathode's bounding square, with a boolean mask for cells
    inside the disk (guide Sec. 8.5's stratification, generalized from radial annuli to a genuine
    2D grid -- see module docstring)."""
    x_edges = np.linspace(-cathode_radius_mm, cathode_radius_mm, n_x + 1)
    y_edges = np.linspace(-cathode_radius_mm, cathode_radius_mm, n_y + 1)
    x_centers_mm = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers_mm = 0.5 * (y_edges[:-1] + y_edges[1:])
    dx_mm = x_edges[1] - x_edges[0]
    dy_mm = y_edges[1] - y_edges[0]
    X, Y = np.meshgrid(x_centers_mm, y_centers_mm, indexing="ij")
    inside_disk = (X ** 2 + Y ** 2) <= cathode_radius_mm ** 2
    dA_mm2 = np.where(inside_disk, dx_mm * dy_mm, 0.0)
    return {"X_mm": X, "Y_mm": Y, "dA_mm2": dA_mm2, "inside_disk": inside_disk,
            "x_centers_mm": x_centers_mm, "y_centers_mm": y_centers_mm}


def _resolve_temperature_K(temperature_field: Optional[TemperatureField], x_mm: np.ndarray, y_mm: np.ndarray, default_T_K: float) -> np.ndarray:
    """Uniform default_T_K (matches every prior version of this code) unless a spatial profile is
    given, in which case it is evaluated at each (x_mm, y_mm) sample position."""
    if temperature_field is None:
        return np.full(x_mm.shape, float(default_T_K))
    if callable(temperature_field):
        return np.asarray(temperature_field(x_mm, y_mm), dtype=float)
    return np.full(x_mm.shape, float(temperature_field))


def _build_fixed_sample(
    n_x: int,
    n_y: int,
    n_t: int,
    cathode_radius_mm: float,
    tau_s: float,
    temperature_field: Optional[TemperatureField],
    default_T_K: float,
    pz0_MeV_c: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Stratified (x,y,t) sample, one macroparticle per populated cell (guide Sec. 8.5), with each
    particle's own local temperature resolved from `temperature_field` at its own (x,y)."""
    grid = _cathode_xy_grid(cathode_radius_mm, n_x, n_y)
    edges_t = np.linspace(0.0, tau_s, n_t + 1)
    t_centers_s = 0.5 * (edges_t[:-1] + edges_t[1:])
    dt_s = edges_t[1] - edges_t[0] if n_t > 0 else 0.0

    X3, T3 = np.meshgrid(grid["X_mm"].ravel(), t_centers_s, indexing="ij")
    Y3, _ = np.meshgrid(grid["Y_mm"].ravel(), t_centers_s, indexing="ij")
    DA3, _ = np.meshgrid(grid["dA_mm2"].ravel(), t_centers_s, indexing="ij")
    IX3, IT3 = np.meshgrid(np.arange(n_x * n_y), np.arange(n_t), indexing="ij")

    x_flat_mm = X3.ravel()
    y_flat_mm = Y3.ravel()
    t_flat_s = T3.ravel()
    dA_flat_mm2 = DA3.ravel()
    xy_bin_idx = IX3.ravel()
    t_bin_idx = IT3.ravel()
    n = x_flat_mm.size

    T_K_flat = _resolve_temperature_K(temperature_field, x_flat_mm, y_flat_mm, default_T_K)
    px, py, pz, _, _ = sample_thermionic_momenta(n, T_K_flat, pz0_MeV_c, pz_model="flux", rng=rng)

    return {
        "grid": grid,
        "n_x": n_x, "n_y": n_y, "n_t": n_t,
        "t_centers_s": t_centers_s, "dt_s": dt_s,
        "x_mm": x_flat_mm, "y_mm": y_flat_mm, "t_flat_s": t_flat_s,
        "dA_flat_mm2": dA_flat_mm2, "T_K_flat": T_K_flat,
        "xy_bin_idx": xy_bin_idx, "t_bin_idx": t_bin_idx,
        "px_MeV_c": px, "py_MeV_c": py, "pz_MeV_c": pz,
    }


def _bunch_matrix_from_weights(sample: Dict[str, np.ndarray], N_i: np.ndarray, mass_MeV: float = ME_MEV) -> np.ndarray:
    n = sample["x_mm"].size
    t0_mm_c = sample["t_flat_s"] * c * 1.0e3
    return np.column_stack([
        sample["x_mm"], sample["px_MeV_c"], sample["y_mm"], sample["py_MeV_c"],
        np.zeros(n), sample["pz_MeV_c"],
        np.full(n, mass_MeV), np.full(n, -1.0), N_i, t0_mm_c,
    ])


#: Electron mass, kg (SI) -- used only for the ballistic z-drift estimate below.
_ME_KG = 9.1093837015e-31


def _ballistic_z_drift_mm(F_local_creation_Vpm: np.ndarray, dt_since_creation_s: np.ndarray, z_max_m: float) -> np.ndarray:
    """Rough kinematic estimate z(dt) = 0.5*(e*F/m_e)*dt^2 of how far an already-emitted electron
    has drifted from the cathode by `dt_since_creation_s` after its own creation, given the
    extraction field magnitude at its creation bin (constant-acceleration-from-rest estimate,
    reasonable this close to the cathode where thermal launch momentum is small relative to the
    RF/SC acceleration).

    Placing every "already emitted" particle exactly at z=0 would put its mirror image at the
    identical point (opposite sign, same location), so source and image would exactly cancel for
    any external probe -- E_SC and E_mirror would then always sum to exactly zero regardless of
    the true charge state, defeating the self-consistency this iteration is meant to capture. This
    estimate keeps each particle at a small but nonzero z once it has had time to drift. Clipped to
    z_max_m (the configured near-cathode region) since the quasi-static approximation is not meant
    to extrapolate particles out of it.
    """
    F = np.maximum(np.asarray(F_local_creation_Vpm, dtype=float), 0.0)
    dt = np.maximum(np.asarray(dt_since_creation_s, dtype=float), 0.0)
    accel = (q_e * F) / _ME_KG  # m/s^2
    z_m = 0.5 * accel * dt ** 2
    return np.minimum(z_m, z_max_m) * 1.0e3  # mm


def _quasi_static_sc_mirror_field(
    rft,
    sample: Dict[str, np.ndarray],
    N_i: np.ndarray,
    F_local_Vpm_grid: np.ndarray,
    x_grid_m: np.ndarray,
    y_grid_m: np.ndarray,
    cfg: EmissionFieldIterationConfig,
    z_probe_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """E_SC(x,y,t), E_mirror(x,y,t) via the quasi-static approximation documented in the module
    docstring, with each already-emitted particle placed at a ballistically-estimated z (not
    exactly z=0 -- see _ballistic_z_drift_mm) so its mirror image does not degenerately coincide
    with it. Returns arrays of shape (n_points, n_t), where n_points = len(x_grid_m).

    `cfg.field_probe_method` selects how the field at (x_grid_m, y_grid_m) is evaluated from the
    already-emitted charge at each time bin -- see EmissionFieldIterationConfig.field_probe_method
    for the two options and their tradeoffs.
    """
    n_t = sample["n_t"]
    t_centers_s = sample["t_centers_s"]
    n_points = x_grid_m.size
    E_sc = np.zeros((n_points, n_t))
    E_mirror = np.zeros((n_points, n_t))

    if not cfg.include_space_charge:
        return E_sc, E_mirror

    method = cfg.field_probe_method
    if method not in ("pic_probe", "analytic_point_charge_image"):
        raise ValueError(
            f"Unknown field_probe_method {method!r}; expected 'pic_probe' or "
            "'analytic_point_charge_image'."
        )

    mass_MeV = ME_MEV
    F_at_creation = F_local_Vpm_grid[sample["xy_bin_idx"], sample["t_bin_idx"]]
    for j, t_j in enumerate(t_centers_s):
        active = sample["t_flat_s"] <= t_j
        if not np.any(active) or not np.any(N_i[active] > 0.0):
            continue
        n_active = int(np.count_nonzero(active))
        dt_since_creation = t_j - sample["t_flat_s"][active]
        z_active_mm = _ballistic_z_drift_mm(F_at_creation[active], dt_since_creation, cfg.z_max_m)

        if method == "pic_probe":
            M_active = np.column_stack([
                sample["x_mm"][active], np.zeros(n_active), sample["y_mm"][active], np.zeros(n_active),
                z_active_mm, np.zeros(n_active),
                np.full(n_active, mass_MeV), np.full(n_active, -1.0), N_i[active], np.zeros(n_active),
            ])
            mirror_z = cfg.mirror_z_m if cfg.include_mirror else None
            E_free, E_plus_mirror = extract_sc_and_mirror_from_snapshot(
                rft, M_active, x_grid_m, y_grid_m, z_probe_m,
                sc_nx=cfg.sc_nx, sc_ny=cfg.sc_ny, sc_nz=cfg.sc_nz,
                mirror_z_m=mirror_z,
            )
        else:
            # Q=-1 (electron) * N_i (real-electron count) * q_e -- same charge convention as the
            # M_active bunch matrix above (column 8=Q=-1, column 9=N), just evaluated directly in
            # Coulombs since this path calls no RF-Track bunch/engine at all.
            source_q_C = -q_e * N_i[active]
            # Each source's own cathode grid cell area, as an effective disk radius -- see
            # analytic_sc_and_mirror_surface_field's docstring on why softening isn't optional in
            # practice: without it, a source's own cell re-evaluated at a small drift height
            # diverges as a bare point charge, which a finite PIC mesh never does.
            source_area_m2 = sample["dA_flat_mm2"][active] * 1.0e-6
            source_softening_m = float(cfg.analytic_softening_scale) * np.sqrt(source_area_m2 / np.pi)
            E_free, E_plus_mirror = analytic_sc_and_mirror_surface_field(
                x_grid_m, y_grid_m,
                sample["x_mm"][active] * 1.0e-3, sample["y_mm"][active] * 1.0e-3, z_active_mm * 1.0e-3,
                source_q_C, source_softening_m=source_softening_m,
            )
            if not cfg.include_mirror:
                E_plus_mirror = E_free

        E_sc[:, j] = E_free
        E_mirror[:, j] = E_plus_mirror - E_free
    return E_sc, E_mirror


def run_emission_field_iteration(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    rf_axis_phasor: complex,
    volume_params,
    emission_params,
    config: EmissionFieldIterationConfig,
    phi_deg: float = 0.0,
    on_iteration: Optional[Callable[[int, "EmissionFieldIterationResult"], None]] = None,
) -> EmissionFieldIterationResult:
    """Under-relaxed Picard iteration coupling emission current to E_SC+E_mirror(+E_BL) at the
    cathode (guide Sec. 7-8). Stage A (iteration 0: RF-only baseline, no SC/mirror/BL update yet)
    and Stage B (iteration 1: first update) fall out naturally as the first two passes of the same
    loop that implements Stage C (fixed-point iteration to convergence).

    Stage D (beam loading): when `config.include_beam_loading=True`, each iteration's own just-
    updated cathode current density is fed through
    `rf_gun.beam_loading_envelope.estimate_beam_induced_cathode_field_from_current_density` (the
    causal TM010 modal-envelope model, `E_BL(x,y,t)=-chi(t)*E_RF(x,y,t)`) to produce an `E_BL`
    term that is added into `E_total` alongside `E_SC`/`E_mirror`, so it feeds back into the *next*
    iteration's emission-law evaluation exactly like the other two terms -- a genuinely self-
    consistent SC+mirror+BL case, not merely a post-hoc estimate over an already-converged BL-off
    result (contrast `rf_gun.beam_loading_envelope.estimate_beam_induced_cathode_field_map`, which
    remains for that post-hoc case). This is *not* a reproduction of RF-Track's own internal
    `BeamLoadingSW` discretization (that binding exposes no mode-amplitude/induced-voltage/
    arbitrary-time query -- see `rf_gun.cathode_fields.extract_beam_loading_field` -- so there is
    no native way to turn its state into a cathode-surface E_BL(r,t) term); it is an independent,
    from-first-principles implementation of the same textbook physics (P. B. Wilson, SLAC-PUB-2884;
    Wangler Sec. 4.7), not yet cross-checked against real RF-Track BeamLoadingSW production
    behavior on a real run (see UPGRADE_PLAN.md Sec. 6e/6f) -- treat E_BL as an order-of-magnitude
    estimate until that comparison is done.
    """
    import time as _time

    if config.include_beam_loading:
        if config.bl_Q_loaded <= 0.0 or config.bl_r_over_q_ohm_per_m <= 0.0 or config.bl_L_eff_m <= 0.0 or config.bl_Veff_V == 0.0:
            raise ValueError(
                "EmissionFieldIterationConfig.include_beam_loading=True requires "
                "bl_Q_loaded>0, bl_r_over_q_ohm_per_m>0, bl_L_eff_m>0, and bl_Veff_V!=0 "
                f"(got bl_Q_loaded={config.bl_Q_loaded}, "
                f"bl_r_over_q_ohm_per_m={config.bl_r_over_q_ohm_per_m}, "
                f"bl_L_eff_m={config.bl_L_eff_m}, bl_Veff_V={config.bl_Veff_V}) -- pass the same "
                "calibration values used for the production BeamLoadingSW attach "
                "(VolumeBuildParams.bl_Q_loaded/bl_r_over_q_ohm_per_m) plus the cavity's effective "
                "length and Veff (see rf_params.r_over_q_per_m)."
            )

    rng = np.random.default_rng(config.random_seed)
    capability_report = inspect_rftrack_capabilities(rft)

    cathode_radius_mm = float(emission_params.cathode_radius_mm)
    default_T_K = float(emission_params.cathode_T_K)
    # Resolve phi_eff(T) as rf_gun.simulation._resolve_work_function does for every other bunch/
    # source builder -- otherwise a work_function_temperature_model would be silently ignored here
    # while the rest of the pipeline uses the resolved value.
    work_function_temperature_model = getattr(emission_params, "work_function_temperature_model", None)
    if work_function_temperature_model is not None:
        work_function_model_params = getattr(emission_params, "work_function_model_params", None) or {}
        phi_eV = float(evaluate_work_function_eV(
            work_function_temperature_model, default_T_K, **work_function_model_params
        ))
    else:
        phi_eV = float(emission_params.work_function_eV)
    beta_enh = float(emission_params.beta_enh) if emission_params.beta_enh is not None else float(emission_params.beta_field)
    emission_law = str(getattr(emission_params, "emission_law", "RDSchottky"))
    f_hz = float(volume_params.f_hz)
    T_rf = 1.0 / f_hz
    tau_s = (float(emission_params.emission_phase_range_deg) / 360.0) * T_rf

    z_probe_m = float(config.z_probe_m) if config.z_probe_m is not None else float(config.z_max_m) / 4.0

    sample = _build_fixed_sample(
        config.n_x_bins, config.n_y_bins, config.n_time_bins, cathode_radius_mm, tau_s,
        config.cathode_temperature_K, default_T_K, float(emission_params.pz0_MeV_c), rng,
    )
    grid = sample["grid"]
    n_x, n_y, n_t = sample["n_x"], sample["n_y"], sample["n_t"]
    x_grid_m = grid["X_mm"].ravel() * 1.0e-3
    y_grid_m = grid["Y_mm"].ravel() * 1.0e-3
    t_grid_s = sample["t_centers_s"]
    area_m2 = grid["dA_mm2"].ravel() * 1.0e-6  # per-cell area, shape (n_x*n_y,)
    T_K_grid = _resolve_temperature_K(
        config.cathode_temperature_K, grid["X_mm"].ravel(), grid["Y_mm"].ravel(), default_T_K
    )  # per-cell temperature, shape (n_x*n_y,) -- constant over t within a cell by construction

    # E_RF(x,y,t): sampled from the *same configured RF-Track field-map element* used for tracking
    # (guide Sec. 6.4), not re-derived from the on-axis phasor alone -- a small, dedicated
    # near-cathode Volume built from the same Er/Ez grids and phase carries the true spatial
    # dependence get_field() exposes.
    from .rftrack_volume import build_volume

    near_cathode_params = volume_params.replace(
        z_min_m=0.0, z_max_m=max(float(config.z_max_m), 5.0 * z_probe_m),
        sc_enabled=False, beam_loading_enabled=False,
    )
    V_field = build_volume(rft, Er_grid, Ez_grid, float(phi_deg), near_cathode_params)
    E_RF = sample_rf_field_on_cathode(rft, V_field, x_grid_m, y_grid_m, t_grid_s, z_probe_m)

    J = np.zeros((n_x * n_y, n_t))
    N_i = np.zeros(sample["x_mm"].size)
    E_sc = np.zeros_like(E_RF)
    E_mirror = np.zeros_like(E_RF)
    E_bl = np.zeros_like(E_RF)
    bl_r_over_q_ohm = float(config.bl_r_over_q_ohm_per_m) * float(config.bl_L_eff_m)

    J_hist, E_rf_hist, E_sc_hist, E_mirror_hist, E_bl_hist, E_tot_hist = [], [], [], [], [], []
    Q_hist, Ipk_hist, epsJ_hist, epsE_hist, epsEinf_hist, epsQ_hist, omega_hist, runtime_hist = (
        [], [], [], [], [], [], [], []
    )

    omega_relax = float(config.relaxation)
    converged_streak = 0
    converged = False
    failure_reason = None
    final_matrix = None

    for k in range(int(config.max_iterations)):
        t_start = _time.time()

        E_total = E_RF + E_sc + E_mirror + E_bl
        F_local = extraction_field(E_total, beta_enh=beta_enh)
        # Flattened, not (n_x*n_y, n_t) directly: the loop-based reference models (rgtf_2019,
        # murphy_good_direct_reference) iterate their field argument element-by-element and
        # expect a matching 1D temperature array, not a 2D one.
        T_broadcast = np.broadcast_to(T_K_grid[:, None], F_local.shape)
        result = evaluate_emission_model(emission_law, F_local.ravel(), T_broadcast.ravel(), phi_eV)
        J_star = np.asarray(result.J_Apm2).reshape(n_x * n_y, n_t)

        if k == 0:
            J_new = J_star  # Stage A: baseline (RF-only field) source, no relaxed update yet
        else:
            J_new = (1.0 - omega_relax) * J + omega_relax * J_star

        N_i_new = (J_new * area_m2[:, None] * sample["dt_s"]).ravel() / q_e

        Q_new = float(np.sum(N_i_new) * q_e)
        I_peak_new = float(np.max(np.sum(J_new * area_m2[:, None], axis=0))) if J_new.size else 0.0

        weights_area_time = (area_m2[:, None] * sample["dt_s"]) * np.ones_like(J_new)
        num_J = float(np.sum(weights_area_time * np.abs(J_new - J)))
        den_J = float(np.sum(weights_area_time * np.abs(J))) + 1.0e-300
        eps_J = num_J / den_J if k > 0 else np.nan

        E_sc_new, E_mirror_new = _quasi_static_sc_mirror_field(
            rft, sample, N_i_new, F_local, x_grid_m, y_grid_m, config, z_probe_m
        )
        if config.include_beam_loading:
            E_bl_new = estimate_beam_induced_cathode_field_from_current_density(
                J_new, area_m2, t_grid_s, E_RF, f_hz,
                float(config.bl_Q_loaded), bl_r_over_q_ohm, float(config.bl_Veff_V),
                detuning_rad_s=float(config.bl_detuning_rad_s),
            )
        else:
            E_bl_new = np.zeros_like(E_RF)
        E_total_new = E_RF + E_sc_new + E_mirror_new + E_bl_new

        eps_E = float(np.linalg.norm((E_total_new - E_total).ravel()) / (np.linalg.norm(E_total.ravel()) + 1e-300)) if k > 0 else np.nan
        eps_E_inf = float(np.max(np.abs(E_total_new - E_total))) if k > 0 else np.nan
        eps_Q = float(abs(Q_new - Q_hist[-1]) / (abs(Q_hist[-1]) + 1e-300)) if k > 0 else np.nan

        runtime_hist.append(_time.time() - t_start)
        J_hist.append(J_new.reshape(n_x, n_y, n_t).copy())
        E_rf_hist.append(E_RF.reshape(n_x, n_y, n_t).copy())
        E_sc_hist.append(E_sc_new.reshape(n_x, n_y, n_t).copy())
        E_mirror_hist.append(E_mirror_new.reshape(n_x, n_y, n_t).copy())
        E_bl_hist.append(E_bl_new.reshape(n_x, n_y, n_t).copy())
        E_tot_hist.append(E_total_new.reshape(n_x, n_y, n_t).copy())
        Q_hist.append(Q_new)
        Ipk_hist.append(I_peak_new)
        epsJ_hist.append(eps_J)
        epsE_hist.append(eps_E)
        epsEinf_hist.append(eps_E_inf)
        epsQ_hist.append(eps_Q)
        omega_hist.append(omega_relax)

        J, N_i = J_new, N_i_new
        E_sc, E_mirror, E_bl = E_sc_new, E_mirror_new, E_bl_new
        final_matrix = _bunch_matrix_from_weights(sample, N_i)

        # A NaN/inf anywhere in this iteration's current/field/charge state invalidates the
        # iteration immediately: preserve this (first failing) iteration's arrays for postmortem
        # inspection, but do not compute further iterations from a corrupted state, and never let
        # relaxation/convergence-streak bookkeeping run on non-finite residuals.
        _finite_checks = (
            ("J_new", J_new), ("Q_new", Q_new), ("I_peak_new", I_peak_new),
            ("E_sc_new", E_sc_new), ("E_mirror_new", E_mirror_new),
            ("E_bl_new", E_bl_new), ("E_total_new", E_total_new),
        )
        _bad = next((name for name, arr in _finite_checks if not np.all(np.isfinite(arr))), None)
        if _bad is not None:
            failure_reason = (
                f"iteration {k}: {_bad} contains non-finite values -- invalidated immediately "
                "rather than continuing from a corrupted current/field state"
            )
            converged = False
            break

        # Adaptive relaxation (guide Sec. 8.3): halve omega on a growing residual; after 3
        # consecutive monotonic decreases, nudge omega up by 20%.
        if k >= 2:
            if epsJ_hist[-1] > epsJ_hist[-2]:
                omega_relax = max(config.relaxation_min, omega_relax * 0.5)
            elif len(epsJ_hist) >= 4 and all(
                epsJ_hist[-i] < epsJ_hist[-i - 1] for i in range(1, 4)
            ):
                omega_relax = min(config.relaxation_max, omega_relax * 1.2)

        if k > 0 and eps_J < config.current_tolerance and eps_E < config.field_tolerance and eps_Q < config.charge_tolerance:
            converged_streak += 1
        else:
            converged_streak = 0
        if converged_streak >= config.min_consecutive_converged:
            converged = True

        if on_iteration is not None:
            partial = EmissionFieldIterationResult(
                x_grid_m=grid["x_centers_mm"] * 1e-3, y_grid_m=grid["y_centers_mm"] * 1e-3,
                t_grid_s=t_grid_s, cathode_radius_mm=cathode_radius_mm,
                temperature_K=T_K_grid.reshape(n_x, n_y),
                J_history_Apm2=J_hist, E_RF_history_Vpm=E_rf_hist, E_SC_history_Vpm=E_sc_hist,
                E_mirror_history_Vpm=E_mirror_hist, E_BL_history_Vpm=E_bl_hist, E_total_history_Vpm=E_tot_hist,
                Q_history_C=Q_hist, I_peak_history_A=Ipk_hist, eps_J_history=epsJ_hist,
                eps_E_history=epsE_hist, eps_E_inf_history=epsEinf_hist, eps_Q_history=epsQ_hist,
                relaxation_history=omega_hist, runtime_history_s=runtime_hist, converged=converged,
                failure_reason=None, fixed_sample=sample, source_bunch_matrix=final_matrix,
                capability_report=capability_report, config=config,
            )
            on_iteration(k, partial)

        if converged:
            break

    if not converged and failure_reason is None:
        failure_reason = f"did not reach {config.min_consecutive_converged} consecutive converged iterations within max_iterations={config.max_iterations}"

    return EmissionFieldIterationResult(
        x_grid_m=grid["x_centers_mm"] * 1e-3, y_grid_m=grid["y_centers_mm"] * 1e-3,
        t_grid_s=t_grid_s, cathode_radius_mm=cathode_radius_mm,
        temperature_K=T_K_grid.reshape(n_x, n_y),
        J_history_Apm2=J_hist, E_RF_history_Vpm=E_rf_hist, E_SC_history_Vpm=E_sc_hist,
        E_mirror_history_Vpm=E_mirror_hist, E_BL_history_Vpm=E_bl_hist, E_total_history_Vpm=E_tot_hist,
        Q_history_C=Q_hist, I_peak_history_A=Ipk_hist, eps_J_history=epsJ_hist,
        eps_E_history=epsE_hist, eps_E_inf_history=epsEinf_hist, eps_Q_history=epsQ_hist,
        relaxation_history=omega_hist, runtime_history_s=runtime_hist, converged=converged,
        failure_reason=failure_reason, fixed_sample=sample, source_bunch_matrix=final_matrix,
        capability_report=capability_report, config=config,
    )


def spatial_source_from_iteration_result(result: EmissionFieldIterationResult) -> Dict[str, np.ndarray]:
    """Build the `prescribed_source`/`spatial_source` dict that
    `rf_gun.simulation.build_bunch_thermionic_spatial` (via `run_transport_with_progress`'s
    `spatial_source` argument) expects from a converged `EmissionFieldIterationResult` -- the one
    place this mapping is defined, so the notebook and `run_thermionic_tm010.py` build it
    identically rather than each hand-rolling their own copy of the dict.

    Besides the emission source itself (`J_Apm2`/`temperature_K`), this also carries disk-averaged
    external (RF-only) and corrected (RF+SC+mirror) signed cathode-field waveforms
    (`Ez_ext_t`/`Ez_corrected_t`, V/m) purely so `rf_gun.plotting.plot_emission_history`'s top panel
    has something to show for a self-consistent run -- `build_bunch_thermionic_spatial` otherwise
    has no field history at all to hand it (unlike `build_bunch_thermionic`'s on-axis path, which
    always populates `Ez_t`/`F_t`). The average is a plain (unweighted) mean over the disk's cells
    at the *final* iteration -- a single representative curve, not a claim that the field is
    uniform across the cathode (see `plot_emission_iteration_near_cathode` for the actual spatial
    maps).
    """
    inside_disk_flat = result.fixed_sample["grid"]["inside_disk"].reshape(-1)
    n_t = result.t_grid_s.size
    E_RF_flat = result.E_RF_history_Vpm[-1].reshape(-1, n_t)
    E_total_flat = result.E_total_history_Vpm[-1].reshape(-1, n_t)
    if np.any(inside_disk_flat):
        Ez_ext_t = np.mean(E_RF_flat[inside_disk_flat], axis=0)
        Ez_corrected_t = np.mean(E_total_flat[inside_disk_flat], axis=0)
    else:
        Ez_ext_t = np.full(n_t, np.nan)
        Ez_corrected_t = np.full(n_t, np.nan)

    return {
        "x_grid_m": result.x_grid_m,
        "y_grid_m": result.y_grid_m,
        "t_grid_s": result.t_grid_s,
        "J_Apm2": result.J_history_Apm2[-1],
        "temperature_K": result.temperature_K,
        "Ez_ext_t": Ez_ext_t,
        "Ez_corrected_t": Ez_corrected_t,
    }
