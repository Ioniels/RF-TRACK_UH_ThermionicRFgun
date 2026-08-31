"""Simulation pipeline helpers."""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace as dataclass_replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Literal
import time
import os
import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .constants import ME_MEV, c, q_e
from .helpers import sample_disk
from .emission_models import (
    delta_phi_schottky_eV,
    evaluate_emission_model,
)
from .work_function_models import evaluate_work_function_eV
from .cathode_fields import sample_rf_field_on_cathode, extraction_field
from .emission_sampling import apply_roughness, sample_thermionic_momenta
from .emission_iteration import TemperatureField
from .rf_params import PhaseCalibrationResult, build_phase_calibration_result
from .rftrack_volume import (
    build_volume,
    track_volume_with_screens,
    VolumeBuildParams,
    ScreenBuildParams,
)
from .diagnostics import (
    build_screen_summary_from_phase_space,
    info_get_first,
    classify_particle_outcomes,
    to_lost_table_array,
)


@dataclass(frozen=True)
class RoughnessParams:
    Ra_um: float = 0.0
    Re_um: float = 0.0


@dataclass(frozen=True)
class EmissionParams:
    cathode_radius_mm: float
    cathode_T_K: float
    work_function_eV: float
    beta_field: float
    emission_phase_range_deg: float
    pz0_MeV_c: float
    pz_model: Literal["constant", "flux"] = "flux"
    emission_law: str = "RDSchottky"
    beta_enh: float = 1.0
    roughness: RoughnessParams = RoughnessParams()
    time_dependent: bool = True
    #: Optional phi_eff(T) model (rf_gun.work_function_models.WORK_FUNCTION_MODEL_NAMES). None
    #: (default) keeps work_function_eV fixed at the value above. When set, it *overrides*
    #: work_function_eV at the cathode temperature cathode_T_K (resolved once in
    #: build_bunch_thermionic(), guide Sec. 2.3).
    work_function_temperature_model: Optional[str] = None
    work_function_model_params: Dict[str, Any] = dataclass_field(default_factory=dict)


#: The project's core 6-column phase-space convention (X, Px, Y, Py, Z, Pz), all in RF-Track's
#: native Bunch6dT units (mm, MeV/c). Extended with %id (particle id, for cross-referencing a
#: row's identity across B0/screens/Bout -- a Screen's own Pz does not reliably carry the true
#: lab-frame sign, so id-based lookups against Bout's classification are the reliable way to tag
#: forward/backward status at a screen), %t (arrival time, mm/c), %E (total energy, MeV), and %K
#: (kinetic energy, MeV) -- all confirmed valid RF-Track format codes. Every consumer that needs
#: exactly the core 6 columns already slices `[:, :6]` explicitly, so this extension is additive.
EXTENDED_PHASE_FMT = "%X %Px %Y %Py %Z %Pz %id %t %E %K"

#: `thermo_info` entries that are per-emission-time-sample arrays (one value per `t_s` sample,
#: typically hundreds), not per-run scalars -- must be excluded from any run summary/config JSON.
THERMO_INFO_TIME_ARRAY_KEYS = frozenset({
    "t_s", "Ez_t", "Ez_corrected_t", "F_t", "dphi_eV_t", "phi_eff_eV_t", "J_Apm2_t",
    "J_th_Apm2_t", "J_fe_Apm2_t", "R_t", "n_t", "I_A_t", "Q_cum_C", "t_emit_s",
})
#: `thermo_info` entries that are per-particle arrays (one value per macroparticle, set at the end
#: of `build_bunch_thermionic`) -- also excluded from any run summary/config JSON, since dumping
#: the full initial phase-space matrix as JSON text would inflate run_summary.json to tens of MB.
THERMO_INFO_PER_PARTICLE_KEYS = frozenset({
    "initial_phase_space", "initial_pz_MeV_c", "initial_t0_mm_c",
})
#: `thermo_info` entries set only by `build_bunch_thermionic_spatial` (the spatial-sampling path,
#: used whenever a converged Emission Fields Iteration source is supplied) -- full (n_x,n_y,n_t)
#: grids, not per-run scalars, so likewise excluded from any run summary/config JSON. Added
#: alongside THERMO_INFO_TIME_ARRAY_KEYS/THERMO_INFO_PER_PARTICLE_KEYS above rather than folded
#: into either: these are neither a single per-time-sample 1D array nor a per-particle array, but
#: a distinct (x,y,t)-resolved shape.
THERMO_INFO_SPATIAL_GRID_KEYS = frozenset({
    "x_grid_m", "y_grid_m", "t_grid_s", "J_xyt_Apm2", "temperature_field_K",
})


def thermo_info_summary(thermo_info: Dict[str, Any]) -> Dict[str, Any]:
    """`thermo_info` with every per-time-sample and per-particle array stripped out.

    The one place both `run_thermionic_tm010.py` and the notebook build their JSON-safe
    thermo_info payload from, so the exclusion list stays in sync between the two.
    """
    exclude = THERMO_INFO_TIME_ARRAY_KEYS | THERMO_INFO_PER_PARTICLE_KEYS | THERMO_INFO_SPATIAL_GRID_KEYS
    return {k: v for k, v in dict(thermo_info).items() if k not in exclude}


@dataclass(frozen=True)
class TrackingParams:
    phi_deg: float
    n_particles: int
    z_screens_m: Optional[Sequence[float]] = None
    phase_fmt: str = EXTENDED_PHASE_FMT
    screen_width_mm: float | None = None
    screen_height_mm: float | None = None
    screen_time_window_mm_c: float | None = None
    screen_t0_mode: Literal["unset", "sync_to_first_crossing", "manual"] = "unset"
    screen_t0_manual_mm_c: float = 0.0
    screen_log: bool = False


@dataclass(frozen=True)
class DiagnosticsParams:
    store_screen_phase_space: bool = False
    store_screen_info: bool = True
    screen_stride: int = 1
    screen_indices: Optional[Sequence[int]] = None
    max_screen_particles: Optional[int] = None
    subsample_screens_random: bool = True
    save_lost_particles: bool = True


@dataclass
class SimulationResult:
    B0: Any
    Bout: Any
    thermo_info: Dict[str, Any]
    M_snaps: List[np.ndarray]
    z_snaps: List[float]
    I_snaps: List[Any]
    screen_summaries: List[Dict[str, float]]
    lost_table: Optional[np.ndarray] = None
    particle_classes: Optional[Dict[str, Any]] = None


_RUNTIME_HISTORY_CACHE = Path(
    os.environ.get(
        "RF_GUN_RUNTIME_CACHE",
        str(Path(__file__).resolve().parents[1] / "outputs" / ".cache" / "rf_gun_transport_runtime_history.json"),
    )
)


def _load_transport_runtime_history() -> Dict[str, float]:
    if not _RUNTIME_HISTORY_CACHE.exists():
        return {}
    try:
        with _RUNTIME_HISTORY_CACHE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        hist = payload.get("history", {}) if isinstance(payload, dict) else {}
        out: Dict[str, float] = {}
        if isinstance(hist, dict):
            for key, val in hist.items():
                try:
                    out[str(key)] = float(val)
                except Exception:
                    continue
        return out
    except Exception:
        return {}


def _save_transport_runtime_history(history: Dict[str, float]) -> None:
    try:
        _RUNTIME_HISTORY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "history": history}
        with _RUNTIME_HISTORY_CACHE.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception:
        pass


_TRANSPORT_RUNTIME_HISTORY: Dict[str, float] = _load_transport_runtime_history()


def _runtime_key_payload(vol_params_eff: VolumeBuildParams, tracking: TrackingParams, n_screens: int) -> Dict[str, Any]:
    """Settings that drive tracking wall-clock cost: particle count, step counts, and which
    solver features are on. Excludes physics values (phase, R/Q) that are re-fit each run and
    would otherwise fragment the cache without changing the actual cost."""
    return {
        "n_particles": int(tracking.n_particles),
        "dt_mm": float(getattr(vol_params_eff, "dt_mm", np.nan)),
        "sc_dt_mm": float(getattr(vol_params_eff, "sc_dt_mm", np.nan)),
        "emission_nsteps": int(getattr(vol_params_eff, "emission_nsteps", 0)),
        "emission_range": float(getattr(vol_params_eff, "emission_range", np.nan)),
        "n_screens": int(n_screens),
        "sc_enabled": bool(getattr(vol_params_eff, "sc_enabled", False)),
        "beam_loading_enabled": bool(getattr(vol_params_eff, "beam_loading_enabled", False)),
        "bl_tinj_mode": str(getattr(vol_params_eff, "bl_tinj_mode", "manual")),
        "fm_nsteps": int(getattr(vol_params_eff, "fm_nsteps", 0)),
        "fm_tt_nsteps": int(getattr(vol_params_eff, "fm_tt_nsteps", 0)),
        "ode_algorithm": str(getattr(vol_params_eff, "ode_algorithm", "")),
        "ode_epsabs": float(getattr(vol_params_eff, "ode_epsabs", np.nan)),
        "cfx_dt_mm": float(getattr(vol_params_eff, "cfx_dt_mm", np.nan)),
        # The deflection magnet forces single-threaded tracking (see rftrack_volume.build_volume),
        # which changes wall-clock cost by roughly the machine's core count -- omitting this let a
        # deflection-on run silently reuse a time estimate cached from a deflection-off (multi-
        # threaded) run, understating the real runtime by 10x or more.
        "deflection_enabled": bool(getattr(vol_params_eff, "deflection_enabled", False)),
        # Mesh size and mirror state change per-kick PIC solve cost materially (guide Sec. 15.4:
        # "mirror-on and mirror-off runs, different PIC meshes... must never share the same
        # runtime-cache key") -- omitting these let a mirror-on or finer-mesh run silently reuse a
        # cheaper run's cached estimate.
        "sc_nx": int(getattr(vol_params_eff, "sc_nx", 0)),
        "sc_ny": int(getattr(vol_params_eff, "sc_ny", 0)),
        "sc_nz": int(getattr(vol_params_eff, "sc_nz", 0)),
        "mirror_charge_enabled": bool(getattr(vol_params_eff, "mirror_charge_enabled", False)),
    }


def _runtime_key_string(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _runtime_key_hash(runtime_key: str) -> str:
    return hashlib.sha1(runtime_key.encode("utf-8")).hexdigest()[:12]


def _build_screen_params(tracking: TrackingParams) -> ScreenBuildParams:
    return ScreenBuildParams(
        width_mm=tracking.screen_width_mm,
        height_mm=tracking.screen_height_mm,
        time_window_mm_c=tracking.screen_time_window_mm_c,
        t0_mode=tracking.screen_t0_mode,
        t0_manual_mm_c=tracking.screen_t0_manual_mm_c,
        log=tracking.screen_log,
    )


def _maybe_screen_params(tracking: TrackingParams) -> Optional[ScreenBuildParams]:
    has_custom = any(
        (
            tracking.screen_width_mm is not None,
            tracking.screen_height_mm is not None,
            tracking.screen_time_window_mm_c is not None,
            str(tracking.screen_t0_mode).strip().lower() != "unset",
            abs(float(tracking.screen_t0_manual_mm_c)) > 0.0,
            bool(tracking.screen_log),
        )
    )
    return _build_screen_params(tracking) if has_custom else None


def _sanitize_screen_positions(
    z_screens_m: Sequence[float],
) -> List[float]:
    if not z_screens_m:
        return []

    z_arr = np.asarray(z_screens_m, dtype=float).reshape(-1)
    z_arr = z_arr[np.isfinite(z_arr)]
    if z_arr.size == 0:
        return []
    return [float(z) for z in z_arr.tolist()]


def _select_screen_indices(n: int, diagnostics: DiagnosticsParams) -> List[int]:
    if n <= 0:
        return []

    if diagnostics.screen_indices is not None:
        out = []
        for idx in diagnostics.screen_indices:
            i = int(idx)
            if 0 <= i < n:
                out.append(i)
        return sorted(set(out))

    stride = max(1, int(diagnostics.screen_stride))
    return list(range(0, n, stride))


def _maybe_subsample_phase_space(
    M: np.ndarray,
    max_particles: Optional[int],
    rng: Optional[np.random.Generator],
    random_subsample: bool,
) -> np.ndarray:
    arr = np.asarray(M)
    if arr.ndim != 2:
        return np.zeros((0, 6), dtype=float)
    if max_particles is None or int(max_particles) <= 0 or arr.shape[0] <= int(max_particles):
        return np.array(arr, copy=True)

    n_keep = int(max_particles)
    if random_subsample:
        rng_eff = np.random.default_rng() if rng is None else rng
        keep = np.sort(rng_eff.choice(arr.shape[0], size=n_keep, replace=False))
    else:
        keep = np.arange(n_keep)
    return np.array(arr[keep], copy=True)


def _extract_lost_particles(V):
    if V is None:
        return None
    get_lost = getattr(V, "get_lost_particles", None)
    if not callable(get_lost):
        return None
    try:
        return to_lost_table_array(get_lost())
    except Exception as exc:
        # Only the binding-not-present case above is silently expected; a raised exception here
        # is either RF-Track's own call failing at runtime or a real bug in
        # `to_lost_table_array` -- swallowing it entirely would silently read downstream as
        # "no lost particles" for the rest of this run. Print loudly and continue (this
        # diagnostic is opt-in via `diagnostics.save_lost_particles`, not worth crashing an
        # otherwise-complete production run over).
        print(f"Warning: failed to extract lost-particle table ({type(exc).__name__}: {exc}); "
              "treating as no lost particles for this run.", flush=True)
        return None


def _try_get_particle_ids(B, selection: str = "all"):
    try:
        ids = np.asarray(B.get_phase_space("%id", selection), dtype=float).reshape(-1)
        return ids if ids.size else None
    except Exception:
        return None


def screen_progress_callback(i: int, z_m: float, stats: Dict[str, float]) -> None:
    """
    Example callback for per-screen progress reporting during transport.
    
    Fires AFTER RF-TRACK transport completes, during screen snapshot extraction.
    This is useful for tracking particle losses and diagnostics across screens.
    
    Note: Live RF-TRACK solver feedback during integration is not available with
    current Python API bindings; RF-Track stepping callbacks are internal to C++.
    
    Args:
        i: Screen index (0, 1, 2, ...)
        z_m: Screen z position in meters
        stats: Dict with keys:
            - N: Number of particles at screen
            - mean_pz: Mean longitudinal momentum [MeV/c]
            - N_lost_since_prev: Particles lost since previous screen (if computed)
    
    Example:
        >>> result, stats = run_transport_with_progress(..., on_screen=screen_progress_callback)
    """
    import sys
    N_lost = stats.get("N_lost_since_prev", None)
    lost_str = f" | lost_since_prev={N_lost}" if N_lost is not None and N_lost > 0 else ""
    print(
        f"  [screen {i}] z={z_m:.4g} m | N={int(stats['N'])} | "
        f"pz_mean={stats['mean_pz']:.4g} MeV/c{lost_str}",
        file=sys.stdout, flush=True
    )


def _resolve_beam_loading_tinj(
    vol_params: VolumeBuildParams,
    B0,
    thermo_info: Dict[str, Any],
) -> VolumeBuildParams:
    if not bool(getattr(vol_params, "beam_loading_enabled", False)):
        return vol_params
    mode = str(getattr(vol_params, "bl_tinj_mode", "manual")).strip().lower()
    if mode != "auto_from_emission":
        return vol_params

    tinj_mm_c = None
    get_t0 = getattr(B0, "get_t0", None)
    if callable(get_t0):
        try:
            t0 = np.asarray(get_t0(), dtype=float).reshape(-1)
            t0 = t0[np.isfinite(t0)]
            if t0.size:
                tinj_mm_c = float(np.min(t0))
        except Exception:
            tinj_mm_c = None

    if tinj_mm_c is None:
        t_model_s = thermo_info.get("t_s", None)
        if t_model_s is not None:
            t_model_s = np.asarray(t_model_s, dtype=float).reshape(-1)
            t_model_s = t_model_s[np.isfinite(t_model_s)]
            if t_model_s.size:
                tinj_mm_c = float(np.min(t_model_s) * c * 1e3)

    if tinj_mm_c is None:
        t_emit_s = thermo_info.get("t_emit_s", None)
        if t_emit_s is not None:
            t_emit_s = np.asarray(t_emit_s, dtype=float).reshape(-1)
            t_emit_s = t_emit_s[np.isfinite(t_emit_s)]
            if t_emit_s.size:
                tinj_mm_c = float(np.quantile(t_emit_s, 1e-3) * c * 1e3)

    if tinj_mm_c is None:
        tinj_mm_c = 0.0
        print("Warning: could not infer emission start time; using tinj=0 mm/c for beam loading.")

    return vol_params.replace(
        bl_tinj_mode="manual",
        bl_tinj_manual_mm_c=float(tinj_mm_c),
    )


def build_bunch_simple(
    rft,
    n: int,
    cathode_radius_mm: float,
    pz0_MeV_c: float,
    q_total_C: float,
    rng: Optional[np.random.Generator] = None,
):
    """Cold emission, sampled over the full cathode disk (no transverse thermal momentum, but a
    finite transverse launch position). This is a *finite-radius* calibration source: useful for
    deliberately probing radial/space-charge-adjacent effects, but not the on-axis reference the
    phase scan needs -- see `build_bunch_on_axis_cold` for that."""
    rng = np.random.default_rng() if rng is None else rng
    x, y = sample_disk(n, cathode_radius_mm, rng=rng)
    px = np.zeros(n)
    py = np.zeros(n)
    z = np.zeros(n)
    pz = np.full(n, float(pz0_MeV_c))

    M = np.column_stack([x, px, y, py, z, pz])
    N_real = float(abs(q_total_C) / q_e)
    B0 = rft.Bunch6dT(ME_MEV, N_real, -1.0, M)
    if hasattr(B0, "set_t0"):
        B0.set_t0(np.zeros(n))
    return B0


def build_bunch_on_axis_cold(rft, n: int, pz0_MeV_c: float, q_total_C: float):
    """Genuinely on-axis, cold calibration source: `x=y=px=py=0` for every particle, exactly (not
    a small-radius disk sample). Every particle in this bunch is identical, so this source carries
    no random-number-generator state at all -- calling it twice, or at any n, at the same phase
    gives the identical result, which is what an RF-only phase-scan calibration needs (the brief:
    "reuse the exact same calibration particle state at every phase, rather than drawing a new disk
    sample from a progressing RNG"). `q_total_C` should be negligible (the calibration current
    should not itself perturb the RF-only field it is measuring); `n` beyond 1 only matters if a
    caller wants nonzero macro-charge spread across more macroparticles, not for phase-scan noise
    reduction, since there is nothing left to average over."""
    x = np.zeros(n)
    y = np.zeros(n)
    px = np.zeros(n)
    py = np.zeros(n)
    z = np.zeros(n)
    pz = np.full(n, float(pz0_MeV_c))

    M = np.column_stack([x, px, y, py, z, pz])
    N_real = float(abs(q_total_C) / q_e)
    B0 = rft.Bunch6dT(ME_MEV, N_real, -1.0, M)
    if hasattr(B0, "set_t0"):
        B0.set_t0(np.zeros(n))
    return B0


def _resolve_work_function(params: EmissionParams) -> EmissionParams:
    """Resolve phi_eff(T) once at `params.cathode_T_K` (guide Sec. 2.3) and return `params` with
    `work_function_eV` replaced by it, when a `work_function_temperature_model` is set -- `params`
    is returned unchanged otherwise. Every downstream computation in each of this module's three
    bunch/source builders (`build_bunch_thermionic`, `build_bunch_thermionic_spatial`,
    `build_cathode_rf_source`) reads `params.work_function_eV`, so replacing it here once is the
    single injection point rather than threading a second work-function value through each
    pipeline separately.
    """
    if params.work_function_temperature_model is None:
        return params
    resolved_phi_eV = evaluate_work_function_eV(
        params.work_function_temperature_model, params.cathode_T_K, **params.work_function_model_params
    )
    return dataclass_replace(params, work_function_eV=resolved_phi_eV)


def build_bunch_thermionic(
    rft,
    n: int,
    phi_deg: float,
    *,
    f_hz: float,
    params: EmissionParams,
    Ez0_phasor_axis: complex,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Thermionic emission with explicit physics blocks."""
    rng = np.random.default_rng() if rng is None else rng
    phi_rad = np.deg2rad(phi_deg)

    params = _resolve_work_function(params)

    Ez0 = float(np.real(Ez0_phasor_axis * np.exp(1j * phi_rad)))
    beta_enh = float(params.beta_enh) if params.beta_enh is not None else float(params.beta_field)
    # Signed extraction field (see the F_ext=max(0,-Ez) note in
    # _compute_emission_waveform_and_current_history): this reference-phase dphi/phi_eff feed
    # only the diagnostics dict below, but should still report 0 lowering, not a spurious nonzero
    # value, if phi_deg happens to fall in the retarding half-cycle.
    F0_ref = beta_enh * max(0.0, -Ez0)
    dphi = delta_phi_schottky_eV(np.array([F0_ref]))[0]
    phi_eff = max(params.work_function_eV - dphi, 0.0)

    area_m2 = np.pi * (params.cathode_radius_mm * 1e-3) ** 2
    area_cm2 = area_m2 * 1e4

    wf = _compute_emission_waveform_and_current_history(
        n,
        phi_rad,
        f_hz,
        params,
        Ez0_phasor_axis,
        area_m2,
        beta_enh,
        rng,
    )

    x, y = sample_disk(n, params.cathode_radius_mm, rng=rng)
    px, py, pz, mean_eps_eV, exp_eps_eV = sample_thermionic_momenta(
        n,
        params.cathode_T_K,
        params.pz0_MeV_c,
        pz_model=params.pz_model,
        rng=rng,
    )

    px_rms0 = float(np.std(px)) if px.size else np.nan
    py_rms0 = float(np.std(py)) if py.size else np.nan
    px, py, sigma_theta = apply_roughness(
        px, py, pz, params.roughness.Ra_um, params.roughness.Re_um, rng=rng
    )
    px_rms = float(np.std(px)) if px.size else np.nan
    py_rms = float(np.std(py)) if py.size else np.nan

    if params.pz_model == "flux":
        print(f"Normal energy: <eps_z>={mean_eps_eV:.4f} eV (expected {exp_eps_eV:.4f} eV)", flush=True)

    t_emit_s = np.asarray(wf.get("t_emit_s", np.zeros(n)), dtype=float)
    t0_mm_c = t_emit_s * c * 1e3

    # Real-particle normalization
    Q_total_C = float(wf.get("Q_total_C", 0.0))
    N_real = float(abs(Q_total_C) / q_e) if Q_total_C > 0.0 else 0.0

    z = np.zeros(n, dtype=float)
    M = np.column_stack([x, px, y, py, z, pz])

    ref_reordered = False

    mass_col = np.full(n, ME_MEV, dtype=float)
    q_col = np.full(n, -1.0, dtype=float)
    N_col = np.full(n, N_real / n if n > 0 else 0.0, dtype=float)

    Mext = np.column_stack([
        x,        # X
        px,       # Px
        y,        # Y
        py,       # Py
        z,        # Z
        pz,       # Pz
        mass_col, # MASS
        q_col,    # Q
        N_col,    # N
        t0_mm_c,  # T0
    ])

    B0 = rft.Bunch6dT(Mext)

    info = _assemble_thermionic_diagnostics(
        wf,
        Ez0=Ez0,
        dphi=dphi,
        phi_eff=phi_eff,
        area_m2=area_m2,
        area_cm2=area_cm2,
        params=params,
        beta_enh=beta_enh,
        mean_eps_eV=mean_eps_eV,
        exp_eps_eV=exp_eps_eV,
        sigma_theta=sigma_theta,
        px_rms0=px_rms0,
        py_rms0=py_rms0,
        px_rms=px_rms,
        py_rms=py_rms,
        has_t0=True,
        reference_particle_reordered=bool(ref_reordered),
    )

    info["initial_phase_space"] = np.asarray(M, dtype=float)
    info["initial_pz_MeV_c"] = np.asarray(M[:, 5], dtype=float)
    info["initial_t0_mm_c"] = np.asarray(t0_mm_c, dtype=float)
    info["t0_span_mm_c"] = float(np.max(t0_mm_c) - np.min(t0_mm_c)) if t0_mm_c.size else 0.0
    info["work_function_temperature_model"] = params.work_function_temperature_model

    return B0, info


def build_bunch_thermionic_spatial(
    rft,
    n: int,
    prescribed_source: Dict[str, np.ndarray],
    *,
    params: EmissionParams,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Thermionic emission with spatially-resolved (x,y,t) RF field and (optionally) cathode
    temperature, an opt-in alternative to build_bunch_thermionic()'s on-axis-only, radially-
    uniform F(t) model (which stays the untouched default -- this is a separate function, not a
    modification of it).

    build_bunch_thermionic() samples position and emission time *independently* (uniform over the
    disk area, time from a single position-independent J(t) waveform derived from the on-axis
    field alone). That assumes both the field and the cathode temperature are uniform across the
    emitting surface. Neither needs to be true: backbombardment and laser heating can produce a
    genuinely asymmetric steady-state temperature profile T(x,y), and this function samples n
    macroparticles *jointly* in (x,y,t) from a real 2D-in-space J(x,y,t) surface supplied via
    `prescribed_source`, built one of two ways by the caller:

    - `build_cathode_rf_source(...)` (below): RF field alone (optionally with a T(x,y) profile
      passed through), sampled on a chosen (n_x, n_y, n_t) grid via the *actual configured
      RF-Track field-map element* (cathode_fields.sample_rf_field_on_cathode, guide Sec. 6.4's own
      preference over re-deriving a second field representation) -- space charge and mirror
      feedback are not included; this is the P1 (signed, spatially-resolved field) upgrade on
      its own.
    - a converged rf_gun.emission_iteration.EmissionFieldIterationResult:
      `{"x_grid_m": result.x_grid_m, "y_grid_m": result.y_grid_m, "t_grid_s": result.t_grid_s,
      "J_Apm2": result.J_history_Apm2[-1], "temperature_K": result.temperature_K}`, which already
      reflects RF+SC+mirror feedback -- the guide Sec. 13.6 "accept a precomputed converged
      source" requirement. Both needs share the same underlying object (a 3D J(x,y,t) surface to
      sample from, plus an optional temperature map), so one function serves both; there is
      deliberately no "compute it myself" branch here, since doing so would need the caller's
      Er_grid/Ez_grid/volume_params anyway -- call build_cathode_rf_source() explicitly instead.

    `prescribed_source["temperature_K"]` (shape (n_x, n_y), optional) gives each sampled
    macroparticle its own local launch-momentum temperature instead of the uniform
    `params.cathode_T_K` -- this is the T(x,y) path guide Sec. 2.3 asks for. Omit it (or pass a
    uniform array) to keep the original uniform-temperature behavior.

    Sampling: standard weighted Monte Carlo -- each of the n macroparticles gets an independently
    drawn (x,y,t) cell (weighted by charge J*area*dt in that cell), a uniform sub-position within
    the cell, and equal per-particle weight N_real/n. This is the guide's own documented fallback
    when a fixed-stratified-sample construction (as in emission_iteration.py, which needs a
    *persistent* sample across outer iterations) is not required -- a one-shot production draw has
    no such persistence constraint.
    """
    rng = np.random.default_rng() if rng is None else rng
    params = _resolve_work_function(params)

    x_grid_m = np.asarray(prescribed_source["x_grid_m"], dtype=float)
    y_grid_m = np.asarray(prescribed_source["y_grid_m"], dtype=float)
    t_centers_s = np.asarray(prescribed_source["t_grid_s"], dtype=float)
    J_xyt = np.asarray(prescribed_source["J_Apm2"], dtype=float)
    n_x, n_y, n_t = x_grid_m.size, y_grid_m.size, t_centers_s.size
    if J_xyt.shape != (n_x, n_y, n_t):
        raise ValueError(f"prescribed_source J_Apm2 shape {J_xyt.shape} does not match (n_x,n_y,n_t)=({n_x},{n_y},{n_t})")

    temperature_field_K = prescribed_source.get("temperature_K")
    if temperature_field_K is not None:
        temperature_field_K = np.asarray(temperature_field_K, dtype=float)
        if temperature_field_K.shape != (n_x, n_y):
            raise ValueError(f"prescribed_source temperature_K shape {temperature_field_K.shape} does not match (n_x,n_y)=({n_x},{n_y})")

    def _edges_from_centers_mm(centers_mm: np.ndarray) -> np.ndarray:
        if centers_mm.size <= 1:
            half = float(params.cathode_radius_mm)
            return np.array([-half, half])
        mid = 0.5 * (centers_mm[:-1] + centers_mm[1:])
        return np.concatenate([[centers_mm[0] - (mid[0] - centers_mm[0])], mid, [centers_mm[-1] + (centers_mm[-1] - mid[-1])]])

    x_centers_mm = x_grid_m * 1e3
    y_centers_mm = y_grid_m * 1e3
    x_edges_mm = _edges_from_centers_mm(x_centers_mm)
    y_edges_mm = _edges_from_centers_mm(y_centers_mm)
    dx_mm = x_edges_mm[1] - x_edges_mm[0] if x_centers_mm.size > 1 else 2.0 * params.cathode_radius_mm
    dy_mm = y_edges_mm[1] - y_edges_mm[0] if y_centers_mm.size > 1 else 2.0 * params.cathode_radius_mm
    cell_area_m2 = (dx_mm * 1e-3) * (dy_mm * 1e-3)

    dt_s = float(t_centers_s[1] - t_centers_s[0]) if t_centers_s.size > 1 else 0.0
    tau_s = float(t_centers_s[-1] - t_centers_s[0] + dt_s) if t_centers_s.size else 0.0

    # x_grid_m/y_grid_m cover the cathode's bounding square, not the disk (same convention as
    # rf_gun.emission_iteration._cathode_xy_grid) -- corner cells outside cathode_radius_mm carry a
    # physically meaningless J_xyt (the iteration zeroes their *area*, not J itself, see
    # _cathode_xy_grid's dA_mm2). Re-masking here keeps charge/sampling consistent with the
    # iteration's own; without it, Q_total_C inflates by ~(4-pi)/pi (~27%) from those corner cells.
    X_mm, Y_mm = np.meshgrid(x_centers_mm, y_centers_mm, indexing="ij")
    inside_disk = (X_mm ** 2 + Y_mm ** 2) <= float(params.cathode_radius_mm) ** 2

    weights_xyt = J_xyt * cell_area_m2 * dt_s  # Coulombs per (x,y,t) cell
    weights_xyt = weights_xyt * inside_disk[:, :, None]
    total_weight = float(np.sum(weights_xyt))
    if total_weight <= 0.0:
        raise RuntimeError("build_bunch_thermionic_spatial: zero total emitted charge over the (x,y,t) grid")
    p_flat = (weights_xyt / total_weight).ravel()

    cell_idx = rng.choice(n_x * n_y * n_t, size=n, p=p_flat)
    ix, iy, it = np.unravel_index(cell_idx, (n_x, n_y, n_t))

    ux, uy = rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)
    x_mm = x_edges_mm[ix] + ux * (x_edges_mm[ix + 1] - x_edges_mm[ix])
    y_mm = y_edges_mm[iy] + uy * (y_edges_mm[iy + 1] - y_edges_mm[iy])

    t_edges_local = np.concatenate([[t_centers_s[0] - 0.5 * dt_s], t_centers_s + 0.5 * dt_s]) if t_centers_s.size > 1 else np.array([0.0, dt_s])
    t_emit_s = rng.uniform(t_edges_local[it], t_edges_local[it + 1])

    T_K_per_particle = temperature_field_K[ix, iy] if temperature_field_K is not None else float(params.cathode_T_K)
    px, py, pz, mean_eps_eV, exp_eps_eV = sample_thermionic_momenta(
        n, T_K_per_particle, params.pz0_MeV_c, pz_model=params.pz_model, rng=rng,
    )
    px, py, sigma_theta = apply_roughness(px, py, pz, params.roughness.Ra_um, params.roughness.Re_um, rng=rng)
    if params.pz_model == "flux":
        mean_exp_eps_eV = float(np.mean(exp_eps_eV)) if np.ndim(exp_eps_eV) else float(exp_eps_eV)
        print(f"Normal energy: <eps_z>={mean_eps_eV:.4f} eV (expected {mean_exp_eps_eV:.4f} eV)", flush=True)

    Q_total_C = total_weight
    N_real = Q_total_C / q_e
    t0_mm_c = t_emit_s * c * 1e3
    z = np.zeros(n, dtype=float)

    Mext = np.column_stack([
        x_mm, px, y_mm, py, z, pz,
        np.full(n, ME_MEV), np.full(n, -1.0), np.full(n, N_real / n if n > 0 else 0.0), t0_mm_c,
    ])
    B0 = rft.Bunch6dT(Mext)

    J_t_marginal = np.sum(weights_xyt, axis=(0, 1)) / (n_x * n_y * cell_area_m2 * dt_s) if dt_s > 0 else np.zeros(n_t)
    beta_enh_report = float(params.beta_enh) if params.beta_enh is not None else float(params.beta_field)

    # Disk-averaged field driving emission, preferring the self-consistent RF+SC+mirror-corrected
    # waveform when available (see the "F_t" entry below).
    _Ez_for_F_t = prescribed_source.get("Ez_corrected_t", prescribed_source.get("Ez_ext_t"))
    F_t = (
        beta_enh_report * np.maximum(-np.asarray(_Ez_for_F_t, dtype=float), 0.0)
        if _Ez_for_F_t is not None else None
    )

    info: Dict[str, Any] = {
        "emission_law": str(params.emission_law),
        "work_function_eV": float(params.work_function_eV),
        "work_function_temperature_model": params.work_function_temperature_model,
        "cathode_T_K": float(params.cathode_T_K),
        "beta_enh": beta_enh_report,
        "spatial_sampling": True,
        "x_grid_m": x_grid_m,
        "y_grid_m": y_grid_m,
        "t_grid_s": t_centers_s,
        "J_xyt_Apm2": J_xyt,
        "temperature_field_K": temperature_field_K,
        "t_s": t_centers_s,
        "J_Apm2_t": J_t_marginal,
        # Disk-averaged external (RF-only)/corrected (RF+SC+mirror) waveforms, when the caller
        # supplied them (see spatial_source_from_iteration_result) -- else None, as on-axis does.
        "Ez_t": np.asarray(prescribed_source["Ez_ext_t"], dtype=float) if "Ez_ext_t" in prescribed_source else None,
        "Ez_corrected_t": np.asarray(prescribed_source["Ez_corrected_t"], dtype=float) if "Ez_corrected_t" in prescribed_source else None,
        # Beta-enhanced extraction field (same F_ext=max(0,-Ez) convention as the on-axis path) --
        # without this, thermo_info["F_t"] stayed absent here, silently skipping the downstream
        # emission-model comparison (it treats an empty F_t history as "nothing to compare").
        "F_t": F_t,
        "Q_total_C": float(Q_total_C),
        "I_avg_A": float(Q_total_C / tau_s) if tau_s > 0 else 0.0,
        "I_peak_A": float(np.max(J_t_marginal) * n_x * n_y * cell_area_m2) if J_t_marginal.size else 0.0,
        "mean_eps_z_eV": float(mean_eps_eV),
        "mean_eps_z_eV_expected": float(np.mean(exp_eps_eV)) if np.ndim(exp_eps_eV) else float(exp_eps_eV),
        "sigma_theta_rad": float(sigma_theta),
        "t_emit_s": t_emit_s,
        "initial_phase_space": np.column_stack([x_mm, px, y_mm, py, z, pz]),
        "initial_pz_MeV_c": pz,
        "initial_t0_mm_c": t0_mm_c,
        "t0_span_mm_c": float(np.max(t0_mm_c) - np.min(t0_mm_c)) if n else 0.0,
    }
    return B0, info


def build_cathode_rf_source(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    phi_deg: float,
    volume_params: "VolumeBuildParams",
    params: EmissionParams,
    n_x_bins: int = 20,
    n_y_bins: int = 20,
    n_time_bins: int = 60,
    z_probe_m: Optional[float] = None,
    temperature_field_K: Optional[TemperatureField] = None,
) -> Dict[str, np.ndarray]:
    """RF-only J(x,y,t) [A/m^2] (guide Sec. 6.4), for build_bunch_thermionic_spatial()'s
    `prescribed_source` argument -- sampled from the *actual configured RF-Track field-map
    element* (not a second, independently reconstructed field), space-charge/mirror-free (that
    feedback is rf_gun.emission_iteration's job, not this function's), on a regular Cartesian grid
    clipped to the cathode disk (see rf_gun.emission_iteration's module docstring for why not a
    radial-only grid: an asymmetric temperature profile needs genuine 2D spatial resolution).

    `temperature_field_K` (optional): a uniform scalar or a callable T_K(x_mm, y_mm) giving each
    grid cell's own local temperature (e.g. fit to a steady-state heater simulation or a measured
    backbombardment power map) -- defaults to params.cathode_T_K everywhere, exactly reproducing
    the single-temperature behavior of every prior version of this function. The resulting map is
    included in the returned dict so build_bunch_thermionic_spatial can also use it for each
    sampled particle's own launch-momentum temperature.
    """
    params = _resolve_work_function(params)

    T_rf = 1.0 / float(volume_params.f_hz)
    tau_s = (max(float(params.emission_phase_range_deg), 0.0) / 360.0) * T_rf
    beta_enh = float(params.beta_enh) if params.beta_enh is not None else float(params.beta_field)
    cathode_radius_mm = float(params.cathode_radius_mm)

    x_edges_mm = np.linspace(-cathode_radius_mm, cathode_radius_mm, n_x_bins + 1)
    y_edges_mm = np.linspace(-cathode_radius_mm, cathode_radius_mm, n_y_bins + 1)
    x_centers_mm = 0.5 * (x_edges_mm[:-1] + x_edges_mm[1:])
    y_centers_mm = 0.5 * (y_edges_mm[:-1] + y_edges_mm[1:])
    X_mm, Y_mm = np.meshgrid(x_centers_mm, y_centers_mm, indexing="ij")
    inside_disk = (X_mm ** 2 + Y_mm ** 2) <= cathode_radius_mm ** 2
    t_centers_s = 0.5 * (np.linspace(0.0, tau_s, n_time_bins + 1)[:-1] + np.linspace(0.0, tau_s, n_time_bins + 1)[1:])

    near_cathode_params = volume_params.replace(
        z_min_m=0.0, z_max_m=max(float(volume_params.z_max_m) * 0.05, 5.0e-3),
        sc_enabled=False, beam_loading_enabled=False, deflection_enabled=False,
    )
    z_probe = float(z_probe_m) if z_probe_m is not None else float(near_cathode_params.z_max_m) / 4.0
    V_field = build_volume(rft, Er_grid, Ez_grid, float(phi_deg), near_cathode_params)

    x_points_m = X_mm.ravel() * 1e-3
    y_points_m = Y_mm.ravel() * 1e-3
    Ez_flat = sample_rf_field_on_cathode(rft, V_field, x_points_m, y_points_m, t_centers_s, z_probe_m=z_probe)
    F_flat = extraction_field(Ez_flat, beta_enh=beta_enh)

    if temperature_field_K is None:
        T_grid_flat = np.full(X_mm.size, float(params.cathode_T_K))
    elif callable(temperature_field_K):
        T_grid_flat = np.asarray(temperature_field_K(X_mm.ravel(), Y_mm.ravel()), dtype=float)
    else:
        T_grid_flat = np.full(X_mm.size, float(temperature_field_K))
    T_broadcast_flat = np.broadcast_to(T_grid_flat[:, None], F_flat.shape)

    J_flat = evaluate_emission_model(
        params.emission_law, F_flat.ravel(), T_broadcast_flat.ravel(), params.work_function_eV
    ).J_Apm2.reshape(F_flat.shape)

    n_x, n_y, n_t = n_x_bins, n_y_bins, t_centers_s.size
    J_xyt = J_flat.reshape(n_x, n_y, n_t) * inside_disk[:, :, None]
    T_grid = T_grid_flat.reshape(n_x, n_y)

    return {
        "x_grid_m": x_centers_mm * 1e-3, "y_grid_m": y_centers_mm * 1e-3, "t_grid_s": t_centers_s,
        "J_Apm2": J_xyt, "temperature_K": T_grid,
    }


def _compute_emission_waveform_and_current_history(
    n: int,
    phi_rad: float,
    f_hz: float,
    params: EmissionParams,
    Ez0_phasor_axis: complex,
    area_m2: float,
    beta_enh: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    f_hz = float(f_hz)
    T = 1.0 / f_hz
    phase_range_deg = max(float(params.emission_phase_range_deg), 0.0)
    tau_s = (phase_range_deg / 360.0) * T

    if not params.time_dependent:
        # Signed extraction field: F_ext = max(0, -Ez), confirmed against this repository's own
        # field-map/phasor convention by tests/test_field_sign_convention.py (an electron is
        # pushed toward +z, into the cavity, when Ez<0 -- verified via a real single-electron
        # RF-Track push at the phase run_phase_scan finds to maximize forward pz). During the
        # retarding half-cycle (Ez>0) this correctly zeroes the Schottky lowering term without
        # zeroing thermionic supply itself: J_rld_schottky(F=0, ...) reduces to plain
        # unenhanced Richardson-Dushman, not zero, matching the implementation guide Sec. 6.2.
        Ez0 = float(np.real(Ez0_phasor_axis * np.exp(1j * phi_rad)))
        F0 = beta_enh * max(0.0, -Ez0)
        res0 = evaluate_emission_model(params.emission_law, np.array([F0]), params.cathode_T_K, params.work_function_eV)
        J0 = float(res0.J_Apm2[0])
        n_t = res0.regime_n
        J_th_t = res0.J_thermionic_Apm2
        J_fe_t = res0.J_field_Apm2

        I_avg = J0 * area_m2
        Q_total_C = float(I_avg * tau_s) if np.isfinite(tau_s) else 0.0
        t_emit_s = rng.uniform(0.0, tau_s, size=n) if tau_s > 0.0 else np.zeros(n)
        R_t = (J_fe_t / np.maximum(J_th_t, 1e-300)) if J_th_t is not None and J_fe_t is not None else None
        return {
            "t_s": None,
            "Ez_t": None,
            "F_t": None,
            "dphi_eV_t": None,
            "phi_eff_eV_t": None,
            "J_Apm2_t": None,
            "J_th_Apm2_t": J_th_t,
            "J_fe_Apm2_t": J_fe_t,
            "R_t": R_t,
            "n_t": n_t,
            "I_A_t": None,
            "Q_cum_C": None,
            "tau_s": float(tau_s),
            "I_avg_A": float(I_avg),
            "I_peak_A": float(I_avg),
            "Q_total_C": float(Q_total_C),
            "t_emit_s": np.asarray(t_emit_s, dtype=float),
            "J_Apm2": float(J0),
            "I_A": float(I_avg),
            "n_at_peak": float(n_t[0]) if n_t is not None and np.size(n_t) else None,
            "n_at_peak_field": float(n_t[0]) if n_t is not None and np.size(n_t) else None,
        }

    omega = 2.0 * np.pi * f_hz
    samples_per_period = max(200, int(phase_range_deg * 2.0))
    n_samples = max(int(samples_per_period * phase_range_deg / 360.0) + 1, 2)
    t_s = np.linspace(0.0, tau_s, n_samples)
    Ez_t = np.real(Ez0_phasor_axis * np.exp(1j * (omega * t_s + phi_rad)))
    # See the identical F_ext=max(0,-Ez) note in the `not time_dependent` branch above.
    F_t = beta_enh * np.maximum(-Ez_t, 0.0)

    res_t = evaluate_emission_model(params.emission_law, F_t, params.cathode_T_K, params.work_function_eV)
    J_t = res_t.J_Apm2
    n_t = res_t.regime_n
    J_th_t = res_t.J_thermionic_Apm2
    J_fe_t = res_t.J_field_Apm2

    dphi_t = delta_phi_schottky_eV(F_t)
    phi_eff_t = np.maximum(params.work_function_eV - dphi_t, 0.0)
    I_t = J_t * area_m2
    R_t = (J_fe_t / np.maximum(J_th_t, 1e-300)) if J_th_t is not None and J_fe_t is not None else None

    dt = t_s[1] - t_s[0] if t_s.size > 1 else 0.0
    Q_cum = np.zeros_like(t_s)
    if t_s.size > 1:
        Q_cum[1:] = np.cumsum((I_t[:-1] + I_t[1:]) * 0.5) * dt

    t_emit_s = _sample_emission_times_from_cumulative_charge(n, t_s, Q_cum, rng)
    Q_total_C = float(Q_cum[-1]) if Q_cum.size else 0.0
    I_avg = float(Q_total_C / tau_s) if tau_s and np.isfinite(tau_s) else 0.0
    I_peak = float(np.max(I_t)) if I_t is not None and I_t.size else 0.0

    return {
        "t_s": t_s,
        "Ez_t": Ez_t,
        "F_t": F_t,
        "dphi_eV_t": dphi_t,
        "phi_eff_eV_t": phi_eff_t,
        "J_Apm2_t": J_t,
        "J_th_Apm2_t": J_th_t,
        "J_fe_Apm2_t": J_fe_t,
        "R_t": R_t,
        "n_t": n_t,
        "I_A_t": I_t,
        "Q_cum_C": Q_cum,
        "tau_s": float(tau_s),
        "I_avg_A": float(I_avg),
        "I_peak_A": float(I_peak),
        "Q_total_C": float(Q_total_C),
        "t_emit_s": np.asarray(t_emit_s, dtype=float),
        "J_Apm2": float(np.max(J_t)) if np.size(J_t) else 0.0,
        "I_A": float(np.max(I_t)) if np.size(I_t) else 0.0,
        "n_at_peak": float(n_t[np.argmax(J_t)]) if n_t is not None and J_t is not None and J_t.size else None,
        "n_at_peak_field": float(n_t[np.argmax(J_fe_t)]) if n_t is not None and J_fe_t is not None and J_fe_t.size else None,
    }


def _sample_emission_times_from_cumulative_charge(
    n: int,
    t_s: np.ndarray,
    Q_cum: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if t_s is None or Q_cum is None or np.size(t_s) == 0 or np.size(Q_cum) == 0:
        return np.zeros(n)
    q_tot = float(Q_cum[-1])
    if q_tot <= 0.0:
        return np.zeros(n)
    return np.interp(rng.random(n) * q_tot, Q_cum, t_s)


def _assemble_thermionic_diagnostics(
    wf: Dict[str, Any],
    *,
    Ez0: float,
    dphi: float,
    phi_eff: float,
    area_m2: float,
    area_cm2: float,
    params: EmissionParams,
    beta_enh: float,
    mean_eps_eV: float,
    exp_eps_eV: float,
    sigma_theta: float,
    px_rms0: float,
    py_rms0: float,
    px_rms: float,
    py_rms: float,
    has_t0: bool,
    reference_particle_reordered: bool,
) -> Dict[str, Any]:
    return {
        "Ez0": Ez0,
        "dphi_eV": float(dphi),
        "phi_eff_eV": float(phi_eff),
        "J_Apm2": float(wf.get("J_Apm2", 0.0)),
        "I_A": float(wf.get("I_A", 0.0)),
        "I_avg_A": float(wf.get("I_avg_A", 0.0)),
        "I_peak_A": float(wf.get("I_peak_A", 0.0)),
        "tau_ns": float(wf.get("tau_s", np.inf) * 1e9) if np.isfinite(wf.get("tau_s", np.inf)) else np.inf,
        "tau_s": float(wf.get("tau_s", np.inf)),
        "Q_total_C": float(wf.get("Q_total_C", 0.0)),
        "emission_phase_range_deg": float(params.emission_phase_range_deg),
        "pz_model": str(params.pz_model),
        "mean_eps_z_eV": float(mean_eps_eV),
        "mean_eps_z_eV_expected": float(exp_eps_eV),
        "Ra_um": float(params.roughness.Ra_um),
        "Re_um": float(params.roughness.Re_um),
        "sigma_theta_rad": float(sigma_theta),
        "px_rms0": float(px_rms0),
        "py_rms0": float(py_rms0),
        "px_rms": float(px_rms),
        "py_rms": float(py_rms),
        "emission_law": str(params.emission_law),
        "beta_enh": float(beta_enh),
        "cathode_T_K": float(params.cathode_T_K),
        "work_function_eV": float(params.work_function_eV),
        "t_s": wf.get("t_s", None),
        "Ez_t": wf.get("Ez_t", None),
        "F_t": wf.get("F_t", None),
        "dphi_eV_t": wf.get("dphi_eV_t", None),
        "phi_eff_eV_t": wf.get("phi_eff_eV_t", None),
        "J_Apm2_t": wf.get("J_Apm2_t", None),
        "J_th_Apm2_t": wf.get("J_th_Apm2_t", None),
        "J_fe_Apm2_t": wf.get("J_fe_Apm2_t", None),
        "R_t": wf.get("R_t", None),
        "n_t": wf.get("n_t", None),
        "n_at_peak": wf.get("n_at_peak", None),
        "n_at_peak_field": wf.get("n_at_peak_field", None),
        "I_A_t": wf.get("I_A_t", None),
        "area_m2": float(area_m2),
        "area_cm2": float(area_cm2),
        "Q_cum_C": wf.get("Q_cum_C", None),
        "t_emit_s": wf.get("t_emit_s", None),
        "has_t0": bool(has_t0),
        "bunch_constructor": "extended_matrix_with_T0",
        "bunch_constructor_full": "Bunch6dT_extended_matrix_with_T0",
        "timing_coordinate_note": (
            "In Bunch6dT, Z is position and T0 is creation time; T0 is separate from the 6D coordinates."
        ),
        "reference_particle_reordered": bool(reference_particle_reordered),
    }


def run_phase_scan(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    vol_params: VolumeBuildParams,
    phase_rel_deg: Sequence[float],
    transport_phase_deg: float,
    n_particles: int,
    pz0_MeV_c: float,
    q_total_C: float = 1e-12,
    refine: bool = True,
    refine_xatol_deg: float = 0.05,
    refine_maxiter: int = 30,
) -> PhaseCalibrationResult:
    """RF-only phase scan with a genuinely on-axis, cold calibration source.

    Every physics switch except the RF field itself is off for this scan (`vol_params_fast` below
    forces space charge and beam loading off regardless of what `vol_params` requests), and the
    same `build_bunch_on_axis_cold` source (`x=y=px=py=0`, negligible `q_total_C`) is reused
    unchanged at every phase -- there is no RNG involved, so this is not "the same distribution
    resampled," it is the identical particle state.

    Runs a coarse scan over `phase_rel_deg` (kept full-range by default so a change in the input
    field maps is still caught), then, unless `refine=False`, brackets the coarse maximum and
    refines it with a bounded scalar search. The refined point is folded into the returned,
    phase-sorted result, so the returned crest reflects the refined value rather than the coarse
    grid resolution. This lets the coarse grid `phase_rel_deg` be much sparser than a fine grid
    would need to be, without losing crest accuracy.

    Returns a `PhaseCalibrationResult`; callers must check `.valid` before using `.crest_*` for
    `Veff`/`R/Q`/any beam-loading-dependent path (see `rf_params.veff_from_phase_calibration`).
    """
    z_span_mm = abs(float(vol_params.z_max_m) - float(vol_params.z_min_m)) * 1e3
    phase_scan_tmax_mm = max(60.0, 2.5 * z_span_mm)

    # A5: RF-only calibration -- space charge, mirror, beam loading, backstop, and deflection are
    # all explicitly off, regardless of what the caller's `vol_params` requests for production.
    vol_params_fast = vol_params.replace(
        sc_enabled=False,
        beam_loading_enabled=False,
        beam_loading_verbose=False,
        mirror_charge_enabled=False,
        deflection_enabled=False,
        cathode_backstop_enabled=False,
        t_max_mm=min(float(getattr(vol_params, "t_max_mm", 2000.0)), float(phase_scan_tmax_mm)),
    )

    def _mean_pz_at(phi_rel: float) -> float:
        phi_abs = (float(phi_rel) + float(transport_phase_deg)) % 360.0
        V = build_volume(rft, Er_grid, Ez_grid, phi_abs, vol_params_fast)
        B0 = build_bunch_on_axis_cold(rft, n_particles, pz0_MeV_c, q_total_C)
        Bout = V.track(B0)
        Mf = Bout.get_phase_space()
        if Mf.shape[0] == 0:
            return np.nan
        return float(np.mean(Mf[:, 5]))

    phi_rel_coarse = np.asarray(phase_rel_deg, dtype=float)
    n_coarse = phi_rel_coarse.size
    # A grid built with endpoint=False over a full 360deg request (run_thermionic_tm010.py's and
    # the notebook's own "don't repeat 0 and 360" fix) satisfies span+step == 360deg exactly; such
    # a grid is physically periodic even though it is stored as a flat array with no explicit
    # wraparound point -- phase 359.x deg is adjacent to phase 0 deg. A genuine partial-range scan
    # (span+step << 360) is not periodic and its array ends are real boundaries.
    circular = False
    if n_coarse > 2:
        span = float(phi_rel_coarse.max() - phi_rel_coarse.min())
        step = span / (n_coarse - 1)
        circular = bool(np.isclose(span + step, 360.0, atol=1e-6))

    phase_scan = []
    for phi in phi_rel_coarse:
        pz = _mean_pz_at(float(phi))
        phi_abs = (float(phi) + float(transport_phase_deg)) % 360.0
        n_ok = 0 if np.isnan(pz) else int(n_particles)
        phase_scan.append((float(phi), float(phi_abs), pz, n_ok))

    refined_applied = False
    if refine and np.any(np.isfinite([row[2] for row in phase_scan])):
        pz_coarse = np.array([row[2] for row in phase_scan], dtype=float)
        i_max = int(np.nanargmax(pz_coarse))
        if circular and (i_max == 0 or i_max == n_coarse - 1):
            # The coarse crest sits at the array seam of a periodic scan: bracket it with its true
            # circular neighbor (wrapping to the other end, offset by a full period) rather than
            # only exploring one side, which a flat-array index computation would otherwise do.
            if i_max == 0:
                lo = float(phi_rel_coarse[-1]) - 360.0
                hi = float(phi_rel_coarse[0])
            else:
                lo = float(phi_rel_coarse[-1])
                hi = float(phi_rel_coarse[0]) + 360.0
        else:
            lo = phi_rel_coarse[max(i_max - 1, 0)]
            hi = phi_rel_coarse[min(i_max + 1, n_coarse - 1)]
        if hi > lo:
            res = minimize_scalar(
                lambda phi: -_mean_pz_at(phi),
                bounds=(float(lo), float(hi)),
                method="bounded",
                options={"xatol": float(refine_xatol_deg), "maxiter": int(refine_maxiter)},
            )
            phi_rel_refined = float(res.x)
            pz_refined = -float(res.fun)
            if np.isfinite(pz_refined):
                phi_abs_refined = (phi_rel_refined + float(transport_phase_deg)) % 360.0
                phase_scan.append((phi_rel_refined % 360.0, phi_abs_refined, pz_refined, int(n_particles)))
                refined_applied = True

    phase_scan_arr = np.array(sorted(phase_scan, key=lambda row: row[0]), dtype=float)
    return build_phase_calibration_result(
        phi_rel_deg=phase_scan_arr[:, 0],
        phi_abs_deg=phase_scan_arr[:, 1],
        pz_mean_MeV_c=phase_scan_arr[:, 2],
        n_ok=phase_scan_arr[:, 3],
        pz0_MeV_c=float(pz0_MeV_c),
        refined=refined_applied,
        circular=circular,
    )


def run_transport_with_progress(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    Ez0_phasor_axis: complex,
    vol_params: VolumeBuildParams,
    emission: EmissionParams,
    tracking: TrackingParams,
    diagnostics: DiagnosticsParams | None = None,
    timing_diagnostics: bool = False,
    slow_step_warn_s: float = 20.0,
    rng: Optional[np.random.Generator] = None,
    on_screen: Optional[Callable[[int, float, Dict[str, float]], None]] = None,
    spatial_source: Optional[Dict[str, np.ndarray]] = None,
):
    """Run transport with staged progress text.

    `spatial_source` (guide Sec. 6.4/13.6): when given (a `{"x_grid_m","y_grid_m","t_grid_s",
    "J_Apm2"}` dict, e.g. from `build_cathode_rf_source(...)` or a converged
    `rf_gun.emission_iteration.EmissionFieldIterationResult`), the bunch is built via
    `build_bunch_thermionic_spatial` instead of the default on-axis-only `build_bunch_thermionic` --
    opt-in, so every existing call site's behavior is unchanged unless it explicitly passes this.

    There is no live progress bar during tracking: RF-Track's Python bindings expose no
    per-step callback, so any in-run percentage would be a wall-clock guess, not real progress.
    With the deflection magnet on, tracking is additionally forced single-threaded (it calls back
    into Python from the tracking thread via `DeflectionField.get_field`), so a concurrent
    background thread printing or refreshing a widget is unsafe here. The `_set_stage` prints
    below (`1/5` .. `5/5`) are the only progress feedback during a run; see
    `track_elapsed_s`/`track_estimate_s` in the returned stats for timing after the fact.

    Returns:
        (SimulationResult, stats_dict)
        where stats_dict has `track_elapsed_s`, `track_estimate_s`, `runtime_key_hash`.
    """

    diagnostics = DiagnosticsParams() if diagnostics is None else diagnostics

    stages = [
        "Build thermionic bunch",
        "Prepare tracking and screens",
        "Run RF-Track transport",
        "Extract diagnostics",
        "Finalize summaries",
    ]

    def _set_stage(i: int):
        print(f"{i} / {len(stages)}: {stages[i-1]}", flush=True)

    diag_enabled = bool(timing_diagnostics)

    def _diag(msg: str) -> None:
        if diag_enabled:
            print(f"[timing] {msg}", flush=True)

    def _warn_slow(label: str, elapsed_s: float) -> None:
        if diag_enabled and np.isfinite(elapsed_s) and elapsed_s >= float(slow_step_warn_s):
            print(
                f"[timing][slow-step] {label}: {elapsed_s:.2f} s "
                f"(threshold {float(slow_step_warn_s):.2f} s)",
                flush=True,
            )

    print("---- Simulation Start ----", flush=True)
    _set_stage(1)
    if spatial_source is not None:
        B0, thermo_info = build_bunch_thermionic_spatial(
            rft, tracking.n_particles, spatial_source, params=emission, rng=rng,
        )
    else:
        B0, thermo_info = build_bunch_thermionic(
            rft,
            tracking.n_particles,
            tracking.phi_deg,
            f_hz=vol_params.f_hz,
            params=emission,
            Ez0_phasor_axis=Ez0_phasor_axis,
            rng=rng,
        )
    vol_params_eff = _resolve_beam_loading_tinj(vol_params, B0, thermo_info)

    if bool(getattr(vol_params_eff, "beam_loading_enabled", False)):
        mode_in = str(getattr(vol_params, "bl_tinj_mode", "manual")).strip().lower()
        mode_eff = str(getattr(vol_params_eff, "bl_tinj_mode", "manual")).strip().lower()
        if mode_in == "auto_from_emission" and mode_eff == "manual":
            print(
                "Beam-loading tinj source: auto_from_emission -> "
                f"resolved tinj={float(getattr(vol_params_eff, 'bl_tinj_manual_mm_c', 0.0)):.4e} mm/c"
            , flush=True)

    _set_stage(2)
    z_snaps_in = list(tracking.z_screens_m) if tracking.z_screens_m is not None else []
    z_snaps = _sanitize_screen_positions(z_snaps_in)

    runtime_payload = _runtime_key_payload(vol_params_eff, tracking, len(z_snaps))
    runtime_payload["spatial_emission_sampling"] = spatial_source is not None
    runtime_key = _runtime_key_string(runtime_payload)
    runtime_key_hash = _runtime_key_hash(runtime_key)
    est_s = _TRANSPORT_RUNTIME_HISTORY.get(runtime_key, None)

    settings_line = (
        f"Tracking settings | N={int(tracking.n_particles):,} | dt_mm={float(getattr(vol_params_eff, 'dt_mm', np.nan)):.4g} "
        f"| sc_dt_mm={float(getattr(vol_params_eff, 'sc_dt_mm', np.nan)):.4g} "
        f"| emission_sc_steps={int(getattr(vol_params_eff, 'emission_nsteps', 0))} "
        f"| screens={len(z_snaps)} | sc={'on' if bool(getattr(vol_params_eff, 'sc_enabled', False)) else 'off'} "
        f"| bl={'on' if bool(getattr(vol_params_eff, 'beam_loading_enabled', False)) else 'off'} "
        f"| key={runtime_key_hash}"
    )
    print(settings_line, flush=True)
    print("(emission_sc_steps = number of SC kicks/substeps applied during emission)", flush=True)

    if bool(getattr(vol_params_eff, "beam_loading_enabled", False)):
        omega = 2.0 * np.pi * float(getattr(vol_params_eff, "f_hz", np.nan))
        tau_s = 2.0 * float(getattr(vol_params_eff, "bl_Q_loaded", 0.0)) / omega if omega > 0.0 else np.nan
        mode = str(getattr(vol_params_eff, "bl_tinj_mode", "manual")).strip().lower()
        tinj_mm_c = float(getattr(vol_params_eff, "bl_tinj_manual_mm_c", 0.0)) if mode == "manual" else 0.0
        tinj_s = (tinj_mm_c * 1e-3) / c
        tinj_tau = tinj_s / tau_s if np.isfinite(tau_s) and tau_s > 0.0 else np.nan
        print(
            "Beam-loading timing | "
            f"tau={tau_s:.4e} s | tinj={tinj_mm_c:.4e} mm/c | tinj/tau={tinj_tau:.4e}"
        , flush=True)

    if est_s is not None and np.isfinite(est_s):
        print(f"Last-run estimate for this key: {float(est_s):.2f} s", flush=True)

    _set_stage(3)
    track_start_s = time.time()

    vol_params_track = vol_params_eff.replace(beam_loading_verbose=False)

    def _run_tracking_once():
        if len(z_snaps) > 0:
            screen_params = _maybe_screen_params(tracking)
            return track_volume_with_screens(
                rft,
                Er_grid,
                Ez_grid,
                tracking.phi_deg,
                vol_params_track,
                B0,
                z_snaps,
                screen_params=screen_params,
                return_volume=True,
            )
        V = build_volume(rft, Er_grid, Ez_grid, tracking.phi_deg, vol_params_track)
        return V.track(B0), [], V

    t_solver_s = time.time()
    Bout, snaps, V = _run_tracking_once()
    solver_elapsed_s = time.time() - t_solver_s
    _diag(f"RF-Track solver return: {solver_elapsed_s:.2f} s")
    _warn_slow("RF-Track solver return", solver_elapsed_s)

    t_extract_phase_s = time.time()
    keep_idx = _select_screen_indices(len(snaps), diagnostics)
    z_snaps_kept = [z_snaps[i] for i in keep_idx] if z_snaps else []

    full_M_snaps = [np.array(snaps[i].get_phase_space(tracking.phase_fmt, "all"), copy=True) for i in keep_idx] if snaps else []

    if diagnostics.store_screen_phase_space and full_M_snaps:
        M_snaps = [
            _maybe_subsample_phase_space(
                raw,
                diagnostics.max_screen_particles,
                rng,
                diagnostics.subsample_screens_random,
            )
            for raw in full_M_snaps
        ]
    else:
        M_snaps = []
    extract_phase_elapsed_s = time.time() - t_extract_phase_s
    _diag(f"Extract phase-space snapshots: {extract_phase_elapsed_s:.2f} s ({len(M_snaps)} screens)")
    _warn_slow("Extract phase-space snapshots", extract_phase_elapsed_s)

    t_extract_info_s = time.time()
    full_I_snaps = [s.get_info() if hasattr(s, "get_info") else None for s in snaps] if snaps else []
    I_snaps = [full_I_snaps[i] for i in keep_idx] if diagnostics.store_screen_info and full_I_snaps else []
    extract_info_elapsed_s = time.time() - t_extract_info_s
    _diag(f"Extract screen info: {extract_info_elapsed_s:.2f} s ({len(I_snaps)} screens)")
    _warn_slow("Extract screen info", extract_info_elapsed_s)

    t_extract_lost_s = time.time()
    lost_table = _extract_lost_particles(V) if diagnostics.save_lost_particles else None
    extract_lost_elapsed_s = time.time() - t_extract_lost_s
    _diag(f"Extract lost-particle table: {extract_lost_elapsed_s:.2f} s")

    track_elapsed_s = time.time() - track_start_s

    t_cache_s = time.time()
    if est_s is None:
        _TRANSPORT_RUNTIME_HISTORY[runtime_key] = track_elapsed_s
    else:
        _TRANSPORT_RUNTIME_HISTORY[runtime_key] = 0.7 * float(est_s) + 0.3 * track_elapsed_s
    _save_transport_runtime_history(_TRANSPORT_RUNTIME_HISTORY)
    cache_elapsed_s = time.time() - t_cache_s
    _diag(f"Update runtime cache: {cache_elapsed_s:.2f} s")
    _warn_slow("Update runtime cache", cache_elapsed_s)

    _set_stage(4)

    if z_snaps and np.any(np.diff(np.asarray(z_snaps, dtype=float)) < 0.0):
        print("Warning: z_screens_m is not monotonic increasing.", flush=True)

    t_callbacks_s = time.time()
    n_initial = int(np.asarray(B0.get_phase_space(tracking.phase_fmt, "all")).shape[0])
    screen_summaries = []
    n_prev = n_initial
    for i, z_m in enumerate(z_snaps_kept):
        rec = build_screen_summary_from_phase_space(
            full_M_snaps[i] if i < len(full_M_snaps) else None,
            screen_index=i,
            z_m=float(z_m),
            n_initial=n_initial,
            n_previous=n_prev,
        )
        info_i = I_snaps[i] if i < len(I_snaps) else None
        rec["rftrack_raw_info"] = {
            "transmission": float(info_get_first(info_i, ["transmission", "Transmission"])) if info_i is not None else np.nan,
            "mean_pz": float(info_get_first(info_i, ["mean_Pz", "mean_P", "mean_pz"])) if info_i is not None else np.nan,
            "sigma_pz": float(info_get_first(info_i, ["sigma_Pz", "sigma_P", "sigma_pz"])) if info_i is not None else np.nan,
        }
        screen_summaries.append(rec)
        n_prev = int(rec.get("N", 0))

    if screen_summaries:
        prev_n = None
        for i, (z_m, rec) in enumerate(zip(z_snaps_kept, screen_summaries)):
            n_cur = float(rec.get("N", np.nan))
            stats = {
                "N": n_cur,
                "mean_pz": float(rec.get("mean_pz", rec.get("mean_pz_info", np.nan))),
                "N_lost_since_prev": float(prev_n - n_cur) if prev_n is not None and np.isfinite(prev_n) and np.isfinite(n_cur) else np.nan,
            }
            prev_n = n_cur
            if on_screen is not None:
                on_screen(i, float(z_m), stats)
    callbacks_elapsed_s = time.time() - t_callbacks_s
    _diag(f"Stage 4 callback loop: {callbacks_elapsed_s:.2f} s ({len(M_snaps)} screens)")
    _warn_slow("Stage 4 callback loop", callbacks_elapsed_s)

    _set_stage(5)
    post_track_elapsed_s = extract_phase_elapsed_s + extract_info_elapsed_s + extract_lost_elapsed_s + cache_elapsed_s + callbacks_elapsed_s
    _diag(
        "Post-track total: "
        f"{post_track_elapsed_s:.2f} s "
            f"(phase={extract_phase_elapsed_s:.2f}s, info={extract_info_elapsed_s:.2f}s, lost={extract_lost_elapsed_s:.2f}s, "
        f"cache={cache_elapsed_s:.2f}s, callbacks={callbacks_elapsed_s:.2f}s)"
    )
    _warn_slow("Post-track total", post_track_elapsed_s)
    print("---- Simulation End ----", flush=True)

    m0 = np.array(B0.get_phase_space(tracking.phase_fmt, "all"), copy=True)
    mf = np.array(Bout.get_phase_space(tracking.phase_fmt, "all"), copy=True)
    t0_mm_c = np.asarray(thermo_info.get("initial_t0_mm_c", []), dtype=float)
    classes = classify_particle_outcomes(m0, mf, t0_mm_c=t0_mm_c if t0_mm_c.size else None, lost_table=lost_table)
    init_ids = _try_get_particle_ids(B0, selection="all")
    classes["initial_t0_mm_c"] = t0_mm_c.tolist() if t0_mm_c.size else []
    classes["initial_pz_MeV_c"] = np.asarray(m0[:, 5], dtype=float).tolist() if m0.ndim == 2 and m0.shape[1] > 5 else []
    classes["particle_id"] = init_ids.tolist() if init_ids is not None else []

    result = SimulationResult(
        B0=B0,
        Bout=Bout,
        thermo_info=thermo_info,
        M_snaps=M_snaps,
        z_snaps=z_snaps_kept,
        I_snaps=I_snaps,
        screen_summaries=screen_summaries,
        lost_table=lost_table,
        particle_classes=classes,
    )
    return result, {
        "track_elapsed_s": float(track_elapsed_s),
        "track_estimate_s": float(_TRANSPORT_RUNTIME_HISTORY[runtime_key]),
        "runtime_key_hash": runtime_key_hash,
        "runtime_cache_file": str(_RUNTIME_HISTORY_CACHE),
        "timing_diagnostics": bool(diag_enabled),
        "timings": {
            "solver_return_s": float(solver_elapsed_s),
            "extract_phase_space_s": float(extract_phase_elapsed_s),
            "extract_screen_info_s": float(extract_info_elapsed_s),
            "extract_lost_particles_s": float(extract_lost_elapsed_s),
            "runtime_cache_update_s": float(cache_elapsed_s),
            "stage4_callback_loop_s": float(callbacks_elapsed_s),
            "post_track_total_s": float(post_track_elapsed_s),
        },
    }
