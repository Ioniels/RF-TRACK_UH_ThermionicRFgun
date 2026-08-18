"""Simulation pipeline helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Literal
import math
import multiprocessing as mp
import time
import os
import json
import hashlib
import threading
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .constants import ME_MEV, c, q_e
from .helpers import sample_disk
from .emission_models import (
    J_rld_schottky,
    J_unified,
    delta_phi_schottky_eV,
    richardson_J_Apm2,
    schottky_delta_phi_eV,
)
from .emission_sampling import apply_roughness, sample_thermionic_momenta
from .rftrack_volume import (
    build_volume,
    track_volume_with_screens,
    track_volume_transport_table,
    VolumeBuildParams,
    ScreenBuildParams,
)
from .diagnostics import (
    snapshot_stats,
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
    emission_law: Literal["RD_schottky", "unified"] = "RD_schottky"
    beta_enh: float = 1.0
    roughness: RoughnessParams = RoughnessParams()
    time_dependent: bool = True


#: The project's core 6-column phase-space convention (X, Px, Y, Py, Z, Pz), all in RF-Track's
#: native Bunch6dT units (mm, MeV/c). Extended with %id (particle id, for cross-referencing a
#: row's identity across B0/screens/Bout -- a Screen's own Pz does not reliably carry the true
#: lab-frame sign, so id-based lookups against Bout's classification are the reliable way to tag
#: forward/backward status at a screen), %t (arrival time, mm/c), %E (total energy, MeV), and %K
#: (kinetic energy, MeV) -- all confirmed valid RF-Track format codes. Every consumer that needs
#: exactly the core 6 columns already slices `[:, :6]` explicitly, so this extension is additive.
EXTENDED_PHASE_FMT = "%X %Px %Y %Py %Z %Pz %id %t %E %K"


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
    use_transport_table_summary: bool = True
    transport_table_dt_mm: Optional[float] = None
    transport_table_fmt: str = "%s %mean_P %sigma_P %sigma_X %sigma_Y"
    time_slice_t_max_mm: Optional[Sequence[float]] = None
    save_screen_json: bool = False
    screen_json_mode: Literal["summary", "full"] = "summary"
    save_npz: bool = True


@dataclass
class SimulationResult:
    B0: Any
    Bout: Any
    thermo_info: Dict[str, Any]
    M_snaps: List[np.ndarray]
    z_snaps: List[float]
    I_snaps: List[Any]
    screen_summaries: List[Dict[str, float]]
    transport_table: Any = None
    lost_table: Optional[np.ndarray] = None
    particle_classes: Optional[Dict[str, Any]] = None


_RUNTIME_HISTORY_CACHE = Path(
    os.environ.get(
        "RF_GUN_RUNTIME_CACHE",
        str(Path(__file__).resolve().parents[1] / ".rf_gun_transport_runtime_history.json"),
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
_ACTIVE_PROGRESS_STOP_EVENT: Optional[threading.Event] = None
_ACTIVE_PROGRESS_THREAD: Optional[threading.Thread] = None


def _stop_progress_worker(stop_event, thread, timeout_s: float = 2.0) -> None:
    if stop_event is not None:
        try:
            stop_event.set()
        except Exception:
            pass
    if thread is not None:
        try:
            thread.join(timeout=float(timeout_s))
        except Exception:
            pass


def _clear_active_progress_worker(timeout_s: float = 0.5) -> None:
    global _ACTIVE_PROGRESS_STOP_EVENT, _ACTIVE_PROGRESS_THREAD
    _stop_progress_worker(_ACTIVE_PROGRESS_STOP_EVENT, _ACTIVE_PROGRESS_THREAD, timeout_s=timeout_s)
    _ACTIVE_PROGRESS_STOP_EVENT = None
    _ACTIVE_PROGRESS_THREAD = None


_ACTIVE_PROGRESS_PROCESS: Optional["mp.process.BaseProcess"] = None


def _stop_progress_process(proc, timeout_s: float = 2.0) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.join(timeout=float(timeout_s))
    except Exception:
        pass


def _clear_active_progress_process(timeout_s: float = 0.5) -> None:
    global _ACTIVE_PROGRESS_PROCESS
    _stop_progress_process(_ACTIVE_PROGRESS_PROCESS, timeout_s=timeout_s)
    _ACTIVE_PROGRESS_PROCESS = None


def _progress_proxy_pct(elapsed_s: float, est_s: float) -> float:
    """Wall-clock proxy percentage against a historical runtime estimate.

    Not real RF-Track tracking progress -- RF-Track's Python bindings expose no
    per-step callback (see `screen_progress_callback`'s docstring), so this is the
    best available estimate: how far `elapsed_s` is through a runtime estimate drawn
    from `_TRANSPORT_RUNTIME_HISTORY` (or a heuristic fallback), asymptoting toward
    99% if the run overruns that estimate.
    """
    est = max(1e-9, float(est_s))
    if elapsed_s <= est:
        return min(98.0, 100.0 * elapsed_s / est)
    over = (elapsed_s - est) / est
    return min(99.0, 98.0 + (1.0 - math.exp(-over)))


def _elapsed_progress_worker(stop_event, est_for_proxy_s: float, poll_interval_s: float) -> None:
    """Background daemon thread only -- never touches RF-Track, numpy, or the tracked bunch.

    A previous implementation ran this in a forked OS subprocess
    (`multiprocessing.get_context("fork")`), concurrently with the blocking call into
    RF-Track's tracking engine. Forking an already multi-threaded process (a Jupyter
    kernel has its own ZMQ/heartbeat threads; RF-Track itself internally distributes
    tracking across threads, per the RF-Track manual) is a well-documented crash/hang
    hazard -- a lock held by any thread other than the calling one at the instant of
    `fork()` is inherited already-locked, forever, in the child, with consequences that
    land at an essentially random point in the run. That was the root cause of the
    intermittent kernel crashes at random progress percentages this project saw. This
    thread-only version never forks anything, and deliberately avoids numpy calls in
    the loop body (plain `math.exp`, not `np.exp`) as further insurance against any
    incidental interaction with a BLAS/OpenMP thread pool from a background thread.
    """
    start_s = time.time()
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    if tqdm is None:
        last_print = -1
        while not stop_event.is_set():
            elapsed_s = time.time() - start_s
            pct = int(_progress_proxy_pct(elapsed_s, est_for_proxy_s))
            if pct != last_print:
                print(f"tracking {pct}% | elapsed={elapsed_s:,.1f}s est={float(est_for_proxy_s):,.1f}s", flush=True)
                last_print = pct
            time.sleep(max(0.1, float(poll_interval_s)))
        print("tracking 100%", flush=True)
        return

    # tqdm.auto already renders an ipywidgets bar in a live Jupyter kernel and a plain
    # text bar otherwise -- this is the "smart progress bar by default in auto mode"
    # with no separate notebook-vs-terminal branching needed on our side.
    with tqdm(total=100, desc="tracking", unit="%", leave=True) as tbar:
        while not stop_event.is_set():
            elapsed_s = time.time() - start_s
            tbar.n = int(_progress_proxy_pct(elapsed_s, est_for_proxy_s))
            tbar.set_postfix_str(f"elapsed={elapsed_s:,.1f}s est={float(est_for_proxy_s):,.1f}s")
            tbar.refresh()
            time.sleep(max(0.1, float(poll_interval_s)))
        tbar.n = 100
        tbar.refresh()


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
    }


def _runtime_key_string(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _runtime_key_hash(runtime_key: str) -> str:
    return hashlib.sha1(runtime_key.encode("utf-8")).hexdigest()[:12]


def _heuristic_runtime_estimate_s(n_particles: int, default_s: float = 300.0) -> float:
    """Progress-bar time estimate for a run with no exact cache match, scaled by particle count
    (the dominant cost driver) via the median per-particle time across cached runs."""
    per_particle: List[float] = []
    for key, val in _TRANSPORT_RUNTIME_HISTORY.items():
        if not (np.isfinite(val) and val > 0.0):
            continue
        try:
            n_k = int(json.loads(key).get("n_particles", 0))
        except Exception:
            continue
        if n_k > 0:
            per_particle.append(val / n_k)
    if not per_particle:
        return float(default_s)
    return max(1.0, float(np.median(per_particle)) * max(1, int(n_particles)))


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
    except Exception:
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
    """Cold emission (no transverse thermal momentum)."""
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

    Ez0 = float(np.real(Ez0_phasor_axis * np.exp(1j * phi_rad)))
    beta_enh = float(params.beta_enh) if params.beta_enh is not None else float(params.beta_field)
    dphi = schottky_delta_phi_eV(Ez0, beta=beta_enh)
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

    return B0, info


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
        F0 = beta_enh * abs(float(np.real(Ez0_phasor_axis * np.exp(1j * phi_rad))))
        if params.emission_law == "unified":
            J0_u, n_t, J_th_t, J_fe_t = J_unified(np.array([F0]), params.cathode_T_K, params.work_function_eV)
            J0 = float(J0_u[0])
        elif params.emission_law == "RD_schottky":
            J0 = float(J_rld_schottky(np.array([F0]), params.cathode_T_K, params.work_function_eV)[0])
            n_t = None
            J_th_t = None
            J_fe_t = None
        else:
            raise ValueError(f"Unknown emission_law: {params.emission_law}")

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
    F_t = beta_enh * np.abs(Ez_t)

    if params.emission_law == "unified":
        J_t, n_t, J_th_t, J_fe_t = J_unified(F_t, params.cathode_T_K, params.work_function_eV)
    elif params.emission_law == "RD_schottky":
        J_t = J_rld_schottky(F_t, params.cathode_T_K, params.work_function_eV)
        n_t = None
        J_th_t = None
        J_fe_t = None
    else:
        raise ValueError(f"Unknown emission_law: {params.emission_law}")

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
    cathode_radius_mm: float,
    pz0_MeV_c: float,
    q_total_C: float = 1e-12,
    rng: Optional[np.random.Generator] = None,
    refine: bool = True,
    refine_xatol_deg: float = 0.05,
    refine_maxiter: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast phase scan (on-axis, cold launch).

    Runs a coarse scan over `phase_rel_deg` (kept full-range by default so a
    change in the input field maps is still caught), then, unless
    `refine=False`, brackets the coarse maximum and refines it with a bounded
    scalar search. The refined point is folded back into the returned arrays,
    so `np.max(pz_mean)` (as used by `veff_from_phase_scan_pz`) reflects the
    refined crest rather than the coarse grid resolution. This lets the coarse
    grid `phase_rel_deg` be much sparser than a fine grid would need to be,
    without losing crest accuracy.
    """
    z_span_mm = abs(float(vol_params.z_max_m) - float(vol_params.z_min_m)) * 1e3
    phase_scan_tmax_mm = max(60.0, 2.5 * z_span_mm)

    vol_params_fast = vol_params.replace(
        sc_enabled=False,
        beam_loading_enabled=False,
        beam_loading_verbose=False,
        t_max_mm=min(float(getattr(vol_params, "t_max_mm", 2000.0)), float(phase_scan_tmax_mm)),
    )

    def _mean_pz_at(phi_rel: float) -> float:
        phi_abs = (float(phi_rel) + float(transport_phase_deg)) % 360.0
        V = build_volume(rft, Er_grid, Ez_grid, phi_abs, vol_params_fast)
        B0 = build_bunch_simple(rft, n_particles, cathode_radius_mm, pz0_MeV_c, q_total_C, rng=rng)
        Bout = V.track(B0)
        Mf = Bout.get_phase_space()
        if Mf.shape[0] == 0:
            return np.nan
        return float(np.mean(Mf[:, 5]))

    phi_rel_coarse = np.asarray(phase_rel_deg, dtype=float)
    phase_scan = []
    for phi in phi_rel_coarse:
        pz = _mean_pz_at(float(phi))
        phi_abs = (float(phi) + float(transport_phase_deg)) % 360.0
        n_ok = 0 if np.isnan(pz) else int(n_particles)
        phase_scan.append((float(phi), float(phi_abs), pz, n_ok))

    if refine and np.any(np.isfinite([row[2] for row in phase_scan])):
        pz_coarse = np.array([row[2] for row in phase_scan], dtype=float)
        i_max = int(np.nanargmax(pz_coarse))
        lo = phi_rel_coarse[max(i_max - 1, 0)]
        hi = phi_rel_coarse[min(i_max + 1, len(phi_rel_coarse) - 1)]
        if hi > lo:
            res = minimize_scalar(
                lambda phi: -_mean_pz_at(phi),
                bounds=(float(lo), float(hi)),
                method="bounded",
                options={"xatol": float(refine_xatol_deg), "maxiter": int(refine_maxiter)},
            )
            phi_rel_refined = float(res.x)
            pz_refined = -float(res.fun)
            phi_abs_refined = (phi_rel_refined + float(transport_phase_deg)) % 360.0
            phase_scan.append((phi_rel_refined, phi_abs_refined, pz_refined, int(n_particles)))

    phase_scan = np.array(sorted(phase_scan, key=lambda row: row[0]), dtype=float)
    phi_rel = phase_scan[:, 0]
    phi_abs = phase_scan[:, 1]
    pz_mean = phase_scan[:, 2]
    return phase_scan, phi_rel, phi_abs, pz_mean


def run_transport_with_progress(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    Ez0_phasor_axis: complex,
    vol_params: VolumeBuildParams,
    emission: EmissionParams,
    tracking: TrackingParams,
    diagnostics: DiagnosticsParams | None = None,
    progress_bar: bool = True,
    use_coarse_progress_proxy: bool = True,
    poll_interval_s: float = 0.5,
    timing_diagnostics: bool = False,
    slow_step_warn_s: float = 20.0,
    rng: Optional[np.random.Generator] = None,
    on_screen: Optional[Callable[[int, float, Dict[str, float]], None]] = None,
    progress_backend: Literal["thread", "spawn"] = "thread",
):
    """Run transport with staged text and a single tracking progress bar.

    `progress_backend` picks how the elapsed-time-based progress percentage (see
    `_progress_proxy_pct`) is printed while the single blocking RF-Track tracking call runs.

    Neither backend shows live progress inside a Jupyter cell during a long tracking call, with
    or without the deflection magnet: RF-Track holds the Python GIL for the whole call, which
    also blocks ipykernel's own output relay. With the magnet on, `DeflectionField.get_field()`'s
    Python callback lets a few ticks through near the start; then the bar freezes until the call
    returns. With the magnet off there is no such callback, so it freezes from the start.

    - `"thread"` (default): in-process daemon thread (`_elapsed_progress_worker`), rendered via
      `tqdm.auto`. No extra process needed; degrades the same way `"spawn"` does in a notebook.
    - `"spawn"`: a separate OS process (`progress_worker.spawn_progress_target`) that keeps
      ticking regardless of the parent's GIL state. Useful for unattended CLI/SLURM runs writing
      to a plain log file (no notebook relay involved there); inside a notebook cell it degrades
      the same as `"thread"`.

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
    runtime_key = _runtime_key_string(runtime_payload)
    runtime_key_hash = _runtime_key_hash(runtime_key)
    est_s = _TRANSPORT_RUNTIME_HISTORY.get(runtime_key, None)
    heuristic_est_s = _heuristic_runtime_estimate_s(int(tracking.n_particles))

    settings_line = (
        f"Tracking settings | N={int(tracking.n_particles):,} | dt_mm={float(getattr(vol_params_eff, 'dt_mm', np.nan)):.4g} "
        f"| sc_dt_mm={float(getattr(vol_params_eff, 'sc_dt_mm', np.nan)):.4g} "
        f"| emission_sc_steps={int(getattr(vol_params_eff, 'emission_nsteps', 0))} "
        f"| screens={len(z_snaps)} | sc={'on' if bool(getattr(vol_params_eff, 'sc_enabled', False)) else 'off'} "
        f"| bl={'on' if bool(getattr(vol_params_eff, 'beam_loading_enabled', False)) else 'off'} "
        f"| mode=elapsed | key={runtime_key_hash}"
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

    est_for_proxy = None
    if est_s is not None and np.isfinite(est_s) and est_s > 0:
        est_for_proxy = float(est_s)
    elif use_coarse_progress_proxy:
        est_for_proxy = float(heuristic_est_s)
    else:
        est_for_proxy = max(1.0, float(heuristic_est_s))

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
        if diagnostics.use_transport_table_summary:
            tt_dt = float(diagnostics.transport_table_dt_mm) if diagnostics.transport_table_dt_mm is not None else float(vol_params_track.dt_mm)
            Bout, T, V = track_volume_transport_table(
                rft,
                Er_grid,
                Ez_grid,
                tracking.phi_deg,
                vol_params_track,
                B0,
                tt_dt_mm=tt_dt,
                table_fmt=str(diagnostics.transport_table_fmt),
                return_volume=True,
            )
            return Bout, [], V, T
        V = build_volume(rft, Er_grid, Ez_grid, tracking.phi_deg, vol_params_track)
        return V.track(B0), [], V, None

    global _ACTIVE_PROGRESS_STOP_EVENT, _ACTIVE_PROGRESS_THREAD, _ACTIVE_PROGRESS_PROCESS

    progress_enabled = bool(progress_bar)
    backend = str(progress_backend).strip().lower()

    if progress_enabled and backend == "spawn":
        # Separate OS process, not a thread -- see `run_transport_with_progress`'s docstring and
        # `progress_worker.py`'s module docstring for why. `progress_worker` deliberately lives
        # outside this package so spawning it doesn't reimport RF_Track.
        import progress_worker

        _clear_active_progress_process(timeout_s=0.25)
        ctx = mp.get_context("spawn")
        progress_process = ctx.Process(
            target=progress_worker.spawn_progress_target,
            args=(time.time(), est_for_proxy, float(poll_interval_s)),
            name="rftrack-progress-spawn",
            daemon=True,
        )
        progress_process.start()
        _ACTIVE_PROGRESS_PROCESS = progress_process

        t_solver_s = time.time()
        try:
            run_out = _run_tracking_once()
        finally:
            _stop_progress_process(progress_process, timeout_s=2.0)
            if progress_process is _ACTIVE_PROGRESS_PROCESS:
                _ACTIVE_PROGRESS_PROCESS = None
        if len(run_out) == 3:
            Bout, snaps, V = run_out
            transport_table = None
        else:
            Bout, snaps, V, transport_table = run_out
        solver_elapsed_s = time.time() - t_solver_s
        _diag(f"RF-Track solver return: {solver_elapsed_s:.2f} s")
        _warn_slow("RF-Track solver return", solver_elapsed_s)
    elif progress_enabled:
        # In-process daemon thread (the default) -- see `run_transport_with_progress`'s docstring
        # for why this and `"spawn"` degrade the same way inside a notebook cell. See
        # `_elapsed_progress_worker`'s docstring for why this is a thread and not a fork()ed
        # subprocess (a fork-based design was the root cause of intermittent crashes).
        _clear_active_progress_worker(timeout_s=0.25)
        progress_stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=_elapsed_progress_worker,
            args=(progress_stop_event, est_for_proxy, float(poll_interval_s)),
            name="rftrack-progress",
            daemon=True,
        )
        progress_thread.start()
        _ACTIVE_PROGRESS_STOP_EVENT = progress_stop_event
        _ACTIVE_PROGRESS_THREAD = progress_thread

        t_solver_s = time.time()
        try:
            run_out = _run_tracking_once()
        finally:
            _stop_progress_worker(progress_stop_event, progress_thread, timeout_s=2.0)
            if progress_thread is _ACTIVE_PROGRESS_THREAD:
                _ACTIVE_PROGRESS_STOP_EVENT = None
                _ACTIVE_PROGRESS_THREAD = None
        if len(run_out) == 3:
            Bout, snaps, V = run_out
            transport_table = None
        else:
            Bout, snaps, V, transport_table = run_out
        solver_elapsed_s = time.time() - t_solver_s
        _diag(f"RF-Track solver return: {solver_elapsed_s:.2f} s")
        _warn_slow("RF-Track solver return", solver_elapsed_s)
    else:
        t_solver_s = time.time()
        run_out = _run_tracking_once()
        if len(run_out) == 3:
            Bout, snaps, V = run_out
            transport_table = None
        else:
            Bout, snaps, V, transport_table = run_out
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
        transport_table=transport_table,
        lost_table=lost_table,
        particle_classes=classes,
    )
    return result, {
        "track_elapsed_s": float(track_elapsed_s),
        "track_estimate_s": float(_TRANSPORT_RUNTIME_HISTORY[runtime_key]),
        "runtime_key_hash": runtime_key_hash,
        "progress_mode": "elapsed",
        "progress_enabled": bool(progress_enabled),
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
