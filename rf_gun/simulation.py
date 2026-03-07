"""Simulation pipeline helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Literal
import time
import os
import json
import hashlib
import multiprocessing as mp
from pathlib import Path

import numpy as np

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
    VolumeBuildParams,
    ScreenBuildParams,
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


@dataclass(frozen=True)
class TrackingParams:
    phi_deg: float
    n_particles: int
    z_screens_m: Optional[Sequence[float]] = None
    phase_fmt: str = "%X %Px %Y %Py %Z %Pz"
    screen_width_mm: float | None = None
    screen_height_mm: float | None = None
    screen_time_window_mm_c: float | None = None
    screen_t0_mode: Literal["unset", "sync_to_first_crossing", "manual"] = "unset"
    screen_t0_manual_mm_c: float = 0.0
    screen_log: bool = False


@dataclass
class SimulationResult:
    B0: Any
    Bout: Any
    thermo_info: Dict[str, Any]
    M_snaps: List[np.ndarray]
    z_snaps: List[float]
    I_snaps: List[Any]


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


def _elapsed_progress_worker(
    stop_event,
    est_for_proxy_s: float,
    poll_interval_s: float,
):
    try:
        from tqdm.auto import tqdm
    except Exception:
        return

    start_s = time.time()
    with tqdm(total=100, desc="tracking", unit="%", leave=False) as tbar:
        while not stop_event.is_set():
            proxy = min(98.0, 100.0 * (time.time() - start_s) / max(1e-9, float(est_for_proxy_s)))
            tbar.n = int(proxy)
            tbar.refresh()

            time.sleep(max(0.1, float(poll_interval_s)))

        tbar.n = 100
        tbar.refresh()


def _runtime_key_payload(vol_params_eff: VolumeBuildParams, tracking: TrackingParams, n_screens: int) -> Dict[str, Any]:
    return {
        "n_particles": int(tracking.n_particles),
        "dt_mm": float(getattr(vol_params_eff, "dt_mm", np.nan)),
        "sc_dt_mm": float(getattr(vol_params_eff, "sc_dt_mm", np.nan)),
        "emission_nsteps": int(getattr(vol_params_eff, "emission_nsteps", 0)),
        "emission_range": float(getattr(vol_params_eff, "emission_range", np.nan)),
        "n_screens": int(n_screens),
        "sc_enabled": bool(getattr(vol_params_eff, "sc_enabled", False)),
        "beam_loading_enabled": bool(getattr(vol_params_eff, "beam_loading_enabled", False)),
        "phi_deg": float(getattr(tracking, "phi_deg", np.nan)),
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


def _snapshot_stats(M: np.ndarray) -> Dict[str, float]:
    if M.size == 0:
        return {"N": 0, "mean_pz": np.nan, "N_lost_since_prev": np.nan}
    return {
        "N": int(M.shape[0]),
        "mean_pz": float(np.mean(M[:, 5])) if M.shape[1] > 5 else np.nan,
        "N_lost_since_prev": np.nan,
    }


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
        t_emit_s = thermo_info.get("t_emit_s", None)
        if t_emit_s is not None:
            t_emit_s = np.asarray(t_emit_s, dtype=float).reshape(-1)
            t_emit_s = t_emit_s[np.isfinite(t_emit_s)]
            if t_emit_s.size:
                tinj_mm_c = float(np.min(t_emit_s) * c * 1e3)

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
    """Thermionic emission with Richardson + Schottky current."""
    rng = np.random.default_rng() if rng is None else rng

    phi_rad = np.deg2rad(phi_deg)
    Ez0 = float(np.real(Ez0_phasor_axis * np.exp(1j * phi_rad)))

    area_m2 = np.pi * (params.cathode_radius_mm * 1e-3) ** 2
    area_cm2 = area_m2 * 1e4

    beta_enh = float(params.beta_enh) if params.beta_enh is not None else float(params.beta_field)
    dphi = schottky_delta_phi_eV(Ez0, beta=beta_enh)
    phi_eff = max(params.work_function_eV - dphi, 0.0)
    J0 = richardson_J_Apm2(params.cathode_T_K, phi_eff)
    I0 = J0 * area_m2

    t_emit_s = None
    t_s = None
    Ez_t = None
    dphi_t = None
    phi_eff_t = None
    J_t = None
    I_t = None
    Q_cum = None
    tau_s = None
    I_avg = None
    I_peak = None
    Q_total_C = 0.0

    if params.time_dependent:
        f_hz = float(f_hz)
        T = 1.0 / f_hz
        omega = 2.0 * np.pi * f_hz

        phase_range_deg = max(float(params.emission_phase_range_deg), 0.0)
        tau_s = (phase_range_deg / 360.0) * T

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
        if J_th_t is not None and J_fe_t is not None:
            R_t = J_fe_t / np.maximum(J_th_t, 1e-300)
        else:
            R_t = None

        dt = t_s[1] - t_s[0] if t_s.size > 1 else 0.0
        Q_cum = np.zeros_like(t_s)
        if t_s.size > 1:
            Q_cum[1:] = np.cumsum((I_t[:-1] + I_t[1:]) * 0.5) * dt

        Q_total_C = float(Q_cum[-1]) if Q_cum.size else 0.0
        if Q_total_C > 0.0:
            t_emit_s = np.interp(rng.random(n) * Q_total_C, Q_cum, t_s)
        else:
            t_emit_s = np.zeros(n)

        I_peak = float(np.max(I_t)) if I_t is not None and I_t.size else 0.0
        I_avg = float(Q_total_C / tau_s) if tau_s and np.isfinite(tau_s) else 0.0
    else:
        f_hz = float(f_hz)
        T = 1.0 / f_hz
        phase_range_deg = max(float(params.emission_phase_range_deg), 0.0)
        tau_s = (phase_range_deg / 360.0) * T
        F0 = beta_enh * abs(Ez0)
        if params.emission_law == "unified":
            J0_u, n_t, J_th_t, J_fe_t = J_unified(np.array([F0]), params.cathode_T_K, params.work_function_eV)
            J0 = float(J0_u[0])
        else:
            J0 = float(J_rld_schottky(np.array([F0]), params.cathode_T_K, params.work_function_eV)[0])
            n_t = None
            J_th_t = None
            J_fe_t = None
        R_t = (J_fe_t / np.maximum(J_th_t, 1e-300)) if J_th_t is not None and J_fe_t is not None else None
        I_avg = J0 * area_m2
        I_peak = I_avg
        Q_total_C = float(I_avg * tau_s) if np.isfinite(tau_s) else 0.0
        if tau_s > 0.0:
            t_emit_s = rng.uniform(0.0, tau_s, size=n)
        else:
            t_emit_s = np.zeros(n)

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
        px,
        py,
        pz,
        params.roughness.Ra_um,
        params.roughness.Re_um,
        rng=rng,
    )
    px_rms = float(np.std(px)) if px.size else np.nan
    py_rms = float(np.std(py)) if py.size else np.nan

    if params.pz_model == "flux":
        print(
            f"Normal energy: <eps_z>={mean_eps_eV:.4f} eV (expected {exp_eps_eV:.4f} eV)"
        )

    if np.isfinite(tau_s) and t_emit_s is not None:
        t = t_emit_s * c * 1e3
    else:
        t = np.zeros(n)

    z = np.zeros(n)

    M = np.column_stack([x, px, y, py, z, pz])

    N_real = float(abs(Q_total_C) / q_e) if Q_total_C > 0.0 else 0.0
    B0 = rft.Bunch6dT(ME_MEV, N_real, -1.0, M)
    if hasattr(B0, "set_t0"):
        B0.set_t0(t)

    info = {
        "Ez0": Ez0,
        "dphi_eV": dphi,
        "phi_eff_eV": phi_eff,
        "J_Apm2": J0,
        "I_A": I0,
        "I_avg_A": I_avg,
        "I_peak_A": I_peak,
        "tau_ns": float(tau_s * 1e9) if np.isfinite(tau_s) else np.inf,
        "tau_s": float(tau_s) if np.isfinite(tau_s) else np.inf,
        "Q_total_C": float(Q_total_C),
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
        "t_s": t_s,
        "Ez_t": Ez_t,
        "F_t": beta_enh * np.abs(Ez_t) if Ez_t is not None else None,
        "dphi_eV_t": dphi_t,
        "phi_eff_eV_t": phi_eff_t,
        "J_Apm2_t": J_t,
        "J_th_Apm2_t": J_th_t,
        "J_fe_Apm2_t": J_fe_t,
        "R_t": R_t,
        "n_t": n_t,
        "n_at_peak": float(n_t[np.argmax(J_t)]) if n_t is not None and J_t is not None and J_t.size else None,
        "n_at_peak_field": float(n_t[np.argmax(J_fe_t)]) if n_t is not None and J_fe_t is not None and J_fe_t.size else None,
        "I_A_t": I_t,
        "area_m2": float(area_m2),
        "area_cm2": float(area_cm2),
        "Q_cum_C": Q_cum,
        "t_emit_s": t_emit_s,
        "has_t0": hasattr(B0, "set_t0") or hasattr(B0, "get_t0"),
    }
    return B0, info


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast phase scan (on-axis, cold launch)."""
    phase_scan = []
    vol_params_fast = vol_params.replace(
        sc_enabled=False,
        beam_loading_enabled=False,
        beam_loading_verbose=False,
    )

    for phi in phase_rel_deg:
        phi_abs = (float(phi) + float(transport_phase_deg)) % 360.0
        V = build_volume(rft, Er_grid, Ez_grid, phi_abs, vol_params_fast)
        B0 = build_bunch_simple(rft, n_particles, cathode_radius_mm, pz0_MeV_c, q_total_C)
        Bout = V.track(B0)
        Mf = Bout.get_phase_space()
        if Mf.shape[0] == 0:
            phase_scan.append((float(phi), float(phi_abs), np.nan, 0))
            continue
        pz = Mf[:, 5]
        phase_scan.append((float(phi), float(phi_abs), float(np.mean(pz)), int(Mf.shape[0])))

    phase_scan = np.array(phase_scan, dtype=float)
    phi_rel = phase_scan[:, 0]
    phi_abs = phase_scan[:, 1]
    pz_mean = phase_scan[:, 2]
    return phase_scan, phi_rel, phi_abs, pz_mean


def run_transport(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    Ez0_phasor_axis: complex,
    vol_params: VolumeBuildParams,
    emission: EmissionParams,
    tracking: TrackingParams,
    on_screen: Optional[Callable[[int, float, Dict[str, float]], None]] = None,
) -> SimulationResult:
    """Run a thermionic transport simulation with optional screens."""
    B0, thermo_info = build_bunch_thermionic(
        rft,
        tracking.n_particles,
        tracking.phi_deg,
        f_hz=vol_params.f_hz,
        params=emission,
        Ez0_phasor_axis=Ez0_phasor_axis,
    )
    vol_params_eff = _resolve_beam_loading_tinj(vol_params, B0, thermo_info)

    z_snaps = []
    if tracking.z_screens_m is not None:
        z_snaps = list(tracking.z_screens_m)
    if len(z_snaps) > 0:
        screen_params = _build_screen_params(tracking)
        Bout, snaps = track_volume_with_screens(
            rft,
            Er_grid,
            Ez_grid,
            tracking.phi_deg,
            vol_params_eff,
            B0,
            z_snaps,
            screen_params=screen_params,
        )
    else:
        V = build_volume(rft, Er_grid, Ez_grid, tracking.phi_deg, vol_params_eff)
        Bout = V.track(B0)
        snaps = []

    M_snaps = [
        np.array(s.get_phase_space(tracking.phase_fmt, "all"), copy=True) for s in snaps
    ] if snaps else []
    I_snaps = [s.get_info() if hasattr(s, "get_info") else None for s in snaps] if snaps else []

    if z_snaps and np.any(np.diff(np.asarray(z_snaps, dtype=float)) < 0.0):
        print("Warning: z_screens_m is not monotonic increasing.")

    if on_screen is not None and M_snaps:
        prev_n = None
        for i, (z_m, M) in enumerate(zip(z_snaps, M_snaps)):
            stats = _snapshot_stats(M)
            if prev_n is not None and np.isfinite(prev_n):
                stats["N_lost_since_prev"] = float(prev_n - stats["N"])
            prev_n = float(stats["N"])
            on_screen(i, float(z_m), stats)

    return SimulationResult(
        B0=B0,
        Bout=Bout,
        thermo_info=thermo_info,
        M_snaps=M_snaps,
        z_snaps=z_snaps,
        I_snaps=I_snaps,
    )


def run_transport_with_progress(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    Ez0_phasor_axis: complex,
    vol_params: VolumeBuildParams,
    emission: EmissionParams,
    tracking: TrackingParams,
    use_coarse_progress_proxy: bool = True,
    poll_interval_s: float = 0.5,
    on_screen: Optional[Callable[[int, float, Dict[str, float]], None]] = None,
):
    """Run transport with staged text and a single tracking progress bar.

    Returns:
        (SimulationResult, stats_dict)
        where stats_dict has `track_elapsed_s`, `track_estimate_s`, `runtime_key_hash`.
    """

    stages = [
        "Build thermionic bunch",
        "Prepare tracking and screens",
        "Run RF-Track transport",
        "Extract phase-space snapshots",
        "Extract screen info",
    ]

    def _set_stage(i: int):
        print(f"{i} / {len(stages)}: {stages[i-1]}")

    print("---- Simulation Start ----")
    _set_stage(1)
    B0, thermo_info = build_bunch_thermionic(
        rft,
        tracking.n_particles,
        tracking.phi_deg,
        f_hz=vol_params.f_hz,
        params=emission,
        Ez0_phasor_axis=Ez0_phasor_axis,
    )
    vol_params_eff = _resolve_beam_loading_tinj(vol_params, B0, thermo_info)

    _set_stage(2)
    z_snaps = list(tracking.z_screens_m) if tracking.z_screens_m is not None else []

    runtime_payload = _runtime_key_payload(vol_params_eff, tracking, len(z_snaps))
    runtime_key = _runtime_key_string(runtime_payload)
    runtime_key_hash = _runtime_key_hash(runtime_key)
    est_s = _TRANSPORT_RUNTIME_HISTORY.get(runtime_key, None)
    history_vals = [float(v) for v in _TRANSPORT_RUNTIME_HISTORY.values() if np.isfinite(v) and float(v) > 0.0]
    heuristic_est_s = float(np.median(history_vals)) if history_vals else 300.0

    settings_line = (
        f"Tracking settings | N={int(tracking.n_particles):,} | dt_mm={float(getattr(vol_params_eff, 'dt_mm', np.nan)):.4g} "
        f"| sc_dt_mm={float(getattr(vol_params_eff, 'sc_dt_mm', np.nan)):.4g} "
        f"| emission_sc_steps={int(getattr(vol_params_eff, 'emission_nsteps', 0))} "
        f"| screens={len(z_snaps)} | sc={'on' if bool(getattr(vol_params_eff, 'sc_enabled', False)) else 'off'} "
        f"| bl={'on' if bool(getattr(vol_params_eff, 'beam_loading_enabled', False)) else 'off'} "
        f"| mode=elapsed | key={runtime_key_hash}"
    )
    print(settings_line)
    print("(emission_sc_steps = number of SC kicks/substeps applied during emission)")

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
        )

    if est_s is not None and np.isfinite(est_s):
        print(f"Last-run estimate for this key: {float(est_s):.2f} s")
    else:
        print(f"No exact runtime estimate for this key yet; using heuristic estimate: {heuristic_est_s:.2f} s")

    _set_stage(3)
    track_start_s = time.time()

    est_for_proxy = None
    if est_s is not None and np.isfinite(est_s) and est_s > 0:
        est_for_proxy = float(est_s)
    elif use_coarse_progress_proxy:
        est_for_proxy = float(heuristic_est_s)
    else:
        est_for_proxy = max(1.0, float(heuristic_est_s))

    stop_event = mp.Event()
    progress_proc = mp.Process(
        target=_elapsed_progress_worker,
        args=(stop_event, est_for_proxy, float(poll_interval_s)),
        daemon=True,
    )
    progress_proc.start()

    vol_params_track = vol_params_eff.replace(beam_loading_verbose=False)

    try:
        if len(z_snaps) > 0:
            screen_params = _build_screen_params(tracking)
            Bout, snaps = track_volume_with_screens(
                rft,
                Er_grid,
                Ez_grid,
                tracking.phi_deg,
                vol_params_track,
                B0,
                z_snaps,
                screen_params=screen_params,
            )
        else:
            V = build_volume(rft, Er_grid, Ez_grid, tracking.phi_deg, vol_params_track)
            Bout = V.track(B0)
            snaps = []
    finally:
        stop_event.set()
        progress_proc.join(timeout=2.0)
        if progress_proc.is_alive():
            progress_proc.terminate()
            progress_proc.join(timeout=1.0)

    M_snaps = [
        np.array(s.get_phase_space(tracking.phase_fmt, "all"), copy=True) for s in snaps
    ] if snaps else []
    I_snaps = [s.get_info() if hasattr(s, "get_info") else None for s in snaps] if snaps else []

    track_elapsed_s = time.time() - track_start_s

    if est_s is None:
        _TRANSPORT_RUNTIME_HISTORY[runtime_key] = track_elapsed_s
    else:
        _TRANSPORT_RUNTIME_HISTORY[runtime_key] = 0.7 * float(est_s) + 0.3 * track_elapsed_s
    _save_transport_runtime_history(_TRANSPORT_RUNTIME_HISTORY)

    _set_stage(4)

    if z_snaps and np.any(np.diff(np.asarray(z_snaps, dtype=float)) < 0.0):
        print("Warning: z_screens_m is not monotonic increasing.")

    if M_snaps:
        prev_n = None
        for i, (z_m, M) in enumerate(zip(z_snaps, M_snaps)):
            stats = _snapshot_stats(M)
            if prev_n is not None and np.isfinite(prev_n):
                stats["N_lost_since_prev"] = float(prev_n - stats["N"])
            prev_n = float(stats["N"])
            if on_screen is not None:
                on_screen(i, float(z_m), stats)

    _set_stage(5)
    print("---- Simulation End ----")

    result = SimulationResult(
        B0=B0,
        Bout=Bout,
        thermo_info=thermo_info,
        M_snaps=M_snaps,
        z_snaps=z_snaps,
        I_snaps=I_snaps,
    )
    return result, {
        "track_elapsed_s": float(track_elapsed_s),
        "track_estimate_s": float(_TRANSPORT_RUNTIME_HISTORY[runtime_key]),
        "runtime_key_hash": runtime_key_hash,
        "progress_mode": "elapsed",
        "runtime_cache_file": str(_RUNTIME_HISTORY_CACHE),
    }
