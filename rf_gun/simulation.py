"""Simulation pipeline helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

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
from .rftrack_volume import build_volume, track_volume_with_screens, VolumeBuildParams


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


@dataclass
class SimulationResult:
    B0: Any
    Bout: Any
    thermo_info: Dict[str, Any]
    M_snaps: List[np.ndarray]
    z_snaps: List[float]


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
    vol_params_fast = vol_params.replace(sc_enabled=False)

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

    z_snaps = []
    if tracking.z_screens_m is not None:
        z_snaps = list(tracking.z_screens_m)
    if len(z_snaps) > 0:
        Bout, snaps = track_volume_with_screens(
            rft,
            Er_grid,
            Ez_grid,
            tracking.phi_deg,
            vol_params,
            B0,
            z_snaps,
        )
    else:
        V = build_volume(rft, Er_grid, Ez_grid, tracking.phi_deg, vol_params)
        Bout = V.track(B0)
        snaps = []

    M_snaps = [
        np.array(s.get_phase_space(tracking.phase_fmt, "good"), copy=True) for s in snaps
    ] if snaps else []

    return SimulationResult(
        B0=B0,
        Bout=Bout,
        thermo_info=thermo_info,
        M_snaps=M_snaps,
        z_snaps=z_snaps,
    )
