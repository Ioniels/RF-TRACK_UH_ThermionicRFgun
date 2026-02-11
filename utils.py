# -*- coding: utf-8 -*-
"""
Utilities for RF-Track gun simulations.

Conventions
- x, y in mm
- Px, Py, Pz in MeV/c
- z, s in m (RF-Track Volume longitudinal coordinate)
- t in mm/c (RF-Track convention)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any, List

import numpy as np
from scipy.constants import c, e as q_e, epsilon_0

ME_MEV = 0.51099895  # electron rest energy [MeV]


# ----------------------------- Generic helpers -----------------------------

def kinetic_energy(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """Return kinetic energy [MeV] from momenta [MeV/c]."""
    p2 = px**2 + py**2 + pz**2
    gamma = np.sqrt(1.0 + p2 / ME_MEV**2)
    return (gamma - 1.0) * ME_MEV


def sample_disk(n: int, radius_mm: float, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random distribution over a disk of radius `radius_mm`."""
    if n <= 0 or radius_mm <= 0:
        return np.zeros(max(n, 0)), np.zeros(max(n, 0))
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(n)
    theta = 2.0 * np.pi * rng.random(n)
    r = radius_mm * np.sqrt(u)
    return r * np.cos(theta), r * np.sin(theta)


# ----------------------------- Thermionic emission -----------------------------

A_RICH = 1.20173e6  # Richardson constant [A/m^2/K^2]

def select_iq_snapshots(t_ns: np.ndarray, Ez_rms: np.ndarray, f_hz: float, search_window: int = 60) -> Tuple[int, int, float, float]:
    """Choose two indices (i0, i90) separated by ~T/4 for I/Q phasor reconstruction.

    The score penalizes timing error relative to T/4 and envelope mismatch.
    Returns (i0, i90, dt_error, amplitude_ratio=Ez_rms[i90]/Ez_rms[i0]).
    """
    T_ns = 1e9 / float(f_hz)
    dt_target = 0.25 * T_ns

    i_peak = int(np.argmax(Ez_rms))
    i_lo = max(0, i_peak - int(search_window))
    i_hi = min(len(Ez_rms) - 1, i_peak + int(search_window))

    best = None
    for i0 in range(i_lo, i_hi + 1):
        t0 = float(t_ns[i0])
        t90 = t0 + dt_target
        i90 = int(np.argmin(np.abs(t_ns - t90)))

        dt_err = abs((float(t_ns[i90]) - t0) - dt_target) / dt_target
        amp_ratio = (float(Ez_rms[i90]) / float(Ez_rms[i0])) if Ez_rms[i0] != 0 else np.inf
        amp_err = abs(np.log(amp_ratio)) if np.isfinite(amp_ratio) else np.inf

        score = 3.0 * dt_err + 1.0 * amp_err
        if best is None or score < best[0]:
            best = (score, i0, i90, dt_err, amp_ratio)

    if best is None:
        raise RuntimeError("select_iq_snapshots: empty candidate set")
    return int(best[1]), int(best[2]), float(best[3]), float(best[4])


def build_iq_phasor(field_0: np.ndarray, field_90: np.ndarray, env_0: float, env_90: float, scale: float = 1.0) -> np.ndarray:
    """Complex phasor from two snapshots at 0° and 90°, normalized by the envelope."""
    e0 = field_0 / (env_0 if env_0 != 0 else 1.0)
    e90 = field_90 / (env_90 if env_90 != 0 else 1.0)
    return (e0 - 1j * e90) * float(scale)


def theoretical_energy_gain(Ez_axis_phasor: np.ndarray, z_m: np.ndarray, phi_rad: float) -> float:
    """Energy gain [MeV] from on-axis phasor: ΔW = -e ∫ Re(Ez·e^{iφ}) dz."""
    Ez_real = np.real(Ez_axis_phasor * np.exp(1j * float(phi_rad)))
    dW_J = (-q_e) * np.trapezoid(Ez_real, z_m)
    return float(dW_J / (q_e * 1e6))


def cavity_wavelength(f_hz: float) -> Dict[str, float]:
    """λ, λ/2, λ/4 for a given frequency."""
    lam = c / float(f_hz)
    return {"lambda": float(lam), "lambda/2": float(lam / 2.0), "lambda/4": float(lam / 4.0)}

def schottky_delta_phi_eV(E_Vm: float, beta: float = 1.0) -> float:
    """Schottky lowering Δφ [eV] for a local normal field magnitude |E| [V/m]."""
    E = abs(E_Vm) * beta
    dphi_J = np.sqrt((q_e**3) * E / (4.0 * np.pi * epsilon_0))
    return float(dphi_J / q_e)


def richardson_J_Apm2(T_K: float, phi_eff_eV: float) -> float:
    """Richardson–Dushman current density J [A/m^2]."""
    kB_eV_per_K = 8.617333262e-5
    return float(A_RICH * (T_K**2) * np.exp(-phi_eff_eV / (kB_eV_per_K * T_K)))


def emission_window_from_charge(Q_C: float, I_A: float) -> float:
    """Return emission duration τ [s] needed to emit charge Q at current I."""
    if I_A <= 0.0:
        return np.inf
    return float(Q_C / I_A)


def sample_thermionic_momenta(
    n: int,
    T_K: float,
    pz0_MeV_c: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thermal transverse momenta for a Maxwellian emitter.
    Longitudinal momentum is initialized around pz0_MeV_c.
    """
    rng = np.random.default_rng() if rng is None else rng

    kB_J_per_K = 1.380649e-23
    me_kg = 9.1093837015e-31

    # Non-relativistic thermal velocity scale; ok for cathode emission.
    sigma_v = np.sqrt(kB_J_per_K * T_K / me_kg)  # [m/s]
    sigma_p_SI = me_kg * sigma_v                 # [kg m/s]

    # Convert to MeV/c: 1 MeV/c = (1e6 * e) / c [kg m/s]
    MeV_c_SI = (1e6 * q_e) / c
    sigma_p_MeV_c = sigma_p_SI / MeV_c_SI

    px = rng.normal(0.0, sigma_p_MeV_c, size=n)
    py = rng.normal(0.0, sigma_p_MeV_c, size=n)
    pz = np.full(n, float(pz0_MeV_c))
    return px, py, pz


# ----------------------------- RF-Track setup -----------------------------

@dataclass(frozen=True)
class VolumeBuildParams:
    @staticmethod
    def from_dict(d: dict) -> 'VolumeBuildParams':
        return VolumeBuildParams(**d)

    def replace(self, **kwargs):
        return replace(self, **kwargs)
    
    f_hz: float
    map_z0_m: float  # z of Ez_grid[0,:] [m] (global z_min)
    z_min_m: float
    z_max_m: float
    hr_m: float
    hz_m: float
    dt_mm: float
    ode_algorithm: str = "rk2"
    ode_epsabs: float = 1e-10
    aperture_m: float = 1.0
    t_max_mm: float = 2000.0

    # Field map integration knobs
    fm_nsteps: int = 400
    fm_tt_nsteps: int = 200

    # Optional space charge during emission
    sc_enabled: bool = False
    sc_dt_mm: float = 1.0
    emission_nsteps: int = 1
    emission_range: float = 0.0


def _coerce_volume_params(p):
    """Accept either VolumeBuildParams or a dict with matching keys."""
    if isinstance(p, VolumeBuildParams):
        return p
    if isinstance(p, dict):
        return VolumeBuildParams.from_dict(p)
    raise TypeError(f"Volume params must be VolumeBuildParams or dict, got {type(p)}")


def build_volume(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    phi_deg: float,
    p: VolumeBuildParams,
    add_screens_z_m: Optional[Sequence[float]] = None,
):
    p = _coerce_volume_params(p)
    """
    Construct a Volume containing a single RF_FieldMap_2d and optional Screens.

    Notes
    - Field maps are placed with reference='entrance' at z=0.
    - RF_FieldMap_2d z0 must match `p.map_z0_m` used for the interpolated grid.
    """
    FM = rft.RF_FieldMap_2d(
        Er_grid, Ez_grid,
        0.0, float(p.map_z0_m),   # r0=0, z0=map_z0_m
        float(p.hr_m), float(p.hz_m),
        -1, float(p.f_hz), +1,
        1.0, 1.0,
    )

    if hasattr(FM, "set_tt_nsteps"):
        FM.set_tt_nsteps(int(p.fm_tt_nsteps))
    if hasattr(FM, "set_nsteps"):
        FM.set_nsteps(int(p.fm_nsteps))
    if hasattr(FM, "set_odeint_algorithm"):
        FM.set_odeint_algorithm(p.ode_algorithm)
    if hasattr(FM, "set_odeint_epsabs"):
        FM.set_odeint_epsabs(p.ode_epsabs)

    FM.set_phid(float(phi_deg))
    if hasattr(FM, "set_t0"):
        FM.set_t0(0.0)

    V = rft.Volume()
    V.add(FM, 0.0, 0.0, 0.0, "entrance")

    # Diagnostics: Screens (captured in the screen frame at traversal)
    if add_screens_z_m:
        for z in add_screens_z_m:
            S = rft.Screen()
            V.add(S, 0.0, 0.0, float(z), "entrance")

    V.dt_mm = float(p.dt_mm)
    V.odeint_algorithm = p.ode_algorithm
    V.odeint_epsabs = float(p.ode_epsabs)
    V.set_s0(float(p.z_min_m))
    V.set_s1(float(p.z_max_m))
    V.set_aperture(float(p.aperture_m), float(p.aperture_m), "circular")
    V.t_max_mm = float(p.t_max_mm)

    if p.sc_enabled:
        # Enable space charge if the RF-Track build exposes the hooks.
        for method_name in (
            "set_sc_on",
            "enable_sc",
            "enable_space_charge",
            "set_space_charge",
        ):
            method = getattr(V, method_name, None)
            if callable(method):
                method(True)
        for attr_name in ("sc_on", "sc_enabled", "sc_enable", "space_charge"):
            if hasattr(V, attr_name):
                setattr(V, attr_name, True)

        if hasattr(V, "sc_dt_mm"):
            V.sc_dt_mm = float(p.sc_dt_mm)
        if hasattr(V, "emission_nsteps"):
            V.emission_nsteps = int(p.emission_nsteps)
        if hasattr(V, "emission_range"):
            V.emission_range = float(p.emission_range)

    return V


def find_Ez_axis_phasor_at_z0(Ez_grid: np.ndarray, z_grid_m: np.ndarray, z0_m: float = 0.0) -> complex:
    """Return on-axis Ez phasor at z≈z0 (r=0 index)."""
    iz0 = int(np.argmin(np.abs(z_grid_m - z0_m)))
    return complex(Ez_grid[iz0, 0])


DEFAULT_CATHODE_RADIUS_MM = 1.0
DEFAULT_PZ0_MEV_C = 0.1
DEFAULT_Q_TOTAL_C = 1e-9  # 1 nC total charge
def build_bunch_simple(
        
    rft,
    n: int,
    cathode_radius_mm: float = DEFAULT_CATHODE_RADIUS_MM,
    pz0_MeV_c: float = DEFAULT_PZ0_MEV_C,
    q_total_C: float = DEFAULT_Q_TOTAL_C,
    rng: Optional[np.random.Generator] = None,
):
    """Cold emission (no transverse thermal momentum)."""
    rng = np.random.default_rng() if rng is None else rng
    x, y = sample_disk(n, cathode_radius_mm, rng=rng)
    px = np.zeros(n)
    py = np.zeros(n)
    z = np.zeros(n)  # mm in phase space convention? use 0
    pz = np.full(n, float(pz0_MeV_c))
    t = np.zeros(n)

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
    cathode_radius_mm: float,
    cathode_T_K: float,
    work_function_eV: float,
    beta_field: float,
    Q_target_C: float,
    pz0_MeV_c: float,
    Ez0_phasor_axis: complex,
    time_dependent: bool = True,
    samples_per_period: int = 200,
    max_periods: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Thermionic emission with Richardson + Schottky current.

    The macro-particle charge matches Q_target_C. Emission times are sampled from a
    time-dependent current I(t) derived from Ez(z=0, t) unless time_dependent=False,
    in which case emission times are uniform over τ_emit.
    """
    rng = np.random.default_rng() if rng is None else rng

    # Field at cathode for the selected RF phase
    phi_rad = np.deg2rad(phi_deg)
    Ez0 = float(np.real(Ez0_phasor_axis * np.exp(1j * phi_rad)))  # [V/m]

    area_m2 = np.pi * (cathode_radius_mm * 1e-3)**2

    dphi = schottky_delta_phi_eV(Ez0, beta=beta_field)
    phi_eff = max(work_function_eV - dphi, 0.0)
    J0 = richardson_J_Apm2(cathode_T_K, phi_eff)  # [A/m^2]
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
    n_periods_used = None
    periods_capped = False

    if time_dependent:
        f_hz = float(f_hz)
        T = 1.0 / f_hz
        omega = 2.0 * np.pi * f_hz

        samples_per_period = max(int(samples_per_period), 10)
        max_periods = max(int(max_periods), 1)

        t_period = np.linspace(0.0, T, samples_per_period + 1)
        Ez_period = np.real(Ez0_phasor_axis * np.exp(1j * (omega * t_period + phi_rad)))
        Eabs_period = np.abs(Ez_period)
        dphi_period = np.sqrt((q_e**3) * Eabs_period / (4.0 * np.pi * epsilon_0)) / q_e
        phi_eff_period = np.maximum(work_function_eV - dphi_period, 0.0)

        kB_eV_per_K = 8.617333262e-5
        J_period = A_RICH * (cathode_T_K**2) * np.exp(-phi_eff_period / (kB_eV_per_K * cathode_T_K))
        I_period = J_period * area_m2
        I_avg = float(np.trapezoid(I_period, t_period) / T) if np.any(I_period) else 0.0

        if I_avg > 0.0:
            n_periods = int(np.ceil(Q_target_C / (I_avg * T)))
        else:
            n_periods = 1
        n_periods = max(1, n_periods)

        if n_periods > max_periods:
            n_periods = max_periods
            periods_capped = True

        n_periods_used = n_periods
        t_s = np.linspace(0.0, n_periods * T, n_periods * samples_per_period + 1)
        Ez_t = np.real(Ez0_phasor_axis * np.exp(1j * (omega * t_s + phi_rad)))
        Eabs_t = np.abs(Ez_t)
        dphi_t = np.sqrt((q_e**3) * Eabs_t / (4.0 * np.pi * epsilon_0)) / q_e
        phi_eff_t = np.maximum(work_function_eV - dphi_t, 0.0)
        J_t = A_RICH * (cathode_T_K**2) * np.exp(-phi_eff_t / (kB_eV_per_K * cathode_T_K))
        I_t = J_t * area_m2

        dt = t_s[1] - t_s[0]
        Q_cum = np.zeros_like(t_s)
        if t_s.size > 1:
            Q_cum[1:] = np.cumsum((I_t[:-1] + I_t[1:]) * 0.5) * dt

        Q_end = float(Q_cum[-1]) if Q_cum.size else 0.0
        if Q_end > 0.0:
            Q_use = min(float(Q_target_C), Q_end)
            t_emit_s = np.interp(rng.random(n) * Q_use, Q_cum, t_s)
            if Q_use < float(Q_target_C):
                tau_s = float(t_s[-1])
            else:
                tau_s = float(np.interp(Q_use, Q_cum, t_s))
        else:
            t_emit_s = np.zeros(n)
            tau_s = np.inf

        I_peak = float(np.max(I_t)) if I_t is not None and I_t.size else 0.0
    else:
        I_avg = I0
        I_peak = I0
        tau_s = emission_window_from_charge(Q_target_C, I0)
        if np.isfinite(tau_s):
            t_emit_s = rng.uniform(0.0, tau_s, size=n)
        else:
            t_emit_s = np.zeros(n)

    # Transverse phase space
    x, y = sample_disk(n, cathode_radius_mm, rng=rng)
    px, py, pz = sample_thermionic_momenta(n, cathode_T_K, pz0_MeV_c, rng=rng)

    # Emission time distribution (mm/c)
    if np.isfinite(tau_s) and t_emit_s is not None:
        t = t_emit_s * c * 1e3  # [mm/c]
    else:
        t = np.zeros(n)

    z = np.zeros(n)

    M = np.column_stack([x, px, y, py, z, pz])

    N_real = float(abs(Q_target_C) / q_e)
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
        "t_s": t_s,
        "Ez_t": Ez_t,
        "dphi_eV_t": dphi_t,
        "phi_eff_eV_t": phi_eff_t,
        "J_Apm2_t": J_t,
        "I_A_t": I_t,
        "Q_cum_C": Q_cum,
        "t_emit_s": t_emit_s,
        "n_periods": n_periods_used,
        "samples_per_period": samples_per_period,
        "periods_capped": periods_capped,
        "has_t0": hasattr(B0, "set_t0") or hasattr(B0, "get_t0"),
    }
    return B0, info


# ----------------------------- Diagnostics during tracking -----------------------------

def track_volume_with_screens(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    phi_deg: float,
    p: VolumeBuildParams,
    B0,
    z_screens_m: Sequence[float],
):
    """
    Track once, capturing phase-space snapshots at `z_screens_m`.

    RF-Track stores the hit particles and creates a Bunch6d per screen. After tracking, use
    Volume.get_bunch_at_screens() (single bunch) to retrieve them. fileciteturn3file0L1-L4
    """
    z_screens_m = [float(z) for z in z_screens_m]
    V = build_volume(rft, Er_grid, Ez_grid, phi_deg, p, add_screens_z_m=z_screens_m)

    Bout = V.track(B0)

    # RF-Track returns one Bunch6d per screen (in the screen reference frame). fileciteturn3file2L30-L36
    snaps = V.get_bunch_at_screens() if hasattr(V, "get_bunch_at_screens") else []
    return Bout, snaps


def track_volume_transport_table(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    phi_deg: float,
    p: VolumeBuildParams,
    B0,
    tt_dt_mm: float,
    table_fmt: str,
):
    """
    Track once and retrieve RF-Track's transport table in a Volume.

    In Volume, transport table sampling is enabled via TrackingOptions.tt_dt_mm (mm/c). fileciteturn4file0L8-L10
    """
    V = build_volume(rft, Er_grid, Ez_grid, phi_deg, p, add_screens_z_m=None)

    opts = rft.TrackingOptions()
    opts.tt_dt_mm = float(tt_dt_mm)

    Bout = V.track(B0, opts)
    T = V.get_transport_table(table_fmt) if hasattr(V, "get_transport_table") else None
    return Bout, T
