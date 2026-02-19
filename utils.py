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
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any, List, Literal

import numpy as np
from scipy.constants import c, e as q_e, epsilon_0
from scipy.interpolate import griddata, UnivariateSpline

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


def min_step(vals: np.ndarray) -> float:
    """Min positive spacing."""
    u = np.unique(np.asarray(vals))
    if u.size < 2:
        return np.nan
    d = np.diff(np.sort(u))
    d = d[d > 0]
    return float(d.min()) if d.size else np.nan


def med_step(vals: np.ndarray) -> float:
    """Median positive spacing."""
    u = np.unique(np.asarray(vals))
    if u.size < 2:
        return np.nan
    d = np.diff(np.sort(u))
    d = d[d > 0]
    return float(np.median(d)) if d.size else np.nan


def fmt_bytes(n: float) -> str:
    """Byte size label."""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


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


def build_iq_phasor(
    field_0: np.ndarray,
    field_90: np.ndarray,
    env_0: float,
    env_90: float,
    scale: float = 1.0,
) -> np.ndarray:
    """Complex phasor from two snapshots at 0° and 90°, normalized by the envelope."""
    e0 = field_0 / (env_0 if env_0 != 0 else 1.0)
    e90 = field_90 / (env_90 if env_90 != 0 else 1.0)
    return (e0 - 1j * e90) * float(scale)


def build_crest_phasor(field_crest: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
    """Simplified phasor using a single crest snapshot (real-only)."""
    field_crest = np.asarray(field_crest, dtype=float)
    if scale is None:
        return field_crest.astype(np.complex128)
    env = float(np.max(np.abs(field_crest))) if field_crest.size else 1.0
    env = env if env != 0.0 else 1.0
    return (field_crest / env) * float(scale)


def rms_from_phasor_over_time(
    phasor: np.ndarray,
    t_ns: np.ndarray,
    f_hz: float,
    phase_deg: float = 0.0,
) -> np.ndarray:
    """RMS of Re{phasor * exp(j*omega*t + j*phase)} over vertices for each time sample."""
    ph = np.asarray(phasor, dtype=np.complex128).reshape(-1)
    if ph.size == 0:
        return np.zeros_like(t_ns, dtype=float)

    a = ph.real
    b = ph.imag
    a2 = float(np.mean(a * a))
    b2 = float(np.mean(b * b))
    ab = float(np.mean(a * b))

    omega = 2.0 * np.pi * float(f_hz)
    t_s = np.asarray(t_ns, dtype=float) * 1e-9
    theta = omega * t_s + np.deg2rad(float(phase_deg))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    rms2 = (a2 * (cos_t**2)) + (b2 * (sin_t**2)) - (2.0 * ab * cos_t * sin_t)
    rms2 = np.clip(rms2, 0.0, None)
    return np.sqrt(rms2)


def interp_cfield(pts: np.ndarray, R: np.ndarray, Z: np.ndarray, phasor: np.ndarray) -> np.ndarray:
    """Complex field interpolation with NaN fill."""
    re_lin = griddata(pts, phasor.real, (R, Z), method="linear")
    im_lin = griddata(pts, phasor.imag, (R, Z), method="linear")
    re_nn = griddata(pts, phasor.real, (R, Z), method="nearest")
    im_nn = griddata(pts, phasor.imag, (R, Z), method="nearest")
    re = np.where(np.isfinite(re_lin), re_lin, re_nn)
    im = np.where(np.isfinite(im_lin), im_lin, im_nn)
    return (re + 1j * im).astype(np.complex128)


def phasor_check(
    Ez_yz: np.ndarray,
    yz_vertices: Optional[np.ndarray],
    t_ns: np.ndarray,
    f_hz: float,
    mode: str,
    i0: int,
    i90: int,
    i_crest: int,
    Ez_rms: np.ndarray,
    t_fit: Optional[np.ndarray] = None,
    Ez_fit: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compare phasor reconstruction against Ez_rms(t)."""
    import matplotlib.pyplot as plt

    mode = str(mode).strip().lower()
    if mode not in ("reconstruct", "simplified"):
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "reconstruct":
        Ez_0 = Ez_yz[:, i0]
        Ez_90 = Ez_yz[:, i90]
        Ez_max_0 = float(np.max(np.abs(Ez_0)))
        Ez_max_90 = float(np.max(np.abs(Ez_90)))
        Ez_ref = max(Ez_max_0, Ez_max_90)
        Ez_phasor = build_iq_phasor(Ez_0, Ez_90, Ez_max_0, Ez_max_90, scale=Ez_ref)
        phase_deg = -360.0 * f_hz * (float(t_ns[i0]) * 1e-9)
        phase_label = f"aligned to i0 @ {t_ns[i0]:.4f} ns"
    else:
        Ez_crest = Ez_yz[:, i_crest]
        Ez_max_crest = float(np.max(np.abs(Ez_crest)))
        Ez_phasor = build_crest_phasor(Ez_crest, scale=Ez_max_crest)
        phase_deg = -360.0 * f_hz * (float(t_ns[i_crest]) * 1e-9)
        phase_label = f"aligned to crest @ {t_ns[i_crest]:.4f} ns"

    Ez_rms_recon = rms_from_phasor_over_time(Ez_phasor, t_ns, f_hz, phase_deg=phase_deg)
    peak_data = float(np.max(Ez_rms)) if Ez_rms.size else np.nan
    peak_recon = float(np.max(Ez_rms_recon)) if Ez_rms_recon.size else np.nan
    ratio = (peak_recon / peak_data) if peak_data else np.nan

    print("Phasor check:")
    print(f"  Mode: {mode}")
    print(f"  Phasor time evolution uses f = {f_hz/1e9:.6f} GHz")
    print(f"  Phase alignment: {phase_label}")
    print("Amplitude:")
    print(f"  Peak Ez_rms (data):   {peak_data:.3e} V/m")
    print(f"  Peak Ez_rms (phasor): {peak_recon:.3e} V/m")
    print(f"  Peak ratio (phasor/data): {ratio:.3f}")
    print(f"  Ez_peak (data RMS max): {float(np.max(Ez_rms)):.3e} V/m")

    out = {
        "peak_data": peak_data,
        "peak_recon": peak_recon,
        "ratio": ratio,
        "phase_deg": phase_deg,
    }

    if t_ns.size <= 1:
        return out

    dt_s = float(np.median(np.diff(t_ns))) * 1e-9
    if dt_s <= 0:
        return out

    rms_per_vertex = np.sqrt(np.mean(Ez_yz**2, axis=1))
    i_vtx = int(np.argmax(rms_per_vertex))
    Ez_trace = Ez_yz[i_vtx, :].astype(float)
    if yz_vertices is not None and len(yz_vertices) > i_vtx:
        vtx_y_mm = float(yz_vertices[i_vtx, 1])
        vtx_z_mm = float(yz_vertices[i_vtx, 2])
    else:
        vtx_y_mm = np.nan
        vtx_z_mm = np.nan

    omega = 2.0 * np.pi * f_hz
    Ez_recon_trace = np.real(
        Ez_phasor[i_vtx] * np.exp(1j * (omega * (t_ns * 1e-9) + np.deg2rad(phase_deg)))
    )
    s_factor_trace = 1e-4 * Ez_trace.var() * Ez_trace.size
    spl_trace = UnivariateSpline(t_ns, Ez_trace, s=s_factor_trace)
    Ez_trace_fit = spl_trace(t_fit) if t_fit is not None else None

    y = Ez_trace - np.mean(Ez_trace)
    yf = np.fft.rfft(y)
    ff = np.fft.rfftfreq(y.size, d=dt_s)
    f_est_hz = np.nan
    if ff.size > 1:
        band = (ff > 0.25 * f_hz) & (ff < 2.0 * f_hz)
        if np.any(band):
            k = int(np.argmax(np.abs(yf[band])))
            f_est_hz = float(ff[band][k])
    span_s = (t_ns[-1] - t_ns[0]) * 1e-9
    df_hz = (1.0 / span_s) if span_s > 0 else np.nan
    nyq_hz = 0.5 / dt_s
    print("Frequency diagnostics (single-vertex Ez):")
    print(f"  FFT peak (banded): {f_est_hz/1e9:.6f} GHz (nominal {f_hz/1e9:.6f} GHz)")
    print(f"  FFT resolution: Delta f approx {df_hz/1e9:.6f} GHz | Nyquist: {nyq_hz/1e9:.6f} GHz")

    sgn = np.sign(Ez_trace)
    zc = np.where(np.diff(sgn) != 0)[0]
    if zc.size > 2:
        t_zc = t_ns[zc].astype(float)
        periods = np.diff(t_zc[::2]) if t_zc.size >= 3 else np.array([])
        f_zc_hz = 1e9 / float(np.median(periods)) if periods.size else np.nan
    else:
        f_zc_hz = np.nan
    print(f"  Zero-crossings: {f_zc_hz/1e9:.6f} GHz")

    t_s = t_ns * 1e-9
    f_min = 0.5 * f_hz
    f_max = 1.5 * f_hz
    n_grid = 2000
    f_grid = np.linspace(f_min, f_max, n_grid)
    best = (np.inf, np.nan, 0.0, 0.0, 0.0)
    for f_try in f_grid:
        w = 2.0 * np.pi * f_try
        cos_wt = np.cos(w * t_s)
        sin_wt = np.sin(w * t_s)
        X = np.column_stack([cos_wt, sin_wt, np.ones_like(cos_wt)])
        coef, _, _, _ = np.linalg.lstsq(X, Ez_trace, rcond=None)
        resid = Ez_trace - X @ coef
        err = float(np.mean(resid**2))
        if err < best[0]:
            best = (err, float(f_try), float(coef[0]), float(coef[1]), float(coef[2]))
    f_fit_hz = best[1]
    A_fit, B_fit, C_fit = best[2], best[3], best[4]
    Ez_fit_trace = A_fit * np.cos(2.0 * np.pi * f_fit_hz * t_s) + B_fit * np.sin(
        2.0 * np.pi * f_fit_hz * t_s
    ) + C_fit
    print(f"  Sinusoid fit: {f_fit_hz/1e9:.6f} GHz")

    a = (Ez_rms - np.mean(Ez_rms))
    b = (Ez_rms_recon - np.mean(Ez_rms_recon))
    if a.size == b.size and a.size > 3:
        corr = np.correlate(a, b, mode="full")
        lag = int(np.argmax(corr) - (a.size - 1))
        lag_s = lag * dt_s
        lag_deg = 360.0 * f_hz * lag_s
        print("Phase and RMS:")
        print(f"  Lag @ max correlation: {lag_s*1e12:.2f} ps (approx {lag_deg:.1f} deg at {f_hz/1e9:.3f} GHz)")
    else:
        print("Phase and RMS:")
        print("  Lag @ max correlation: n/a")

    rms_data = float(np.sqrt(np.mean(Ez_rms**2)))
    rms_recon = float(np.sqrt(np.mean(Ez_rms_recon**2)))
    rms_ratio = (rms_recon / rms_data) if rms_data else np.nan
    print(f"  RMS Ez_rms (data):   {rms_data:.3e} V/m")
    print(f"  RMS Ez_rms (phasor): {rms_recon:.3e} V/m")
    print(f"  RMS ratio (phasor/data): {rms_ratio:.3f}")
    if np.isfinite(vtx_y_mm) and np.isfinite(vtx_z_mm):
        print(f"  Vertex (RMS-max): index={i_vtx}, y={vtx_y_mm:.3f} mm, z={vtx_z_mm:.3f} mm")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9.0), sharex=False)
    mag = np.abs(yf)
    axes[0].plot(ff * 1e-9, mag, lw=1.4, color="tab:blue")
    axes[0].axvline(f_hz * 1e-9, color="gray", ls="--", lw=1.0, label="Nominal f")
    if np.isfinite(f_est_hz):
        axes[0].axvline(f_est_hz * 1e-9, color="tab:red", ls="--", lw=1.0, label="FFT peak")
    if np.isfinite(f_zc_hz):
        axes[0].axvline(f_zc_hz * 1e-9, color="tab:green", ls=":", lw=1.0, label="Zero-cross")
    axes[0].set_xlim(0.0, min(6.0, nyq_hz * 1e-9))
    axes[0].set_xlabel("Frequency [GHz]")
    axes[0].set_ylabel("|FFT|")
    axes[0].set_title("Ez(t) FFT (single vertex)")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_ns, Ez_rms, "o", ms=4, alpha=0.6, label="Ez RMS (data)")
    axes[1].plot(t_ns, Ez_rms_recon, "-", lw=2, label=f"Phasor RMS ({mode})")
    if t_fit is not None and Ez_fit is not None:
        axes[1].plot(t_fit, Ez_fit, "-", lw=1.8, color="red", label="Spline fit (RMS)")
    axes[1].set_ylabel("Ez RMS [V/m]")
    axes[1].set_title(f"Phasor vs measured envelope ({phase_label})")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_ns, Ez_trace, "o", ms=3, alpha=0.5, label="Ez (data vertex)")
    axes[2].plot(t_ns, Ez_recon_trace, "-", lw=1.8, label="Ez (phasor vertex)")
    if np.isfinite(f_fit_hz):
        axes[2].plot(t_ns, Ez_fit_trace, "-", lw=1.2, color="tab:red", label="Sinusoid fit (Ez)")
    if Ez_trace_fit is not None:
        axes[2].plot(t_fit, Ez_trace_fit, "-", lw=1.6, color="red", label="Spline fit (Ez)")
    axes[2].set_xlabel("Time [ns]")
    axes[2].set_ylabel("Ez [V/m]")
    if np.isfinite(vtx_y_mm) and np.isfinite(vtx_z_mm):
        axes[2].set_title(f"Ez(t) at vertex {i_vtx} (y={vtx_y_mm:.3f} mm, z={vtx_z_mm:.3f} mm)")
    else:
        axes[2].set_title(f"Ez(t) at vertex {i_vtx}")
    axes[2].legend(frameon=False, loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Phasor time evolution uses f = {f_hz/1e9:.6f} GHz", fontsize=11)
    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    plt.show()

    return out


def field_maps(
    xy: Dict[str, np.ndarray],
    yz: Dict[str, np.ndarray],
    t_ns: np.ndarray,
    t_crest: float,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    Ez_grid: np.ndarray,
    lambda_m: float,
):
    """Plot raw field maps and RF-Track grid."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import matplotlib.colors as colors

    i_snap = int(np.argmin(np.abs(t_ns - t_crest)))

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    verts_xy = xy["vertices"]
    tri_xy = xy["facets"]
    Ux = verts_xy[:, 0]
    Vy = verts_xy[:, 1]
    Fx = np.asarray(xy["Ez"])[:, i_snap]

    triang_xy = mtri.Triangulation(Ux, Vy, triangles=tri_xy)
    ax_xy = fig.add_subplot(gs[0, 0])
    p_xy = float(np.percentile(np.abs(Fx), 99)) if Fx.size else 1.0
    p_xy = p_xy if p_xy > 0 else 1.0
    norm_xy = colors.TwoSlopeNorm(vcenter=0.0, vmin=-p_xy, vmax=p_xy)
    levels_xy = np.linspace(-p_xy, p_xy, 257)
    cf_xy = ax_xy.tricontourf(triang_xy, Fx, levels=levels_xy, cmap="coolwarm", norm=norm_xy)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x [mm]")
    ax_xy.set_ylabel("y [mm]")
    ax_xy.set_title("Raw mesh: XY Ez at crest")
    plt.colorbar(cf_xy, ax=ax_xy, label="Ez [V/m]")

    verts_yz = yz["vertices"]
    tri_yz = yz["facets"]
    Uy = verts_yz[:, 1]
    Vz = verts_yz[:, 2]
    Fy = np.asarray(yz["Ez"])[:, i_snap]

    triang_yz = mtri.Triangulation(Vz, Uy, triangles=tri_yz)
    ax_yz = fig.add_subplot(gs[0, 1])
    p_yz = float(np.percentile(np.abs(Fy), 99)) if Fy.size else 1.0
    p_yz = p_yz if p_yz > 0 else 1.0
    norm_yz = colors.TwoSlopeNorm(vcenter=0.0, vmin=-p_yz, vmax=p_yz)
    levels_yz = np.linspace(-p_yz, p_yz, 257)
    cf_yz = ax_yz.tricontourf(triang_yz, Fy, levels=levels_yz, cmap="coolwarm", norm=norm_yz)
    ax_yz.set_aspect("equal", adjustable="box")
    ax_yz.set_xlabel("z [mm]")
    ax_yz.set_ylabel("y [mm]")
    ax_yz.set_title("Raw mesh (rotated): YZ Ez at crest")
    plt.colorbar(cf_yz, ax=ax_yz, label="Ez [V/m]")

    r_neg = -r_grid[::-1]
    r_full = np.concatenate([r_neg, r_grid[1:]])
    Ez_full = np.concatenate([Ez_grid[:, ::-1], Ez_grid[:, 1:]], axis=1)

    ax_rf = fig.add_subplot(gs[1, :])
    extent_full = [z_grid[0] * 1e3, z_grid[-1] * 1e3, r_full[0] * 1e3, r_full[-1] * 1e3]
    im = ax_rf.imshow(
        np.real(Ez_full.T),
        aspect="auto",
        origin="lower",
        extent=extent_full,
        cmap="plasma",
    )
    ax_rf.axvline(0, color="white", ls="--", lw=1, alpha=0.5, label="Cathode (z=0)")
    ax_rf.axvline(lambda_m / 4 * 1e3, color="cyan", ls="--", lw=1, alpha=0.7, label="lambda/4")
    ax_rf.axhline(0, color="white", ls=":", lw=0.8, alpha=0.4)
    ax_rf.set_xlabel("z [mm]")
    ax_rf.set_ylabel("r [mm]")
    ax_rf.set_title("RF-Track field map: Re(Ez)")
    legend = ax_rf.legend(frameon=False, loc="upper right", fontsize=16)
    for text in legend.get_texts():
        text.set_color("white")
    ax_rf.text(
        0,
        r_full[-1] * 1e3 * 0.93,
        "Cathode (z=0)",
        color="white",
        fontsize=16,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.15, edgecolor="none"),
    )
    ax_rf.text(
        lambda_m / 4 * 1e3,
        r_full[-1] * 1e3 * 0.93,
        "lambda/4",
        color="white",
        fontsize=16,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.15, edgecolor="none"),
    )
    plt.colorbar(im, ax=ax_rf, label="Ez [V/m]")

    plt.tight_layout()
    plt.show()


def axis_phase(
    Ez_axis: np.ndarray,
    z_grid: np.ndarray,
    Ez0_phasor_axis: complex,
    emission_phase_start: float,
    emission_phase_range: float,
    lambda_m: float,
) -> Tuple[float, float]:
    """Auto phase from on-axis phasor and plot Ez(z)."""
    import matplotlib.pyplot as plt

    phi_opt = -np.angle(Ez_axis[np.argmax(np.abs(Ez_axis))])
    phi_zero_deg = (90.0 - np.rad2deg(np.angle(Ez0_phasor_axis))) % 360.0
    transport_phase_deg = (phi_zero_deg + float(emission_phase_start)) % 360.0

    print(f"Auto phase: Ez0 crosses 0 at phi approx {phi_zero_deg:.2f} deg")
    print(
        f"Transport phase (t=0): phi = {transport_phase_deg:.2f} deg "
        f"(start shift {float(emission_phase_start):.1f} deg)"
    )
    print(f"Emission window: {float(emission_phase_range):.1f} deg")

    offsets_deg = [0, 30, 60, 90, 120, 150, 180]
    phases_deg_plot = [np.rad2deg(phi_opt) + d for d in offsets_deg]
    for extra_phase in (phi_zero_deg, transport_phase_deg):
        if not any(np.isclose(extra_phase, p, atol=1e-3) for p in phases_deg_plot):
            phases_deg_plot.append(extra_phase)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for deg in phases_deg_plot:
        phi = np.deg2rad(deg)
        Ez_phase = np.real(Ez_axis * np.exp(1j * phi))
        ax.plot(z_grid * 1e3, Ez_phase, lw=1.5, label=f"phi = {deg:.1f} deg")

    ax.axvline(0, color="red", ls="--", lw=1, alpha=0.6, label="Cathode")
    ax.axvline(
        lambda_m / 4 * 1e3,
        color="blue",
        ls="--",
        lw=1,
        alpha=0.6,
        label=f"lambda/4 = {lambda_m/4*1e3:.2f} mm",
    )
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Re{Ez(r=0, z, phi)} [V/m]")
    ax.set_title("On-axis field at selected phases")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return float(transport_phase_deg), float(phi_zero_deg)


def theory_plot(phi_deg: np.ndarray, dW_vals: np.ndarray, pz_theory: np.ndarray):
    """Plot theory phase scan for energy gain."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(phi_deg, dW_vals, "o-", ms=4, lw=1.6, color="tab:blue", label="Delta W")
    ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.6)

    ax2 = ax.twinx()
    ax2.plot(phi_deg, pz_theory, "s--", ms=3, lw=1.2, color="tab:orange", alpha=0.9, label="pz (end)")
    ax2.set_ylabel("pz [MeV/c]", color="tab:orange")
    ax2.tick_params(axis="y", colors="tab:orange")

    if np.any(np.isfinite(dW_vals)):
        i_max = int(np.nanargmax(dW_vals))
        ax.axvline(phi_deg[i_max], color="tab:red", ls="--", lw=1.2, alpha=0.7)
        ax.plot(phi_deg[i_max], dW_vals[i_max], "o", ms=7, color="tab:red", zorder=5)
        ax.text(
            phi_deg[i_max],
            dW_vals[i_max],
            f"  peak {dW_vals[i_max]:.3f} MeV @ {phi_deg[i_max]:.1f} deg",
            va="bottom",
            ha="left",
            fontsize=9,
            color="tab:red",
        )

    ax.set_xlabel("RF phase [deg]")
    ax.set_ylabel("Delta W [MeV]", color="tab:blue")
    ax.tick_params(axis="y", colors="tab:blue")
    ax.set_title("Theory: energy gain vs phase")
    ax.grid(alpha=0.3)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")

    plt.tight_layout()
    plt.show()


def phase_plot(phi_abs: np.ndarray, pz_mean: np.ndarray):
    """Plot fast phase scan pz vs phase."""
    import matplotlib.pyplot as plt

    mask = np.isfinite(pz_mean)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(phi_abs[mask], pz_mean[mask], "o-", ms=4, lw=1.6, color="tab:blue")
    ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.6)
    if np.any(mask):
        i_max = int(np.nanargmax(pz_mean))
        ax.axvline(phi_abs[i_max], color="tab:red", ls="--", lw=1.2, alpha=0.7)
        ax.plot(phi_abs[i_max], pz_mean[i_max], "o", ms=7, color="tab:red", zorder=5)
        ax.text(
            phi_abs[i_max],
            pz_mean[i_max],
            f"  peak {pz_mean[i_max]:.3f} MeV/c @ {phi_abs[i_max]:.1f} deg",
            va="bottom",
            ha="left",
            fontsize=9,
            color="tab:red",
        )
    ax.set_xlabel("RF phase [deg] (absolute)")
    ax.set_ylabel("Mean pz at exit [MeV/c]")
    ax.set_title("Fast phase scan: mean pz vs phase")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def bunch_diag(
    B0,
    Bout,
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    transport_phase_deg: float,
    clean_e: bool = False,
    show_zle0: bool = True,
):
    """Plot spectra, phase space, and z evolution."""
    import matplotlib.pyplot as plt

    PHASE_FMT = "%X %Px %Y %Py %Z %Pz"
    ID_FMTS = ["%id"]

    def _safe_get_phase_space(bunch, selection):
        return np.array(bunch.get_phase_space(PHASE_FMT, selection), copy=True)

    def _try_get_ids(bunch, selection):
        for fmt in ID_FMTS:
            try:
                ids = np.array(bunch.get_phase_space(fmt, selection), copy=True).reshape(-1)
                if ids.size:
                    return ids
            except Exception:
                continue
        return None

    def _clean_output_particles(M):
        if not clean_e or M is None or M.shape[0] == 0:
            return M
        mask = M[:, 4] > 0.0
        return M[mask]

    try:
        Mf_f_all = _safe_get_phase_space(Bout, "all")
        Mf_launch_all = _safe_get_phase_space(B0, "all")
        has_all = True
    except Exception:
        Mf_f_all = _safe_get_phase_space(Bout, "good")
        Mf_launch_all = _safe_get_phase_space(B0, "good")
        has_all = False

    finite_z = np.isfinite(Mf_f_all[:, 4])
    mask_bad = finite_z & (Mf_f_all[:, 4] <= 0.0)
    mask_good = finite_z & (Mf_f_all[:, 4] > 0.0)

    Mf_f_good = Mf_f_all[mask_good]
    Mf_f = Mf_f_good if clean_e else Mf_f_all[finite_z]
    M_snaps = [_clean_output_particles(M) for M in M_snaps]

    ids_exit = _try_get_ids(Bout, "all") if has_all else None
    ids_launch = _try_get_ids(B0, "all") if has_all else None
    lost_ids = None
    if ids_exit is not None and ids_exit.size == Mf_f_all.shape[0]:
        lost_ids = ids_exit[mask_bad]
        print(f"Lost particles (z <= 0): {lost_ids.size} of {ids_exit.size}")
        if lost_ids.size:
            preview = np.array2string(lost_ids[:20], separator=", ")
            print(f"Lost particle IDs (first 20): {preview}")

    total_tracked = int(np.sum(finite_z))
    lost_count = int(np.sum(mask_bad))
    good_count = int(np.sum(mask_good))
    transmission_pct = 100.0 * good_count / total_tracked if total_tracked > 0 else 0.0
    print(f"Lost particles (z <= 0): {lost_count} / {total_tracked}")
    print(f"Transmission (z > 0): {transmission_pct:.2f}% (tracked)")

    Mf_launch = Mf_launch_all
    launch_mask_bad = None
    if has_all and lost_ids is not None and ids_launch is not None:
        if ids_launch.size == Mf_launch_all.shape[0]:
            launch_mask_bad = np.isin(ids_launch, lost_ids)
        else:
            print("Warning: launch ID array length does not match launch phase space.")
    elif has_all and Mf_launch_all.shape[0] == Mf_f_all.shape[0]:
        launch_mask_bad = mask_bad
    elif has_all and Mf_launch_all.shape[0] != Mf_f_all.shape[0]:
        print(
            "Warning: 'all' selections differ in size; cannot map launch to exit one-to-one. "
            "Skipping red overlay on launch plots."
        )
    Mf_launch_bad = Mf_launch_all[launch_mask_bad] if launch_mask_bad is not None else np.empty((0, 6))

    if Mf_f.shape[0] > 0:
        pz_f = Mf_f[:, 5]
        tof_ns = (Mf_f[:, 4] * 1e-3 / c) * 1e9
        tof_ns = tof_ns[np.isfinite(tof_ns)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(pz_f, bins=60, alpha=0.75, edgecolor="black", lw=0.5)
        axes[0].set_xlabel("Pz [MeV/c]")
        axes[0].set_ylabel("Counts")
        axes[0].grid(alpha=0.3)
        axes[0].set_title(f"Final longitudinal momentum @ phi = {transport_phase_deg:.1f} deg")

        if tof_ns.size > 0:
            axes[1].hist(tof_ns, bins=60, alpha=0.75, edgecolor="black", lw=0.5)
            axes[1].set_xlabel("ToF [ns]")
            axes[1].set_ylabel("Counts")
            axes[1].grid(alpha=0.3)
            axes[1].set_title("Time of flight histogram")
        else:
            axes[1].axis("off")
            axes[1].text(0.5, 0.5, "ToF not available", ha="center", va="center")

        plt.tight_layout()
        plt.show()

    if Mf_f.shape[0] > 0 and Mf_launch.shape[0] > 0:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))

        axes[0, 0].plot(Mf_launch[:, 0], Mf_launch[:, 1], ".", ms=2, alpha=0.5)
        if show_zle0 and Mf_launch_bad.shape[0] > 0:
            axes[0, 0].plot(Mf_launch_bad[:, 0], Mf_launch_bad[:, 1], ".", ms=2, alpha=0.7, color="red")
        axes[0, 0].set_xlabel("x [mm]")
        axes[0, 0].set_ylabel("px [MeV/c]")
        axes[0, 0].set_title("Launch: x-px")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(Mf_launch[:, 2], Mf_launch[:, 3], ".", ms=2, alpha=0.5)
        if show_zle0 and Mf_launch_bad.shape[0] > 0:
            axes[0, 1].plot(Mf_launch_bad[:, 2], Mf_launch_bad[:, 3], ".", ms=2, alpha=0.7, color="red")
        axes[0, 1].set_xlabel("y [mm]")
        axes[0, 1].set_ylabel("py [MeV/c]")
        axes[0, 1].set_title("Launch: y-py")
        axes[0, 1].grid(alpha=0.3)

        axes[0, 2].plot(Mf_launch[:, 4], Mf_launch[:, 5], ".", ms=2, alpha=0.5)
        if show_zle0 and Mf_launch_bad.shape[0] > 0:
            axes[0, 2].plot(Mf_launch_bad[:, 4], Mf_launch_bad[:, 5], ".", ms=2, alpha=0.7, color="red")
        axes[0, 2].set_xlabel("z [mm]")
        axes[0, 2].set_ylabel("pz [MeV/c]")
        axes[0, 2].set_title("Launch: z-pz")
        axes[0, 2].grid(alpha=0.3)

        axes[1, 0].plot(Mf_f[:, 0], Mf_f[:, 1], ".", ms=2, alpha=0.5)
        axes[1, 0].set_xlabel("x [mm]")
        axes[1, 0].set_ylabel("px [MeV/c]")
        axes[1, 0].set_title("Exit: x-px")
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(Mf_f[:, 2], Mf_f[:, 3], ".", ms=2, alpha=0.5)
        axes[1, 1].set_xlabel("y [mm]")
        axes[1, 1].set_ylabel("py [MeV/c]")
        axes[1, 1].set_title("Exit: y-py")
        axes[1, 1].grid(alpha=0.3)

        axes[1, 2].plot(Mf_f[:, 4], Mf_f[:, 5], ".", ms=2, alpha=0.5)
        axes[1, 2].set_xlabel("z [mm]")
        axes[1, 2].set_ylabel("pz [MeV/c]")
        axes[1, 2].set_title("Exit: z-pz")
        axes[1, 2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    if len(M_snaps) and len(z_snaps) == len(M_snaps):
        z_mm = 1e3 * np.asarray(z_snaps)
        sig_x = np.array([np.std(M[:, 0]) if M.shape[0] else np.nan for M in M_snaps])
        sig_y = np.array([np.std(M[:, 2]) if M.shape[0] else np.nan for M in M_snaps])
        pz_m = np.array([np.mean(M[:, 5]) if M.shape[0] else np.nan for M in M_snaps])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(z_mm, sig_x, "o-", ms=3, label="sigma x")
        ax.plot(z_mm, sig_y, "o-", ms=3, label="sigma y")
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("RMS size [mm]")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Transverse RMS size vs z")
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(z_mm, pz_m, "o-", ms=3)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("Mean Pz [MeV/c]")
        ax.grid(alpha=0.3)
        ax.set_title("Mean longitudinal momentum vs z")
        plt.show()
    else:
        print("No snapshots available (M_snaps empty or z_snaps mismatch).")


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


def sample_pz_flux(
    n: int,
    T_K: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Flux model for normal energy: eps_z ~ Exp(mean=kT).
    Returns pz [MeV/c], mean eps_z [eV], expected mean [eV].
    """
    rng = np.random.default_rng() if rng is None else rng

    kB_J_per_K = 1.380649e-23
    kB_eV_per_K = 8.617333262e-5
    me_kg = 9.1093837015e-31
    MeV_c_SI = (1e6 * q_e) / c

    eps_z_J = rng.exponential(scale=kB_J_per_K * T_K, size=n)
    pz_SI = np.sqrt(2.0 * me_kg * eps_z_J)
    pz_MeV_c = pz_SI / MeV_c_SI

    mean_eps_eV = float(np.mean(eps_z_J) / q_e) if n > 0 else 0.0
    exp_eps_eV = float(kB_eV_per_K * T_K)
    return pz_MeV_c, mean_eps_eV, exp_eps_eV


def roughness_slope_rms(Ra_um: float, Re_um: float) -> float:
    """
    RMS surface slope from sinusoidal roughness.
    Assume a ~ sqrt(2)*Ra and lambda ~ Re.
    """
    Ra_um = float(Ra_um)
    Re_um = float(Re_um)
    if Ra_um <= 0.0 or Re_um <= 0.0:
        return 0.0
    amp_um = np.sqrt(2.0) * Ra_um
    return float((2.0 * np.pi * amp_um) / Re_um)


def apply_roughness(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    Ra_um: float,
    Re_um: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Small-angle rotation from local surface slopes.
    px <- px + pz*theta_x, py <- py + pz*theta_y.
    """
    rng = np.random.default_rng() if rng is None else rng
    sigma_theta = roughness_slope_rms(Ra_um, Re_um)
    if sigma_theta <= 0.0:
        return px, py, 0.0
    theta_x = rng.normal(0.0, sigma_theta, size=px.size)
    theta_y = rng.normal(0.0, sigma_theta, size=py.size)
    px = px + pz * theta_x
    py = py + pz * theta_y
    return px, py, float(sigma_theta)


def sample_thermionic_momenta(
    n: int,
    T_K: float,
    pz0_MeV_c: float,
    pz_model: Literal["constant", "flux"] = "flux",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Maxwellian transverse momenta with optional flux-normal pz.
    pz_model='flux' samples eps_z ~ Exp(kT); 'constant' uses pz0_MeV_c.
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

    if pz_model == "flux":
        pz, mean_eps_eV, exp_eps_eV = sample_pz_flux(n, T_K, rng=rng)
    elif pz_model == "constant":
        pz = np.full(n, float(pz0_MeV_c))
        mean_eps_eV = np.nan
        exp_eps_eV = float(8.617333262e-5 * T_K)
    else:
        raise ValueError(f"Unknown pz_model: {pz_model}")
    return px, py, pz, float(mean_eps_eV), float(exp_eps_eV)


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
    emission_phase_range_deg: float,
    pz0_MeV_c: float,
    Ez0_phasor_axis: complex,
    time_dependent: bool = True,
    pz_model: Literal["constant", "flux"] = "flux",
    Ra_um: float = 0.0,
    Re_um: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Thermionic emission with Richardson + Schottky current.

    The macro-particle charge matches the total charge emitted within the selected
    phase window. Emission times are sampled from a time-dependent current I(t)
    derived from Ez(z=0, t) unless time_dependent=False, in which case emission
    times are uniform over the phase window.
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
    Q_total_C = 0.0

    if time_dependent:
        f_hz = float(f_hz)
        T = 1.0 / f_hz
        omega = 2.0 * np.pi * f_hz

        phase_range_deg = max(float(emission_phase_range_deg), 0.0)
        tau_s = (phase_range_deg / 360.0) * T

        samples_per_period = max(200, int(phase_range_deg * 2.0))
        n_samples = max(int(samples_per_period * phase_range_deg / 360.0) + 1, 2)

        t_s = np.linspace(0.0, tau_s, n_samples)
        Ez_t = np.real(Ez0_phasor_axis * np.exp(1j * (omega * t_s + phi_rad)))
        Eabs_t = np.abs(Ez_t)
        dphi_t = np.sqrt((q_e**3) * Eabs_t / (4.0 * np.pi * epsilon_0)) / q_e
        phi_eff_t = np.maximum(work_function_eV - dphi_t, 0.0)

        kB_eV_per_K = 8.617333262e-5
        J_t = A_RICH * (cathode_T_K**2) * np.exp(-phi_eff_t / (kB_eV_per_K * cathode_T_K))
        I_t = J_t * area_m2

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
        phase_range_deg = max(float(emission_phase_range_deg), 0.0)
        tau_s = (phase_range_deg / 360.0) * T
        I_avg = I0
        I_peak = I0
        Q_total_C = float(I0 * tau_s) if np.isfinite(tau_s) else 0.0
        if tau_s > 0.0:
            t_emit_s = rng.uniform(0.0, tau_s, size=n)
        else:
            t_emit_s = np.zeros(n)

    # Transverse phase space
    x, y = sample_disk(n, cathode_radius_mm, rng=rng)
    px, py, pz, mean_eps_eV, exp_eps_eV = sample_thermionic_momenta(
        n,
        cathode_T_K,
        pz0_MeV_c,
        pz_model=pz_model,
        rng=rng,
    )

    px_rms0 = float(np.std(px)) if px.size else np.nan
    py_rms0 = float(np.std(py)) if py.size else np.nan
    px, py, sigma_theta = apply_roughness(px, py, pz, Ra_um, Re_um, rng=rng)
    px_rms = float(np.std(px)) if px.size else np.nan
    py_rms = float(np.std(py)) if py.size else np.nan

    if pz_model == "flux":
        print(
            f"Normal energy: <eps_z>={mean_eps_eV:.4f} eV (expected {exp_eps_eV:.4f} eV)"
        )

    # Emission time distribution (mm/c)
    if np.isfinite(tau_s) and t_emit_s is not None:
        t = t_emit_s * c * 1e3  # [mm/c]
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
        "emission_phase_range_deg": float(emission_phase_range_deg),
        "pz_model": str(pz_model),
        "mean_eps_z_eV": float(mean_eps_eV),
        "mean_eps_z_eV_expected": float(exp_eps_eV),
        "Ra_um": float(Ra_um),
        "Re_um": float(Re_um),
        "sigma_theta_rad": float(sigma_theta),
        "px_rms0": float(px_rms0),
        "py_rms0": float(py_rms0),
        "px_rms": float(px_rms),
        "py_rms": float(py_rms),
        "t_s": t_s,
        "Ez_t": Ez_t,
        "dphi_eV_t": dphi_t,
        "phi_eff_eV_t": phi_eff_t,
        "J_Apm2_t": J_t,
        "I_A_t": I_t,
        "Q_cum_C": Q_cum,
        "t_emit_s": t_emit_s,
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
