"""Phasor construction and checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
from scipy.interpolate import griddata, UnivariateSpline
from scipy.spatial import Delaunay, cKDTree


def select_iq_snapshots(
    t_ns: np.ndarray,
    Ez_rms: np.ndarray,
    f_hz: float,
    search_window: int = 60,
) -> Tuple[int, int, float, float]:
    """Choose two indices separated by ~T/4 for I/Q reconstruction."""
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
    """Complex phasor from two snapshots at 0 deg and 90 deg.

    `env_0`/`env_90` are each expected strictly positive (the field-map snapshots' own peak
    instantaneous amplitude) -- an exactly-zero envelope means the snapshot at that time is
    degenerate (all-zero field), which should never happen for a real field-map snapshot. Raising
    here surfaces that immediately rather than silently normalizing by 1.0, which would otherwise
    produce a phasor with the wrong amplitude with no indication anything was wrong.
    """
    if env_0 == 0.0 or env_90 == 0.0:
        raise ValueError(f"build_iq_phasor: degenerate (zero) envelope, env_0={env_0!r}, env_90={env_90!r}")
    e0 = field_0 / env_0
    e90 = field_90 / env_90
    return (e0 - 1j * e90) * float(scale)


def build_crest_phasor(field_crest: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
    """Simplified phasor using a crest snapshot (real-only). See `build_iq_phasor`'s docstring for
    why a zero envelope raises rather than silently normalizing by 1.0."""
    field_crest = np.asarray(field_crest, dtype=float)
    if scale is None:
        return field_crest.astype(np.complex128)
    env = float(np.max(np.abs(field_crest))) if field_crest.size else 0.0
    if env == 0.0:
        raise ValueError("build_crest_phasor: degenerate (all-zero) crest snapshot")
    return (field_crest / env) * float(scale)


def rms_from_phasor_over_time(
    phasor: np.ndarray,
    t_ns: np.ndarray,
    f_hz: float,
    phase_deg: float = 0.0,
) -> np.ndarray:
    """RMS of Re{phasor * exp(j*omega*t + j*phase)} over vertices."""
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


@dataclass
class FieldInterpolationContext:
    """Reusable source-mesh triangulation and target-grid hull membership.

    Building the Delaunay triangulation and target-grid hull test is the expensive, field-value-
    independent step of `interp_cfield`. It depends only on the source vertex locations (`pts`)
    and the target grid (`R`, `Z`), so one context is built once and reused for every field
    component (Er/Ez, real/imaginary) evaluated on that same grid, instead of rebuilding it (and
    re-running `Delaunay`) on every call as the previous implementation did.

    `repaired_hole_fraction` is mutated by `interp_cfield` to the largest fraction seen across every
    component interpolated with this context (isolated interior holes are rare; a handful of
    components sharing one context should not silently overwrite each other's diagnostic).
    """

    kdtree: cKDTree
    target_pts: np.ndarray  # (n_target, 2), flattened (r, z) target grid points
    inside_hull: np.ndarray  # bool, flattened target-grid mask: True where inside native support
    shape: Tuple[int, int]
    repaired_hole_fraction: float = 0.0

    @property
    def outside_hull_fraction(self) -> float:
        return float(np.mean(~self.inside_hull)) if self.inside_hull.size else float("nan")


def build_field_interpolation_context(pts: np.ndarray, R: np.ndarray, Z: np.ndarray) -> FieldInterpolationContext:
    """Triangulate the source vertices once and classify every target-grid point as inside or
    outside the native convex hull. Reuse the returned context across every component passed to
    `interp_cfield` for the same (pts, R, Z)."""
    pts = np.asarray(pts, dtype=float)
    tri = Delaunay(pts)
    target_pts = np.column_stack([np.asarray(R).ravel(), np.asarray(Z).ravel()])
    inside = tri.find_simplex(target_pts) >= 0
    kdtree = cKDTree(pts)
    return FieldInterpolationContext(
        kdtree=kdtree, target_pts=target_pts, inside_hull=inside, shape=np.asarray(R).shape
    )


def interp_cfield(
    pts: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    phasor: np.ndarray,
    *,
    ctx: Optional[FieldInterpolationContext] = None,
    outside_value: complex = 0.0,
) -> np.ndarray:
    """Complex field interpolation onto (R, Z) from scattered source vertices `pts`.

    Outside the native convex hull of `pts` the field is set to `outside_value` (zero by default):
    the measured map carries no information there, so nearest-neighbor extrapolation would invent
    a nonzero field the data never supported. A handful of points strictly inside the hull can
    still come back non-finite from the linear interpolant (e.g. an exact simplex-edge
    degeneracy); those isolated interior holes are repaired with a KD-tree nearest lookup, not
    treated as outside-support points. Pass a `FieldInterpolationContext` built once via
    `build_field_interpolation_context` to avoid re-triangulating the source mesh for every
    component (real/imag, Er/Ez) sharing the same (pts, R, Z).

    `ctx.outside_hull_fraction` and `ctx.repaired_hole_fraction` (updated in place, largest fraction
    seen across every component interpolated with this `ctx`) are available for run provenance.
    """
    if ctx is None:
        ctx = build_field_interpolation_context(pts, R, Z)
    shape = ctx.shape

    re_lin = griddata(pts, phasor.real, (R, Z), method="linear").reshape(-1)
    im_lin = griddata(pts, phasor.imag, (R, Z), method="linear").reshape(-1)

    inside = ctx.inside_hull
    re_flat = np.where(inside, re_lin, float(np.real(outside_value)))
    im_flat = np.where(inside, im_lin, float(np.imag(outside_value)))

    hole = inside & (~np.isfinite(re_lin) | ~np.isfinite(im_lin))
    if np.any(hole):
        _, nn_idx = ctx.kdtree.query(ctx.target_pts[hole])
        re_flat[hole] = phasor.real[nn_idx]
        im_flat[hole] = phasor.imag[nn_idx]

    out = (re_flat + 1j * im_flat).astype(np.complex128).reshape(shape)
    repaired_hole_fraction = float(np.mean(hole)) if hole.size else 0.0
    ctx.repaired_hole_fraction = max(ctx.repaired_hole_fraction, repaired_hole_fraction)
    return out


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
    print(f"  FFT resolution: Delta f approx {df_hz/1e9:.6f} GHz | Nyquist: {nyq_hz/1e9:.6f} GHz)")

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
        print(
            f"  Lag @ max correlation: {lag_s*1e12:.2f} ps (approx {lag_deg:.1f} deg at {f_hz/1e9:.3f} GHz)"
        )
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
        axes[2].set_title(
            f"Ez(t) at vertex {i_vtx} (y={vtx_y_mm:.3f} mm, z={vtx_z_mm:.3f} mm)"
        )
    else:
        axes[2].set_title(f"Ez(t) at vertex {i_vtx}")
    axes[2].legend(frameon=False, loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Phasor time evolution uses f = {f_hz/1e9:.6f} GHz", fontsize=11)
    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    plt.show()

    return out
