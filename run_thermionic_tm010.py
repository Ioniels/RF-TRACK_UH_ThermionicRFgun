#!/usr/bin/env python3
"""Batch-friendly thermionic TM010 RF-gun run (no Jupyter dependencies)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import time

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thermionic TM010 transport with RF-Track.")

    parser.add_argument("--preset", choices=["none", "quick"], default="none")
    parser.add_argument("--output", type=Path, default=Path("outputs") / f"tm010_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--phase_deg", type=float, default=0.0)
    parser.add_argument("--emission_phase_start", type=float, default=0.0)
    parser.add_argument("--n_particles", type=int, default=100_000)

    parser.add_argument("--xy_fieldmap", type=Path, default=Path("field_maps/XYplanarSensorData.mat"))
    parser.add_argument("--yz_fieldmap", type=Path, default=Path("field_maps/YZplanarSensorData.mat"))
    parser.add_argument("--phasor_mode", choices=["reconstruct", "simplified"], default="reconstruct")

    parser.add_argument("--f_hz", type=float, default=2.856e9)
    parser.add_argument("--y_cathode_mm", type=float, default=12.75)
    parser.add_argument("--r_max_m", type=float, default=0.01)
    parser.add_argument("--dr_um", type=float, default=4.0)
    parser.add_argument("--dz_um", type=float, default=13.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=None)
    parser.add_argument("--ext_zmax", type=float, default=0.0075)

    parser.add_argument("--dt_mm", type=float, default=0.1)
    parser.add_argument("--sc_dt_mm", type=float, default=0.2)
    parser.add_argument("--emission_nsteps", type=int, default=100)
    parser.add_argument("--emission_range", type=float, default=10.0)
    parser.add_argument("--fm_nsteps", type=int, default=100)
    parser.add_argument("--fm_tt_nsteps", type=int, default=100)
    parser.add_argument("--cfx_dt_mm", type=float, default=0.1)
    parser.add_argument("--ode_algorithm", type=str, default="rk2")
    parser.add_argument("--ode_epsabs", type=float, default=1e-6)
    parser.add_argument("--aperture_m", type=float, default=0.01)

    parser.add_argument("--sc_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--beam_loading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bl_q0", type=float, default=4000.0)
    parser.add_argument("--bl_qext", type=float, default=3500.0)
    parser.add_argument("--bl_p_fwd_w", type=float, default=1.0e6)
    parser.add_argument("--bl_r_over_q_ohm_per_m", type=float, default=1.0)
    parser.add_argument("--bl_ncells", type=int, default=1)
    parser.add_argument("--bl_tinj_mode", choices=["auto_from_emission", "manual"], default="auto_from_emission")
    parser.add_argument("--bl_tinj_manual_mm_c", type=float, default=0.0)

    parser.add_argument("--n_screens", type=int, default=100)
    parser.add_argument("--n_z_snap", type=int, default=None)
    parser.add_argument("--screens_z", type=float, nargs="*", default=None)
    parser.add_argument("--no-screens", action="store_true", default=False)

    parser.add_argument("--screen_width_mm", type=float, default=None)
    parser.add_argument("--screen_height_mm", type=float, default=None)
    parser.add_argument("--screen_time_window_mm_c", type=float, default=None)
    parser.add_argument("--screen_t0_mode", choices=["unset", "sync_to_first_crossing", "manual"], default="unset")
    parser.add_argument("--screen_t0_manual_mm_c", type=float, default=0.0)
    parser.add_argument("--screen_log", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--r_cathode_mm", type=float, default=3.14 / 2)
    parser.add_argument("--emission_scale", type=float, default=1.0)
    parser.add_argument("--use_const_pz", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pz_init_mevc", type=float, default=4.0e-3)
    parser.add_argument("--ra_um", type=float, default=1.0)
    parser.add_argument("--re_um", type=float, default=10.0)
    parser.add_argument("--emission_law", choices=["RD_schottky", "unified"], default="RD_schottky")
    parser.add_argument("--t_cathode_k", type=float, default=1700.0)
    parser.add_argument("--phi_eff_ev", type=float, default=2.1)
    parser.add_argument("--beta_f", type=float, default=1.0)
    parser.add_argument("--emission_phase_range", type=float, default=180.0)

    parser.add_argument("--poll_interval_s", type=float, default=0.5)
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-screen-json", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "quick":
        return
    args.n_particles = 1_000
    args.no_screens = False
    args.sc_enabled = True
    args.sc_dt_mm = 0.2
    args.beam_loading = True
    args.dt_mm = 0.1
    args.cfx_dt_mm = 0.1


def set_thread_environment(threads: int) -> None:
    os.environ["RF_TRACK_NUMBER_OF_THREADS"] = str(int(threads))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def resolve_threads(args: argparse.Namespace) -> int:
    if args.threads is not None:
        return max(1, int(args.threads))
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    try:
        return max(1, int(slurm_cpus))
    except Exception:
        return 1


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size <= 64:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.nanmin(value)) if np.issubdtype(value.dtype, np.number) else None,
            "max": float(np.nanmax(value)) if np.issubdtype(value.dtype, np.number) else None,
        }
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def format_duration(seconds: float) -> str:
    s = float(seconds)
    if not np.isfinite(s):
        return "n/a"
    if s < 1.0:
        return f"{1e3 * s:.0f} ms"
    if s < 120.0:
        return f"{s:.2f} s"
    m = int(s // 60)
    return f"{m} min {s - 60 * m:.1f} s"


def save_figure(fig, output_dir: Path, name: str) -> Tuple[Path, Path]:
    png_path = output_dir / f"{name}.png"
    eps_path = output_dir / f"{name}.eps"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    return png_path, eps_path


def save_screen_distributions_json(
    output_dir: Path,
    save_enabled: bool,
    z_snaps: Sequence[float],
    M_snaps: Sequence[np.ndarray],
    I_snaps: Sequence[Any] | None,
) -> int:
    if not save_enabled or not z_snaps or not M_snaps:
        return 0

    out_dir = output_dir / "screen_distributions_json"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, (z_m, M) in enumerate(zip(z_snaps, M_snaps)):
        arr = np.asarray(M)
        payload: Dict[str, Any] = {
            "screen_index": int(i),
            "z_m": float(z_m),
            "columns": ["x_mm", "px_MeV_c", "y_mm", "py_MeV_c", "z_mm", "pz_MeV_c"],
            "n_particles": int(arr.shape[0]) if arr.ndim == 2 else 0,
            "phase_space": arr.tolist() if arr.size else [],
        }

        if I_snaps is not None and i < len(I_snaps):
            info_i = I_snaps[i]
            if info_i is not None:
                payload["screen_info"] = str(info_i)

        file_path = out_dir / f"screen_{i:04d}_z_{float(z_m):.6f}m.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        saved += 1

    return saved


def _make_phase_space_fig(M: np.ndarray, x_idx: int, y_idx: int, x_label: str, y_label: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if M.ndim == 2 and M.shape[0] > 0 and M.shape[1] > max(x_idx, y_idx):
        x = M[:, x_idx]
        y = M[:, y_idx]
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size > 0:
            ax.hexbin(x, y, gridsize=90, bins="log", cmap="viridis")
        else:
            ax.text(0.5, 0.5, "No finite particles", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No particles", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _twiss_from_moments(u: np.ndarray, pu: np.ndarray) -> Tuple[float, float]:
    if u.size < 2 or pu.size < 2:
        return np.nan, np.nan
    u0 = u - np.mean(u)
    pu0 = pu - np.mean(pu)
    s11 = float(np.mean(u0 * u0))
    s22 = float(np.mean(pu0 * pu0))
    s12 = float(np.mean(u0 * pu0))
    det = s11 * s22 - s12 * s12
    if not np.isfinite(det) or det <= 0.0:
        return np.nan, np.nan
    eps = np.sqrt(det)
    alpha = -s12 / eps
    beta = s11 / eps
    return float(alpha), float(beta)


def _info_get_first(info: Any, keys: Sequence[str]) -> float:
    if info is None:
        return np.nan
    for key in keys:
        val = np.nan
        if isinstance(info, dict):
            val = info.get(key, info.get(key.lower(), info.get(key.upper(), np.nan)))
        else:
            for candidate in (key, key.lower(), key.upper(), f"get_{key}"):
                if hasattr(info, candidate):
                    attr = getattr(info, candidate)
                    try:
                        val = attr() if callable(attr) else attr
                    except Exception:
                        val = np.nan
                    if np.isfinite(float(val)):
                        break
        try:
            fval = float(val)
        except Exception:
            continue
        if np.isfinite(fval):
            return fval
    return np.nan


def _make_emission_diagnostics_fig(thermo_info: Dict[str, Any]):
    import matplotlib.pyplot as plt

    t_s = thermo_info.get("t_s", None)
    if t_s is None:
        return None

    t_ns = np.asarray(t_s, dtype=float) * 1e9
    I_t = thermo_info.get("I_A_t", None)
    J_t = thermo_info.get("J_Apm2_t", None)
    F_t = thermo_info.get("F_t", None)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    if F_t is not None:
        axes[0].plot(t_ns, np.asarray(F_t, dtype=float), lw=1.5)
        axes[0].set_ylabel("F [V/m]")
    else:
        axes[0].text(0.5, 0.5, "F(t) unavailable", transform=axes[0].transAxes, ha="center", va="center")
    axes[0].grid(alpha=0.3)

    if J_t is not None:
        axes[1].plot(t_ns, np.asarray(J_t, dtype=float) * 1e-4, lw=1.5)
        axes[1].set_ylabel("J [A/cm^2]")
    else:
        axes[1].text(0.5, 0.5, "J(t) unavailable", transform=axes[1].transAxes, ha="center", va="center")
    axes[1].grid(alpha=0.3)

    if I_t is not None:
        axes[2].plot(t_ns, np.asarray(I_t, dtype=float), lw=1.5)
        axes[2].set_ylabel("I [A]")
    else:
        axes[2].text(0.5, 0.5, "I(t) unavailable", transform=axes[2].transAxes, ha="center", va="center")
    axes[2].set_xlabel("t [ns]")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Emission diagnostics")
    fig.tight_layout()
    return fig


def _make_longitudinal_evolution_fig(M_snaps: Sequence[np.ndarray], z_snaps: Sequence[float]):
    import matplotlib.pyplot as plt

    if not M_snaps or not z_snaps or len(M_snaps) != len(z_snaps):
        return None

    z_mm = 1e3 * np.asarray(z_snaps, dtype=float)
    mean_pz = []
    sig_pz = []
    for M in M_snaps:
        if M.ndim == 2 and M.shape[0] > 0 and M.shape[1] > 5:
            mean_pz.append(float(np.mean(M[:, 5])))
            sig_pz.append(float(np.std(M[:, 5])))
        else:
            mean_pz.append(np.nan)
            sig_pz.append(np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(z_mm, mean_pz, "o-", ms=3)
    axes[0].set_ylabel("mean pz [MeV/c]")
    axes[0].grid(alpha=0.3)
    axes[1].plot(z_mm, sig_pz, "o-", ms=3, color="tab:orange")
    axes[1].set_ylabel("sigma pz [MeV/c]")
    axes[1].set_xlabel("z [mm]")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Longitudinal evolution")
    fig.tight_layout()
    return fig


def _make_twiss_evolution_fig(M_snaps: Sequence[np.ndarray], z_snaps: Sequence[float]):
    import matplotlib.pyplot as plt

    if not M_snaps or not z_snaps or len(M_snaps) != len(z_snaps):
        return None

    z_mm = 1e3 * np.asarray(z_snaps, dtype=float)
    alpha_x, beta_x, alpha_y, beta_y, alpha_z, beta_z = [], [], [], [], [], []
    for M in M_snaps:
        if M.ndim == 2 and M.shape[0] > 1 and M.shape[1] > 5:
            ax, bx = _twiss_from_moments(M[:, 0], M[:, 1])
            ay, by = _twiss_from_moments(M[:, 2], M[:, 3])
            az, bz = _twiss_from_moments(M[:, 4], M[:, 5])
        else:
            ax = bx = ay = by = az = bz = np.nan
        alpha_x.append(ax)
        beta_x.append(bx)
        alpha_y.append(ay)
        beta_y.append(by)
        alpha_z.append(az)
        beta_z.append(bz)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].plot(z_mm, alpha_x, "o-", ms=3, label="alpha_x")
    axes[0, 0].plot(z_mm, alpha_y, "o-", ms=3, label="alpha_y")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_ylabel("alpha")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(z_mm, beta_x, "o-", ms=3, label="beta_x")
    axes[0, 1].plot(z_mm, beta_y, "o-", ms=3, label="beta_y")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].set_ylabel("beta")
    axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(z_mm, alpha_z, "o-", ms=3, color="tab:blue")
    axes[1, 0].set_ylabel("alpha_z")
    axes[1, 0].set_xlabel("z [mm]")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(z_mm, beta_z, "o-", ms=3, color="tab:orange")
    axes[1, 1].set_ylabel("beta_z")
    axes[1, 1].set_xlabel("z [mm]")
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle("Twiss parameter evolution")
    fig.tight_layout()
    return fig


def _make_screen_snapshot_fig(M_snaps: Sequence[np.ndarray], z_snaps: Sequence[float], I_snaps: Sequence[Any]):
    import matplotlib.pyplot as plt

    if not M_snaps or not z_snaps or len(M_snaps) != len(z_snaps):
        return None

    z_mm = 1e3 * np.asarray(z_snaps, dtype=float)
    n_particles = []
    mean_pz = []
    transmission = []

    for idx, M in enumerate(M_snaps):
        if M.ndim == 2 and M.shape[0] > 0 and M.shape[1] > 5:
            n_particles.append(float(M.shape[0]))
            mean_pz.append(float(np.mean(M[:, 5])))
        else:
            n_particles.append(0.0)
            mean_pz.append(np.nan)

        info = I_snaps[idx] if I_snaps is not None and idx < len(I_snaps) else None
        transmission.append(_info_get_first(info, ["transmission"]))

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(z_mm, n_particles, "o-", ms=3)
    axes[0].set_ylabel("N")
    axes[0].grid(alpha=0.3)
    axes[1].plot(z_mm, mean_pz, "o-", ms=3, color="tab:orange")
    axes[1].set_ylabel("mean pz [MeV/c]")
    axes[1].grid(alpha=0.3)
    axes[2].plot(z_mm, transmission, "o-", ms=3, color="tab:green")
    axes[2].set_ylabel("transmission")
    axes[2].set_xlabel("z [mm]")
    axes[2].grid(alpha=0.3)
    fig.suptitle("Screen snapshots")
    fig.tight_layout()
    return fig


def _make_beam_loading_fig(z_snaps: Sequence[float], I_snaps: Sequence[Any]):
    import matplotlib.pyplot as plt

    if not z_snaps or not I_snaps or len(z_snaps) != len(I_snaps):
        return None

    z_mm = 1e3 * np.asarray(z_snaps, dtype=float)
    transmission = np.asarray([_info_get_first(info, ["transmission"]) for info in I_snaps], dtype=float)
    mean_p = np.asarray([_info_get_first(info, ["mean_P", "mean_Pz", "mean_pz"]) for info in I_snaps], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(z_mm, transmission, "o-", ms=3)
    axes[0].set_ylabel("transmission")
    axes[0].grid(alpha=0.3)
    axes[1].plot(z_mm, mean_p, "o-", ms=3, color="tab:red")
    axes[1].set_ylabel("mean P")
    axes[1].set_xlabel("z [mm]")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Beam loading evolution")
    fig.tight_layout()
    return fig


def generate_and_save_figures(
    output_dir: Path,
    save_figures_enabled: bool,
    m0: np.ndarray,
    mf: np.ndarray,
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    I_snaps: Sequence[Any],
    thermo_info: Dict[str, Any],
    beam_loading_enabled: bool,
) -> List[str]:
    if not save_figures_enabled:
        return []

    import matplotlib.pyplot as plt

    saved: List[str] = []

    figure_specs = [
        ("fig_01_initial_phase_space_x_px", _make_phase_space_fig(m0, 0, 1, "x [mm]", "px [MeV/c]", "Initial phase space: x-px")),
        ("fig_02_initial_phase_space_y_py", _make_phase_space_fig(m0, 2, 3, "y [mm]", "py [MeV/c]", "Initial phase space: y-py")),
        ("fig_03_initial_phase_space_z_pz", _make_phase_space_fig(m0, 4, 5, "z [mm]", "pz [MeV/c]", "Initial phase space: z-pz")),
        ("fig_04_final_phase_space_x_px", _make_phase_space_fig(mf, 0, 1, "x [mm]", "px [MeV/c]", "Final phase space: x-px")),
        ("fig_05_final_phase_space_y_py", _make_phase_space_fig(mf, 2, 3, "y [mm]", "py [MeV/c]", "Final phase space: y-py")),
        ("fig_06_final_phase_space_z_pz", _make_phase_space_fig(mf, 4, 5, "z [mm]", "pz [MeV/c]", "Final phase space: z-pz")),
        ("fig_07_longitudinal_evolution", _make_longitudinal_evolution_fig(M_snaps, z_snaps)),
        ("fig_08_twiss_evolution", _make_twiss_evolution_fig(M_snaps, z_snaps)),
        ("fig_09_screen_snapshots", _make_screen_snapshot_fig(M_snaps, z_snaps, I_snaps)),
        ("fig_10_emission_diagnostics", _make_emission_diagnostics_fig(thermo_info)),
    ]

    if beam_loading_enabled:
        figure_specs.append(("fig_11_beam_loading_evolution", _make_beam_loading_fig(z_snaps, I_snaps)))

    for name, fig in figure_specs:
        if fig is None:
            continue
        png, eps = save_figure(fig, output_dir, name)
        saved.append(png.name)
        saved.append(eps.name)
        plt.close(fig)

    return saved


def main() -> None:
    t_sim_start = time.time()

    args = parse_args()
    apply_preset(args)
    args.threads = resolve_threads(args)
    set_thread_environment(args.threads)

    import RF_Track as rft
    import rf_gun as rg
    from rf_params import delivered_power_on_resonance

    try:
        rft.cvar.number_of_threads = int(args.threads)
    except Exception:
        pass

    print(f"RF-Track max threads: {rft.max_number_of_threads}")
    print(f"RF-Track chosen threads: {getattr(rft.cvar, 'number_of_threads', 'n/a')}")
    print("---- Simulation Main Parameters ----")
    print(f"Particles: {int(args.n_particles):,}")
    print(f"N_Z_SNAP: {int(args.n_z_snap) if args.n_z_snap is not None else int(args.n_screens)}")
    print(f"EMISSION_PHASE_START: {float(args.emission_phase_start):.3f} deg")
    print(f"EMISSION_PHASE_RANGE: {float(args.emission_phase_range):.3f} deg")
    print(f"T_CATHODE_K: {float(args.t_cathode_k):.1f}")
    print(f"PHI_EFF_EV: {float(args.phi_eff_ev):.3f}")
    print(f"BETA_F: {float(args.beta_f):.3f}")
    print(f"Space charge enabled: {bool(args.sc_enabled)}")
    print(f"Beam loading enabled: {bool(args.beam_loading)}")
    print(f"dt_mm: {float(args.dt_mm)}")
    print(f"sc_dt_mm: {float(args.sc_dt_mm)}")
    print(f"cfx_dt_mm: {float(args.cfx_dt_mm)}")
    print(f"Env RF_TRACK_NUMBER_OF_THREADS={os.environ.get('RF_TRACK_NUMBER_OF_THREADS')}")
    print(
        "BLAS thread env: "
        f"OMP={os.environ.get('OMP_NUM_THREADS')} "
        f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
        f"MKL={os.environ.get('MKL_NUM_THREADS')} "
        f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}"
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    xy = rg.load_fieldmap_mat(str(args.xy_fieldmap), verbose=False)
    yz = rg.load_fieldmap_mat(str(args.yz_fieldmap), verbose=False)

    t_ns = yz["time"].astype(np.float64)
    t_ns = t_ns - t_ns[0]
    ez_rms = np.sqrt(np.mean(yz["Ez"] ** 2, axis=0))

    f_hz = float(args.f_hz)
    lambda_m = rg.c / f_hz
    z_max = float(args.z_max) if args.z_max is not None else (lambda_m / 4.0 + float(args.ext_zmax))
    z_min = float(args.z_min)

    x_mm = xy["vertices"][:, 0]
    y_mm = xy["vertices"][:, 1]
    r_m = np.abs(x_mm) * 1e-3
    z_m = (float(args.y_cathode_mm) - y_mm) * 1e-3

    t_maps_start = time.time()
    mode = str(args.phasor_mode).strip().lower()
    t_phasor_start = time.time()
    if mode == "reconstruct":
        i0, i90, _, _ = rg.select_iq_snapshots(t_ns, ez_rms, f_hz)
        ex_0 = xy["Ex"][:, i0]
        ex_90 = xy["Ex"][:, i90]
        ey_0 = xy["Ey"][:, i0]
        ey_90 = xy["Ey"][:, i90]

        ey_max_0 = np.max(np.abs(ey_0))
        ey_max_90 = np.max(np.abs(ey_90))
        ex_max_0 = np.max(np.abs(ex_0))
        ex_max_90 = np.max(np.abs(ex_90))
        e_ref = max(ey_max_0, ey_max_90)

        ex_phasor = rg.build_iq_phasor(ex_0, ex_90, ex_max_0, ex_max_90, e_ref)
        ey_phasor = rg.build_iq_phasor(ey_0, ey_90, ey_max_0, ey_max_90, e_ref)
    else:
        i_crest = int(np.argmax(ez_rms))
        ex_crest = xy["Ex"][:, i_crest]
        ey_crest = xy["Ey"][:, i_crest]
        e_ref = float(np.max(np.abs(ey_crest))) if ey_crest.size else 1.0
        ex_phasor = rg.build_crest_phasor(ex_crest, scale=e_ref)
        ey_phasor = rg.build_crest_phasor(ey_crest, scale=e_ref)
    t_phasor_elapsed = time.time() - t_phasor_start

    er_vertices = np.sign(x_mm) * ex_phasor
    ez_vertices = ey_phasor

    dr_um = float(args.dr_um)
    dz_um = float(args.dz_um)
    nr = int(float(args.r_max_m) * 1e6 / dr_um) + 1
    nz = int((z_max - z_min) * 1e6 / dz_um) + 1

    r_grid = np.linspace(0.0, float(args.r_max_m), nr)
    z_grid = np.linspace(z_min, z_max, nz)
    z_grid[np.argmin(np.abs(z_grid))] = 0.0
    R, Z = np.meshgrid(r_grid, z_grid)

    hr = float(r_grid[1] - r_grid[0]) if r_grid.size > 1 else 0.0
    hz = float(z_grid[1] - z_grid[0]) if z_grid.size > 1 else 0.0
    pts = np.column_stack([r_m, z_m])

    t_interp_start = time.time()
    er_grid = rg.interp_cfield(pts, R, Z, er_vertices)
    ez_grid = rg.interp_cfield(pts, R, Z, ez_vertices)
    ez0_phasor_axis = rg.find_Ez_axis_phasor_at_z0(ez_grid, z_grid, z0_m=0.0)
    t_interp_elapsed = time.time() - t_interp_start
    t_maps_elapsed = time.time() - t_maps_start

    print("Field maps generated:")
    print(f"  Phasor mode: {mode}")
    print(f"  Grid size: NR={nr}, NZ={nz} (shape={ez_grid.shape[0]}x{ez_grid.shape[1]})")
    print(f"  Resolution: dr={dr_um:.3f} um, dz={dz_um:.3f} um")
    print(
        f"  Extents: r=[0.000, {float(args.r_max_m) * 1e3:.3f}] mm, "
        f"z=[{z_min * 1e3:.3f}, {z_max * 1e3:.3f}] mm"
    )
    print(
        "  Timing: "
        f"phasor={format_duration(t_phasor_elapsed)}, "
        f"interpolation={format_duration(t_interp_elapsed)}, "
        f"total={format_duration(t_maps_elapsed)}"
    )

    q_loaded = 1.0 / (1.0 / float(args.bl_qext) + 1.0 / float(args.bl_q0))
    p_del_w = delivered_power_on_resonance(float(args.bl_p_fwd_w), float(args.bl_q0), float(args.bl_qext))
    print(f"Loaded Q={q_loaded:.2f}, delivered power={p_del_w/1e6:.3f} MW")

    vol_params = rg.VolumeBuildParams(
        f_hz=f_hz,
        map_z0_m=z_min,
        z_min_m=z_min,
        z_max_m=z_max,
        hr_m=hr,
        hz_m=hz,
        dt_mm=float(args.dt_mm),
        ode_algorithm=str(args.ode_algorithm),
        ode_epsabs=float(args.ode_epsabs),
        aperture_m=float(args.aperture_m),
        sc_enabled=bool(args.sc_enabled),
        sc_dt_mm=float(args.sc_dt_mm),
        emission_nsteps=int(args.emission_nsteps),
        emission_range=float(args.emission_range),
        fm_nsteps=int(args.fm_nsteps),
        fm_tt_nsteps=int(args.fm_tt_nsteps),
        cfx_dt_mm=float(args.cfx_dt_mm),
        beam_loading_enabled=bool(args.beam_loading),
        bl_Q_loaded=float(q_loaded),
        bl_r_over_q_ohm_per_m=float(args.bl_r_over_q_ohm_per_m),
        bl_ncells=int(args.bl_ncells),
        bl_tinj_mode=str(args.bl_tinj_mode),
        bl_tinj_manual_mm_c=float(args.bl_tinj_manual_mm_c),
    )

    pz_model = "constant" if bool(args.use_const_pz) else "flux"
    cathode_radius_mm = float(args.r_cathode_mm) / max(1e-12, float(args.emission_scale))

    roughness = rg.RoughnessParams(Ra_um=float(args.ra_um), Re_um=float(args.re_um))
    emission = rg.EmissionParams(
        cathode_radius_mm=cathode_radius_mm,
        cathode_T_K=float(args.t_cathode_k),
        work_function_eV=float(args.phi_eff_ev),
        beta_field=float(args.beta_f),
        emission_phase_range_deg=float(args.emission_phase_range),
        pz0_MeV_c=float(args.pz_init_mevc),
        pz_model=pz_model,
        emission_law=str(args.emission_law),
        beta_enh=float(args.beta_f),
        roughness=roughness,
        time_dependent=True,
    )

    if args.no_screens:
        z_snaps = None
    elif args.screens_z:
        z_snaps = [float(z) for z in args.screens_z]
    else:
        n_snap = int(args.n_z_snap) if args.n_z_snap is not None else int(args.n_screens)
        n_screens = max(0, n_snap)
        if n_screens <= 0:
            z_snaps = None
        elif n_screens == 1:
            z_snaps = [float(z_max)]
        else:
            z_snaps = np.linspace(z_min, z_max, n_screens).tolist()

    phase_deg_transport = float(args.phase_deg) + float(args.emission_phase_start)

    tracking = rg.TrackingParams(
        phi_deg=float(phase_deg_transport),
        n_particles=int(args.n_particles),
        z_screens_m=z_snaps,
        phase_fmt="%X %Px %Y %Py %Z %Pz",
        screen_width_mm=args.screen_width_mm,
        screen_height_mm=args.screen_height_mm,
        screen_time_window_mm_c=args.screen_time_window_mm_c,
        screen_t0_mode=str(args.screen_t0_mode),
        screen_t0_manual_mm_c=float(args.screen_t0_manual_mm_c),
        screen_log=bool(args.screen_log),
    )

    result, progress_stats = rg.run_transport_with_progress(
        rft,
        er_grid,
        ez_grid,
        ez0_phasor_axis,
        vol_params,
        emission,
        tracking,
        use_coarse_progress_proxy=True,
        poll_interval_s=float(args.poll_interval_s),
    )

    phase_fmt = "%X %Px %Y %Py %Z %Pz"
    m0 = np.array(result.B0.get_phase_space(phase_fmt, "all"), copy=True)
    mf = np.array(result.Bout.get_phase_space(phase_fmt, "all"), copy=True)
    m_snaps_obj = np.array([np.array(m, copy=True) for m in result.M_snaps], dtype=object)
    z_snaps_arr = np.asarray(result.z_snaps, dtype=float)

    npz_path = output_dir / "beam_data.npz"
    np.savez_compressed(
        npz_path,
        M0=m0,
        Mf=mf,
        M_snaps=m_snaps_obj,
        z_snaps=z_snaps_arr,
    )

    thermo_summary = {
        k: to_jsonable(v)
        for k, v in dict(result.thermo_info).items()
        if k not in {"t_s", "Ez_t", "F_t", "dphi_eV_t", "phi_eff_eV_t", "J_Apm2_t", "J_th_Apm2_t", "J_fe_Apm2_t", "R_t", "n_t", "I_A_t", "Q_cum_C", "t_emit_s"}
    }

    saved_figures = generate_and_save_figures(
        output_dir=output_dir,
        save_figures_enabled=bool(args.save_figures),
        m0=m0,
        mf=mf,
        M_snaps=list(result.M_snaps),
        z_snaps=list(result.z_snaps),
        I_snaps=list(result.I_snaps),
        thermo_info=dict(result.thermo_info),
        beam_loading_enabled=bool(args.beam_loading),
    )
    saved_screen_json = save_screen_distributions_json(
        output_dir=output_dir,
        save_enabled=bool(args.save_screen_json),
        z_snaps=list(result.z_snaps),
        M_snaps=list(result.M_snaps),
        I_snaps=list(result.I_snaps),
    )

    run_metadata: Dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "args": to_jsonable(vars(args)),
        "rftrack": {
            "max_number_of_threads": to_jsonable(getattr(rft, "max_number_of_threads", None)),
            "number_of_threads": to_jsonable(getattr(rft.cvar, "number_of_threads", None)),
        },
        "thread_env": {
            "RF_TRACK_NUMBER_OF_THREADS": os.environ.get("RF_TRACK_NUMBER_OF_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "grid": {
            "nr": int(nr),
            "nz": int(nz),
            "dr_um": float(dr_um),
            "dz_um": float(dz_um),
            "z_min_m": float(z_min),
            "z_max_m": float(z_max),
        },
        "counts": {
            "N0": int(m0.shape[0]) if m0.ndim == 2 else 0,
            "Nf": int(mf.shape[0]) if mf.ndim == 2 else 0,
            "n_screens": int(len(z_snaps_arr)),
        },
        "progress_stats": to_jsonable(progress_stats),
        "timing": {
            "total_simulation_s": float(time.time() - t_sim_start),
            "field_map_total_s": float(t_maps_elapsed),
            "field_map_phasor_s": float(t_phasor_elapsed),
            "field_map_interpolation_s": float(t_interp_elapsed),
        },
        "thermo_info": thermo_summary,
        "saved_figures": saved_figures,
        "saved_screen_json_count": int(saved_screen_json),
    }

    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, sort_keys=True)

    progress_path = output_dir / "progress_stats.json"
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(progress_stats), f, indent=2, sort_keys=True)

    t_sim_elapsed = time.time() - t_sim_start
    print(f"\nRun complete, simulation time: {format_duration(t_sim_elapsed)}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Saved: {npz_path.name}, {metadata_path.name}, {progress_path.name}")
    if saved_figures:
        print(f"Saved {len(saved_figures)} figure files (.png/.eps)")
    if saved_screen_json:
        print(f"Saved {saved_screen_json} per-screen JSON files")


if __name__ == "__main__":
    main()
