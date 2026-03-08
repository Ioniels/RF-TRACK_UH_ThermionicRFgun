#!/usr/bin/env python3
"""Batch-friendly thermionic TM010 RF-gun run (no Jupyter dependencies)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
    parser.add_argument("--emission_phase_start", type=float, default=45.0)
    parser.add_argument("--n_particles", type=int, default=100_000)
    parser.add_argument("--run-family", type=str, default="thermionic")
    parser.add_argument("--scan-tags", type=str, nargs="*", default=None)

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

    parser.add_argument("--dt_mm", type=float, default=0.01)
    parser.add_argument("--sc_dt_mm", type=float, default=0.01)
    parser.add_argument("--emission_nsteps", type=int, default=200)
    parser.add_argument("--emission_range", type=float, default=10.0)
    parser.add_argument("--fm_nsteps", type=int, default=200)
    parser.add_argument("--fm_tt_nsteps", type=int, default=200)
    parser.add_argument("--cfx_dt_mm", type=float, default=0.01)
    parser.add_argument("--ode_algorithm", type=str, default="rk2")
    parser.add_argument("--ode_epsabs", type=float, default=1e-6)
    parser.add_argument("--aperture_m", type=float, default=1.0)

    parser.add_argument("--sc_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--beam_loading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bl_q0", type=float, default=4000.0)
    parser.add_argument("--bl_qext", type=float, default=3500.0)
    parser.add_argument("--bl_p_fwd_w", type=float, default=1.0e6)
    parser.add_argument("--bl_r_over_q_ohm_per_m", type=float, default=1.0)
    parser.add_argument("--bl_ncells", type=int, default=1)
    parser.add_argument("--bl_tinj_mode", choices=["auto_from_emission", "manual"], default="auto_from_emission")
    parser.add_argument("--bl_tinj_manual_mm_c", type=float, default=0.0)

    parser.add_argument("--n_screens", type=int, default=0)
    parser.add_argument("--n_z_snap", type=int, default=3)
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
    parser.add_argument("--ra_um", type=float, default=0.0)
    parser.add_argument("--re_um", type=float, default=0.0)
    parser.add_argument("--emission_law", choices=["RD_schottky", "unified"], default="RD_schottky")
    parser.add_argument("--t_cathode_k", type=float, default=1700.0)
    parser.add_argument("--phi_eff_ev", type=float, default=2.1)
    parser.add_argument("--beta_f", type=float, default=1.0)
    parser.add_argument("--emission_phase_range", type=float, default=90.0)

    parser.add_argument("--phase_scan_min", type=float, default=0.0)
    parser.add_argument("--phase_scan_max", type=float, default=360.0)
    parser.add_argument("--phase_scan_n", type=int, default=90)
    parser.add_argument("--phase_scan_n_part", type=int, default=20)
    parser.add_argument("--phase_scan_dt_mm", type=float, default=0.5)

    parser.add_argument("--poll_interval_s", type=float, default=0.5)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-notebook-mode", choices=["minimal", "verbose", "auto"], default="auto")
    parser.add_argument("--timing-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--slow-step-warn-s", type=float, default=20.0)
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-screen-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-json-mode", choices=["summary", "full"], default="summary")
    parser.add_argument("--store-screen-phase-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--store-screen-info", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-stride", type=int, default=1)
    parser.add_argument("--screen-indices", type=int, nargs="*", default=None)
    parser.add_argument("--max-screen-particles", type=int, default=None)
    parser.add_argument("--subsample-screens-random", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-lost-particles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-beam-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-beam-summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-screen-phase-space-batch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-screen-phase-space-json", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-class-phase-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-e", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-except-zpz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-zle0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibrate-bl-r-over-q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--t_max_mm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "quick":
        return
    args.n_particles = 1_000
    # Keep quick preset aligned with notebook defaults for diagnostics.
    args.sc_enabled = True
    args.sc_dt_mm = 0.2
    args.beam_loading = True
    args.dt_mm = 0.1
    args.cfx_dt_mm = 0.1
    args.phase_scan_n = 90
    args.phase_scan_n_part = 2
    args.phase_scan_dt_mm = 2.0


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
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


def summarize_array(values: np.ndarray, *, with_span: bool = False) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    out: dict[str, Any] = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if with_span:
        out["span"] = None
    if finite.size == 0:
        return out
    out["min"] = float(np.min(finite))
    out["max"] = float(np.max(finite))
    out["mean"] = float(np.mean(finite))
    out["std"] = float(np.std(finite))
    if with_span:
        out["span"] = float(np.max(finite) - np.min(finite))
    return out


def _twiss_summary_from_phase_space(rg, M_snaps: list[np.ndarray]) -> dict[str, Any]:
    alpha_x = []
    beta_x = []
    alpha_y = []
    beta_y = []
    alpha_z = []
    beta_z = []
    for M in M_snaps:
        M_arr = np.asarray(M, dtype=float)
        if M_arr.ndim != 2 or M_arr.shape[0] < 2 or M_arr.shape[1] < 6:
            alpha_x.append(np.nan)
            beta_x.append(np.nan)
            alpha_y.append(np.nan)
            beta_y.append(np.nan)
            alpha_z.append(np.nan)
            beta_z.append(np.nan)
            continue
        ax, bx, _ = rg.twiss_from_moments(M_arr[:, 0], M_arr[:, 1])
        ay, by, _ = rg.twiss_from_moments(M_arr[:, 2], M_arr[:, 3])
        az, bz, _ = rg.twiss_from_moments(M_arr[:, 4], M_arr[:, 5])
        alpha_x.append(ax)
        beta_x.append(bx)
        alpha_y.append(ay)
        beta_y.append(by)
        alpha_z.append(az)
        beta_z.append(bz)
    return {
        "available": True,
        "alpha_x": summarize_array(np.asarray(alpha_x, dtype=float)),
        "beta_x": summarize_array(np.asarray(beta_x, dtype=float)),
        "alpha_y": summarize_array(np.asarray(alpha_y, dtype=float)),
        "beta_y": summarize_array(np.asarray(beta_y, dtype=float)),
        "alpha_z": summarize_array(np.asarray(alpha_z, dtype=float)),
        "beta_z": summarize_array(np.asarray(beta_z, dtype=float)),
    }


def _emittance_summary_from_info(rg, info_snaps: list[Any]) -> dict[str, Any]:
    geom_keys = {
        "x": ["emit_x", "emittance_x", "eps_x", "epsilon_x", "ex"],
        "y": ["emit_y", "emittance_y", "eps_y", "epsilon_y", "ey"],
        "z": ["emit_z", "emittance_z", "eps_z", "epsilon_z", "ez"],
    }
    norm_keys = {
        "x": ["emitt_x", "emit_nx", "emitnx", "norm_emit_x", "normalized_emittance_x", "eps_nx", "epsilon_nx"],
        "y": ["emitt_y", "emit_ny", "emitny", "norm_emit_y", "normalized_emittance_y", "eps_ny", "epsilon_ny"],
        "z": ["emitt_z", "emit_nz", "emitnz", "norm_emit_z", "normalized_emittance_z", "eps_nz", "epsilon_nz"],
    }
    out: dict[str, Any] = {"available": bool(len(info_snaps) > 0)}
    for axis in ("x", "y", "z"):
        geom = np.asarray([rg.info_get_first(info, geom_keys[axis]) for info in info_snaps], dtype=float)
        norm = np.asarray([rg.info_get_first(info, norm_keys[axis]) for info in info_snaps], dtype=float)
        out[f"eps_{axis}"] = summarize_array(geom)
        out[f"eps_n{axis}"] = summarize_array(norm)
    if not bool(len(info_snaps) > 0):
        out["note"] = "No RF-Track info snapshots were returned"
    return out


def _screen_summaries_from_arrays(
    *,
    rg,
    z_snaps: list[float],
    M_snaps: list[np.ndarray],
    I_snaps: list[Any],
    n_initial: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n_prev = int(n_initial)
    n = min(len(z_snaps), len(M_snaps))
    for i in range(n):
        rec = rg.build_screen_summary_from_phase_space(
            np.asarray(M_snaps[i], dtype=float),
            screen_index=i,
            z_m=float(z_snaps[i]),
            n_initial=int(n_initial),
            n_previous=n_prev,
        )
        info_i = I_snaps[i] if i < len(I_snaps) else None
        rec["rftrack_raw_info"] = {
            "transmission": rg.info_get_first(info_i, ["transmission", "Transmission"]),
            "mean_pz": rg.info_get_first(info_i, ["mean_Pz", "mean_P", "mean_pz"]),
            "sigma_pz": rg.info_get_first(info_i, ["sigma_Pz", "sigma_P", "sigma_pz"]),
        }
        out.append(rec)
        n_prev = int(rec.get("N", 0))
    return out


def _evolution_summary_from_screens(screens: list[dict[str, Any]]) -> dict[str, Any]:
    n_arr = np.asarray([float(rec.get("N", np.nan)) for rec in screens], dtype=float)
    tr_arr = np.asarray([float(rec.get("transmission_from_initial", np.nan)) for rec in screens], dtype=float)
    mpz_arr = np.asarray([float(rec.get("mean_pz_MeV_c", np.nan)) for rec in screens], dtype=float)
    spz_arr = np.asarray([float(rec.get("sigma_pz_MeV_c", np.nan)) for rec in screens], dtype=float)
    return {
        "N": summarize_array(n_arr),
        "transmission_from_initial": summarize_array(tr_arr),
        "mean_pz_MeV_c": summarize_array(mpz_arr),
        "sigma_pz_MeV_c": summarize_array(spz_arr),
    }


def _particle_classes_summary(classes: dict[str, Any], n_initial: int) -> dict[str, Any]:
    out: dict[str, Any] = {"n_initial": int(n_initial), "classes": {}}
    for key in ("transmitted", "backward_returned", "lost"):
        rec = dict(classes.get(key, {})) if isinstance(classes.get(key), dict) else {}
        count = int(rec.get("count", 0) or 0)
        frac = float(count / n_initial) if n_initial > 0 else None
        out["classes"][key] = {
            "count": count,
            "fraction": frac,
            "mean_initial_t0_mm_c": rec.get("initial_t0_mean_mm_c", None),
            "mean_initial_pz_MeV_c": rec.get("initial_pz_mean", None),
            "mean_final_z_mm": rec.get("final_z_mean_mm", None),
            "mean_final_pz_MeV_c": rec.get("final_pz_mean", None),
        }
    return out


def _consistency_warnings(
    *,
    n_initial: int,
    screen_summaries: list[dict[str, Any]],
    classes_summary: dict[str, Any],
    has_t0: bool,
    t0_summary: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if any((int(rec.get("N", -1)) < 0) for rec in screen_summaries):
        warnings.append("Negative screen particle count detected")
    for rec in screen_summaries:
        for key in ("transmission_from_initial", "transmission_from_previous"):
            val = rec.get(key, None)
            if val is None:
                continue
            if not (0.0 <= float(val) <= 1.0):
                warnings.append(f"{key} out of [0,1] at screen_index={rec.get('screen_index')}")

    cls = classes_summary.get("classes", {}) if isinstance(classes_summary, dict) else {}
    c_trans = int((cls.get("transmitted", {}) or {}).get("count", 0) or 0)
    c_back = int((cls.get("backward_returned", {}) or {}).get("count", 0) or 0)
    c_lost = int((cls.get("lost", {}) or {}).get("count", 0) or 0)
    if abs((c_trans + c_back + c_lost) - int(n_initial)) > 1:
        warnings.append("Class counts do not match initial bunch size within tolerance")

    if bool(has_t0):
        span = t0_summary.get("span", None)
        if span is None or not np.isfinite(float(span)) or float(span) <= 0.0:
            warnings.append("has_t0 is True but initial_t0_mm_c_summary.span is not positive")
    return warnings


def _build_beam_summary(
    *,
    rg,
    args: argparse.Namespace,
    run_name: str,
    result,
    M0: np.ndarray,
    Mf: np.ndarray,
    phase_deg_transport: float,
    phi_zero_deg: float,
    phi_crest_deg: float,
) -> dict[str, Any]:
    z_snaps = [float(z) for z in list(result.z_snaps)]
    M_snaps = [np.asarray(M, dtype=float) for M in list(result.M_snaps)]
    I_snaps = list(result.I_snaps)

    n_initial = int(M0.shape[0]) if M0.ndim == 2 else 0
    n_final = int(Mf.shape[0]) if Mf.ndim == 2 else 0
    screen_summaries = [dict(rec) for rec in list(getattr(result, "screen_summaries", []) or [])]
    if not screen_summaries:
        screen_summaries = _screen_summaries_from_arrays(
            rg=rg,
            z_snaps=z_snaps,
            M_snaps=M_snaps,
            I_snaps=I_snaps,
            n_initial=n_initial,
        )

    screen_z_mm = [1e3 * float(z) for z in z_snaps]

    particle_classes_raw = dict(result.particle_classes) if isinstance(result.particle_classes, dict) else {}
    particle_classes_summary = _particle_classes_summary(particle_classes_raw, n_initial)

    t0_arr = np.asarray(result.thermo_info.get("initial_t0_mm_c", []), dtype=float)
    t0_summary = summarize_array(t0_arr, with_span=True)
    pz0_summary = summarize_array(np.asarray(M0[:, 5], dtype=float) if M0.ndim == 2 and M0.shape[1] > 5 else np.asarray([], dtype=float))

    twiss_summary: dict[str, Any]
    if M_snaps:
        twiss_summary = _twiss_summary_from_phase_space(rg, M_snaps)
    else:
        twiss_summary = {"available": False, "reason": "Twiss data not returned in current run"}

    emittance_summary = _emittance_summary_from_info(rg, I_snaps)

    warnings = _consistency_warnings(
        n_initial=n_initial,
        screen_summaries=screen_summaries,
        classes_summary=particle_classes_summary,
        has_t0=bool(result.thermo_info.get("has_t0", False)),
        t0_summary=t0_summary,
    )

    c = particle_classes_summary.get("classes", {})
    trans_count = int((c.get("transmitted", {}) or {}).get("count", 0) or 0)
    back_count = int((c.get("backward_returned", {}) or {}).get("count", 0) or 0)
    lost_count = int((c.get("lost", {}) or {}).get("count", max(0, n_initial - n_final)) or 0)

    return {
        "run_name": str(run_name),
        "run_family": str(args.run_family),
        "scan_tags": [str(x) for x in (args.scan_tags or [])],
        "scanned_parameters": {
            "t_cathode_k": float(args.t_cathode_k),
            "emission_phase_start_deg": float(args.emission_phase_start),
            "emission_phase_range_deg": float(args.emission_phase_range),
            "sc_dt_mm": float(args.sc_dt_mm),
            "bl_cfx_dt_mm": float(args.cfx_dt_mm),
            "ra_um": float(args.ra_um),
            "re_um": float(args.re_um),
        },
        "particle_counts": {
            "requested": int(args.n_particles),
            "initial_bunch_rows": int(n_initial),
            "final_bunch_rows": int(n_final),
            "transmitted_count": trans_count,
            "backward_returned_count": back_count,
            "lost_count": lost_count,
        },
        "timing_emission": {
            "initial_t0_mm_c_summary": t0_summary,
            "initial_pz_MeV_c_summary": pz0_summary,
            "emission_phase_range_deg": float(args.emission_phase_range),
            "emission_phase_start_deg": float(args.emission_phase_start),
            "transport_phase_deg": float(phase_deg_transport),
            "phi_zero_deg": float(phi_zero_deg),
            "phi_crest_deg": float(phi_crest_deg),
            "has_t0": bool(result.thermo_info.get("has_t0", False)),
            "t0_readback_ok": result.thermo_info.get("t0_readback_ok", None),
        },
        "screen_count": int(len(screen_summaries)),
        "screen_positions_m": z_snaps,
        "screen_positions_mm": screen_z_mm,
        "screens": screen_summaries,
        "evolution_summary": _evolution_summary_from_screens(screen_summaries),
        "twiss_summary": twiss_summary,
        "emittance_summary": emittance_summary,
        "particle_classes_summary": particle_classes_summary,
        "consistency_warnings": warnings,
    }


def main() -> None:
    t_sim_start = time.time()

    import rf_gun as rg

    args = parse_args()
    apply_preset(args)
    rng = np.random.default_rng(int(args.seed) if args.seed is not None else None)

    if args.threads is not None:
        args.threads = rg.resolve_threads(requested=args.threads, default=1)
        rg.set_thread_environment(args.threads, pin_blas_threads=True)

    import RF_Track as rft
    from rf_gun.rf_params import (
        delivered_power_on_resonance,
        effective_length_from_abs_ez,
        r_over_q_per_m,
        veff_from_phase_scan_pz,
    )

    if args.threads is not None:
        try:
            rft.cvar.number_of_threads = int(args.threads)
        except Exception:
            pass

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "unset")
    rftrack_max_threads = getattr(rft, "max_number_of_threads", "n/a")
    rftrack_chosen_threads = getattr(rft.cvar, "number_of_threads", "n/a")

    print(f"SLURM_CPUS_PER_TASK: {slurm_cpus}")
    print(f"RF-Track max threads (detected): {rftrack_max_threads}")
    print(f"RF-Track chosen threads (effective): {rftrack_chosen_threads}")
    print("---- Simulation Main Parameters ----")
    print(f"Particles: {int(args.n_particles):,}")
    print(f"N_Z_SNAP: {int(args.n_z_snap) if args.n_z_snap is not None else int(args.n_screens)}")
    print(f"EMISSION_PHASE_START: {float(args.emission_phase_start):.3f} deg")
    print(f"EMISSION_PHASE_RANGE: {float(args.emission_phase_range):.3f} deg")
    print(f"RNG seed: {int(args.seed) if args.seed is not None else 'random'}")
    print(f"T_CATHODE_K: {float(args.t_cathode_k):.1f}")
    print(f"PHI_EFF_EV: {float(args.phi_eff_ev):.3f}")
    print(f"BETA_F: {float(args.beta_f):.3f}")
    print(f"Space charge enabled: {bool(args.sc_enabled)}")
    print(f"Beam loading enabled: {bool(args.beam_loading)}")
    print(f"dt_mm: {float(args.dt_mm)}")
    print(f"sc_dt_mm: {float(args.sc_dt_mm)}")
    print(f"cfx_dt_mm: {float(args.cfx_dt_mm)}")
    if args.threads is None:
        print("Thread policy: automatic detection")
    else:
        print(f"Thread policy: forced --threads={int(args.threads)}")
    print(f"Env RF_TRACK_NUMBER_OF_THREADS={os.environ.get('RF_TRACK_NUMBER_OF_THREADS', 'unset')}")
    print(
        "BLAS thread env: "
        f"OMP={os.environ.get('OMP_NUM_THREADS', 'unset')} "
        f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset')} "
        f"MKL={os.environ.get('MKL_NUM_THREADS', 'unset')} "
        f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS', 'unset')}"
    )
    if bool(args.timing_diagnostics):
        print(f"Timing diagnostics: ON (slow-step threshold={float(args.slow_step_warn_s):.2f} s)")
    else:
        print("Timing diagnostics: OFF")

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

    ez_axis = ez_grid[:, 0]
    l_eff_m = effective_length_from_abs_ez(z_grid, ez_axis, tail_frac=1e-3)
    phi_zero_deg = (90.0 - np.rad2deg(np.angle(ez0_phasor_axis))) % 360.0
    phi_crest_deg = (phi_zero_deg + 90.0) % 360.0
    phase_deg_transport = (phi_zero_deg + float(args.emission_phase_start) + float(args.phase_deg)) % 360.0

    print(f"Leff = {l_eff_m*1e3:.3f} mm")
    print(f"Auto phase: Ez0 crosses 0 at phi approx {phi_zero_deg:.2f} deg")
    print(f"Auto crest phase at cathode: phi approx {phi_crest_deg:.2f} deg")
    print(
        f"Transport phase (t=0): phi = {phase_deg_transport:.2f} deg "
        f"(zero-crossing reference + start shift {float(args.emission_phase_start):.1f} deg)"
    )
    print(f"Emission window: {float(args.emission_phase_range):.1f} deg")

    phase_scan_n = max(3, int(args.phase_scan_n))
    phase_scan_n_part = max(1, int(args.phase_scan_n_part))
    phase_scan_rel = np.linspace(float(args.phase_scan_min), float(args.phase_scan_max), phase_scan_n)
    vol_params_cal = rg.VolumeBuildParams(
        f_hz=f_hz,
        map_z0_m=z_min,
        z_min_m=z_min,
        z_max_m=z_max,
        hr_m=hr,
        hz_m=hz,
        dt_mm=float(args.phase_scan_dt_mm),
        ode_algorithm=str(args.ode_algorithm),
        ode_epsabs=float(args.ode_epsabs),
        aperture_m=float(args.aperture_m),
        sc_enabled=False,
        sc_dt_mm=float(args.sc_dt_mm),
        emission_nsteps=int(args.emission_nsteps),
        emission_range=float(args.emission_range),
        fm_nsteps=int(args.fm_nsteps),
        fm_tt_nsteps=int(args.fm_tt_nsteps),
        cfx_dt_mm=float(args.cfx_dt_mm),
        beam_loading_enabled=False,
        bl_Q_loaded=float(q_loaded),
        bl_r_over_q_ohm_per_m=float(args.bl_r_over_q_ohm_per_m),
        bl_ncells=int(args.bl_ncells),
        bl_tinj_mode=str(args.bl_tinj_mode),
        bl_tinj_manual_mm_c=float(args.bl_tinj_manual_mm_c),
    )
    print(
        "Phase scan calibration: "
        f"N={phase_scan_n}, N_part={phase_scan_n_part}, dt_mm={float(args.phase_scan_dt_mm):.3g}",
        flush=True,
    )
    t_phase_scan_start = time.time()
    _, _, _, pz_mean_scan = rg.run_phase_scan(
        rft,
        er_grid,
        ez_grid,
        vol_params_cal,
        phase_scan_rel,
        phase_deg_transport,
        phase_scan_n_part,
        float(args.r_cathode_mm),
        float(args.pz_init_mevc),
        q_total_C=1e-12,
        rng=rng,
    )
    t_phase_scan_elapsed = time.time() - t_phase_scan_start
    veff_v = veff_from_phase_scan_pz(np.asarray(pz_mean_scan, dtype=float), float(args.pz_init_mevc), me_MeV=rg.ME_MEV)
    r_over_q_ohm = (veff_v**2) / (p_del_w * q_loaded)
    bl_r_over_q_ohm_per_m = r_over_q_per_m(veff_v, p_del_w, q_loaded, l_eff_m)

    if bool(args.beam_loading) and bool(args.calibrate_bl_r_over_q):
        print("Beam-loading R/Q per m updated from phase scan.")
    else:
        bl_r_over_q_ohm_per_m = float(args.bl_r_over_q_ohm_per_m)
        print("Beam-loading R/Q per m kept fixed from CLI/default value.")
    print(f"Phase scan elapsed: {format_duration(t_phase_scan_elapsed)}")
    print(f"Veff = {veff_v/1e6:.6f} MV")
    print(f"(R/Q) from scan = {r_over_q_ohm:.3e} Ω")
    print(f"(R/Q)/m from scan = {bl_r_over_q_ohm_per_m:.3e} Ω/m")
    print(f"(R/Q)/m used in transport = {bl_r_over_q_ohm_per_m:.3e} Ω/m")

    tmax_default_mm = rg.estimate_default_tmax_mm(
        z_min,
        z_max,
        f_hz,
        float(args.emission_phase_range),
        safety_factor=2.0,
    )
    t_max_mm = float(args.t_max_mm) if args.t_max_mm is not None else float(tmax_default_mm)

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
        bl_r_over_q_ohm_per_m=float(bl_r_over_q_ohm_per_m),
        bl_ncells=int(args.bl_ncells),
        bl_tinj_mode=str(args.bl_tinj_mode),
        bl_tinj_manual_mm_c=float(args.bl_tinj_manual_mm_c),
        t_max_mm=t_max_mm,
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
            z_snaps = [0.5 * (float(z_min) + float(z_max))]
        else:
            z_snaps = np.linspace(float(z_min), float(z_max), n_screens + 2)[1:-1].tolist()

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

    diagnostics = rg.DiagnosticsParams(
        store_screen_phase_space=bool(args.store_screen_phase_space),
        store_screen_info=bool(args.store_screen_info),
        screen_stride=max(1, int(args.screen_stride)),
        screen_indices=args.screen_indices,
        max_screen_particles=args.max_screen_particles,
        subsample_screens_random=bool(args.subsample_screens_random),
        save_lost_particles=bool(args.save_lost_particles),
        use_transport_table_summary=True,
        transport_table_dt_mm=float(args.dt_mm),
        save_screen_json=bool(args.save_screen_json),
        screen_json_mode=str(args.screen_json_mode),
        save_npz=True,
    )

    result, progress_stats = rg.run_transport_with_progress(
        rft,
        er_grid,
        ez_grid,
        ez0_phasor_axis,
        vol_params,
        emission,
        tracking,
        diagnostics=diagnostics,
        progress_bar=bool(args.progress_bar),
        progress_notebook_mode=str(args.progress_notebook_mode),
        use_coarse_progress_proxy=True,
        poll_interval_s=float(args.poll_interval_s),
        timing_diagnostics=bool(args.timing_diagnostics),
        slow_step_warn_s=float(args.slow_step_warn_s),
        rng=rng,
    )

    phase_fmt = "%X %Px %Y %Py %Z %Pz"
    m0 = np.array(result.B0.get_phase_space(phase_fmt, "all"), copy=True)
    mf = np.array(result.Bout.get_phase_space(phase_fmt, "all"), copy=True)
    z_snaps_arr = np.asarray(result.z_snaps, dtype=float)

    npz_path = output_dir / "beam_data.npz"
    npz_payload: Dict[str, Any] = {"M0": m0, "Mf": mf, "z_snaps": z_snaps_arr}
    if result.transport_table is not None:
        npz_payload["transport_table"] = np.asarray(result.transport_table)
    if result.M_snaps:
        for i, M in enumerate(result.M_snaps):
            npz_payload[f"screen_phase_space_{i:04d}"] = np.asarray(M)
    np.savez_compressed(npz_path, **npz_payload)

    b0_t0 = np.asarray(result.thermo_info.get("initial_t0_mm_c", []), dtype=float)
    t0_summary = summarize_array(b0_t0, with_span=True)
    pz0_summary = summarize_array(np.asarray(m0[:, 5], dtype=float) if m0.ndim == 2 and m0.shape[1] > 5 else np.asarray([], dtype=float))

    b0_json_path = None
    bout_json_path = None
    b0_timing_path = output_dir / "B0_timing.json"
    if bool(args.save_beam_json):
        b0_json_path = rg.save_beam_phase_space_json(
            output_dir / "B0.json",
            m0,
            label="B0",
            extra_metadata={"t0_mm_c_summary": t0_summary},
        )
        bout_json_path = rg.save_beam_phase_space_json(output_dir / "Bout.json", mf, label="Bout")

    with b0_timing_path.open("w", encoding="utf-8") as f:
        json.dump(
            sanitize_for_json(
                {
                    "particle_count": int(m0.shape[0]) if m0.ndim == 2 else 0,
                    "t0_mm_c_summary": t0_summary,
                }
            ),
            f,
            indent=2,
            sort_keys=True,
        )

    thermo_summary = {
        k: sanitize_for_json(v)
        for k, v in dict(result.thermo_info).items()
        if k not in {"t_s", "Ez_t", "F_t", "dphi_eV_t", "phi_eff_eV_t", "J_Apm2_t", "J_th_Apm2_t", "J_fe_Apm2_t", "R_t", "n_t", "I_A_t", "Q_cum_C", "t_emit_s"}
    }
    thermo_summary["has_t0"] = True
    thermo_summary["bunch_constructor"] = "extended_matrix_with_T0"
    thermo_summary["bunch_constructor_full"] = "Bunch6dT_extended_matrix_with_T0"
    thermo_summary["initial_t0_mm_c_summary"] = t0_summary
    thermo_summary["initial_pz_MeV_c_summary"] = pz0_summary
    thermo_summary["emission_phase_start_deg"] = float(args.emission_phase_start)
    thermo_summary["emission_phase_range_deg"] = float(args.emission_phase_range)
    thermo_summary["phi_zero_deg"] = float(phi_zero_deg)
    thermo_summary["phi_crest_deg"] = float(phi_crest_deg)
    thermo_summary["timing_coordinate_note"] = (
        "In Bunch6dT, Z is particle position; T0 is creation time and is stored separately from the 6D phase-space coordinates."
    )

    saved_figures = []
    if bool(args.save_figures):
        n_real_ref = abs(float(result.thermo_info.get("Q_total_C", np.nan))) / rg.q_e if result.thermo_info else None
        saved_figures = rg.save_run_figures(
            output_dir=output_dir,
            B0=result.B0,
            Bout=result.Bout,
            transport_phase_deg=float(phase_deg_transport),
            thermo_info=dict(result.thermo_info),
            M_snaps=list(result.M_snaps),
            z_snaps=list(result.z_snaps),
            I_snaps=list(result.I_snaps),
            phase_fmt=phase_fmt,
            clean_e=bool(args.clean_e),
            clean_except_zpz=bool(args.clean_except_zpz),
            show_zle0=bool(args.show_zle0),
            n_real_ref=n_real_ref,
            n_macroparticles=int(args.n_particles),
            lost_table=result.lost_table,
        )

    screen_phase_space_batch = None
    if bool(args.save_screen_phase_space_batch):
        n_real_ref = abs(float(result.thermo_info.get("Q_total_C", np.nan))) / rg.q_e if result.thermo_info else None
        plot_style = rg.PlotStyleConfig(dezoom_frac=0.05)
        screen_phase_space_batch = rg.save_screen_phase_space_batch(
            output_dir=output_dir,
            M_snaps=list(result.M_snaps),
            z_snaps=list(result.z_snaps),
            info_snaps=list(result.I_snaps),
            B0=result.B0,
            Bout=result.Bout,
            phase_fmt=phase_fmt,
            clean_e=bool(args.clean_e),
            clean_except_zpz=bool(args.clean_except_zpz),
            n_real_ref=n_real_ref,
            n_macroparticles=int(args.n_particles),
            style=plot_style,
            highlight_mode="zlt0",
            show_colorbar=False,
            save_json=bool(args.save_screen_phase_space_json),
        )

    saved_screen_json = 0
    if bool(args.save_screen_json):
        saved_screen_json = rg.save_screen_distributions_json(
            output_dir=output_dir,
            z_snaps=list(result.z_snaps),
            M_snaps=list(result.M_snaps),
            I_snaps=list(result.I_snaps),
            mode=str(args.screen_json_mode),
            n_initial=int(m0.shape[0]) if m0.ndim == 2 else 0,
            robust_summaries=list(result.screen_summaries),
        )

    lost_path = rg.save_lost_particles_json(output_dir, result.lost_table) if bool(args.save_lost_particles) else None

    beam_summary_path = None
    beam_summary = None
    if bool(args.save_beam_summary):
        beam_summary = _build_beam_summary(
            rg=rg,
            args=args,
            run_name=output_dir.name,
            result=result,
            M0=m0,
            Mf=mf,
            phase_deg_transport=float(phase_deg_transport),
            phi_zero_deg=float(phi_zero_deg),
            phi_crest_deg=float(phi_crest_deg),
        )
        beam_summary_path = output_dir / "beam_summary.json"
        with beam_summary_path.open("w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(beam_summary), f, indent=2, sort_keys=True)

    classes_raw = dict(result.particle_classes) if isinstance(result.particle_classes, dict) else {}
    classes_summary = _particle_classes_summary(classes_raw, int(m0.shape[0]) if m0.ndim == 2 else 0)

    particle_classes_summary_path = output_dir / "particle_classes_summary.json"
    with particle_classes_summary_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(classes_summary), f, indent=2, sort_keys=True)

    if bool(args.save_class_phase_space):
        if mf.ndim == 2 and mf.shape[0] > 0 and mf.shape[1] >= 6:
            zf = np.asarray(mf[:, 4], dtype=float)
            pzf = np.asarray(mf[:, 5], dtype=float)
            mask_trans = np.isfinite(zf) & np.isfinite(pzf) & (zf >= 0.0) & (pzf > 0.0)
            mask_back = np.isfinite(zf) & np.isfinite(pzf) & ~mask_trans
            rg.save_beam_phase_space_json(output_dir / "B_transmitted.json", mf[mask_trans], label="B_transmitted")
            rg.save_beam_phase_space_json(output_dir / "B_backward_returned.json", mf[mask_back], label="B_backward_returned")

    robust_screen_summaries = [dict(rec) for rec in list(result.screen_summaries)]
    if not robust_screen_summaries:
        robust_screen_summaries = _screen_summaries_from_arrays(
            rg=rg,
            z_snaps=[float(z) for z in list(result.z_snaps)],
            M_snaps=[np.asarray(M, dtype=float) for M in list(result.M_snaps)],
            I_snaps=list(result.I_snaps),
            n_initial=int(m0.shape[0]) if m0.ndim == 2 else 0,
        )
    consistency_warnings = _consistency_warnings(
        n_initial=int(m0.shape[0]) if m0.ndim == 2 else 0,
        screen_summaries=robust_screen_summaries,
        classes_summary=classes_summary,
        has_t0=True,
        t0_summary=t0_summary,
    )
    for w in consistency_warnings:
        print(f"WARNING: {w}")

    ref_note = "RF-Track may switch to centroid reference if first particle is lost; robust summaries are computed from explicit phase-space arrays."
    ref_warn = bool(result.thermo_info.get("reference_particle_reordered", False))

    run_metadata: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_family": str(args.run_family),
        "scan_tags": [str(x) for x in (args.scan_tags or [])],
        "args": sanitize_for_json(vars(args)),
        "plotting_defaults": {
            "clean_e": bool(args.clean_e),
            "clean_except_zpz": bool(args.clean_except_zpz),
            "show_zle0": bool(args.show_zle0),
        },
        "rftrack": {
            "max_number_of_threads": sanitize_for_json(getattr(rft, "max_number_of_threads", None)),
            "number_of_threads": sanitize_for_json(getattr(rft.cvar, "number_of_threads", None)),
            "thread_policy": "automatic detection" if args.threads is None else f"forced ({int(args.threads)})",
        },
        "thread_env": {
            "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
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
        "derived": {
            "calibrate_bl_r_over_q": True,
            "L_eff_m": float(l_eff_m),
            "phi_zero_deg": float(phi_zero_deg),
            "phi_crest_deg": float(phi_crest_deg),
            "phase_deg_transport": float(phase_deg_transport),
            "bl_r_over_q_ohm_per_m_used": float(bl_r_over_q_ohm_per_m),
            "phase_scan_n": int(phase_scan_rel.size),
            "phase_scan_n_part": int(phase_scan_n_part),
            "t_max_mm": float(t_max_mm),
        },
        "counts": {
            "N0": int(m0.shape[0]) if m0.ndim == 2 else 0,
            "Nf": int(mf.shape[0]) if mf.ndim == 2 else 0,
            "n_screens": int(len(z_snaps_arr)),
        },
        "progress_stats": sanitize_for_json(progress_stats),
        "diagnostics": sanitize_for_json({
            "store_screen_phase_space": diagnostics.store_screen_phase_space,
            "store_screen_info": diagnostics.store_screen_info,
            "screen_stride": diagnostics.screen_stride,
            "screen_indices": diagnostics.screen_indices,
            "max_screen_particles": diagnostics.max_screen_particles,
            "subsample_screens_random": diagnostics.subsample_screens_random,
            "screen_json_mode": diagnostics.screen_json_mode,
        }),
        "timing": {
            "total_simulation_s": float(time.time() - t_sim_start),
            "field_map_total_s": float(t_maps_elapsed),
            "field_map_phasor_s": float(t_phasor_elapsed),
            "field_map_interpolation_s": float(t_interp_elapsed),
        },
        "thermo_info": thermo_summary,
        "screen_summaries": sanitize_for_json(robust_screen_summaries),
        "evolution_summary": sanitize_for_json(_evolution_summary_from_screens(robust_screen_summaries)),
        "particle_classes": sanitize_for_json(classes_summary),
        "warnings": consistency_warnings,
        "reference_particle_warning": bool(ref_warn),
        "reference_particle_note": ref_note,
        "saved_figures": saved_figures,
        "screen_phase_space_batch": sanitize_for_json(screen_phase_space_batch),
        "saved_screen_json_count": int(saved_screen_json),
        "beam_json": {
            "B0": str(b0_json_path) if b0_json_path is not None else None,
            "Bout": str(bout_json_path) if bout_json_path is not None else None,
            "B0_timing": str(b0_timing_path),
        },
        "particle_classes_summary_file": str(particle_classes_summary_path),
        "beam_summary_file": str(beam_summary_path) if beam_summary_path is not None else None,
        "lost_particles_file": str(lost_path) if lost_path is not None else None,
    }

    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(run_metadata), f, indent=2, sort_keys=True)

    progress_path = output_dir / "progress_stats.json"
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(progress_stats), f, indent=2, sort_keys=True)

    t_sim_elapsed = time.time() - t_sim_start
    print(f"\nRun complete, simulation time: {format_duration(t_sim_elapsed)}")
    n0 = int(m0.shape[0]) if m0.ndim == 2 else 0
    if n0 > 0 and len(result.M_snaps) > 0:
        first_n = int(np.asarray(result.M_snaps[0]).shape[0])
        last_n = int(np.asarray(result.M_snaps[-1]).shape[0])
        first_pct = 100.0 * first_n / n0
        last_pct = 100.0 * last_n / n0
        z_first = float(result.z_snaps[0])
        z_last = float(result.z_snaps[-1])
        print(
            f"Screen transmission: first screen (z={z_first*1e3:.3f} mm) = "
            f"{first_n}/{n0} ({first_pct:.2f}%)"
        )
        print(
            f"Screen transmission: last screen (z={z_last*1e3:.3f} mm) = "
            f"{last_n}/{n0} ({last_pct:.2f}%)"
        )
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Saved: {npz_path.name}, {metadata_path.name}, {progress_path.name}")
    if b0_json_path is not None and bout_json_path is not None:
        print(f"Saved beam JSON: {b0_json_path.name}, {bout_json_path.name}")
    if beam_summary_path is not None:
        print(f"Saved beam summary: {beam_summary_path.name}")
    if saved_figures:
        print(f"Saved {len(saved_figures)} figure files (.png/.eps)")
    if screen_phase_space_batch is not None:
        print(f"Saved cinematic phase-space frames: {int(screen_phase_space_batch.get('frame_count', 0))}")
    if saved_screen_json:
        print(f"Saved {saved_screen_json} per-screen JSON files")
    if lost_path is not None:
        print(f"Saved lost-particle diagnostics: {lost_path.name}")


if __name__ == "__main__":
    main()
