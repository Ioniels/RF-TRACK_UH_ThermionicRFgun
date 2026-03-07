#!/usr/bin/env python3
"""Batch-friendly thermionic TM010 RF-gun run (no Jupyter dependencies)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict

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

    parser.add_argument("--n_screens", type=int, default=0)
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

    parser.add_argument("--phase_scan_n", type=int, default=90)
    parser.add_argument("--phase_scan_n_part", type=int, default=20)
    parser.add_argument("--phase_scan_dt_mm", type=float, default=0.5)

    parser.add_argument("--poll_interval_s", type=float, default=0.5)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
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
    from rf_params import (
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

    print(f"RF-Track max threads: {rft.max_number_of_threads}")
    print(f"RF-Track chosen threads: {getattr(rft.cvar, 'number_of_threads', 'n/a')}")
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
        print("Thread policy: unmanaged (RF-Track/environment defaults)")
    else:
        print(f"Thread policy: forced --threads={int(args.threads)}")
    print(f"Env RF_TRACK_NUMBER_OF_THREADS={os.environ.get('RF_TRACK_NUMBER_OF_THREADS')}")
    print(
        "BLAS thread env: "
        f"OMP={os.environ.get('OMP_NUM_THREADS')} "
        f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
        f"MKL={os.environ.get('MKL_NUM_THREADS')} "
        f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}"
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
    phase_scan_rel = np.linspace(0.0, 360.0, phase_scan_n)
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

    print("Beam-loading R/Q per m updated from phase scan.")
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

    thermo_summary = {
        k: to_jsonable(v)
        for k, v in dict(result.thermo_info).items()
        if k not in {"t_s", "Ez_t", "F_t", "dphi_eV_t", "phi_eff_eV_t", "J_Apm2_t", "J_th_Apm2_t", "J_fe_Apm2_t", "R_t", "n_t", "I_A_t", "Q_cum_C", "t_emit_s"}
    }

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
            clean_e=False,
            show_zle0=True,
            n_real_ref=n_real_ref,
            n_macroparticles=int(args.n_particles),
            lost_table=result.lost_table,
        )

    saved_screen_json = 0
    if bool(args.save_screen_json):
        saved_screen_json = rg.save_screen_distributions_json(
            output_dir=output_dir,
            z_snaps=list(result.z_snaps),
            M_snaps=list(result.M_snaps),
            I_snaps=list(result.I_snaps),
            mode=str(args.screen_json_mode),
        )

    lost_path = rg.save_lost_particles_json(output_dir, result.lost_table) if bool(args.save_lost_particles) else None

    classes_summary = dict(result.particle_classes) if isinstance(result.particle_classes, dict) else {}
    for heavy_key in ("initial_t0_mm_c", "initial_pz_MeV_c", "particle_id"):
        if heavy_key in classes_summary:
            classes_summary.pop(heavy_key)

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
        "progress_stats": to_jsonable(progress_stats),
        "diagnostics": to_jsonable({
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
        "screen_summaries": to_jsonable(result.screen_summaries),
        "particle_classes": to_jsonable(classes_summary),
        "saved_figures": saved_figures,
        "saved_screen_json_count": int(saved_screen_json),
        "lost_particles_file": str(lost_path) if lost_path is not None else None,
    }

    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, sort_keys=True)

    progress_path = output_dir / "progress_stats.json"
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(progress_stats), f, indent=2, sort_keys=True)

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
    if saved_figures:
        print(f"Saved {len(saved_figures)} figure files (.png/.eps)")
    if saved_screen_json:
        print(f"Saved {saved_screen_json} per-screen JSON files")
    if lost_path is not None:
        print(f"Saved lost-particle diagnostics: {lost_path.name}")


if __name__ == "__main__":
    main()
