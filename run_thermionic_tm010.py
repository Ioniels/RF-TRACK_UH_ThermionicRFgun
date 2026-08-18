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

# Kept separate from rf_gun.finesse_presets.FINESSE_TIERS so `--help` doesn't have to import
# rf_gun (and RF_Track with it) just to parse arguments; must match it by hand.
_FINESSE_TIER_NAMES = ("extra_fine", "fine", "medium", "coarse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thermionic TM010 transport with RF-Track.")

    parser.add_argument("--preset", choices=["none", "quick"], default="none")
    # Solver/meshing finesse tier -- see rf_gun/finesse_presets.py. Applied after --preset, so it
    # always wins over --preset quick's own values.
    parser.add_argument("--finesse", choices=list(_FINESSE_TIER_NAMES), default=None)
    parser.add_argument("--output", type=Path, default=None)

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

    # Physical exit-aperture channel (post-hoc geometric cut + explicit entrance/exit screens) --
    # a separate concept from --aperture_m (RF-Track's own whole-Volume aperture bound above).
    # Matches the notebook's APERTURE_ENABLED/APERTURE_START_M/APERTURE_END_M/APERTURE_DIAMETER_MM;
    # opt-in (default disabled) so existing scripts/SLURM jobs that don't pass these flags are
    # unaffected. See `--save-openpmd-beam` for the aperture-aware exit-beam export this enables.
    parser.add_argument("--aperture_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aperture_start_m", type=float, default=0.028854)
    parser.add_argument("--aperture_end_m", type=float, default=0.040589)
    parser.add_argument("--aperture_diameter_mm", type=float, default=5.0546)

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

    parser.add_argument("--deflection_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deflection_current_A", type=float, default=0.0)
    parser.add_argument("--deflection_B_pk_per_A_T", type=float, default=None)
    parser.add_argument("--deflection_z_p_mm", type=float, default=None)
    parser.add_argument("--deflection_w_mm", type=float, default=None)

    parser.add_argument("--poll_interval_s", type=float, default=0.5)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    # "spawn" prints from a separate OS process, useful for an unattended CLI/SLURM run writing
    # to a log file -- see rf_gun/simulation.py::run_transport_with_progress.
    parser.add_argument("--progress-backend", choices=["thread", "spawn"], default="thread")
    # Accepted for compatibility with existing SLURM scripts; this script never renders a
    # notebook progress widget, so the value has no effect beyond being echoed in run_metadata.
    parser.add_argument("--progress-notebook-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--timing-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--slow-step-warn-s", type=float, default=20.0)
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-screen-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-json-mode", choices=["summary", "full"], default="summary")
    parser.add_argument("--save-screen-hdf5", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--store-screen-phase-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--store-screen-info", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-stride", type=int, default=1)
    parser.add_argument("--screen-indices", type=int, nargs="*", default=None)
    parser.add_argument("--max-screen-particles", type=int, default=None)
    parser.add_argument("--subsample-screens-random", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-lost-particles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-beam-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-beam-summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-openpmd-beam", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-screen-phase-space-batch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-screen-phase-space-json", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--screen-frame-formats", type=str, nargs="+", default=["png"])
    parser.add_argument("--screen-frame-timing-log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-class-phase-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exclude-backward-losses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-aperture-losses", action=argparse.BooleanOptionalAction, default=True)
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


def _requested_screen_count(args: argparse.Namespace) -> int:
    if bool(args.no_screens):
        return 0
    if args.screens_z:
        return int(len(args.screens_z))
    n_snap = int(args.n_z_snap) if args.n_z_snap is not None else int(args.n_screens)
    return max(0, int(n_snap))


def build_scientific_run_name(args: argparse.Namespace) -> str:
    """Date/time first, then the parameters that most determine a run's physics -- matching the
    `outputs/runs/<stamp>_T<T>K_SC<on/off>_BL<on/off>` convention used by the notebook's own
    `SAVE_DATA` run-directory setup, so runs from either entry point sort and group the same way.
    """
    sc_tag = "on" if bool(args.sc_enabled) else "off"
    bl_tag = "on" if bool(args.beam_loading) else "off"
    n_particles = int(args.n_particles)
    n_screens = _requested_screen_count(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{stamp}_T{float(args.t_cathode_k):.0f}K_SC{sc_tag}_BL{bl_tag}"
        f"_N{n_particles}_ZSNAPS{n_screens}"
    )


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    return Path("outputs") / "runs" / build_scientific_run_name(args)


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


def _beam_parameters_summary_from_table(table: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Twiss + emittance summaries from `rg.compute_beam_properties`'s per-screen table.

    Replaces two previous, independent implementations: a hand-rolled Twiss computation that
    passed raw `px` (not `px/pz`) as the divergence, and an emittance summary read from RF-Track's
    per-screen `get_info()` (`info_snaps`) -- found unreliable since a `Screen`'s `get_info()`
    returns an internal `Bunch6d` object, not `Bunch6dT` (confirmed against RF-Track 2.5.4 and
    2.6.3 -- see `UPGRADE_PLAN_notebook_and_architecture.md`). `compute_beam_properties` forward-
    filters via `%id` lookup against `Bout`'s own reliable absolute z/pz
    (`rf_gun.particle_tags.ParticleTags`) instead of a raw z/pz sign check on the screen's own
    columns, and additionally restricts to the aperture-surviving population when tagged -- the
    same population used by every other summary and figure built from this run.
    """

    def _col(key):
        return np.asarray([row.get(key, np.nan) for row in table], dtype=float)

    twiss_summary = {
        "available": True,
        "alpha_x": summarize_array(_col("alpha_x")),
        "beta_x": summarize_array(_col("beta_x")),
        "alpha_y": summarize_array(_col("alpha_y")),
        "beta_y": summarize_array(_col("beta_y")),
        "alpha_t": summarize_array(_col("alpha_t")),
        "beta_t": summarize_array(_col("beta_t")),
    }
    emittance_summary = {
        "available": True,
        "eps_nx": summarize_array(_col("emitt_x_norm")),
        "eps_ny": summarize_array(_col("emitt_y_norm")),
        "eps_nt": summarize_array(_col("emitt_t")),
        "note": "normalized emittance only (mm*mrad for x/y; this project's (ToF, pz/mean(pz)) convention for the longitudinal plane, using %t rather than %Z since a screen's own %Z is not a lab-frame position) -- no separate geometric emittance is computed",
    }
    return twiss_summary, emittance_summary


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
    tags,
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
    emittance_summary: dict[str, Any]
    if M_snaps:
        beam_properties_table = rg.compute_beam_properties(M_snaps, z_snaps, tags, rg.ME_MEV)
        twiss_summary, emittance_summary = _beam_parameters_summary_from_table(beam_properties_table)
    else:
        twiss_summary = {"available": False, "reason": "Twiss data not returned in current run"}
        emittance_summary = {"available": False, "reason": "Twiss data not returned in current run"}

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

    args = parse_args()
    threads_requested_explicit = args.threads is not None
    apply_preset(args)

    import rf_gun as rg
    from rf_gun.finesse_presets import apply_finesse_preset_to_args

    apply_finesse_preset_to_args(args, args.finesse)

    output_dir = resolve_output_dir(args)
    args.output = output_dir
    rng = np.random.default_rng(int(args.seed) if args.seed is not None else None)

    inherited_thread_env = {
        "RF_TRACK_NUMBER_OF_THREADS": os.environ.get("RF_TRACK_NUMBER_OF_THREADS", "unset"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "unset"),
    }

    effective_threads = rg.resolve_threads(requested=args.threads, default=1)
    rg.set_thread_environment(effective_threads, pin_blas_threads=True)
    args.threads = int(effective_threads)

    import RF_Track as rft
    from rf_gun.rf_params import (
        delivered_power_on_resonance,
        effective_length_from_abs_ez,
        r_over_q_per_m,
        veff_from_phase_scan_pz,
    )

    try:
        rft.cvar.number_of_threads = int(effective_threads)
    except Exception:
        pass

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "unset")
    rftrack_max_threads = getattr(rft, "max_number_of_threads", "n/a")
    rftrack_chosen_threads = getattr(rft.cvar, "number_of_threads", "n/a")

    print("---- Pre-configuration environment snapshot ----")
    print(f"Inherited RF_TRACK_NUMBER_OF_THREADS={inherited_thread_env['RF_TRACK_NUMBER_OF_THREADS']}")
    print(f"Inherited OMP_NUM_THREADS={inherited_thread_env['OMP_NUM_THREADS']}")
    print(f"Inherited OPENBLAS_NUM_THREADS={inherited_thread_env['OPENBLAS_NUM_THREADS']}")
    print(f"Inherited MKL_NUM_THREADS={inherited_thread_env['MKL_NUM_THREADS']}")
    print(f"Inherited NUMEXPR_NUM_THREADS={inherited_thread_env['NUMEXPR_NUM_THREADS']}")

    print("---- Applied thread configuration ----")
    print(f"SLURM_CPUS_PER_TASK: {slurm_cpus}")
    print(f"Requested threads: {int(args.threads) if threads_requested_explicit else 'auto'}")
    print(f"Resolved threads: {int(effective_threads)}")
    print(f"Applied RF_TRACK_NUMBER_OF_THREADS={os.environ.get('RF_TRACK_NUMBER_OF_THREADS', 'unset')}")
    print(f"Applied OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"Applied OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset')}")
    print(f"Applied MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'unset')}")
    print(f"Applied NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS', 'unset')}")
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
    if threads_requested_explicit:
        print(f"Thread policy: forced --threads={int(effective_threads)}")
    else:
        print(f"Thread policy: auto-resolved from scheduler/default -> {int(effective_threads)}")
    if bool(args.timing_diagnostics):
        print(f"Timing diagnostics: ON (slow-step threshold={float(args.slow_step_warn_s):.2f} s)")
    else:
        print("Timing diagnostics: OFF")

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
    _, _, phi_abs_scan, pz_mean_scan = rg.run_phase_scan(
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
    i_peak_scan = int(np.argmax(pz_mean_scan))
    crest_phase_deg = float(np.asarray(phi_abs_scan, dtype=float)[i_peak_scan])
    veff_v = veff_from_phase_scan_pz(np.asarray(pz_mean_scan, dtype=float), float(args.pz_init_mevc), me_MeV=rg.ME_MEV)
    r_over_q_ohm = (veff_v**2) / (p_del_w * q_loaded)
    # Kept distinct from `bl_r_over_q_ohm_per_m` (the value actually used in transport, below) so
    # the from-scan estimate is still reported correctly even when --no-calibrate-bl-r-over-q
    # overrides it with the fixed CLI/default value -- previously both prints read the same
    # (possibly-overridden) variable, so "from scan" silently showed the fixed value instead.
    bl_r_over_q_ohm_per_m_from_scan = r_over_q_per_m(veff_v, p_del_w, q_loaded, l_eff_m)
    bl_r_over_q_ohm_per_m = bl_r_over_q_ohm_per_m_from_scan

    if bool(args.beam_loading) and bool(args.calibrate_bl_r_over_q):
        print("Beam-loading R/Q per m updated from phase scan.")
    else:
        bl_r_over_q_ohm_per_m = float(args.bl_r_over_q_ohm_per_m)
        print("Beam-loading R/Q per m kept fixed from CLI/default value.")
    print(f"Phase scan elapsed: {format_duration(t_phase_scan_elapsed)}")
    print(f"Phase scan: {phase_scan_n} coarse points -> crest at {crest_phase_deg:.3f} deg")
    print(f"Veff = {veff_v/1e6:.6f} MV")
    print(f"(R/Q) from scan = {r_over_q_ohm:.3e} Ω")
    print(f"(R/Q)/m from scan = {bl_r_over_q_ohm_per_m_from_scan:.3e} Ω/m")
    print(f"(R/Q)/m used in transport = {bl_r_over_q_ohm_per_m:.3e} Ω/m")

    # NOTE: estimate_default_tmax_mm() assumes light-speed transit (cavity_length / c), which
    # badly underestimates the time a thermionic gun needs: particles start near rest and
    # accelerate gradually, so light-speed transit time is not a valid proxy for tracking
    # duration here. Confirmed empirically: at the ~120-200 mm/c this produces for this gun's
    # parameters, Volume.get_bunch_at_screens() returns an empty list (not screens with zero
    # rows) because no particle reaches even the first screen in time -- and Bout itself would
    # reflect a run cut off before slower/backward-turning particles finish evolving. The
    # notebook never calls this estimator at all and instead relies on VolumeBuildParams's own
    # t_max_mm default (2000.0 mm/c); match that here unless the user overrides explicitly.
    t_max_mm = float(args.t_max_mm) if args.t_max_mm is not None else 2000.0

    deflection_B_pk_per_A_T = (
        float(args.deflection_B_pk_per_A_T) if args.deflection_B_pk_per_A_T is not None else rg.DEFAULT_B_PK_PER_A_T
    )
    deflection_z_p_mm = float(args.deflection_z_p_mm) if args.deflection_z_p_mm is not None else rg.DEFAULT_Z_P_MM
    deflection_w_mm = float(args.deflection_w_mm) if args.deflection_w_mm is not None else rg.DEFAULT_W_MM

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
        deflection_enabled=bool(args.deflection_enabled),
        deflection_current_A=float(args.deflection_current_A),
        deflection_B_pk_per_A_T=deflection_B_pk_per_A_T,
        deflection_z_p_mm=deflection_z_p_mm,
        deflection_w_mm=deflection_w_mm,
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

    # Insert explicit aperture entrance/exit screens ("before"/"after" views), matching the
    # notebook's APERTURE_ENABLED handling -- needed so the openPMD exit-beam export below (see
    # --save-openpmd-beam) has an exact screen at the aperture-exit z to select from.
    if bool(args.aperture_enabled):
        z_snaps = sorted(set(z_snaps or []) | {float(args.aperture_start_m), float(args.aperture_end_m)})

    tracking = rg.TrackingParams(
        phi_deg=float(phase_deg_transport),
        n_particles=int(args.n_particles),
        z_screens_m=z_snaps,
        phase_fmt=rg.EXTENDED_PHASE_FMT,
        screen_width_mm=args.screen_width_mm,
        screen_height_mm=args.screen_height_mm,
        screen_time_window_mm_c=args.screen_time_window_mm_c,
        screen_t0_mode=str(args.screen_t0_mode),
        screen_t0_manual_mm_c=float(args.screen_t0_manual_mm_c),
        screen_log=bool(args.screen_log),
    )

    if (bool(args.save_screen_hdf5) or bool(args.save_openpmd_beam)) and not bool(args.store_screen_phase_space):
        print(
            "Note: --save-screen-hdf5/--save-openpmd-beam need each screen's raw phase-space "
            "array; auto-enabling --store-screen-phase-space."
        )
        args.store_screen_phase_space = True

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
        progress_backend=str(args.progress_backend),
    )

    phase_fmt = rg.EXTENDED_PHASE_FMT
    m0 = np.array(result.B0.get_phase_space(phase_fmt, "all"), copy=True)
    mf = np.array(result.Bout.get_phase_space(phase_fmt, "all"), copy=True)
    z_snaps_arr = np.asarray(result.z_snaps, dtype=float)

    # Backward-tagging only: this pipeline has no post-hoc aperture entrance/exit screens (the
    # notebook's aperture cell is the only place that pairing exists -- see rf_gun/aperture.py's
    # module docstring), so aperture-loss tagging is empty here and `--exclude-aperture-losses` is
    # a no-op until/unless this script grows an equivalent post-hoc aperture. Computed once and
    # reused by every figure and every JSON summary built from this run.
    #
    # A particle whose radius exceeds the field map's r_max_m extent can pick up an unphysical,
    # runaway Pz, yet still reads as forward at Bout, corrupting Twiss/emittance for any screen it
    # reaches. The notebook catches this for free via its aperture radius cut; this script has no
    # equivalent geometric screen. Not filtered here, since a hidden energy/radius cut would risk
    # silently dropping legitimate particles too -- pair a tight --aperture_m with r_max_m, or add
    # a real geometric screen, instead of a numeric cutoff on the tagging itself.
    tags = rg.build_particle_tags(mf, None, None, 0.0, False)

    # Save the final (forward-going, aperture-clipped when --aperture_enabled) 6D beam as
    # openPMD-beamphysics HDF5 -- mirrors the notebook's "Export the exit beam" cell exactly
    # (same source-selection logic, same `Bout_sout<mm>mm_T<K>K_SC<on/off>_BL<on/off>.h5` naming).
    openpmd_h5_path = None
    openpmd_exit_beam_summary = None
    if bool(args.save_openpmd_beam):
        openpmd_dir = output_dir / "openpmd"
        _sc_tag = "on" if bool(args.sc_enabled) else "off"
        _bl_tag = "on" if bool(args.beam_loading) else "off"
        _skip_export = False

        if bool(args.aperture_enabled):
            aperture_radius_mm = float(args.aperture_diameter_mm) / 2.0
            _i_ext_save = int(np.argmin(np.abs(z_snaps_arr - float(args.aperture_end_m))))
            _M_ap_exit_raw = np.asarray(result.M_snaps[_i_ext_save], dtype=float)
            _is_bw_save, _ = rg.tag_mask(_M_ap_exit_raw, tags, screen_z_m=float(z_snaps_arr[_i_ext_save]))
            _M_ap_exit_fwd = _M_ap_exit_raw[~_is_bw_save] if _M_ap_exit_raw.shape[0] else _M_ap_exit_raw
            _r_mask_save = rg.aperture_survival_mask(_M_ap_exit_fwd, aperture_radius_mm)
            _M_to_save = _M_ap_exit_fwd[_r_mask_save] if _M_ap_exit_fwd.shape[0] else _M_ap_exit_fwd

            print(
                f"Aperture-exit screen (z={z_snaps_arr[_i_ext_save]*1e3:.3f} mm): "
                f"{_M_ap_exit_raw.shape[0]} particles reached this plane, "
                f"{_M_ap_exit_fwd.shape[0]} forward-going, "
                f"{_M_to_save.shape[0]} within aperture radius ({aperture_radius_mm:.4f} mm)."
            )

            if _M_to_save.shape[0] == 0:
                print("WARNING: no particles within the aperture radius -- skipping openPMD exit-beam export.")
                _skip_export = True
            else:
                # Uniform per-macroparticle weighting (matches build_bunch_thermionic's own N
                # column, split evenly across all requested macroparticles).
                _n_real_per_macro = (abs(float(result.thermo_info.get("Q_total_C", 0.0))) / rg.q_e) / float(args.n_particles)
                _N_real_saved = _n_real_per_macro * _M_to_save.shape[0]
                _bunch_to_save = rft.Bunch6dT(rg.ME_MEV, float(_N_real_saved), -1.0, _M_to_save[:, :6])
                _which = "all"
                _forward_only_save = False
                _aperture_radius_for_save = None  # already filtered above
                s_out_m = float(z_snaps_arr[_i_ext_save])
                _save_source = "aperture-exit screen (exact z of aperture crossing)"
        else:
            _bunch_to_save = result.Bout
            _which = "good"
            _forward_only_save = True
            _aperture_radius_for_save = None
            s_out_m = float(z_max)
            _save_source = "Bout (final tracking time, no aperture)"

        if not _skip_export:
            s_out_mm = s_out_m * 1e3
            _stem = f"Bout_sout{s_out_mm:.1f}mm_T{float(args.t_cathode_k):.0f}K_SC{_sc_tag}_BL{_bl_tag}"
            _meta = {
                "run_name": output_dir.name,
                "s_out_m": s_out_m,
                "save_source": _save_source,
                "aperture_radius_mm": (float(args.aperture_diameter_mm) / 2.0) if bool(args.aperture_enabled) else None,
                "transport_phase_deg": float(phase_deg_transport),
                "f_hz": float(f_hz),
                "cathode_T_K": float(args.t_cathode_k),
                "work_function_eV": float(args.phi_eff_ev),
                "space_charge": bool(args.sc_enabled),
                "beam_loading": bool(args.beam_loading),
                "Q_total_C": float(result.thermo_info.get("Q_total_C", float("nan"))),
            }
            openpmd_h5_path = rg.save_beam_openpmd(
                openpmd_dir / f"{_stem}.h5",
                _bunch_to_save,
                which=_which,
                forward_only=_forward_only_save,
                aperture_radius_mm=_aperture_radius_for_save,
                species="electron",
                extra_attrs=_meta,
            )

            from pmd_beamphysics import ParticleGroup

            _pg = ParticleGroup(h5=str(openpmd_h5_path))
            openpmd_exit_beam_summary = {
                "file": str(openpmd_h5_path.resolve()),
                "source": _save_source,
                "s_out_m": s_out_m,
                "n_saved": int(_pg.n_particle),
                "total_charge_C": float(_pg.charge),
                "mean_energy_eV": float(_pg["mean_energy"]),
                "norm_emit_x_m": float(_pg["norm_emit_x"]),
                "norm_emit_y_m": float(_pg["norm_emit_y"]),
            }
            print(f"Saved exit beam to: {openpmd_h5_path.resolve()}")
            print(f"Source                   : {_save_source}")
            print(f"z of saved distribution  : s_out = {s_out_mm:.3f} mm from cathode (z=0)")
            print(f"Saved                    : {_pg.n_particle}")

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
        saved_figures = rg.save_run_figures(
            output_dir=output_dir / "figures",
            B0=result.B0,
            Bout=result.Bout,
            transport_phase_deg=float(phase_deg_transport),
            thermo_info=dict(result.thermo_info),
            M_snaps=list(result.M_snaps),
            z_snaps=list(result.z_snaps),
            tags=tags,
            phase_fmt=phase_fmt,
            exclude_backward_losses=bool(args.exclude_backward_losses),
            exclude_aperture_losses=bool(args.exclude_aperture_losses),
            n_macroparticles=int(args.n_particles),
            lost_table=result.lost_table,
        )

    screen_phase_space_batch = None
    if bool(args.save_screen_phase_space_batch):
        plot_style = rg.PlotStyleConfig(dezoom_frac=0.05)
        screen_phase_space_batch = rg.save_screen_phase_space_batch(
            output_dir=output_dir,
            M_snaps=list(result.M_snaps),
            z_snaps=list(result.z_snaps),
            B0=result.B0,
            tags=tags,
            phase_fmt=phase_fmt,
            exclude_backward_losses=bool(args.exclude_backward_losses),
            exclude_aperture_losses=bool(args.exclude_aperture_losses),
            n_macroparticles=int(args.n_particles),
            style=plot_style,
            show_colorbar=False,
            save_json=bool(args.save_screen_phase_space_json),
            figure_formats=tuple(str(fmt).strip().lower() for fmt in args.screen_frame_formats if str(fmt).strip()),
            timing_log=bool(args.screen_frame_timing_log),
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

    saved_screen_hdf5_paths: List[Path] = []
    if bool(args.save_screen_hdf5):
        _sc_tag = "on" if bool(args.sc_enabled) else "off"
        _bl_tag = "on" if bool(args.beam_loading) else "off"
        saved_screen_hdf5_paths = rg.save_screen_distributions_hdf5(
            output_dir=output_dir / "screen_distributions_hdf5",
            z_snaps=list(result.z_snaps),
            M_snaps=list(result.M_snaps),
            I_snaps=list(result.I_snaps),
            n_initial=int(m0.shape[0]) if m0.ndim == 2 else 0,
            q_total_C=float(result.thermo_info.get("Q_total_C", 0.0)),
            filename_stem=f"screen_T{float(args.t_cathode_k):.0f}K_SC{_sc_tag}_BL{_bl_tag}",
            extra_attrs={
                "run_name": output_dir.name,
                "cathode_T_K": float(args.t_cathode_k),
                "space_charge": bool(args.sc_enabled),
                "beam_loading": bool(args.beam_loading),
                "transport_phase_deg": float(phase_deg_transport),
            },
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
            tags=tags,
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
            "exclude_backward_losses": bool(args.exclude_backward_losses),
            "exclude_aperture_losses": bool(args.exclude_aperture_losses),
        },
        "rftrack": {
            "max_number_of_threads": sanitize_for_json(getattr(rft, "max_number_of_threads", None)),
            "number_of_threads": sanitize_for_json(getattr(rft.cvar, "number_of_threads", None)),
            "thread_policy": (
                f"forced ({int(effective_threads)})"
                if threads_requested_explicit
                else f"auto-resolved ({int(effective_threads)})"
            ),
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
        "saved_screen_hdf5_files": [str(p) for p in saved_screen_hdf5_paths],
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

    # Same `run_summary.json` shape as the notebook's `SAVE_DATA` run -- both call
    # `rg.save_run_summary` so a run started either way is read the same way later.
    # `openpmd_exit_beam` is populated when `--save-openpmd-beam` was passed (see above, right
    # after `tags`); `acceptance_scan`/`aperture_summary`/`back_bombardment`/`beam_properties_csv`
    # stay None -- this script doesn't run those notebook-only diagnostics (acceptance scan,
    # back-bombardment, beam-properties-vs-z).
    run_summary_path = rg.save_run_summary(
        output_dir,
        run_name=output_dir.name,
        source="script:run_thermionic_tm010.py",
        hardcoded_parameters={
            "cavity": {
                "f_hz": f_hz,
                "y_cathode_mm": float(args.y_cathode_mm),
                "r_max_m": float(args.r_max_m),
                "dr_um": float(args.dr_um),
                "dz_um": float(args.dz_um),
                "ext_zmax_m": float(args.ext_zmax),
                "ext_zmin_m": z_min,
                "xy_fieldmap": str(args.xy_fieldmap),
                "yz_fieldmap": str(args.yz_fieldmap),
                "phasor_mode": str(args.phasor_mode),
            },
            "cathode_emission": {
                "r_cathode_mm": float(args.r_cathode_mm),
                "emission_scale": float(args.emission_scale),
                "use_const_pz": bool(args.use_const_pz),
                "pz_init_mevc": float(args.pz_init_mevc),
                "ra_um": float(args.ra_um),
                "re_um": float(args.re_um),
                "emission_law": str(args.emission_law),
                "t_cathode_k": float(args.t_cathode_k),
                "phi_eff_ev": float(args.phi_eff_ev),
                "beta_f": float(args.beta_f),
                "emission_phase_range_deg": float(args.emission_phase_range),
                "emission_phase_start_deg": float(args.emission_phase_start),
            },
            "integration": {
                "dt_mm": float(args.dt_mm),
                "ode_algorithm": str(args.ode_algorithm),
                "ode_epsabs": float(args.ode_epsabs),
                "fm_nsteps": int(args.fm_nsteps),
                "fm_tt_nsteps": int(args.fm_tt_nsteps),
            },
            "space_charge": {
                "enabled": bool(args.sc_enabled),
                "sc_dt_mm": float(args.sc_dt_mm),
                "emission_nsteps": int(args.emission_nsteps),
                "emission_range": float(args.emission_range),
            },
            "beam_loading": {
                "enabled": bool(args.beam_loading),
                "Q0": float(args.bl_q0),
                "Qext": float(args.bl_qext),
                "Q_loaded": float(q_loaded),
                "P_fwd_W": float(args.bl_p_fwd_w),
                "P_del_W": float(p_del_w),
                "r_over_q_ohm_per_m_used": float(bl_r_over_q_ohm_per_m),
                "calibrate_from_scan": bool(args.calibrate_bl_r_over_q),
                "n_cells": int(args.bl_ncells),
                "tinj_mode": str(args.bl_tinj_mode),
                "tinj_manual_mm_c": float(args.bl_tinj_manual_mm_c),
                "cfx_dt_mm": float(args.cfx_dt_mm),
            },
            "phase_scan_settings": {
                "min_deg": float(args.phase_scan_min),
                "max_deg": float(args.phase_scan_max),
                "n_points": int(phase_scan_n),
                "n_particles": int(phase_scan_n_part),
            },
            "transport": {
                "n_particles": int(args.n_particles),
                "n_z_snap": int(z_snaps_arr.size),
                "screen_width_mm": args.screen_width_mm,
                "screen_height_mm": args.screen_height_mm,
                "screen_time_window_mm_c": args.screen_time_window_mm_c,
                "screen_t0_mode": str(args.screen_t0_mode),
                "screen_t0_manual_mm_c": float(args.screen_t0_manual_mm_c),
            },
            "aperture": {
                "volume_aperture_m": args.aperture_m,
                "physical_exit_aperture_enabled": bool(args.aperture_enabled),
                "physical_exit_aperture_start_m": float(args.aperture_start_m) if bool(args.aperture_enabled) else None,
                "physical_exit_aperture_end_m": float(args.aperture_end_m) if bool(args.aperture_enabled) else None,
                "physical_exit_aperture_diameter_mm": float(args.aperture_diameter_mm) if bool(args.aperture_enabled) else None,
                "note": (
                    "'volume_aperture_m' is RF-Track's own whole-Volume aperture bound; the "
                    "'physical_exit_aperture_*' fields are the separate, opt-in (--aperture_enabled) "
                    "post-hoc geometric cut matching the notebook's APERTURE_START_M/END_M/"
                    "APERTURE_DIAMETER_MM, used only for the --save-openpmd-beam export."
                ),
            },
            "deflection_magnet": {
                "enabled": bool(args.deflection_enabled),
                "current_A": float(args.deflection_current_A),
                "B_pk_per_A_T": deflection_B_pk_per_A_T,
                "z_p_mm": deflection_z_p_mm,
                "w_mm": deflection_w_mm,
            },
        },
        derived_parameters={
            "lambda_m": lambda_m,
            "grid": {"nr": int(nr), "nz": int(nz)},
            "z_min_m": z_min,
            "z_max_m": z_max,
            "l_eff_m": float(l_eff_m),
            "phi_zero_deg": float(phi_zero_deg),
            "phi_crest_deg": float(phi_crest_deg),
            "transport_phase_deg": float(phase_deg_transport),
            "phase_scan": {
                "n_coarse_points": int(phase_scan_rel.size),
                "crest_phase_deg": crest_phase_deg,
                "veff_V": float(veff_v),
                "r_over_q_ohm": float(r_over_q_ohm),
                "r_over_q_ohm_per_m_from_scan": float(bl_r_over_q_ohm_per_m_from_scan),
                "r_over_q_ohm_per_m_used": float(bl_r_over_q_ohm_per_m),
            },
        },
        results={
            "thermo_info": thermo_summary,
            "progress_stats": dict(progress_stats),
            "particle_classes_summary": classes_summary,
            "acceptance_scan": None,
            "aperture_summary": None,
            "back_bombardment": None,
            "openpmd_exit_beam": openpmd_exit_beam_summary,
            "screen_summaries": robust_screen_summaries,
        },
        output_files={
            "figures_dir": str((output_dir / "figures").resolve()) if saved_figures else None,
            "openpmd_dir": str(openpmd_h5_path.parent.resolve()) if openpmd_h5_path is not None else None,
            "screens_dir": str((output_dir / "screen_distributions_hdf5").resolve()) if saved_screen_hdf5_paths else None,
            "beam_properties_csv": None,
            "lost_particles_json": str(lost_path) if lost_path is not None else None,
        },
    )

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
    if openpmd_h5_path is not None:
        print(f"Saved openPMD exit beam: {openpmd_h5_path.relative_to(output_dir)}")
    if saved_figures:
        print(f"Saved {len(saved_figures)} figure files (.png/.eps) to figures/")
    if screen_phase_space_batch is not None:
        print(f"Saved cinematic phase-space frames: {int(screen_phase_space_batch.get('frame_count', 0))}")
    if saved_screen_json:
        print(f"Saved {saved_screen_json} per-screen JSON files")
    if saved_screen_hdf5_paths:
        print(f"Saved {len(saved_screen_hdf5_paths)} per-screen HDF5 files")
    if lost_path is not None:
        print(f"Saved lost-particle diagnostics: {lost_path.name}")
    print(f"Saved run summary: {run_summary_path.name}")


if __name__ == "__main__":
    main()
