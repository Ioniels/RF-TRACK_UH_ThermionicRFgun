#!/usr/bin/env python3
"""Batch-friendly thermionic TM010 RF-gun run (no Jupyter dependencies)."""

from __future__ import annotations

import argparse
import hashlib
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
# Kept separate from rf_gun.aperture.R_CAV_MM/DEFAULT_DELTA_CATHODE_CHAMFER_MM for the same
# reason as _FINESSE_TIER_NAMES above -- must match by hand.
_R_CAV_MM = 34.0145
_DEFAULT_DELTA_CATHODE_CHAMFER_MM = 0.0
# Kept separate from rf_gun.back_bombardment.DEFAULT_CATHODE_CHAMFER_WIDTH_MM for the same reason
# as _FINESSE_TIER_NAMES above -- must match by hand. (Was previously referenced directly as
# rg.DEFAULT_CATHODE_CHAMFER_WIDTH_MM inside parse_args(), which runs before `import rf_gun as rg`
# in main() -- a NameError on every invocation, caught by actually running the script for the
# first time in a working environment rather than only reading it.)
_DEFAULT_CATHODE_CHAMFER_WIDTH_MM = 0.2
# Kept separate from rf_gun.aperture.DEFAULT_CATHODE_BACKSTOP_THICKNESS_MM for the same reason as
# _FINESSE_TIER_NAMES above -- must match by hand (parse_args() runs before `import rf_gun as rg`).
_DEFAULT_CATHODE_BACKSTOP_THICKNESS_MM = 2.0


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
    parser.add_argument("--r_max_m", type=float, default=_R_CAV_MM * 1e-3)
    # Field-grid resolution: independent of --finesse/--preset (see rf_gun.finesse_presets module
    # docstring) -- these defaults match FIXED_DR_UM/FIXED_DZ_UM there (hand-copied, not imported,
    # for the same parse-args-before-`import rf_gun` reason as _FINESSE_TIER_NAMES above). An
    # explicit --dr_um/--dz_um always wins; apply_finesse_preset_to_args() never overwrites these.
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

    # Dynamic radial aperture R(z): the cavity's real transverse channel (narrow cathode-side
    # chamfer, wide body, narrow exit transition), enforced by RF-Track itself during tracking
    # (see rf_gun.aperture). `--delta_cathode_chamfer_mm` shifts the whole profile relative to the
    # cathode (0 = cathode exactly at the chamfer start; see rf_gun.aperture's module docstring for
    # the sign convention) -- a tunable CLI flag so different cathode insertion depths can be tried.
    parser.add_argument("--delta_cathode_chamfer_mm", type=float, default=None)

    parser.add_argument("--sc_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sc-nx", dest="sc_nx", type=int, default=32)
    parser.add_argument("--sc-ny", dest="sc_ny", type=int, default=32)
    parser.add_argument("--sc-nz", dest="sc_nz", type=int, default=32)
    parser.add_argument("--mirror-charges", dest="mirror_charges", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mirror-z-m", dest="mirror_z_m", type=float, default=0.0)
    parser.add_argument("--mirror-charge-tolerance", dest="mirror_charge_tolerance", type=float, default=None)
    parser.add_argument("--beam_loading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bl_q0", type=float, default=4000.0)
    parser.add_argument("--bl_qext", type=float, default=3500.0)
    parser.add_argument("--bl_p_fwd_w", type=float, default=1.0e6)
    parser.add_argument("--bl_r_over_q_ohm_per_m", type=float, default=1.0)
    parser.add_argument("--bl_ncells", type=int, default=1)
    parser.add_argument("--bl_tinj_mode", choices=["auto_from_emission", "manual"], default="auto_from_emission")
    parser.add_argument("--bl_tinj_manual_mm_c", type=float, default=0.0)

    parser.add_argument("--n_screens", type=int, default=0)
    parser.add_argument("--screens_z", type=float, nargs="*", default=None)
    parser.add_argument("--no-screens", action="store_true", default=False)

    parser.add_argument("--screen_width_mm", type=float, default=None)
    parser.add_argument("--screen_height_mm", type=float, default=None)
    parser.add_argument("--screen_time_window_mm_c", type=float, default=None)
    parser.add_argument("--screen_t0_mode", choices=["unset", "sync_to_first_crossing", "manual"], default="unset")
    parser.add_argument("--screen_t0_manual_mm_c", type=float, default=0.0)
    parser.add_argument("--screen_log", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--r_cathode_mm", type=float, default=2.80 / 2,
        help="Emission radius [mm]. Default 1.40mm is the physical flat-face radius (2.80mm "
             "diameter disk, excluding the 0.2mm 45deg bevel/chamfer -- matches the notebook's "
             "R_CATHODE_MM). The previous default here (3.14/2=1.57mm) was an accidental value "
             "from an earlier version, inconsistent with the notebook and undocumented; pass a "
             "larger value (e.g. 1.6mm, the bevel-inclusive full-disk radius) deliberately, as a "
             "stress test for space-charge/mirror-charge sensitivity, not as a production default "
             "-- see Upgrade_history/run_thermionic_tm010_stress_sc_bevel.slurm for that stress-test "
             "template (not part of the tracked KOA_slurm_scripts/ production suite).",
    )
    parser.add_argument(
        "--cathode_chamfer_width_mm", type=float, default=_DEFAULT_CATHODE_CHAMFER_WIDTH_MM,
        help="Radial width of the cathode's 45deg outer chamfer; impacts there still heat the "
             "cathode (see rg.classify_impact_surface), unlike the holder/wall beyond it.",
    )
    # On by default: wires rf_gun.rftrack_volume.VolumeBuildParams.cathode_backstop_enabled (a thin
    # absorbing element just behind the cathode plane) into the production Volume and runs the
    # validated v2 return-event capture after transport (plain back_bombardment_events.h5); the
    # legacy_ballistic reconstruction moves to back_bombardment_events_legacy_v1.h5 for comparison.
    # Disable only for a deliberate A/B comparison against the pre-backstop behavior.
    parser.add_argument("--cathode_backstop_enabled", action=argparse.BooleanOptionalAction, default=True,
                         help="Cathode backstop element and v2 back-bombardment event capture/HDF5 "
                              "(written to the plain back_bombardment_events.h5; the "
                              "legacy_ballistic reconstruction moves to "
                              "back_bombardment_events_legacy_v1.h5). Requires --save-lost-particles "
                              "(auto-enabled with a printed note if unset). Disable "
                              "(--no-cathode_backstop_enabled) only for a deliberate comparison.")
    parser.add_argument("--cathode_backstop_thickness_mm", type=float,
                         default=_DEFAULT_CATHODE_BACKSTOP_THICKNESS_MM,
                         help="Thickness of the --cathode_backstop_enabled Aperture_1d element "
                              "(rf_gun.aperture.build_cathode_backstop); also used to reconstruct "
                              "the backstop_z_min_m band for backstop/dynamic-aperture loss "
                              "separation in rg.extract_back_bombardment_events.")
    parser.add_argument("--emission_scale", type=float, default=1.0)
    parser.add_argument("--use_const_pz", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pz_init_mevc", type=float, default=4.0e-3)
    parser.add_argument("--ra_um", type=float, default=0.0)
    parser.add_argument("--re_um", type=float, default=0.0)
    parser.add_argument(
        "--emission_law",
        choices=[
            # Current names -- see rf_gun.emission_models module docstring for the convention.
            "RDSchottky", "jensen2014_RDSchottky_MurphyGood_additive",
            "jensen_gtf_2007", "jensen2019_RDSchottky_MurphyGood_transition",
            "murphygood1956_SchottkyNordheim_integral",
            # This session's first-pass (author_papercode_year) names, now superseded.
            "richardson_dushman_schottky", "schottky_murphygood_additive_legacy",
            "jensen_rgtf_2019", "murphy_good_1956_integral",
            # Pre-refactor names, kept accepted here (not just via EMISSION_MODEL_ALIASES) so an
            # old saved shell script or SLURM submission keeps working unchanged.
            "RD_schottky", "rld_schottky_plus_mg", "unified", "rgtf_2019", "murphy_good_direct_reference",
        ],
        default="RDSchottky",
    )
    parser.add_argument("--compare-emission-models", dest="compare_emission_models", type=str, nargs="*", default=None)
    parser.add_argument("--richardson-constant", dest="richardson_constant", type=float, default=None)
    parser.add_argument("--chemical-potential-eV", dest="chemical_potential_eV", type=float, default=None)
    parser.add_argument(
        "--beta-application", dest="beta_application",
        choices=["total_macro_field", "external_field_only"], default="total_macro_field",
    )

    parser.add_argument(
        "--emission-field-iteration", dest="emission_field_iteration",
        action=argparse.BooleanOptionalAction, default=False,
    )
    # The iteration's real sample size is n_x_bins*n_y_bins*n_time_bins (see
    # rf_gun.emission_iteration.EmissionFieldIterationConfig and its module docstring's "fixed
    # source samples" section) -- a formerly-accepted --emission-iteration-particles flag did not
    # control this and was removed rather than left as a misleading no-op (brief Sec. 4.2.7); use
    # the two flags below (and --emission-iteration-nt, further below) to change it.
    parser.add_argument("--emission-iteration-nx", dest="emission_iteration_nx", type=int, default=12,
                         help="Cathode x-grid resolution for the Emission Fields Iteration (and its near-cathode figures).")
    parser.add_argument("--emission-iteration-ny", dest="emission_iteration_ny", type=int, default=12,
                         help="Cathode y-grid resolution for the Emission Fields Iteration (and its near-cathode figures).")
    parser.add_argument("--emission-iteration-nt", dest="emission_iteration_nt", type=int, default=24,
                         help="Emission-time-bin resolution for the Emission Fields Iteration (matches "
                              "EmissionFieldIterationConfig.n_time_bins's own default).")
    parser.add_argument("--emission-iteration-finesse", dest="emission_iteration_finesse", choices=["coarse", "medium"], default="medium")
    parser.add_argument("--emission-iteration-max-iterations", dest="emission_iteration_max_iterations", type=int, default=12)
    parser.add_argument("--emission-iteration-relaxation", dest="emission_iteration_relaxation", type=float, default=0.30)
    parser.add_argument("--emission-iteration-current-tolerance", dest="emission_iteration_current_tolerance", type=float, default=1.0e-2)
    parser.add_argument("--emission-iteration-field-tolerance", dest="emission_iteration_field_tolerance", type=float, default=1.0e-2)
    parser.add_argument("--emission-iteration-charge-tolerance", dest="emission_iteration_charge_tolerance", type=float, default=1.0e-2)
    parser.add_argument("--emission-iteration-z-max-mm", dest="emission_iteration_z_max_mm", type=float, default=2.0)
    parser.add_argument("--emission-field-probe-z-um", dest="emission_field_probe_z_um", type=float, default=None)
    parser.add_argument(
        "--emission-iteration-include-beam-loading", dest="emission_iteration_include_beam_loading",
        action=argparse.BooleanOptionalAction, default=False,
        help="Fold a causal TM010 modal-envelope beam-loading estimate "
             "(E_BL(x,y,t)=-chi(t)*E_RF(x,y,t), rf_gun.beam_loading_envelope) into the Emission "
             "Fields Iteration's own self-consistency loop, using the same Q_L/(R/Q)/Veff "
             "calibration as the real production BeamLoadingSW attach below -- a genuinely "
             "self-consistent SC+mirror+BL case, not a post-hoc estimate. Not yet cross-validated "
             "against RF-Track's own production BeamLoadingSW gradient reduction on a real run "
             "(see UPGRADE_PLAN.md) -- treat E_BL as order-of-magnitude, not precision, "
             "information regardless.",
    )
    parser.add_argument(
        "--emission-iteration-field-probe-method", dest="emission_iteration_field_probe_method",
        choices=["pic_probe", "analytic_point_charge_image"], default="pic_probe",
        help="How the Emission Fields Iteration evaluates space-charge/mirror field from "
             "already-emitted charge at the cathode: 'pic_probe' (default) uses zero-weight "
             "probe particles in RF-Track's own SpaceCharge_PIC_FreeSpace engine at a fixed probe "
             "distance; 'analytic_point_charge_image' uses a closed-form conductor-image kernel "
             "evaluated exactly at the cathode surface (no probe distance, no PIC mesh) -- see "
             "rf_gun.emission_iteration.EmissionFieldIterationConfig.field_probe_method and "
             "UPGRADE_PLAN.md. NEITHER is actually converged for the *peak* near-cathode field "
             "(confirmed: pic_probe's own peak field grows >10x from sc_nx=16 to 64) -- treat "
             "that specific number as order-of-magnitude, not precision, information regardless "
             "of which method is selected.",
    )
    parser.add_argument(
        "--emission-iteration-analytic-softening-scale", dest="emission_iteration_analytic_softening_scale",
        type=float, default=1.0,
        help="Only used by --emission-iteration-field-probe-method analytic_point_charge_image: "
             "multiplier on each source macroparticle's natural softening length (its own "
             "cathode grid cell's disk-equivalent radius) -- see "
             "EmissionFieldIterationConfig.analytic_softening_scale. Scan this the way sc_nx/ny/nz "
             "is scanned for pic_probe to characterize this method's own analogous sensitivity.",
    )
    parser.add_argument(
        "--spatial-emission-sampling", dest="spatial_emission_sampling",
        action=argparse.BooleanOptionalAction, default=False,
        help="Sample the production bunch jointly in (x,y,t) from the RF field alone (guide Sec. "
             "6.4), instead of the default independent radius-uniform/on-axis-F(t) model. "
             "Superseded by --use-converged-iteration-source when both are set and the iteration "
             "converged.",
    )
    parser.add_argument(
        "--use-converged-iteration-source", dest="use_converged_iteration_source",
        action=argparse.BooleanOptionalAction, default=False,
        help="Requires --emission-field-iteration. If the iteration converges, use its final "
             "J(x,y,t) (RF+SC+mirror) directly as the production source (guide Sec. 13.6) instead "
             "of the prescribed on-axis model. Falls back clearly (prints a message, keeps the "
             "prescribed source) if the iteration does not converge.",
    )
    # 1650 K matches the notebook's default and the brief's requested uniform default for the
    # KOA production studies -- previously 1700.0 here, an undocumented mismatch between entry
    # points (see RF_GUN_REPOSITORY_UPGRADE_INSTRUCTIONS.md Sec. 5, item C4 / Sec. 2 finding #5).
    parser.add_argument("--t_cathode_k", type=float, default=1650.0)
    parser.add_argument("--phi_eff_ev", type=float, default=2.1)
    parser.add_argument(
        "--work-function-temperature-model", dest="work_function_temperature_model",
        choices=["constant_phi_eff", "linear_tcwf", "piecewise_surface_evolution"], default=None,
        help="LaB6<100> phi_eff(T) model (manual_references/LaB6_100_work_function_models.md); "
             "overrides --phi_eff_ev with a temperature-resolved value when set.",
    )
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

    parser.add_argument("--timing-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--slow-step-warn-s", type=float, default=20.0)
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-screen-hdf5", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--store-screen-phase-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--store-screen-info", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-stride", type=int, default=1)
    parser.add_argument("--screen-indices", type=int, nargs="*", default=None)
    parser.add_argument("--max-screen-particles", type=int, default=None)
    parser.add_argument("--subsample-screens-random", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-lost-particles", action=argparse.BooleanOptionalAction, default=True)
    # Default True: this is now the canonical B0/Bout output (screen_distributions_hdf5/B0.h5,
    # screen_distributions_hdf5/Bout_*.h5), replacing the old always-on beam_data.npz.
    parser.add_argument("--save-openpmd-beam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-screen-phase-space-batch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-screen-phase-space-json", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--screen-frame-formats", type=str, nargs="+", default=["png"])
    parser.add_argument("--screen-frame-timing-log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-class-phase-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exclude-backward-losses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibrate-bl-r-over-q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--t_max_mm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "quick":
        return
    args.n_particles = 1_000
    args.sc_enabled = True
    args.beam_loading = True
    if args.finesse is None:
        args.finesse = "coarse"


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file's bytes, for compact field-map provenance (Section 6.1: enough
    to detect a changed/wrong raw field-map file without saving the file itself in run metadata)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _requested_screen_count(args: argparse.Namespace) -> int:
    if bool(args.no_screens):
        return 0
    if args.screens_z:
        return int(len(args.screens_z))
    return max(0, int(args.n_screens))


def build_scientific_run_name(args: argparse.Namespace, *, rg) -> str:
    """Date/time first, then the parameters that most determine a run's physics -- matching the
    `outputs/runs/<stamp>_T<T>K_SC<on/off>_BL<on/off>` convention used by the notebook's own
    `SAVE_DATA` run-directory setup, so runs from either entry point sort and group the same way.
    """
    n_particles = int(args.n_particles)
    n_screens = _requested_screen_count(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{stamp}_T{float(args.t_cathode_k):.0f}K_{rg.sc_bl_tag(bool(args.sc_enabled), bool(args.beam_loading))}"
        f"_N{n_particles}_ZSNAPS{n_screens}"
    )


def resolve_output_dir(args: argparse.Namespace, *, rg) -> Path:
    if args.output is not None:
        return Path(args.output)
    return Path("outputs") / "runs" / build_scientific_run_name(args, rg=rg)


def _beam_parameters_summary_from_table(table: list[dict[str, Any]], *, rg) -> tuple[dict[str, Any], dict[str, Any]]:
    """Twiss + emittance summaries from `rg.compute_beam_properties`'s per-screen table.

    `compute_beam_properties` uses `px/pz` (not raw `px`) as the divergence, and forward-filters
    via `%id` lookup against `Bout`'s own reliable absolute z/pz (`rf_gun.particle_tags.
    ParticleTags`) rather than a screen's own `get_info()` (a `Screen`'s `get_info()` returns an
    internal `Bunch6d` object, not `Bunch6dT`, and its z/pz do not reliably carry lab-frame sign).
    It additionally restricts to the aperture-surviving population when tagged -- the same
    population used by every other summary and figure built from this run.
    """

    def _col(key):
        return np.asarray([row.get(key, np.nan) for row in table], dtype=float)

    twiss_summary = {
        "available": True,
        "alpha_x": rg.summarize_array(_col("alpha_x")),
        "beta_x": rg.summarize_array(_col("beta_x")),
        "alpha_y": rg.summarize_array(_col("alpha_y")),
        "beta_y": rg.summarize_array(_col("beta_y")),
        "alpha_t": rg.summarize_array(_col("alpha_t")),
        "beta_t": rg.summarize_array(_col("beta_t")),
    }
    emittance_summary = {
        "available": True,
        "eps_nx": rg.summarize_array(_col("emitt_x_norm")),
        "eps_ny": rg.summarize_array(_col("emitt_y_norm")),
        "eps_nt": rg.summarize_array(_col("emitt_t")),
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


def _evolution_summary_from_screens(screens: list[dict[str, Any]], *, rg) -> dict[str, Any]:
    n_arr = np.asarray([float(rec.get("N", np.nan)) for rec in screens], dtype=float)
    tr_arr = np.asarray([float(rec.get("transmission_from_initial", np.nan)) for rec in screens], dtype=float)
    mpz_arr = np.asarray([float(rec.get("mean_pz_MeV_c", np.nan)) for rec in screens], dtype=float)
    spz_arr = np.asarray([float(rec.get("sigma_pz_MeV_c", np.nan)) for rec in screens], dtype=float)
    return {
        "N": rg.summarize_array(n_arr),
        "transmission_from_initial": rg.summarize_array(tr_arr),
        "mean_pz_MeV_c": rg.summarize_array(mpz_arr),
        "sigma_pz_MeV_c": rg.summarize_array(spz_arr),
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
        if key == "lost":
            # "lost" combines two disjoint sources -- see
            # rf_gun.diagnostics.classify_particle_outcomes's docstring.
            out["classes"][key]["count_aperture"] = int(rec.get("count_aperture", 0) or 0)
            out["classes"][key]["count_unphysical_energy"] = int(rec.get("count_unphysical_energy", 0) or 0)
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
    t0_summary = rg.summarize_array(t0_arr, with_span=True)
    pz0_summary = rg.summarize_array(np.asarray(M0[:, 5], dtype=float) if M0.ndim == 2 and M0.shape[1] > 5 else np.asarray([], dtype=float))

    twiss_summary: dict[str, Any]
    emittance_summary: dict[str, Any]
    if M_snaps:
        beam_properties_table = rg.compute_beam_properties(M_snaps, z_snaps, tags, rg.ME_MEV)
        twiss_summary, emittance_summary = _beam_parameters_summary_from_table(beam_properties_table, rg=rg)
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
        "evolution_summary": _evolution_summary_from_screens(screen_summaries, rg=rg),
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

    if args.richardson_constant is not None or args.chemical_potential_eV is not None:
        raise NotImplementedError(
            "--richardson-constant and --chemical-potential-eV are not yet threaded into the "
            "tracking pipeline's emission law (rf_gun.simulation calls J_rld_schottky/J_unified "
            "directly with their built-in defaults) -- they are only honored by direct "
            "evaluate_emission_model(...) calls, e.g. in the sensitivity/comparison figures. "
            "Wiring a per-run material-parameter override through EmissionParams into the "
            "tracking source is a separate follow-on change; leave both flags unset for now."
        )

    if str(args.beta_application) == "external_field_only":
        raise NotImplementedError(
            "--beta-application external_field_only is not wired into the tracking pipeline yet "
            "(guide Sec. 6.6): beta_enh is currently always applied to the total macroscopic field "
            "(E_RF+E_SC+E_mirror), not just the external RF component. Use "
            "--beta-application total_macro_field (the default) or leave the flag unset."
        )

    import rf_gun as rg
    from rf_gun.finesse_presets import apply_finesse_preset_to_args

    apply_finesse_preset_to_args(args, args.finesse)

    output_dir = resolve_output_dir(args, rg=rg)
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
        veff_from_phase_calibration,
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
    print(f"N_SCREENS: {int(args.n_screens)}")
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
    # One Delaunay triangulation of the source vertices, reused for both Er and Ez (real+imag) --
    # the triangulation/hull test depends only on (pts, R, Z), not on the field values.
    interp_ctx = rg.build_field_interpolation_context(pts, R, Z)
    er_grid = rg.interp_cfield(pts, R, Z, er_vertices, ctx=interp_ctx)
    ez_grid = rg.interp_cfield(pts, R, Z, ez_vertices, ctx=interp_ctx)
    ez0_phasor_axis = rg.find_Ez_axis_phasor_at_z0(ez_grid, z_grid, z0_m=0.0)
    t_interp_elapsed = time.time() - t_interp_start
    t_maps_elapsed = time.time() - t_maps_start
    outside_hull_fraction = interp_ctx.outside_hull_fraction
    repaired_hole_fraction = interp_ctx.repaired_hole_fraction

    print("Field maps generated:")
    print(f"  Phasor mode: {mode}")
    print(f"  Grid size: NR={nr}, NZ={nz} (shape={ez_grid.shape[0]}x{ez_grid.shape[1]})")
    print(f"  Resolution: dr={dr_um:.3f} um, dz={dz_um:.3f} um")
    print(
        f"  Extents: r=[0.000, {float(args.r_max_m) * 1e3:.3f}] mm, "
        f"z=[{z_min * 1e3:.3f}, {z_max * 1e3:.3f}] mm"
    )
    print(
        f"  Interpolation support: outside-native-hull fraction={outside_hull_fraction:.4%} "
        f"(set to zero field, not extrapolated), repaired-interior-hole fraction={repaired_hole_fraction:.4%}"
    )
    print(
        "  Timing: "
        f"phasor={rg.format_duration(t_phasor_elapsed)}, "
        f"interpolation={rg.format_duration(t_interp_elapsed)}, "
        f"total={rg.format_duration(t_maps_elapsed)}"
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

    delta_cathode_chamfer_mm = (
        float(args.delta_cathode_chamfer_mm)
        if args.delta_cathode_chamfer_mm is not None
        else rg.DEFAULT_DELTA_CATHODE_CHAMFER_MM
    )

    phase_scan_n = max(3, int(args.phase_scan_n))
    phase_scan_n_part = max(1, int(args.phase_scan_n_part))
    # A full 0-360deg sweep must not repeat the same physical phase at both endpoints (0 and 360
    # are the same phase); a genuine partial-range scan (e.g. 0 to 90) keeps both requested ends.
    phase_scan_full_circle = (
        np.isclose((float(args.phase_scan_max) - float(args.phase_scan_min)) % 360.0, 0.0)
        and float(args.phase_scan_max) > float(args.phase_scan_min)
    )
    phase_scan_rel = np.linspace(
        float(args.phase_scan_min), float(args.phase_scan_max), phase_scan_n,
        endpoint=not phase_scan_full_circle,
    )
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
        aperture_delta_mm=delta_cathode_chamfer_mm,
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
    phase_cal = rg.run_phase_scan(
        rft,
        er_grid,
        ez_grid,
        vol_params_cal,
        phase_scan_rel,
        phase_deg_transport,
        phase_scan_n_part,
        float(args.pz_init_mevc),
        q_total_C=1e-12,
    )
    t_phase_scan_elapsed = time.time() - t_phase_scan_start
    print(f"Phase scan elapsed: {rg.format_duration(t_phase_scan_elapsed)}")
    print(
        f"Phase scan: {phase_scan_n} coarse points, on-axis cold source, "
        f"{phase_cal.valid_fraction:.1%} finite coverage, "
        f"{len(phase_cal.valid_intervals)} contiguous valid interval(s)"
    )
    if not phase_cal.valid:
        # Priority-A gate (README): an unresolved RF-only calibration invalidates Veff/(R/Q) and
        # every beam-loading-dependent input derived from them. Fail here, before any of that NaN
        # can reach run_config.json, the emission iteration, or BeamLoadingSW -- do not continue
        # with a degraded/NaN calibration at production scale.
        raise RuntimeError(
            "RF-only phase-scan calibration failed: "
            f"{phase_cal.invalid_reason} (finite coverage {phase_cal.valid_fraction:.1%} of "
            f"{phase_cal.phi_rel_deg.size} points). Cannot derive Veff/(R/Q)/BeamLoadingSW inputs "
            "from this scan; check the field map, grid resolution, and phase_scan_dt_mm before "
            "retrying."
        )
    crest_phase_deg = float(phase_cal.crest_phi_abs_deg)
    veff_v = veff_from_phase_calibration(phase_cal, me_MeV=rg.ME_MEV)
    # (R/Q) = Veff^2/(P_del*Q0), with Q0 (unloaded/intrinsic Q) -- NOT q_loaded -- since p_del_w is
    # the wall-dissipated power (see rf_params.r_over_q_per_m's docstring for the derivation from
    # Q0 = omega*U/P_wall and R/Q = V^2/(omega*U)). q_loaded remains the correct Q to pass to
    # BeamLoadingSW itself (below); it is a different quantity from the Q0 used in this
    # calibration step.
    r_over_q_ohm = (veff_v**2) / (p_del_w * float(args.bl_q0))
    # Kept distinct from `bl_r_over_q_ohm_per_m` (the value actually used in transport, below) so
    # the from-scan estimate is still reported correctly even when --no-calibrate-bl-r-over-q
    # overrides it with the fixed CLI/default value.
    bl_r_over_q_ohm_per_m_from_scan = r_over_q_per_m(veff_v, p_del_w, float(args.bl_q0), l_eff_m)
    bl_r_over_q_ohm_per_m = bl_r_over_q_ohm_per_m_from_scan

    if bool(args.beam_loading) and bool(args.calibrate_bl_r_over_q):
        print("Beam-loading R/Q per m updated from phase scan.")
    else:
        bl_r_over_q_ohm_per_m = float(args.bl_r_over_q_ohm_per_m)
        print("Beam-loading R/Q per m kept fixed from CLI/default value.")
    print(f"Phase scan: crest at {crest_phase_deg:.3f} deg")
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
        aperture_delta_mm=delta_cathode_chamfer_mm,
        sc_enabled=bool(args.sc_enabled),
        sc_dt_mm=float(args.sc_dt_mm),
        sc_nx=int(args.sc_nx),
        sc_ny=int(args.sc_ny),
        sc_nz=int(args.sc_nz),
        mirror_charge_enabled=bool(args.mirror_charges),
        mirror_z_m=float(args.mirror_z_m),
        mirror_charge_tolerance=(
            float(args.mirror_charge_tolerance) if args.mirror_charge_tolerance is not None else None
        ),
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
        cathode_backstop_enabled=bool(args.cathode_backstop_enabled),
        cathode_backstop_thickness_mm=float(args.cathode_backstop_thickness_mm),
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
        work_function_temperature_model=args.work_function_temperature_model,
        beta_enh=float(args.beta_f),
        roughness=roughness,
        time_dependent=True,
    )

    if args.no_screens:
        z_snaps = None
    elif args.screens_z:
        z_snaps = [float(z) for z in args.screens_z]
    else:
        n_screens = max(0, int(args.n_screens))
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

    if bool(args.cathode_backstop_enabled) and not bool(args.save_lost_particles):
        print(
            "Note: --cathode_backstop_enabled needs simulation_result.lost_table populated "
            "(RF-Track only fills it when DiagnosticsParams(save_lost_particles=True), see "
            "rf_gun.simulation._extract_lost_particles) to run v2 back-bombardment event capture; "
            "auto-enabling --save-lost-particles for this run."
        )
        args.save_lost_particles = True

    diagnostics = rg.DiagnosticsParams(
        store_screen_phase_space=bool(args.store_screen_phase_space),
        store_screen_info=bool(args.store_screen_info),
        screen_stride=max(1, int(args.screen_stride)),
        screen_indices=args.screen_indices,
        max_screen_particles=args.max_screen_particles,
        subsample_screens_random=bool(args.subsample_screens_random),
        save_lost_particles=bool(args.save_lost_particles),
    )

    emission_iteration_result = None
    if bool(args.emission_field_iteration):
        iter_z_probe_m = (
            float(args.emission_field_probe_z_um) * 1.0e-6
            if args.emission_field_probe_z_um is not None else None
        )
        iteration_config = rg.EmissionFieldIterationConfig(
            enabled=True,
            finesse=str(args.emission_iteration_finesse),
            max_iterations=int(args.emission_iteration_max_iterations),
            n_x_bins=int(args.emission_iteration_nx),
            n_y_bins=int(args.emission_iteration_ny),
            n_time_bins=int(args.emission_iteration_nt),
            relaxation=float(args.emission_iteration_relaxation),
            current_tolerance=float(args.emission_iteration_current_tolerance),
            field_tolerance=float(args.emission_iteration_field_tolerance),
            charge_tolerance=float(args.emission_iteration_charge_tolerance),
            z_probe_m=iter_z_probe_m,
            z_max_m=float(args.emission_iteration_z_max_mm) * 1.0e-3,
            include_space_charge=True,
            include_mirror=bool(args.mirror_charges),
            include_beam_loading=bool(args.emission_iteration_include_beam_loading),
            # Same calibration used for the real production BeamLoadingSW attach (VolumeBuildParams
            # below), so the Emission Fields Iteration's causal envelope and the actual tracked
            # bunch's beam loading use identical Q_L/(R/Q)/Veff -- see
            # EmissionFieldIterationConfig.bl_Q_loaded's docstring.
            bl_Q_loaded=float(q_loaded),
            bl_r_over_q_ohm_per_m=float(bl_r_over_q_ohm_per_m),
            bl_L_eff_m=float(l_eff_m),
            bl_Veff_V=float(veff_v),
            field_probe_method=str(args.emission_iteration_field_probe_method),
            analytic_softening_scale=float(args.emission_iteration_analytic_softening_scale),
            sc_nx=int(args.sc_nx), sc_ny=int(args.sc_ny), sc_nz=int(args.sc_nz),
            mirror_z_m=float(args.mirror_z_m),
            random_seed=int(args.seed),
        )
        print(
            f"Emission Fields Iteration: grid={iteration_config.n_x_bins}x{iteration_config.n_y_bins} "
            f"x {iteration_config.n_time_bins} time bins "
            f"({iteration_config.n_x_bins * iteration_config.n_y_bins * iteration_config.n_time_bins} "
            "total fixed samples), "
            f"max_iter={iteration_config.max_iterations}, omega0={iteration_config.relaxation}",
            flush=True,
        )
        t_iter_start = time.time()
        emission_iteration_result = rg.run_emission_field_iteration(
            rft, er_grid, ez_grid, ez0_phasor_axis, vol_params, emission,
            iteration_config, phi_deg=float(phase_deg_transport),
        )
        print(
            f"Emission Fields Iteration finished in {rg.format_duration(time.time() - t_iter_start)}: "
            f"converged={emission_iteration_result.converged} "
            f"({len(emission_iteration_result.eps_J_history)} iterations)"
        )
        if not emission_iteration_result.converged:
            print(f"  Reason: {emission_iteration_result.failure_reason}")
        try:
            rg.plot_emission_iteration_convergence(emission_iteration_result)
            rg.plot_emission_iteration_waveforms(emission_iteration_result)
            rg.plot_emission_iteration_near_cathode(emission_iteration_result)
        except Exception as exc:
            print(f"  Warning: iteration figures failed: {exc}")

        # Large (x,y,t)-and-iteration-resolved arrays go in a compressed NPZ, not run_results.json
        # (guide Sec. 15.3) -- only a scalar summary is recorded in run_results.json below.
        emission_iteration_npz_path = output_dir / "emission_iteration.npz"
        np.savez_compressed(
            emission_iteration_npz_path,
            schema_version=np.array(3),  # v3 adds E_BL_history_Vpm (causal beam-loading envelope)
            x_grid_m=emission_iteration_result.x_grid_m,
            y_grid_m=emission_iteration_result.y_grid_m,
            t_grid_s=emission_iteration_result.t_grid_s,
            temperature_K=emission_iteration_result.temperature_K,
            J_history_Apm2=np.array(emission_iteration_result.J_history_Apm2),
            E_RF_history_Vpm=np.array(emission_iteration_result.E_RF_history_Vpm),
            E_SC_history_Vpm=np.array(emission_iteration_result.E_SC_history_Vpm),
            E_mirror_history_Vpm=np.array(emission_iteration_result.E_mirror_history_Vpm),
            E_BL_history_Vpm=np.array(emission_iteration_result.E_BL_history_Vpm),
            E_total_history_Vpm=np.array(emission_iteration_result.E_total_history_Vpm),
            Q_history_C=np.array(emission_iteration_result.Q_history_C),
            I_peak_history_A=np.array(emission_iteration_result.I_peak_history_A),
            eps_J_history=np.array(emission_iteration_result.eps_J_history),
            eps_E_history=np.array(emission_iteration_result.eps_E_history),
            eps_Q_history=np.array(emission_iteration_result.eps_Q_history),
            relaxation_history=np.array(emission_iteration_result.relaxation_history),
        )
    else:
        emission_iteration_npz_path = None

    spatial_source = None
    if bool(args.use_converged_iteration_source):
        if emission_iteration_result is None:
            raise ValueError("--use-converged-iteration-source requires --emission-field-iteration")
        if emission_iteration_result.converged:
            spatial_source = rg.spatial_source_from_iteration_result(emission_iteration_result)
            print("Using the converged Emission Fields Iteration source (RF+SC+mirror) for production tracking.")
        else:
            print(
                "--use-converged-iteration-source requested but the iteration did not converge "
                f"({emission_iteration_result.failure_reason}); falling back to the prescribed source."
            )
    elif bool(args.spatial_emission_sampling):
        spatial_source = rg.build_cathode_rf_source(
            rft, er_grid, ez_grid, float(phase_deg_transport), vol_params, emission,
        )
        print("Using RF-only spatially-resolved emission sampling (no SC/mirror feedback) for production tracking.")

    result, progress_stats = rg.run_transport_with_progress(
        rft,
        er_grid,
        ez_grid,
        ez0_phasor_axis,
        vol_params,
        emission,
        tracking,
        diagnostics=diagnostics,
        timing_diagnostics=bool(args.timing_diagnostics),
        slow_step_warn_s=float(args.slow_step_warn_s),
        rng=rng,
        spatial_source=spatial_source,
    )

    phase_fmt = rg.EXTENDED_PHASE_FMT
    m0 = np.array(result.B0.get_phase_space(phase_fmt, "all"), copy=True)
    mf = np.array(result.Bout.get_phase_space(phase_fmt, "all"), copy=True)
    z_snaps_arr = np.asarray(result.z_snaps, dtype=float)

    # `Bout` only ever contains particles the dynamic aperture (rf_gun.aperture) did not remove
    # during tracking -- there is no separate post-hoc radius cut left to apply, so tagging here
    # is just backward-vs-forward (from Bout's reliable absolute z/pz) plus lost (id-based, from
    # RF-Track's own lost-particle table). Computed once and reused by every figure and every
    # JSON summary built from this run.
    #
    # The acceptance scan's trailing-particle removal (`extra_backward_ids`) is folded in here to
    # match the notebook's own tagging exactly (see its "Particle classification" cell) -- this
    # was previously omitted from this script, so its "backward" population (and everything
    # downstream: figures, particle_classes, beam-property tables) classified more loosely than
    # the notebook for the same physics run.
    _backward_ids_strict = rg.backward_ids_from_bout(mf, threshold_backward_mevc=0.0)
    _unphysical_ids = rg.unphysical_ids_from_bout(mf)
    acceptance_scan = rg.scan_acceptance(mf, _backward_ids_strict | _unphysical_ids)
    print(
        f"Acceptance scan: n_forward={acceptance_scan.n_forward} | "
        f"k_core={acceptance_scan.k_core:.2f} (main-beam selection, reference only, NOT applied) | "
        f"k_trailing={acceptance_scan.k_trailing:.2f} (trailing removal, applied) | "
        f"{len(acceptance_scan.trailing_ids)} particle(s) newly tagged backward (trailing)."
    )
    tags = rg.build_particle_tags(mf, result.lost_table, extra_backward_ids=acceptance_scan.trailing_ids)

    # Save B0 (initial launch state) and Bout (final, forward-going, dynamic-aperture-surviving
    # state) as openPMD-beamphysics HDF5, beside the per-screen HDF5 files in
    # screen_distributions_hdf5/ -- the canonical location for every particle-distribution output
    # (Section 6.2: openpmd/ as a separate top-level directory, and the redundant beam_data.npz,
    # are both retired in favor of this one schema/location).
    openpmd_h5_path = None
    b0_h5_path = None
    openpmd_exit_beam_summary = None
    if bool(args.save_openpmd_beam):
        screen_hdf5_dir = output_dir / "screen_distributions_hdf5"

        _which = "good"
        s_out_m = float(z_max)
        _save_source = "Bout (final tracking time, dynamic-aperture survivors)"

        s_out_mm = s_out_m * 1e3
        _run_tag = f"T{float(args.t_cathode_k):.0f}K_{rg.sc_bl_tag(bool(args.sc_enabled), bool(args.beam_loading))}"
        _stem = f"Bout_sout{s_out_mm:.1f}mm_{_run_tag}"
        _meta = {
            "run_name": output_dir.name,
            "s_out_m": s_out_m,
            "save_source": _save_source,
            "delta_cathode_chamfer_mm": float(delta_cathode_chamfer_mm),
            "transport_phase_deg": float(phase_deg_transport),
            "f_hz": float(f_hz),
            "cathode_T_K": float(args.t_cathode_k),
            "work_function_eV": float(args.phi_eff_ev),
            "space_charge": bool(args.sc_enabled),
            "beam_loading": bool(args.beam_loading),
            "Q_total_C": float(result.thermo_info.get("Q_total_C", float("nan"))),
        }
        openpmd_h5_path = rg.save_beam_openpmd(
            screen_hdf5_dir / f"{_stem}.h5",
            result.Bout,
            which=_which,
            forward_only=True,
            aperture_radius_mm=None,
            species="electron",
            extra_attrs=_meta,
        )

        # B0: the as-launched distribution, no forward/aperture filtering (nothing has happened
        # to it yet) -- "which='all'" since RF-Track has not classified any particle as lost yet.
        b0_h5_path = rg.save_beam_openpmd(
            screen_hdf5_dir / f"B0_{_run_tag}.h5",
            result.B0,
            which="all",
            forward_only=False,
            aperture_radius_mm=None,
            species="electron",
            extra_attrs={**_meta, "save_source": "B0 (as-launched distribution)"},
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
        print(f"Saved B0 (as-launched)   : {b0_h5_path.relative_to(output_dir)}")

    # Full B0/Bout phase-space arrays: screen_distributions_hdf5/{B0,Bout_*}.h5 above are the
    # canonical (and, since Section 6.2, only) full-array export -- the previous always-on
    # beam_data.npz duplicated exactly this content in a second, non-openPMD format and is removed.
    # Per-screen full arrays are likewise only in screen_distributions_hdf5/ (--save-screen-hdf5).

    b0_t0 = np.asarray(result.thermo_info.get("initial_t0_mm_c", []), dtype=float)
    t0_summary = rg.summarize_array(b0_t0, with_span=True)
    pz0_summary = rg.summarize_array(np.asarray(m0[:, 5], dtype=float) if m0.ndim == 2 and m0.shape[1] > 5 else np.asarray([], dtype=float))

    # B0/Bout.json were removed as pure duplication of the HDF5 arrays above; at production
    # particle counts (1e5-1e6) each JSON file alone ran 20-30 MB.
    thermo_summary = rg.thermo_info_summary(result.thermo_info)
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
    peak_current_A = float(thermo_summary.get("I_peak_A", float("nan")))
    peak_current_density_A_cm2 = float(thermo_summary.get("J_Apm2", float("nan"))) * 1e-4

    emission_model_comparison = None
    if args.compare_emission_models:
        F_t_hist = np.asarray(result.thermo_info.get("F_t", []), dtype=float)
        F_t_hist = F_t_hist[np.isfinite(F_t_hist) & (F_t_hist > 0.0)]
        if F_t_hist.size == 0:
            print("--compare-emission-models: no populated extraction-field history to compare over; skipping.")
        else:
            # The resolved value actually used by this run (from thermo_info, not the raw
            # --phi_eff_ev CLI value, which --work-function-temperature-model may override) --
            # matches the notebook's own PHI_EFF_EV_USED convention (see its "Emission geometry"/
            # thermionic config cell).
            phi_eff_ev_used = float(result.thermo_info.get("work_function_eV", args.phi_eff_ev))
            models_to_compare = list(args.compare_emission_models)
            F_domain = rg.select_operating_field_domain(F_t_hist)
            F_scan = np.geomspace(max(F_domain[0], 1.0), max(F_domain[2], F_domain[0] + 1.0), 60)
            comparison = rg.compare_emission_models(models_to_compare, F_scan, float(args.t_cathode_k), phi_eff_ev_used)
            print("Emission-model comparison (charge-weighted error vs "
                  f"{comparison['reference_model']}):")
            for m, err in comparison["charge_weighted_error"].items():
                print(f"  {m}: {'n/a (not implemented)' if err is None else f'{err:.4%}'}")
            emission_model_comparison = {
                "models": models_to_compare,
                "reference_model": comparison["reference_model"],
                "charge_weighted_error": comparison["charge_weighted_error"],
                "F_domain_Vpm": [float(x) for x in F_domain],
            }
            try:
                with rg.capture_figures("emission_sensitivity", output_dir / "figures", formats=("png",)):
                    rg.plot_emission_model_sensitivities(
                        F_scan, float(args.t_cathode_k), phi_eff_ev_used,
                        models=models_to_compare, F_populated_range=F_domain,
                    )
            except Exception as exc:
                print(f"  Warning: sensitivity figure failed: {exc}")

    # Back-bombardment: reconstructed for free from Bout's already-tracked state (no extra
    # tracking cost -- see rf_gun.back_bombardment's module docstring), so always computed,
    # independent of --save-figures.
    back_bombardment_data = rg.compute_back_bombardment(
        mf,
        list(result.M_snaps),
        list(result.z_snaps),
        q_total_C=float(result.thermo_info.get("Q_total_C", 0.0)),
        n_macroparticles=int(args.n_particles),
        r_max_mm=float(args.r_max_m) * 1e3,
        cathode_radius_mm=cathode_radius_mm,
        cathode_chamfer_width_mm=float(args.cathode_chamfer_width_mm),
    )
    back_bombardment_summary: Dict[str, Any] = {
        "n_behind_cathode": int(back_bombardment_data.n_behind_cathode),
        "n_valid": int(back_bombardment_data.n_valid),
        "n_never_reached_a_screen": int(np.sum(back_bombardment_data.n_screens_reached == 0)),
        "n_cathode_face": int(back_bombardment_data.n_cathode_face),
        "n_cathode_chamfer": int(back_bombardment_data.n_cathode_chamfer),
        "n_excluded_geometry": int(back_bombardment_data.n_excluded_geometry),
        "total_deposited_energy_J": None,
        "energy_map_file": None,
        "events_file": None,
        "events_v2_file": None,
        "events_v2_n_events": None,
    }
    # Legacy `legacy_ballistic` write moves to an explicitly-labeled filename so it can keep
    # coexisting, for comparison, alongside the new v2 schema below -- see the v2 block's own
    # comment for why the PLAIN `back_bombardment_events.h5` name is reserved for the new schema,
    # not the legacy one (plan Sec. 2.3/4.2: `resolve_back_bombardment_study_input`'s `load_run`
    # mode and `run_back_bombardment_macropulse_study`'s own writer both hardcode that plain name
    # as the v2 file; the legacy writer must not claim it once v2 capture is enabled for this run).
    back_bombardment_events_path = rg.save_back_bombardment_events_hdf5(
        output_dir, back_bombardment_data,
        filename="back_bombardment_events_legacy_v1.h5" if bool(args.cathode_backstop_enabled)
        else "back_bombardment_events.h5",
    )
    if back_bombardment_events_path is not None:
        back_bombardment_summary["events_file"] = str(back_bombardment_events_path)

    # ---- Back-bombardment v2 event capture (opt-in, Work Package 1 completion) -------------
    # BACK_BOMBARDMENT_MACROPULSE_IMPLEMENTATION_PLAN.md Sec. 3.2/4.1/13, addendum Sec. 19.2/19.6.
    # Deliberately additive and gated behind --cathode_backstop_enabled: the legacy
    # `rg.compute_back_bombardment`/`rg.save_back_bombardment_events_hdf5` call above is left
    # functionally untouched (same data, same population) and keeps running by default -- plan
    # Sec. 3.2's `legacy_ballistic` locator "remains available ... for comparison". Only its
    # OUTPUT FILENAME moves aside (to `back_bombardment_events_legacy_v1.h5`, above) when v2
    # capture is enabled, so the PLAIN `back_bombardment_events.h5` name is free for the new
    # `backstop_raycast_v1` events -- matching `rf_gun.back_bombardment_events.
    # resolve_back_bombardment_study_input`'s `load_run` mode and `rf_gun.studies.
    # back_bombardment_macropulse.run_back_bombardment_macropulse_study`'s own writer, which BOTH
    # hardcode the plain name as the v2 schema's canonical path (plan Sec. 2.3: "writes the same
    # object to the run directory as back_bombardment_events.h5"). An earlier version of this
    # script instead wrote v2 events to `back_bombardment_events_v2.h5`, which silently broke
    # `--source-mode load_run` for exactly these runs (the loader looked for the plain name and
    # found only the pre-v2-format legacy file there) -- fixed here by reserving the plain name for
    # v2 and relabeling the legacy output instead, not by teaching the loader a second filename.
    back_bombardment_events_v2_path = None
    if bool(args.cathode_backstop_enabled):
        bb_events_v2 = rg.extract_back_bombardment_events(
            result,
            geometry=rg.CathodeGeometry(),
            capture_config=rg.BackBombardmentCaptureConfig(
                backstop_thickness_mm=float(args.cathode_backstop_thickness_mm),
            ),
            f_hz=f_hz,
            run_id=output_dir.name,
            vol_params=vol_params,
            emission_iteration_enabled=bool(args.emission_field_iteration),
        )
        back_bombardment_events_v2_path = rg.write_back_bombardment_events_h5(
            output_dir / "back_bombardment_events.h5", bb_events_v2,
        )
        print(
            f"Back-bombardment v2 (backstop_raycast_v1): {bb_events_v2.n_events} qualified "
            f"events -> {back_bombardment_events_v2_path.name} (legacy_ballistic output moved to "
            f"back_bombardment_events_legacy_v1.h5)"
        )
        rg.display_back_bombardment_event_schema(bb_events_v2, h5_path=back_bombardment_events_v2_path)
        back_bombardment_summary["events_v2_file"] = str(back_bombardment_events_v2_path)
        back_bombardment_summary["events_v2_n_events"] = int(bb_events_v2.n_events)

    saved_figures: List[str] = []
    back_bombardment_energy_map = None
    if bool(args.save_figures):
        figures_result = rg.save_run_figures(
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
            n_macroparticles=int(args.n_particles),
            lost_table=result.lost_table,
            back_bombardment_data=back_bombardment_data,
            back_bombardment_cathode_radius_mm=cathode_radius_mm,
        )
        saved_figures = figures_result["saved_figures"]
        back_bombardment_energy_map = figures_result["back_bombardment_energy_map"]

    # Independent of figures/: the 2D map is data (xedges/yedges/density_J_per_mm2/total_J), not a
    # rendering of it -- see rf_gun.save_back_bombardment_energy_map. Only produced when
    # --save-figures also ran (that's currently the only code path that bins the map).
    back_bombardment_map_path = rg.save_back_bombardment_energy_map(output_dir, back_bombardment_energy_map)
    if back_bombardment_map_path is not None:
        back_bombardment_summary["total_deposited_energy_J"] = float(back_bombardment_energy_map["total_J"])
        back_bombardment_summary["energy_map_file"] = str(back_bombardment_map_path)

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
            n_macroparticles=int(args.n_particles),
            style=plot_style,
            show_colorbar=False,
            save_json=bool(args.save_screen_phase_space_json),
            figure_formats=tuple(str(fmt).strip().lower() for fmt in args.screen_frame_formats if str(fmt).strip()),
            timing_log=bool(args.screen_frame_timing_log),
            thermo_info=dict(result.thermo_info),
        )

    # Per-screen JSON files (screen_distributions_json/) were removed: they duplicated the same
    # per-screen summaries already recorded in run_results.json's screen curve arrays (Section 6.2).
    saved_screen_hdf5_paths: List[Path] = []
    if bool(args.save_screen_hdf5):
        saved_screen_hdf5_paths = rg.save_screen_distributions_hdf5(
            output_dir=output_dir / "screen_distributions_hdf5",
            z_snaps=list(result.z_snaps),
            M_snaps=list(result.M_snaps),
            I_snaps=list(result.I_snaps),
            n_initial=int(m0.shape[0]) if m0.ndim == 2 else 0,
            q_total_C=float(result.thermo_info.get("Q_total_C", 0.0)),
            filename_stem=f"screen_T{float(args.t_cathode_k):.0f}K_{rg.sc_bl_tag(bool(args.sc_enabled), bool(args.beam_loading))}",
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

    # One consolidated beam-properties-vs-z summary (particle counts, per-screen curves,
    # evolution, Twiss/emittance, classification, consistency checks), written only inside
    # run_results.json (below) since every one of these is a small per-screen/per-class quantity,
    # never a per-particle array.
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
    for w in beam_summary["consistency_warnings"]:
        print(f"WARNING: {w}")

    if bool(args.save_class_phase_space):
        if mf.ndim == 2 and mf.shape[0] > 0 and mf.shape[1] >= 6:
            is_backward, _is_lost = rg.tag_mask(mf, tags)
            rg.save_beam_phase_space_json(output_dir / "B_transmitted.json", mf[~is_backward], label="B_transmitted")
            rg.save_beam_phase_space_json(output_dir / "B_backward_returned.json", mf[is_backward], label="B_backward_returned")

    ref_note = "RF-Track may switch to centroid reference if first particle is lost; robust summaries are computed from explicit phase-space arrays."
    ref_warn = bool(result.thermo_info.get("reference_particle_reordered", False))

    # `run_config.json`: everything this run was set up to do -- every input parameter (cavity/
    # field-map, solver/finesse, cathode/emission, beam-loading, aperture, deflection, screen/
    # particle-count settings) plus the handful of values derived from them before tracking even
    # starts (grid sizes, phase-scan crest/Veff/R-over-Q, effective length). No per-particle or
    # per-time-sample data anywhere in it. Same shape as the notebook's `SAVE_DATA` run (both call
    # `rg.save_run_config`) so a run started either way is read the same way later.
    run_config_path = rg.save_run_config(
        output_dir,
        run_name=output_dir.name,
        source="script:run_thermionic_tm010.py",
        hardcoded_parameters={
            "run_identity": {
                "run_family": str(args.run_family),
                "scan_tags": [str(x) for x in (args.scan_tags or [])],
                "seed": args.seed,
                "finesse_tier": str(args.finesse) if args.finesse is not None else None,
                "args": rg.to_json_safe(vars(args)),
            },
            "runtime_environment": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "rftrack": {
                    "max_number_of_threads": rg.to_json_safe(getattr(rft, "max_number_of_threads", None)),
                    "number_of_threads": rg.to_json_safe(getattr(rft.cvar, "number_of_threads", None)),
                    "thread_policy": (
                        f"forced ({int(effective_threads)})"
                        if threads_requested_explicit
                        else f"auto-resolved ({int(effective_threads)})"
                    ),
                },
                "thread_env": dict(inherited_thread_env),
            },
            "plotting_defaults": {
                "exclude_backward_losses": bool(args.exclude_backward_losses),
            },
            "diagnostics_config": {
                "store_screen_phase_space": diagnostics.store_screen_phase_space,
                "store_screen_info": diagnostics.store_screen_info,
                "screen_stride": diagnostics.screen_stride,
                "screen_indices": diagnostics.screen_indices,
                "max_screen_particles": diagnostics.max_screen_particles,
                "subsample_screens_random": diagnostics.subsample_screens_random,
            },
            "field_provenance": {
                "source_maps": {
                    tag: {
                        "path": str(path),
                        "sha256": _sha256_file(path),
                        "size_bytes": int(Path(path).stat().st_size),
                        "n_vertices": int(data["vertices"].shape[0]),
                        "n_facets": int(data["facets"].shape[0]) if data["facets"] is not None else None,
                        "edge_length_stats_mm": (
                            rg.mesh_edge_length_stats(data["vertices"], data["facets"])
                            if data["facets"] is not None else None
                        ),
                        "raw_keys": list(data["keys"]),
                    }
                    for tag, path, data in (("xy", args.xy_fieldmap, xy), ("yz", args.yz_fieldmap, yz))
                },
                "phasor": {"mode": str(args.phasor_mode), "frequency_hz": f_hz},
                "interpolation": {
                    "method": (
                        "linear on the source Delaunay triangulation; zero field outside the "
                        "native convex hull (not extrapolated); KD-tree nearest-neighbor repair "
                        "only for isolated interior holes"
                    ),
                    "requested_dr_um": dr_um,
                    "requested_dz_um": dz_um,
                    "realized_hr_m": hr,
                    "realized_hz_m": hz,
                    "nr": int(nr),
                    "nz": int(nz),
                    "outside_hull_fraction": outside_hull_fraction,
                    "repaired_hole_fraction": repaired_hole_fraction,
                },
                "magnetic_field_source": {
                    "status": "none",
                    "reason": (
                        "raw field-map .mat files expose only TotalField_E_X/Y/Z (confirmed by "
                        "direct inspection of every top-level .mat key) -- no B/H component is "
                        "available to measure, so Bt=Bz=0.0 in RF_FieldMap_2d reflects the source "
                        "data, not an assumption. Results are E-only; treat any transverse-force "
                        "or emission-field claim that would depend on the missing RF Bphi as "
                        "unverified against a measured/Maxwell-consistent magnetic map."
                    ),
                    "enabled": False,
                },
            },
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
                "cathode_chamfer_width_mm": float(args.cathode_chamfer_width_mm),
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
                "work_function_temperature_model": args.work_function_temperature_model,
                "spatial_emission_sampling": bool(args.spatial_emission_sampling),
                "use_converged_iteration_source": bool(args.use_converged_iteration_source),
                "compare_emission_models": list(args.compare_emission_models) if args.compare_emission_models else None,
                "richardson_constant_Apm2K2": float(args.richardson_constant) if args.richardson_constant is not None else None,
                "chemical_potential_eV": float(args.chemical_potential_eV) if args.chemical_potential_eV is not None else None,
                "beta_application": str(args.beta_application),
            },
            "emission_field_iteration": {
                "enabled": bool(args.emission_field_iteration),
                "n_x_bins": int(args.emission_iteration_nx),
                "n_y_bins": int(args.emission_iteration_ny),
                "n_time_bins": int(args.emission_iteration_nt),
                "n_fixed_samples": int(args.emission_iteration_nx) * int(args.emission_iteration_ny) * int(args.emission_iteration_nt),
                "finesse": str(args.emission_iteration_finesse),
                "max_iterations": int(args.emission_iteration_max_iterations),
                "relaxation": float(args.emission_iteration_relaxation),
                "current_tolerance": float(args.emission_iteration_current_tolerance),
                "field_tolerance": float(args.emission_iteration_field_tolerance),
                "charge_tolerance": float(args.emission_iteration_charge_tolerance),
                "z_max_mm": float(args.emission_iteration_z_max_mm),
                "probe_z_um": float(args.emission_field_probe_z_um) if args.emission_field_probe_z_um is not None else None,
                "include_beam_loading": bool(args.emission_iteration_include_beam_loading),
                "bl_Q_loaded": float(q_loaded) if bool(args.emission_iteration_include_beam_loading) else None,
                "bl_r_over_q_ohm_per_m": float(bl_r_over_q_ohm_per_m) if bool(args.emission_iteration_include_beam_loading) else None,
                "bl_L_eff_m": float(l_eff_m) if bool(args.emission_iteration_include_beam_loading) else None,
                "bl_Veff_V": float(veff_v) if bool(args.emission_iteration_include_beam_loading) else None,
                "field_probe_method": str(args.emission_iteration_field_probe_method),
                "analytic_softening_scale": float(args.emission_iteration_analytic_softening_scale),
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
                "sc_nx": int(args.sc_nx),
                "sc_ny": int(args.sc_ny),
                "sc_nz": int(args.sc_nz),
                "mirror_charge_enabled": bool(args.mirror_charges),
                "mirror_z_m": float(args.mirror_z_m),
                "mirror_charge_tolerance": (
                    float(args.mirror_charge_tolerance) if args.mirror_charge_tolerance is not None else None
                ),
                "emission_nsteps": int(args.emission_nsteps),
                "emission_range": float(args.emission_range),
                "rftrack_capabilities": rg.inspect_rftrack_capabilities(rft),
            },
            "beam_loading": {
                "enabled": bool(args.beam_loading),
                "Q0": float(args.bl_q0),
                "Qext": float(args.bl_qext),
                "Q_loaded": float(q_loaded),
                "P_fwd_W": float(args.bl_p_fwd_w),
                "P_del_W": float(p_del_w),
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
                "n_screens": int(z_snaps_arr.size),
                "screen_width_mm": args.screen_width_mm,
                "screen_height_mm": args.screen_height_mm,
                "screen_time_window_mm_c": args.screen_time_window_mm_c,
                "screen_t0_mode": str(args.screen_t0_mode),
                "screen_t0_manual_mm_c": float(args.screen_t0_manual_mm_c),
            },
            "aperture": {
                "delta_cathode_chamfer_mm": float(delta_cathode_chamfer_mm),
                "r1_mm": rg.R1_MM,
                "r2_mm": rg.R2_MM,
                "R_cav_mm": rg.R_CAV_MM,
                "chamfer_len_mm": rg.CHAMFER_LEN_MM,
                "chamfer_angle_deg": rg.CHAMFER_ANGLE_DEG,
                "rho_mm": rg.RHO_MM,
                "L_mm": rg.L_MM,
                "note": (
                    "Dynamic radial aperture R(z), enforced by RF-Track's own Aperture_1d element "
                    "during tracking (see rf_gun.aperture)."
                ),
            },
            "cathode_backstop": {
                "enabled": bool(args.cathode_backstop_enabled),
                "thickness_mm": float(args.cathode_backstop_thickness_mm),
                "events_v2_file": (
                    str(back_bombardment_events_v2_path.name)
                    if back_bombardment_events_v2_path is not None else None
                ),
                "note": (
                    "Opt-in Aperture_1d backstop just behind the cathode plane (rf_gun.aperture."
                    "build_cathode_backstop), enabling backstop_raycast_v1 v2 event capture "
                    "(BACK_BOMBARDMENT_MACROPULSE_IMPLEMENTATION_PLAN.md Sec. 3.2/4.1). "
                    "'events_v2_file' non-null means the v2 schema was written to the PLAIN "
                    "back_bombardment_events.h5 in this run directory (the canonical load_run "
                    "path -- rf_gun.resolve_back_bombardment_study_input hardcodes this name), "
                    "with the legacy_ballistic reconstruction moved aside to "
                    "back_bombardment_events_legacy_v1.h5 for comparison -- a load_run consumer "
                    "can check this field without opening any HDF5 file."
                ),
            },
            "deflection_magnet": {
                "enabled": bool(args.deflection_enabled),
                # "current_A" kept for backward compatibility with existing readers -- always
                # equals configured_current_A, i.e. the requested value, not what tracking saw.
                "current_A": float(args.deflection_current_A),
                "configured_current_A": float(args.deflection_current_A),
                "applied_current_A": float(args.deflection_current_A) if bool(args.deflection_enabled) else 0.0,
                "B_pk_per_A_T": deflection_B_pk_per_A_T,
                "z_p_mm": deflection_z_p_mm,
                "w_mm": deflection_w_mm,
            },
        },
        derived_parameters={
            "lambda_m": lambda_m,
            "grid": {"nr": int(nr), "nz": int(nz), "dr_um": float(dr_um), "dz_um": float(dz_um)},
            "z_min_m": z_min,
            "z_max_m": z_max,
            "l_eff_m": float(l_eff_m),
            "phi_zero_deg": float(phi_zero_deg),
            "phi_crest_deg": float(phi_crest_deg),
            "transport_phase_deg": float(phase_deg_transport),
            "t_max_mm": float(t_max_mm),
            "phase_scan": {
                "n_coarse_points": int(phase_scan_rel.size),
                "calibration_source": "on_axis_cold",
                "valid_fraction": float(phase_cal.valid_fraction),
                "n_valid_intervals": int(len(phase_cal.valid_intervals)),
                "crest_bracketed": bool(phase_cal.crest_bracketed),
                "crest_phase_deg": crest_phase_deg,
                "veff_V": float(veff_v),
                "r_over_q_ohm": float(r_over_q_ohm),
                "r_over_q_ohm_per_m_from_scan": float(bl_r_over_q_ohm_per_m_from_scan),
                "r_over_q_ohm_per_m_used": float(bl_r_over_q_ohm_per_m),
            },
        },
    )

    # `run_results.json`: everything this run *found* -- R/Q and Veff actually used (in
    # run_config.json's derived_parameters.phase_scan, calibrated before tracking), peak current/
    # current density, the beam-property curves vs z (transmission, Twiss, beam size -- one row
    # per screen, in beam_summary below), particle classification, aperture/back-bombardment
    # summaries, the openPMD exit-beam summary, and every other output file this run wrote.
    run_results_path = rg.save_run_results(
        output_dir,
        run_name=output_dir.name,
        source="script:run_thermionic_tm010.py",
        results={
            "thermo_info": thermo_summary,
            "peak_current_A": peak_current_A,
            "peak_current_density_A_cm2": peak_current_density_A_cm2,
            "tracking_timing": {
                "total_simulation_s": float(time.time() - t_sim_start),
                "field_map_total_s": float(t_maps_elapsed),
                "field_map_phasor_s": float(t_phasor_elapsed),
                "field_map_interpolation_s": float(t_interp_elapsed),
                "phase_scan_s": float(t_phase_scan_elapsed),
                "tracking_call": rg.to_json_safe(progress_stats),
            },
            "beam_summary": rg.to_json_safe(beam_summary),
            "back_bombardment": back_bombardment_summary,
            "openpmd_exit_beam": openpmd_exit_beam_summary,
            "reference_particle_warning": bool(ref_warn),
            "reference_particle_note": ref_note,
            "screen_phase_space_batch": rg.to_json_safe(screen_phase_space_batch),
            "saved_screen_hdf5_files": [str(p) for p in saved_screen_hdf5_paths],
            "emission_model_comparison": emission_model_comparison,
            "emission_iteration": (
                {
                    "converged": bool(emission_iteration_result.converged),
                    "n_iterations": len(emission_iteration_result.eps_J_history),
                    "failure_reason": emission_iteration_result.failure_reason,
                    "final_eps_J": emission_iteration_result.eps_J_history[-1] if emission_iteration_result.eps_J_history else None,
                    "final_eps_E": emission_iteration_result.eps_E_history[-1] if emission_iteration_result.eps_E_history else None,
                    "final_eps_Q": emission_iteration_result.eps_Q_history[-1] if emission_iteration_result.eps_Q_history else None,
                    "initial_charge_C": emission_iteration_result.Q_history_C[0] if emission_iteration_result.Q_history_C else None,
                    "final_charge_C": emission_iteration_result.Q_history_C[-1] if emission_iteration_result.Q_history_C else None,
                    "initial_peak_current_A": emission_iteration_result.I_peak_history_A[0] if emission_iteration_result.I_peak_history_A else None,
                    "final_peak_current_A": emission_iteration_result.I_peak_history_A[-1] if emission_iteration_result.I_peak_history_A else None,
                    "capability_report": emission_iteration_result.capability_report,
                }
                if emission_iteration_result is not None else None
            ),
        },
        output_files={
            "emission_iteration_npz": str(emission_iteration_npz_path.resolve()) if emission_iteration_npz_path is not None else None,
            "figures_dir": str((output_dir / "figures").resolve()) if saved_figures else None,
            "bout_h5": str(openpmd_h5_path) if openpmd_h5_path is not None else None,
            "b0_h5": str(b0_h5_path) if b0_h5_path is not None else None,
            "screens_dir": (
                str(output_dir / "screen_distributions_hdf5")
                if (saved_screen_hdf5_paths or openpmd_h5_path is not None) else None
            ),
            "back_bombardment_energy_map": str(back_bombardment_map_path) if back_bombardment_map_path is not None else None,
            "beam_properties_csv": None,
            "lost_particles_json": str(lost_path) if lost_path is not None else None,
        },
    )

    _bout_ids = mf[:, 6] if mf.ndim == 2 and mf.shape[1] > 6 else np.asarray([])
    _finite_ids = _bout_ids[np.isfinite(_bout_ids)]
    _n_duplicate_ids = int(_finite_ids.size - np.unique(_finite_ids).size) if _finite_ids.size else 0

    _derived_rf_values = {
        "veff_V": veff_v, "r_over_q_ohm": r_over_q_ohm, "l_eff_m": float(l_eff_m),
        "r_over_q_ohm_per_m_used": bl_r_over_q_ohm_per_m,
    }
    _nonfinite_rf = [k for k, v in _derived_rf_values.items() if not np.isfinite(v)]

    # Provisional threshold: no pilot convergence study has yet set a defensible production value
    # (Section 9 acknowledges this explicitly) -- 5% is a conservative placeholder that flags a
    # clearly mismatched grid/domain without over-triggering on normal edge effects.
    _outside_hull_threshold = 0.05
    validation_checks: Dict[str, Any] = {
        "phase_calibration": {
            "passed": bool(phase_cal.valid),
            "valid_fraction": float(phase_cal.valid_fraction),
            "crest_bracketed": bool(phase_cal.crest_bracketed),
            "reason": phase_cal.invalid_reason,
        },
        "particle_ids_unique": {
            "passed": _n_duplicate_ids == 0,
            "n_duplicate_ids": _n_duplicate_ids,
            "n_ids_checked": int(_finite_ids.size),
        },
        "derived_rf_parameters_finite": {
            "passed": len(_nonfinite_rf) == 0,
            "nonfinite_fields": _nonfinite_rf,
        },
        "field_interpolation_support": {
            "passed": bool(outside_hull_fraction <= _outside_hull_threshold),
            "outside_hull_fraction": float(outside_hull_fraction),
            "threshold": _outside_hull_threshold,
            "threshold_status": "provisional -- not yet set from a pilot convergence study",
        },
    }
    if emission_iteration_result is not None:
        validation_checks["emission_field_iteration_converged"] = {
            "passed": bool(emission_iteration_result.converged),
            "failure_reason": emission_iteration_result.failure_reason,
        }
    validation_report = rg.build_validation_report(validation_checks)
    validation_path = rg.save_validation_report(output_dir, validation_checks)
    if validation_report["status"] == "ok":
        (output_dir / ".run_complete").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    else:
        print(f"VALIDATION FAILED: {validation_report['failed_checks']} -- see {validation_path.name}")

    t_sim_elapsed = time.time() - t_sim_start
    print(f"\nRun complete, simulation time: {rg.format_duration(t_sim_elapsed)}")
    print(f"Validation: {validation_report['status']} ({validation_path.name})")
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
    print(f"Saved: {run_config_path.name}, {run_results_path.name}")
    if openpmd_h5_path is not None:
        print(f"Saved openPMD exit beam: {openpmd_h5_path.relative_to(output_dir)}")
    if saved_figures:
        print(f"Saved {len(saved_figures)} figure files (.png/.eps) to figures/")
    if back_bombardment_map_path is not None:
        print(f"Saved back-bombardment energy map: {back_bombardment_map_path.name}")
    if screen_phase_space_batch is not None:
        print(f"Saved cinematic phase-space frames: {int(screen_phase_space_batch.get('frame_count', 0))}")
    if saved_screen_hdf5_paths:
        print(f"Saved {len(saved_screen_hdf5_paths)} per-screen HDF5 files")
    if lost_path is not None:
        print(f"Saved lost-particle diagnostics: {lost_path.name}")


if __name__ == "__main__":
    main()
