#!/usr/bin/env python3
"""Thin CLI wrapper around the back-bombardment/configurable-macropulse study (implementation plan
Sec. 10.1: "Thin argparse wrapper around the shared study function"; Sec. 13's example commands).

This script contains NO physics logic of its own. Every step delegates to already-implemented
`rf_gun` library functions -- `rf_gun.resolve_back_bombardment_study_input`,
`rf_gun.default_uh_back_bombardment_study_config`, `rf_gun.run_back_bombardment_macropulse_study`,
`rf_gun.validate_back_bombardment_study`, `rf_gun.load_comsol_thermal_result`,
`rf_gun.compare_python_comsol_thermal`, `rf_gun.plot_back_bombardment_source_qualification`, and
`rf_gun.plot_back_bombardment_macropulse` -- exactly the same functions the notebook's back-
bombardment cells call (plan Sec. 1: "The notebook and batch/SLURM paths call the same library
functions and construct the same dataclasses.").

Two stages (`--stage`):

  * `run` (default): resolve a study input, run the full L2_one_way study, validate it, print a
    summary, and write `study_config.json`/`study_results.json` (plan Sec. 4.2) plus, optionally,
    the two study figures.
  * `compare`: import a completed COMSOL thermal result and compare it against an already-
    completed study's *scalar* thermal histories (`T_center`/`T_max`/`T_area_average` vs macro
    time). This stage is intentionally partial -- see `_run_stage_compare`'s docstring below for
    exactly what it can and cannot do, and why.

`import rf_gun as rg` (which transitively does `import RF_Track`) is deferred to inside `main()`,
matching `run_thermionic_tm010.py`'s own convention, so `--help` stays fast and does not require a
working RF-Track installation just to print usage.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

_THERMAL_BACKENDS = ("python_xy_layered", "python_xy_sheet", "lumped_energy_check", "uh_legacy_1d")
_SOURCE_MODES = ("current_notebook", "load_run")
_STAGES = ("run", "compare")


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser. Kept separate from `parse_args()` so tests (and any future
    caller) can inspect the parser -- e.g. its option strings -- without also running the
    post-parse cross-field validation in `parse_args()`."""
    parser = argparse.ArgumentParser(
        description=(
            "Run (or compare) the back-bombardment/configurable-macropulse heating study from a "
            "completed RF-Track run's qualified event capture. This is a thin wrapper around "
            "rf_gun.studies.back_bombardment_macropulse.run_back_bombardment_macropulse_study -- "
            "it implements no physics of its own."
        )
    )

    parser.add_argument(
        "--source-mode",
        choices=_SOURCE_MODES,
        default="load_run",
        help=(
            "Study input mode (rf_gun.resolve_back_bombardment_study_input). 'load_run' reads a "
            "completed run directory's back_bombardment_events.h5 (v2 schema) and is the only mode "
            "this CLI can actually use. 'current_notebook' is accepted here only for API symmetry "
            "with resolve_back_bombardment_study_input's two modes -- it requires an in-memory "
            "BackBombardmentEvents object produced by a live Python/notebook session, which cannot "
            "exist in a fresh CLI process, so choosing it here always fails immediately with a "
            "clear error rather than silently doing something else. Default: load_run."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Completed run directory containing back_bombardment_events.h5 (v2 schema). Required "
            "for --source-mode load_run."
        ),
    )
    parser.add_argument(
        "--thermal-backend",
        choices=_THERMAL_BACKENDS,
        default="python_xy_layered",
        help="rf_gun.ThermalConfig.backend. Default: python_xy_layered.",
    )
    parser.add_argument(
        "--material-set",
        dest="material_set",
        default="LaB6_UH_recommended_v1",
        help="Named LaB6 property set (rf_gun.CathodeMaterialSelection.property_set). Default: LaB6_UH_recommended_v1.",
    )
    parser.add_argument(
        "--deposition-model",
        dest="deposition_model",
        default="BB0_TIO",
        choices=["BB0_TIO"],
        help=(
            "Named energy-deposition model (rf_gun.DepositionConfig.model). Only 'BB0_TIO' "
            "(the TIO/CSDA baseline) is implemented; 'BB1_uncertainty'/'BB2_response_library' "
            "are not yet implemented and are not exposed as CLI choices until they are (they "
            "still exist as documented, NotImplementedError-raising placeholders for anyone "
            "calling rf_gun.DepositionConfig directly as a library). Default: BB0_TIO."
        ),
    )
    parser.add_argument(
        "--macropulse-duration-us",
        dest="macropulse_duration_us",
        type=float,
        default=8.0,
        help=(
            "RF macropulse duration in microseconds, converted once at parsing to the SI "
            "duration_s stored in rf_gun.MacropulseConfig (plan Sec. 10.3). Default: 8.0."
        ),
    )
    parser.add_argument(
        "--thermal-bin-ns",
        dest="thermal_bin_ns",
        type=float,
        default=50.0,
        help=(
            "Macro-time thermal bin width in nanoseconds, forwarded to "
            "rf_gun.run_back_bombardment_macropulse_study's thermal_bin_s (plan Sec. 8.1: "
            "'initial thermal bins of roughly 20-100ns are reasonable'). Previously only "
            "settable by calling the library directly; the CLI silently used the library's own "
            "50ns default. Default: 50.0 (matches that prior default; Study IV requires a "
            "20/10/5ns pilot convergence comparison before adopting a production value)."
        ),
    )
    parser.add_argument(
        "--thermal-dt-ns",
        dest="thermal_dt_ns",
        type=float,
        default=None,
        help=(
            "Sub-step interval in nanoseconds within each thermal bin's implicit solve "
            "(rf_gun.ThermalConfig.dt_s) -- decouples time refinement from --thermal-bin-ns "
            "(plan Sec. 6.2). Default: None, meaning step exactly at each thermal bin's own edge "
            "(no sub-stepping) -- the ThermalConfig.dt_s default."
        ),
    )

    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument(
        "--initial-temperature-uniform-k",
        dest="initial_temperature_uniform_k",
        type=float,
        default=None,
        help="Build a rf_gun.ConstantTemperatureMap(T0_K) initial condition. Mutually exclusive with --initial-temperature-map.",
    )
    init_group.add_argument(
        "--initial-temperature-map",
        dest="initial_temperature_map",
        type=Path,
        default=None,
        help=(
            "Path to an initial_temperature_xy.h5 asymmetric T(x,y) map (plan Sec. 6.1), to be "
            "loaded as a rf_gun.TemperatureMap2D. NOT YET IMPLEMENTED: no loader for this file "
            "exists anywhere in rf_gun yet, so passing this flag raises a clear NotImplementedError "
            "at run time rather than silently falling back to a uniform map. Use "
            "--initial-temperature-uniform-k instead until that loader is added."
        ),
    )

    parser.add_argument(
        "--comsol-results",
        dest="comsol_results",
        type=Path,
        default=None,
        help=(
            "Optional COMSOL thermal-result file (comsol_thermal_result_v1 HDF5), loaded via "
            "rf_gun.load_comsol_thermal_result and fed into the study's (interface-only) COMSOL "
            "comparison. Only used with --stage run; for comparing COMSOL results against an "
            "already-completed study, use --stage compare instead."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Study output directory (the study's own output_dir). Required for --stage run -- "
            "study_config.json/study_results.json, back_bombardment_events.h5, "
            "back_bombardment_heat_source.h5, back_bombardment_macropulse.h5, and (with "
            "--save-figures) figures/ are all written here. For --stage compare, defaults to "
            "--study-h5's own parent directory if omitted."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=_STAGES,
        default="run",
        help=(
            "'run' (default): run the full study from a resolved study input. 'compare': import a "
            "COMSOL result and compare it against an already-completed study's HDF5 output -- see "
            "--study-h5. Default: run."
        ),
    )
    parser.add_argument(
        "--study-h5",
        dest="study_h5",
        type=Path,
        default=None,
        help=(
            "Completed back_bombardment_macropulse.h5 to compare against (required for --stage "
            "compare). Only the scalar /thermal time histories in this file are read -- see "
            "--stage compare's --help note printed at run time for exactly what this does and does "
            "not do."
        ),
    )
    parser.add_argument(
        "--save-figures",
        dest="save_figures",
        action="store_true",
        default=False,
        help=(
            "Generate both study figures (rf_gun.plot_back_bombardment_source_qualification, "
            "rf_gun.plot_back_bombardment_macropulse) and save them under <output>/figures/. Off "
            "by default, matching run_thermionic_tm010.py's convention that figure generation is "
            "opt-in. Only used with --stage run."
        ),
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.stage == "run":
        if args.source_mode == "current_notebook":
            parser.error(
                "--source-mode current_notebook cannot be used from this CLI: it requires an "
                "in-memory BackBombardmentEvents object produced by a live Python/notebook "
                "session (rf_gun.resolve_back_bombardment_study_input's current_events "
                "argument), which does not exist in a fresh CLI process. It is accepted as a "
                "--source-mode choice only for API symmetry with "
                "resolve_back_bombardment_study_input's two modes. Use --source-mode load_run "
                "(the default) instead."
            )
        if args.source_mode == "load_run" and args.run_dir is None:
            parser.error("--run-dir is required for --source-mode load_run.")
        if args.output is None:
            parser.error("--output is required for --stage run.")
        n_init = sum(
            v is not None
            for v in (args.initial_temperature_uniform_k, args.initial_temperature_map)
        )
        if n_init != 1:
            parser.error(
                "exactly one of --initial-temperature-uniform-k / --initial-temperature-map is "
                "required for --stage run."
            )
    elif args.stage == "compare":
        if args.study_h5 is None or args.comsol_results is None:
            parser.error("--stage compare requires both --study-h5 and --comsol-results.")

    return args


# ------------------------------------------------------------------------------------------------
# --stage run
# ------------------------------------------------------------------------------------------------

def _run_stage_run(rg: Any, args: argparse.Namespace) -> int:
    print(f"Resolving back-bombardment study input (source_mode={args.source_mode!r}) ...")
    study_input = rg.resolve_back_bombardment_study_input(
        source_mode=args.source_mode,
        run_dir=args.run_dir,
    )
    print(
        f"  origin_run_id={study_input.origin_run_id!r}  source_path={study_input.source_path}\n"
        f"  event_file_hash={study_input.event_file_hash}\n"
        f"  n_events={study_input.events.n_events}  n_launched={study_input.events.n_launched}\n"
        f"  event_locator={study_input.events.event_locator!r}  "
        f"sample_represents={study_input.events.sample_represents!r}"
    )

    if args.initial_temperature_map is not None:
        raise NotImplementedError(
            f"--initial-temperature-map {args.initial_temperature_map} was given, but no loader "
            "exists yet anywhere in rf_gun for reading an initial_temperature_xy.h5 asymmetric "
            "T(x,y) map (plan Sec. 6.1's 'standard map file') into a rf_gun.TemperatureMap2D. "
            "Use --initial-temperature-uniform-k instead until that loader is implemented."
        )
    initial_temperature = rg.ConstantTemperatureMap(args.initial_temperature_uniform_k)

    thermal_dt_s = float(args.thermal_dt_ns) * 1.0e-9 if args.thermal_dt_ns is not None else None
    base_config = rg.default_uh_back_bombardment_study_config()
    config = base_config.replace(
        macropulse=rg.MacropulseConfig(duration_s=args.macropulse_duration_us * 1.0e-6),
        material=rg.CathodeMaterialSelection(material_id="LaB6", property_set=args.material_set),
        deposition=rg.DepositionConfig(model=args.deposition_model),
        thermal=dataclasses.replace(base_config.thermal, backend=args.thermal_backend, dt_s=thermal_dt_s),
        coupling=rg.CouplingConfig(level="L2_one_way"),
    )
    thermal_bin_s = float(args.thermal_bin_ns) * 1.0e-9

    comsol_results = None
    if args.comsol_results is not None:
        print(f"Loading COMSOL results from {args.comsol_results} ...")
        comsol_results = rg.load_comsol_thermal_result(args.comsol_results)

    print()
    print("Resolved study configuration:")
    print(json.dumps(rg.to_json_safe(dataclasses.asdict(config)), indent=2, sort_keys=True))
    print()

    print("Running back-bombardment macropulse study ...")
    study = rg.run_back_bombardment_macropulse_study(
        study_input=study_input,
        config=config,
        initial_temperature=initial_temperature,
        comsol_results=comsol_results,
        output_dir=args.output,
        thermal_bin_s=thermal_bin_s,
    )

    print()
    print(
        "Validating study (charge balance, BB0 energy closure, thermal energy residual, "
        "coupling level) ..."
    )
    rg.validate_back_bombardment_study(study)
    print("  validation passed.")

    _print_run_summary(study)

    output_dir = Path(args.output)
    _write_study_json(rg, study, output_dir)

    if args.save_figures:
        _save_figures(rg, study, output_dir)

    return 0


def _print_run_summary(study: Any) -> None:
    events = study.study_input.events
    accounting = events.accounting if isinstance(events.accounting, dict) else {}
    charges = accounting.get("charge_C", {})
    energy = accounting.get("energy_J", {})

    print()
    print("==== Study summary ====")
    print(f"  material: {study.material.material_id} / {study.material.property_set}")
    print(f"  deposition model: {study.config.deposition.model}")
    print(f"  thermal backend: {study.thermal_result.backend}")
    print(f"  coupling level: {study.config.coupling.level}")
    print(
        f"  macropulse: duration={study.config.macropulse.duration_s * 1.0e6:.6g} us  "
        f"envelope={study.config.macropulse.envelope!r}"
    )
    print(f"  n_events={events.n_events}  n_launched={events.n_launched}")
    print(f"  charge_C (accounting): {json.dumps(charges)}")
    print(
        f"  incident energy_J before/after filter: "
        f"{energy.get('incident_before_filter')} / {energy.get('incident_after_filter')}"
    )
    print(
        f"  charge_balance_ok={study.charge_balance_ok}  "
        f"charge_balance_error={study.charge_balance_error!r}"
    )
    print(
        f"  BB0 energy closure: incident={study.heat_source.total_incident_energy_J:.6e} J  "
        f"deposited={study.heat_source.total_deposited_energy_J:.6e} J"
    )
    print(
        f"  T_center final={float(study.thermal_result.T_center_t[-1]):.6g} K  "
        f"T_area_average final={float(study.thermal_result.T_area_average_t[-1]):.6g} K  "
        f"T_max final={float(study.thermal_result.T_max_t[-1]):.6g} K"
    )
    print(f"  thermal energy_residual_normalized={study.thermal_result.energy_residual_normalized:.3e}")
    print(f"  comsol_available={study.comparison.comsol_available}")
    print(f"  events_h5={study.events_h5}")
    print(f"  heat_source_h5={study.heat_source_h5}")
    print(f"  macropulse_h5={study.macropulse_h5}")


def _write_study_json(rg: Any, study: Any, output_dir: Path) -> None:
    """Write study_config.json / study_results.json (plan Sec. 4.2: 'human-readable controls and
    scalar summaries, using the existing run-config/run-results philosophy'). Uses
    `rf_gun.to_json_safe` throughout, matching `rf_gun.io.save_run_config`/`save_run_results`'s own
    convention for `run_thermionic_tm010.py`."""
    config_path = output_dir / "study_config.json"
    config_path.write_text(
        json.dumps(rg.to_json_safe(dataclasses.asdict(study.config)), indent=2, sort_keys=True)
    )

    events = study.study_input.events
    results = {
        "source_mode": study.study_input.source_mode,
        "origin_run_id": study.study_input.origin_run_id,
        "source_path": str(study.study_input.source_path) if study.study_input.source_path else None,
        "event_file_hash": study.study_input.event_file_hash,
        "material": {
            "material_id": study.material.material_id,
            "property_set": study.material.property_set,
        },
        "n_events": events.n_events,
        "n_launched": events.n_launched,
        "accounting": events.accounting,
        "charge_balance_ok": study.charge_balance_ok,
        "charge_balance_error": study.charge_balance_error,
        "bb0_energy_closure": {
            "total_incident_energy_J": study.heat_source.total_incident_energy_J,
            "total_deposited_energy_J": study.heat_source.total_deposited_energy_J,
            "escaping_energy_geometric_J_total": study.heat_source.escaping_energy_geometric_J_total,
            "escaping_energy_below_tio_validity_J_total": (
                study.heat_source.escaping_energy_below_tio_validity_J_total
            ),
            "excluded_non_lab6_energy_J_total": study.heat_source.excluded_non_lab6_energy_J_total,
        },
        "thermal": {
            "backend": study.thermal_result.backend,
            "T_center_final_K": float(study.thermal_result.T_center_t[-1]),
            "T_area_average_final_K": float(study.thermal_result.T_area_average_t[-1]),
            "T_max_final_K": float(study.thermal_result.T_max_t[-1]),
            "energy_residual_normalized": study.thermal_result.energy_residual_normalized,
        },
        "comsol_available": study.comparison.comsol_available,
        "output_files": {
            "events_h5": str(study.events_h5) if study.events_h5 else None,
            "heat_source_h5": str(study.heat_source_h5) if study.heat_source_h5 else None,
            "macropulse_h5": str(study.macropulse_h5) if study.macropulse_h5 else None,
        },
    }
    results_path = output_dir / "study_results.json"
    results_path.write_text(json.dumps(rg.to_json_safe(results), indent=2, sort_keys=True))

    print()
    print(f"Wrote {config_path}")
    print(f"Wrote {results_path}")


def _save_figures(rg: Any, study: Any, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_a = rg.plot_back_bombardment_source_qualification(study)
    path_a = figures_dir / "back_bombardment_source_qualification.png"
    fig_a.savefig(path_a, dpi=150)
    print(f"Wrote {path_a}")

    fig_b = rg.plot_back_bombardment_macropulse(study)
    path_b = figures_dir / "back_bombardment_macropulse.png"
    fig_b.savefig(path_b, dpi=150)
    print(f"Wrote {path_b}")


# ------------------------------------------------------------------------------------------------
# --stage compare
# ------------------------------------------------------------------------------------------------

class _ThermalHistoryProxy:
    """Minimal duck-typed stand-in for a `rf_gun.thermal.ThermalResult`, carrying only the scalar
    time histories `rf_gun.comsol_io.compare_python_comsol_thermal` actually reads
    (`t_grid_s`/`T_center_K`/`T_max_K`/`T_area_average_K`) -- see `_run_stage_compare`'s docstring
    for why a full `ThermalResult` cannot be reconstructed here."""

    def __init__(
        self,
        t_grid_s: Any,
        T_center_K: Any,
        T_max_K: Any,
        T_area_average_K: Any,
    ) -> None:
        self.t_grid_s = t_grid_s
        self.T_center_K = T_center_K
        self.T_max_K = T_max_K
        self.T_area_average_K = T_area_average_K


def _read_macropulse_thermal_proxy(study_h5_path: Path) -> _ThermalHistoryProxy:
    """Read just the `/thermal` group's scalar time histories out of a
    `back_bombardment_macropulse.h5` file (plan Sec. 4.2), by raw h5py access -- there is no
    `read_back_bombardment_macropulse_h5` function anywhere in this project (only the writer,
    `rf_gun.studies.back_bombardment_macropulse.write_back_bombardment_macropulse_h5, exists), and
    building one that reconstructs a full `rf_gun.thermal.ThermalResult` is out of scope for this
    pass: that file does not persist the full `T_surface_xyt`/`T_layer_xyzt` spatial history at all
    (by design, to bound file size -- see that module's docstring), so a full `ThermalResult`
    could not be reconstructed from it even with such a loader.
    """
    import h5py
    import numpy as np

    if not study_h5_path.is_file():
        raise ValueError(f"{study_h5_path}: file not found.")

    with h5py.File(str(study_h5_path), "r") as h5f:
        if "thermal" not in h5f:
            raise ValueError(
                f"{study_h5_path}: no '/thermal' group found -- this does not look like a "
                "back_bombardment_macropulse.h5 file written by "
                "rf_gun.studies.back_bombardment_macropulse.write_back_bombardment_macropulse_h5."
            )
        thermal_grp = h5f["thermal"]
        t_grid_s = np.asarray(thermal_grp["t_grid_s"][()], dtype=float)
        T_center_K = np.asarray(thermal_grp["T_center_K"][()], dtype=float)
        T_max_K = np.asarray(thermal_grp["T_max_K"][()], dtype=float)
        T_area_average_K = np.asarray(thermal_grp["T_area_average_K"][()], dtype=float)

    return _ThermalHistoryProxy(t_grid_s, T_center_K, T_max_K, T_area_average_K)


def _run_stage_compare(rg: Any, args: argparse.Namespace) -> int:
    """Import a COMSOL thermal result and compare it against an already-completed study.

    Scope, stated explicitly rather than faked (plan Sec. 13's `--stage compare --study-h5 ...
    --comsol-results ...` example): a full regeneration of Figure B's COMSOL overlay/difference
    maps would require reading back a complete `rf_gun.thermal.ThermalResult` (including its
    `(x,y,t)` spatial fields) from `--study-h5`. No such reader exists anywhere in this project,
    and `back_bombardment_macropulse.h5` itself does not even store the full spatial history (only
    scalar `T_center`/`T_area_average`/`T_max`/... time series -- see that file's writer's
    docstring), so a full reconstruction is not possible from this file regardless.

    What IS implemented: the real `rf_gun.compare_python_comsol_thermal` function is called
    directly against a minimal duck-typed proxy object (`_ThermalHistoryProxy`) built from those
    stored scalar histories. This reproduces the scalar (`T_center`/`T_max`/`T_area_average` vs
    macro time) half of the comparison exactly as the library function defines it; the full spatial
    surface-difference-map/hotspot-displacement fields are automatically left `None` (the same
    function gracefully skips them when the duck-typed object has no `T_surface_K`/
    `x_centers_m`/`y_centers_m` attributes) rather than fabricated.
    """
    print(f"Loading COMSOL results from {args.comsol_results} ...")
    comsol_result = rg.load_comsol_thermal_result(args.comsol_results)

    print(f"Reading resolved Python thermal scalar histories from {args.study_h5} ...")
    proxy = _read_macropulse_thermal_proxy(args.study_h5)

    print()
    print(
        "NOTE: --stage compare only compares the SCALAR thermal histories (T_center/T_max/"
        "T_area_average vs macro time) stored in back_bombardment_macropulse.h5's /thermal group. "
        "That file does not persist the full T_surface(x,y,t) spatial history (by design, to "
        "bound file size), and no loader exists yet anywhere in this project that reconstructs a "
        "full rf_gun.thermal.ThermalResult from it. A full spatial Figure-B-style difference-map "
        "comparison is therefore NOT available from this CLI stage -- only the scalar comparison "
        "below, computed by the real rf_gun.compare_python_comsol_thermal function."
    )
    print()

    comparison = rg.compare_python_comsol_thermal(proxy, comsol_result)

    print("==== COMSOL comparison summary (scalar histories only) ====")
    print(f"  comsol_available={comparison.comsol_available}")
    if comparison.max_abs_temperature_diff_K is not None:
        print(f"  max_abs_temperature_diff_K={comparison.max_abs_temperature_diff_K:.6g}")
    if comparison.mean_abs_temperature_diff_K is not None:
        print(f"  mean_abs_temperature_diff_K={comparison.mean_abs_temperature_diff_K:.6g}")
    if comparison.notes:
        print(f"  notes: {comparison.notes}")

    output_dir = Path(args.output) if args.output is not None else args.study_h5.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comsol_comparison.json"
    payload = {
        "scope": "scalar_thermal_history_only",
        "study_h5": str(args.study_h5),
        "comsol_results": str(args.comsol_results),
        "comsol_available": comparison.comsol_available,
        "max_abs_temperature_diff_K": comparison.max_abs_temperature_diff_K,
        "mean_abs_temperature_diff_K": comparison.mean_abs_temperature_diff_K,
        "surface_diff_norm_K": comparison.surface_diff_norm_K,
        "hotspot_displacement_m": comparison.hotspot_displacement_m,
        "notes": comparison.notes,
    }
    out_path.write_text(json.dumps(rg.to_json_safe(payload), indent=2, sort_keys=True))
    print(f"Wrote {out_path}")

    return 0


# ------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Deferred: importing rf_gun transitively imports RF_Track (rf_gun.config). Deferring it here
    # (rather than at module import time) matches run_thermionic_tm010.py's own convention, so
    # --help stays fast and does not itself require a working RF-Track installation.
    import rf_gun as rg

    if args.stage == "compare":
        return _run_stage_compare(rg, args)
    return _run_stage_run(rg, args)


if __name__ == "__main__":
    sys.exit(main())
