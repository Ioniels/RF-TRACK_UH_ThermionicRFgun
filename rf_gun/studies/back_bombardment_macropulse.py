"""High-level back-bombardment/macropulse study orchestration (implementation plan Sec. 4.2, 10.1,
10.2, 10.3; addendum Sec. 19.2) -- the `run_back_bombardment_macropulse_study` entry point the
notebook and CLI both call (plan Sec. 1: "The notebook and batch/SLURM paths call the same library
functions and construct the same dataclasses.").

Scope: pure Python, NO RF-Track dependency. This module wires together, in order:

    load_cathode_material                          (rf_gun.materials)
    -> build_back_bombardment_heat_source            (rf_gun.back_bombardment_deposition, BB0)
    -> build_macropulse_time_grid                    (rf_gun.macropulse)
    -> build_macropulse_heat_source                  (rf_gun.macropulse)
    -> build_macropulse_current_history              (rf_gun.macropulse)
    -> validate_charge_balance                       (rf_gun.macropulse -- WARN, do not hard-crash)
    -> solve_xy_layered_thermal                      (rf_gun.thermal)
    -> compare_python_comsol_thermal                 (rf_gun.comsol_io -- comsol_result=None by
                                                       default, addendum Sec. 19.2's "stub only for
                                                       now" decision for this pass)

into one `BackBombardmentMacropulseStudy` result object, with an explicit `coupling_level=
"L2_one_way"` guard (this implementation runs no thermal/emission/cavity feedback loop at all --
Work Package 6, addendum Sec. 19.2 -- so a caller asking for `"L3_..."`/`"L4_..."` gets a clear,
immediate `ValueError` rather than a silently mislabeled L2 result).

Charge-balance failure handling (explicit design decision, since the plan does not fix one): a
`validate_charge_balance` failure is a DATA-QUALITY problem in the upstream event-capture run, not a
bug in this orchestration code, and the rest of the study (deposition, thermal solve, comparison) is
still independently informative even if the input's charge accounting does not close. This function
therefore catches the `ValueError`, emits a `RuntimeWarning` AND a prominent `print`, records the
failure on the returned `BackBombardmentMacropulseStudy` (`charge_balance_ok`/
`charge_balance_error`), and continues -- it does NOT raise and abort the whole study. Contrast this
with `validate_back_bombardment_study` below, a later, deliberate, hard post-hoc check a caller
invokes explicitly on an already-built study; there, the same failure DOES raise (see that
function's docstring).

`back_bombardment_macropulse.h5` layout (plan Sec. 4.2: "current, current density, cavity envelope,
Python temperature, optional imported COMSOL temperature, and benchmark metrics" -- exact layout is
this implementation's own choice, following `back_bombardment_events.py`/`comsol_io.py`'s existing
schema-versioning/provenance conventions):

    (root attrs)      schema_version="back_bombardment_macropulse_v1", git_commit, timestamp_utc,
                       source_mode, origin_run_id, event_file_hash, coupling_level,
                       macropulse_duration_s, envelope, rf_frequency_Hz, N_RF, comsol_available,
                       charge_balance_ok
    /current/*         t_s, I_emit_A, I_return_A, I_transmitted_A, I_other_loss_A, I_useful_A,
                       J_emit_mean_A_m2 (coarse flat-face-average diagnostic, NOT a spatial map --
                       a full J(x,y,t) map is Figure B/plotting territory, out of this module's
                       scope)
    /cavity_envelope/* t_s, envelope_value (`evaluate_rf_envelope` at the same bin centers)
    /thermal/*          t_grid_s, T_center_K, T_area_average_K, T_max_K, T_flat_mean_K,
                       T_bevel_mean_K, stored_energy_J_t, radiation_loss_power_W_t,
                       contact_loss_power_W_t (attrs: backend, material_property_set,
                       total_input_energy_J, energy_residual_normalized)
    /comsol/*           present only if comsol_result was supplied; attrs comsol_available,
                       max_abs_temperature_diff_K, mean_abs_temperature_diff_K,
                       surface_diff_norm_K, hotspot_displacement_m, notes; datasets
                       aligned_time_s/T_center_diff_K/T_max_diff_K/T_area_average_diff_K when
                       available. When `comsol_result=None`, this group holds ONLY
                       `comsol_available=False` and `notes` -- never fabricated COMSOL data (plan
                       Sec. 8.2/12; addendum Sec. 19.2).
    /benchmark_metrics  a single `json` attribute: charge-balance numbers (Q_emit/Q_return/
                       Q_transmitted/Q_other_loss/Q_surviving/charge_balance_ok), BB0 energy-
                       closure numbers (incident/deposited/escaping totals), and the thermal
                       energy residual.
    /config             a single `json` attribute: `to_json_safe(dataclasses.asdict(config))`.
    /provenance         a single `json` attribute: schema_version, git_commit, timestamp_utc,
                       source_mode, origin_run_id, source_path, event_file_hash.
"""
from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..back_bombardment_deposition import (
    BackBombardmentHeatSource,
    build_back_bombardment_heat_source,
    validate_energy_closure,
    write_back_bombardment_heat_source_h5,
)
from ..back_bombardment_events import (
    BackBombardmentStudyInput,
    _git_commit_hash,
    write_back_bombardment_events_h5,
)
from ..back_bombardment_study_config import BackBombardmentStudyConfig
from ..cathode_geometry import SURFACE_UNKNOWN
from ..comsol_io import ComsolComparison, ComsolThermalResult, compare_python_comsol_thermal
from ..io import to_json_safe
from ..macropulse import (
    MacropulseCurrentHistory,
    build_macropulse_current_history,
    build_macropulse_heat_source,
    build_macropulse_time_grid,
    compute_n_rf_periods,
    evaluate_rf_envelope,
    validate_charge_balance,
)
from ..materials.base import CathodeMaterialSet
from ..materials.registry import load_cathode_material
from ..thermal import ThermalResult, VolumetricHeatSourceTimeSeries, solve_xy_layered_thermal

__all__ = [
    "BACK_BOMBARDMENT_MACROPULSE_SCHEMA_VERSION",
    "BackBombardmentMacropulseStudy",
    "run_back_bombardment_macropulse_study",
    "validate_back_bombardment_study",
    "write_back_bombardment_macropulse_h5",
]

#: Bump only when a field is added/removed/reinterpreted in a way that would break a reader written
#: against an earlier version -- same convention as `BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION`/
#: `COMSOL_SOURCE_SCHEMA_VERSION`.
BACK_BOMBARDMENT_MACROPULSE_SCHEMA_VERSION = "back_bombardment_macropulse_v1"

#: Only coupling level this orchestration function actually implements (module docstring). Any
#: other `config.coupling.level` raises immediately in `run_back_bombardment_macropulse_study` --
#: never silently produces an L2 result under a different label.
_IMPLEMENTED_COUPLING_LEVEL = "L2_one_way"


@dataclass(frozen=True)
class BackBombardmentMacropulseStudy:
    """High-level result of `run_back_bombardment_macropulse_study` (plan Sec. 10.2: "file paths,
    configuration, accounting tables, event/source objects, Python thermal result, optional COMSOL
    result, benchmark metrics, and figure-ready arrays").

    `charge_balance_ok`/`charge_balance_error`: outcome of the (non-fatal, warn-only)
    `validate_charge_balance` check run during construction -- see module docstring's "Charge-
    balance failure handling" paragraph. `charge_balance_error` is `None` exactly when
    `charge_balance_ok` is `True`.

    `events_h5`/`heat_source_h5`/`macropulse_h5`: `Path | None`, populated only when `output_dir`
    was supplied to `run_back_bombardment_macropulse_study` AND that file was actually written.
    `heat_source_h5` is `<output_dir>/back_bombardment_heat_source.h5` (plan Sec. 4.2), written via
    `rf_gun.back_bombardment_deposition.write_back_bombardment_heat_source_h5` with
    `source_events_hash=study_input.event_file_hash` (plan Sec. 4.1's "the hash of the input event
    file", applied to this exact file).
    """

    study_input: BackBombardmentStudyInput
    config: BackBombardmentStudyConfig
    material: CathodeMaterialSet
    heat_source: BackBombardmentHeatSource
    macropulse_heat_source: VolumetricHeatSourceTimeSeries
    current_history: MacropulseCurrentHistory
    thermal_result: ThermalResult
    comsol_result: ComsolThermalResult | None
    comparison: ComsolComparison
    charge_balance_ok: bool
    charge_balance_error: str | None
    events_h5: Path | None = None
    heat_source_h5: Path | None = None
    macropulse_h5: Path | None = None
    output_dir: Path | None = None


def run_back_bombardment_macropulse_study(
    study_input: BackBombardmentStudyInput,
    config: BackBombardmentStudyConfig,
    initial_temperature: Any,
    comsol_results: ComsolThermalResult | None = None,
    *,
    output_dir: str | Path | None = None,
    thermal_bin_s: float = 50e-9,
) -> BackBombardmentMacropulseStudy:
    """Run the full `coupling_level="L2_one_way"` back-bombardment/macropulse study (plan Sec. 10.2's
    suggested public function; addendum Sec. 19.2's Work Package 0-4 scope).

    `thermal_bin_s`: forwarded to `rf_gun.macropulse.build_macropulse_time_grid` -- a normal,
    caller-configurable macro-time-grid resolution (plan Sec. 8.1), never hardcoded here.

    Raises `ValueError` immediately if `config.coupling.level != "L2_one_way"`: this function
    implements no thermal/emission/cavity feedback loop at all (Work Package 6, addendum
    Sec. 19.2), so any other coupling level would be produced under a mislabeled name.

    A `validate_charge_balance` failure on `study_input.events` is WARNED about (via both
    `warnings.warn` and a prominent `print`) and recorded on the returned study
    (`charge_balance_ok`/`charge_balance_error`), but does NOT abort the study -- see this module's
    docstring for the reasoning.

    If `output_dir` is given, writes `<output_dir>/back_bombardment_macropulse.h5` (see this
    module's docstring for its layout) and sets `macropulse_h5` accordingly. Also writes
    `<output_dir>/back_bombardment_events.h5` (`events_h5`) unless `study_input` already came from
    exactly that path (`source_mode="load_run"` reading from the same directory), in which case the
    existing file is reused rather than rewritten.
    """
    if config.coupling.level != _IMPLEMENTED_COUPLING_LEVEL:
        raise ValueError(
            f"run_back_bombardment_macropulse_study: config.coupling.level="
            f"{config.coupling.level!r} is not implemented. This function runs no thermal/"
            f"emission/cavity feedback loop at all (Work Package 6, addendum Sec. 19.2) -- only "
            f"{_IMPLEMENTED_COUPLING_LEVEL!r} is supported. Using a different coupling level here "
            "would silently mislabel an L2-only result."
        )

    events = study_input.events

    material = load_cathode_material(config.material.material_id, config.material.property_set)
    print(
        f"run_back_bombardment_macropulse_study: material_id={config.material.material_id!r} "
        f"property_set={config.material.property_set!r} resolved."
    )

    heat_source = build_back_bombardment_heat_source(
        events, config.geometry, material, config.deposition
    )
    print(
        "run_back_bombardment_macropulse_study: BB0 deposition built -- "
        f"total_incident_energy_J={heat_source.total_incident_energy_J:.6e}, "
        f"total_deposited_energy_J={heat_source.total_deposited_energy_J:.6e}."
    )

    t_grid_edges_s = build_macropulse_time_grid(config.macropulse, thermal_bin_s=thermal_bin_s)
    n_rf = compute_n_rf_periods(events.rf_frequency_Hz, config.macropulse)
    print(
        f"run_back_bombardment_macropulse_study: macropulse duration_s="
        f"{config.macropulse.duration_s:.6g}, envelope={config.macropulse.envelope!r}, "
        f"rf_frequency_Hz={events.rf_frequency_Hz:.6g}, N_RF={n_rf:.6g}, "
        f"n_thermal_bins={t_grid_edges_s.size - 1} (thermal_bin_s={thermal_bin_s:.6g})."
    )

    macropulse_heat_source = build_macropulse_heat_source(
        heat_source, events.rf_frequency_Hz, config.macropulse, t_grid_edges_s
    )
    current_history = build_macropulse_current_history(events, config.macropulse, t_grid_edges_s)

    charge_balance_ok = True
    charge_balance_error: str | None = None
    try:
        validate_charge_balance(events)
    except ValueError as exc:
        charge_balance_ok = False
        charge_balance_error = str(exc)
        warnings.warn(
            "run_back_bombardment_macropulse_study: charge balance check FAILED for this study's "
            f"input events (study continues -- see module docstring): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        print(f"WARNING: run_back_bombardment_macropulse_study: charge balance check FAILED: {exc}")

    thermal_result = solve_xy_layered_thermal(
        macropulse_heat_source, initial_temperature, config.geometry, material, config.thermal
    )
    print(
        "run_back_bombardment_macropulse_study: thermal solve complete -- backend="
        f"{thermal_result.backend!r}, T_max_final={float(thermal_result.T_max_t[-1]):.6g} K, "
        f"energy_residual_normalized={thermal_result.energy_residual_normalized:.3e}."
    )

    comparison = compare_python_comsol_thermal(thermal_result, comsol_results)
    print(
        f"run_back_bombardment_macropulse_study: COMSOL comparison -- "
        f"comsol_available={comparison.comsol_available}."
    )

    events_h5: Path | None = None
    heat_source_h5: Path | None = None
    macropulse_h5: Path | None = None

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        candidate_events_h5 = output_dir / "back_bombardment_events.h5"
        if study_input.source_path is not None and Path(study_input.source_path).resolve() == candidate_events_h5.resolve():
            events_h5 = candidate_events_h5
            print(f"run_back_bombardment_macropulse_study: reusing existing {events_h5}.")
        else:
            events_h5 = write_back_bombardment_events_h5(candidate_events_h5, events)
            print(f"run_back_bombardment_macropulse_study: wrote {events_h5}.")

        heat_source_h5 = write_back_bombardment_heat_source_h5(
            output_dir / "back_bombardment_heat_source.h5",
            heat_source,
            source_events_hash=study_input.event_file_hash,
        )
        print(f"run_back_bombardment_macropulse_study: wrote {heat_source_h5}.")

        macropulse_h5 = write_back_bombardment_macropulse_h5(
            output_dir / "back_bombardment_macropulse.h5",
            study_input=study_input,
            config=config,
            heat_source=heat_source,
            macropulse_heat_source=macropulse_heat_source,
            current_history=current_history,
            thermal_result=thermal_result,
            comparison=comparison,
            charge_balance_ok=charge_balance_ok,
            charge_balance_error=charge_balance_error,
            n_rf=n_rf,
        )
        print(f"run_back_bombardment_macropulse_study: wrote {macropulse_h5}.")

    return BackBombardmentMacropulseStudy(
        study_input=study_input,
        config=config,
        material=material,
        heat_source=heat_source,
        macropulse_heat_source=macropulse_heat_source,
        current_history=current_history,
        thermal_result=thermal_result,
        comsol_result=comsol_results,
        comparison=comparison,
        charge_balance_ok=charge_balance_ok,
        charge_balance_error=charge_balance_error,
        events_h5=events_h5,
        heat_source_h5=heat_source_h5,
        macropulse_h5=macropulse_h5,
        output_dir=output_dir,
    )


def validate_back_bombardment_study(study: BackBombardmentMacropulseStudy) -> None:
    """Post-hoc self-consistency guard on an already-built `BackBombardmentMacropulseStudy` (plan
    Sec. 10.2).

    Unlike `run_back_bombardment_macropulse_study`'s own construction-time charge-balance check
    (which WARNS and continues -- see that function's docstring), this is a deliberate, explicit
    HARD check a caller invokes on a finished study: it raises `ValueError` (propagated from
    whichever sub-check failed, unchanged) if ANY of the following do not hold:

      1. `validate_charge_balance(study.study_input.events)` (plan Sec. 8.5).
      2. `validate_energy_closure(study.heat_source, study.study_input.events)` (BB0 energy
         closure, plan Sec. 3.4/15.2).
      3. `study.thermal_result.energy_residual_normalized <= study.config.thermal.energy_residual_tol`
         (plan Sec. 15.3's own configured tolerance).
      4. `study.config.coupling.level == "L2_one_way"` -- this project's only implemented coupling
         level (module docstring); a mismatch here means the study's OWN configuration claims a
         feedback loop ran that this code base cannot actually run, a basic self-consistency guard
         against a future caller accidentally mislabeling a run (plan Sec. 10.2).
      5. The fraction of qualified events with `surface_code == SURFACE_UNKNOWN` does not exceed
         `study.config.capture.max_unknown_surface_fraction`. `extract_back_bombardment_events`
         only *warns* about this at capture time (it does not know whether its caller is an
         exploratory or production run); this is the point where it becomes fatal.
    """
    validate_charge_balance(study.study_input.events)
    validate_energy_closure(study.heat_source, study.study_input.events)

    residual = study.thermal_result.energy_residual_normalized
    tol = study.config.thermal.energy_residual_tol
    if abs(residual) > tol:
        raise ValueError(
            f"validate_back_bombardment_study: thermal energy_residual_normalized={residual:.3e} "
            f"exceeds config.thermal.energy_residual_tol={tol:.3e} (plan Sec. 15.3)."
        )

    if study.config.coupling.level != _IMPLEMENTED_COUPLING_LEVEL:
        raise ValueError(
            f"validate_back_bombardment_study: study.config.coupling.level="
            f"{study.config.coupling.level!r} but run_back_bombardment_macropulse_study only ever "
            f"implements {_IMPLEMENTED_COUPLING_LEVEL!r} (no feedback loop ran) -- this study's own "
            "configuration is mislabeled relative to what was actually computed."
        )

    surface_code = np.asarray(study.study_input.events.surface_code)
    n_events = int(surface_code.size)
    unknown_frac = float(np.mean(surface_code == SURFACE_UNKNOWN)) if n_events else 0.0
    max_unknown_frac = float(study.config.capture.max_unknown_surface_fraction)
    if unknown_frac > max_unknown_frac:
        raise ValueError(
            f"validate_back_bombardment_study: {unknown_frac:.3%} of {n_events} qualified events "
            f"landed on SURFACE_UNKNOWN (no physical surface intersection), exceeding "
            f"config.capture.max_unknown_surface_fraction={max_unknown_frac:.3%} -- the placeholder "
            "geometry is not adequate for this run's return distribution; heating/current claims "
            "from it cannot be trusted at production scale."
        )


def write_back_bombardment_macropulse_h5(
    path: str | Path,
    *,
    study_input: BackBombardmentStudyInput,
    config: BackBombardmentStudyConfig,
    heat_source: BackBombardmentHeatSource,
    macropulse_heat_source: VolumetricHeatSourceTimeSeries,
    current_history: MacropulseCurrentHistory,
    thermal_result: ThermalResult,
    comparison: ComsolComparison,
    charge_balance_ok: bool,
    charge_balance_error: str | None,
    n_rf: float,
) -> Path:
    """Write `back_bombardment_macropulse.h5` (plan Sec. 4.2) -- see this module's docstring for
    the exact group layout. Follows `back_bombardment_events.py`/`comsol_io.py`'s conventions: a
    strict `schema_version` root attribute, JSON blobs in a single `json` group attribute for
    caller-defined nested dicts (`/config`, `/provenance`, `/benchmark_metrics`), and plain HDF5
    datasets (with a `units` attribute where meaningful) for numeric time series.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "h5py is required to write back_bombardment_macropulse_v1 HDF5 files. "
            "Install it with 'pip install h5py'."
        ) from exc

    events = study_input.events
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t_centers_s = current_history.t_s
    envelope_value = evaluate_rf_envelope(t_centers_s, config.macropulse)

    flat_area_m2 = float(config.geometry.flat_area_mm2) * 1.0e-6
    J_emit_mean_A_m2 = current_history.I_emit_A / flat_area_m2 if flat_area_m2 > 0.0 else np.full_like(current_history.I_emit_A, np.nan)

    charges = events.accounting.get("charge_C", {}) if isinstance(events.accounting, dict) else {}
    Q_emit = float(charges.get("emitted", np.nan))
    Q_return = float(charges.get("returned_after_filter", np.nan))
    Q_transmitted = float(charges.get("transmitted", np.nan))
    Q_other_loss = float(charges.get("other_lost", np.nan))
    Q_surviving = Q_emit - (Q_return + Q_transmitted + Q_other_loss)

    benchmark_metrics = {
        "charge_balance": {
            "Q_emit_C": Q_emit,
            "Q_return_after_filter_C": Q_return,
            "Q_transmitted_C": Q_transmitted,
            "Q_other_loss_C": Q_other_loss,
            "Q_surviving_C": Q_surviving,
            "charge_balance_ok": bool(charge_balance_ok),
            "charge_balance_error": charge_balance_error,
        },
        "bb0_energy_closure": {
            "total_incident_energy_J": float(heat_source.total_incident_energy_J),
            "total_deposited_energy_J": float(heat_source.total_deposited_energy_J),
            "escaping_energy_geometric_J_total": float(heat_source.escaping_energy_geometric_J_total),
            "escaping_energy_below_tio_validity_J_total": float(
                heat_source.escaping_energy_below_tio_validity_J_total
            ),
            "excluded_non_lab6_energy_J_total": float(heat_source.excluded_non_lab6_energy_J_total),
        },
        "thermal": {
            "energy_residual_normalized": float(thermal_result.energy_residual_normalized),
            "total_input_energy_J": float(thermal_result.total_input_energy_J),
        },
        "n_rf_periods": float(n_rf),
    }

    with h5py.File(str(path), "w") as h5f:
        h5f.attrs["schema_version"] = BACK_BOMBARDMENT_MACROPULSE_SCHEMA_VERSION
        h5f.attrs["git_commit"] = _git_commit_hash()
        h5f.attrs["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        h5f.attrs["source_mode"] = study_input.source_mode
        h5f.attrs["origin_run_id"] = study_input.origin_run_id if study_input.origin_run_id is not None else ""
        h5f.attrs["event_file_hash"] = study_input.event_file_hash
        h5f.attrs["coupling_level"] = config.coupling.level
        h5f.attrs["macropulse_duration_s"] = config.macropulse.duration_s
        h5f.attrs["envelope"] = config.macropulse.envelope
        h5f.attrs["rf_frequency_Hz"] = events.rf_frequency_Hz
        h5f.attrs["N_RF"] = float(n_rf)
        h5f.attrs["comsol_available"] = comparison.comsol_available
        h5f.attrs["charge_balance_ok"] = bool(charge_balance_ok)

        current_grp = h5f.create_group("current")
        for name, arr, unit in (
            ("t_s", t_centers_s, "s"),
            ("I_emit_A", current_history.I_emit_A, "A"),
            ("I_return_A", current_history.I_return_A, "A"),
            ("I_transmitted_A", current_history.I_transmitted_A, "A"),
            ("I_other_loss_A", current_history.I_other_loss_A, "A"),
            ("I_useful_A", current_history.I_useful_A, "A"),
            ("J_emit_mean_A_m2", J_emit_mean_A_m2, "A/m^2"),
        ):
            ds = current_grp.create_dataset(name, data=np.asarray(arr, dtype=float))
            ds.attrs["units"] = unit

        cavity_grp = h5f.create_group("cavity_envelope")
        cavity_grp.create_dataset("t_s", data=t_centers_s).attrs["units"] = "s"
        cavity_grp.create_dataset("envelope_value", data=envelope_value).attrs["units"] = ""

        thermal_grp = h5f.create_group("thermal")
        for name, arr, unit in (
            ("t_grid_s", thermal_result.t_grid_s, "s"),
            ("T_center_K", thermal_result.T_center_t, "K"),
            ("T_area_average_K", thermal_result.T_area_average_t, "K"),
            ("T_max_K", thermal_result.T_max_t, "K"),
            ("T_flat_mean_K", thermal_result.T_flat_mean_t, "K"),
            ("T_bevel_mean_K", thermal_result.T_bevel_mean_t, "K"),
            ("stored_energy_J_t", thermal_result.stored_energy_J_t, "J"),
            ("radiation_loss_power_W_t", thermal_result.radiation_loss_power_W_t, "W"),
            ("contact_loss_power_W_t", thermal_result.contact_loss_power_W_t, "W"),
        ):
            ds = thermal_grp.create_dataset(name, data=np.asarray(arr, dtype=float))
            ds.attrs["units"] = unit
        thermal_grp.attrs["backend"] = thermal_result.backend
        thermal_grp.attrs["material_property_set"] = thermal_result.material_property_set
        thermal_grp.attrs["total_input_energy_J"] = float(thermal_result.total_input_energy_J)
        thermal_grp.attrs["energy_residual_normalized"] = float(thermal_result.energy_residual_normalized)

        comsol_grp = h5f.create_group("comsol")
        comsol_grp.attrs["comsol_available"] = comparison.comsol_available
        comsol_grp.attrs["notes"] = comparison.notes
        if comparison.comsol_available:
            if comparison.max_abs_temperature_diff_K is not None:
                comsol_grp.attrs["max_abs_temperature_diff_K"] = float(comparison.max_abs_temperature_diff_K)
            if comparison.mean_abs_temperature_diff_K is not None:
                comsol_grp.attrs["mean_abs_temperature_diff_K"] = float(comparison.mean_abs_temperature_diff_K)
            if comparison.surface_diff_norm_K is not None:
                comsol_grp.attrs["surface_diff_norm_K"] = float(comparison.surface_diff_norm_K)
            if comparison.hotspot_displacement_m is not None:
                comsol_grp.attrs["hotspot_displacement_m"] = float(comparison.hotspot_displacement_m)
            for name, arr in (
                ("aligned_time_s", comparison.aligned_time_s),
                ("T_center_diff_K", comparison.T_center_diff_K),
                ("T_max_diff_K", comparison.T_max_diff_K),
                ("T_area_average_diff_K", comparison.T_area_average_diff_K),
            ):
                if arr is not None:
                    comsol_grp.create_dataset(name, data=np.asarray(arr, dtype=float))

        h5f.create_group("benchmark_metrics").attrs["json"] = json.dumps(to_json_safe(benchmark_metrics))
        h5f.create_group("config").attrs["json"] = json.dumps(to_json_safe(dataclasses.asdict(config)))
        h5f.create_group("provenance").attrs["json"] = json.dumps(
            to_json_safe(
                {
                    "schema_version": BACK_BOMBARDMENT_MACROPULSE_SCHEMA_VERSION,
                    "git_commit": _git_commit_hash(),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "source_mode": study_input.source_mode,
                    "origin_run_id": study_input.origin_run_id,
                    "source_path": study_input.source_path,
                    "event_file_hash": study_input.event_file_hash,
                }
            )
        )

    return path
