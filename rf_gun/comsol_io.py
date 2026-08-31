"""COMSOL exchange: heat-source export, thermal-result import, and Python/COMSOL comparison
(implementation plan Sec. 7; addendum Sec. 19.2 -- Work Package 5, DATA-CONTRACT/INTERFACE ONLY).

Explicit scope per the addendum's user-confirmed decision (Sec. 19.2): "implement the
data-contract/export/import interface only ... so it is ready to receive a real model later; no
COMSOL run exists yet, and comparison figures must mark the COMSOL curves as unavailable rather
than fabricate placeholder data." Concretely, this means:

  * `export_comsol_heat_source` writes a real, self-consistent adapter file pair from this
    project's own `BackBombardmentHeatSource` -- no COMSOL software is invoked or required.
  * `load_comsol_thermal_result`/`ComsolThermalResult` define and validate a documented HDF5
    contract for a COMSOL-produced result -- since no such file exists yet, this module is tested
    by round-tripping against files this project's OWN test suite writes as stand-ins
    (`tests/test_comsol_io.py`'s `write_synthetic_comsol_result_for_testing`), never against a
    real COMSOL export.
  * `compare_python_comsol_thermal(python_result, comsol_result=None)` is the single most
    important function in this module: when no COMSOL result is available, it returns
    `comsol_available=False` with every COMSOL-side field `None` -- it NEVER fabricates,
    interpolates, or defaults in placeholder COMSOL data (plan Sec. 8.2/12; addendum Sec. 19.2).

--------------------------------------------------------------------------------------------
1. Export contract: `<directory>/comsol_source/comsol_heat_source.csv` +
   `comsol_source_manifest.json` (schema `"comsol_source_v1"`)
--------------------------------------------------------------------------------------------

Coordinate/normal convention (identical to `rf_gun.cathode_geometry`/`rf_gun.back_bombardment_events`,
restated here so COMSOL side has no ambiguity): the origin `(0,0,0)` is the CENTER of the flat
emitting face, AT the cathode's emission surface. `x,y` are lateral Cartesian coordinates in the
cathode's own frame (untouched by `insertion_offset_mm`, which is pure bookkeeping metadata, not a
coordinate shift -- see `CathodeGeometry`'s own docstring). `z=0` is the emission surface; the
LaB6 solid occupies `z<=0` (increasing depth into the cathode is increasingly NEGATIVE z); vacuum
and the beam channel occupy `z>=0`. This is the axis that is "into the cathode" COMSOL needs.

CSV columns, one row per `(lateral cell, depth layer)` pair inside `heat_source.cathode_footprint_mask`
(i.e. every cell of the uniform lateral grid within `bevel_outer_radius_mm`, times every depth
layer -- a full rectangular grid suitable for a COMSOL interpolation function, including cells/
layers with zero deposited energy, not just the cells an event happened to land in):

  * `x_m`, `y_m`: lateral cell-center coordinates [m].
  * `z_m`: depth-layer MIDPOINT coordinate [m] (`z_lo_m`, `z_hi_m` are also given as the layer's
    exact bounds) -- the actual 3D coordinate, per plan Sec. 7's explicit requirement, not merely
    an `(x, y, layer-index)` tuple.
  * `layer_index`: integer index into `heat_source.layer_boundaries_um` (0-based).
  * `surface_code`, `zone_label`: `rf_gun.cathode_geometry` surface-zone code/label
    (`classify_surface_by_radius`, radius-only classification is exact here because
    `cathode_footprint_mask` never extends past `bevel_outer_radius_mm`, so every row is
    `cathode_flat` or `cathode_bevel` -- NEVER `holder`; see the "holder" paragraph below) --
    lets COMSOL separate flat/bevel contributions per plan Sec. 7 without re-deriving geometry.
  * `q_BB_W_m3`: volumetric heating-power density -- see "Power-density conversion" below for the
    exact, explicitly documented meaning (this is NOT a macropulse-integrated or time-varying
    source; that is Work Package 3/4's job).
  * `deposited_J`: this row's EXACT deposited energy for the one representative RF period
    `heat_source` represents (`heat_source.q_layer_J[ix,iy,layer]`, reused verbatim -- never
    recomputed).
  * `incident_energy_attributed_J`, `escaping_energy_attributed_J`: see "Per-row incident/escaping
    attribution" below -- a documented bookkeeping APPORTIONMENT (not a spatially resolved
    incident/escape model), constructed so that summing each column over every exported row
    reproduces `heat_source`'s own recorded `total_incident_energy_J` and
    `escaping_energy_geometric_J_total + escaping_energy_below_tio_validity_J_total` totals
    exactly (up to floating-point round-off) -- this is the "correct closure" the task requires
    and `tests/test_comsol_io.py` checks directly.

Power-density conversion (documented explicitly, per plan Sec. 7/8.1, so a reader never has to
guess): `BackBombardmentHeatSource.q_layer_J` is deposited ENERGY [J] for ONE representative RF
period (plan Sec. 6.2's `q''_layer(x,y,ell,t)` with the macropulse time axis `t` collapsed to one
slice) -- no macropulse/keyframe time axis exists yet at this stage of the pipeline (that is Work
Package 3/4's job, deferred here per addendum Sec. 19.2). Three modes, selected by the caller:

  * `rf_frequency_Hz` given (the default): `q_BB_W_m3 = deposited_J * rf_frequency_Hz /
    cell_volume_m3`, i.e. plan Sec. 8.1's own `P_dep,n = f_RF * sum(E_dep,i,n)` formula applied
    per cell -- the instantaneous volumetric power density IF this one representative period's
    source repeated unchanged at `f_RF` forever (the "documented 8us top-hat" idealization of plan
    Sec. 8.2, collapsed to its own steady-state limit). This is a REPRESENTATIVE-PERIOD SNAPSHOT,
    not a real macropulse time history -- `manifest["power_density_conversion"]` states this
    explicitly and records the exact `rf_frequency_Hz` used.
  * `q_layer_W_m3_override` given instead (an array shaped like `heat_source.q_layer_J`): the
    caller already has a genuinely time-resolved (or otherwise independently computed) volumetric
    power density for this snapshot and wants it exported verbatim in place of the
    frequency-scaled conversion above -- this is the "accept an optional already-time-resolved
    array as an alternative input" option the task allows.
  * both `rf_frequency_Hz=None` and no override: `q_BB_W_m3` is written as `nan` for every row and
    the manifest records `power_density_conversion.mode = "energy_density_only"` -- explicitly NOT
    a power density, so a reader can never mistake a stale/default frequency assumption for a
    validated one.

Per-row incident/escaping attribution: BB0 (`back_bombardment_deposition.py`) tracks per-cell
DEPOSITED energy exactly, but `total_incident_energy_J` and the two escape totals
(`escaping_energy_geometric_J_total`, `escaping_energy_below_tio_validity_J_total`) are aggregate
scalars -- a single event's incident energy spreads across several depth layers along its own CSDA
path, so there is no physically well-defined per-cell incident/escape split in
`BackBombardmentHeatSource` itself. This module apportions each aggregate total across rows in
proportion to that row's share of `total_deposited_energy_J` (rows with zero deposited energy get
zero attributed incident/escaping energy) purely so file-level sums reproduce the source's own
recorded totals exactly -- this is bookkeeping, not a spatially resolved incident/escape model, and
is stated as such in the manifest's `per_row_attribution_note`.

Holder contributions: `heat_source.cathode_footprint_mask` never extends past
`bevel_outer_radius_mm` (BB0 excludes non-LaB6 events entirely -- plan Sec. 3.3, "excluded from
LaB6-only Python source"), so no CSV row is ever labeled `holder`. The holder's aggregate incident
energy (`heat_source.excluded_non_lab6_energy_J_total`) is still recorded in the manifest (plan
Sec. 7's "separate flat, bevel, and holder contributions" is satisfied at the metadata level for
holder, since BB0 has no spatial holder-deposition model to export) but is NOT part of any CSV row
or its closure sums.

The manifest additionally records the resolved material-property table (`material.to_manifest_dict()`),
the resolved `CathodeGeometry` fields, `schema_version`, git commit, timestamp, and -- only if the
caller supplies them via `extra_metadata` -- forward-compatible RF-envelope/keyframe-interpolation
metadata PLACEHOLDER fields (plan Sec. 7's "RF envelope and keyframe interpolation metadata"; no
macropulse/keyframe system is wired in yet, so these are never fabricated by this function, only
passed through verbatim when supplied).

--------------------------------------------------------------------------------------------
2. Import contract: `ComsolThermalResult` / `load_comsol_thermal_result` (schema
   `"comsol_thermal_result_v1"`)
--------------------------------------------------------------------------------------------

A single HDF5 file, following the same strict schema-version-gated convention as
`rf_gun.back_bombardment_events.read_back_bombardment_events_h5` (reject a file with no/mismatched
`schema_version` root attribute, clear actionable error message -- never silently accept an
unversioned or wrong-version file):

  * Root attrs: `schema_version` (must equal `COMSOL_THERMAL_RESULT_SCHEMA_VERSION` exactly),
    `mesh_id` (str), `time_step_id` (str).
  * `/thermal/{time_s, T_center_K, T_flat_edge_K, T_bevel_K, T_max_K, T_area_average_K,
    stored_energy_J, radiation_loss_W, boundary_heat_flow_W}`: required 1D datasets, all shape
    `(n_t,)`.
  * `/convergence` group: a `json` attribute holding an arbitrary JSON-encoded
    `convergence_metrics` dict (mesh/time-step convergence results, plan Sec. 7's "mesh/time-step
    identifiers and convergence results") -- same lossless-JSON-blob-in-one-attribute convention
    `back_bombardment_events.py` already uses for its own caller-defined nested dicts
    (`/accounting`, `/provenance`, `/source_state`). Optional; defaults to `{}` if absent.
  * `/surface_field` group (OPTIONAL): `x_m` `(n_x,)`, `y_m` `(n_y,)`,
    `T_surface_full_K` `(n_x, n_y, n_t)` -- the optional full surface-temperature field. Present
    only if the writer chose to include it; `load_comsol_thermal_result` returns `None` for
    `T_surface_full_K`/`x_m`/`y_m` when this group is absent.

`write_synthetic_comsol_result_for_testing` (deliberately NOT defined in this production module --
see `tests/test_comsol_io.py`) writes a self-consistent fake result in exactly this format, purely
so `load_comsol_thermal_result` has something real to round-trip against; no genuine COMSOL output
exists yet (addendum Sec. 19.2).

--------------------------------------------------------------------------------------------
3. Comparison: `compare_python_comsol_thermal` / `ComsolComparison`
--------------------------------------------------------------------------------------------

`python_result` is DELIBERATELY duck-typed, not imported from `rf_gun.thermal` (that module is
still being built by a concurrent task -- plan Sec. 10.1's `rf_gun/thermal.py` had not landed at
the time this module was written). The minimal attributes this function actually reads (best
guesses at `rf_gun.thermal`'s eventual naming, from plan Sec. 6.2's stated output list -- "Report
`T_center`, area average, `T_max` ... hotspot centroid, x/y lineouts, stored energy, ..." -- a
later integration pass reconciles exact names once `rf_gun/thermal.py` lands):

  * `t_grid_s` (REQUIRED -- macro-time grid, seconds; `ValueError` if missing entirely, since there
    is then nothing to align against).
  * `T_center_K`, `T_max_K`, `T_area_average_K` (all optional, each `(n_t,)`, matching
    `ComsolThermalResult`'s own field names) -- any subset may be absent; this function degrades
    gracefully (records a note, skips that one diff) rather than crashing, since a partial python
    result is still useful to compare against COMSOL where both sides overlap.
  * `T_surface_K` `(n_x, n_y, n_t)` plus `x_centers_m` `(n_x,)`, `y_centers_m` `(n_y,)` (all
    optional) -- enables the full surface difference map and hotspot-displacement metric ONLY
    when both sides supply a full field AND the two lateral grids match exactly (a mismatched
    grid is documented as skipped, not silently regridded/averaged away).

`compare_python_comsol_thermal` aligns time by interpolating the COMSOL time histories onto the
overlap of the two time grids (`np.interp`, matching plan Sec. 7's "aligns coordinates/time");
COMSOL surface data are placed onto the Python grid without azimuthal averaging when grids match
exactly, exactly as plan Sec. 7 requires ("interpolated onto the Python Cartesian grid without
azimuthal averaging") -- a genuine COMSOL mesh interpolation onto an arbitrary Python grid is
Work Package 6 scope; here, an exact-match check is a correct, honest special case, not a
placeholder for later regridding logic silently pretending to be complete.

**`comsol_result=None` is the load-bearing behavior of this whole module** (plan Sec. 8.2/12,
addendum Sec. 19.2's explicit user decision): the return is `ComsolComparison(comsol_available=False,
...)` with every COMSOL-side field `None` -- never fabricated, interpolated, or defaulted.
`tests/test_comsol_io.py` tests this explicitly and is the single most important test in that file.

--------------------------------------------------------------------------------------------
4. Equivalence-test scaffold (plan Sec. 7's "Before using the real 3D geometry, run an
   equivalence case ...") -- DOCUMENTED HERE ONLY, NOT IMPLEMENTED
--------------------------------------------------------------------------------------------

No real COMSOL model exists to run this against (addendum Sec. 19.2). A future equivalence test,
built on top of this module's export contract, would need to check -- all simultaneously, on the
SAME Cartesian cathode footprint used by `python_xy_layered`:

  * identical `(x, y)` cell centers/footprint mask (export `heat_source.x_centers_m`,
    `y_centers_m`, `cathode_footprint_mask` directly; do not let COMSOL re-derive its own grid);
  * CONSTANT material properties on both sides (e.g. `LaB6_constant_verification_v1`, plan
    Sec. 5.2's dedicated "never a production physics set" analytic-test material) so any
    discrepancy cannot be blamed on differing nonlinear property evaluation;
  * identical boundary maps (the same `h_contact(x,y)`/`T_mount(x,y,t)` convention on both sides,
    not COMSOL's own full heater/support geometry -- that full-geometry comparison is the LATER,
    separate step plan Sec. 7 describes after the equivalence case passes);
  * identical depth layers and volumetric source (export exactly `heat_source.layer_boundaries_um`
    and `q_layer_J`/`q_BB_W_m3` as computed here -- COMSOL must consume this file's rows directly,
    not an independently-derived source);
  * identical initial `T(x, y)` field (the same `ConstantTemperatureMap`/`TemperatureMap2D`, plan
    Sec. 6.1, fed to both solvers).

Only once all five match exactly can a Python/COMSOL difference be attributed to genuine 3D/
heater/support/radiation modeling differences rather than an implementation error in either
solver or in this export/import adapter itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .back_bombardment_events import _git_commit_hash
from .cathode_geometry import SURFACE_LABELS
from .io import to_json_safe

#: Schema version for the `comsol_source/` export adapter (plan Sec. 4.2:
#: `comsol_source/*.csv: adapter output with explicit columns/units ...`). Bump only when a field
#: is added/removed/reinterpreted in a way that would break a reader written against an earlier
#: version -- same convention as `BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION`/`RUN_CONFIG_SCHEMA_VERSION`.
COMSOL_SOURCE_SCHEMA_VERSION = "comsol_source_v1"

#: Schema version for the COMSOL thermal-result import contract (plan Sec. 7's "documented CSV/
#: HDF5 contract"). No real COMSOL output has ever been written against this contract yet
#: (addendum Sec. 19.2) -- it is tested only against this project's own synthetic stand-in files
#: (`tests/test_comsol_io.py`'s `write_synthetic_comsol_result_for_testing`).
COMSOL_THERMAL_RESULT_SCHEMA_VERSION = "comsol_thermal_result_v1"

#: Default RF frequency [Hz] used for the "representative-period snapshot" `q_BB_W_m3` power-
#: density conversion (see module docstring, Sec. 1) -- this project's standard S-band frequency,
#: matching the value already used as an explicit default throughout
#: `rf_gun/back_bombardment_events.py`'s own test fixtures and provenance examples (2.856 GHz).
#: Overridable via `export_comsol_heat_source`'s `rf_frequency_Hz` keyword; passing `None` there
#: disables the power-density conversion entirely rather than silently assuming this value.
DEFAULT_RF_FREQUENCY_HZ = 2.856e9

__all__ = [
    "COMSOL_SOURCE_SCHEMA_VERSION",
    "COMSOL_THERMAL_RESULT_SCHEMA_VERSION",
    "DEFAULT_RF_FREQUENCY_HZ",
    "ComsolThermalResult",
    "ComsolComparison",
    "export_comsol_heat_source",
    "load_comsol_thermal_result",
    "compare_python_comsol_thermal",
]


# ------------------------------------------------------------------------------------------------
# 1. Export: Python heat source -> COMSOL adapter files
# ------------------------------------------------------------------------------------------------


def export_comsol_heat_source(
    directory: str | Path,
    heat_source: "Any",
    geometry: "Any",
    material: "Any",
    *,
    extra_metadata: dict[str, Any] | None = None,
    rf_frequency_Hz: float | None = DEFAULT_RF_FREQUENCY_HZ,
    q_layer_W_m3_override: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write `<directory>/comsol_source/{comsol_heat_source.csv, comsol_source_manifest.json}`
    from an already-built `rf_gun.back_bombardment_deposition.BackBombardmentHeatSource`,
    `rf_gun.cathode_geometry.CathodeGeometry`, and `rf_gun.materials.base.CathodeMaterialSet`.

    See this module's docstring (Sec. 1) for the exact column layout, coordinate convention, and
    power-density conversion this writes. Returns `{"csv": csv_path, "manifest": manifest_path}`.

    `rf_frequency_Hz`/`q_layer_W_m3_override` control how (or whether) `q_BB_W_m3` is populated --
    see the module docstring's "Power-density conversion" paragraph; passing both is an error
    (ambiguous which conversion the caller actually wants).
    """
    if rf_frequency_Hz is not None and q_layer_W_m3_override is not None:
        raise ValueError(
            "export_comsol_heat_source: pass at most one of rf_frequency_Hz, "
            "q_layer_W_m3_override (ambiguous which q_BB_W_m3 conversion is wanted); see the "
            "module docstring's 'Power-density conversion' paragraph."
        )

    q_layer_J = np.asarray(heat_source.q_layer_J, dtype=float)
    mask = np.asarray(heat_source.cathode_footprint_mask, dtype=bool)
    n_layers = q_layer_J.shape[2]
    layer_boundaries_um = np.asarray(heat_source.layer_boundaries_um, dtype=float)

    if q_layer_W_m3_override is not None:
        q_layer_W_m3_override = np.asarray(q_layer_W_m3_override, dtype=float)
        if q_layer_W_m3_override.shape != q_layer_J.shape:
            raise ValueError(
                f"q_layer_W_m3_override shape {q_layer_W_m3_override.shape} does not match "
                f"heat_source.q_layer_J shape {q_layer_J.shape}"
            )
        power_mode = "time_resolved_override"
    elif rf_frequency_Hz is not None:
        power_mode = "representative_period_snapshot"
    else:
        power_mode = "energy_density_only"

    total_deposited_J = float(heat_source.total_deposited_energy_J)
    total_incident_J = float(heat_source.total_incident_energy_J)
    total_escaping_J = float(
        heat_source.escaping_energy_geometric_J_total
        + heat_source.escaping_energy_below_tio_validity_J_total
    )

    ix_idx, iy_idx = np.nonzero(mask)
    n_lateral_cells = ix_idx.size
    n_rows = n_lateral_cells * n_layers
    uniform_frac = 1.0 / n_rows if n_rows > 0 else 0.0

    rows: list[tuple[Any, ...]] = []
    for ix, iy in zip(ix_idx.tolist(), iy_idx.tolist()):
        x_m = float(heat_source.x_centers_m[ix])
        y_m = float(heat_source.y_centers_m[iy])
        r_mm = float(np.hypot(x_m, y_m) * 1.0e3)
        surface_code = int(geometry.classify_surface_by_radius(np.array([r_mm]))[0])
        zone_label = SURFACE_LABELS.get(surface_code, "unknown")
        for ell in range(n_layers):
            deposited_J = float(q_layer_J[ix, iy, ell])
            frac = (deposited_J / total_deposited_J) if total_deposited_J > 0.0 else uniform_frac
            incident_J_row = total_incident_J * frac
            escaping_J_row = total_escaping_J * frac

            z_lo_m = -float(layer_boundaries_um[ell]) * 1.0e-6
            z_hi_m = -float(layer_boundaries_um[ell + 1]) * 1.0e-6
            z_m = 0.5 * (z_lo_m + z_hi_m)

            if power_mode == "time_resolved_override":
                q_W_m3 = float(q_layer_W_m3_override[ix, iy, ell])
            elif power_mode == "representative_period_snapshot":
                cell_vol_m3 = float(heat_source.cell_volume_m3(ell))
                q_W_m3 = (deposited_J * float(rf_frequency_Hz) / cell_vol_m3) if cell_vol_m3 > 0.0 else 0.0
            else:
                q_W_m3 = float("nan")

            rows.append(
                (
                    x_m, y_m, z_m, z_lo_m, z_hi_m, ell, surface_code, zone_label,
                    q_W_m3, deposited_J, incident_J_row, escaping_J_row,
                )
            )

    out_dir = Path(directory) / "comsol_source"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comsol_heat_source.csv"
    manifest_path = out_dir / "comsol_source_manifest.json"

    import csv

    header = [
        "x_m", "y_m", "z_m", "z_lo_m", "z_hi_m", "layer_index", "surface_code", "zone_label",
        "q_BB_W_m3", "deposited_J", "incident_energy_attributed_J", "escaping_energy_attributed_J",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    manifest = {
        "schema_version": COMSOL_SOURCE_SCHEMA_VERSION,
        "git_commit": _git_commit_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": (
            "Origin (0,0,0) is the CENTER of the flat emitting face, AT the cathode emission "
            "surface. x,y are lateral Cartesian coordinates in the cathode's own frame "
            "(insertion_offset_mm does not shift this frame -- bookkeeping metadata only, see "
            "CathodeGeometry's own docstring)."
        ),
        "normal_convention": (
            "z=0 is the emission surface; the LaB6 solid occupies z<=0 (increasing depth into "
            "the cathode is increasingly NEGATIVE z, i.e. 'into the cathode' is the -z "
            "direction); vacuum/beam channel occupies z>=0. Identical to "
            "rf_gun.cathode_geometry/rf_gun.back_bombardment_events."
        ),
        "csv_columns": {
            "x_m": "lateral cell-center x [m]",
            "y_m": "lateral cell-center y [m]",
            "z_m": "depth-layer midpoint z [m] (negative, into the solid)",
            "z_lo_m": "depth-layer near (shallow) boundary z [m]",
            "z_hi_m": "depth-layer far (deep) boundary z [m]",
            "layer_index": "0-based index into layer_boundaries_um below",
            "surface_code": "rf_gun.cathode_geometry surface-zone numeric code (radius-only; "
                "never 'holder' here, see holder_note below)",
            "zone_label": "human-readable surface-zone label",
            "q_BB_W_m3": "volumetric heating-power density; see power_density_conversion below "
                "for exact meaning -- NOT a macropulse time history",
            "deposited_J": "exact per-cell deposited energy for the one representative RF period "
                "heat_source represents (heat_source.q_layer_J, reused verbatim)",
            "incident_energy_attributed_J": "bookkeeping apportionment of "
                "total_incident_energy_J proportional to this row's deposited_J share -- NOT a "
                "spatially resolved incident-energy model, see per_row_attribution_note",
            "escaping_energy_attributed_J": "bookkeeping apportionment of the two aggregate "
                "escape totals, same convention as incident_energy_attributed_J",
        },
        "power_density_conversion": {
            "mode": power_mode,
            "rf_frequency_Hz": float(rf_frequency_Hz) if power_mode == "representative_period_snapshot" else None,
            "explanation": (
                "q_BB_W_m3 = deposited_J * rf_frequency_Hz / cell_volume_m3 (plan Sec. 8.1's "
                "P_dep,n = f_RF * sum(E_dep,i,n), applied per cell): the instantaneous volumetric "
                "power density IF this one representative-period source repeated unchanged at "
                "rf_frequency_Hz forever. This is a REPRESENTATIVE-PERIOD SNAPSHOT pending the "
                "real macropulse time axis from Work Package 3/4 -- it is NOT yet an 8us-"
                "integrated or genuinely time-varying macropulse source."
                if power_mode == "representative_period_snapshot"
                else (
                    "q_BB_W_m3 was supplied by the caller as an already time-resolved array "
                    "(q_layer_W_m3_override), exported verbatim, not derived from rf_frequency_Hz."
                    if power_mode == "time_resolved_override"
                    else (
                        "No rf_frequency_Hz or time-resolved override was supplied: q_BB_W_m3 is "
                        "NaN for every row. Use deposited_J (an energy, not a power density) "
                        "instead, or re-export with rf_frequency_Hz/q_layer_W_m3_override set."
                    )
                )
            ),
        },
        "per_row_attribution_note": (
            "BB0 tracks per-cell DEPOSITED energy exactly, but total incident energy and the two "
            "escape totals are aggregate scalars (a single event's incident energy spreads across "
            "several depth layers along its own CSDA path -- see "
            "rf_gun.back_bombardment_deposition's module docstring). This file apportions each "
            "aggregate total across rows in proportion to that row's deposited_J share of "
            "total_deposited_energy_J purely so file-level column sums reproduce heat_source's "
            "own recorded totals exactly; it is bookkeeping, not a spatially resolved model."
        ),
        "holder_note": (
            "No CSV row is ever labeled 'holder': heat_source.cathode_footprint_mask never "
            "extends past bevel_outer_radius_mm (BB0 excludes non-LaB6 events entirely). The "
            "holder's aggregate incident energy is recorded below as "
            "excluded_non_lab6_energy_J_total but is NOT part of any CSV row or its closure sums "
            "-- plan Sec. 7's flat/bevel/holder separation is satisfied for holder only at this "
            "metadata level, since BB0 has no spatial holder-deposition model to export."
        ),
        "energy_totals_J": {
            "total_incident_energy_J": total_incident_J,
            "total_deposited_energy_J": total_deposited_J,
            "escaping_energy_geometric_J_total": float(heat_source.escaping_energy_geometric_J_total),
            "escaping_energy_below_tio_validity_J_total": float(
                heat_source.escaping_energy_below_tio_validity_J_total
            ),
            "excluded_non_lab6_energy_J_total": float(heat_source.excluded_non_lab6_energy_J_total),
        },
        "deposition_model": heat_source.model,
        "density_used_kg_m3": float(heat_source.density_used_kg_m3),
        "layer_boundaries_um": layer_boundaries_um.tolist(),
        "xy_cell_area_m2": float(heat_source.xy_cell_area_m2),
        "state_ids": np.asarray(heat_source.state_ids).tolist(),
        "geometry": {
            "flat_radius_mm": float(geometry.flat_radius_mm),
            "bevel_width_mm": float(geometry.bevel_width_mm),
            "bevel_angle_deg": float(geometry.bevel_angle_deg),
            "cathode_length_mm": float(geometry.cathode_length_mm),
            "insertion_offset_mm": float(geometry.insertion_offset_mm),
            "holder_outer_radius_mm": float(geometry.holder_outer_radius_mm),
            "bevel_outer_radius_mm": float(geometry.bevel_outer_radius_mm),
            "bevel_true_area_mm2": float(geometry.bevel_true_area_mm2),
        },
        "material": material.to_manifest_dict(),
        "rf_envelope_keyframe_placeholders": {
            "note": (
                "No macropulse/keyframe system is wired in yet (addendum Sec. 19.2, Work "
                "Package 5 is interface-only). The fields under 'values' are forward-compatible "
                "PLACEHOLDERS for the RF envelope/keyframe-interpolation metadata plan Sec. 7 "
                "specifies COMSOL will eventually need -- populated ONLY from caller-supplied "
                "extra_metadata, never fabricated by this function."
            ),
            "values": to_json_safe(extra_metadata) if extra_metadata else {},
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(manifest), f, indent=2, sort_keys=False)

    return {"csv": csv_path, "manifest": manifest_path}


# ------------------------------------------------------------------------------------------------
# 2. Import: ComsolThermalResult / load_comsol_thermal_result
# ------------------------------------------------------------------------------------------------


def _decode_attr_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class ComsolThermalResult:
    """In-memory form of the `comsol_thermal_result_v1` HDF5 contract (module docstring Sec. 2;
    plan Sec. 7's "documented CSV/HDF5 contract" for COMSOL result import).

    `time_s` and every scalar time-history field are shape `(n_t,)`. `T_surface_full_K` (optional,
    `(n_x, n_y, n_t)`) requires `x_m`/`y_m` (`(n_x,)`/`(n_y,)`) alongside it -- both `None` together
    if no full field was supplied.
    """

    time_s: np.ndarray
    T_center_K: np.ndarray
    T_flat_edge_K: np.ndarray
    T_bevel_K: np.ndarray
    T_max_K: np.ndarray
    T_area_average_K: np.ndarray
    stored_energy_J: np.ndarray
    radiation_loss_W: np.ndarray
    boundary_heat_flow_W: np.ndarray
    mesh_id: str
    time_step_id: str
    convergence_metrics: dict[str, Any] = field(default_factory=dict)
    T_surface_full_K: np.ndarray | None = None
    x_m: np.ndarray | None = None
    y_m: np.ndarray | None = None
    schema_version: str = COMSOL_THERMAL_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", np.asarray(self.time_s, dtype=float))
        n_t = self.time_s.shape[0]
        for name in (
            "T_center_K", "T_flat_edge_K", "T_bevel_K", "T_max_K", "T_area_average_K",
            "stored_energy_J", "radiation_loss_W", "boundary_heat_flow_W",
        ):
            arr = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, arr)
            if arr.shape != (n_t,):
                raise ValueError(
                    f"ComsolThermalResult.{name} has shape {arr.shape}, expected ({n_t},) to "
                    f"match time_s"
                )
        has_field = self.T_surface_full_K is not None
        has_grid = self.x_m is not None or self.y_m is not None
        if has_field != has_grid or (has_field and (self.x_m is None or self.y_m is None)):
            raise ValueError(
                "ComsolThermalResult: T_surface_full_K and x_m/y_m must be supplied together or "
                "not at all (got T_surface_full_K="
                f"{'set' if has_field else 'None'}, x_m={'set' if self.x_m is not None else 'None'}, "
                f"y_m={'set' if self.y_m is not None else 'None'})"
            )
        if has_field:
            x_m = np.asarray(self.x_m, dtype=float)
            y_m = np.asarray(self.y_m, dtype=float)
            T_full = np.asarray(self.T_surface_full_K, dtype=float)
            object.__setattr__(self, "x_m", x_m)
            object.__setattr__(self, "y_m", y_m)
            object.__setattr__(self, "T_surface_full_K", T_full)
            expected = (x_m.shape[0], y_m.shape[0], n_t)
            if T_full.shape != expected:
                raise ValueError(
                    f"ComsolThermalResult.T_surface_full_K has shape {T_full.shape}, expected "
                    f"{expected} = (len(x_m), len(y_m), len(time_s))"
                )


def load_comsol_thermal_result(path: str | Path) -> ComsolThermalResult:
    """Strict reader for `comsol_thermal_result_v1` HDF5 files (module docstring Sec. 2).

    Rejects, with a specific actionable `ValueError` (never a generic `KeyError`), a file with no
    `schema_version` root attribute at all, or one that does not exactly equal
    `COMSOL_THERMAL_RESULT_SCHEMA_VERSION` -- matching
    `rf_gun.back_bombardment_events.read_back_bombardment_events_h5`'s precedent exactly (plan
    Sec. 2.3/4.2's "the loader checks the exact schema major version and fails clearly on older
    files").
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "h5py is required to read comsol_thermal_result_v1 HDF5 files. Install it with "
            "'pip install h5py'."
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: file not found.")

    with h5py.File(str(path), "r") as h5f:
        if "schema_version" not in h5f.attrs:
            raise ValueError(
                f"{path}: no 'schema_version' root attribute found. This file is NOT a "
                f"supported {COMSOL_THERMAL_RESULT_SCHEMA_VERSION!r} input (module "
                "rf_gun.comsol_io's documented import contract) -- regenerate it with a writer "
                "that follows that contract (e.g. tests/test_comsol_io.py's "
                "write_synthetic_comsol_result_for_testing for a test stand-in) before loading."
            )
        schema_version = _decode_attr_str(h5f.attrs["schema_version"])
        if schema_version != COMSOL_THERMAL_RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version={schema_version!r} does not match the required "
                f"{COMSOL_THERMAL_RESULT_SCHEMA_VERSION!r}. Only files following "
                "rf_gun.comsol_io's documented comsol_thermal_result_v1 contract are supported; "
                "regenerate this file with the current schema."
            )
        if "thermal" not in h5f:
            raise ValueError(
                f"{path}: schema_version matches but the required '/thermal' group is missing "
                "-- this file is corrupt or was not written by this contract's writer."
            )

        thermal_grp = h5f["thermal"]
        required = [
            "time_s", "T_center_K", "T_flat_edge_K", "T_bevel_K", "T_max_K", "T_area_average_K",
            "stored_energy_J", "radiation_loss_W", "boundary_heat_flow_W",
        ]
        arrays: dict[str, np.ndarray] = {}
        for name in required:
            if name not in thermal_grp:
                raise ValueError(f"{path}: missing required dataset /thermal/{name}.")
            arrays[name] = np.asarray(thermal_grp[name][()], dtype=float)

        for attr_name in ("mesh_id", "time_step_id"):
            if attr_name not in h5f.attrs:
                raise ValueError(f"{path}: missing required root attribute {attr_name!r}.")
        mesh_id = _decode_attr_str(h5f.attrs["mesh_id"])
        time_step_id = _decode_attr_str(h5f.attrs["time_step_id"])

        convergence_metrics: dict[str, Any] = {}
        if "convergence" in h5f and "json" in h5f["convergence"].attrs:
            convergence_metrics = json.loads(h5f["convergence"].attrs["json"])

        T_surface_full_K = x_m = y_m = None
        if "surface_field" in h5f:
            sf = h5f["surface_field"]
            for name in ("x_m", "y_m", "T_surface_full_K"):
                if name not in sf:
                    raise ValueError(
                        f"{path}: '/surface_field' group is present but missing dataset {name}."
                    )
            x_m = np.asarray(sf["x_m"][()], dtype=float)
            y_m = np.asarray(sf["y_m"][()], dtype=float)
            T_surface_full_K = np.asarray(sf["T_surface_full_K"][()], dtype=float)

        return ComsolThermalResult(
            time_s=arrays["time_s"],
            T_center_K=arrays["T_center_K"],
            T_flat_edge_K=arrays["T_flat_edge_K"],
            T_bevel_K=arrays["T_bevel_K"],
            T_max_K=arrays["T_max_K"],
            T_area_average_K=arrays["T_area_average_K"],
            stored_energy_J=arrays["stored_energy_J"],
            radiation_loss_W=arrays["radiation_loss_W"],
            boundary_heat_flow_W=arrays["boundary_heat_flow_W"],
            mesh_id=mesh_id,
            time_step_id=time_step_id,
            convergence_metrics=convergence_metrics,
            T_surface_full_K=T_surface_full_K,
            x_m=x_m,
            y_m=y_m,
            schema_version=schema_version,
        )


# ------------------------------------------------------------------------------------------------
# 3. Comparison: compare_python_comsol_thermal / ComsolComparison
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComsolComparison:
    """Result of `compare_python_comsol_thermal` (module docstring Sec. 3).

    `comsol_available=False` (COMSOL result was `None`) means every field below is `None` except
    `notes` -- see `compare_python_comsol_thermal`'s docstring; this is the single most important
    behavioral contract in this module (plan Sec. 8.2/12, addendum Sec. 19.2).
    """

    comsol_available: bool
    aligned_time_s: np.ndarray | None = None
    T_center_diff_K: np.ndarray | None = None
    T_max_diff_K: np.ndarray | None = None
    T_area_average_diff_K: np.ndarray | None = None
    max_abs_temperature_diff_K: float | None = None
    mean_abs_temperature_diff_K: float | None = None
    surface_diff_map_K: np.ndarray | None = None
    surface_diff_norm_K: float | None = None
    hotspot_displacement_m: float | None = None
    notes: str = ""


def compare_python_comsol_thermal(
    python_result: Any, comsol_result: ComsolThermalResult | None
) -> ComsolComparison:
    """Align and compare a Python thermal result against an optional COMSOL result (module
    docstring Sec. 3).

    `comsol_result is None` -> returns `ComsolComparison(comsol_available=False, ...)` with every
    COMSOL-side field `None` -- NEVER fabricates, interpolates, or defaults in placeholder COMSOL
    data (plan Sec. 8.2/12; addendum Sec. 19.2's explicit user decision). This is the single most
    important behavior in this function; `tests/test_comsol_io.py` tests it explicitly.

    `python_result` is a loosely-typed duck-typed object -- see the module docstring's "Comparison"
    section for the exact (best-guess) attribute names read, all optional except `t_grid_s`.
    """
    if comsol_result is None:
        return ComsolComparison(
            comsol_available=False,
            notes=(
                "COMSOL result unavailable (comsol_result=None): no COMSOL run exists yet "
                "(implementation plan addendum Sec. 19.2, Work Package 5 is an interface-only "
                "pass). All comsol-side fields are None; this function never fabricates, "
                "interpolates, or defaults in placeholder COMSOL data (plan Sec. 8.2/12)."
            ),
        )
    if not isinstance(comsol_result, ComsolThermalResult):
        raise TypeError(
            f"compare_python_comsol_thermal: comsol_result must be a ComsolThermalResult or "
            f"None, got {type(comsol_result)!r}"
        )

    t_py_raw = getattr(python_result, "t_grid_s", None)
    if t_py_raw is None:
        raise ValueError(
            "compare_python_comsol_thermal: python_result must supply a 't_grid_s' attribute "
            "(macro-time grid, seconds) -- see this module's docstring (Sec. 3) for the minimal "
            "duck-typed protocol this function needs."
        )
    t_py = np.asarray(t_py_raw, dtype=float)
    t_comsol = np.asarray(comsol_result.time_s, dtype=float)

    notes_parts: list[str] = []

    lo = max(float(np.min(t_py)), float(np.min(t_comsol)))
    hi = min(float(np.max(t_py)), float(np.max(t_comsol)))
    if not (hi > lo):
        return ComsolComparison(
            comsol_available=True,
            notes=(
                f"Python time grid [{np.min(t_py):.6g}, {np.max(t_py):.6g}] s and COMSOL time "
                f"grid [{np.min(t_comsol):.6g}, {np.max(t_comsol):.6g}] s do not overlap; no "
                "difference metrics computed."
            ),
        )

    t_common = t_py[(t_py >= lo) & (t_py <= hi)]
    if t_common.size < 2:
        t_common = np.linspace(lo, hi, 2)

    def _interp_py(name: str) -> np.ndarray | None:
        arr = getattr(python_result, name, None)
        if arr is None:
            return None
        return np.interp(t_common, t_py, np.asarray(arr, dtype=float))

    def _interp_comsol(arr: np.ndarray) -> np.ndarray:
        return np.interp(t_common, t_comsol, np.asarray(arr, dtype=float))

    diffs: dict[str, np.ndarray] = {}
    for py_name, comsol_arr, diff_key in (
        ("T_center_K", comsol_result.T_center_K, "T_center_diff_K"),
        ("T_max_K", comsol_result.T_max_K, "T_max_diff_K"),
        ("T_area_average_K", comsol_result.T_area_average_K, "T_area_average_diff_K"),
    ):
        py_series = _interp_py(py_name)
        if py_series is None:
            notes_parts.append(f"python_result has no '{py_name}' attribute; {diff_key} skipped.")
            continue
        diffs[diff_key] = py_series - _interp_comsol(comsol_arr)

    all_diff_values = list(diffs.values())
    max_abs = float(max(np.max(np.abs(d)) for d in all_diff_values)) if all_diff_values else None
    mean_abs = (
        float(np.mean(np.concatenate([np.abs(d) for d in all_diff_values])))
        if all_diff_values
        else None
    )

    surface_diff_map_K: np.ndarray | None = None
    surface_diff_norm_K: float | None = None
    hotspot_displacement_m: float | None = None

    py_T_surface = getattr(python_result, "T_surface_K", None)
    py_x = getattr(python_result, "x_centers_m", None)
    py_y = getattr(python_result, "y_centers_m", None)
    if py_T_surface is None or py_x is None or py_y is None:
        notes_parts.append(
            "python_result has no full T_surface_K/x_centers_m/y_centers_m field; surface "
            "difference map and hotspot displacement skipped."
        )
    elif comsol_result.T_surface_full_K is None:
        notes_parts.append(
            "COMSOL result has no T_surface_full_K field; surface difference map and hotspot "
            "displacement skipped."
        )
    else:
        py_x = np.asarray(py_x, dtype=float)
        py_y = np.asarray(py_y, dtype=float)
        py_T_surface = np.asarray(py_T_surface, dtype=float)
        if py_x.shape == comsol_result.x_m.shape and np.allclose(py_x, comsol_result.x_m) and (
            py_y.shape == comsol_result.y_m.shape and np.allclose(py_y, comsol_result.y_m)
        ):
            i_py = int(np.argmin(np.abs(t_py - hi)))
            i_co = int(np.argmin(np.abs(t_comsol - hi)))
            map_py = py_T_surface[:, :, i_py]
            map_co = comsol_result.T_surface_full_K[:, :, i_co]
            diff_map = map_py - map_co
            surface_diff_map_K = diff_map
            surface_diff_norm_K = float(np.sqrt(np.nanmean(diff_map**2)))

            ix_py, iy_py = np.unravel_index(int(np.nanargmax(map_py)), map_py.shape)
            ix_co, iy_co = np.unravel_index(int(np.nanargmax(map_co)), map_co.shape)
            x_py_hot, y_py_hot = float(py_x[ix_py]), float(py_y[iy_py])
            x_co_hot, y_co_hot = float(comsol_result.x_m[ix_co]), float(comsol_result.y_m[iy_co])
            hotspot_displacement_m = float(np.hypot(x_py_hot - x_co_hot, y_py_hot - y_co_hot))
        else:
            notes_parts.append(
                "Python and COMSOL lateral grids (x_centers_m/y_centers_m vs. x_m/y_m) differ; "
                "surface difference map and hotspot displacement skipped (this function never "
                "silently regrids/azimuthally averages COMSOL data onto a mismatched grid -- "
                "plan Sec. 7's 'without azimuthal averaging' requirement)."
            )

    return ComsolComparison(
        comsol_available=True,
        aligned_time_s=t_common,
        T_center_diff_K=diffs.get("T_center_diff_K"),
        T_max_diff_K=diffs.get("T_max_diff_K"),
        T_area_average_diff_K=diffs.get("T_area_average_diff_K"),
        max_abs_temperature_diff_K=max_abs,
        mean_abs_temperature_diff_K=mean_abs,
        surface_diff_map_K=surface_diff_map_K,
        surface_diff_norm_K=surface_diff_norm_K,
        hotspot_displacement_m=hotspot_displacement_m,
        notes="; ".join(notes_parts),
    )
