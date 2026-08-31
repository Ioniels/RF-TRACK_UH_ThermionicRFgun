"""BB0 (TIO/CSDA baseline) electron energy-deposition model (implementation plan Sec. 3.4, 4.2,
6.2, 10.1, 10.2; addendum Sec. 19.2/19.3) -- Work Package 2.

Scope: pure Python/numpy, NO RF-Track dependency. Consumes an already-populated
`rf_gun.back_bombardment_events.BackBombardmentEvents` object (real or synthetic) plus a
`rf_gun.materials.base.CathodeMaterialSet`, and produces a depth-resolved, laterally-binned
volumetric energy-deposition source -- one Cartesian `(x, y, layer)` tensor of deposited energy
[J] for the single representative RF period the events represent (`events.state_id`). Converting
this per-period energy tensor into a macropulse-integrated power density `q'''(x,y,z,t)` (plan
Sec. 6.2's `q''_layer(x,y,ell,t)`) is explicitly Work Package 3's job (the thermal solver,
`rf_gun/thermal.py`) -- this module only supplies the raw per-period deposited-energy tensor plus
the geometric normalization (lateral cell area, per-layer thickness) needed to divide it into a
volumetric density later.

Physics summary (plan Sec. 3.4 "BB0 -- TIO/CSDA baseline"):
  * Each LaB6-heating event's incoming path direction inside the solid continues straight along
    its own incidence direction (not assumed normal) -- `cos_incidence`/`incidence_angle_rad`
    already carry this (plan Sec. 3.1/4.1, produced by `CathodeGeometry.intersect_ray`).
  * `R_i = tio_range_um(K_i_keV, Z_eff, A_eff, rho0)` (this project's own TIO implementation,
    `rf_gun.materials.electron_range.tio_range_um` -- see that module's docstring for the
    Tabata/Bakr a1-coefficient discrepancy this project resolves) is the total PATH-LENGTH range at
    the material's reference density (`ElectronDepositionComponent.rho0_kg_m3`, Bakr's
    `rho0=4720 kg/m3`) -- the BB0 baseline density, NOT a temperature-corrected one (plan Sec. 3.4;
    the `R(E,T)=R(E,T0)*rho(T0)/rho(T)` correction of Sec. 5.3 is a later production refinement).
    `density_override_kg_m3` is the named hook for a future caller wanting that correction: pass
    the density at the temperature of interest and this module uses it in place of `rho0` for
    every event, unchanged otherwise.
  * The continuous-slowing-down-approximation (CSDA) residual-range method turns `R_i` into a
    depth-dose profile: at path distance `s` (`0<=s<=R_i`) the residual range is `R_i-s`, inverted
    to a residual kinetic energy `K_res(s)` via a once-per-call PCHIP-free linear lookup table
    (`np.interp` on a dense, verified-monotonic `R(E)` grid -- see `_ResidualEnergyLookup` below;
    this is the "fast lookup/interpolation table" option plan Sec. 3.4/BB0's instructions offer as
    an alternative to a per-sample-point `scipy.optimize.brentq` root-find, chosen here purely for
    speed with many events -- both are declared valid by the plan).
  * The energy deposited between path positions `s` and `s+ds` is `K_res(s)-K_res(s+ds)`. Binning
    this onto the plan Sec. 6.2 depth-layer grid (`DEFAULT_DEPTH_LAYER_BOUNDARIES_UM`) is done by
    INTEGRATING (not point-sampling) the exact CSDA energy loss between each layer's path-length
    bounds, using the identity that adjacent layers' shared path-length boundary makes the
    per-layer differences telescope exactly (see `_deposit_one_event`'s docstring for the closure
    argument this buys).
  * The critical, easy-to-invert-backwards projection (plan's own explicit warning): a path
    element at distance `s` along the electron's own direction of travel corresponds to NORMAL
    depth (perpendicular to the surface, the axis the depth-layer grid uses) `z_local = s *
    cos(incidence_angle_rad)`. A grazing hit (`cos(incidence_angle)` small) therefore spreads the
    SAME path-length range over LESS normal depth than a normal-incidence hit at the same energy --
    i.e. the oblique event's deposition is MORE concentrated near the surface in normal-depth
    terms, not less. `tests/test_back_bombardment_deposition.py` tests this direction explicitly
    (do not "fix" this back to `z_local = s / cos(incidence_angle)` without re-reading this
    paragraph and that test).

Two named escape/validity mechanisms (plan Sec. 3.4/5.3, "must be tracked separately and by
name so the two 'does not close' reasons are never conflated in diagnostics"):
  * `escaping_energy_geometric_J_total`: energy corresponding to path length beyond the modeled
    depth limit -- `min(layer_boundaries_um[-1], geometry.cathode_length_mm*1000)` in normal depth
    (the smaller of the layer grid's own deepest boundary and the actual physical LaB6 thickness --
    an electron cannot deposit energy past the real solid, regardless of how deep the layer grid
    happens to extend) -- is genuinely NOT deposited into any layer; booked here.
  * `escaping_energy_below_tio_validity_J_total`: plan Sec. 5.3's "BB0 deposits the residual energy
    below the approximately 0.3 keV TIO validity limit in the terminal deposition cell" -- once the
    CSDA inversion's residual energy would drop below `tio_validity_floor_keV` (default 0.3 keV),
    this module stops inverting and dumps the ENTIRE remaining residual energy into whatever layer
    the CSDA tracking had reached at that point (the "current terminal layer"). Because that energy
    IS deposited (not discarded), this named total is, BY THIS MODULE'S OWN DESIGN, always exactly
    0.0 -- it exists as a distinctly-named diagnostic placeholder in the closure equation precisely
    so a reader can see that BB0 chose "deposit, don't discard" for the sub-floor tail, and so a
    future implementation that chose differently (e.g. genuinely discarding it) would have a
    natural place to report a nonzero number without conflating it with (a). See
    `_deposit_one_event` for exactly where this branch is taken.

Energy closure (plan Sec. 3.4/15.2): `sum(deposited_J) + escaping_geometric_J_total +
escaping_below_tio_validity_J_total == sum(incident_energy_J for heats_lab6 events)`. This
module's specific numerical design (see `_deposit_one_event`) makes this an EXACT telescoping
identity in the per-electron kinetic-energy bookkeeping (`K_res(0)` is anchored to the event's own
exact `kinetic_energy_eV`, not read back off the interpolation table, so table-resolution error
cancels out of the total and only affects how energy is split *between* layers/escape buckets).
The only sources of closure error are then genuine floating-point round-off accumulated over
`n_layers` subtractions per event (order 1e-14 relative) -- ACHIEVED closure tolerance in this
implementation is therefore far tighter than plan Sec. 15.2's "provisional 1e-6 relative
tolerance" target; `tests/test_back_bombardment_deposition.py` checks it at `rtol=1e-9`
(deliberately tighter than the plan's own provisional target, honestly reflecting what this
specific numerical scheme actually achieves -- see that test for the exact number observed).

Holder/unknown-surface events (`~events.heats_lab6_mask`) are never deposited -- their incident
energy is tracked separately in `excluded_non_lab6_energy_J_total` (plan's "excluded, not
deposited" accounting bucket) and is NOT part of the closure sum above.

BB1 (uncertainty scan)/BB2 (Geant4 response library) are out of scope for this pass (plan addendum
Sec. 19.2, Work Package 2 vs. later work) -- `_build_bb1_uncertainty_heat_source`/
`_build_bb2_response_library_heat_source` are documented `NotImplementedError` stubs with the
correct signature, dispatched from `build_back_bombardment_heat_source` via
`deposition_config.model`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import-order concerns at module load
    from .back_bombardment_events import BackBombardmentEvents
    from .back_bombardment_study_config import DepositionConfig
    from .cathode_geometry import CathodeGeometry
    from .materials.base import CathodeMaterialSet

#: Provisional depth-layer boundaries in normal depth [um] (plan Sec. 6.2: "a provisional
#: geometric sequence such as [0,1,3,10,30,100,300,1000] um is suitable for development"). A
#: module-level constant, override-able per call via `build_back_bombardment_heat_source`'s
#: `layer_boundaries_um` keyword -- layer count/boundaries are convergence parameters (plan
#: Sec. 6.2), not fixed physics.
DEFAULT_DEPTH_LAYER_BOUNDARIES_UM = np.array([0, 1, 3, 10, 30, 100, 300, 1000], dtype=float)

#: TIO validity floor [keV] (plan Sec. 5.3: "the approximately 0.3 keV TIO validity limit"). Below
#: this residual energy the CSDA inversion is not trusted; the entire remaining residual energy is
#: dumped into the current terminal layer instead of being tracked further (see module docstring's
#: `escaping_energy_below_tio_validity_J_total` paragraph).
DEFAULT_TIO_VALIDITY_FLOOR_KEV = 0.3

#: Number of log-spaced energy points in the CSDA residual-range lookup table built once per
#: `build_back_bombardment_heat_source` call (shared across all events at that material/density).
#: Chosen generously fine (this is a cheap one-time cost, `tio_range_um` is a closed-form
#: evaluation, not a simulation) so that per-layer *distribution* error stays small; recall from
#: the module docstring that this resolution does NOT affect the total energy closure, only how a
#: single event's energy is split between layers/escape buckets.
DEFAULT_CSDA_LOOKUP_POINTS = 4000

__all__ = [
    "DEFAULT_DEPTH_LAYER_BOUNDARIES_UM",
    "DEFAULT_TIO_VALIDITY_FLOOR_KEV",
    "DEFAULT_CSDA_LOOKUP_POINTS",
    "BackBombardmentHeatSource",
    "build_back_bombardment_heat_source",
    "validate_energy_closure",
    "BACK_BOMBARDMENT_HEAT_SOURCE_SCHEMA_VERSION",
    "write_back_bombardment_heat_source_h5",
    "read_back_bombardment_heat_source_h5",
]


# ------------------------------------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BackBombardmentHeatSource:
    """Depth-resolved, laterally-binned BB0 deposition source for one representative RF period
    (plan Sec. 6.2's `q''_layer(x,y,ell,t)` with `t` collapsed to one slice, matching
    `events.state_id`; plan Sec. 4.2's `back_bombardment_heat_source.h5` contract, in-memory form).

    `q_layer_J[ix, iy, ell]` is deposited energy [J] in lateral cell `(ix,iy)` and depth layer
    `ell`, for the ONE representative RF period `events` represents -- NOT yet a power density.
    Converting to `q'''` [W/m^3] or a macropulse-integrated power is Work Package 3's job (the
    thermal solver): `cell_volume_m3(ell) = xy_cell_area_m2 * layer_thickness_m[ell]` (for cells
    inside `cathode_footprint_mask`) is exactly the geometric factor that later division needs.

    `total_incident_energy_J`/`total_deposited_energy_J` plus the two escape totals are cached
    here (computed once, at construction) precisely so `validate_energy_closure` can be a small,
    standalone, testable function that does not have to re-walk the event array itself.
    """

    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    layer_boundaries_um: np.ndarray
    q_layer_J: np.ndarray
    xy_cell_area_m2: float
    layer_thickness_m: np.ndarray
    cathode_footprint_mask: np.ndarray
    escaping_energy_geometric_J_total: float
    escaping_energy_below_tio_validity_J_total: float
    excluded_non_lab6_energy_J_total: float
    total_incident_energy_J: float
    total_deposited_energy_J: float
    model: str = "BB0_TIO"
    density_used_kg_m3: float = 0.0
    n_events_included: int = 0
    n_events_excluded: int = 0
    state_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    def cell_volume_m3(self, layer_index: int) -> float:
        """Convenience for Work Package 3: geometric volume [m^3] of one lateral cell in depth
        layer `layer_index` (the same value for every `(x,y)` cell -- the lateral grid is uniform;
        multiply by `cathode_footprint_mask` to zero cells outside the physical disk)."""
        return float(self.xy_cell_area_m2 * self.layer_thickness_m[layer_index])


def validate_energy_closure(
    source: BackBombardmentHeatSource, events: "BackBombardmentEvents", rtol: float = 1e-6
) -> None:
    """Standalone energy-closure check (plan Sec. 3.4/15.2): raises `ValueError` (naming the
    actual numbers) unless

        deposited + escaping_geometric + escaping_below_tio_validity
            ~= sum(incident_energy_J for heats_lab6 events)

    to relative tolerance `rtol`. Recomputes the incident-energy sum directly from `events` (the
    authoritative source, plan Sec. 1) rather than trusting `source.total_incident_energy_J` blindly,
    so a mismatch between the two also surfaces here rather than only inside `source`'s own cached
    number.
    """
    mask = events.heats_lab6_mask
    total_incident = float(np.sum(np.asarray(events.incident_energy_J)[mask]))
    total_deposited = float(np.sum(source.q_layer_J))
    total_escaping = (
        source.escaping_energy_geometric_J_total + source.escaping_energy_below_tio_validity_J_total
    )
    closure_sum = total_deposited + total_escaping
    denom = max(abs(total_incident), 1e-30)
    rel_err = abs(closure_sum - total_incident) / denom
    if rel_err > rtol:
        raise ValueError(
            "BB0 energy closure failed: "
            f"incident(heats_lab6)={total_incident:.9e} J, "
            f"deposited={total_deposited:.9e} J, "
            f"escaping_geometric={source.escaping_energy_geometric_J_total:.9e} J, "
            f"escaping_below_tio_validity={source.escaping_energy_below_tio_validity_J_total:.9e} J, "
            f"closure_sum={closure_sum:.9e} J, "
            f"relative_error={rel_err:.3e} (rtol={rtol:.1e}). "
            "See rf_gun.back_bombardment_deposition module docstring for the expected achieved "
            "tolerance and how the two escape totals are defined."
        )


# ------------------------------------------------------------------------------------------------
# HDF5 v1 writer/reader (plan Sec. 4.2's `back_bombardment_heat_source.h5` contract: "binned
# q'''(x,y,z,t_macro) for COMSOL and the corresponding depth-layer source q''_layer(x,y,layer,
# t_macro) for Python, with energy closure and source uncertainty"; Sec. 4.1's "the hash of the
# input event file" requirement, applied here as `source_events_hash`) -- Work Package 4's small
# missing data-contract piece: at the start of this pass NO writer for this file existed anywhere
# in the project (`BackBombardmentHeatSource`'s own docstring only documented the in-memory form).
#
# This in-memory object already collapses the macropulse time axis to the single representative RF
# period it represents (see the class docstring above) -- it is not itself a macropulse time
# series (that is `rf_gun.macropulse`'s `VolumetricHeatSourceTimeSeries`, downstream of this file
# and not written by this writer). Follows `rf_gun.back_bombardment_events.
# write_back_bombardment_events_h5`/`read_back_bombardment_events_h5`'s exact conventions: a
# `/deposition` group of plain HDF5 datasets (each numeric dataset carries a `units` attribute),
# scalar root attributes for the closure/escape totals and model identifier, and a strict
# `schema_version`-gated reader that rejects anything else with a specific, actionable `ValueError`
# (never a generic `KeyError` or a silent best-effort read) -- matching
# `read_back_bombardment_events_h5`/`rf_gun.comsol_io.load_comsol_thermal_result`'s precedent.
# ------------------------------------------------------------------------------------------------

#: Bump only when a field is added/removed/reinterpreted in a way that would break a reader
#: written against an earlier version -- same convention as `BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION`.
BACK_BOMBARDMENT_HEAT_SOURCE_SCHEMA_VERSION = "back_bombardment_heat_source_v1"

#: Minimal local dtype map for this module's own `/deposition/*` datasets (mirrors
#: `back_bombardment_events._NUMPY_DTYPE_MAP`'s convention, not imported from it, to keep this
#: module's HDF5 writer/reader self-contained).
_HEAT_SOURCE_NUMPY_DTYPE_MAP: dict[str, Any] = {
    "float64": np.float64,
    "int32": np.int32,
    "bool": np.bool_,
}

#: Root-metadata scalar fields written/read verbatim, in order. `model`/`density_used_kg_m3`
#: identify which deposition model/density produced this file (plan Sec. 4.2: "the deposition
#: model and material-property version belong to back_bombardment_heat_source.h5 ... not to the
#: immutable event file"); the remaining fields are exactly `validate_energy_closure`'s inputs plus
#: the two population counts, so a reader never has to re-run BB0 to audit closure from the file
#: alone. `xy_cell_area_m2` is included so `cell_volume_m3` is reconstructable without recomputing
#: the lateral grid spacing from `x_centers_m`/`y_centers_m`.
_HEAT_SOURCE_ROOT_SCALAR_FIELDS: tuple[str, ...] = (
    "model",
    "density_used_kg_m3",
    "xy_cell_area_m2",
    "escaping_energy_geometric_J_total",
    "escaping_energy_below_tio_validity_J_total",
    "excluded_non_lab6_energy_J_total",
    "total_incident_energy_J",
    "total_deposited_energy_J",
    "n_events_included",
    "n_events_excluded",
)

#: `/deposition/<name>` dataset names, dtype, and unit, in write/read order. Matches
#: `BackBombardmentHeatSource`'s own per-array fields exactly (the footprint mask and the
#: per-period-state `state_ids` array are included here per the task's explicit "Write ... the
#: footprint mask" instruction, alongside the grid/source arrays plan Sec. 4.2 names directly).
_HEAT_SOURCE_ARRAY_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("x_centers_m", "float64", "m"),
    ("y_centers_m", "float64", "m"),
    ("layer_boundaries_um", "float64", "um"),
    ("layer_thickness_m", "float64", "m"),
    ("q_layer_J", "float64", "J"),
    ("cathode_footprint_mask", "bool", ""),
    ("state_ids", "int32", ""),
)


def write_back_bombardment_heat_source_h5(
    path: str | Path,
    heat_source: BackBombardmentHeatSource,
    *,
    source_events_hash: str | None = None,
    extra_root_attrs: dict[str, Any] | None = None,
) -> Path:
    """Write `heat_source` as a strict `back_bombardment_heat_source_v1` HDF5 file (plan Sec. 4.2).

    `source_events_hash`: the sha256 of the input `back_bombardment_events.h5` this source was
    built from (plan Sec. 4.1's "the hash of the input event file", applied here to this exact
    file) -- typically `BackBombardmentStudyInput.event_file_hash`. `None` (the default, e.g. an
    unsaved `source_mode="current_notebook"` study with no event file on disk yet) is written as an
    empty string root attribute rather than omitted, so a reader can always find the attribute and
    distinguish "no hash recorded" from "attribute missing entirely / corrupt file".

    `extra_root_attrs`: forwarded verbatim as additional root attributes (same pattern as
    `write_back_bombardment_events_h5`); `None` values are skipped, non-HDF5-representable values
    are stringified rather than raising, matching that writer's own fallback.

    No numeric source-uncertainty is recorded (plan Sec. 4.2 mentions "source uncertainty" in the
    same breath as energy closure): BB0 (the only deposition model implemented in this pass) has no
    uncertainty model at all -- that is BB1_uncertainty, out of scope here (see this module's
    docstring) -- so a `source_uncertainty_note` root attribute states this explicitly instead of
    fabricating a number.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "h5py is required to write back_bombardment_heat_source_v1 HDF5 files. "
            "Install it with 'pip install h5py'."
        ) from exc

    from .back_bombardment_events import _git_commit_hash

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(path), "w") as h5f:
        dep_grp = h5f.create_group("deposition")
        for name, dtype_name, unit in _HEAT_SOURCE_ARRAY_FIELDS:
            np_dtype = _HEAT_SOURCE_NUMPY_DTYPE_MAP[dtype_name]
            arr = np.asarray(getattr(heat_source, name)).astype(np_dtype)
            ds = dep_grp.create_dataset(name, data=arr)
            ds.attrs["units"] = unit

        h5f.attrs["schema_version"] = BACK_BOMBARDMENT_HEAT_SOURCE_SCHEMA_VERSION
        h5f.attrs["git_commit"] = _git_commit_hash()
        h5f.attrs["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        h5f.attrs["source_events_hash"] = source_events_hash if source_events_hash is not None else ""
        h5f.attrs["source_uncertainty_note"] = (
            "BB0 (TIO/CSDA baseline, the only deposition model implemented in this pass) has no "
            "uncertainty model -- that is BB1_uncertainty (plan Sec. 3.4), out of scope here (see "
            "this module's docstring). No numeric source uncertainty is recorded in this file."
        )
        for name in _HEAT_SOURCE_ROOT_SCALAR_FIELDS:
            h5f.attrs[name] = getattr(heat_source, name)
        if extra_root_attrs:
            for key, value in extra_root_attrs.items():
                if value is None:
                    continue
                try:
                    h5f.attrs[str(key)] = value
                except (TypeError, ValueError):
                    h5f.attrs[str(key)] = str(value)

    return path


def _decode_attr_str_heat_source(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_back_bombardment_heat_source_h5(path: str | Path) -> BackBombardmentHeatSource:
    """Strict reader for `back_bombardment_heat_source_v1` HDF5 files.

    Rejects (with a specific, actionable `ValueError` -- never a generic `KeyError`) a file with no
    `schema_version` root attribute at all, or one that does not exactly equal
    `BACK_BOMBARDMENT_HEAT_SOURCE_SCHEMA_VERSION` -- the same strict-version-rejection convention
    as `rf_gun.back_bombardment_events.read_back_bombardment_events_h5` and
    `rf_gun.comsol_io.load_comsol_thermal_result` (plan Sec. 2.3/4.2: "the loader requires the
    exact declared schema major version and rejects ... rather than silently inventing missing
    ... values").
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "h5py is required to read back_bombardment_heat_source_v1 HDF5 files. "
            "Install it with 'pip install h5py'."
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: file not found.")

    with h5py.File(str(path), "r") as h5f:
        if "schema_version" not in h5f.attrs:
            raise ValueError(
                f"{path}: no 'schema_version' root attribute found. This file is NOT a supported "
                f"{BACK_BOMBARDMENT_HEAT_SOURCE_SCHEMA_VERSION!r} input -- regenerate it with "
                "write_back_bombardment_heat_source_h5 (this module's writer) before loading."
            )
        schema_version = _decode_attr_str_heat_source(h5f.attrs["schema_version"])
        if schema_version != BACK_BOMBARDMENT_HEAT_SOURCE_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version={schema_version!r} does not match the required "
                f"{BACK_BOMBARDMENT_HEAT_SOURCE_SCHEMA_VERSION!r}. Only files written by "
                "write_back_bombardment_heat_source_h5 (this module's writer) are supported; "
                "regenerate this file with the current schema."
            )
        if "deposition" not in h5f:
            raise ValueError(
                f"{path}: schema_version matches but the required '/deposition' group is missing "
                "-- this file is corrupt or was not written by write_back_bombardment_heat_source_h5."
            )

        dep_grp = h5f["deposition"]
        arrays: dict[str, np.ndarray] = {}
        for name, dtype_name, _unit in _HEAT_SOURCE_ARRAY_FIELDS:
            if name not in dep_grp:
                raise ValueError(f"{path}: missing required dataset /deposition/{name}.")
            arrays[name] = np.asarray(dep_grp[name][()], dtype=_HEAT_SOURCE_NUMPY_DTYPE_MAP[dtype_name])

        def _root_attr(name: str) -> Any:
            if name not in h5f.attrs:
                raise ValueError(f"{path}: missing required root attribute {name!r}.")
            return h5f.attrs[name]

        return BackBombardmentHeatSource(
            x_centers_m=arrays["x_centers_m"],
            y_centers_m=arrays["y_centers_m"],
            layer_boundaries_um=arrays["layer_boundaries_um"],
            q_layer_J=arrays["q_layer_J"],
            xy_cell_area_m2=float(_root_attr("xy_cell_area_m2")),
            layer_thickness_m=arrays["layer_thickness_m"],
            cathode_footprint_mask=arrays["cathode_footprint_mask"],
            escaping_energy_geometric_J_total=float(_root_attr("escaping_energy_geometric_J_total")),
            escaping_energy_below_tio_validity_J_total=float(
                _root_attr("escaping_energy_below_tio_validity_J_total")
            ),
            excluded_non_lab6_energy_J_total=float(_root_attr("excluded_non_lab6_energy_J_total")),
            total_incident_energy_J=float(_root_attr("total_incident_energy_J")),
            total_deposited_energy_J=float(_root_attr("total_deposited_energy_J")),
            model=_decode_attr_str_heat_source(_root_attr("model")),
            density_used_kg_m3=float(_root_attr("density_used_kg_m3")),
            n_events_included=int(_root_attr("n_events_included")),
            n_events_excluded=int(_root_attr("n_events_excluded")),
            state_ids=arrays["state_ids"].astype(np.int32),
        )


# ------------------------------------------------------------------------------------------------
# CSDA residual-energy lookup
# ------------------------------------------------------------------------------------------------


def _build_residual_range_lookup(
    electron_deposition: "object",
    rho_kg_m3: float,
    *,
    max_kinetic_energy_keV: float,
    floor_keV: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a dense, log-spaced `(E_grid_keV, R_grid_um)` table for `tio_range_um` at fixed
    `(Z_eff, A_eff, rho_kg_m3)`, and verify `R_grid_um` is strictly monotonically increasing (the
    assumption every CSDA residual-range inversion in this module relies on -- plan Sec. 3.4/19.3's
    "verify monotonicity holds ... before relying on it").

    Raises `RuntimeError` (not a silent fallback) if monotonicity fails anywhere in
    `[floor_keV*0.1, max_kinetic_energy_keV]` -- per the task's explicit instruction not to proceed
    silently; a future caller hitting this would need to restrict the energy range (matching plan
    Sec. 5.3's own ~0.3 keV lower-validity-floor language) rather than have this module guess.
    """
    e_min = floor_keV * 0.1
    e_max = max(max_kinetic_energy_keV, floor_keV * 10.0)
    E_grid_keV = np.logspace(np.log10(e_min), np.log10(e_max), n_points)
    R_grid_um = np.asarray(
        electron_deposition.range_um(E_grid_keV, rho_kg_m3=rho_kg_m3), dtype=float
    )
    if not np.all(np.diff(R_grid_um) > 0.0):
        bad = np.nonzero(np.diff(R_grid_um) <= 0.0)[0]
        raise RuntimeError(
            "tio_range_um is not strictly monotonically increasing in kinetic energy over the "
            f"lookup grid [{e_min:.4g}, {e_max:.4g}] keV at rho={rho_kg_m3:.6g} kg/m3 (first "
            f"failure near E={E_grid_keV[bad[0]]:.4g} keV) -- the CSDA residual-range inversion "
            "this module relies on (plan Sec. 3.4/19.3) assumes strict monotonicity and refuses "
            "to silently proceed. Restrict the energy range (e.g. raise "
            "tio_validity_floor_keV) or investigate rf_gun.materials.electron_range.tio_range_um "
            "at the offending energy before retrying."
        )
    return E_grid_keV, R_grid_um


# ------------------------------------------------------------------------------------------------
# BB0 per-event depth-dose deposition
# ------------------------------------------------------------------------------------------------


def _residual_energy_keV(
    s_um: np.ndarray,
    K_i_keV: float,
    R_i_um: float,
    R_grid_um: np.ndarray,
    E_grid_keV: np.ndarray,
) -> np.ndarray:
    """CSDA residual kinetic energy `K_res(s)` [keV] at path distance `s_um` (array-like, um) into
    an electron of incident energy `K_i_keV` and total range `R_i_um`.

    Deliberately anchors the two exact endpoints WITHOUT going through the interpolation table:
    `K_res(0) := K_i_keV` exactly (the electron's own known incident energy -- no need to invert
    what is already known analytically) and `K_res(s>=R_i_um) := 0.0` exactly (fully stopped). Only
    intermediate points are read off the `(E_grid_keV, R_grid_um)` lookup table via `np.interp`.
    This is what makes the per-event energy bookkeeping close exactly in `_deposit_one_event` (see
    module docstring) -- table-resolution error can only affect intermediate/leftover values, never
    the anchors the telescoping sum depends on.
    """
    s = np.atleast_1d(np.asarray(s_um, dtype=float))
    out = np.empty_like(s)
    at_zero = s <= 0.0
    out[at_zero] = K_i_keV
    R_query = R_i_um - s
    stopped = (~at_zero) & (R_query <= 0.0)
    out[stopped] = 0.0
    mid = (~at_zero) & (~stopped)
    if np.any(mid):
        # R_grid_um/E_grid_keV are both increasing, so a query below the grid's range corresponds
        # to an energy below the grid's own minimum, and above -> above the grid's own maximum;
        # explicit left/right make that constant-extrapolation direction unambiguous (it also
        # happens to be np.interp's own unspecified-left/right default, restated here for clarity).
        out[mid] = np.interp(R_query[mid], R_grid_um, E_grid_keV, left=E_grid_keV[0], right=E_grid_keV[-1])
    return out


def _deposit_one_event(
    *,
    K_i_keV: float,
    cos_theta: float,
    R_i_um: float,
    incident_energy_J: float,
    layer_boundaries_um: np.ndarray,
    z_max_um: float,
    R_floor_um: float,
    R_grid_um: np.ndarray,
    E_grid_keV: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Deposit one electron's incident energy onto the `layer_boundaries_um` depth grid.

    Returns `(deposited_J[n_layers], escaping_geometric_J, escaping_below_floor_J)`.

    Method (plan Sec. 3.4/6.2): path-length layer boundaries are `layer_boundaries_um / cos_theta`
    (the projection this module's docstring documents -- `z_local = s*cos_theta`, inverted here to
    `s = z_local/cos_theta`). CSDA tracking is only trusted out to
    `s_track_max = min(R_i_um - R_floor_um, z_max_um/cos_theta)` -- whichever of the TIO validity
    floor or the modeled/physical depth limit binds first. Per-layer deposited energy is
    `K_res(s_lo_clipped) - K_res(s_hi_clipped)`, clipped to `[0, s_track_max]`; because adjacent
    layers share the same (pre-clip) boundary value, these differences telescope EXACTLY to
    `K_res(0) - K_res(s_track_max) = K_i_keV - K_res(s_track_max)` regardless of the lookup table's
    resolution (see `_residual_energy_keV`'s docstring) -- the remaining `K_res(s_track_max)` is
    then booked as exactly one of the two escape mechanisms (never both, never neither), so the
    per-electron total always closes to `K_i_keV` exactly modulo floating-point round-off.
    """
    n_layers = layer_boundaries_um.size - 1
    if K_i_keV <= 0.0 or incident_energy_J <= 0.0:
        return np.zeros(n_layers, dtype=float), 0.0, 0.0

    s_lo = layer_boundaries_um[:-1] / cos_theta
    s_hi = layer_boundaries_um[1:] / cos_theta

    s_upper_valid = max(0.0, R_i_um - R_floor_um)
    s_upper_geom = z_max_um / cos_theta
    is_floor_case = s_upper_valid <= s_upper_geom
    s_track_max = min(s_upper_valid, s_upper_geom)

    s_lo_c = np.clip(s_lo, 0.0, s_track_max)
    s_hi_c = np.clip(s_hi, 0.0, s_track_max)
    K_lo = _residual_energy_keV(s_lo_c, K_i_keV, R_i_um, R_grid_um, E_grid_keV)
    K_hi = _residual_energy_keV(s_hi_c, K_i_keV, R_i_um, R_grid_um, E_grid_keV)
    deposited_keV = K_lo - K_hi  # shape (n_layers,), telescopes to K_i_keV - K_res(s_track_max)

    leftover_keV = float(
        _residual_energy_keV(np.array([s_track_max]), K_i_keV, R_i_um, R_grid_um, E_grid_keV)[0]
    )

    # keV(single electron) -> J(this macroparticle's real electrons): deposited_keV/K_i_keV is the
    # FRACTION of this electron's total kinetic energy deposited in that layer (it telescopes to
    # exactly 1.0 across all layers + leftover, per this function's docstring), so multiplying by
    # the event's own already-computed incident_energy_J (rather than independently re-deriving
    # macro_weight_electrons * K_i_keV * 1000 * q_e) guarantees this module's Joule bookkeeping
    # matches events.incident_energy_J's own w*K*e convention (plan Sec. 4.1) exactly, with no
    # second, independently-rounded unit-conversion path to drift out of sync with it.
    to_J = incident_energy_J / K_i_keV  # J per keV of this electron's own energy
    deposited_J = deposited_keV * to_J

    escaping_geometric_J = 0.0
    escaping_below_floor_J = 0.0
    if is_floor_case:
        # Below the TIO validity floor: dump the entire leftover into the current terminal layer
        # (plan Sec. 5.3) -- fully deposited, never discarded, so the "below validity" escape
        # bucket for this event is exactly 0.0 (see module docstring).
        z_track = s_track_max * cos_theta
        term_j = int(
            np.clip(np.searchsorted(layer_boundaries_um, z_track, side="right") - 1, 0, n_layers - 1)
        )
        deposited_J[term_j] += leftover_keV * to_J
        escaping_below_floor_J = 0.0
    else:
        # Genuine geometric escape: path continues beyond the modeled/physical depth limit while
        # still above the TIO validity floor -- this energy is NOT deposited anywhere.
        escaping_geometric_J = leftover_keV * to_J

    return deposited_J, escaping_geometric_J, escaping_below_floor_J


# ------------------------------------------------------------------------------------------------
# BB0 driver
# ------------------------------------------------------------------------------------------------


def _build_bb0_tio_heat_source(
    events: "BackBombardmentEvents",
    geometry: "CathodeGeometry",
    material: "CathodeMaterialSet",
    *,
    xy_grid_n: int,
    layer_boundaries_um: np.ndarray,
    density_override_kg_m3: float | None,
    tio_validity_floor_keV: float,
    csda_lookup_points: int,
) -> BackBombardmentHeatSource:
    from .materials.registry import validate_material_for

    validate_material_for(material, "bb0_deposition")
    electron_deposition = material.electron_deposition

    layer_boundaries_um = np.asarray(layer_boundaries_um, dtype=float)
    if layer_boundaries_um.ndim != 1 or layer_boundaries_um.size < 2:
        raise ValueError(
            f"layer_boundaries_um must be a 1D array with at least 2 entries, got shape "
            f"{layer_boundaries_um.shape}"
        )
    if not np.all(np.diff(layer_boundaries_um) > 0.0):
        raise ValueError(f"layer_boundaries_um must be strictly increasing, got {layer_boundaries_um}")
    n_layers = layer_boundaries_um.size - 1

    rho_used = (
        float(density_override_kg_m3)
        if density_override_kg_m3 is not None
        else float(electron_deposition.rho0_kg_m3)
    )

    # The modeled/physical depth limit is the SMALLER of the layer grid's own deepest boundary and
    # the actual physical LaB6 thickness -- an electron cannot deposit past the real solid (plan
    # Sec. 3.4: "clip the path to the actual LaB6 volume"), regardless of how deep the (convergence
    # -parameter) layer grid happens to extend.
    z_max_um = min(float(layer_boundaries_um[-1]), float(geometry.cathode_length_mm) * 1000.0)

    mask_lab6 = events.heats_lab6_mask
    incident_energy_J = np.asarray(events.incident_energy_J, dtype=float)
    K_eV = np.asarray(events.kinetic_energy_eV, dtype=float)
    cos_inc = np.asarray(events.cos_incidence, dtype=float)
    x_hit = np.asarray(events.x_hit_m, dtype=float)
    y_hit = np.asarray(events.y_hit_m, dtype=float)

    excluded_non_lab6_energy_J_total = float(np.sum(incident_energy_J[~mask_lab6]))
    total_incident_energy_J = float(np.sum(incident_energy_J[mask_lab6]))

    idx_lab6 = np.nonzero(mask_lab6)[0]
    K_keV_included = K_eV[idx_lab6] / 1000.0 if idx_lab6.size else np.array([0.0])
    max_kinetic_energy_keV = float(np.max(K_keV_included)) if K_keV_included.size else 0.0

    E_grid_keV, R_grid_um = _build_residual_range_lookup(
        electron_deposition,
        rho_used,
        max_kinetic_energy_keV=max(2000.0, 1.5 * max_kinetic_energy_keV),
        floor_keV=tio_validity_floor_keV,
        n_points=csda_lookup_points,
    )
    R_floor_um = float(electron_deposition.range_um(tio_validity_floor_keV, rho_kg_m3=rho_used))

    # Lateral grid: NxN cells over [-bevel_outer_radius, +bevel_outer_radius] in both x and y (plan
    # Sec. 6.2/10.2), cells outside the physical disk masked out (never populated, since no event
    # can land there, but the mask is exposed for later plotting/masking convenience).
    R_max_m = float(geometry.bevel_outer_radius_mm) * 1.0e-3
    n = int(xy_grid_n)
    if n < 1:
        raise ValueError(f"xy_grid_n must be a positive integer, got {xy_grid_n!r}")
    edges = np.linspace(-R_max_m, R_max_m, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = float(edges[1] - edges[0])
    xy_cell_area_m2 = dx * dx
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    rr_mm = np.hypot(xx, yy) * 1.0e3
    cathode_footprint_mask = rr_mm <= float(geometry.bevel_outer_radius_mm)

    layer_thickness_m = np.diff(layer_boundaries_um) * 1.0e-6
    q_layer_J = np.zeros((n, n, n_layers), dtype=float)
    escaping_geometric_total = 0.0
    escaping_below_floor_total = 0.0

    for i in idx_lab6:
        K_i_keV = K_eV[i] / 1000.0
        if K_i_keV <= 0.0 or incident_energy_J[i] <= 0.0:
            continue
        # cos_incidence should always be strictly positive for a qualified heating event (plan
        # Sec. 3.1's p.n_in>0 test); guard the degenerate exactly-grazing edge case to avoid a
        # divide-by-zero rather than assume upstream data is perfectly clean.
        cos_theta = max(float(cos_inc[i]), 1.0e-6)
        R_i_um = float(electron_deposition.range_um(K_i_keV, rho_kg_m3=rho_used))

        deposited_J, esc_geom_J, esc_floor_J = _deposit_one_event(
            K_i_keV=K_i_keV,
            cos_theta=cos_theta,
            R_i_um=R_i_um,
            incident_energy_J=float(incident_energy_J[i]),
            layer_boundaries_um=layer_boundaries_um,
            z_max_um=z_max_um,
            R_floor_um=R_floor_um,
            R_grid_um=R_grid_um,
            E_grid_keV=E_grid_keV,
        )

        ix = int(np.clip(np.searchsorted(edges, x_hit[i], side="right") - 1, 0, n - 1))
        iy = int(np.clip(np.searchsorted(edges, y_hit[i], side="right") - 1, 0, n - 1))
        q_layer_J[ix, iy, :] += deposited_J
        escaping_geometric_total += esc_geom_J
        escaping_below_floor_total += esc_floor_J

    state_ids = np.unique(np.asarray(events.state_id)[mask_lab6]).astype(np.int32)

    return BackBombardmentHeatSource(
        x_centers_m=centers,
        y_centers_m=centers,
        layer_boundaries_um=layer_boundaries_um,
        q_layer_J=q_layer_J,
        xy_cell_area_m2=xy_cell_area_m2,
        layer_thickness_m=layer_thickness_m,
        cathode_footprint_mask=cathode_footprint_mask,
        escaping_energy_geometric_J_total=float(escaping_geometric_total),
        escaping_energy_below_tio_validity_J_total=float(escaping_below_floor_total),
        excluded_non_lab6_energy_J_total=excluded_non_lab6_energy_J_total,
        total_incident_energy_J=total_incident_energy_J,
        total_deposited_energy_J=float(np.sum(q_layer_J)),
        model="BB0_TIO",
        density_used_kg_m3=rho_used,
        n_events_included=int(idx_lab6.size),
        n_events_excluded=int(events.n_events - idx_lab6.size),
        state_ids=state_ids,
    )


def _build_bb1_uncertainty_heat_source(*_args, **_kwargs) -> BackBombardmentHeatSource:
    """BB1 -- uncertainty model (plan Sec. 3.4): "retain BB0's depth shape while scanning or
    tabulating `eta_dep(K,theta)` and lateral spread. This supplies honest uncertainty bands before
    a validated low-energy transport calculation exists." Out of scope for this pass (plan addendum
    Sec. 19.2, Work Package 2); not implemented here.
    """
    raise NotImplementedError(
        "BB1_uncertainty (plan Sec. 3.4) is out of scope for the current implementation pass "
        "(addendum Sec. 19.2, Work Package 2 covers BB0 only). It needs a scanned/tabulated "
        "retained-energy fraction eta_dep(K,theta) and a lateral-spread model on top of BB0's "
        "depth shape, neither of which exist yet."
    )


def _build_bb2_response_library_heat_source(*_args, **_kwargs) -> BackBombardmentHeatSource:
    """BB2 -- response library (plan Sec. 3.4): "convolve events with a Geant4 Livermore/Penelope
    response library `G(dr; K, theta, T)`". A later accuracy upgrade, not a prerequisite for the
    first Python/COMSOL comparison (plan Sec. 3.4); out of scope for this pass (addendum Sec. 19.2).
    """
    raise NotImplementedError(
        "BB2_response_library (plan Sec. 3.4) is out of scope for the current implementation pass "
        "(addendum Sec. 19.2). It requires a Geant4 Livermore/Penelope response library convolution "
        "that does not exist yet -- 'a later accuracy upgrade, not a prerequisite for the first "
        "Python/COMSOL comparison' per the plan itself."
    )


# ------------------------------------------------------------------------------------------------
# Public dispatch entry point
# ------------------------------------------------------------------------------------------------


def build_back_bombardment_heat_source(
    events: "BackBombardmentEvents",
    geometry: "CathodeGeometry",
    material: "CathodeMaterialSet",
    deposition_config: "DepositionConfig",
    *,
    xy_grid_n: int = 41,
    layer_boundaries_um: np.ndarray = DEFAULT_DEPTH_LAYER_BOUNDARIES_UM,
    density_override_kg_m3: float | None = None,
    tio_validity_floor_keV: float = DEFAULT_TIO_VALIDITY_FLOOR_KEV,
    csda_lookup_points: int = DEFAULT_CSDA_LOOKUP_POINTS,
) -> BackBombardmentHeatSource:
    """Build the volumetric BB0/BB1/BB2 back-bombardment heat source for one representative RF
    period (plan Sec. 10.2's suggested public function; addendum Sec. 19.2, Work Package 2 scope).

    Dispatches on `deposition_config.model`:
      * `"BB0_TIO"` (default, the only model implemented so far) -- see this module's docstring.
      * `"BB1_uncertainty"` / `"BB2_response_library"` -- raise `NotImplementedError` (plan Sec. 3.4,
        both are later work, addendum Sec. 19.2).

    `xy_grid_n`: lateral grid resolution (cells per axis) over
    `[-bevel_outer_radius_mm, +bevel_outer_radius_mm]` in both `x` and `y` (plan Sec. 6.2/10.2);
    default 41 is a modest, fast-to-test resolution, not a validated production value -- mesh
    convergence (like the depth-layer boundaries) is Work Package 3's job.

    `density_override_kg_m3`: named hook for a future caller wanting the `R(E,T)=R(E,T0)*rho(T0)/
    rho(T)` temperature correction of plan Sec. 5.3 -- pass the density at the temperature of
    interest. Defaults to `None`, which uses the material's own reference density
    (`electron_deposition.rho0_kg_m3`, Bakr's `rho0=4720 kg/m3`), i.e. the BB0 baseline behavior
    plan Sec. 3.4 specifies.
    """
    model = deposition_config.model
    if model == "BB0_TIO":
        return _build_bb0_tio_heat_source(
            events,
            geometry,
            material,
            xy_grid_n=xy_grid_n,
            layer_boundaries_um=np.asarray(layer_boundaries_um, dtype=float),
            density_override_kg_m3=density_override_kg_m3,
            tio_validity_floor_keV=tio_validity_floor_keV,
            csda_lookup_points=csda_lookup_points,
        )
    if model == "BB1_uncertainty":
        return _build_bb1_uncertainty_heat_source()
    if model == "BB2_response_library":
        return _build_bb2_response_library_heat_source()
    raise ValueError(
        f"Unknown deposition_config.model {model!r}; valid values: 'BB0_TIO' (implemented), "
        "'BB1_uncertainty' (not implemented, plan Sec. 3.4), "
        "'BB2_response_library' (not implemented, plan Sec. 3.4)."
    )
