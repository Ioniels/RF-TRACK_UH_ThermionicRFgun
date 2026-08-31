"""HDF5 v2 back-bombardment event schema (implementation plan Sec. 2.3, 3.1, 3.3, 4.1, 4.2,
10.2; addendum Sec. 19.2/19.5).

Scope: this module is PURE PYTHON / HDF5 -- it has NO RF-Track dependency and performs NO
particle tracking. It defines:

  * `BackBombardmentEvents`, the in-memory container for one qualified representative-RF-period
    event set (plan Sec. 4.1's `/events/*` per-event columns plus the mandatory root metadata and
    `/accounting`, `/geometry/surfaces`, `/source_state`, `/provenance` groups);
  * the strict v2 HDF5 writer/reader (`write_back_bombardment_events_h5`/
    `read_back_bombardment_events_h5`), which rejects anything other than an exact
    `schema_version` match -- including, deliberately, the *existing* unversioned
    `back_bombardment_events.h5` files already on disk under `outputs/runs/*/`, written by the
    old `rf_gun.io.save_back_bombardment_events_hdf5` (plan Sec. 2.3: "the loader checks the
    exact schema major version and fails clearly on older files");
  * `resolve_back_bombardment_study_input`, the `current_notebook` / `load_run` mode resolver
    (plan Sec. 2.3);
  * `display_back_bombardment_event_schema`, the notebook's post-run schema printout (plan
    Sec. 4.1's "the notebook prints this table immediately after the run", Sec. 11 Cell 2);
  * `extract_back_bombardment_events`, the Work Package 1 event-capture implementation: RF-Track's
    backstop/loss-table (`Volume.get_lost_particles()`) separated from dynamic-aperture losses via
    `rf_gun.backstop_loss_separation.identify_backstop_loss_candidates` (plan Sec. 3.2 step 3,
    addendum Sec. 19.6 -- NOT the plan's original literal "negative-z backstop band" wording, which
    that module's docstring shows is empirically false for the physically dominant transit case),
    then ray-cast through `rf_gun.cathode_geometry.CathodeGeometry.intersect_ray` (plan Sec. 3.2
    step 4) and joined by particle ID to each event's emission-time record (plan Sec. 3.2 step 2).

What this module deliberately does NOT do: it does not implement BB0/BB1/BB2 deposition
(`back_bombardment_deposition.py`, a later module), and it does not touch the existing legacy
`rf_gun.back_bombardment.BackBombardmentData` / `rf_gun.io.save_back_bombardment_events_hdf5`
(kept exactly as-is, retained as `legacy_ballistic` per plan Sec. 3.2 -- see that module's
docstring -- for the current notebook cell until a later pass rewires it). `extract_back_bombardment_events`
below does duck-type against RF-Track bunch objects (`simulation_result.B0`/`.Bout`, via
`.get_phase_space(...)`) to join emission records and account for transmitted charge -- exactly
the "later module that does depend on RF-Track" the original stub anticipated, just implemented
here per this pass's explicit direction rather than in a separate file; it still never imports
`RF_Track` itself, so a caller without RF-Track installed can still import this module freely (it
only fails, same as always, if it is actually asked to call a method an incompatible object
doesn't have).

Coordinate/normal convention: identical to `rf_gun.cathode_geometry` -- z=0 is the cathode
emission surface, inward normals point from vacuum into the solid, and a genuine heating event
requires `p_hit . n_in > 0` (plan Sec. 3.1). `coordinate_system`/`normal_convention` are stored as
explicit string metadata (not just implied by code) so a reader never has to guess.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .cathode_geometry import (
    CathodeGeometry,
    SURFACE_CATHODE_BEVEL,
    SURFACE_CATHODE_FLAT,
    SURFACE_CATHODE_SIDE,
    SURFACE_HOLDER,
    SURFACE_LABELS,
    SURFACE_UNKNOWN,
    SURFACE_ZONE_INFO,
)
from .constants import c, q_e

#: Bump only when a field is added, removed, or its meaning changes in a way that would break a
#: reader written against an earlier version (matches `rf_gun.io.RUN_CONFIG_SCHEMA_VERSION`'s own
#: reasoning). Exact string per plan Sec. 4.1/2.3: "schema_version = 'back_bombardment_events_v2'".
BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION = "back_bombardment_events_v2"

#: LaB6 zone codes for `BackBombardmentEvents.heats_lab6_mask` (plan Sec. 3.3's table: flat and
#: bevel are always LaB6; the side wall is LaB6 "if exposed" -- included here since this project's
#: side-wall zone, when it appears at all, is by definition an exposed LaB6 surface).
_LAB6_SURFACE_CODES = (int(SURFACE_CATHODE_FLAT), int(SURFACE_CATHODE_BEVEL), int(SURFACE_CATHODE_SIDE))

#: `quality_flags` bit layout (uint32 bitmask) for `extract_back_bombardment_events` -- this
#: implementation's own choice; plan Sec. 4.1 only requires the column to exist ("Interpolation,
#: boundary, or validity flags"), it does not fix exact bits.
#:
#:   bit 0 (value 1): the loss-table row's particle ID had no matching row in the initial
#:     (emission-time) B0 record (`rf_gun.simulation._try_get_particle_ids`'s own creation-row
#:     lookup). Should be rare/never -- RF-Track should never report a lost-particle ID that did
#:     not originate in B0 -- but is counted and flagged rather than silently dropped, per this
#:     pass's explicit instruction. `x_emit_m`/`y_emit_m`/`z_emit_m`/`t_emit_rf_s`/
#:     `rf_phase_emit_rad`/`return_time_s` are `nan` for a flagged row.
#:   bit 1 (value 2): `geometry.intersect_ray` found no physical surface for this candidate
#:     (`RayIntersection.hit=False`) -- e.g. a numerically pathological ray. `x_hit_m`/`y_hit_m`/
#:     `z_hit_m`/`t_hit_rf_s`/`return_time_s`/`n_in_*`/`cos_incidence`/`incidence_angle_rad` are
#:     `nan` and `surface_code=SURFACE_UNKNOWN` for a flagged row -- still retained as an event
#:     row, per plan Sec. 3.2's "select the first physical intersection" leaving no candidate.
QUALITY_FLAG_ID_JOIN_FAILED = np.uint32(1 << 0)
QUALITY_FLAG_RAY_NO_HIT = np.uint32(1 << 1)

_NUMPY_DTYPE_MAP: dict[str, Any] = {
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "uint32": np.uint32,
    "uint8": np.uint8,
    "float64": np.float64,
    "bool": np.bool_,
}

#: Single source of truth for the `/events/*` column layout (plan Sec. 4.1's table): used to
#: build the dataclass's per-event array fields, to drive the strict HDF5 writer/reader (dtype +
#: `units` dataset attribute), and to populate `/schema/columns` / `display_back_bombardment_event_schema`.
EVENT_SCHEMA_COLUMNS: list[dict[str, str]] = [
    {"name": "event_id", "dtype": "int64", "unit": "", "description": "Unique event row ID"},
    {"name": "particle_id", "dtype": "int64", "unit": "", "description": "RF-Track %id"},
    {"name": "state_id", "dtype": "int32", "unit": "", "description": "Macropulse/keyframe state that produced the event"},
    {"name": "x_emit_m", "dtype": "float64", "unit": "m", "description": "Emission point x"},
    {"name": "y_emit_m", "dtype": "float64", "unit": "m", "description": "Emission point y"},
    {"name": "z_emit_m", "dtype": "float64", "unit": "m", "description": "Emission point z"},
    {"name": "t_emit_rf_s", "dtype": "float64", "unit": "s", "description": "Time within the representative RF period at emission"},
    {"name": "rf_phase_emit_rad", "dtype": "float64", "unit": "rad", "description": "Emission RF phase"},
    {"name": "x_hit_m", "dtype": "float64", "unit": "m", "description": "True surface intersection x"},
    {"name": "y_hit_m", "dtype": "float64", "unit": "m", "description": "True surface intersection y"},
    {"name": "z_hit_m", "dtype": "float64", "unit": "m", "description": "True surface intersection z"},
    {"name": "t_hit_rf_s", "dtype": "float64", "unit": "s", "description": "Impact time in the representative RF period"},
    {"name": "return_time_s", "dtype": "float64", "unit": "s", "description": "t_hit - t_emit"},
    {"name": "px_MeV_c", "dtype": "float64", "unit": "MeV/c", "description": "Impact momentum x"},
    {"name": "py_MeV_c", "dtype": "float64", "unit": "MeV/c", "description": "Impact momentum y"},
    {"name": "pz_MeV_c", "dtype": "float64", "unit": "MeV/c", "description": "Impact momentum z"},
    {"name": "kinetic_energy_eV", "dtype": "float64", "unit": "eV", "description": "Single-electron incident kinetic energy"},
    {"name": "macro_weight_electrons", "dtype": "float64", "unit": "electrons", "description": "Physical electrons represented"},
    {"name": "incident_energy_J", "dtype": "float64", "unit": "J", "description": "w*K*e"},
    {"name": "n_in_x", "dtype": "float64", "unit": "", "description": "Unit normal into struck solid, x component"},
    {"name": "n_in_y", "dtype": "float64", "unit": "", "description": "Unit normal into struck solid, y component"},
    {"name": "n_in_z", "dtype": "float64", "unit": "", "description": "Unit normal into struck solid, z component"},
    {"name": "cos_incidence", "dtype": "float64", "unit": "", "description": "Cosine of incidence angle relative to inward normal"},
    {"name": "incidence_angle_rad", "dtype": "float64", "unit": "rad", "description": "Incidence angle relative to inward normal"},
    {"name": "surface_code", "dtype": "uint8", "unit": "", "description": "Zone code defined in /geometry/surfaces"},
    {"name": "heats_lab6", "dtype": "bool", "unit": "", "description": "True for LaB6 zones"},
    {"name": "furthest_z_m", "dtype": "float64", "unit": "m", "description": "Diagnostic maximum screen reach"},
    {"name": "n_screens_reached", "dtype": "int16", "unit": "", "description": "Diagnostic screen count"},
    {"name": "quality_flags", "dtype": "uint32", "unit": "", "description": "Interpolation, boundary, or validity flags bit mask"},
]

#: Just the field names, in display/storage order.
EVENT_ARRAY_FIELDS: tuple[str, ...] = tuple(col["name"] for col in EVENT_SCHEMA_COLUMNS)

#: Mandatory root-metadata field names (plan Sec. 4.1's bullet list), in the order they're written
#: as HDF5 root attributes. `n_events` is also written but is derived (len of the arrays), not a
#: `BackBombardmentEvents` field, so it's added separately in the writer.
_ROOT_METADATA_FIELDS: tuple[str, ...] = (
    "schema_version",
    "coordinate_system",
    "normal_convention",
    "rf_frequency_Hz",
    "rf_period_s",
    "sample_represents",
    "emission_phase_window",
    "n_launched",
    "q_emitted_per_sample_C",
    "event_locator",
    "sc_enabled",
    "mirror_charge_enabled",
    "beam_loading_enabled",
    "deflection_magnet_enabled",
    "emission_iteration_enabled",
    "flat_radius_mm",
    "bevel_width_mm",
    "bevel_angle_deg",
    "cathode_length_mm",
    "insertion_offset_mm",
)


@dataclass(frozen=True)
class BackBombardmentEvents:
    """One qualified representative-RF-period back-bombardment event set (plan Sec. 4.1's HDF5 v2
    data contract, held in memory). This is the v2 replacement/superset of the legacy
    `rf_gun.back_bombardment.BackBombardmentData` -- a different dataclass, in a different module,
    so both coexist without collision (the legacy one stays wired to the current notebook cell
    via `legacy_ballistic` until Work Package 1 rewires event capture; see plan Sec. 3.2).

    Per-event arrays (all share one length, `n_events`; see `EVENT_SCHEMA_COLUMNS` for the
    authoritative name/dtype/unit/description table, transcribed verbatim from plan Sec. 4.1):
    `event_id, particle_id, state_id, x_emit_m, y_emit_m, z_emit_m, t_emit_rf_s,
    rf_phase_emit_rad, x_hit_m, y_hit_m, z_hit_m, t_hit_rf_s, return_time_s, px_MeV_c, py_MeV_c,
    pz_MeV_c, kinetic_energy_eV, macro_weight_electrons, incident_energy_J, n_in_x, n_in_y,
    n_in_z, cos_incidence, incidence_angle_rad, surface_code, heats_lab6, furthest_z_m,
    n_screens_reached, quality_flags`.

    Mandatory root-metadata scalars (plan Sec. 4.1's bullet list): `schema_version` (must equal
    `BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION`), `coordinate_system`, `normal_convention`,
    `rf_frequency_Hz`, `rf_period_s`, `sample_represents` (default `"one_rf_period"`),
    `emission_phase_window`, `n_launched`, `q_emitted_per_sample_C`, `event_locator`
    (`"backstop_raycast_v1"` or `"legacy_ballistic"`), the five physics-switch booleans
    (`sc_enabled`, `mirror_charge_enabled`, `beam_loading_enabled`, `deflection_magnet_enabled`,
    `emission_iteration_enabled`), and the five geometry scalars (`flat_radius_mm`,
    `bevel_width_mm`, `bevel_angle_deg`, `cathode_length_mm`, `insertion_offset_mm` -- these
    mirror `rf_gun.cathode_geometry.CathodeGeometry`'s own fields exactly; `resolved_geometry()`
    reconstructs a `CathodeGeometry` instance from them on demand rather than this dataclass
    embedding one directly, so the mandatory HDF5 root attrs plan Sec. 4.1 requires stay flat
    scalars with no nested-object indirection).

    `accounting` (plan Sec. 3.1/4.1: "Counts, charge, and kinetic energy are recorded before and
    after every filter" / "`/accounting` stores launched/emitted/transmitted/returned/other-lost
    counts, charge, and energy both before and after filters"). This implementation's chosen
    nested-dict shape (documented here since the plan does not fix exact key names):

        {
          "counts": {
            "n_launched": int, "n_emitted": int, "n_transmitted": int,
            "n_returned_before_filter": int, "n_returned_after_filter": int,
            "n_other_lost": int,
            "by_surface": {"<surface_code>": int, ...},   # qualified/returned events per zone
          },
          "charge_C": {
            "emitted": float, "transmitted": float, "other_lost": float,
            "returned_before_filter": float, "returned_after_filter": float,
          },
          "energy_J": {
            "incident_before_filter": float, "incident_after_filter": float,
            "by_surface": {"<surface_code>": float, ...},
          },
        }

    `validate()` (also run from `__post_init__`) checks `charge_C`/`energy_J`'s
    `*_after_filter` does not exceed `*_before_filter` beyond a small relative tolerance whenever
    both are present -- filtering can only remove events, never add charge/energy -- and raises
    `ValueError` naming the offending numbers if it does. Missing keys are tolerated (accounting
    can be partially populated by an upstream caller); only a *present* violation raises.

    `provenance` (plan Sec. 4.1: "schema version, git commit, run ID, timestamp, RF-Track
    version, random seed, command line, and hashes of field maps/configuration"); see
    `build_back_bombardment_provenance` for the helper that assembles this dict (git commit via
    `subprocess`, `sys.argv`, etc.) -- this dataclass itself just stores whatever dict it's given.

    `source_state` (plan Sec. 4.1: "`/source_state` stores the temperature/current/cavity-
    amplitude values that define the keyframe"). Empty by default (`{}`) -- populated by the
    later macropulse-keyframe machinery (Work Package 1/2/3), not by this data-layer module.

    Internal consistency is checked eagerly by `__post_init__` -> `validate()`: every per-event
    array must have the same length (`n_events`), and `schema_version` must match the module
    constant exactly. Both raise `ValueError` with the specific offending values/lengths -- never
    silently truncate or renumber. Note this dataclass is frozen (matching
    `rf_gun.cathode_geometry.CathodeGeometry`/`RayIntersection`'s convention): `__post_init__`
    only *validates*, it never mutates a field via `object.__setattr__`.
    """

    # ---- Per-event arrays (length n_events) -----------------------------------------------
    event_id: np.ndarray
    particle_id: np.ndarray
    state_id: np.ndarray
    x_emit_m: np.ndarray
    y_emit_m: np.ndarray
    z_emit_m: np.ndarray
    t_emit_rf_s: np.ndarray
    rf_phase_emit_rad: np.ndarray
    x_hit_m: np.ndarray
    y_hit_m: np.ndarray
    z_hit_m: np.ndarray
    t_hit_rf_s: np.ndarray
    return_time_s: np.ndarray
    px_MeV_c: np.ndarray
    py_MeV_c: np.ndarray
    pz_MeV_c: np.ndarray
    kinetic_energy_eV: np.ndarray
    macro_weight_electrons: np.ndarray
    incident_energy_J: np.ndarray
    n_in_x: np.ndarray
    n_in_y: np.ndarray
    n_in_z: np.ndarray
    cos_incidence: np.ndarray
    incidence_angle_rad: np.ndarray
    surface_code: np.ndarray
    heats_lab6: np.ndarray
    furthest_z_m: np.ndarray
    n_screens_reached: np.ndarray
    quality_flags: np.ndarray

    # ---- Mandatory root metadata (plan Sec. 4.1), no defaults -----------------------------
    schema_version: str
    coordinate_system: str
    normal_convention: str
    rf_frequency_Hz: float
    rf_period_s: float
    emission_phase_window: str
    n_launched: int
    q_emitted_per_sample_C: float
    event_locator: str
    sc_enabled: bool
    mirror_charge_enabled: bool
    beam_loading_enabled: bool
    deflection_magnet_enabled: bool
    emission_iteration_enabled: bool
    flat_radius_mm: float
    bevel_width_mm: float
    bevel_angle_deg: float
    cathode_length_mm: float
    insertion_offset_mm: float

    # ---- Accounting / provenance dicts (plan Sec. 4.1), no defaults -----------------------
    accounting: dict[str, Any]
    provenance: dict[str, Any]

    # ---- Defaulted fields (must trail all fields above per dataclass ordering rules) -----
    sample_represents: str = "one_rf_period"
    source_state: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise `ValueError` (with the specific offending values) if this event set is
        internally inconsistent. Never silently passes on a mismatch -- see class docstring."""
        if self.schema_version != BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION:
            raise ValueError(
                f"BackBombardmentEvents.schema_version={self.schema_version!r} does not match "
                f"the required {BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION!r}."
            )

        lengths: dict[str, int] = {}
        for name in EVENT_ARRAY_FIELDS:
            arr = np.asarray(getattr(self, name))
            lengths[name] = int(arr.shape[0]) if arr.ndim >= 1 else -1
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                "BackBombardmentEvents: per-event arrays must all share one length, but got "
                f"mismatched lengths: {lengths}"
            )

        if not isinstance(self.accounting, dict):
            raise ValueError(
                f"BackBombardmentEvents.accounting must be a dict, got {type(self.accounting)!r}"
            )

        tol = 1e-6
        charge = self.accounting.get("charge_C", {}) if isinstance(self.accounting, dict) else {}
        if isinstance(charge, dict):
            before = charge.get("returned_before_filter")
            after = charge.get("returned_after_filter")
            if before is not None and after is not None and after > before * (1.0 + tol) + 1e-30:
                raise ValueError(
                    "BackBombardmentEvents.accounting['charge_C']: 'returned_after_filter' "
                    f"({after!r}) exceeds 'returned_before_filter' ({before!r}) beyond tolerance "
                    f"{tol!r} -- filtering can only remove events, never add charge."
                )

        energy = self.accounting.get("energy_J", {}) if isinstance(self.accounting, dict) else {}
        if isinstance(energy, dict):
            e_before = energy.get("incident_before_filter")
            e_after = energy.get("incident_after_filter")
            if e_before is not None and e_after is not None and e_after > e_before * (1.0 + tol) + 1e-30:
                raise ValueError(
                    "BackBombardmentEvents.accounting['energy_J']: 'incident_after_filter' "
                    f"({e_after!r}) exceeds 'incident_before_filter' ({e_before!r}) beyond "
                    f"tolerance {tol!r} -- filtering can only remove events, never add energy."
                )

    # ------------------------------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------------------------------

    @property
    def n_events(self) -> int:
        return int(np.asarray(self.event_id).shape[0])

    @property
    def heats_lab6_mask(self) -> np.ndarray:
        """True where `surface_code` is one of the LaB6 zones (flat/bevel/side) -- the v2
        equivalent of the legacy `BackBombardmentData.heating_relevant` concept, computed
        directly from `surface_code` rather than trusted blindly from the stored `heats_lab6`
        column (useful as a cross-check between the two -- see
        `tests/test_back_bombardment_events.py`)."""
        return np.isin(np.asarray(self.surface_code), _LAB6_SURFACE_CODES)

    def resolved_geometry(self) -> CathodeGeometry:
        """Reconstruct the `CathodeGeometry` these events were captured against, from this
        object's own flat geometry scalars (plan Sec. 4.1's mandatory root metadata)."""
        return CathodeGeometry(
            flat_radius_mm=self.flat_radius_mm,
            bevel_width_mm=self.bevel_width_mm,
            bevel_angle_deg=self.bevel_angle_deg,
            cathode_length_mm=self.cathode_length_mm,
            insertion_offset_mm=self.insertion_offset_mm,
        )

    def geometry_surfaces(self) -> dict[str, Any]:
        """`/geometry/surfaces` content (plan Sec. 4.1: "stores code, label, material, analytic
        parameters, and physical area"): `rf_gun.cathode_geometry.SURFACE_ZONE_INFO`'s static
        label/material_owner/plot_treatment table, augmented with this event set's own resolved
        `CathodeGeometry` areas/radii for the zones that have them. Keyed by the surface code as
        a string (HDF5 group names must be strings)."""
        geometry = self.resolved_geometry()
        out: dict[str, Any] = {}
        for code, info in SURFACE_ZONE_INFO.items():
            entry = dict(info)
            if code == int(SURFACE_CATHODE_FLAT):
                entry["area_mm2"] = geometry.flat_area_mm2
                entry["outer_radius_mm"] = geometry.flat_radius_mm
            elif code == int(SURFACE_CATHODE_BEVEL):
                entry["area_mm2"] = geometry.bevel_true_area_mm2
                entry["inner_radius_mm"] = geometry.flat_radius_mm
                entry["outer_radius_mm"] = geometry.bevel_outer_radius_mm
            elif code == int(SURFACE_HOLDER):
                entry["inner_radius_mm"] = geometry.bevel_outer_radius_mm
                entry["outer_radius_mm"] = geometry.holder_outer_radius_mm
            out[str(code)] = entry
        return out


# --------------------------------------------------------------------------------------------
# Provenance helper (plan Sec. 4.1: "/provenance stores schema version, git commit, run ID,
# timestamp, RF-Track version, random seed, command line, and hashes of field maps/configuration")
# --------------------------------------------------------------------------------------------

def _git_commit_hash() -> str:
    """`git rev-parse HEAD` at call time, `"unknown"` on any failure (not in a git repo, git not
    installed, timeout, ...) -- never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).resolve().parent),
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            if commit:
                return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def build_back_bombardment_provenance(
    *,
    run_id: str,
    rf_track_version: str | None = None,
    random_seed: int | None = None,
    config_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the `/provenance` dict (plan Sec. 4.1) at call time.

    `rf_track_version` is accepted as a plain caller-supplied string -- this module never
    imports RF-Track itself (it is pure Python/HDF5). `config_hashes` is likewise accepted
    as-is (e.g. hashes of field-map files/configuration dataclasses) -- computing those hashes
    is the caller's responsibility, not this module's.
    """
    return {
        "schema_version": BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
        "git_commit": _git_commit_hash(),
        "run_id": str(run_id),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rf_track_version": rf_track_version,
        "random_seed": random_seed,
        "command_line": " ".join(sys.argv),
        "config_hashes": dict(config_hashes) if config_hashes else {},
    }


# --------------------------------------------------------------------------------------------
# Strict HDF5 v2 writer/reader
# --------------------------------------------------------------------------------------------

def write_back_bombardment_events_h5(
    path: str | Path,
    events: BackBombardmentEvents,
    *,
    extra_root_attrs: dict[str, Any] | None = None,
) -> Path:
    """Write `events` as a strict `back_bombardment_events_v2` HDF5 file (plan Sec. 4.1's group
    layout): `/events/<column>` datasets (each with a `units` attribute, per Sec. 4.1: "assign a
    units attribute to every numeric dataset"), `/geometry/surfaces/<code>` subgroups,
    `/accounting`, `/source_state`, and `/provenance` groups (each holding one JSON-encoded
    `json` attribute -- a deliberate implementation choice, documented here: these three dicts
    have caller-defined nesting/keys that this module does not own, so a single lossless
    JSON blob round-trips them exactly without this module having to also standardize an
    arbitrary flattened-attribute convention for arbitrary nested keys), `/schema/columns` (name/
    unit/description arrays, for `display_back_bombardment_event_schema` and manual inspection),
    and the mandatory root attributes of plan Sec. 4.1's bullet list plus `n_events`.

    Follows `rf_gun.io.save_back_bombardment_events_hdf5`'s h5py conventions (the `ImportError`
    guard, `Path`/`mkdir` handling). Does NOT reuse that function -- this is a new, coexisting
    writer for the new schema; the legacy v1 writer/file is untouched.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "h5py is required to write back_bombardment_events_v2 HDF5 files. "
            "Install it with 'pip install h5py'."
        ) from exc

    from .io import to_json_safe

    events.validate()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(path), "w") as h5f:
        events_grp = h5f.create_group("events")
        for col in EVENT_SCHEMA_COLUMNS:
            name = col["name"]
            np_dtype = _NUMPY_DTYPE_MAP[col["dtype"]]
            arr = np.asarray(getattr(events, name)).astype(np_dtype)
            ds = events_grp.create_dataset(name, data=arr)
            ds.attrs["units"] = col["unit"]

        surfaces_grp = h5f.create_group("geometry/surfaces")
        for code_str, info in events.geometry_surfaces().items():
            sub = surfaces_grp.create_group(code_str)
            for key, value in info.items():
                sub.attrs[str(key)] = value

        h5f.create_group("accounting").attrs["json"] = json.dumps(to_json_safe(events.accounting))
        h5f.create_group("provenance").attrs["json"] = json.dumps(to_json_safe(events.provenance))
        h5f.create_group("source_state").attrs["json"] = json.dumps(to_json_safe(events.source_state))

        schema_grp = h5f.create_group("schema/columns")
        str_dtype = h5py.string_dtype(encoding="utf-8")
        schema_grp.create_dataset(
            "name", data=np.array([c["name"] for c in EVENT_SCHEMA_COLUMNS], dtype=object), dtype=str_dtype
        )
        schema_grp.create_dataset(
            "unit", data=np.array([c["unit"] for c in EVENT_SCHEMA_COLUMNS], dtype=object), dtype=str_dtype
        )
        schema_grp.create_dataset(
            "description",
            data=np.array([c["description"] for c in EVENT_SCHEMA_COLUMNS], dtype=object),
            dtype=str_dtype,
        )

        root_attrs: dict[str, Any] = {name: getattr(events, name) for name in _ROOT_METADATA_FIELDS}
        root_attrs["n_events"] = events.n_events
        if extra_root_attrs:
            root_attrs.update(extra_root_attrs)
        for key, value in root_attrs.items():
            if value is None:
                continue
            try:
                h5f.attrs[str(key)] = value
            except (TypeError, ValueError):
                h5f.attrs[str(key)] = str(value)

    return path


def _decode_attr_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_back_bombardment_events_h5(path: str | Path) -> BackBombardmentEvents:
    """Strict reader for `back_bombardment_events_v2` HDF5 files.

    Rejects (with a specific, actionable `ValueError` -- never a generic `KeyError`) any file
    whose root `schema_version` attribute is missing entirely (e.g. every existing
    `back_bombardment_events.h5` under `outputs/runs/*/`, written by the pre-v2
    `rf_gun.io.save_back_bombardment_events_hdf5`) or does not exactly equal
    `BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION` (plan Sec. 2.3/4.2: "Only files written by the new
    implementation are supported. The loader requires the exact declared schema major version and
    rejects the current unversioned HDF5 rather than silently inventing missing IDs, geometry,
    timing, or normalization.").
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "h5py is required to read back_bombardment_events_v2 HDF5 files. "
            "Install it with 'pip install h5py'."
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: file not found.")

    with h5py.File(str(path), "r") as h5f:
        if "schema_version" not in h5f.attrs:
            raise ValueError(
                f"{path}: no 'schema_version' root attribute found. This file predates the "
                f"{BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION!r} schema (plan Sec. 2.3/4.2) and is "
                "NOT a supported back_bombardment_events_v2 input -- for example, every "
                "back_bombardment_events.h5 written by the old rf_gun.io."
                "save_back_bombardment_events_hdf5 carries no schema_version attribute at all. "
                "Regenerate this run directory with write_back_bombardment_events_h5 (the new "
                "event writer) before using it as a study input; see "
                "resolve_back_bombardment_study_input's 'load_run' mode."
            )
        schema_version = _decode_attr_str(h5f.attrs["schema_version"])
        if schema_version != BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version={schema_version!r} does not match the required "
                f"{BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION!r}. Only files written by "
                "write_back_bombardment_events_h5 (this module's writer) are supported; "
                "regenerate this file with the current schema."
            )

        if "events" not in h5f:
            raise ValueError(
                f"{path}: schema_version matches but the required '/events' group is missing "
                "-- this file is corrupt or was not written by write_back_bombardment_events_h5."
            )
        events_grp = h5f["events"]
        arrays: dict[str, np.ndarray] = {}
        for col in EVENT_SCHEMA_COLUMNS:
            name = col["name"]
            if name not in events_grp:
                raise ValueError(f"{path}: missing required dataset /events/{name}.")
            arrays[name] = np.asarray(events_grp[name][()])

        def _root_attr(name: str) -> Any:
            if name not in h5f.attrs:
                raise ValueError(f"{path}: missing required root attribute {name!r}.")
            return h5f.attrs[name]

        accounting = json.loads(h5f["accounting"].attrs["json"]) if "accounting" in h5f else {}
        provenance = json.loads(h5f["provenance"].attrs["json"]) if "provenance" in h5f else {}
        source_state = json.loads(h5f["source_state"].attrs["json"]) if "source_state" in h5f else {}

        return BackBombardmentEvents(
            **arrays,
            schema_version=schema_version,
            coordinate_system=_decode_attr_str(_root_attr("coordinate_system")),
            normal_convention=_decode_attr_str(_root_attr("normal_convention")),
            rf_frequency_Hz=float(_root_attr("rf_frequency_Hz")),
            rf_period_s=float(_root_attr("rf_period_s")),
            emission_phase_window=_decode_attr_str(_root_attr("emission_phase_window")),
            n_launched=int(_root_attr("n_launched")),
            q_emitted_per_sample_C=float(_root_attr("q_emitted_per_sample_C")),
            event_locator=_decode_attr_str(_root_attr("event_locator")),
            sc_enabled=bool(_root_attr("sc_enabled")),
            mirror_charge_enabled=bool(_root_attr("mirror_charge_enabled")),
            beam_loading_enabled=bool(_root_attr("beam_loading_enabled")),
            deflection_magnet_enabled=bool(_root_attr("deflection_magnet_enabled")),
            emission_iteration_enabled=bool(_root_attr("emission_iteration_enabled")),
            flat_radius_mm=float(_root_attr("flat_radius_mm")),
            bevel_width_mm=float(_root_attr("bevel_width_mm")),
            bevel_angle_deg=float(_root_attr("bevel_angle_deg")),
            cathode_length_mm=float(_root_attr("cathode_length_mm")),
            insertion_offset_mm=float(_root_attr("insertion_offset_mm")),
            accounting=accounting,
            provenance=provenance,
            sample_represents=_decode_attr_str(_root_attr("sample_represents")),
            source_state=source_state,
        )


# --------------------------------------------------------------------------------------------
# Study-input resolver (plan Sec. 2.3)
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BackBombardmentStudyInput:
    """The single validated object every downstream study step consumes, regardless of which of
    the two source modes produced it (plan Sec. 2.3: "Both modes are converted immediately to the
    same validated object.").

    `event_file_hash`: sha256 of the actual HDF5 file bytes for `source_mode="load_run"`. For
    `source_mode="current_notebook"` there is no file on disk (that is the whole point of the
    mode), so this is instead a sha256 over a canonical in-memory serialization of `events`
    (`schema_version` plus every per-event array's raw bytes, in `EVENT_ARRAY_FIELDS` order) --
    still a stable, reproducible fingerprint of the event content, just not a file hash. This
    distinction is deliberate and documented here rather than silently reusing the same field
    name for two different things without comment.
    """

    events: BackBombardmentEvents
    source_mode: str
    origin_run_id: str | None
    source_path: Path | None
    event_file_hash: str


_VALID_SOURCE_MODES: tuple[str, ...] = ("current_notebook", "load_run")


def _sha256_of_in_memory_events(events: BackBombardmentEvents) -> str:
    hasher = hashlib.sha256()
    hasher.update(events.schema_version.encode("utf-8"))
    for name in EVENT_ARRAY_FIELDS:
        hasher.update(np.asarray(getattr(events, name)).tobytes())
    return hasher.hexdigest()


def resolve_back_bombardment_study_input(
    source_mode: str,
    *,
    current_events: BackBombardmentEvents | None = None,
    run_dir: str | Path | None = None,
) -> BackBombardmentStudyInput:
    """Resolve either study-input mode (plan Sec. 2.3) to one `BackBombardmentStudyInput`.

    `source_mode="current_notebook"`: wraps `current_events` (the `BackBombardmentEvents` object
    already produced by the preceding in-notebook RF-Track cell); raises `ValueError` if
    `current_events is None`.

    `source_mode="load_run"`: requires `run_dir`; loads `<run_dir>/back_bombardment_events.h5` via
    `read_back_bombardment_events_h5`, which raises its own specific `ValueError` (propagated
    unchanged here, not wrapped or reworded) if that file predates the v2 schema -- e.g. the
    example run directory named in plan Sec. 2.3,
    `outputs/runs/20260830_004543_T1650K_SCon_BLon/`, is NOT currently a valid `load_run` input
    for exactly this reason, and this function does not work around that.

    Any other `source_mode` raises `ValueError` listing the two valid options.
    """
    if source_mode not in _VALID_SOURCE_MODES:
        raise ValueError(
            f"Unknown source_mode {source_mode!r}; valid options: {list(_VALID_SOURCE_MODES)}."
        )

    if source_mode == "current_notebook":
        if current_events is None:
            raise ValueError(
                "source_mode='current_notebook' requires current_events (the BackBombardmentEvents "
                "object produced by the preceding in-notebook RF-Track back-bombardment cell); "
                "got None."
            )
        return BackBombardmentStudyInput(
            events=current_events,
            source_mode=source_mode,
            origin_run_id=None,
            source_path=None,
            event_file_hash=_sha256_of_in_memory_events(current_events),
        )

    # source_mode == "load_run"
    if run_dir is None:
        raise ValueError("source_mode='load_run' requires run_dir (a completed run directory).")
    run_dir = Path(run_dir)
    event_path = run_dir / "back_bombardment_events.h5"
    if not event_path.is_file():
        raise ValueError(
            f"{run_dir}: no back_bombardment_events.h5 found (expected at {event_path})."
        )
    events = read_back_bombardment_events_h5(event_path)
    event_file_hash = hashlib.sha256(event_path.read_bytes()).hexdigest()
    return BackBombardmentStudyInput(
        events=events,
        source_mode=source_mode,
        origin_run_id=run_dir.name,
        source_path=event_path,
        event_file_hash=event_file_hash,
    )


# --------------------------------------------------------------------------------------------
# Notebook schema printout (plan Sec. 4.1/11 Cell 2)
# --------------------------------------------------------------------------------------------

def display_back_bombardment_event_schema(
    events: BackBombardmentEvents, *, h5_path: Path | None = None
) -> None:
    """Print the schema version, ordered `/schema/columns` (name/unit/description), surface-zone
    event counts, and charge/energy closure numbers -- "so that the user sees `ID, x, y, px, ...`
    without opening the file manually" (plan Sec. 4.1). When `h5_path is None` (an intentionally
    unsaved `current_notebook` study, plan Sec. 11), prints that no file was written instead of
    inventing a path.
    """
    from .io import to_json_safe

    print(f"Back-bombardment event schema: {events.schema_version}")
    print(
        f"  event_locator={events.event_locator!r}  sample_represents={events.sample_represents!r}  "
        f"coordinate_system={events.coordinate_system!r}  normal_convention={events.normal_convention!r}"
    )
    print(
        f"  rf_frequency_Hz={events.rf_frequency_Hz:.6g}  rf_period_s={events.rf_period_s:.6g}  "
        f"n_events={events.n_events}  n_launched={events.n_launched}"
    )
    print()
    print(f"  {'name':<24}{'unit':<10}description")
    for col in EVENT_SCHEMA_COLUMNS:
        print(f"  {col['name']:<24}{col['unit']:<10}{col['description']}")
    print()

    codes = np.asarray(events.surface_code)
    print("  Surface-zone event counts:")
    for code in sorted(int(c) for c in np.unique(codes)):
        label = SURFACE_LABELS.get(code, "unknown")
        count = int(np.sum(codes == code))
        print(f"    code {code:>3} ({label}): {count}")
    print()

    print("  Accounting:")
    print(json.dumps(to_json_safe(events.accounting), indent=2, sort_keys=True))

    if h5_path is not None:
        print(f"\n  Written to: {h5_path}")
    else:
        print("\n  No file written (in-memory current_notebook study); file path unavailable.")


# --------------------------------------------------------------------------------------------
# Event-capture stub (Work Package 1, deferred -- see module docstring)
# --------------------------------------------------------------------------------------------

def _try_get_phase_space_column(B: Any, code: str, selection: str = "all") -> np.ndarray | None:
    """Same defensive pattern as `rf_gun.simulation._try_get_particle_ids`, generalized to any
    single RF-Track phase-space code (here, `"%N"` for per-macroparticle weight) -- `None` on any
    failure (wrong/missing binding, incompatible object), never raises."""
    try:
        vals = np.asarray(B.get_phase_space(code, selection), dtype=float).reshape(-1)
        return vals if vals.size else None
    except Exception:
        return None


def extract_back_bombardment_events(
    simulation_result: Any,
    geometry: CathodeGeometry,
    capture_config: Any,
    *,
    f_hz: float,
    run_id: str = "unknown",
    vol_params: Any = None,
    emission_iteration_enabled: bool = False,
) -> BackBombardmentEvents:
    """Extract a qualified `BackBombardmentEvents` set from a completed RF-Track production run
    (plan Sec. 3.2's backstop/loss-table + ray-cast procedure, Sec. 10.2's suggested public
    function signature; Work Package 1).

    Keeps the plan's exact three positional arguments (`simulation_result, geometry,
    capture_config`); `f_hz` is added as a required keyword-only argument because RF frequency is
    a run parameter in this codebase (`rf_gun.rftrack_volume.VolumeBuildParams.f_hz`), not a
    project-wide constant -- see `rf_gun.rf_params`/`run_thermionic_tm010.py` for how it is
    normally threaded through; nothing here silently assumes 2.856 GHz. `run_id` and `vol_params`
    are further optional keyword-only additions this implementation needed and the plan does not
    fix: `run_id` feeds `build_back_bombardment_provenance` (no `SimulationResult` field carries a
    run identifier); `vol_params` (a `rf_gun.rftrack_volume.VolumeBuildParams`, or any object with
    the same attribute names) supplies the mandatory `sc_enabled`/`mirror_charge_enabled`/
    `beam_loading_enabled`/`deflection_magnet_enabled` root-metadata switches, since those live on
    `VolumeBuildParams` (used to build the `Volume` this run tracked), not on `SimulationResult`
    or `capture_config` -- passing `None` (the default) records all four as `False` rather than
    guessing. `emission_iteration_enabled` has no `VolumeBuildParams` analog at all (it reflects
    whether `rf_gun.emission_iteration.run_emission_field_iteration` produced this run's source),
    so it is its own explicit keyword, default `False`.

    `simulation_result.lost_table` must be present and non-empty: RF-Track only ever populates it
    when the run was made with `VolumeBuildParams(cathode_backstop_enabled=True)` (or the dynamic
    aperture actually removed something) *and* `DiagnosticsParams(save_lost_particles=True)` --
    see `rf_gun.simulation._extract_lost_particles`. An empty/missing table here almost always
    means the backstop was not enabled for this run, a genuine misconfiguration this function
    raises `ValueError` on rather than silently returning an empty (and misleadingly "clean")
    event set.

    Procedure (plan Sec. 3.2):
      1. (Precondition, not performed here) the caller must have already tracked with a thin
         absorbing `Aperture_1d` backstop just behind the cathode plane
         (`rf_gun.aperture.build_cathode_backstop`, wired into `VolumeBuildParams.cathode_backstop_enabled`).
      2. `simulation_result.lost_table` is RF-Track's own `Volume.get_lost_particles()` table,
         already normalized by `rf_gun.diagnostics.to_lost_table_array` to `(n, 11)` =
         `X, Px, Y, Py, Z, Pz, T, MASS, Q, N, ID` (mm/MeV-c/mm-c convention; this is `Volume`'s own
         loss-table column order, distinct from `Lattice.get_lost_particles()`'s different order --
         plan addendum Sec. 19.5).
      3. Backstop candidates are identified by
         `rf_gun.backstop_loss_separation.identify_backstop_loss_candidates` -- NOT the plan's
         original literal "negative-z backstop band" wording (`Pz<0` and `Z<=0`): that module's own
         docstring and validation (`tests/test_backstop_loss_separation.py`) show a genuine transit
         backstop hit is recorded at a small *positive* residual `Z` (up to ~0.63mm observed), so
         the actual rule is `Pz<0` and `Z` within `[backstop_z_min_m, 0 + z_slack_m]` and a valid
         particle ID (addendum Sec. 19.6). `backstop_z_min_m` is derived here from
         `capture_config.backstop_thickness_mm` (`-thickness_mm*1e-3`, this project's
         `z0_global=0` cathode-frame convention). Dynamic-aperture losses (everything the mask does
         not select) remain a disjoint class, accounted for as `n_other_lost`, never merged in.
      4. Every backstop candidate (hit or not) is ray-cast through `geometry.intersect_ray`,
         called directly with the loss-table's raw `(X, Y, Z, Px, Py, Pz)` -- `intersect_ray`
         propagates forward along the momentum direction and does the actual physical back-solve
         from a small positive residual `Z` down through `z=0` and, where applicable, into the
         bevel; this function does not pre-correct `Z` itself (see `rf_gun.backstop_loss_separation`'s
         module docstring for why that division of labor is intentional). A candidate with
         `RayIntersection.hit=False` becomes an event row with `QUALITY_FLAG_RAY_NO_HIT` set and
         `surface_code=SURFACE_UNKNOWN` rather than being dropped (plan: "never a reason to delete
         an event").
      5. `t_hit` is corrected for the short field-free segment between the loss-table state and the
         true ray-cast intersection: `t_hit_mm_c = T_lost_mm_c + path_length_mm / beta`, where
         `beta = |p|/E` at the (momentum-conserving, field-free) loss-table state and
         `path_length_mm` is the exact 3D distance from the loss-table position to the ray-cast hit
         point (the same "mm/c" convention `rf_gun.back_bombardment.compute_back_bombardment`
         already uses for its own 1D `z/beta_z` version of this correction).
      6. `geometry.intersect_ray` already enforces `p_hit . n_in > 0` as part of its own `hit`
         criterion (plan Sec. 3.1) -- a ray failing that test is exactly a `hit=False` row here,
         handled by step 4/`QUALITY_FLAG_RAY_NO_HIT` above, not a second independent filter.
         `capture_config.require_inward_momentum=False` is accepted but currently a no-op: doing
         anything else would require a second, unfiltered ray-cast mode in `cathode_geometry`
         itself, out of scope for this pass -- documented here rather than silently ignored.
      7. Emission-time fields (`x_emit_m`/`y_emit_m`/`z_emit_m`/`t_emit_rf_s`/`rf_phase_emit_rad`)
         are joined by RF-Track's own `%id` against `rf_gun.simulation._try_get_particle_ids(B0,
         selection="all")`'s creation-row-ordered ID array, indexing into
         `thermo_info["initial_phase_space"]` (columns `x_mm, px, y_mm, py, z_mm(=0), pz`) and
         `thermo_info["initial_t0_mm_c"]`. An unmatched ID is a data-integrity problem, not a row
         to drop -- it is counted (`accounting["counts"]["n_id_join_failed"]`) and flagged
         (`QUALITY_FLAG_ID_JOIN_FAILED`), leaving its emission fields `nan`.
      8. Accounting/provenance are assembled (see the class docstring's accounting shape, reused
         verbatim/extended here) and a validated `BackBombardmentEvents` is returned with
         `event_locator` taken from `capture_config.event_locator` (default
         `"backstop_raycast_v1"`).
    """
    from .backstop_loss_separation import identify_backstop_loss_candidates
    from .back_bombardment import _screen_reach
    from .simulation import _try_get_particle_ids

    lost_table_raw = getattr(simulation_result, "lost_table", None)
    lost_arr = np.asarray(lost_table_raw, dtype=float) if lost_table_raw is not None else None
    if lost_arr is None or lost_arr.ndim != 2 or lost_arr.shape[0] == 0:
        raise ValueError(
            "extract_back_bombardment_events: simulation_result.lost_table is empty or missing. "
            "RF-Track only populates this table when the run was tracked with "
            "VolumeBuildParams(cathode_backstop_enabled=True) and "
            "DiagnosticsParams(save_lost_particles=True) (see rf_gun.simulation._extract_lost_particles) "
            "-- without the backstop enabled there is no return-event loss table to extract events "
            "from at all (plan Sec. 3.2 step 1). If this run genuinely had zero back-bombardment "
            "events despite the backstop being enabled, lost_table would still be a (0, 11) array, "
            "not None/absent; a None/missing table means the backstop was not enabled for this run."
        )

    thermo_info: dict[str, Any] = dict(getattr(simulation_result, "thermo_info", None) or {})
    B0 = getattr(simulation_result, "B0", None)
    Bout = getattr(simulation_result, "Bout", None)
    M_snaps = list(getattr(simulation_result, "M_snaps", None) or [])
    z_snaps = list(getattr(simulation_result, "z_snaps", None) or [])

    backstop_thickness_mm = float(getattr(capture_config, "backstop_thickness_mm"))
    mask = identify_backstop_loss_candidates(
        lost_arr,
        backstop_z_min_m=-backstop_thickness_mm * 1e-3,
        backstop_z_max_m=0.0,
    )
    n_candidates = int(np.sum(mask))
    n_other_lost = int(lost_arr.shape[0] - n_candidates)
    candidates = lost_arr[mask]
    n = candidates.shape[0]

    x0, px = candidates[:, 0], candidates[:, 1]
    y0, py = candidates[:, 2], candidates[:, 3]
    z0, pz = candidates[:, 4], candidates[:, 5]
    t0_lost_mm_c = candidates[:, 6]
    mass = candidates[:, 7]
    macro_weight = candidates[:, 9]
    particle_id = np.rint(candidates[:, 10]).astype(np.int64)

    # -- Ray-cast every candidate (step 4) --------------------------------------------------
    ray = geometry.intersect_ray(x0, y0, z0, px, py, pz)

    # -- Kinetic energy at impact: momentum is unchanged over the short field-free segment --
    p_norm = np.sqrt(px**2 + py**2 + pz**2)
    E_MeV = np.sqrt(p_norm**2 + mass**2)
    K_MeV = E_MeV - mass
    kinetic_energy_eV = K_MeV * 1.0e6
    incident_energy_J = macro_weight * kinetic_energy_eV * q_e

    # -- Time correction over the field-free segment (step 5) -------------------------------
    dx = ray.x_hit_mm - x0
    dy = ray.y_hit_mm - y0
    dz = ray.z_hit_mm - z0
    path_len_mm = np.sqrt(dx**2 + dy**2 + dz**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(E_MeV > 0.0, p_norm / np.where(E_MeV > 0.0, E_MeV, 1.0), np.nan)
        t_travel_mm_c = np.where(
            ray.hit & np.isfinite(beta) & (beta > 0.0),
            path_len_mm / np.where(beta > 0.0, beta, 1.0),
            np.nan,
        )
    t_hit_mm_c = np.where(ray.hit, t0_lost_mm_c + t_travel_mm_c, np.nan)
    t_hit_rf_s = t_hit_mm_c * 1.0e-3 / c

    # -- Emission-time join by particle ID (step 7) ------------------------------------------
    init_ids = _try_get_particle_ids(B0, selection="all") if B0 is not None else None
    initial_phase_space = np.asarray(thermo_info.get("initial_phase_space", np.zeros((0, 6))), dtype=float)
    initial_t0_mm_c = np.asarray(thermo_info.get("initial_t0_mm_c", np.zeros((0,))), dtype=float)

    x_emit_m = np.full(n, np.nan)
    y_emit_m = np.full(n, np.nan)
    z_emit_m = np.full(n, np.nan)
    t_emit_rf_s = np.full(n, np.nan)
    id_join_failed = np.ones(n, dtype=bool)

    have_init = (
        init_ids is not None
        and init_ids.size
        and initial_phase_space.ndim == 2
        and initial_phase_space.shape[0] == init_ids.size
    )
    if have_init and n:
        init_ids_int = np.rint(init_ids).astype(np.int64)
        order = np.argsort(init_ids_int)
        sorted_ids = init_ids_int[order]
        pos = np.searchsorted(sorted_ids, particle_id)
        pos_clipped = np.clip(pos, 0, sorted_ids.size - 1)
        found = sorted_ids[pos_clipped] == particle_id
        rows0 = order[pos_clipped]

        x_emit_m = np.where(found, initial_phase_space[rows0, 0] * 1e-3, np.nan)
        y_emit_m = np.where(found, initial_phase_space[rows0, 2] * 1e-3, np.nan)
        z_emit_m = np.where(found, initial_phase_space[rows0, 4] * 1e-3, np.nan)
        if initial_t0_mm_c.size == init_ids.size:
            t0_emit_mm_c = np.where(found, initial_t0_mm_c[rows0], np.nan)
        else:
            t0_emit_mm_c = np.full(n, np.nan)
        t_emit_rf_s = t0_emit_mm_c * 1.0e-3 / c
        id_join_failed = ~found

    return_time_s = t_hit_rf_s - t_emit_rf_s
    rf_phase_emit_rad = np.mod(2.0 * np.pi * float(f_hz) * t_emit_rf_s, 2.0 * np.pi)

    # -- Diagnostic screen-reach fields -------------------------------------------------------
    n_screens_reached_i, last_screen_z_mm = _screen_reach(particle_id, M_snaps, z_snaps)
    n_screens_reached = n_screens_reached_i.astype(np.int16)
    furthest_z_m = last_screen_z_mm * 1e-3

    # -- Quality flags -------------------------------------------------------------------------
    quality_flags = np.zeros(n, dtype=np.uint32)
    quality_flags |= id_join_failed.astype(np.uint32) * QUALITY_FLAG_ID_JOIN_FAILED
    quality_flags |= (~ray.hit).astype(np.uint32) * QUALITY_FLAG_RAY_NO_HIT

    n_id_join_failed = int(np.sum(id_join_failed))
    n_ray_no_hit = int(np.sum(~ray.hit))
    if n_id_join_failed:
        print(
            f"Warning: extract_back_bombardment_events: {n_id_join_failed} of {n} backstop "
            "candidate(s) had a particle ID with no matching row in B0's initial phase space "
            "(RF-Track should never report a lost-particle ID that did not originate in B0) -- "
            "these events are retained with QUALITY_FLAG_ID_JOIN_FAILED set and nan emission fields.",
            flush=True,
        )

    unknown_frac = (n_ray_no_hit / n) if n else 0.0
    max_unknown_frac = float(getattr(capture_config, "max_unknown_surface_fraction", 0.01))
    if unknown_frac > max_unknown_frac:
        print(
            f"Warning: extract_back_bombardment_events: {unknown_frac:.3%} of qualified backstop "
            f"candidates found no physical surface intersection (SURFACE_UNKNOWN), exceeding "
            f"max_unknown_surface_fraction={max_unknown_frac:.3%} (plan Sec. 3.3).",
            flush=True,
        )

    # -- Charge/weight bookkeeping and cross-check --------------------------------------------
    Q_total_C = float(thermo_info.get("Q_total_C", 0.0))
    q_emitted_abs_C = abs(Q_total_C)
    if init_ids is not None and init_ids.size:
        n_launched = int(init_ids.size)
    else:
        n_launched = int(initial_phase_space.shape[0]) if initial_phase_space.ndim == 2 else 0
    weight_uniform = (
        q_emitted_abs_C / q_e / n_launched if n_launched > 0 and q_emitted_abs_C > 0.0 else 0.0
    )
    if weight_uniform > 0.0 and macro_weight.size:
        rel_dev = float(np.max(np.abs(macro_weight - weight_uniform))) / weight_uniform
        if rel_dev > 1e-3:
            print(
                "Warning: extract_back_bombardment_events: lost_table's N (macro_weight_electrons) "
                f"deviates from thermo_info['Q_total_C']/q_e/n_launched by {rel_dev:.3%} "
                f"(expected {weight_uniform:.6g}, observed range "
                f"[{float(np.min(macro_weight)):.6g}, {float(np.max(macro_weight)):.6g}]).",
                flush=True,
            )

    ids_transmitted = _try_get_particle_ids(Bout, selection="all") if Bout is not None else None
    n_transmitted = int(ids_transmitted.size) if ids_transmitted is not None else 0
    weights_transmitted = _try_get_phase_space_column(Bout, "%N", "all") if Bout is not None else None
    if weights_transmitted is not None and weights_transmitted.size == n_transmitted:
        Q_transmitted_C = q_e * float(np.sum(weights_transmitted))
    else:
        Q_transmitted_C = q_e * weight_uniform * n_transmitted

    Q_other_lost_C = q_e * float(np.sum(lost_arr[~mask, 9])) if n_other_lost else 0.0
    Q_returned_before_C = q_e * float(np.sum(macro_weight)) if n else 0.0
    Q_returned_after_C = q_e * float(np.sum(macro_weight[ray.hit])) if n else 0.0

    E_incident_before_J = float(np.sum(incident_energy_J)) if n else 0.0
    E_incident_after_J = float(np.sum(incident_energy_J[ray.hit])) if n else 0.0

    surface_code = ray.surface_code
    heats_lab6 = np.isin(surface_code, _LAB6_SURFACE_CODES)

    counts_by_surface: dict[str, int] = {}
    energy_by_surface: dict[str, float] = {}
    for code in (np.unique(surface_code) if n else np.asarray([], dtype=np.uint8)):
        code_mask = surface_code == code
        counts_by_surface[str(int(code))] = int(np.sum(code_mask))
        energy_by_surface[str(int(code))] = float(np.sum(incident_energy_J[code_mask]))

    accounting: dict[str, Any] = {
        "counts": {
            "n_launched": n_launched,
            "n_emitted": n_launched,
            "n_transmitted": n_transmitted,
            "n_returned_before_filter": n_candidates,
            "n_returned_after_filter": int(np.sum(ray.hit)) if n else 0,
            "n_other_lost": n_other_lost,
            "n_id_join_failed": n_id_join_failed,
            "n_ray_no_hit": n_ray_no_hit,
            "by_surface": counts_by_surface,
        },
        "charge_C": {
            "emitted": q_emitted_abs_C,
            "transmitted": Q_transmitted_C,
            "other_lost": Q_other_lost_C,
            "returned_before_filter": Q_returned_before_C,
            "returned_after_filter": Q_returned_after_C,
        },
        "energy_J": {
            "incident_before_filter": E_incident_before_J,
            "incident_after_filter": E_incident_after_J,
            "by_surface": energy_by_surface,
        },
    }

    provenance = build_back_bombardment_provenance(run_id=run_id)

    return BackBombardmentEvents(
        event_id=np.arange(n, dtype=np.int64),
        particle_id=particle_id,
        state_id=np.zeros(n, dtype=np.int32),
        x_emit_m=x_emit_m,
        y_emit_m=y_emit_m,
        z_emit_m=z_emit_m,
        t_emit_rf_s=t_emit_rf_s,
        rf_phase_emit_rad=rf_phase_emit_rad,
        x_hit_m=ray.x_hit_mm * 1e-3,
        y_hit_m=ray.y_hit_mm * 1e-3,
        z_hit_m=ray.z_hit_mm * 1e-3,
        t_hit_rf_s=t_hit_rf_s,
        return_time_s=return_time_s,
        px_MeV_c=px,
        py_MeV_c=py,
        pz_MeV_c=pz,
        kinetic_energy_eV=kinetic_energy_eV,
        macro_weight_electrons=macro_weight,
        incident_energy_J=incident_energy_J,
        n_in_x=ray.n_in_x,
        n_in_y=ray.n_in_y,
        n_in_z=ray.n_in_z,
        cos_incidence=ray.cos_incidence,
        incidence_angle_rad=ray.incidence_angle_rad,
        surface_code=surface_code,
        heats_lab6=heats_lab6,
        furthest_z_m=furthest_z_m,
        n_screens_reached=n_screens_reached,
        quality_flags=quality_flags,
        schema_version=BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
        coordinate_system="cathode_z0_vacuum_positive_z",
        normal_convention="inward_vacuum_to_solid",
        rf_frequency_Hz=float(f_hz),
        rf_period_s=1.0 / float(f_hz),
        emission_phase_window="full_rf_period",
        n_launched=n_launched,
        q_emitted_per_sample_C=q_emitted_abs_C,
        event_locator=str(getattr(capture_config, "event_locator", "backstop_raycast_v1")),
        sc_enabled=bool(getattr(vol_params, "sc_enabled", False)),
        mirror_charge_enabled=bool(getattr(vol_params, "mirror_charge_enabled", False)),
        beam_loading_enabled=bool(getattr(vol_params, "beam_loading_enabled", False)),
        deflection_magnet_enabled=bool(getattr(vol_params, "deflection_enabled", False)),
        emission_iteration_enabled=bool(emission_iteration_enabled),
        flat_radius_mm=geometry.flat_radius_mm,
        bevel_width_mm=geometry.bevel_width_mm,
        bevel_angle_deg=geometry.bevel_angle_deg,
        cathode_length_mm=geometry.cathode_length_mm,
        insertion_offset_mm=geometry.insertion_offset_mm,
        accounting=accounting,
        provenance=provenance,
    )
