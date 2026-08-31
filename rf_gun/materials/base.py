"""Generic, material-agnostic cathode-property interfaces (implementation plan Sec. 5,
"Selectable cathode-material property system").

This module knows nothing about LaB6 specifically -- it defines the on-disk schema for one
tabulated/interpolated scalar property versus one independent variable (`ScalarPropertyDataset`
plus its YAML/CSV loader `load_property_dataset_yaml`), the citation record every dataset must
carry (`PropertyReference`), and the four-component container a complete cathode material
resolves into (`CathodeMaterialSet`: thermal, electron_deposition, emission, optical). A later
dispenser cathode or other thermionic material reuses all of this unchanged; only
`rf_gun/materials/lab6.py` (and a future `rf_gun/materials/<other_material>.py`) is
material-specific.

Design goals directly from plan Sec. 5.1:
  * "Each YAML/JSON data file contains only values, units, interpolation rules, validity ranges,
    uncertainties, and complete references. Python code provides validation and interpolation."
  * "The resolved run manifest records the composite set name and the source key used for each
    individual property. Selecting a paper-specific set with a missing required property raises
    an error; values are never silently borrowed from the default composite set."
  * "Use shape-preserving PCHIP interpolation for tabulated scalar properties. Extrapolation is
    disabled by default."

Every resolved property therefore carries both an evaluator (a `ScalarPropertyDataset`, a plain
constant, or -- for the closed-form legacy set -- a Python callable) *and* the dataset/source key
that produced it, via `ResolvedProperty`, so `CathodeMaterialSet.to_manifest_dict()` can record
exactly which sources backed a given run (matching the provenance-recording style of
`rf_gun/io.py`'s `to_json_safe`/run-manifest helpers).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.interpolate import PchipInterpolator

#: The four independently selectable components of a complete cathode material (plan Sec. 5.1).
COMPONENT_NAMES: tuple[str, ...] = ("thermal", "electron_deposition", "emission", "optical")


@dataclass(frozen=True)
class PropertyReference:
    """A citation record attached to every dataset (plan Sec. 5.1: "complete references").

    `key` is the short attribution key used throughout this package (e.g.
    `"Tanaka_OsakaThesis_1981"`); `title`/`year`/`url` are for human-readable provenance in plots
    and manifests. `url` is optional (many of the source PDFs in `manual_references/` have no
    stable public URL). `notes` carries anything a reader needs before trusting the number, e.g. a
    co-source attribution correction or a known discrepancy against another paper's stated value.
    """

    key: str
    title: str
    year: int
    url: str | None = None
    notes: str = ""

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "year": int(self.year),
            "url": self.url,
            "notes": self.notes,
        }


@dataclass
class ScalarPropertyDataset:
    """One tabulated/interpolated scalar material property versus one independent variable
    (almost always temperature), transcribed from a single YAML file per plan Sec. 5.1's example
    schema.

    `x`/`y` are the raw digitized/tabulated points (validated strictly monotonic increasing in
    `x`, all-finite, by `load_property_dataset_yaml`). `evaluate(T)` shape-preserves the input
    (scalar in, scalar out; array in, array out), matching the convention already used throughout
    `rf_gun` (e.g. `work_function_models.phi_eff_constant`).
    """

    dataset_id: str
    material: str
    property: str
    quantity_symbol: str
    independent_variable: str
    x: np.ndarray
    y: np.ndarray
    unit_x: str
    unit_y: str
    interpolation: str
    extrapolation: str
    status: str
    uncertainty: dict[str, Any]
    reference: PropertyReference
    notes: str = ""

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"{self.dataset_id}: x/y shape mismatch {self.x.shape} vs {self.y.shape}"
            )

    def evaluate(self, value: Any) -> Any:
        """Evaluate the property at `value` (scalar or array-like, in `unit_x`).

        Raises `ValueError` if `extrapolation == 'forbidden'` and ANY query point lies outside
        `[x.min(), x.max()]` -- the check is vectorized (all points checked before raising) and
        the message reports which value(s) were out of range and the valid range, per plan
        Sec. 5.1's "extrapolation is disabled by default." Raises `NotImplementedError` for any
        `interpolation` mode other than `"pchip"` -- no silent fallback to e.g. linear
        interpolation.
        """
        original = np.asarray(value, dtype=float)
        T = np.atleast_1d(original)

        lo = float(self.x.min())
        hi = float(self.x.max())
        if self.extrapolation == "forbidden":
            out_of_range = (T < lo) | (T > hi)
            if np.any(out_of_range):
                bad = np.unique(T[out_of_range]).tolist()
                raise ValueError(
                    f"{self.dataset_id}: query value(s) {bad} {self.unit_x} lie outside the "
                    f"valid range [{lo}, {hi}] {self.unit_x} and extrapolation is forbidden for "
                    f"this dataset (status={self.status!r})."
                )
        elif self.extrapolation == "clamp":
            T = np.clip(T, lo, hi)
        else:
            raise NotImplementedError(
                f"{self.dataset_id}: extrapolation mode {self.extrapolation!r} is not "
                f"implemented; only 'forbidden' and 'clamp' are supported."
            )

        if self.interpolation != "pchip":
            raise NotImplementedError(
                f"{self.dataset_id}: interpolation mode {self.interpolation!r} is not "
                f"implemented; only 'pchip' is currently supported (shape-preserving PCHIP per "
                f"plan Sec. 5.2)."
            )
        interpolator = PchipInterpolator(self.x, self.y, extrapolate=False)
        result = interpolator(T)

        return result if original.ndim > 0 else float(result[0])

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "material": self.material,
            "property": self.property,
            "quantity_symbol": self.quantity_symbol,
            "independent_variable": self.independent_variable,
            "unit_x": self.unit_x,
            "unit_y": self.unit_y,
            "interpolation": self.interpolation,
            "extrapolation": self.extrapolation,
            "status": self.status,
            "uncertainty": dict(self.uncertainty),
            "reference": self.reference.to_manifest_dict(),
            "notes": self.notes,
            "x_range": [float(self.x.min()), float(self.x.max())],
        }


def load_property_dataset_yaml(path: Path) -> ScalarPropertyDataset:
    """Load and validate one `ScalarPropertyDataset` from a YAML file following plan Sec. 5.1's
    schema, e.g.::

        dataset_id: LaB6_cp_Tanaka_OsakaThesis_1981
        material: LaB6
        property: specific_heat_capacity
        quantity_symbol: cp
        independent_variable: temperature_K
        temperature_K: [1000, 1200, 1400, 1600, 1800, 2000]
        value_J_kg_K: [801, 824, 839, 850, 858, 865]
        interpolation: pchip
        extrapolation: forbidden
        status: cp_approximated_by_cV
        uncertainty: {type: relative_model, value: 0.03}
        reference: {key: ..., title: ..., year: 1981, url: ...}
        notes: ...

    Validates: required keys present; the independent-variable axis is strictly monotonic
    increasing; all x/y values are finite; x and y have matching length (at least 2 points).

    For large digitized curves the YAML instead points at a neighboring CSV via a `csv_file:
    <relative filename>` key (plan Sec. 5.1: "the YAML metadata points to a neighboring CSV with
    columns such as temperature_K,value,uncertainty,source_point_id"); this loader reads
    `temperature_K` and `value` from that CSV for x/y (uncertainty/source_point_id columns, if
    present, are informational only -- the dataset's aggregate `uncertainty` block still comes
    from the YAML itself, since `ScalarPropertyDataset.uncertainty` is a single dict, not a
    per-point array). When `csv_file` is used, the YAML must also carry a `value_unit` key (there
    is no `value_<unit>` array key to infer the unit from).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    required_keys = [
        "dataset_id",
        "material",
        "property",
        "quantity_symbol",
        "independent_variable",
        "interpolation",
        "extrapolation",
        "status",
        "uncertainty",
        "reference",
    ]
    missing = [k for k in required_keys if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")

    indep_key = raw["independent_variable"]
    unit_y: str | None

    if "csv_file" in raw:
        csv_path = path.parent / raw["csv_file"]
        if not csv_path.is_file():
            raise ValueError(f"{path}: csv_file {raw['csv_file']!r} not found at {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as cf:
            reader = csv.DictReader(cf)
            fieldnames = reader.fieldnames or []
            if "temperature_K" not in fieldnames or "value" not in fieldnames:
                raise ValueError(
                    f"{csv_path}: expected at least columns temperature_K,value "
                    f"(got {fieldnames})"
                )
            rows = sorted(reader, key=lambda r: float(r["temperature_K"]))
        x = np.array([float(r["temperature_K"]) for r in rows], dtype=float)
        y = np.array([float(r["value"]) for r in rows], dtype=float)
        if "value_unit" not in raw:
            raise ValueError(
                f"{path}: a 'value_unit' key is required when using csv_file (no value_<unit> "
                f"array key is present to infer it from)"
            )
        unit_y = raw["value_unit"]
    else:
        if indep_key not in raw:
            raise ValueError(
                f"{path}: independent_variable key {indep_key!r} not found among top-level keys"
            )
        x = np.array(raw[indep_key], dtype=float)
        value_keys = [k for k in raw if k.startswith("value_")]
        if len(value_keys) != 1:
            raise ValueError(
                f"{path}: expected exactly one 'value_<unit>' key, found {value_keys}"
            )
        value_key = value_keys[0]
        y = np.array(raw[value_key], dtype=float)
        unit_y = raw.get("unit_y", value_key[len("value_") :])

    if x.size < 2:
        raise ValueError(f"{path}: need at least 2 points, got {x.size}")
    if x.shape != y.shape:
        raise ValueError(f"{path}: x/y shape mismatch {x.shape} vs {y.shape}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{path}: dataset contains non-finite value(s)")
    if not np.all(np.diff(x) > 0):
        raise ValueError(
            f"{path}: independent-variable axis must be strictly monotonic increasing, "
            f"got {x.tolist()}"
        )

    ref_raw = raw["reference"]
    for ref_key in ("key", "title", "year"):
        if ref_key not in ref_raw:
            raise ValueError(f"{path}: reference block missing required key {ref_key!r}")
    reference = PropertyReference(
        key=ref_raw["key"],
        title=ref_raw["title"],
        year=int(ref_raw["year"]),
        url=ref_raw.get("url"),
        notes=ref_raw.get("notes", ""),
    )

    unit_x = raw.get("unit_x", indep_key)

    return ScalarPropertyDataset(
        dataset_id=raw["dataset_id"],
        material=raw["material"],
        property=raw["property"],
        quantity_symbol=raw["quantity_symbol"],
        independent_variable=indep_key,
        x=x,
        y=y,
        unit_x=unit_x,
        unit_y=unit_y,
        interpolation=raw["interpolation"],
        extrapolation=raw["extrapolation"],
        status=raw["status"],
        uncertainty=dict(raw["uncertainty"]),
        reference=reference,
        notes=raw.get("notes", ""),
    )


PropertyEvaluator = "ScalarPropertyDataset | float | Callable[[Any], Any]"


@dataclass
class ResolvedProperty:
    """One resolved scalar property plus the provenance needed to record it in a run manifest
    (plan Sec. 5.1: "the source key used for each individual property").

    `evaluator` is one of:
      * a `ScalarPropertyDataset` (tabulated/PCHIP-interpolated property), or
      * a plain constant `float` (e.g. a reference density with no T-dependence enabled), or
      * a Python callable `T -> value` (used only for the closed-form legacy equations in
        `lab6.py`, which are equations, not tabulated data).
    """

    evaluator: Any
    dataset_id: str
    reference_key: str
    unit: str
    status: str = ""

    def evaluate(self, T: Any) -> Any:
        if isinstance(self.evaluator, ScalarPropertyDataset):
            return self.evaluator.evaluate(T)
        if callable(self.evaluator):
            return self.evaluator(T)
        T_arr = np.asarray(T, dtype=float)
        result = T_arr * 0.0 + float(self.evaluator)
        return result if T_arr.ndim > 0 else float(result)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "reference_key": self.reference_key,
            "unit": self.unit,
            "status": self.status,
        }


@dataclass
class ThermalComponent:
    """`rho(T)`, `cp(T)`, `k(T)`, and an optional thermal-expansion correction (plan Sec. 5.1).

    `rho0_kg_m3` is the reference (typically room/near-room-temperature) density; `expansion`, if
    present, enables the optional correction
    `rho(T) = rho(T0) * exp[-3 * integral(alpha_L(T') dT', T0, T)]` (plan Sec. 5.2), which is
    disabled by default -- `rho_kg_m3()` only applies it when `apply_expansion=True` is passed
    explicitly, since selecting a thermal set does not itself imply the expansion correction is
    wanted.
    """

    rho0: ResolvedProperty
    cp: ResolvedProperty
    k: ResolvedProperty
    expansion: ResolvedProperty | None = None
    T0_K: float = 293.15

    def rho_kg_m3(self, T: Any, *, apply_expansion: bool = False) -> Any:
        T_arr = np.asarray(T, dtype=float)
        rho0_val = float(self.rho0.evaluate(self.T0_K))
        if not apply_expansion or self.expansion is None:
            result = T_arr * 0.0 + rho0_val
            return result if T_arr.ndim > 0 else float(result)
        alpha_arr = np.asarray(self.expansion.evaluate(T), dtype=float)
        result = rho0_val * np.exp(-3.0 * alpha_arr * (T_arr - self.T0_K))
        return result if T_arr.ndim > 0 else float(result)

    def cp_J_kg_K(self, T: Any) -> Any:
        return self.cp.evaluate(T)

    def k_W_m_K(self, T: Any) -> Any:
        return self.k.evaluate(T)

    def to_manifest_dict(self) -> dict[str, Any]:
        out = {
            "rho0": self.rho0.to_manifest_dict(),
            "cp": self.cp.to_manifest_dict(),
            "k": self.k.to_manifest_dict(),
        }
        if self.expansion is not None:
            out["expansion"] = self.expansion.to_manifest_dict()
        return out


@dataclass
class ElectronDepositionComponent:
    """Composition and TIO range/stopping-power model for back-bombardment energy deposition
    (plan Sec. 5.1/5.3). The actual range law lives in `electron_range.tio_range_um`; this
    component only carries the composition (`Z_eff`, `A_eff_g_mol`) and reference density used to
    evaluate it, plus provenance.
    """

    Z_eff: float
    A_eff_g_mol: float
    rho0_kg_m3: float
    dataset_id: str
    reference_key: str
    notes: str = ""

    def range_um(self, kinetic_energy_keV: Any, rho_kg_m3: float | None = None) -> Any:
        from .electron_range import tio_range_um

        rho = self.rho0_kg_m3 if rho_kg_m3 is None else float(rho_kg_m3)
        return tio_range_um(kinetic_energy_keV, self.Z_eff, self.A_eff_g_mol, rho)

    def entrance_stopping_power_keV_per_um(
        self, kinetic_energy_keV: Any, rho_kg_m3: float | None = None
    ) -> Any:
        from .electron_range import tio_entrance_stopping_power_kev_per_um

        rho = self.rho0_kg_m3 if rho_kg_m3 is None else float(rho_kg_m3)
        return tio_entrance_stopping_power_kev_per_um(
            kinetic_energy_keV, self.Z_eff, self.A_eff_g_mol, rho
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "reference_key": self.reference_key,
            "Z_eff": float(self.Z_eff),
            "A_eff_g_mol": float(self.A_eff_g_mol),
            "rho0_kg_m3": float(self.rho0_kg_m3),
            "notes": self.notes,
        }


@dataclass
class EmissionComponent:
    """Bookkeeping/provenance only (plan Sec. 5.4): "Thermal feedback changes current only
    through the same emission model used by the production simulation. The material selection
    therefore records, but does not duplicate, the resolved emission inputs." The actual emission
    physics stays in `rf_gun.emission_models`/`rf_gun.work_function_models`.
    """

    work_function_model_id: str
    reference_key: str
    dataset_id: str
    phi_fit_eV: float | None = None
    A_R_A_cm2_K2: float | None = None
    notes: str = ""

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "reference_key": self.reference_key,
            "work_function_model_id": self.work_function_model_id,
            "phi_fit_eV": self.phi_fit_eV,
            "A_R_A_cm2_K2": self.A_R_A_cm2_K2,
            "notes": self.notes,
        }


@dataclass
class OpticalComponent:
    """Spectral emissivity/absorptivity anchors for radiation and the synthetic UH optical
    diagnostic (plan Sec. 5.1/5.5). `spectral_anchors_nm` maps wavelength [nm] -> dimensionless
    emissivity/absorptivity; per plan Sec. 5.5 these anchors must NOT be substituted directly for
    total hemispherical emissivity.
    """

    spectral_anchors_nm: dict[float, float]
    dataset_id: str
    reference_key: str
    notes: str = ""

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "reference_key": self.reference_key,
            "spectral_anchors_nm": {str(k): v for k, v in self.spectral_anchors_nm.items()},
            "notes": self.notes,
        }


@dataclass
class CathodeMaterialSet:
    """The four independently selectable cathode-material components (plan Sec. 5.1), resolved
    for one `material_id` + `property_set` name.

    A component is `None` when the selected `property_set` does not supply it (e.g. the legacy
    Kowalczyk set has no electron_deposition/optical data) -- callers must go through
    `registry.validate_material_for()` before relying on a component so a missing one raises a
    clear error instead of an `AttributeError` deep inside a solver.
    """

    material_id: str
    property_set: str
    thermal: ThermalComponent | None = None
    electron_deposition: ElectronDepositionComponent | None = None
    emission: EmissionComponent | None = None
    optical: OpticalComponent | None = None

    def to_manifest_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "material_id": self.material_id,
            "property_set": self.property_set,
        }
        for name in COMPONENT_NAMES:
            component = getattr(self, name)
            out[name] = component.to_manifest_dict() if component is not None else None
        return out
