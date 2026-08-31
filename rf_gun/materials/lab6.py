"""LaB6 cathode-material composition and model assembly (implementation plan Sec. 5.2/5.3),
registered into `rf_gun.materials.registry` at import time via `register_lab6_materials()`.

Assembles the three named complete sets from plan Sec. 5.2:
  * `LaB6_UH_recommended_v1` (default) -- the modern high-temperature composite, each property
    resolved from the best identified source;
  * `LaB6_Kowalczyk_PRSTAB120402_2014_legacy` -- closed-form equations reproducing the historical
    UH one-dimensional thermal calculation only;
  * `LaB6_constant_verification_v1` -- constant rho/cp/k for analytic Python-COMSOL tests, never
    a production physics set.

and registers the underlying paper-specific datasets (e.g. `LaB6_cp_Tanaka_OsakaThesis_1981`,
`LaB6_stopping_Bakr_PRSTAB060708_2011`) as standalone, directly-loadable partial components (plan
Sec. 5.1/5.2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .base import (
    CathodeMaterialSet,
    ElectronDepositionComponent,
    EmissionComponent,
    OpticalComponent,
    PropertyReference,
    ResolvedProperty,
    ScalarPropertyDataset,
    ThermalComponent,
    load_property_dataset_yaml,
)
from .registry import register_complete_material, register_partial_component

DATA_DIR = Path(__file__).parent / "data"

MATERIAL_ID = "LaB6"

# ---------------------------------------------------------------------------------------------
# Raw-YAML helpers for files that are NOT ScalarPropertyDataset-shaped (constants, composite
# manifests, closed-form-equation records) -- these are read directly rather than through
# base.load_property_dataset_yaml(), which expects a temperature_K/value_* (or csv_file) axis.
# ---------------------------------------------------------------------------------------------


def _read_yaml(filename: str) -> dict[str, Any]:
    path = DATA_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _reference_from_block(ref_raw: dict[str, Any]) -> PropertyReference:
    return PropertyReference(
        key=ref_raw["key"],
        title=ref_raw["title"],
        year=int(ref_raw["year"]),
        url=ref_raw.get("url"),
        notes=ref_raw.get("notes", ""),
    )


def _scalar_dataset(filename: str) -> ScalarPropertyDataset:
    return load_property_dataset_yaml(DATA_DIR / filename)


# ---------------------------------------------------------------------------------------------
# thermal.k composite: PCHIP over the concatenated Sun (<=1273 K, provisional) and Tanaka
# (1300-2000 K, provisional) points (plan Sec. 5.2: "Final PCHIP interpolation of digitized Sun
# data through 1273 K and Tanaka data from 1300-2000 K").
# ---------------------------------------------------------------------------------------------

_K_SUN_FILE = "LaB6_k_Sun_JNST2192728_2023.yaml"
_K_TANAKA_FILE = "LaB6_k_Tanaka_OsakaThesis_1981.yaml"


def _build_lab6_k_composite_dataset() -> ScalarPropertyDataset:
    """Stitch the Sun (<=1273 K) and Tanaka (1300-2000 K) provisional thermal-conductivity
    datasets into a single PCHIP fit over their concatenated points, per plan Sec. 5.2. Both
    constituent datasets are individually selectable via `load_material_component()`; this
    composite is what `LaB6_UH_recommended_v1`'s `thermal.k` actually evaluates.
    """
    sun = _scalar_dataset(_K_SUN_FILE)
    tanaka = _scalar_dataset(_K_TANAKA_FILE)

    x = np.concatenate([sun.x, tanaka.x])
    y = np.concatenate([sun.y, tanaka.y])
    order = np.argsort(x)
    x, y = x[order], y[order]
    if not np.all(np.diff(x) > 0):
        raise ValueError(
            "LaB6_k composite: concatenated Sun+Tanaka temperature axis is not strictly "
            f"monotonic increasing after sorting: {x.tolist()}"
        )

    reference = PropertyReference(
        key="Sun2023+Tanaka1981_composite",
        title=(
            "Composite of LaB6_k_Sun_JNST2192728_2023 (T<=1273 K) and "
            "LaB6_k_Tanaka_OsakaThesis_1981 (1300-2000 K); see those two files individually for "
            "full per-segment provenance."
        ),
        year=2023,
        url=None,
        notes="Both constituent segments are status=provisional_digitization -- see their own files.",
    )
    return ScalarPropertyDataset(
        dataset_id="LaB6_k_composite_Sun2023_Tanaka1981_v1",
        material=MATERIAL_ID,
        property="thermal_conductivity",
        quantity_symbol="k",
        independent_variable="temperature_K",
        x=x,
        y=y,
        unit_x="temperature_K",
        unit_y="W_m_K",
        interpolation="pchip",
        extrapolation="forbidden",
        status="provisional_digitization",
        uncertainty={"type": "model", "value": 0.15},
        reference=reference,
        notes=(
            "PCHIP fit over the concatenated Sun (<=1273 K, placeholder-digitized) and Tanaka "
            "(1300-2000 K, solver-development-values-only) segments. Both segments are "
            "provisional -- see LaB6_k_Sun_JNST2192728_2023.yaml/.csv and "
            "LaB6_k_Tanaka_OsakaThesis_1981.yaml for the full caveats before any production use."
        ),
    )


# ---------------------------------------------------------------------------------------------
# LaB6_Kowalczyk_PRSTAB120402_2014_legacy closed-form equations (plan Sec. 5.2). These are
# equations, not tabulated data, so they are plain Python functions rather than
# ScalarPropertyDataset instances; LaB6_Kowalczyk_PRSTAB120402_2014_legacy.yaml records the same
# equations as text plus the citation and check values used to validate them below.
# ---------------------------------------------------------------------------------------------

_LEGACY_RHO_KG_M3 = 4720.0


def _legacy_k_UH_W_m_K(T_K: Any) -> Any:
    """k_UH(T) = 418.4 * (116.7/T + 0.0366) W/m/K (plan Sec. 5.2)."""
    T = np.asarray(T_K, dtype=float)
    result = 418.4 * (116.7 / T + 0.0366)
    return result if T.ndim > 0 else float(result)


def _legacy_D_UH_m2_s(T_K: Any) -> Any:
    """D_UH(T) = 12.7675 * (T - 273.15)^(-0.640226) * 1e-4 m^2/s (plan Sec. 5.2)."""
    T = np.asarray(T_K, dtype=float)
    result = 12.7675 * (T - 273.15) ** (-0.640226) * 1.0e-4
    return result if T.ndim > 0 else float(result)


def _legacy_cp_UH_J_kg_K(T_K: Any) -> Any:
    """cp_UH(T) = k_UH(T) / (rho * D_UH(T)), rho=4720 kg/m3 (plan Sec. 5.2, C_V,UH = k_UH/D_UH)."""
    T = np.asarray(T_K, dtype=float)
    k = _legacy_k_UH_W_m_K(T)
    D = _legacy_D_UH_m2_s(T)
    result = k / (_LEGACY_RHO_KG_M3 * D)
    return result if T.ndim > 0 else float(result)


def _legacy_rho_kg_m3(T_K: Any) -> Any:
    """Constant rho=4720 kg/m3, broadcast to the shape of T_K (the legacy set carries no
    temperature-dependent density correction)."""
    T = np.asarray(T_K, dtype=float)
    result = T * 0.0 + _LEGACY_RHO_KG_M3
    return result if T.ndim > 0 else float(result)


# ---------------------------------------------------------------------------------------------
# Complete-material loaders
# ---------------------------------------------------------------------------------------------


def _load_LaB6_UH_recommended_v1() -> CathodeMaterialSet:
    manifest = _read_yaml("LaB6_UH_recommended_v1.yaml")

    density_raw = _read_yaml("LaB6_density_Ivashchenko_PhysicaB531_2018.yaml")
    density_ref = _reference_from_block(density_raw["reference"])
    rho0 = ResolvedProperty(
        evaluator=float(density_raw["value_kg_m3"]),
        dataset_id=density_raw["dataset_id"],
        reference_key=density_ref.key,
        unit="kg_m3",
        status=density_raw["status"],
    )

    expansion_raw = _read_yaml("LaB6_expansion_Williams_LBL27907_1989.yaml")
    expansion_ref = _reference_from_block(expansion_raw["reference"])
    expansion = ResolvedProperty(
        evaluator=float(expansion_raw["value_per_K"]),
        dataset_id=expansion_raw["dataset_id"],
        reference_key=expansion_ref.key,
        unit="per_K",
        status=expansion_raw["status"],
    )

    cp_dataset = _scalar_dataset("LaB6_cp_Tanaka_OsakaThesis_1981.yaml")
    cp = ResolvedProperty(
        evaluator=cp_dataset,
        dataset_id=cp_dataset.dataset_id,
        reference_key=cp_dataset.reference.key,
        unit=cp_dataset.unit_y,
        status=cp_dataset.status,
    )

    k_dataset = _build_lab6_k_composite_dataset()
    k = ResolvedProperty(
        evaluator=k_dataset,
        dataset_id=k_dataset.dataset_id,
        reference_key=k_dataset.reference.key,
        unit=k_dataset.unit_y,
        status=k_dataset.status,
    )

    thermal = ThermalComponent(rho0=rho0, cp=cp, k=k, expansion=expansion)

    stopping_raw = _read_yaml("LaB6_stopping_Bakr_PRSTAB060708_2011.yaml")
    stopping_ref = _reference_from_block(stopping_raw["reference"])
    electron_deposition = ElectronDepositionComponent(
        Z_eff=float(stopping_raw["Z_eff"]),
        A_eff_g_mol=float(stopping_raw["A_eff_g_mol"]),
        rho0_kg_m3=float(stopping_raw["rho0_kg_m3"]),
        dataset_id=stopping_raw["dataset_id"],
        reference_key=stopping_ref.key,
        notes=stopping_raw["notes"],
    )

    # Bookkeeping only (plan Sec. 5.4): the real emission model/parameters come from the
    # originating run_config.json in load_run mode, not from this material file.
    emission = EmissionComponent(
        work_function_model_id="inherited_from_run_config",
        reference_key="inherited_from_run_config",
        dataset_id="inherited_from_run_config",
        phi_fit_eV=None,
        A_R_A_cm2_K2=None,
        notes=(
            "LaB6_UH_recommended_v1 does not fix an emission model: plan Sec. 5.4 states the "
            "material selection 'records, but does not duplicate, the resolved emission "
            "inputs' -- in load_run mode these are inherited from the originating "
            "run_config.json. An explicit override is only valid if the run manifest labels "
            "it as a new material/emission scenario."
        ),
    )

    optical_raw = _read_yaml("LaB6_optical_Kowalczyk_IJT035-1538_2014.yaml")
    optical_ref = _reference_from_block(optical_raw["reference"])
    optical = OpticalComponent(
        spectral_anchors_nm={float(k): float(v) for k, v in optical_raw["anchors_nm"].items()},
        dataset_id=optical_raw["dataset_id"],
        reference_key=optical_ref.key,
        notes=optical_raw["notes"],
    )

    return CathodeMaterialSet(
        material_id=MATERIAL_ID,
        property_set="LaB6_UH_recommended_v1",
        thermal=thermal,
        electron_deposition=electron_deposition,
        emission=emission,
        optical=optical,
    )


def _load_LaB6_Kowalczyk_PRSTAB120402_2014_legacy() -> CathodeMaterialSet:
    raw = _read_yaml("LaB6_Kowalczyk_PRSTAB120402_2014_legacy.yaml")
    ref = _reference_from_block(raw["reference"])
    dataset_id = raw["dataset_id"]

    rho0 = ResolvedProperty(
        evaluator=_legacy_rho_kg_m3,
        dataset_id=dataset_id,
        reference_key=ref.key,
        unit="kg_m3",
        status=raw["status"],
    )
    cp = ResolvedProperty(
        evaluator=_legacy_cp_UH_J_kg_K,
        dataset_id=dataset_id,
        reference_key=ref.key,
        unit="J_kg_K",
        status=raw["status"],
    )
    k = ResolvedProperty(
        evaluator=_legacy_k_UH_W_m_K,
        dataset_id=dataset_id,
        reference_key=ref.key,
        unit="W_m_K",
        status=raw["status"],
    )
    thermal = ThermalComponent(rho0=rho0, cp=cp, k=k, expansion=None)

    emission_fit = raw["emission_legacy_fit"]
    emission = EmissionComponent(
        work_function_model_id="Kowalczyk_PRSTAB120402_2014_legacy_fit",
        reference_key=ref.key,
        dataset_id=dataset_id,
        phi_fit_eV=float(emission_fit["phi_fit_eV"]),
        A_R_A_cm2_K2=float(emission_fit["A_R_A_cm2_K2"]),
        notes=(
            "Fitted values reproducing the reduced Kowalczyk 2014 legacy model only (plan "
            "Sec. 5.4); never silently substituted for a production run's selected "
            "work-function/emission configuration."
        ),
    )

    # The legacy paper supplies no LaB6 composition/TIO-stopping data or spectral optical
    # anchors -- electron_deposition and optical are intentionally None so
    # registry.validate_material_for() raises a clear error if an operation needing them (e.g.
    # bb0_deposition, uh_optical_diagnostic) is requested against this benchmark-only set.
    return CathodeMaterialSet(
        material_id=MATERIAL_ID,
        property_set="LaB6_Kowalczyk_PRSTAB120402_2014_legacy",
        thermal=thermal,
        electron_deposition=None,
        emission=emission,
        optical=None,
    )


def _load_LaB6_constant_verification_v1() -> CathodeMaterialSet:
    raw = _read_yaml("LaB6_constant_verification_v1.yaml")
    ref = _reference_from_block(raw["reference"])
    dataset_id = raw["dataset_id"]
    values = raw["values"]

    rho0 = ResolvedProperty(
        evaluator=float(values["rho_kg_m3"]),
        dataset_id=dataset_id,
        reference_key=ref.key,
        unit="kg_m3",
        status=raw["status"],
    )
    cp = ResolvedProperty(
        evaluator=float(values["cp_J_kg_K"]),
        dataset_id=dataset_id,
        reference_key=ref.key,
        unit="J_kg_K",
        status=raw["status"],
    )
    k = ResolvedProperty(
        evaluator=float(values["k_W_m_K"]),
        dataset_id=dataset_id,
        reference_key=ref.key,
        unit="W_m_K",
        status=raw["status"],
    )
    thermal = ThermalComponent(rho0=rho0, cp=cp, k=k, expansion=None)

    return CathodeMaterialSet(
        material_id=MATERIAL_ID,
        property_set="LaB6_constant_verification_v1",
        thermal=thermal,
        electron_deposition=None,
        emission=None,
        optical=None,
    )


# ---------------------------------------------------------------------------------------------
# Standalone paper-specific components (plan Sec. 5.1: "also directly selectable as components"),
# refused as complete materials by registry.load_cathode_material().
# ---------------------------------------------------------------------------------------------


def _load_component_cp_tanaka() -> ScalarPropertyDataset:
    return _scalar_dataset("LaB6_cp_Tanaka_OsakaThesis_1981.yaml")


def _load_component_k_sun() -> ScalarPropertyDataset:
    return _scalar_dataset("LaB6_k_Sun_JNST2192728_2023.yaml")


def _load_component_k_tanaka() -> ScalarPropertyDataset:
    return _scalar_dataset("LaB6_k_Tanaka_OsakaThesis_1981.yaml")


def _load_component_stopping_bakr() -> ElectronDepositionComponent:
    raw = _read_yaml("LaB6_stopping_Bakr_PRSTAB060708_2011.yaml")
    ref = _reference_from_block(raw["reference"])
    return ElectronDepositionComponent(
        Z_eff=float(raw["Z_eff"]),
        A_eff_g_mol=float(raw["A_eff_g_mol"]),
        rho0_kg_m3=float(raw["rho0_kg_m3"]),
        dataset_id=raw["dataset_id"],
        reference_key=ref.key,
        notes=raw["notes"],
    )


def _load_component_density() -> ResolvedProperty:
    raw = _read_yaml("LaB6_density_Ivashchenko_PhysicaB531_2018.yaml")
    ref = _reference_from_block(raw["reference"])
    return ResolvedProperty(
        evaluator=float(raw["value_kg_m3"]),
        dataset_id=raw["dataset_id"],
        reference_key=ref.key,
        unit="kg_m3",
        status=raw["status"],
    )


def _load_component_expansion() -> ResolvedProperty:
    raw = _read_yaml("LaB6_expansion_Williams_LBL27907_1989.yaml")
    ref = _reference_from_block(raw["reference"])
    return ResolvedProperty(
        evaluator=float(raw["value_per_K"]),
        dataset_id=raw["dataset_id"],
        reference_key=ref.key,
        unit="per_K",
        status=raw["status"],
    )


def _load_component_optical() -> OpticalComponent:
    raw = _read_yaml("LaB6_optical_Kowalczyk_IJT035-1538_2014.yaml")
    ref = _reference_from_block(raw["reference"])
    return OpticalComponent(
        spectral_anchors_nm={float(k): float(v) for k, v in raw["anchors_nm"].items()},
        dataset_id=raw["dataset_id"],
        reference_key=ref.key,
        notes=raw["notes"],
    )


def register_lab6_materials() -> None:
    """Register the three complete LaB6 material sets and the standalone paper-specific
    components with `rf_gun.materials.registry`. Called once from
    `rf_gun/materials/__init__.py` at import time; idempotent (re-registering the same names is
    harmless, just overwrites the loader with an identical one).
    """
    register_complete_material(MATERIAL_ID, "LaB6_UH_recommended_v1", _load_LaB6_UH_recommended_v1)
    register_complete_material(
        MATERIAL_ID,
        "LaB6_Kowalczyk_PRSTAB120402_2014_legacy",
        _load_LaB6_Kowalczyk_PRSTAB120402_2014_legacy,
    )
    register_complete_material(
        MATERIAL_ID, "LaB6_constant_verification_v1", _load_LaB6_constant_verification_v1
    )

    register_partial_component(
        MATERIAL_ID, "LaB6_cp_Tanaka_OsakaThesis_1981", _load_component_cp_tanaka
    )
    register_partial_component(MATERIAL_ID, "LaB6_k_Sun_JNST2192728_2023", _load_component_k_sun)
    register_partial_component(
        MATERIAL_ID, "LaB6_k_Tanaka_OsakaThesis_1981", _load_component_k_tanaka
    )
    register_partial_component(
        MATERIAL_ID, "LaB6_stopping_Bakr_PRSTAB060708_2011", _load_component_stopping_bakr
    )
    register_partial_component(
        MATERIAL_ID, "LaB6_density_Ivashchenko_PhysicaB531_2018", _load_component_density
    )
    register_partial_component(
        MATERIAL_ID, "LaB6_expansion_Williams_LBL27907_1989", _load_component_expansion
    )
    register_partial_component(
        MATERIAL_ID, "LaB6_optical_Kowalczyk_IJT035-1538_2014", _load_component_optical
    )
