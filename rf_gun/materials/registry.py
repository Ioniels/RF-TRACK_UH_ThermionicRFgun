"""Validated `material_id` + `property_set` -> `CathodeMaterialSet` lookup (plan Sec. 5.1, 5.4).

Two kinds of names are registered per material:
  * "complete" property sets, loadable via `load_cathode_material()`, which resolve all four
    components a downstream operation might need (some components may still be `None` if that
    named set genuinely omits them, e.g. the legacy Kowalczyk set has no optical data -- callers
    must use `validate_material_for()` before relying on a specific component);
  * "partial" (paper-specific) components, e.g. `LaB6_cp_Tanaka_OsakaThesis_1981`, which are
    directly selectable as a single dataset (plan Sec. 5.1: "Paper-specific datasets ... are also
    directly selectable as components") via `load_material_component()`, but are refused as a
    complete material by `load_cathode_material()` (plan Sec. 5.2: "The registry refuses to
    promote a paper-specific component to a complete material if that paper does not provide all
    properties needed by the requested model").

This module is itself material-agnostic; `lab6.py` calls `register_complete_material()` /
`register_partial_component()` at import time (via `register_lab6_materials()`) to populate it.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import CathodeMaterialSet

# material_id -> {property_set_name: zero-arg loader}
_MATERIAL_REGISTRY: dict[str, dict[str, Callable[[], CathodeMaterialSet]]] = {}

# material_id -> {component_dataset_name: zero-arg loader} for standalone paper-specific datasets
_COMPONENT_REGISTRY: dict[str, dict[str, Callable[[], Any]]] = {}

# property_set names that are intentionally partial (paper-specific) and must never be resolved
# as a complete material, even if a caller asks for them by name via load_cathode_material().
_PARTIAL_SET_NAMES: set[str] = set()

#: Operations `python_xy_layered`/`bb0_deposition`/etc. and the CathodeMaterialSet component
#: name(s) each requires (plan Sec. 5.4 "Required component checks are model-dependent" table).
_REQUIRED_COMPONENTS: dict[str, frozenset[str]] = {
    "bb_event_analysis": frozenset(),  # particle mass/charge and geometry only; no material data
    "bb0_deposition": frozenset({"thermal", "electron_deposition"}),
    "python_xy_layered": frozenset({"thermal"}),
    "comsol_export": frozenset({"thermal", "electron_deposition"}),
    "temperature_to_emission_feedback": frozenset({"thermal", "electron_deposition", "emission"}),
    "uh_optical_diagnostic": frozenset({"optical"}),
}


def register_complete_material(
    material_id: str, property_set: str, loader: Callable[[], CathodeMaterialSet]
) -> None:
    """Register a zero-argument `loader` producing a complete `CathodeMaterialSet` for
    `(material_id, property_set)`."""
    _MATERIAL_REGISTRY.setdefault(material_id, {})[property_set] = loader


def register_partial_component(
    material_id: str, name: str, loader: Callable[[], Any]
) -> None:
    """Register a standalone paper-specific dataset `name` (e.g.
    `LaB6_cp_Tanaka_OsakaThesis_1981`) as directly loadable via `load_material_component()`, and
    mark it as refused when requested as a complete material via `load_cathode_material()`."""
    _COMPONENT_REGISTRY.setdefault(material_id, {})[name] = loader
    _PARTIAL_SET_NAMES.add(name)


def load_cathode_material(material_id: str, property_set: str) -> CathodeMaterialSet:
    """Public entry point (plan Sec. 5.1)::

        material = rg.load_cathode_material(material_id="LaB6", property_set="LaB6_UH_recommended_v1")

    Raises `ValueError` for an unknown `material_id`, listing valid material IDs. Raises
    `ValueError` for an unknown `property_set`, listing the valid complete property sets for that
    material. Raises `ValueError` -- distinctly worded -- if `property_set` names a registered
    paper-specific partial component instead of a complete material (plan Sec. 5.2's "refuses to
    promote" contract): that dataset must instead be loaded via `load_material_component()`.
    """
    if material_id not in _MATERIAL_REGISTRY:
        raise ValueError(
            f"Unknown material_id {material_id!r}; valid material_id(s): "
            f"{sorted(_MATERIAL_REGISTRY)}"
        )
    complete_sets = _MATERIAL_REGISTRY[material_id]

    if property_set in complete_sets:
        return complete_sets[property_set]()

    if property_set in _PARTIAL_SET_NAMES:
        raise ValueError(
            f"{property_set!r} is a paper-specific, intentionally partial dataset for "
            f"{material_id!r} (it does not supply every property a complete material set needs). "
            f"The registry refuses to promote it to a complete material (plan Sec. 5.2) -- load it "
            f"as a single component instead via load_material_component({material_id!r}, "
            f"{property_set!r}), or select one of the complete sets: {sorted(complete_sets)}"
        )

    raise ValueError(
        f"Unknown property_set {property_set!r} for material_id {material_id!r}; valid "
        f"complete property_set(s): {sorted(complete_sets)}"
    )


def load_material_component(material_id: str, name: str) -> Any:
    """Load a single paper-specific dataset/component directly (plan Sec. 5.1: "Paper-specific
    datasets ... are also directly selectable as components"), bypassing the complete-material
    completeness check.

    Raises `ValueError` for an unknown `material_id` or an unknown component `name`.
    """
    if material_id not in _COMPONENT_REGISTRY:
        raise ValueError(
            f"Unknown material_id {material_id!r} (no standalone components registered); valid "
            f"material_id(s): {sorted(_COMPONENT_REGISTRY)}"
        )
    components = _COMPONENT_REGISTRY[material_id]
    if name not in components:
        raise ValueError(
            f"Unknown component/dataset {name!r} for material_id {material_id!r}; valid "
            f"component(s): {sorted(components)}"
        )
    return components[name]()


def required_components_for(operation: str) -> frozenset[str]:
    """Return the set of `CathodeMaterialSet` component name(s) (a subset of
    `rf_gun.materials.base.COMPONENT_NAMES`) required for `operation`, per plan Sec. 5.4's
    "Required component checks are model-dependent" table. Raises `ValueError` for an unrecognized
    `operation`, listing the valid operation names.
    """
    if operation not in _REQUIRED_COMPONENTS:
        raise ValueError(
            f"Unknown operation {operation!r}; valid operation(s): "
            f"{sorted(_REQUIRED_COMPONENTS)}"
        )
    return _REQUIRED_COMPONENTS[operation]


def validate_material_for(material: CathodeMaterialSet, operation: str) -> None:
    """Raise `ValueError` naming the missing component(s) if `material` lacks any component
    `operation` requires (plan Sec. 5.4). No-op if `material` supplies everything `operation`
    needs.
    """
    required = required_components_for(operation)
    missing = [name for name in required if getattr(material, name) is None]
    if missing:
        raise ValueError(
            f"Material property_set={material.property_set!r} (material_id="
            f"{material.material_id!r}) is missing required component(s) {missing} for "
            f"operation {operation!r}; it cannot be used for this operation. Required "
            f"component(s) for {operation!r}: {sorted(required)}."
        )
