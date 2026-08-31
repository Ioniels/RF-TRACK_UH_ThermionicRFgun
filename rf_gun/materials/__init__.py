"""Selectable cathode-material property system (implementation plan Sec. 5).

Generic interfaces live in `base.py` and `registry.py` (material-agnostic); `lab6.py` composes
the LaB6-specific datasets in `data/` into the three named sets from plan Sec. 5.2 and registers
them here at import time so `rg.load_cathode_material("LaB6", ...)` works immediately after
`import rf_gun`.
"""
from __future__ import annotations

from .base import (
    COMPONENT_NAMES,
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
from .electron_range import tio_entrance_stopping_power_kev_per_um, tio_range_um
from .registry import (
    load_cathode_material,
    load_material_component,
    required_components_for,
    validate_material_for,
)
from .lab6 import register_lab6_materials

register_lab6_materials()

__all__ = [
    "COMPONENT_NAMES",
    "CathodeMaterialSet",
    "ElectronDepositionComponent",
    "EmissionComponent",
    "OpticalComponent",
    "PropertyReference",
    "ResolvedProperty",
    "ScalarPropertyDataset",
    "ThermalComponent",
    "load_property_dataset_yaml",
    "tio_range_um",
    "tio_entrance_stopping_power_kev_per_um",
    "load_cathode_material",
    "load_material_component",
    "required_components_for",
    "validate_material_for",
    "register_lab6_materials",
]
