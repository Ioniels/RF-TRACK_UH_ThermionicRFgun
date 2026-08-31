"""High-level study orchestration (implementation plan Sec. 10.1/10.2; addendum Sec. 19.2).

`rf_gun.studies.back_bombardment_macropulse` wires together the material registry, BB0 deposition,
the configurable macropulse model (`rf_gun.macropulse`), the `python_xy_layered` thermal solver, and
the (interface-only, per addendum Sec. 19.2) COMSOL comparison into one shared function the
notebook and CLI entry points both call (plan Sec. 1: "The notebook and batch/SLURM paths call the
same library functions and construct the same dataclasses.").
"""
from __future__ import annotations

from .back_bombardment_macropulse import (
    BACK_BOMBARDMENT_MACROPULSE_SCHEMA_VERSION,
    BackBombardmentMacropulseStudy,
    run_back_bombardment_macropulse_study,
    validate_back_bombardment_study,
    write_back_bombardment_macropulse_h5,
)

__all__ = [
    "BACK_BOMBARDMENT_MACROPULSE_SCHEMA_VERSION",
    "BackBombardmentMacropulseStudy",
    "run_back_bombardment_macropulse_study",
    "validate_back_bombardment_study",
    "write_back_bombardment_macropulse_h5",
]
