"""Four-tier solver/meshing finesse presets: extra_fine, fine, medium, coarse.

Bundles the numerical resolution knobs (field-map integration steps, ODE tolerance,
space-charge/beam-loading step size, emission-sampling resolution, phase-scan resolution) into
one selectable tier. Machine parameters (particle count, screen count, cathode
temperature/roughness, cavity R/Q, deflection current, SC/BL/deflection/aperture on-off switches)
are untouched by this module and stay directly user-controlled.

`fine` matches `run_thermionic_tm010_scanT_1400_1700.slurm`; `coarse` matches
`UH_gun_tracking_demo.ipynb`'s defaults.

Not tiered: `r_max_m`/`ext_zmax`/`z_min` (physical domain size, not a step size) and `bl_ncells`
(a physical cell count). The field-map grid step `dr_um`/`dz_um` is also fixed across every tier
-- coarsening it causes particle energies to diverge to unphysical values with the deflection
magnet on, so it is not a safe speed/precision tradeoff and is not offered as one.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict

#: Field-map grid resolution: fixed across every tier -- see the module docstring for why.
FIXED_DR_UM = 4.0
FIXED_DZ_UM = 13.0

#: CLI flag names a finesse tier controls.
FINESSE_KEYS = (
    "fm_nsteps",
    "fm_tt_nsteps",
    "ode_epsabs",
    "dt_mm",
    "sc_dt_mm",
    "cfx_dt_mm",
    "emission_nsteps",
    "phase_scan_n",
    "phase_scan_n_part",
    "phase_scan_dt_mm",
)

FINESSE_PRESETS: Dict[str, Dict[str, Any]] = {
    "extra_fine": {
        "fm_nsteps": 400,
        "fm_tt_nsteps": 400,
        "ode_epsabs": 1e-7,
        "dt_mm": 0.005,
        "sc_dt_mm": 0.005,
        "cfx_dt_mm": 0.005,
        "emission_nsteps": 400,
        "phase_scan_n": 180,
        "phase_scan_n_part": 40,
        "phase_scan_dt_mm": 0.25,
    },
    "fine": {
        "fm_nsteps": 200,
        "fm_tt_nsteps": 200,
        "ode_epsabs": 1e-6,
        "dt_mm": 0.01,
        "sc_dt_mm": 0.01,
        "cfx_dt_mm": 0.01,
        "emission_nsteps": 200,
        "phase_scan_n": 90,
        "phase_scan_n_part": 20,
        "phase_scan_dt_mm": 0.5,
    },
    "medium": {
        "fm_nsteps": 150,
        "fm_tt_nsteps": 150,
        "ode_epsabs": 1e-6,
        "dt_mm": 0.05,
        "sc_dt_mm": 0.05,
        "cfx_dt_mm": 0.1,
        "emission_nsteps": 150,
        "phase_scan_n": 60,
        "phase_scan_n_part": 20,
        "phase_scan_dt_mm": 0.5,
    },
    "coarse": {
        "fm_nsteps": 100,
        "fm_tt_nsteps": 100,
        "ode_epsabs": 1e-6,
        "dt_mm": 0.1,
        "sc_dt_mm": 0.1,
        "cfx_dt_mm": 0.2,
        "emission_nsteps": 100,
        "phase_scan_n": 24,
        "phase_scan_n_part": 20,
        "phase_scan_dt_mm": 0.5,
    },
}

FINESSE_TIERS = tuple(FINESSE_PRESETS.keys())


def finesse_preset_dict(tier: str) -> Dict[str, Any]:
    """Plain `{flag_name: value}` dict for `tier`, including the fixed `dr_um`/`dz_um`."""
    if tier not in FINESSE_PRESETS:
        raise ValueError(f"Unknown finesse tier {tier!r}; expected one of {FINESSE_TIERS}")
    return {"dr_um": FIXED_DR_UM, "dz_um": FIXED_DZ_UM, **FINESSE_PRESETS[tier]}


def apply_finesse_preset_to_args(args: argparse.Namespace, tier: str | None) -> None:
    """Mutate `args` in place with tier's values. Applied after `--preset`, so `--finesse` wins
    over `--preset quick` when both are given. `tier=None` is a no-op."""
    if tier is None:
        return
    preset = finesse_preset_dict(tier)
    for key, value in preset.items():
        setattr(args, key, value)
