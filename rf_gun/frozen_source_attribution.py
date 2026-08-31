"""Frozen-source physics attribution (UPGRADE_PLAN.md Sec. 4, "Beam loading -- major correction to
the plan", point 5): track the *identical* fixed-seed emitted source through several transport
physics configurations, so a downstream difference between two cases is attributable purely to
that transport physics (space charge / cathode mirror / beam loading), not to a different
macroparticle realization.

This complements, rather than replaces, rf_gun.emission_iteration's self-consistency loop: it
measures forces/transport with a *fixed* source (the emission current J(x,y,t) never changes
between cases), not emission feedback -- it cannot attribute how much of a difference in the real
gun is due to the emission current itself responding to a stronger/weaker field. That coupled
question is what rf_gun.emission_iteration.run_emission_field_iteration and
rf_gun.plotting.plot_emission_iteration_submodel_comparison address instead, for space charge and
the cathode mirror specifically (beam loading cannot be included there at all yet -- see
EmissionFieldIterationConfig.include_beam_loading).

Why the same seed gives a bit-for-bit identical initial bunch across cases: `build_bunch_thermionic`/
`build_bunch_thermionic_spatial` draw thermal momenta and emission-time samples from `emission`,
`tracking`, and `rng` alone -- never from `vol_params.sc_enabled`/`mirror_charge_enabled`/
`beam_loading_enabled`, which only affect what happens to the bunch *after* it is created. So
re-seeding the same `rng` state before each case's `run_transport_with_progress` call reproduces
the same B0 exactly, regardless of which of those three flags is set on that case's `vol_params`.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .diagnostics import build_screen_summary_from_phase_space
from .rftrack_volume import VolumeBuildParams
from .simulation import DiagnosticsParams, EmissionParams, TrackingParams, run_transport_with_progress

#: The five standard cases from UPGRADE_PLAN.md's frozen-source attribution table, as
#: (label, vol_params overrides on top of a caller-supplied baseline VolumeBuildParams). Any
#: baseline value for sc_enabled/mirror_charge_enabled/beam_loading_enabled is overridden here --
#: every other field (field map, finesse/integration, bl_Q_loaded/bl_r_over_q_ohm_per_m, etc.)
#: passes through from the caller's vol_params_base unchanged.
FROZEN_SOURCE_ATTRIBUTION_CASES: Tuple[Tuple[str, Dict[str, bool]], ...] = (
    ("RF only", dict(sc_enabled=False, mirror_charge_enabled=False, beam_loading_enabled=False)),
    ("RF + BL", dict(sc_enabled=False, mirror_charge_enabled=False, beam_loading_enabled=True)),
    ("RF + SC (free space)", dict(sc_enabled=True, mirror_charge_enabled=False, beam_loading_enabled=False)),
    ("RF + SC + mirror", dict(sc_enabled=True, mirror_charge_enabled=True, beam_loading_enabled=False)),
    ("RF + SC + mirror + BL", dict(sc_enabled=True, mirror_charge_enabled=True, beam_loading_enabled=True)),
)


@dataclass
class FrozenSourceAttributionResult:
    labels: List[str]
    overrides: List[Dict[str, bool]]
    results: List[Any] = dataclass_field(repr=False)          # one SimulationResult per case
    exit_summaries: List[Dict[str, Any]] = dataclass_field(default_factory=list)
    #: Sanity check that the frozen-source claim actually held (same Q_total_C/peak current for
    #: every case, since none of sc/mirror/BL affect emission itself) -- populated by
    #: run_frozen_source_attribution, not asserted internally, so a caller can decide how to react
    #: to an unexpected mismatch rather than have this module raise on their behalf.
    emitted_charge_C: List[float] = dataclass_field(default_factory=list)


def run_frozen_source_attribution(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    Ez0_phasor_axis: complex,
    vol_params_base: VolumeBuildParams,
    emission: EmissionParams,
    tracking: TrackingParams,
    cases: Sequence[Tuple[str, Dict[str, bool]]] = FROZEN_SOURCE_ATTRIBUTION_CASES,
    seed: int = 42,
    diagnostics: Optional[DiagnosticsParams] = None,
    spatial_source: Optional[Dict[str, np.ndarray]] = None,
) -> FrozenSourceAttributionResult:
    """Track the identical fixed-seed source through each of `cases` (default
    FROZEN_SOURCE_ATTRIBUTION_CASES), varying only sc_enabled/mirror_charge_enabled/
    beam_loading_enabled on top of `vol_params_base`. See the module docstring for why the same
    seed reproduces a bit-for-bit identical B0 across every case.

    A "RF + BL"/"... + BL" case requires `vol_params_base.bl_Q_loaded`/`bl_r_over_q_ohm_per_m`
    (and, if using `bl_tinj_mode="manual"`, `bl_tinj_manual_mm_c`) to already be calibrated by the
    caller (e.g. from a phase scan) -- this function does not calibrate beam loading itself.

    Returns a FrozenSourceAttributionResult with one exit-beam summary per case (via
    rf_gun.diagnostics.build_screen_summary_from_phase_space on that case's `Bout`, independent of
    whether `tracking.z_screens_m` configures any intermediate screens) and each case's total
    emitted charge (should match across every case -- a mismatch would indicate the frozen-source
    assumption was violated, e.g. by a non-fixed seed or emission parameters that do change between
    cases, and should be investigated rather than trusted).
    """
    diagnostics = (
        DiagnosticsParams(store_screen_phase_space=False, save_lost_particles=False)
        if diagnostics is None else diagnostics
    )

    result_obj = FrozenSourceAttributionResult(labels=[], overrides=[], results=[])

    for label, override in cases:
        vol_params_case = vol_params_base.replace(**override)
        rng = np.random.default_rng(seed)
        result, _stats = run_transport_with_progress(
            rft, Er_grid, Ez_grid, Ez0_phasor_axis, vol_params_case, emission, tracking,
            diagnostics=diagnostics, rng=rng, spatial_source=spatial_source,
        )

        n_initial = int(np.asarray(result.B0.get_phase_space(tracking.phase_fmt, "all")).shape[0])
        M_out = np.asarray(result.Bout.get_phase_space(tracking.phase_fmt, "all")) if result.Bout is not None else None
        exit_summary = build_screen_summary_from_phase_space(
            M_out, screen_index=-1, z_m=float(vol_params_case.z_max_m), n_initial=n_initial,
        )

        result_obj.labels.append(label)
        result_obj.overrides.append(dict(override))
        result_obj.results.append(result)
        result_obj.exit_summaries.append(exit_summary)
        result_obj.emitted_charge_C.append(float(result.thermo_info.get("Q_total_C", np.nan)))

    return result_obj
