"""Back-bombardment/macropulse study configuration dataclasses (implementation plan Sec. 10.2,
10.3).

Pure Python, no RF-Track dependency. This module gives the notebook, CLI, and SLURM entry points
one shared set of small, replaceable-field configuration dataclasses and the single
`default_uh_back_bombardment_study_config()` factory that is "the one source of defaults" (plan
Sec. 10.3) -- in particular, the 8 microsecond default macropulse duration lives here, in
`MacropulseConfig.duration_s`, and nowhere else; no notebook constant, CLI default, or SLURM
literal may silently redefine it (or cathode geometry, or material data -- plan Sec. 10.3).

Use SI units throughout (plan Sec. 10.3: "Use SI units in dataclasses. CLI suffixes such as
`--macropulse-duration-us` are converted once at parsing.").
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .aperture import DEFAULT_CATHODE_BACKSTOP_THICKNESS_MM
from .cathode_geometry import CathodeGeometry


@dataclass(frozen=True)
class CathodeMaterialSelection:
    """Which cathode material and named property set to resolve via
    `rf_gun.materials.load_cathode_material(material_id, property_set)` (plan Sec. 5, 10.2).
    Defaults to the project's recommended modern LaB6 composite (plan Sec. 5.2)."""

    material_id: str = "LaB6"
    property_set: str = "LaB6_UH_recommended_v1"


@dataclass(frozen=True)
class MacropulseConfig:
    """The configurable RF macropulse (plan Sec. 8, 10.3). `duration_s` is THE single place the
    8 microsecond default lives -- it is a normal, replaceable SI-unit field used consistently by
    time grids, RF-cycle normalization, HDF5 metadata, and plots (plan Sec. 10.3); it must never
    be re-hardcoded elsewhere.

    `envelope`: the macropulse's prescribed current/power envelope shape. `"top_hat"` is the only
    implemented value at present -- a deliberate, explicitly-labeled idealization (plan Sec. 8.2,
    confirmed by addendum Sec. 19.4: "neither the `LaB6_heating` note nor the RF-Track manual
    supplies a measured or modeled fill/flat-top/decay envelope for any pulse duration ... The
    8 microsecond top-hat remains a deliberate, explicitly-labeled idealization ... there is
    nothing to substitute it with yet"). A future measured/modeled fill-decay envelope would add a
    new `envelope` value here, not change this field's meaning.
    """

    duration_s: float = 8.0e-6
    envelope: str = "top_hat"

    def __post_init__(self) -> None:
        if not (self.duration_s > 0.0):
            raise ValueError(f"duration_s must be positive, got {self.duration_s!r}")


@dataclass(frozen=True)
class DepositionConfig:
    """Which named energy-deposition model level to use (plan Sec. 3.4): `"BB0_TIO"` (TIO/CSDA
    baseline, the default and the only one implemented so far), `"BB1_uncertainty"`, or
    `"BB2_response_library"` (both future work)."""

    model: str = "BB0_TIO"


#: Provisional total hemispherical emissivity used by `rf_gun.thermal`'s radiative-loss term
#: (plan Sec. 6.2/10.2's `ThermalConfig.total_hemispherical_emissivity`, Sec. 5.5's "the effective
#: radiative-loss model carries its own dataset name and uncertainty").
#:
#: THIS IS NOT A MEASURED TOTAL HEMISPHERICAL EMISSIVITY DATASET -- one does not exist yet for
#: LaB6 in this project. `materials.base.OpticalComponent.spectral_anchors_nm` only carries two
#: *spectral* anchors (0.687 @550nm / 0.713 @1064nm, Kowalczyk Int. J. Thermophys. 2014) and plan
#: Sec. 5.5 explicitly forbids substituting those directly for a total hemispherical value ("Do
#: not substitute the 550/1064 nm anchors directly for total hemispherical emissivity."). This
#: constant is therefore a deliberately separate, clearly-provisional order-of-magnitude value
#: (0.8, typical of LaB6 and similar refractory borides/carbides in the literature) chosen only so
#: `rf_gun.thermal`'s radiative-loss term has *something* physically reasonable to evaluate with
#: before a real total-emissivity dataset/model exists -- a future `LaB6_emissivity_total_*.yaml`
#: dataset (a genuine `materials` component, resolved and provenance-tracked like every other
#: property) is the correct way to replace this, not a silent change to this bare constant.
DEFAULT_TOTAL_HEMISPHERICAL_EMISSIVITY = 0.8


@dataclass(frozen=True)
class ThermalConfig:
    """Which thermal solver backend to use (plan Sec. 6, 10.1) plus every numerical/boundary-
    condition control `rf_gun.thermal`'s four backends need (plan Sec. 6.2/6.3, 10.2, 15.3;
    Work Package 3). Lives here (not in `rf_gun/thermal.py` itself) so it stays alongside the
    project's other study-configuration dataclasses per plan Sec. 10.2's "Core dataclasses" list
    and `BackBombardmentStudyConfig`'s existing `thermal: ThermalConfig` field -- `rf_gun.thermal`
    imports this class rather than redefining it.

    `backend`: `"python_xy_layered"` (default, the asymmetric Cartesian depth-layered solver),
    `"python_xy_sheet"` (one depth-integrated sheet), `"lumped_energy_check"` (0D energy/units
    sanity check), `"uh_legacy_1d"` (historical 1D depth-only Kowalczyk/McKee benchmark), or
    (once available, out of this pass's scope) `"comsol_3d"`.

    The spatial/time grid itself is NOT a field here -- every backend takes its `(x,y)` grid,
    depth-layer boundaries, cathode footprint mask, and macro-time grid from the mandatory
    `heat_source` argument to `rf_gun.thermal.solve_xy_layered_thermal` (a
    `rf_gun.thermal.VolumetricHeatSourceTimeSeries`), since that object already has to define all
    of those consistently with the deposited-power tensor it carries; duplicating an "override"
    grid here would only invite the two to silently disagree.

    Nonlinear/implicit-solve controls (plan Sec. 6.2):
      * `dt_s`: `None` (default) steps the implicit solve exactly at `heat_source.t_grid_s`'s own
        bin edges (piecewise-constant power held over each bin). A positive float instead
        sub-steps each bin at intervals of at most `dt_s`, still holding that bin's own power
        constant across every sub-step -- this decouples TIME refinement from the heat source's
        own time-binning resolution, which is what plan Sec. 15.3's mesh/time convergence check
        needs to vary independently of mesh (spatial) refinement.
      * `picard_max_iter`/`picard_tol_K`: nonlinear-property (`rho(T)`, `cp(T)`, `k(T)`, and the
        radiative-loss term) fixed-point iteration cap and convergence tolerance per timestep
        (max absolute temperature change between successive iterates, Kelvin). Raises rather than
        silently returning a non-converged state if `picard_max_iter` is exceeded.
      * `energy_residual_tol`: plan Sec. 15.3's "provisional production target ... below 1e-3" for
        the normalized energy residual `rf_gun.thermal.ThermalResult.energy_residual_normalized`.

    Radiation/boundary terms (plan Sec. 5.5, 6.2):
      * `total_hemispherical_emissivity`: see `DEFAULT_TOTAL_HEMISPHERICAL_EMISSIVITY`'s own
        docstring above for why this is a named, clearly-provisional constant and not derived from
        `materials.base.OpticalComponent`. `0.0` disables radiative loss entirely (used by the
        analytic-validation tests, which need a case with no radiation to compare against a closed
        -form energy balance).
      * `contact_h_W_m2K`: uniform bottom-layer (mount-facing) contact conductance to
        `T_mount_K` [W/m^2/K]. `0.0` (default) means an adiabatic bottom boundary -- a *named
        simplification* (plan Sec. 6.2: "A uniform boundary coefficient is a named simplification,
        not an assumed symmetry"), not a physical claim that the real mount is perfectly insulating.
      * `T_mount_K`/`T_environment_K`: boundary temperatures for the contact and radiative-loss
        terms respectively (plan Sec. 5.5: "boundary-condition sets, not LaB6 bulk properties").

    Backend-specific parameters:
      * `h_eff_m`: `python_xy_sheet`'s declared effective thickness (plan Sec. 6.3: "`h_eff` is a
        declared model parameter"), used to build its single areal heat capacity/conductance
        `C_A=rho*cp*h_eff`, `K_A=k*h_eff`. Unused by every other backend.
      * `uh_legacy_n_depth_layers`: `uh_legacy_1d`'s own independent uniform depth-grid resolution
        (a convergence parameter for that benchmark's 1D solve, unrelated to the layered solver's
        own `layer_boundaries_um` convergence parameter). Unused by every other backend.

    `store_layer_history`: if `True`, `rf_gun.thermal.ThermalResult` retains the full
    `T_layer(x,y,ell,t)` 4D history (plan Sec. 6.2's "Store `T_layer(x,y,ell,t)` when requested");
    if `False` (default), only the always-stored `T_surface(x,y,t)` and the final full-depth state
    are kept, to bound memory for large grids/long runs.
    """

    backend: str = "python_xy_layered"
    dt_s: float | None = None
    picard_max_iter: int = 20
    picard_tol_K: float = 1.0e-3
    energy_residual_tol: float = 1.0e-3
    total_hemispherical_emissivity: float = DEFAULT_TOTAL_HEMISPHERICAL_EMISSIVITY
    contact_h_W_m2K: float = 0.0
    T_mount_K: float = 300.0
    T_environment_K: float = 300.0
    h_eff_m: float = 100.0e-6
    uh_legacy_n_depth_layers: int = 21
    store_layer_history: bool = False

    def __post_init__(self) -> None:
        if self.dt_s is not None and not (self.dt_s > 0.0):
            raise ValueError(f"dt_s must be positive or None, got {self.dt_s!r}")
        if not (self.picard_max_iter >= 1):
            raise ValueError(f"picard_max_iter must be >= 1, got {self.picard_max_iter!r}")
        if not (self.picard_tol_K > 0.0):
            raise ValueError(f"picard_tol_K must be positive, got {self.picard_tol_K!r}")
        if not (self.energy_residual_tol > 0.0):
            raise ValueError(f"energy_residual_tol must be positive, got {self.energy_residual_tol!r}")
        if not (0.0 <= self.total_hemispherical_emissivity <= 1.0):
            raise ValueError(
                f"total_hemispherical_emissivity must be in [0,1], got "
                f"{self.total_hemispherical_emissivity!r}"
            )
        if not (self.contact_h_W_m2K >= 0.0):
            raise ValueError(f"contact_h_W_m2K must be non-negative, got {self.contact_h_W_m2K!r}")
        if not (self.h_eff_m > 0.0):
            raise ValueError(f"h_eff_m must be positive, got {self.h_eff_m!r}")
        if not (self.uh_legacy_n_depth_layers >= 2):
            raise ValueError(
                f"uh_legacy_n_depth_layers must be >= 2, got {self.uh_legacy_n_depth_layers!r}"
            )


@dataclass(frozen=True)
class CouplingConfig:
    """Which physics-coupling level this study run represents (plan Sec. 1's L1-L5 staging):
    `"L2_one_way"` (the default -- one qualified source, no temperature-to-emission feedback),
    `"L3_thermal_emission_feedback"`, or `"L4_cavity_feedback"` (both deferred, plan addendum
    Sec. 19.2's Work Package 6)."""

    level: str = "L2_one_way"


@dataclass(frozen=True)
class BackBombardmentCaptureConfig:
    """Event-capture method and quality-filter thresholds (plan Sec. 3.2, 10.2).

    `event_locator`: `"backstop_raycast_v1"` (the production method once Work Package 1's
    RF-Track backstop/ray-cast integration lands) or `"legacy_ballistic"` (the existing `Bout`
    drift reconstruction, retained for comparison only per plan Sec. 3.2).

    The remaining fields are quality-filter thresholds this implementation chose (the plan does
    not fix exact numeric values, only the general requirement of "counts, charge, and kinetic
    energy ... recorded before and after every filter", plan Sec. 3.1):
      * `require_inward_momentum`: enforce `p_hit . n_in > 0` (plan Sec. 3.1's exact heating-event
        test) -- always `True` for the production `backstop_raycast_v1` locator; kept as an
        explicit, named field rather than a hardcoded assumption so a diagnostic/debug run can
        disable it and see what the unfiltered candidate population looks like.
      * `max_unknown_surface_fraction`: the largest allowed fraction of qualified events landing
        on `SURFACE_UNKNOWN` before capture should be treated as a production warning/failure
        (plan Sec. 3.3's table: "Causes a production warning/failure above a set fraction"). 1%
        is a conservative starting point, not a validated production tolerance.
      * `backstop_thickness_mm`: the thickness of the `Aperture_1d` backstop element the run was
        (or must be) tracked with (`rf_gun.aperture.build_cathode_backstop`'s own `thickness_mm`,
        also `rf_gun.rftrack_volume.VolumeBuildParams.cathode_backstop_thickness_mm`) -- needed by
        `rf_gun.back_bombardment_events.extract_back_bombardment_events` to reconstruct the exact
        `backstop_z_min_m` band edge for `rf_gun.backstop_loss_separation.identify_backstop_loss_candidates`
        (`backstop_z_min_m = -backstop_thickness_mm * 1e-3`, cathode-frame `z0_global=0`
        convention). Defaults to the same project-wide default the backstop itself uses, so a
        capture config built without an explicit override still matches an unmodified
        `VolumeBuildParams` backstop by construction.
    """

    event_locator: str = "backstop_raycast_v1"
    require_inward_momentum: bool = True
    max_unknown_surface_fraction: float = 0.01
    backstop_thickness_mm: float = DEFAULT_CATHODE_BACKSTOP_THICKNESS_MM


@dataclass(frozen=True)
class BackBombardmentStudyConfig:
    """Bundles every sub-configuration a back-bombardment/macropulse study needs (plan Sec. 10.2's
    core-dataclass list), plus the authoritative `CathodeGeometry` (plan addendum Sec. 19.2's
    confirmed as-built hardware geometry by default).

    Nested-field replacement (read this before calling `.replace()`): `.replace(**overrides)` is a
    thin wrapper around `dataclasses.replace(self, **overrides)`. Because `BackBombardmentStudyConfig`
    is frozen and its sub-configs are themselves separate frozen dataclasses, `.replace(macropulse=...)`
    swaps the *entire* `macropulse` field for the exact object you pass -- it does NOT deep-merge
    field-by-field with the existing `macropulse`. For example::

        cfg = default_uh_back_bombardment_study_config()
        cfg2 = cfg.replace(macropulse=MacropulseConfig(duration_s=12.0e-6))

    `cfg2.macropulse` is a brand new `MacropulseConfig(duration_s=12.0e-6, envelope="top_hat")`
    (picking up `MacropulseConfig`'s own default for every field you didn't set) -- it is not
    `cfg.macropulse` with only `duration_s` patched. If you want to change one field of an
    *existing*, already-customized nested config while keeping its other overridden fields, build
    the nested replacement from that existing object explicitly, e.g.
    `cfg.replace(macropulse=dataclasses.replace(cfg.macropulse, duration_s=12.0e-6))` -- this
    module does not attempt to auto-deep-merge, since guessing which nested fields the caller
    intended to keep would be more surprising than requiring it explicitly. This matches exactly
    how plan Sec. 11's notebook example calls `.replace(macropulse=..., material=..., ...)`: each
    keyword there always constructs a complete, fully-specified nested config object.
    """

    macropulse: MacropulseConfig = field(default_factory=MacropulseConfig)
    material: CathodeMaterialSelection = field(default_factory=CathodeMaterialSelection)
    deposition: DepositionConfig = field(default_factory=DepositionConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    capture: BackBombardmentCaptureConfig = field(default_factory=BackBombardmentCaptureConfig)
    geometry: CathodeGeometry = field(default_factory=CathodeGeometry)

    def replace(self, **overrides: Any) -> "BackBombardmentStudyConfig":
        """`dataclasses.replace(self, **overrides)` -- see the class docstring for exactly how
        nested-config overrides are meant to be constructed (whole-object replacement per
        top-level field, not a deep merge)."""
        return replace(self, **overrides)


def default_uh_back_bombardment_study_config() -> BackBombardmentStudyConfig:
    """The one source of defaults (plan Sec. 10.3): 8 microsecond top-hat macropulse,
    `LaB6_UH_recommended_v1` material, `BB0_TIO` deposition, `python_xy_layered` thermal backend,
    `L2_one_way` coupling, and the confirmed as-built `CathodeGeometry()` default (plan addendum
    Sec. 19.2). Every field here is every other sub-dataclass's own already-stated default, so
    this factory just documents that combination as the project's single canonical default --
    it does not introduce any new literal value of its own.
    """
    return BackBombardmentStudyConfig()
