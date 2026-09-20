"""Tests for `rf_gun.back_bombardment_deposition` (implementation plan Sec. 3.4, 4.2, 6.2, 10.1,
10.2, 15.2; addendum Sec. 19.2/19.3) -- Work Package 2's BB0 (TIO/CSDA baseline) deposition model.

Pure Python/numpy -- no RF-Track dependency. Synthetic `BackBombardmentEvents` are built directly
(matching the dataclass's own constructor/`validate()` requirements) rather than depending on the
still-in-progress real event-extraction pipeline (`extract_back_bombardment_events` is a documented
`NotImplementedError` stub -- see `rf_gun/back_bombardment_events.py`).
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from scipy.optimize import brentq

import rf_gun as rg
from rf_gun.back_bombardment_events import (
    BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
    BackBombardmentEvents,
    build_back_bombardment_provenance,
)
from rf_gun.cathode_geometry import SURFACE_CATHODE_FLAT, SURFACE_HOLDER, CathodeGeometry
from rf_gun.materials.electron_range import tio_range_um

Q_E = 1.602176634e-19  # elementary charge [C] -- matches rf_gun.constants.q_e (scipy CODATA)


# ------------------------------------------------------------------------------------------------
# Synthetic event builder
# ------------------------------------------------------------------------------------------------


def _make_events(
    *,
    kinetic_energy_eV: list[float],
    cos_incidence: list[float],
    surface_code: list[int],
    x_hit_m: list[float],
    y_hit_m: list[float],
    macro_weight_electrons: float = 1.0e6,
) -> BackBombardmentEvents:
    """Build a small synthetic `BackBombardmentEvents` set directly, matching plan Sec. 4.1's
    schema. `incident_energy_J = w*K*e` (plan Sec. 4.1) is computed here with the same elementary
    charge `back_bombardment_deposition.py` implicitly relies on via `events.incident_energy_J`
    (it never independently reconverts kinetic_energy_eV -> J -- see that module's docstring), so
    keeping this helper's convention explicit matters for the closure tests below.
    """
    n = len(kinetic_energy_eV)
    K_eV = np.asarray(kinetic_energy_eV, dtype=float)
    cos_inc = np.asarray(cos_incidence, dtype=float)
    surf = np.asarray(surface_code, dtype=np.uint8)
    heats_lab6 = np.isin(surf, (int(SURFACE_CATHODE_FLAT), 1, 2))
    w = np.full(n, macro_weight_electrons, dtype=float)
    incident_energy_J = w * K_eV * Q_E

    return BackBombardmentEvents(
        event_id=np.arange(n, dtype=np.int64),
        particle_id=np.arange(100, 100 + n, dtype=np.int64),
        state_id=np.zeros(n, dtype=np.int32),
        x_emit_m=np.zeros(n, dtype=np.float64),
        y_emit_m=np.zeros(n, dtype=np.float64),
        z_emit_m=np.zeros(n, dtype=np.float64),
        t_emit_rf_s=np.zeros(n, dtype=np.float64),
        rf_phase_emit_rad=np.zeros(n, dtype=np.float64),
        x_hit_m=np.asarray(x_hit_m, dtype=np.float64),
        y_hit_m=np.asarray(y_hit_m, dtype=np.float64),
        z_hit_m=np.zeros(n, dtype=np.float64),
        t_hit_rf_s=np.zeros(n, dtype=np.float64),
        return_time_s=np.zeros(n, dtype=np.float64),
        px_MeV_c=np.zeros(n, dtype=np.float64),
        py_MeV_c=np.zeros(n, dtype=np.float64),
        pz_MeV_c=-np.ones(n, dtype=np.float64) * 0.01,
        kinetic_energy_eV=K_eV,
        macro_weight_electrons=w,
        incident_energy_J=incident_energy_J,
        n_in_x=np.zeros(n, dtype=np.float64),
        n_in_y=np.zeros(n, dtype=np.float64),
        n_in_z=-np.ones(n, dtype=np.float64),
        cos_incidence=cos_inc,
        incidence_angle_rad=np.arccos(np.clip(cos_inc, -1.0, 1.0)),
        surface_code=surf,
        heats_lab6=heats_lab6,
        furthest_z_m=np.zeros(n, dtype=np.float64),
        n_screens_reached=np.zeros(n, dtype=np.int16),
        quality_flags=np.zeros(n, dtype=np.uint32),
        schema_version=BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
        coordinate_system="cathode_z0_vacuum_positive_z",
        normal_convention="inward_vacuum_to_solid",
        rf_frequency_Hz=2.856e9,
        rf_period_s=1.0 / 2.856e9,
        emission_phase_window="full_rf_period",
        n_launched=1_000_000,
        q_emitted_per_sample_C=1.0e-9,
        event_locator="legacy_ballistic",
        sc_enabled=False,
        mirror_charge_enabled=False,
        beam_loading_enabled=False,
        deflection_magnet_enabled=False,
        emission_iteration_enabled=False,
        flat_radius_mm=1.4,
        bevel_width_mm=0.2,
        bevel_angle_deg=45.0,
        cathode_length_mm=1.0,
        insertion_offset_mm=0.0,
        accounting={},
        provenance=build_back_bombardment_provenance(run_id="test_back_bombardment_deposition"),
    )


@pytest.fixture(scope="module")
def geometry() -> CathodeGeometry:
    return CathodeGeometry()


@pytest.fixture(scope="module")
def material():
    return rg.load_cathode_material("LaB6", "LaB6_UH_recommended_v1")


@pytest.fixture(scope="module")
def deposition_config():
    return rg.DepositionConfig(model="BB0_TIO")


def _cell_index(geometry: CathodeGeometry, value_m: float, n: int = 41) -> int:
    R_max_m = geometry.bevel_outer_radius_mm * 1.0e-3
    edges = np.linspace(-R_max_m, R_max_m, n + 1)
    return int(np.clip(np.searchsorted(edges, value_m, side="right") - 1, 0, n - 1))


# ------------------------------------------------------------------------------------------------
# 1. tio_range_um monotonicity guard (task item: verify before relying on it for CSDA inversion)
# ------------------------------------------------------------------------------------------------


def test_tio_range_monotonicity_over_relied_upon_energy_range(material):
    """Guard test for the CSDA-inversion assumption `back_bombardment_deposition.py` depends on:
    `tio_range_um` must be strictly monotonically increasing in kinetic energy over the range this
    module actually uses it (from the TIO validity floor up to comfortably above any event energy
    in these tests). If this ever failed, `_build_residual_range_lookup` raises `RuntimeError`
    rather than silently proceeding (see that function) -- this test independently confirms the
    assumption holds today, over a wide margin, so that guard is not masking a real problem.
    """
    ed = material.electron_deposition
    E_grid_keV = np.logspace(np.log10(0.03), np.log10(5000.0), 5000)
    R_grid_um = tio_range_um(E_grid_keV, ed.Z_eff, ed.A_eff_g_mol, ed.rho0_kg_m3)
    assert np.all(np.diff(R_grid_um) > 0.0), (
        "tio_range_um is not strictly monotonically increasing over [0.03, 5000] keV -- "
        "back_bombardment_deposition.py's CSDA residual-range inversion assumes this; if this "
        "assertion ever fails, the module's own RuntimeError guard (in "
        "_build_residual_range_lookup) must be investigated, not bypassed."
    )


def test_tio_range_reproduces_reference_table(material):
    """Cross-check against plan Sec. 3.4/5.3's reference range table (sub-percent agreement is the
    documented working decision in rf_gun/materials/electron_range.py -- this is a lightweight
    reminder check local to this test file, not a duplicate of tests/test_materials.py's own more
    thorough regression test)."""
    ed = material.electron_deposition
    E_keV = np.array([1, 10, 100, 300, 500, 1000], dtype=float)
    R_expected_um = np.array([0.030, 0.623, 23.35, 126.2, 262.3, 663.5])
    R_actual_um = tio_range_um(E_keV, ed.Z_eff, ed.A_eff_g_mol, ed.rho0_kg_m3)
    np.testing.assert_allclose(R_actual_um, R_expected_um, rtol=0.01)


# ------------------------------------------------------------------------------------------------
# 2. Peak-layer sanity: low-energy vs. high-energy normal-incidence events
# ------------------------------------------------------------------------------------------------


def test_low_energy_normal_incidence_peaks_in_shallow_layers(geometry, material, deposition_config):
    """A ~20 keV normal-incidence event has TIO range ~1-2 um (interpolating the reference table's
    0.623 um @10 keV / 23.35 um @100 keV) -- its deposited energy should be concentrated in the
    shallowest 1-3 layers (boundaries at [0,1,3,10] um), not spread into the deep layers."""
    events = _make_events(
        kinetic_energy_eV=[20_000.0],
        cos_incidence=[1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0],
        y_hit_m=[0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    ix = _cell_index(geometry, 0.0)
    profile = source.q_layer_J[ix, ix, :]
    assert profile.sum() > 0.0
    peak_layer = int(np.argmax(profile))
    assert peak_layer <= 2, f"expected peak in one of the first 3 layers (<10 um), got layer {peak_layer}"
    # Deep layers (300-1000 um and beyond) must be essentially empty.
    assert profile[-1] == 0.0
    assert profile[-2] == 0.0


def test_high_energy_normal_incidence_spreads_into_deep_layers(geometry, material, deposition_config):
    """A ~500 keV normal-incidence event has TIO range ~262 um (reference table) -- its deposited
    energy should reach the 100-300 um layer (and possibly 30-100 um), unlike the low-energy case,
    and should NOT escape geometrically (262 um is comfortably inside the default 1000 um limit)."""
    events = _make_events(
        kinetic_energy_eV=[500_000.0],
        cos_incidence=[1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0],
        y_hit_m=[0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    ix = _cell_index(geometry, 0.0)
    profile = source.q_layer_J[ix, ix, :]
    # layer_boundaries_um = [0,1,3,10,30,100,300,1000] -> index 5 is [100,300), index 6 is [300,1000)
    assert profile[5] > 0.0 or profile[4] > 0.0, "expected non-trivial deposition reaching >=30 um"
    assert profile.sum() > 0.0
    assert source.escaping_energy_geometric_J_total == 0.0
    rg.validate_energy_closure(source, events, rtol=1e-9)


# ------------------------------------------------------------------------------------------------
# 3. Oblique vs. normal incidence: the cos(incidence_angle) projection direction
# ------------------------------------------------------------------------------------------------


def test_oblique_incidence_is_more_concentrated_near_surface(geometry, material, deposition_config):
    """Same kinetic energy, two incidence angles (normal vs. 60 deg oblique) -- the oblique event's
    normal-depth deposition profile must be MORE concentrated near the surface (this is the
    plan's explicit warning: z_local = path_length * cos(incidence_angle), so a grazing hit spreads
    its range over LESS normal depth for the same path length -- getting this backwards is an easy,
    plan-flagged bug)."""
    K_keV = 300.0
    cos_oblique = np.cos(np.deg2rad(60.0))
    events = _make_events(
        kinetic_energy_eV=[K_keV * 1000.0, K_keV * 1000.0],
        cos_incidence=[1.0, cos_oblique],
        surface_code=[int(SURFACE_CATHODE_FLAT), int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0, 0.0003],
        y_hit_m=[0.0, 0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    ix_normal = _cell_index(geometry, 0.0)
    ix_oblique = _cell_index(geometry, 0.0003)

    normal_profile = source.q_layer_J[ix_normal, ix_normal, :]
    oblique_profile = source.q_layer_J[ix_oblique, ix_normal, :]
    assert normal_profile.sum() > 0.0 and oblique_profile.sum() > 0.0

    # Qualitative check: cumulative fraction deposited within the first 3 layers (normal depth
    # <10 um) must be LARGER for the oblique event.
    cum_normal = normal_profile[:3].sum() / normal_profile.sum()
    cum_oblique = oblique_profile[:3].sum() / oblique_profile.sum()
    assert cum_oblique > cum_normal, (
        f"oblique incidence should concentrate more energy within the first 3 layers than normal "
        f"incidence at the same energy; got cum_oblique={cum_oblique:.4f} <= cum_normal={cum_normal:.4f}"
    )

    # Exact check: independently (brentq, not the module's own lookup table) predict the energy
    # deposited by normal depth z=10 um at each angle, via K_i - K_res(10/cos(theta)), and compare
    # to the module's own first-three-layers sum (the layer grid's boundary at 10 um is exact, so
    # this is a like-for-like comparison, not an approximation of one).
    ed = material.electron_deposition
    R_i_um = float(tio_range_um(K_keV, ed.Z_eff, ed.A_eff_g_mol, ed.rho0_kg_m3))

    def _predicted_deposited_by_z_keV(z_um: float, cos_theta: float) -> float:
        s = z_um / cos_theta

        def _residual(E_keV: float) -> float:
            return float(tio_range_um(E_keV, ed.Z_eff, ed.A_eff_g_mol, ed.rho0_kg_m3)) - (R_i_um - s)

        K_res = brentq(_residual, 1.0e-6, K_keV, xtol=1.0e-10)
        return K_keV - K_res

    w = 1.0e6
    to_keV = 1.0 / (w * Q_E * 1000.0)
    actual_normal_keV = float(normal_profile[:3].sum()) * to_keV
    actual_oblique_keV = float(oblique_profile[:3].sum()) * to_keV
    predicted_normal_keV = _predicted_deposited_by_z_keV(10.0, 1.0)
    predicted_oblique_keV = _predicted_deposited_by_z_keV(10.0, cos_oblique)

    np.testing.assert_allclose(actual_normal_keV, predicted_normal_keV, rtol=1.0e-3)
    np.testing.assert_allclose(actual_oblique_keV, predicted_oblique_keV, rtol=1.0e-3)
    assert predicted_oblique_keV > predicted_normal_keV


# ------------------------------------------------------------------------------------------------
# 4. Energy closure: fully-contained low/medium-energy synthetic case
# ------------------------------------------------------------------------------------------------


def test_energy_closure_fully_contained_case(geometry, material, deposition_config):
    """A handful of low/medium-energy events, all well within the default 1000 um depth range,
    all on the flat face or bevel. `validate_energy_closure` must pass to a tight tolerance --
    this implementation's per-electron bookkeeping is designed to telescope exactly (see
    back_bombardment_deposition.py's module docstring), so `rtol=1e-9` is used here deliberately
    tighter than plan Sec. 15.2's own provisional 1e-6 target, to honestly reflect what this
    specific numerical scheme achieves (floating-point round-off only, not interpolation-table
    error, since the total closure per event does not depend on the lookup table's resolution)."""
    events = _make_events(
        kinetic_energy_eV=[5_000.0, 20_000.0, 80_000.0, 250_000.0],
        cos_incidence=[1.0, 0.95, 0.8, 0.6],
        surface_code=[int(SURFACE_CATHODE_FLAT), int(SURFACE_CATHODE_FLAT), 1, 1],
        x_hit_m=[0.0, 0.0002, 0.0011, -0.0012],
        y_hit_m=[0.0, -0.0003, 0.0004, 0.0005],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    rg.validate_energy_closure(source, events, rtol=1e-9)
    assert source.escaping_energy_geometric_J_total == 0.0
    assert source.escaping_energy_below_tio_validity_J_total == 0.0


def test_rim_hit_is_remapped_into_thermal_footprint(geometry, material, deposition_config):
    """A hit can be inside the circular LaB6 face while its Cartesian bin centre lies just
    outside the centre-based thermal mask. Its energy must be conserved in the nearest active
    cell, not stranded in a cell the thermal solver never assembles."""
    events = _make_events(
        kinetic_energy_eV=[20_000.0],
        cos_incidence=[1.0],
        surface_code=[1],
        x_hit_m=[1.4e-3],
        y_hit_m=[0.75e-3],
    )
    source = rg.build_back_bombardment_heat_source(
        events, geometry, material, deposition_config, xy_grid_n=41
    )

    assert np.hypot(events.x_hit_m[0], events.y_hit_m[0]) < geometry.bevel_outer_radius_mm * 1.0e-3
    assert np.sum(source.q_layer_J[~source.cathode_footprint_mask, :]) == 0.0
    assert np.sum(source.q_layer_J[source.cathode_footprint_mask, :]) == pytest.approx(
        source.total_deposited_energy_J, rel=1.0e-12
    )
    rg.validate_energy_closure(source, events, rtol=1.0e-9)


def test_energy_closure_fails_loudly_on_a_corrupted_source(geometry, material, deposition_config):
    """`validate_energy_closure` must actually raise (not silently pass) when the closure equation
    is violated -- sanity check for the check itself, using a deliberately corrupted `q_layer_J`."""
    events = _make_events(
        kinetic_energy_eV=[20_000.0],
        cos_incidence=[1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0],
        y_hit_m=[0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    import dataclasses

    corrupted = dataclasses.replace(source, q_layer_J=source.q_layer_J * 0.5)
    with pytest.raises(ValueError):
        rg.validate_energy_closure(corrupted, events, rtol=1e-9)


# ------------------------------------------------------------------------------------------------
# 5. Geometric escape: a deliberately very-high-energy event exceeding the 1000 um layer grid
# ------------------------------------------------------------------------------------------------


def test_very_high_energy_event_books_geometric_escape_and_still_closes(geometry, material, deposition_config):
    """A 3 MeV normal-incidence event has TIO range ~2.5 mm (>> the default 1000 um deepest layer
    boundary and >> the default 1.0 mm cathode_length_mm), so it must book non-zero
    `escaping_energy_geometric_J_total`, deposit nothing in "escaping_energy_below_tio_validity"
    (the floor is never reached -- the modeled depth limit binds first), and closure must still
    hold when the geometric-escape term is included."""
    events = _make_events(
        kinetic_energy_eV=[3_000_000.0],
        cos_incidence=[1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0],
        y_hit_m=[0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    assert source.escaping_energy_geometric_J_total > 0.0
    assert source.escaping_energy_below_tio_validity_J_total == 0.0
    assert source.total_deposited_energy_J < source.total_incident_energy_J
    rg.validate_energy_closure(source, events, rtol=1e-9)


# ------------------------------------------------------------------------------------------------
# 6. Holder/unknown-surface event: excluded from deposition, tracked separately
# ------------------------------------------------------------------------------------------------


def test_holder_event_is_excluded_not_deposited(geometry, material, deposition_config):
    """A holder-surface event (surface_code=SURFACE_HOLDER) must contribute zero deposited energy
    anywhere and must be tracked in `excluded_non_lab6_energy_J_total`, not silently dropped."""
    events = _make_events(
        kinetic_energy_eV=[20_000.0, 40_000.0],
        cos_incidence=[1.0, 1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT), int(SURFACE_HOLDER)],
        x_hit_m=[0.0, 0.002],
        y_hit_m=[0.0, 0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    holder_incident_J = float(events.incident_energy_J[1])
    assert source.excluded_non_lab6_energy_J_total == pytest.approx(holder_incident_J, rel=1e-12)
    assert source.n_events_excluded == 1
    assert source.n_events_included == 1
    # The holder event's energy must not appear anywhere in q_layer_J: total deposited should equal
    # only the flat-face event's own incident energy (fully contained, low energy).
    assert source.total_deposited_energy_J == pytest.approx(float(events.incident_energy_J[0]), rel=1e-9)
    rg.validate_energy_closure(source, events, rtol=1e-9)


# ------------------------------------------------------------------------------------------------
# 7. Below-TIO-validity floor: dumped into the terminal layer, not tracked as true escape
# ------------------------------------------------------------------------------------------------


def test_below_floor_energy_is_deposited_not_escaped(geometry, material, deposition_config):
    """A very low-energy event (well under the ~0.3 keV TIO validity floor) has its entire energy
    dumped into the shallowest layer (plan Sec. 5.3) -- fully deposited, so
    escaping_energy_below_tio_validity_J_total stays exactly 0.0 by this module's own design (see
    its docstring), and all incident energy shows up in layer 0."""
    events = _make_events(
        kinetic_energy_eV=[100.0],  # 0.1 keV, below DEFAULT_TIO_VALIDITY_FLOOR_KEV=0.3 keV
        cos_incidence=[1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0],
        y_hit_m=[0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    assert source.escaping_energy_below_tio_validity_J_total == 0.0
    assert source.escaping_energy_geometric_J_total == 0.0
    ix = _cell_index(geometry, 0.0)
    profile = source.q_layer_J[ix, ix, :]
    assert profile[0] == pytest.approx(float(events.incident_energy_J[0]), rel=1e-9)
    assert profile[1:].sum() == 0.0
    rg.validate_energy_closure(source, events, rtol=1e-9)


# ------------------------------------------------------------------------------------------------
# 8. Bevel-face event and BB1/BB2 dispatch
# ------------------------------------------------------------------------------------------------


def test_bevel_event_deposits_and_closes(geometry, material, deposition_config):
    """A medium-energy event on the bevel (surface_code=1, still LaB6-heating) at a position
    outside the flat-face radius must deposit correctly and close, exercising the lateral-binning
    footprint beyond r=flat_radius_mm."""
    r_bevel_m = 0.0015  # between flat_radius_mm=1.4mm and bevel_outer_radius_mm=1.6mm
    events = _make_events(
        kinetic_energy_eV=[60_000.0],
        cos_incidence=[0.9],
        surface_code=[1],
        x_hit_m=[r_bevel_m],
        y_hit_m=[0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    assert source.total_deposited_energy_J > 0.0
    assert source.cathode_footprint_mask[_cell_index(geometry, r_bevel_m), _cell_index(geometry, 0.0)]
    rg.validate_energy_closure(source, events, rtol=1e-9)


def test_bb1_and_bb2_raise_not_implemented(geometry, material):
    events = _make_events(
        kinetic_energy_eV=[20_000.0],
        cos_incidence=[1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0],
        y_hit_m=[0.0],
    )
    with pytest.raises(NotImplementedError):
        rg.build_back_bombardment_heat_source(
            events, geometry, material, rg.DepositionConfig(model="BB1_uncertainty")
        )
    with pytest.raises(NotImplementedError):
        rg.build_back_bombardment_heat_source(
            events, geometry, material, rg.DepositionConfig(model="BB2_response_library")
        )


def test_unknown_model_raises_value_error(geometry, material):
    events = _make_events(
        kinetic_energy_eV=[20_000.0],
        cos_incidence=[1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0],
        y_hit_m=[0.0],
    )
    with pytest.raises(ValueError):
        rg.build_back_bombardment_heat_source(
            events, geometry, material, rg.DepositionConfig(model="not_a_real_model")
        )


# ------------------------------------------------------------------------------------------------
# 9. back_bombardment_heat_source.h5 round trip (plan Sec. 4.2; Work Package 4's small missing
#    data-contract piece -- at the start of this pass no writer for this file existed anywhere).
# ------------------------------------------------------------------------------------------------


def test_heat_source_h5_round_trip(geometry, material, deposition_config, tmp_path):
    events = _make_events(
        kinetic_energy_eV=[5_000.0, 20_000.0, 80_000.0, 250_000.0],
        cos_incidence=[1.0, 0.95, 0.8, 0.6],
        surface_code=[int(SURFACE_CATHODE_FLAT), int(SURFACE_CATHODE_FLAT), 1, 1],
        x_hit_m=[0.0, 0.0002, 0.0011, -0.0012],
        y_hit_m=[0.0, -0.0003, 0.0004, 0.0005],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)

    path = tmp_path / "back_bombardment_heat_source.h5"
    written_path = rg.write_back_bombardment_heat_source_h5(
        path, source, source_events_hash="deadbeef" * 8
    )
    assert written_path == path
    assert path.is_file()

    with h5py.File(str(path), "r") as h5f:
        assert h5f.attrs["schema_version"] == rg.BACK_BOMBARDMENT_HEAT_SOURCE_SCHEMA_VERSION
        assert h5f.attrs["source_events_hash"] == "deadbeef" * 8
        assert "deposition" in h5f
        for name in ("x_centers_m", "y_centers_m", "layer_boundaries_um", "q_layer_J", "cathode_footprint_mask"):
            assert name in h5f["deposition"]

    round_tripped = rg.read_back_bombardment_heat_source_h5(path)
    np.testing.assert_allclose(round_tripped.x_centers_m, source.x_centers_m)
    np.testing.assert_allclose(round_tripped.y_centers_m, source.y_centers_m)
    np.testing.assert_allclose(round_tripped.layer_boundaries_um, source.layer_boundaries_um)
    np.testing.assert_allclose(round_tripped.layer_thickness_m, source.layer_thickness_m)
    np.testing.assert_allclose(round_tripped.q_layer_J, source.q_layer_J)
    np.testing.assert_array_equal(round_tripped.cathode_footprint_mask, source.cathode_footprint_mask)
    np.testing.assert_array_equal(round_tripped.state_ids, source.state_ids)
    assert round_tripped.model == source.model
    assert round_tripped.density_used_kg_m3 == pytest.approx(source.density_used_kg_m3)
    assert round_tripped.xy_cell_area_m2 == pytest.approx(source.xy_cell_area_m2)
    assert round_tripped.escaping_energy_geometric_J_total == pytest.approx(source.escaping_energy_geometric_J_total)
    assert round_tripped.escaping_energy_below_tio_validity_J_total == pytest.approx(
        source.escaping_energy_below_tio_validity_J_total
    )
    assert round_tripped.excluded_non_lab6_energy_J_total == pytest.approx(source.excluded_non_lab6_energy_J_total)
    assert round_tripped.total_incident_energy_J == pytest.approx(source.total_incident_energy_J)
    assert round_tripped.total_deposited_energy_J == pytest.approx(source.total_deposited_energy_J)
    assert round_tripped.n_events_included == source.n_events_included
    assert round_tripped.n_events_excluded == source.n_events_excluded

    # Energy closure must still hold on the round-tripped object (checked against the same events).
    rg.validate_energy_closure(round_tripped, events, rtol=1e-9)


def test_heat_source_h5_rejects_missing_schema_version(tmp_path):
    path = tmp_path / "not_a_heat_source.h5"
    with h5py.File(str(path), "w") as h5f:
        h5f.create_group("deposition")
    with pytest.raises(ValueError, match="schema_version"):
        rg.read_back_bombardment_heat_source_h5(path)


def test_heat_source_h5_rejects_wrong_schema_version(tmp_path):
    path = tmp_path / "wrong_version.h5"
    with h5py.File(str(path), "w") as h5f:
        h5f.attrs["schema_version"] = "back_bombardment_heat_source_v0"
        h5f.create_group("deposition")
    with pytest.raises(ValueError, match="schema_version"):
        rg.read_back_bombardment_heat_source_h5(path)


def test_heat_source_h5_no_hash_written_as_empty_string(geometry, material, deposition_config, tmp_path):
    events = _make_events(
        kinetic_energy_eV=[20_000.0],
        cos_incidence=[1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0],
        y_hit_m=[0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    path = tmp_path / "no_hash.h5"
    rg.write_back_bombardment_heat_source_h5(path, source)
    with h5py.File(str(path), "r") as h5f:
        assert h5f.attrs["source_events_hash"] == ""


def test_rim_reassignment_is_recorded_not_just_performed(geometry, material, deposition_config):
    """The remap conserves energy, but it is a boundary discretisation artefact.

    Its size must be visible in the source record, otherwise "energy is conserved" hides how
    much of the deposit was actually moved to a neighbouring cell.
    """
    events = _make_events(
        kinetic_energy_eV=[20_000.0],
        cos_incidence=[1.0],
        surface_code=[1],
        x_hit_m=[1.4e-3],
        y_hit_m=[0.75e-3],
    )
    source = rg.build_back_bombardment_heat_source(
        events, geometry, material, deposition_config, xy_grid_n=41
    )
    assert source.rim_reassigned_event_count == 1
    assert source.rim_reassigned_energy_J_total > 0.0
    assert source.rim_reassigned_energy_J_total == pytest.approx(
        source.total_deposited_energy_J, rel=1.0e-12
    )


def test_centre_hit_records_no_rim_reassignment(geometry, material, deposition_config):
    """The counter must discriminate: a well-inside hit is not a rim event."""
    events = _make_events(
        kinetic_energy_eV=[20_000.0],
        cos_incidence=[1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0],
        y_hit_m=[0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)
    assert source.rim_reassigned_event_count == 0
    assert source.rim_reassigned_energy_J_total == 0.0


def test_non_finite_hit_position_is_accounted_not_binned_into_a_corner_cell(
    geometry, material, deposition_config
):
    """A NaN coordinate must never be silently pinned to one fixed cell.

    ``searchsorted(edges, nan)`` lands past the last edge and clips to a corner, so without a
    guard the event's whole energy would be deposited in an arbitrary place.
    """
    events = _make_events(
        kinetic_energy_eV=[20_000.0, 20_000.0],
        cos_incidence=[1.0, 1.0],
        surface_code=[int(SURFACE_CATHODE_FLAT), int(SURFACE_CATHODE_FLAT)],
        x_hit_m=[0.0, float("nan")],
        y_hit_m=[0.0, 0.0],
    )
    source = rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config)

    assert source.non_finite_position_event_count == 1
    assert source.non_finite_position_energy_J_total > 0.0
    # The good event alone accounts for every deposited joule; the NaN event contributed none.
    assert np.count_nonzero(source.q_layer_J.sum(axis=2)) >= 1
    assert np.sum(source.q_layer_J[~source.cathode_footprint_mask, :]) == 0.0
    # The balance must still close, with the dropped energy named explicitly.
    rg.validate_energy_closure(source, events, rtol=1.0e-9)
