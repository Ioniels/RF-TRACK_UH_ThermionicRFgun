"""Tests for `rf_gun.comsol_io` (implementation plan Sec. 7; addendum Sec. 19.2 -- Work Package 5,
data-contract/interface-only pass).

Pure Python/HDF5/CSV -- no RF-Track dependency, no real COMSOL software or license involved. There
is no genuine COMSOL output to test against (addendum Sec. 19.2 is explicit that none exists yet
and that this is fine/expected): `load_comsol_thermal_result` is instead round-tripped against
`write_synthetic_comsol_result_for_testing`, a TEST-ONLY writer defined in this file (not in
`rf_gun/comsol_io.py` itself) that writes a self-consistent fake result in exactly the documented
`comsol_thermal_result_v1` contract, purely so the reader has something real to load.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import rf_gun as rg
from rf_gun.back_bombardment_events import (
    BACK_BOMBARDMENT_EVENTS_SCHEMA_VERSION,
    BackBombardmentEvents,
    build_back_bombardment_provenance,
)
from rf_gun.cathode_geometry import SURFACE_CATHODE_BEVEL, SURFACE_CATHODE_FLAT, CathodeGeometry
from rf_gun.comsol_io import COMSOL_THERMAL_RESULT_SCHEMA_VERSION, ComsolThermalResult

Q_E = 1.602176634e-19  # elementary charge [C], matches rf_gun.constants.q_e


# ------------------------------------------------------------------------------------------------
# Synthetic BackBombardmentEvents / BackBombardmentHeatSource builder (same pattern as
# tests/test_back_bombardment_deposition.py's `_make_events`)
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
    n = len(kinetic_energy_eV)
    K_eV = np.asarray(kinetic_energy_eV, dtype=float)
    cos_inc = np.asarray(cos_incidence, dtype=float)
    surf = np.asarray(surface_code, dtype=np.uint8)
    heats_lab6 = np.isin(surf, (int(SURFACE_CATHODE_FLAT), int(SURFACE_CATHODE_BEVEL), 2))
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
        provenance=build_back_bombardment_provenance(run_id="test_comsol_io"),
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


@pytest.fixture()
def heat_source(geometry, material, deposition_config):
    events = _make_events(
        kinetic_energy_eV=[20_000.0, 80_000.0, 300_000.0],
        cos_incidence=[1.0, 0.8, 0.6],
        surface_code=[int(SURFACE_CATHODE_FLAT), int(SURFACE_CATHODE_FLAT), int(SURFACE_CATHODE_BEVEL)],
        x_hit_m=[0.0, 0.3e-3, 1.45e-3],
        y_hit_m=[0.0, -0.2e-3, 0.1e-3],
    )
    return rg.build_back_bombardment_heat_source(events, geometry, material, deposition_config, xy_grid_n=15)


# ------------------------------------------------------------------------------------------------
# 1. export_comsol_heat_source: files produced, closure exact
# ------------------------------------------------------------------------------------------------


def test_export_produces_expected_files(tmp_path, heat_source, geometry, material):
    paths = rg.export_comsol_heat_source(tmp_path, heat_source, geometry, material)
    assert set(paths.keys()) == {"csv", "manifest"}
    assert paths["csv"].is_file()
    assert paths["manifest"].is_file()
    assert paths["csv"].parent == tmp_path / "comsol_source"
    assert paths["csv"].name == "comsol_heat_source.csv"
    assert paths["manifest"].name == "comsol_source_manifest.json"


def test_export_closure_matches_heat_source_totals(tmp_path, heat_source, geometry, material):
    """The task's central export test: sum of exported per-row incident/deposited/escaping
    energy must equal the source's own recorded totals exactly (to floating-point tolerance)."""
    import csv

    paths = rg.export_comsol_heat_source(tmp_path, heat_source, geometry, material)
    with paths["csv"].open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) > 0
    sum_deposited = sum(float(r["deposited_J"]) for r in rows)
    sum_incident = sum(float(r["incident_energy_attributed_J"]) for r in rows)
    sum_escaping = sum(float(r["escaping_energy_attributed_J"]) for r in rows)

    np.testing.assert_allclose(sum_deposited, heat_source.total_deposited_energy_J, rtol=1e-9)
    np.testing.assert_allclose(sum_incident, heat_source.total_incident_energy_J, rtol=1e-9)
    total_escaping_expected = (
        heat_source.escaping_energy_geometric_J_total
        + heat_source.escaping_energy_below_tio_validity_J_total
    )
    np.testing.assert_allclose(sum_escaping, total_escaping_expected, rtol=1e-9, atol=1e-30)

    # Every row must fall within the LaB6 footprint -- never 'holder' (module docstring's
    # holder_note: BB0's footprint mask never extends past bevel_outer_radius_mm).
    zone_labels = {r["zone_label"] for r in rows}
    assert zone_labels <= {"cathode_flat", "cathode_bevel"}


def test_export_manifest_records_material_geometry_and_schema(tmp_path, heat_source, geometry, material):
    paths = rg.export_comsol_heat_source(tmp_path, heat_source, geometry, material)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == rg.COMSOL_SOURCE_SCHEMA_VERSION
    assert manifest["material"]["material_id"] == material.material_id
    assert manifest["material"]["property_set"] == material.property_set
    assert manifest["geometry"]["flat_radius_mm"] == pytest.approx(geometry.flat_radius_mm)
    assert manifest["power_density_conversion"]["mode"] == "representative_period_snapshot"
    assert manifest["power_density_conversion"]["rf_frequency_Hz"] == pytest.approx(
        rg.DEFAULT_RF_FREQUENCY_HZ
    )
    # No extra_metadata supplied -> placeholders must stay empty, never fabricated.
    assert manifest["rf_envelope_keyframe_placeholders"]["values"] == {}


def test_export_extra_metadata_is_passed_through_not_fabricated(tmp_path, heat_source, geometry, material):
    extra = {"keyframe_state_id": 3, "rf_envelope": "documented_8us_tophat"}
    paths = rg.export_comsol_heat_source(
        tmp_path, heat_source, geometry, material, extra_metadata=extra
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["rf_envelope_keyframe_placeholders"]["values"] == extra


def test_export_energy_density_only_mode_when_no_frequency(tmp_path, heat_source, geometry, material):
    import csv

    paths = rg.export_comsol_heat_source(
        tmp_path, heat_source, geometry, material, rf_frequency_Hz=None
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["power_density_conversion"]["mode"] == "energy_density_only"
    assert manifest["power_density_conversion"]["rf_frequency_Hz"] is None
    with paths["csv"].open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert all(r["q_BB_W_m3"].lower() == "nan" for r in rows)


def test_export_rejects_both_frequency_and_override(tmp_path, heat_source, geometry, material):
    override = np.zeros_like(heat_source.q_layer_J)
    with pytest.raises(ValueError):
        rg.export_comsol_heat_source(
            tmp_path, heat_source, geometry, material, q_layer_W_m3_override=override
        )


# ------------------------------------------------------------------------------------------------
# 2. Synthetic COMSOL result writer (TEST-ONLY -- not part of rf_gun.comsol_io's public API) +
#    round-trip through load_comsol_thermal_result
# ------------------------------------------------------------------------------------------------


def write_synthetic_comsol_result_for_testing(
    path: str | Path,
    *,
    n_t: int = 6,
    n_x: int = 4,
    n_y: int = 5,
    mesh_id: str = "test_mesh_v1",
    time_step_id: str = "test_dt_v1",
    convergence_metrics: dict | None = None,
    include_surface_field: bool = True,
    seed: int = 42,
) -> Path:
    """TEST-ONLY: writes a self-consistent fake `comsol_thermal_result_v1` HDF5 file (NOT a
    production function -- there is no real COMSOL output to test against, addendum Sec. 19.2).
    Deliberately lives in this test file, not in `rf_gun/comsol_io.py`, so the production module
    never carries a synthetic-data generator.

    Matches `rf_gun.comsol_io.load_comsol_thermal_result`'s documented contract exactly (module
    docstring Sec. 2): root attrs `schema_version`/`mesh_id`/`time_step_id`, `/thermal/*` required
    1D datasets, `/convergence`'s `json` attr, optional `/surface_field` group.
    """
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    time_s = np.linspace(0.0, 8.0e-6, n_t)
    T_center_K = 1650.0 + 10.0 * time_s / 8.0e-6
    T_flat_edge_K = 1645.0 + 8.0 * time_s / 8.0e-6
    T_bevel_K = 1640.0 + 6.0 * time_s / 8.0e-6
    T_max_K = T_center_K + 2.0
    T_area_average_K = 0.5 * (T_center_K + T_flat_edge_K)
    stored_energy_J = 1.0e-3 + 1.0e-4 * time_s / 8.0e-6
    radiation_loss_W = 5.0 + 0.1 * rng.standard_normal(n_t)
    boundary_heat_flow_W = 2.0 + 0.05 * rng.standard_normal(n_t)

    with h5py.File(str(path), "w") as h5f:
        h5f.attrs["schema_version"] = COMSOL_THERMAL_RESULT_SCHEMA_VERSION
        h5f.attrs["mesh_id"] = mesh_id
        h5f.attrs["time_step_id"] = time_step_id

        thermal_grp = h5f.create_group("thermal")
        thermal_grp.create_dataset("time_s", data=time_s)
        thermal_grp.create_dataset("T_center_K", data=T_center_K)
        thermal_grp.create_dataset("T_flat_edge_K", data=T_flat_edge_K)
        thermal_grp.create_dataset("T_bevel_K", data=T_bevel_K)
        thermal_grp.create_dataset("T_max_K", data=T_max_K)
        thermal_grp.create_dataset("T_area_average_K", data=T_area_average_K)
        thermal_grp.create_dataset("stored_energy_J", data=stored_energy_J)
        thermal_grp.create_dataset("radiation_loss_W", data=radiation_loss_W)
        thermal_grp.create_dataset("boundary_heat_flow_W", data=boundary_heat_flow_W)

        conv_grp = h5f.create_group("convergence")
        conv_grp.attrs["json"] = json.dumps(convergence_metrics if convergence_metrics is not None else {
            "mesh_refinement_level": 2,
            "time_step_convergence_rel_change": 0.004,
        })

        if include_surface_field:
            x_m = np.linspace(-1.4e-3, 1.4e-3, n_x)
            y_m = np.linspace(-1.4e-3, 1.4e-3, n_y)
            T_surface_full_K = (
                T_center_K[np.newaxis, np.newaxis, :]
                + rng.standard_normal((n_x, n_y, n_t)) * 0.0  # deterministic, zero-noise field
            )
            sf_grp = h5f.create_group("surface_field")
            sf_grp.create_dataset("x_m", data=x_m)
            sf_grp.create_dataset("y_m", data=y_m)
            sf_grp.create_dataset("T_surface_full_K", data=T_surface_full_K)

    return path


def test_synthetic_writer_round_trips_every_field(tmp_path):
    path = tmp_path / "synthetic_comsol_result.h5"
    write_synthetic_comsol_result_for_testing(path, convergence_metrics={"foo": "bar", "n": 3})
    result = rg.load_comsol_thermal_result(path)

    assert isinstance(result, ComsolThermalResult)
    assert result.mesh_id == "test_mesh_v1"
    assert result.time_step_id == "test_dt_v1"
    assert result.convergence_metrics == {"foo": "bar", "n": 3}
    assert result.schema_version == COMSOL_THERMAL_RESULT_SCHEMA_VERSION

    with h5py.File(str(path), "r") as h5f:
        np.testing.assert_array_equal(result.time_s, h5f["thermal"]["time_s"][()])
        np.testing.assert_array_equal(result.T_center_K, h5f["thermal"]["T_center_K"][()])
        np.testing.assert_array_equal(result.T_flat_edge_K, h5f["thermal"]["T_flat_edge_K"][()])
        np.testing.assert_array_equal(result.T_bevel_K, h5f["thermal"]["T_bevel_K"][()])
        np.testing.assert_array_equal(result.T_max_K, h5f["thermal"]["T_max_K"][()])
        np.testing.assert_array_equal(result.T_area_average_K, h5f["thermal"]["T_area_average_K"][()])
        np.testing.assert_array_equal(result.stored_energy_J, h5f["thermal"]["stored_energy_J"][()])
        np.testing.assert_array_equal(result.radiation_loss_W, h5f["thermal"]["radiation_loss_W"][()])
        np.testing.assert_array_equal(
            result.boundary_heat_flow_W, h5f["thermal"]["boundary_heat_flow_W"][()]
        )
        np.testing.assert_array_equal(result.x_m, h5f["surface_field"]["x_m"][()])
        np.testing.assert_array_equal(result.y_m, h5f["surface_field"]["y_m"][()])
        np.testing.assert_array_equal(
            result.T_surface_full_K, h5f["surface_field"]["T_surface_full_K"][()]
        )


def test_synthetic_writer_round_trip_without_surface_field(tmp_path):
    path = tmp_path / "synthetic_no_surface.h5"
    write_synthetic_comsol_result_for_testing(path, include_surface_field=False)
    result = rg.load_comsol_thermal_result(path)
    assert result.T_surface_full_K is None
    assert result.x_m is None
    assert result.y_m is None


# ------------------------------------------------------------------------------------------------
# 3. Schema-version rejection (matches read_back_bombardment_events_h5's precedent style)
# ------------------------------------------------------------------------------------------------


def test_load_rejects_missing_schema_version(tmp_path):
    path = tmp_path / "no_schema.h5"
    with h5py.File(str(path), "w") as h5f:
        h5f.create_group("thermal")
    with pytest.raises(ValueError, match="schema_version"):
        rg.load_comsol_thermal_result(path)


def test_load_rejects_wrong_schema_version(tmp_path):
    path = tmp_path / "wrong_schema.h5"
    write_synthetic_comsol_result_for_testing(path)
    with h5py.File(str(path), "a") as h5f:
        del h5f.attrs["schema_version"]
        h5f.attrs["schema_version"] = "comsol_thermal_result_v0_old"
    with pytest.raises(ValueError, match="comsol_thermal_result_v0_old"):
        rg.load_comsol_thermal_result(path)


def test_load_rejects_missing_required_dataset(tmp_path):
    path = tmp_path / "missing_dataset.h5"
    write_synthetic_comsol_result_for_testing(path)
    with h5py.File(str(path), "a") as h5f:
        del h5f["thermal"]["T_max_K"]
    with pytest.raises(ValueError, match="T_max_K"):
        rg.load_comsol_thermal_result(path)


def test_load_rejects_missing_file():
    with pytest.raises(ValueError, match="not found"):
        rg.load_comsol_thermal_result("/nonexistent/path/does_not_exist.h5")


# ------------------------------------------------------------------------------------------------
# 4. compare_python_comsol_thermal: comsol_result=None is the single most important behavior
# ------------------------------------------------------------------------------------------------


def test_compare_with_none_marks_comsol_unavailable_and_fabricates_nothing():
    python_result = SimpleNamespace(
        t_grid_s=np.linspace(0.0, 8.0e-6, 6),
        T_center_K=np.full(6, 1650.0),
        T_max_K=np.full(6, 1655.0),
        T_area_average_K=np.full(6, 1648.0),
    )
    comparison = rg.compare_python_comsol_thermal(python_result, None)

    assert comparison.comsol_available is False
    assert comparison.aligned_time_s is None
    assert comparison.T_center_diff_K is None
    assert comparison.T_max_diff_K is None
    assert comparison.T_area_average_diff_K is None
    assert comparison.max_abs_temperature_diff_K is None
    assert comparison.mean_abs_temperature_diff_K is None
    assert comparison.surface_diff_map_K is None
    assert comparison.surface_diff_norm_K is None
    assert comparison.hotspot_displacement_m is None
    assert "unavailable" in comparison.notes.lower()


def test_compare_requires_t_grid_s_on_python_result(tmp_path):
    path = tmp_path / "synthetic_for_error.h5"
    write_synthetic_comsol_result_for_testing(path)
    comsol_result = rg.load_comsol_thermal_result(path)
    python_result = SimpleNamespace(T_center_K=np.zeros(3))
    with pytest.raises(ValueError, match="t_grid_s"):
        rg.compare_python_comsol_thermal(python_result, comsol_result)


# ------------------------------------------------------------------------------------------------
# 5. compare_python_comsol_thermal with both real: identical case -> exactly zero diff
# ------------------------------------------------------------------------------------------------


def test_compare_identical_python_and_comsol_gives_zero_diff(tmp_path):
    path = tmp_path / "synthetic_identical.h5"
    write_synthetic_comsol_result_for_testing(path, n_t=6, n_x=4, n_y=5)
    comsol_result = rg.load_comsol_thermal_result(path)

    python_result = SimpleNamespace(
        t_grid_s=comsol_result.time_s.copy(),
        T_center_K=comsol_result.T_center_K.copy(),
        T_max_K=comsol_result.T_max_K.copy(),
        T_area_average_K=comsol_result.T_area_average_K.copy(),
        T_surface_K=comsol_result.T_surface_full_K.copy(),
        x_centers_m=comsol_result.x_m.copy(),
        y_centers_m=comsol_result.y_m.copy(),
    )

    comparison = rg.compare_python_comsol_thermal(python_result, comsol_result)

    assert comparison.comsol_available is True
    np.testing.assert_allclose(comparison.T_center_diff_K, 0.0, atol=1e-10)
    np.testing.assert_allclose(comparison.T_max_diff_K, 0.0, atol=1e-10)
    np.testing.assert_allclose(comparison.T_area_average_diff_K, 0.0, atol=1e-10)
    assert comparison.max_abs_temperature_diff_K == pytest.approx(0.0, abs=1e-10)
    assert comparison.mean_abs_temperature_diff_K == pytest.approx(0.0, abs=1e-10)
    assert comparison.surface_diff_norm_K == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(comparison.surface_diff_map_K, 0.0, atol=1e-10)
    assert comparison.hotspot_displacement_m == pytest.approx(0.0, abs=1e-12)


def test_compare_with_real_but_nontrivial_difference(tmp_path):
    path = tmp_path / "synthetic_nontrivial.h5"
    write_synthetic_comsol_result_for_testing(path, n_t=6)
    comsol_result = rg.load_comsol_thermal_result(path)

    python_result = SimpleNamespace(
        t_grid_s=comsol_result.time_s.copy(),
        T_center_K=comsol_result.T_center_K.copy() + 5.0,
        T_max_K=comsol_result.T_max_K.copy() + 3.0,
    )
    comparison = rg.compare_python_comsol_thermal(python_result, comsol_result)

    assert comparison.comsol_available is True
    np.testing.assert_allclose(comparison.T_center_diff_K, 5.0, atol=1e-9)
    np.testing.assert_allclose(comparison.T_max_diff_K, 3.0, atol=1e-9)
    assert comparison.T_area_average_diff_K is None
    assert "T_area_average_K" in comparison.notes
    assert comparison.max_abs_temperature_diff_K == pytest.approx(5.0, abs=1e-9)
