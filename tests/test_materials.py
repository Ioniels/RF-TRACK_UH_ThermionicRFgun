"""Tests for rf_gun.materials (implementation plan Sec. 5, "Selectable cathode-material property
system", and Sec. 19.3's TIO/Bakr electron-range verification findings)."""
from __future__ import annotations

import warnings

import pytest

import rf_gun as rg
from rf_gun.materials.electron_range import tio_range_um
from rf_gun.materials.lab6 import LAB6_MELTING_POINT_K
from rf_gun.materials.registry import required_components_for, validate_material_for

# plan Sec. 5.3 / 19.3 reference range table for LaB6 (Z_eff=40.447, A_eff=94.735 g/mol,
# rho0=4720 kg/m3).
_REF_ENERGY_KEV = [1, 10, 100, 300, 500, 1000]
_REF_RANGE_UM = [0.030, 0.623, 23.35, 126.2, 262.3, 663.5]
_Z_EFF = 40.447
_A_EFF = 94.735
_RHO0_KG_M3 = 4720.0


@pytest.mark.parametrize("E_keV,R_expected_um", list(zip(_REF_ENERGY_KEV, _REF_RANGE_UM)))
def test_tio_range_reproduces_reference_table(E_keV, R_expected_um):
    """See rf_gun/materials/electron_range.py's module docstring: this implementation uses
    Tabata's original b1=0.2335 coefficient (not Bakr's printed a1=2.335*A/Z^1.209), which is the
    convention found to reproduce this table -- a few percent tolerance covers rounding in the
    plan's own tabulated reference values."""
    R = tio_range_um(E_keV, _Z_EFF, _A_EFF, _RHO0_KG_M3)
    assert R == pytest.approx(R_expected_um, rel=0.05)


def test_tio_range_vectorized_matches_scalar_calls():
    import numpy as np

    R_vec = tio_range_um(_REF_ENERGY_KEV, _Z_EFF, _A_EFF, _RHO0_KG_M3)
    R_scalar = [tio_range_um(E, _Z_EFF, _A_EFF, _RHO0_KG_M3) for E in _REF_ENERGY_KEV]
    assert np.asarray(R_vec) == pytest.approx(R_scalar, rel=1e-9)


def test_tio_range_unit_conversion_hazard_regression():
    """Plan Sec. 19.3's explicit units hazard: rho0 is given in kg/m3 but Tabata's range law
    needs rho in g/cm3 (R = Rex/rho_g_cm3); an accidental 10x or 1000x unit-conversion bug must
    be caught. The ratio between two reference energies is invariant to a pure multiplicative
    unit-conversion bug (it cancels), so also check an absolute value directly -- together these
    catch both a wrong-ratio bug (rare) and a constant-factor bug (the actual hazard flagged)."""
    R_1000 = tio_range_um(1000.0, _Z_EFF, _A_EFF, _RHO0_KG_M3)
    R_1 = tio_range_um(1.0, _Z_EFF, _A_EFF, _RHO0_KG_M3)

    # Absolute value check: catches a 10x (a1 ambiguity) or 1000x (skipped kg/m3->g/cm3
    # conversion) error, either of which a pure ratio check alone would miss.
    assert R_1000 == pytest.approx(663.5, rel=0.05)

    # Ratio check: independently confirms the *shape* of R(E) is right (this would still pass
    # under a pure constant-factor bug, hence the absolute check above is required too).
    ratio_expected = 663.5 / 0.030
    assert (R_1000 / R_1) == pytest.approx(ratio_expected, rel=0.10)


def test_load_lab6_uh_recommended_v1_succeeds_and_manifest_has_all_components():
    material = rg.load_cathode_material("LaB6", "LaB6_UH_recommended_v1")
    manifest = material.to_manifest_dict()

    assert manifest["material_id"] == "LaB6"
    assert manifest["property_set"] == "LaB6_UH_recommended_v1"
    for component_name in ("thermal", "electron_deposition", "emission", "optical"):
        assert manifest[component_name] is not None, f"missing {component_name} in manifest"
        assert "dataset_id" in manifest[component_name] or "rho0" in manifest[component_name]

    # thermal sub-manifest carries rho0/cp/k provenance individually (plan Sec. 5.1: "the source
    # key used for each individual property").
    thermal_manifest = manifest["thermal"]
    for prop in ("rho0", "cp", "k"):
        assert prop in thermal_manifest
        assert thermal_manifest[prop]["dataset_id"]
        assert thermal_manifest[prop]["reference_key"]


def test_lab6_uh_recommended_v1_thermal_evaluates_in_range():
    material = rg.load_cathode_material("LaB6", "LaB6_UH_recommended_v1")
    cp = material.thermal.cp_J_kg_K(1500.0)
    k = material.thermal.k_W_m_K(1500.0)
    rho = material.thermal.rho_kg_m3(1500.0)
    assert cp > 0
    assert k > 0
    assert rho == pytest.approx(4720.0)


def test_lab6_uh_recommended_v1_electron_deposition_range():
    material = rg.load_cathode_material("LaB6", "LaB6_UH_recommended_v1")
    R = material.electron_deposition.range_um(1000.0)
    assert R == pytest.approx(663.5, rel=0.05)


def test_load_lab6_legacy_reproduces_check_values_at_1720K():
    material = rg.load_cathode_material("LaB6", "LaB6_Kowalczyk_PRSTAB120402_2014_legacy")
    T = 1720.0
    k = material.thermal.k_W_m_K(T)
    D = 12.7675 * (T - 273.15) ** (-0.640226) * 1.0e-4
    cp = material.thermal.cp_J_kg_K(T)

    assert k == pytest.approx(43.7, rel=0.02)
    assert D == pytest.approx(1.21e-5, rel=0.02)
    assert cp == pytest.approx(765.0, rel=0.02)

    assert material.emission.phi_fit_eV == pytest.approx(2.485)
    assert material.emission.A_R_A_cm2_K2 == pytest.approx(29.0)


def test_lab6_legacy_missing_electron_deposition_and_optical():
    material = rg.load_cathode_material("LaB6", "LaB6_Kowalczyk_PRSTAB120402_2014_legacy")
    assert material.electron_deposition is None
    assert material.optical is None
    with pytest.raises(ValueError):
        validate_material_for(material, "bb0_deposition")
    with pytest.raises(ValueError):
        validate_material_for(material, "uh_optical_diagnostic")
    # python_xy_layered only needs thermal, which the legacy set does supply.
    validate_material_for(material, "python_xy_layered")


def test_constant_verification_set_is_verification_only():
    material = rg.load_cathode_material("LaB6", "LaB6_constant_verification_v1")
    assert material.thermal.rho_kg_m3(1000.0) == pytest.approx(4720.0)
    assert material.thermal.cp_J_kg_K(1000.0) == pytest.approx(850.0)
    assert material.thermal.k_W_m_K(1000.0) == pytest.approx(45.0)
    assert material.electron_deposition is None


def test_paper_specific_partial_dataset_rejected_as_complete_material():
    with pytest.raises(ValueError):
        rg.load_cathode_material("LaB6", "LaB6_cp_Tanaka_OsakaThesis_1981")


def test_paper_specific_partial_dataset_directly_loadable_as_component():
    dataset = rg.load_material_component("LaB6", "LaB6_cp_Tanaka_OsakaThesis_1981")
    assert dataset.dataset_id == "LaB6_cp_Tanaka_OsakaThesis_1981"
    assert dataset.evaluate(1500.0) > 0


def test_scalar_property_dataset_refuses_downward_extrapolation():
    dataset = rg.load_material_component("LaB6", "LaB6_cp_Tanaka_OsakaThesis_1981")
    assert dataset.extrapolation == "linear_to_limit"
    with pytest.raises(ValueError):
        dataset.evaluate(500.0)  # below the tabulated 1000-2000 K range


def test_scalar_property_dataset_hard_limit_is_melting_not_table_edge():
    """The hard failure belongs at the physical limit (LaB6 melting, 2483 K), not at the last
    knot of a fitted table (2000 K).

    Every Study IV macropulse case aborted at 2000.003-2000.285 K -- fractions of a kelvin past
    the cp table's final knot -- which is a data-coverage limit, not a physics one.
    """
    dataset = rg.load_material_component("LaB6", "LaB6_cp_Tanaka_OsakaThesis_1981")
    assert dataset.extrapolation_limit_x == pytest.approx(LAB6_MELTING_POINT_K)

    # The band between the table edge and melting evaluates, but warns that it is extrapolated.
    with pytest.warns(RuntimeWarning, match="above the tabulated range"):
        value = dataset.evaluate(2000.2029548049272)  # an actual Study IV abort value
    assert value > 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Monotone continuation of the final segment, still near the Dulong-Petit plateau.
        assert dataset.evaluate(2000.0) < dataset.evaluate(2400.0) < 1.05 * dataset.evaluate(2000.0)

    # Past melting the solid-phase model is genuinely undefined and must still fail hard.
    with pytest.raises(ValueError, match="physical limit"):
        dataset.evaluate(LAB6_MELTING_POINT_K + 1.0)


def test_thermal_conductivity_shares_the_same_physical_limit():
    dataset = rg.load_material_component("LaB6", "LaB6_k_Tanaka_OsakaThesis_1981")
    assert dataset.extrapolation_limit_x == pytest.approx(LAB6_MELTING_POINT_K)
    with pytest.raises(ValueError, match="physical limit"):
        dataset.evaluate(LAB6_MELTING_POINT_K + 1.0)


def test_scalar_property_dataset_in_range_does_not_raise():
    dataset = rg.load_material_component("LaB6", "LaB6_cp_Tanaka_OsakaThesis_1981")
    value = dataset.evaluate(1500.0)
    assert value > 0


def test_unknown_material_id_raises():
    with pytest.raises(ValueError):
        rg.load_cathode_material("Dispenser", "anything")


def test_unknown_property_set_raises():
    with pytest.raises(ValueError):
        rg.load_cathode_material("LaB6", "not_a_real_property_set")


def test_required_components_for_unknown_operation_raises():
    with pytest.raises(ValueError):
        required_components_for("not_a_real_operation")


def test_required_components_for_known_operations():
    assert required_components_for("bb_event_analysis") == frozenset()
    assert required_components_for("uh_optical_diagnostic") == frozenset({"optical"})
    assert "thermal" in required_components_for("python_xy_layered")
    assert "electron_deposition" in required_components_for("bb0_deposition")


def test_exported_from_package():
    assert rg.load_cathode_material is not None
    material = rg.load_cathode_material("LaB6", "LaB6_UH_recommended_v1")
    assert isinstance(material, rg.CathodeMaterialSet)
