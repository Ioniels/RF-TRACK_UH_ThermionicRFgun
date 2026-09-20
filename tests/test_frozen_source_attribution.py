"""Tests for rf_gun.frozen_source_attribution.

A full end-to-end test (actually calling run_frozen_source_attribution) needs real field maps and
a full RF-Track transport run per case, which is expensive; these tests instead verify the two
structural claims the whole module's docstring rests on, which don't need any of that:

1. FROZEN_SOURCE_ATTRIBUTION_CASES is well-formed (5 cases, valid VolumeBuildParams field names).
2. The same-seed-gives-identical-B0 claim: build_bunch_thermionic's signature never receives
   sc_enabled/mirror_charge_enabled/beam_loading_enabled at all (it takes f_hz and
   Ez0_phasor_axis, not a VolumeBuildParams), so two calls with the same rng seed and the same
   emission/tracking parameters produce a bit-for-bit identical bunch regardless of what those
   three flags will later be set to on the vol_params used for tracking -- confirmed here by
   inspecting the actual function signature, not just asserting it in a docstring.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import rf_gun as rg
from rf_gun.frozen_source_attribution import FROZEN_SOURCE_ATTRIBUTION_CASES
from rf_gun.rftrack_volume import VolumeBuildParams
from rf_gun.simulation import build_bunch_thermionic


def test_frozen_source_attribution_cases_are_well_formed():
    labels = [label for label, _ in FROZEN_SOURCE_ATTRIBUTION_CASES]
    assert len(labels) == 5
    assert len(set(labels)) == 5, "case labels must be unique"

    valid_fields = set(VolumeBuildParams.__dataclass_fields__.keys())
    for label, override in FROZEN_SOURCE_ATTRIBUTION_CASES:
        assert set(override.keys()) <= valid_fields, f"{label}: unknown VolumeBuildParams field in {override}"
        assert set(override.keys()) == {"sc_enabled", "mirror_charge_enabled", "beam_loading_enabled"}

    # Exactly one "RF only" (nothing on) and one "everything on" case, as the two extremes.
    all_off = [o for _, o in FROZEN_SOURCE_ATTRIBUTION_CASES if not any(o.values())]
    all_on = [o for _, o in FROZEN_SOURCE_ATTRIBUTION_CASES if all(o.values())]
    assert len(all_off) == 1
    assert len(all_on) == 1


def test_build_bunch_thermionic_signature_is_independent_of_vol_params():
    """build_bunch_thermionic must not accept sc_enabled/mirror_charge_enabled/
    beam_loading_enabled (or a VolumeBuildParams at all) -- this is what guarantees the frozen-
    source module's central claim (same seed -> same B0 regardless of those three flags)."""
    sig = inspect.signature(build_bunch_thermionic)
    param_names = set(sig.parameters.keys())
    assert "sc_enabled" not in param_names
    assert "mirror_charge_enabled" not in param_names
    assert "beam_loading_enabled" not in param_names
    assert "vol_params" not in param_names


def test_build_bunch_thermionic_same_seed_gives_identical_bunch():
    """Direct empirical confirmation of the seed-reproducibility claim, independent of the
    signature-inspection test above: two calls with the same seed and the same emission/tracking
    parameters (f_hz, phi_deg, Ez0_phasor_axis, params) must produce identical phase space."""
    emission = rg.EmissionParams(
        cathode_radius_mm=0.7, cathode_T_K=1700.0, work_function_eV=2.1,
        beta_field=1.0, emission_phase_range_deg=90.0, pz0_MeV_c=4.0e-3,
        pz_model="flux", emission_law="RDSchottky",
    )
    kwargs = dict(
        n=200, phi_deg=0.0, f_hz=2.856e9,
        params=emission, Ez0_phasor_axis=complex(-8.0e6, 0.0),
    )
    B0_a, thermo_a = build_bunch_thermionic(rg.rft, rng=np.random.default_rng(123), **kwargs)
    B0_b, thermo_b = build_bunch_thermionic(rg.rft, rng=np.random.default_rng(123), **kwargs)
    M_a = np.asarray(B0_a.get_phase_space(rg.EXTENDED_PHASE_FMT, "all"))
    M_b = np.asarray(B0_b.get_phase_space(rg.EXTENDED_PHASE_FMT, "all"))
    np.testing.assert_allclose(M_a, M_b)
    assert thermo_a.get("Q_total_C") == pytest.approx(thermo_b.get("Q_total_C"))
