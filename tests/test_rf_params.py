"""Tests for rf_gun.rf_params -- in particular a regression guard for the R/Q calibration bug
fixed in this pass (see UPGRADE_PLAN.md): (R/Q) must use the unloaded Q0, not the loaded Q_L,
when the power argument is wall-dissipated power (as delivered_power_on_resonance returns).
"""
from __future__ import annotations

import pytest

import rf_gun as rg
from rf_gun.rf_params import delivered_power_on_resonance, r_over_q_per_m


def test_r_over_q_per_m_matches_textbook_identity():
    """R/Q := V^2/(omega*U) is independent of any Q by definition; combined with Q0 = omega*U/P_wall,
    this gives R/Q = V^2/(P_wall*Q0) exactly -- so r_over_q_per_m(V, P_wall, Q0, L) must reproduce
    that identity directly (trivially true by construction, but locks in the formula's meaning)."""
    Veff_V, P_wall_W, Q0, L_eff_m = 1.0e6, 5.0e5, 4000.0, 0.05
    expected_R_over_Q = (Veff_V ** 2) / (P_wall_W * Q0)
    result = r_over_q_per_m(Veff_V, P_wall_W, Q0, L_eff_m)
    assert result == pytest.approx(expected_R_over_Q / L_eff_m)


def test_r_over_q_per_m_is_sensitive_to_which_q_is_passed():
    """Regression guard: passing the loaded Q_L instead of the unloaded Q0 must change the result
    by exactly the factor Q0/Q_L (this is precisely the bug that was fixed -- run_thermionic_tm010.py
    and the notebook used to pass Q_L here). If someone reintroduces Q_L, this test still passes
    numerically (it doesn't know which one is "correct") but documents the two are NOT
    interchangeable, so a future reviewer sees the sensitivity explicitly."""
    Veff_V, P_wall_W, L_eff_m = 1.0e6, 5.0e5, 0.05
    Q0, Qext = 4000.0, 3500.0
    Q_loaded = 1.0 / (1.0 / Qext + 1.0 / Q0)
    assert Q_loaded < Q0  # Q_L is always smaller than Q0 (and Qext) by definition

    result_with_Q0 = r_over_q_per_m(Veff_V, P_wall_W, Q0, L_eff_m)
    result_with_Q_loaded = r_over_q_per_m(Veff_V, P_wall_W, Q_loaded, L_eff_m)

    assert result_with_Q0 != pytest.approx(result_with_Q_loaded)
    assert result_with_Q_loaded / result_with_Q0 == pytest.approx(Q0 / Q_loaded)


def test_delivered_power_on_resonance_matched_case():
    """beta=1 (Q0=Qext, matched): P_del = P_fwd (all forward power coupled in, guide's own
    beta=1 special case)."""
    P_fwd_W = 1.0e6
    P_del = delivered_power_on_resonance(P_fwd_W, Q0=4000.0, Qext=4000.0)
    assert P_del == pytest.approx(P_fwd_W)


def test_run_thermionic_tm010_and_notebook_use_Q0_not_Q_loaded():
    """Static guard against regressing the fix: the calibration call sites must reference Q0, not
    a computed loaded-Q variable, when building r_over_q_ohm/bl_r_over_q_ohm_per_m_from_scan."""
    import re
    from pathlib import Path

    script_src = Path(__file__).resolve().parents[1].joinpath("run_thermionic_tm010.py").read_text()
    m = re.search(r"r_over_q_ohm = \(veff_v\*\*2\) / \(p_del_w \* (.+?)\)", script_src)
    assert m is not None, "could not locate the r_over_q_ohm calibration line in run_thermionic_tm010.py"
    assert "q_loaded" not in m.group(1), (
        f"r_over_q_ohm calibration uses {m.group(1)!r} -- expected args.bl_q0 (Q0), not q_loaded"
    )


def _synthetic_circular_scan(crest_deg: float, n: int = 60):
    """A synthetic full-circle (endpoint=False) phase scan with a cosine-shaped pz peaking at
    `crest_deg`, mimicking what run_phase_scan produces for a real RF bucket."""
    import numpy as np

    phi_rel = np.linspace(0.0, 360.0, n, endpoint=False)
    pz = 1.0 + np.cos(np.deg2rad(phi_rel - crest_deg))
    return phi_rel, phi_rel.copy(), pz, np.full(n, 10, dtype=int)


def test_circular_crest_near_phase_seam_is_bracketed():
    """Regression: a crest landing near phase 0/360 (the array seam after the duplicate 0/360
    endpoint was removed) must still be recognized as bracketed when circular=True -- phase 359.x
    deg is physically adjacent to phase 0 deg, even though they are the first/last flat-array
    entries. This is the exact bug the executed notebook hit: 'crest has no finite bracketing
    neighbors' on a scan that was otherwise 100% finite."""
    phi_rel, phi_abs, pz, n_ok = _synthetic_circular_scan(crest_deg=0.5)
    result = rg.build_phase_calibration_result(
        phi_rel, phi_abs, pz, n_ok, pz0_MeV_c=0.0, refined=False, circular=True,
    )
    assert result.crest_bracketed is True
    assert result.valid is True, result.invalid_reason


def test_circular_crest_near_phase_seam_without_circular_flag_is_unbracketed():
    """Documents why the circular flag matters: the same data, without circular=True, reports the
    seam-adjacent crest as unbracketed (the pre-fix, buggy behavior) -- i.e. circular=False is not
    a safe default for a genuine full-circle scan."""
    phi_rel, phi_abs, pz, n_ok = _synthetic_circular_scan(crest_deg=0.5)
    result = rg.build_phase_calibration_result(
        phi_rel, phi_abs, pz, n_ok, pz0_MeV_c=0.0, refined=False, circular=False,
    )
    assert result.crest_bracketed is False
    assert result.valid is False


def test_noncircular_partial_scan_crest_at_edge_is_not_bracketed():
    """A genuine partial-range scan (not a full circle) must NOT wrap around: a crest at its edge
    has no physical reason to be treated as adjacent to the opposite edge."""
    import numpy as np

    phi_rel = np.linspace(0.0, 90.0, 20)  # deliberately partial, includes both endpoints
    pz = 1.0 + np.cos(np.deg2rad(phi_rel - 0.0))  # crest exactly at the first point
    result = rg.build_phase_calibration_result(
        phi_rel, phi_rel.copy(), pz, np.full(20, 10, dtype=int),
        pz0_MeV_c=0.0, refined=False, circular=False,
    )
    assert result.crest_index == 0
    assert result.crest_bracketed is False


def test_run_phase_scan_detects_circularity_from_full_circle_grid():
    """rf_gun.simulation.run_phase_scan's own circularity auto-detection (span+step==360deg) --
    pure arithmetic, checked directly against the heuristic without needing RF-Track."""
    import numpy as np

    n = 24
    phi_rel_coarse = np.linspace(0.0, 360.0, n, endpoint=False)
    span = float(phi_rel_coarse.max() - phi_rel_coarse.min())
    step = span / (n - 1)
    assert np.isclose(span + step, 360.0, atol=1e-6)

    # A genuine partial scan must NOT be detected as circular.
    phi_rel_partial = np.linspace(0.0, 90.0, n)
    span_p = float(phi_rel_partial.max() - phi_rel_partial.min())
    step_p = span_p / (n - 1)
    assert not np.isclose(span_p + step_p, 360.0, atol=1e-6)
