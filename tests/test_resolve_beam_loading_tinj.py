"""Tests for `rf_gun.simulation._resolve_beam_loading_tinj`'s mean-based (not min-based)
`auto_from_emission` resolution, current-weighted where a current history is available.

Root cause this guards against (see the function's own docstring for the full, corrected
diagnosis): in real KOA production runs, `thermo_info["t_s"]` is the converged Emission Fields
Iteration's own fixed-bin-count time grid, and `min(t_s)` -- the center of its very first bin --
is a discretization artifact (shrinks with bin count, not with anything physical) that happened
to land RF-Track 2.7.0's BeamLoadingSW in a numerical-instability region (non-finite `get_TT1()`
after construction). The mean does not depend on bin count; weighting it by the actual emitted-
current history (`I_A_t`/`J_Apm2_t`) makes it the physical "center of charge injection" rather
than merely the grid's own midpoint. Pure Python/numpy -- no RF-Track dependency (B0 is a
lightweight mock exposing only `get_t0()`, itself only a defensive/dead code path in practice --
see the corrected docstring).
"""
from __future__ import annotations

import numpy as np
import pytest

from rf_gun.rftrack_volume import VolumeBuildParams
from rf_gun.simulation import _resolve_beam_loading_tinj


class _MockB0:
    def __init__(self, t0_mm_c):
        self._t0 = np.asarray(t0_mm_c, dtype=float)

    def get_t0(self):
        return self._t0


def _base_params(**overrides):
    defaults = dict(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=0.0, z_max_m=0.03,
        hr_m=1.0e-5, hz_m=1.0e-5, dt_mm=1.0,
        beam_loading_enabled=True, bl_tinj_mode="auto_from_emission",
    )
    defaults.update(overrides)
    return VolumeBuildParams(**defaults)


def test_uses_mean_not_min():
    # An asymmetric emission-time distribution where mean and min differ substantially.
    t0_mm_c = np.array([0.001, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = _resolve_beam_loading_tinj(_base_params(), _MockB0(t0_mm_c), thermo_info={})
    assert result.bl_tinj_mode == "manual"
    assert result.bl_tinj_manual_mm_c == pytest.approx(float(np.mean(t0_mm_c)))
    assert result.bl_tinj_manual_mm_c != pytest.approx(float(np.min(t0_mm_c)))


def test_large_n_mean_does_not_collapse_toward_zero_like_min_would():
    """The actual failure mode: as n grows, min(t0) shrinks toward the window's start, but the
    mean stays anchored near the distribution's true central tendency regardless of n."""
    rng = np.random.default_rng(0)

    def resolved_tinj(n):
        t0_mm_c = rng.uniform(0.0, 50.0, size=n)  # a stand-in emission-time window
        result = _resolve_beam_loading_tinj(_base_params(), _MockB0(t0_mm_c), thermo_info={})
        return result.bl_tinj_manual_mm_c, float(np.min(t0_mm_c))

    tinj_small_n, min_small_n = resolved_tinj(1_000)
    tinj_large_n, min_large_n = resolved_tinj(300_000)

    # The minimum collapses by roughly the sample-size ratio (order-statistics scaling); the
    # resolved (mean-based) tinj stays within a small factor of its small-n counterpart.
    assert min_large_n < min_small_n / 50.0
    assert tinj_large_n == pytest.approx(tinj_small_n, rel=0.5)
    assert tinj_large_n > 10.0  # comfortably large, not collapsed like the old min-based value


def test_disabled_beam_loading_is_a_no_op():
    params = _base_params(beam_loading_enabled=False)
    result = _resolve_beam_loading_tinj(params, _MockB0([1.0, 2.0]), thermo_info={})
    assert result is params


def test_manual_mode_is_a_no_op():
    params = _base_params(bl_tinj_mode="manual", bl_tinj_manual_mm_c=3.0)
    result = _resolve_beam_loading_tinj(params, _MockB0([1.0, 2.0]), thermo_info={})
    assert result is params


def test_falls_back_to_thermo_info_t_s_mean_when_get_t0_unavailable():
    class _NoT0:
        pass

    thermo_info = {"t_s": np.array([1e-10, 2e-10, 3e-10])}
    result = _resolve_beam_loading_tinj(_base_params(), _NoT0(), thermo_info=thermo_info)
    expected_mm_c = float(np.mean(thermo_info["t_s"])) * 2.99792458e8 * 1e3
    assert result.bl_tinj_manual_mm_c == pytest.approx(expected_mm_c, rel=1e-6)


class _NoT0:
    pass


def test_t_s_path_is_current_weighted_when_a_current_history_is_present():
    """The physically-meaningful case this function is actually exercised through in production
    (converged Emission Fields Iteration source): an asymmetric current profile should pull the
    resolved tinj toward where the charge is actually concentrated, not just the grid's midpoint."""
    t_s = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
    I_t = np.array([100.0, 0.0, 0.0, 0.0, 0.0])  # all current concentrated at the first bin
    thermo_info = {"t_s": t_s, "I_A_t": I_t}
    result = _resolve_beam_loading_tinj(_base_params(), _NoT0(), thermo_info=thermo_info)
    expected_mm_c = float(t_s[0]) * 2.99792458e8 * 1e3  # weighted mean collapses to the one bin carrying current
    plain_mean_mm_c = float(np.mean(t_s)) * 2.99792458e8 * 1e3
    assert result.bl_tinj_manual_mm_c == pytest.approx(expected_mm_c, rel=1e-6)
    assert result.bl_tinj_manual_mm_c != pytest.approx(plain_mean_mm_c, rel=1e-3)


def test_t_s_path_falls_back_to_plain_mean_when_current_all_zero():
    t_s = np.array([1e-10, 2e-10, 3e-10])
    I_t = np.zeros_like(t_s)
    thermo_info = {"t_s": t_s, "I_A_t": I_t}
    result = _resolve_beam_loading_tinj(_base_params(), _NoT0(), thermo_info=thermo_info)
    expected_mm_c = float(np.mean(t_s)) * 2.99792458e8 * 1e3
    assert result.bl_tinj_manual_mm_c == pytest.approx(expected_mm_c, rel=1e-6)


def test_t_s_path_falls_back_to_plain_mean_when_weight_shape_mismatched():
    thermo_info = {"t_s": np.array([1e-10, 2e-10, 3e-10]), "I_A_t": np.array([1.0, 2.0])}
    result = _resolve_beam_loading_tinj(_base_params(), _NoT0(), thermo_info=thermo_info)
    expected_mm_c = float(np.mean(thermo_info["t_s"])) * 2.99792458e8 * 1e3
    assert result.bl_tinj_manual_mm_c == pytest.approx(expected_mm_c, rel=1e-6)


def test_t_s_path_uses_j_apm2_t_when_i_a_t_absent():
    t_s = np.array([1e-10, 2e-10, 3e-10])
    J_t = np.array([0.0, 0.0, 50.0])  # all current concentrated at the last bin
    thermo_info = {"t_s": t_s, "J_Apm2_t": J_t}
    result = _resolve_beam_loading_tinj(_base_params(), _NoT0(), thermo_info=thermo_info)
    expected_mm_c = float(t_s[-1]) * 2.99792458e8 * 1e3
    assert result.bl_tinj_manual_mm_c == pytest.approx(expected_mm_c, rel=1e-6)
