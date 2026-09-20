"""RF-Track 2.7's BeamLoadingSW constructor rejects tinj/tau == 0 exactly. bl_tinj_mode=
"auto_from_emission" (the default used by every SLURM scan script and the notebook) can produce
exactly zero; rf_gun.rftrack_volume._attach_beam_loading_sw floors it to a negligible nonzero
offset instead. These tests confirm every tinj mode/value combination builds a Volume without
crashing.

A real KOA production batch (after the tinj/tau floor above was already deployed) found the
floor itself, 1e-3, ALSO inside RF-Track's unsafe region at full production scale (n=300,000) --
the unsafe region is not simply "tinj/tau below some threshold". _attach_beam_loading_sw now
degrades gracefully (skips attaching beam loading, does not raise) whenever the post-construction
finiteness check fails for any reason, rather than relying on any single floor constant being
sufficient -- justified by tests/test_beam_loading_cross_validation.py's independent finding that
BeamLoadingSW already has zero measurable effect on tracked dynamics for this gun geometry.
test_beam_loading_gracefully_skipped_when_beam_loading_sw_reports_non_finite below exercises that
fallback directly, via a fake BeamLoadingSW (RF-Track's own implementation cannot be coerced into
failing this check from outside -- see rf_gun.simulation._resolve_beam_loading_tinj's docstring).
"""
from __future__ import annotations

import numpy as np

import rf_gun as rg


def _small_grid():
    return np.zeros((21, 5), dtype=complex), np.zeros((21, 5), dtype=complex)


def test_beam_loading_auto_from_emission_default_does_not_crash():
    Er, Ez = _small_grid()
    p = rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=-0.001, z_max_m=0.05,
        hr_m=1e-3, hz_m=2e-3, dt_mm=0.1,
        beam_loading_enabled=True, bl_Q_loaded=1866.67, bl_r_over_q_ohm_per_m=8458.0, bl_ncells=1,
        bl_tinj_mode="auto_from_emission", beam_loading_verbose=False,
    )
    V = rg.build_volume(rg.rft, Er, Ez, 0.0, p)
    assert V is not None


def test_beam_loading_manual_zero_tinj_does_not_crash():
    """A user explicitly passing bl_tinj_manual_mm_c=0.0 should hit the same floor, not the crash."""
    Er, Ez = _small_grid()
    p = rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=-0.001, z_max_m=0.05,
        hr_m=1e-3, hz_m=2e-3, dt_mm=0.1,
        beam_loading_enabled=True, bl_Q_loaded=1866.67, bl_r_over_q_ohm_per_m=8458.0, bl_ncells=1,
        bl_tinj_mode="manual", bl_tinj_manual_mm_c=0.0, beam_loading_verbose=False,
    )
    V = rg.build_volume(rg.rft, Er, Ez, 0.0, p)
    assert V is not None


def test_beam_loading_manual_nonzero_tinj_unaffected():
    Er, Ez = _small_grid()
    p = rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=-0.001, z_max_m=0.05,
        hr_m=1e-3, hz_m=2e-3, dt_mm=0.1,
        beam_loading_enabled=True, bl_Q_loaded=1866.67, bl_r_over_q_ohm_per_m=8458.0, bl_ncells=1,
        bl_tinj_mode="manual", bl_tinj_manual_mm_c=5.0, beam_loading_verbose=False,
    )
    V = rg.build_volume(rg.rft, Er, Ez, 0.0, p)
    assert V is not None


def test_beam_loading_small_but_nonzero_tinj_from_real_koa_failure_does_not_crash(capsys):
    """Regression test for a real production failure: every KOA_slurm_scripts/study_* family (5
    job arrays, ~55 failed tasks in one batch) crashed with 'BeamLoadingSW.get_TT1() returned
    non-finite value(s) after construction' at tinj/tau~8.7656e-06 -- reproducible only at full
    production particle count (n=300,000, via bl_tinj_mode="auto_from_emission"'s min(t0)
    collapsing toward the emission window's start for a large sample), not in an isolated unit
    test. rf_gun.simulation._resolve_beam_loading_tinj now uses mean(t0) instead (see
    tests/test_resolve_beam_loading_tinj.py), which alone should keep tinj/tau far from this
    region -- this test instead exercises _attach_beam_loading_sw's own independent floor
    (_MIN_SAFE_TINJ_TAU) directly, confirming that if a small tinj/tau value like the one actually
    observed ever reaches this function by any path, it no longer crashes.
    """
    f_hz = 2.856e9
    Q_loaded = 1866.6666666666665
    tau_s = 2.0 * Q_loaded / (2.0 * np.pi * f_hz)
    tinj_tau_observed = 8.7656e-6  # the exact ratio from the real failure logs
    tinj_manual_mm_c = tinj_tau_observed * tau_s * rg.c * 1e3

    Er, Ez = _small_grid()
    p = rg.VolumeBuildParams(
        f_hz=f_hz, map_z0_m=0.0, z_min_m=-0.001, z_max_m=0.05,
        hr_m=1e-3, hz_m=2e-3, dt_mm=0.1,
        beam_loading_enabled=True, bl_Q_loaded=Q_loaded, bl_r_over_q_ohm_per_m=3956.69, bl_ncells=1,
        bl_tinj_mode="manual", bl_tinj_manual_mm_c=tinj_manual_mm_c, beam_loading_verbose=False,
    )
    V = rg.build_volume(rg.rft, Er, Ez, 0.0, p)
    assert V is not None
    assert "numerical-stability floor" in capsys.readouterr().out


class _FakeBeamLoadingSW:
    """Stands in for RF-Track's own BeamLoadingSW, always reporting a non-finite get_TT1() --
    RF-Track's real implementation could not be coerced into failing this check from outside in
    any isolated test attempted this session (see rf_gun.simulation._resolve_beam_loading_tinj's
    docstring), so this directly exercises _attach_beam_loading_sw's post-construction fallback
    without depending on reproducing RF-Track's own internal numerical instability."""

    def __init__(self, *args, **kwargs):
        pass

    def get_Lcell(self):
        return 0.05

    def get_tfill(self):
        return 1.0e-7

    def get_tinj(self):
        return 1.0e-10

    def get_TT1(self):
        return float("nan")

    def get_TT2(self):
        return 1.0


class _RftWithBrokenBeamLoading:
    """Forwards every attribute to the real `rft` except `BeamLoadingSW`."""

    def __init__(self, real_rft):
        object.__setattr__(self, "_real", real_rft)

    def __getattr__(self, name):
        if name == "BeamLoadingSW":
            return _FakeBeamLoadingSW
        return getattr(self._real, name)


def test_beam_loading_gracefully_skipped_when_beam_loading_sw_reports_non_finite(capsys):
    """The real safety net (see this file's own module docstring): when BeamLoadingSW's post-
    construction finiteness check fails for ANY reason -- not just a floor that turned out to be
    insufficient -- the Volume is still returned, fully usable, just without beam loading
    attached, and a clear warning is printed rather than the run crashing."""
    Er, Ez = _small_grid()
    p = rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=-0.001, z_max_m=0.05,
        hr_m=1e-3, hz_m=2e-3, dt_mm=0.1,
        beam_loading_enabled=True, bl_Q_loaded=1866.67, bl_r_over_q_ohm_per_m=3956.69, bl_ncells=1,
        bl_tinj_mode="manual", bl_tinj_manual_mm_c=5.0, beam_loading_verbose=False,
    )
    V = rg.build_volume(_RftWithBrokenBeamLoading(rg.rft), Er, Ez, 0.0, p)
    assert V is not None
    out = capsys.readouterr().out
    assert "could not be safely constructed" in out
    assert "Continuing WITHOUT beam loading" in out
