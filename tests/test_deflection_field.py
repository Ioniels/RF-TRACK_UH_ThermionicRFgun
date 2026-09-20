"""Deflection-field geometry/sign checks and the configured-vs-applied current distinction
(RF_GUN_REPOSITORY_UPGRADE_INSTRUCTIONS.md Priority B3).

Verifies: (1) RF-Track's own `UserField.get_field()` query matches the analytic `b0_deflection_T`
function it is supposed to reproduce, at real field-query points; (2) `configured_current_A` (what
was requested) and `applied_current_A` (what the run actually experienced) are distinct, with
`applied_current_A` exactly zero whenever the magnet is disabled regardless of the configured
value; (3) `plot_deflection_field_profile` refuses to draw a nonzero-looking curve for a
disabled/unapplied current.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from rf_gun.deflection_field import DeflectionField, b0_deflection_T
from rf_gun.plotting.fields import plot_deflection_field_profile
from rf_gun.rftrack_volume import VolumeBuildParams


def test_userfield_get_field_matches_analytic_b0_deflection_T():
    current_A = 0.8
    length_m = 0.05
    field = DeflectionField(length_m, current_A)

    z_mm_samples = np.linspace(-40.0, 40.0, 9)
    for z_mm in z_mm_samples:
        _, B = field.get_field(0.0, 0.0, float(z_mm), 0.0)
        expected = float(b0_deflection_T(z_mm, current_A))
        assert np.isclose(float(B[0]), expected, rtol=1e-10), (
            f"UserField.get_field Bx={B[0]!r} T at z={z_mm} mm does not match the analytic "
            f"b0_deflection_T={expected!r} T it is supposed to reproduce"
        )
        assert B[1] == 0.0 and B[2] == 0.0, "deflection field must be purely horizontal (Bx only)"


def test_userfield_sign_follows_current_polarity():
    length_m = 0.05
    field_pos = DeflectionField(length_m, 0.5)
    field_neg = DeflectionField(length_m, -0.5)
    _, B_pos = field_pos.get_field(0.0, 0.0, 0.0, 0.0)
    _, B_neg = field_neg.get_field(0.0, 0.0, 0.0, 0.0)
    assert float(B_pos[0]) > 0.0
    assert float(B_neg[0]) < 0.0
    assert np.isclose(float(B_pos[0]), -float(B_neg[0]))


def test_applied_current_is_zero_when_disabled_regardless_of_configured_value():
    p_enabled = VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=0.0, z_max_m=0.05, hr_m=1e-3, hz_m=1e-3, dt_mm=0.1,
        deflection_enabled=True, deflection_current_A=0.8,
    )
    p_disabled = p_enabled.replace(deflection_enabled=False)

    assert p_enabled.applied_deflection_current_A == 0.8
    assert p_disabled.applied_deflection_current_A == 0.0
    # The configured value is untouched by enabling/disabling -- only "applied" changes.
    assert p_disabled.deflection_current_A == 0.8


def test_plot_deflection_field_profile_skips_when_not_applied():
    z_mm = np.linspace(-100.0, 100.0, 50)
    result = plot_deflection_field_profile(z_mm, configured_current_A=0.8, applied_current_A=0.0)
    assert result is None, "a disabled magnet must not produce a plotted (nonzero-looking) profile"


def test_plot_deflection_field_profile_draws_when_applied():
    z_mm = np.linspace(-100.0, 100.0, 50)
    result = plot_deflection_field_profile(z_mm, configured_current_A=0.8, applied_current_A=0.8)
    assert result is not None
    assert np.any(result["Bx_T"] != 0.0)
