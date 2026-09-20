"""Tests for `rf_gun.field_io.detect_pec_boundary_along_axis`: locating a PEC (perfectly-
conducting) boundary crossing -- e.g. the cathode's emitting face -- along a planar field map's
own y-axis, from the field data itself rather than a hand-set constant.

Synthetic field maps only (no dependency on the real, large `field_maps/*.mat` files): a small
regular (x, y) grid with a few time samples, where the field is a smooth decaying function of y
for y < y_pec and exactly the solver's own numerical noise floor for y >= y_pec.
"""
from __future__ import annotations

import numpy as np
import pytest

from rf_gun.field_io import detect_pec_boundary_along_axis

RNG = np.random.default_rng(0)


def _make_synthetic_fieldmap(y_pec_mm: float, *, x_values_mm=None, y_values_mm=None, noise_floor: float = 1.0e-7):
    x_values_mm = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) if x_values_mm is None else np.asarray(x_values_mm)
    y_values_mm = np.arange(10.0, 16.0, 0.25) if y_values_mm is None else np.asarray(y_values_mm)
    n_time = 6

    X, Y = np.meshgrid(x_values_mm, y_values_mm, indexing="ij")
    x_flat = X.ravel()
    y_flat = Y.ravel()
    n = x_flat.size
    vertices = np.column_stack([x_flat, y_flat, np.zeros(n)])

    open_mask = y_flat < y_pec_mm
    Ex = np.zeros((n, n_time))
    Ey = np.zeros((n, n_time))
    for it in range(n_time):
        vacuum_field = 1.0e7 * np.cos(0.3 * y_flat[open_mask] + it)
        Ex[open_mask, it] = vacuum_field
        Ey[open_mask, it] = 0.5 * vacuum_field
        Ex[~open_mask, it] = noise_floor * RNG.standard_normal(np.sum(~open_mask))
        Ey[~open_mask, it] = noise_floor * RNG.standard_normal(np.sum(~open_mask))

    return {"vertices": vertices, "Ex": Ex, "Ey": Ey}


def test_finds_the_exact_bracket_on_axis():
    fieldmap = _make_synthetic_fieldmap(y_pec_mm=13.0)
    result = detect_pec_boundary_along_axis(fieldmap, x_target_mm=0.0)
    assert result["x_actual_mm"] == pytest.approx(0.0)
    assert result["last_open_y_mm"] == pytest.approx(12.75)
    assert result["first_shadowed_y_mm"] == pytest.approx(13.0)
    assert result["midpoint_y_mm"] == pytest.approx(12.875)
    assert result["drop_ratio"] < 1.0e-4


def test_consistent_across_x_within_the_disk():
    fieldmap = _make_synthetic_fieldmap(y_pec_mm=13.0)
    for x_target in (-1.0, 0.0, 1.0, 2.0):
        result = detect_pec_boundary_along_axis(fieldmap, x_target_mm=x_target)
        assert result["last_open_y_mm"] == pytest.approx(12.75)
        assert result["first_shadowed_y_mm"] == pytest.approx(13.0)


def test_snaps_to_nearest_available_x_column():
    fieldmap = _make_synthetic_fieldmap(y_pec_mm=13.0)
    result = detect_pec_boundary_along_axis(fieldmap, x_target_mm=0.3)
    assert result["x_actual_mm"] == pytest.approx(0.0)  # nearest of [-2,-1,0,1,2]


def test_no_boundary_raises_value_error():
    """A column with no PEC crossing (smoothly decaying field everywhere) must raise, not report
    a spurious boundary from an ordinary field gradient."""
    y_values_mm = np.arange(10.0, 16.0, 0.25)
    x_values_mm = np.array([0.0])
    X, Y = np.meshgrid(x_values_mm, y_values_mm, indexing="ij")
    n = X.size
    vertices = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n)])
    Ex = np.zeros((n, 4))
    Ey = np.zeros((n, 4))
    for it in range(4):
        Ex[:, it] = 1.0e7 * np.exp(-0.05 * Y.ravel())  # smooth decay, no sharp drop anywhere
        Ey[:, it] = 0.5 * Ex[:, it]
    fieldmap = {"vertices": vertices, "Ex": Ex, "Ey": Ey}

    with pytest.raises(ValueError, match="no PEC boundary crossing"):
        detect_pec_boundary_along_axis(fieldmap, x_target_mm=0.0, zero_floor_ratio=1.0e-6)


def test_too_few_rows_raises_value_error():
    fieldmap = {
        "vertices": np.array([[0.0, 10.0, 0.0]]),
        "Ex": np.array([[1.0e7]]),
        "Ey": np.array([[5.0e6]]),
    }
    with pytest.raises(ValueError, match="fewer than 2 rows"):
        detect_pec_boundary_along_axis(fieldmap, x_target_mm=0.0)


def test_single_field_component_option():
    fieldmap = _make_synthetic_fieldmap(y_pec_mm=13.0)
    result = detect_pec_boundary_along_axis(fieldmap, x_target_mm=0.0, field_components=("Ex",))
    assert result["last_open_y_mm"] == pytest.approx(12.75)
    assert result["first_shadowed_y_mm"] == pytest.approx(13.0)
