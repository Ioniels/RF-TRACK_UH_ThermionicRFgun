from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from rf_gun.fieldmaps.coordinates import XFD_TD_CATHODE_TO_BEAM
from rf_gun.fieldmaps.comparison import compare_tail_control
from rf_gun.fieldmaps.models import AxisymmetricRFField
from rf_gun.fieldmaps.workflow import (
    apply_tail_policy,
    common_beam_frame_support,
    plan_axisymmetric_grid,
    qualify_evanescent_tail,
)


def _coordinate_only_fixture(path: Path) -> None:
    axes = {
        "x": np.linspace(-0.040, 0.040, 9),
        "y": np.linspace(-0.033, 0.008, 42),
        "z": np.linspace(-0.038, 0.038, 9),
    }
    with h5py.File(path, "w") as handle:
        for quantity in ("E", "H"):
            for component in "xyz":
                label = quantity + component
                for index, axis in enumerate("xyz"):
                    centered = axes[axis]
                    handle.create_dataset(
                        f"/coordinates/cathode_centered/{label}/{axis}_m", data=centered
                    )
                    handle.create_dataset(
                        f"/coordinates/native/{label}/{axis}_m",
                        data=centered + XFD_TD_CATHODE_TO_BEAM.origin_native_m[index],
                    )


def test_grid_plan_uses_common_support_and_aligns_measured_tail(tmp_path: Path):
    source = tmp_path / "coordinates.h5"
    _coordinate_only_fixture(source)
    support = common_beam_frame_support(str(source))
    assert support.z_max_m == pytest.approx(0.033)
    assert support.full_azimuth_radius_m == pytest.approx(0.038)

    plan = plan_axisymmetric_grid(
        str(source),
        target_z_max_m=0.040589,
        requested_dr_m=0.0001,
        requested_dz_m=0.00005,
    )
    assert plan.full_z_m[-1] == pytest.approx(0.040589)
    assert plan.measured_z_m[-1] <= support.z_max_m
    assert plan.measured_z_m[-1] + plan.dz_m > support.z_max_m
    assert plan.missing_length_m == pytest.approx(
        plan.target_z_max_m - plan.measured_z_max_m
    )
    assert plan.measured_vacuum_mask.shape == (
        plan.measured_z_m.size,
        plan.r_m.size,
    )
    # At the cathode the physical channel is narrow, while the cavity body is wide.
    assert np.count_nonzero(plan.measured_vacuum_mask[0]) < np.count_nonzero(
        plan.measured_vacuum_mask[100]
    )


def test_tail_qualification_and_modal_zero_policies_are_explicit(tmp_path: Path):
    source = tmp_path / "coordinates.h5"
    _coordinate_only_fixture(source)
    plan = plan_axisymmetric_grid(
        str(source),
        target_z_max_m=0.040589,
        requested_dr_m=0.001,
        requested_dz_m=0.0001,
        r_max_m=0.034,
    )
    z = plan.measured_z_m
    r = plan.r_m
    decay_length = 1.053e-3
    axial = 4.0e7 * np.exp(-z / decay_length) * np.exp(0.2j)
    # Keep a cavity-scale plateau before the downstream fitting window, then use
    # a clean evanescent tail from 28 mm onward.
    axial = np.where(
        z < 0.028,
        4.0e7 * np.exp(0.2j),
        4.0e7 * np.exp(-(z - 0.028) / decay_length) * np.exp(0.2j),
    )
    radial_shape = np.maximum(0.0, 1.0 - (r / r[-1]) ** 2)
    ez = axial[:, None] * radial_shape[None, :]
    er = 0.03 * ez * (r[None, :] / r[-1])
    bt = ez / 299_792_458.0
    bz = 0.01 * bt
    for values in (er, ez, bt, bz):
        values[~plan.measured_vacuum_mask] = 0.0
    field = AxisymmetricRFField(2.86541722e9, r, z, er, ez, bt, bz)

    qualification = qualify_evanescent_tail(
        field, target_z_max_m=plan.target_z_max_m, fit_start_z_m=0.030
    )
    assert qualification.passed
    assert qualification.metrics["endpoint_e_fraction_of_map_peak"] < 0.01
    modal = apply_tail_policy(field, plan, qualification, policy="modal")
    zero = apply_tail_policy(field, plan, qualification, policy="zero")
    np.testing.assert_allclose(modal.z_m, plan.full_z_m)
    np.testing.assert_allclose(zero.z_m, plan.full_z_m)
    assert np.any(np.abs(modal.Ez_Vpm[z.size :]) > 0.0)
    assert np.all(zero.Ez_Vpm[z.size :] == 0.0)
    assert modal.metadata["spatial_support"]["extension"].startswith("extrapolated")
    assert zero.metadata["spatial_support"]["extension"] == "explicit_zero_field_control"
    comparison = compare_tail_control(modal, zero, measured_z_max_m=z[-1])
    expected_difference = abs(np.trapezoid(modal.Ez_Vpm[:, 0] - zero.Ez_Vpm[:, 0], modal.z_m))
    assert comparison["complex_voltage_difference_V"] == pytest.approx(expected_difference)
    # A stale control with the same geometry but a different field amplitude is
    # not a tail-only comparison, even if both files are individually qualified.
    stale = AxisymmetricRFField(
        zero.frequency_hz, zero.r_m, zero.z_m,
        zero.Er_Vpm, 1.02 * zero.Ez_Vpm, zero.Btheta_T, zero.Bz_T,
    )
    with pytest.raises(ValueError, match="different measured Ez"):
        compare_tail_control(modal, stale, measured_z_max_m=z[-1])
