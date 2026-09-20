from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from rf_gun.fieldmaps import (
    FrameTransform,
    XFD_TD_CATHODE_TO_BEAM,
    XFdtdH5,
    check_xfdtd_cathode_origin,
)


def _make_cathode_step_fixture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write an irregular-time E/H fixture with a physical PEC step at the origin."""

    frequency_hz = 1.37e9
    phase_samples = np.array([0.00, 0.071, 0.193, 0.337, 0.501, 0.692, 0.887, 1.137, 1.404])
    electric_time_s = phase_samples / frequency_hz
    magnetic_time_s = electric_time_s + 0.013 / frequency_hz
    reference_time_s = float(np.mean(electric_time_s))
    origin = np.asarray(XFD_TD_CATHODE_TO_BEAM.origin_native_m)

    # Ey owns deliberately staggered coordinates.  Under the project transform
    # beam z = -(native y-origin_y), beam x = native x-origin_x, and beam y =
    # native z-origin_z.  The closest on-axis samples are therefore offset by
    # +30 um and -40 um, which the result must expose rather than hide.
    offsets = {
        "x": np.array([-0.20, 0.03, 0.30]) * 1.0e-3,
        "y": np.array([-0.25, -0.05, 0.05, 0.15, 0.25]) * 1.0e-3,
        "z": np.array([-0.20, -0.04, 0.20]) * 1.0e-3,
    }
    native_axes = {
        name: origin[index] + offsets[name]
        for index, name in enumerate("xyz")
    }
    beam_z_native_order = -offsets["y"]
    amplitude = np.where(beam_z_native_order < 0.0, 100.0, 1.0e6)
    # Give the farthest vacuum point a distinct amplitude so the transition's
    # vacuum-signal fraction is tested against the actual line peak.
    amplitude[np.argmax(beam_z_native_order)] = 1.1e6
    expected_beam_phasor_native_order = amplitude * np.exp(0.37j)
    beam_slope_per_s = expected_beam_phasor_native_order * 2.0e5
    tau = electric_time_s - reference_time_s
    temporal_phase = np.exp(1j * 2.0 * np.pi * frequency_hz * tau)
    beam_samples = 17.0 + np.real(
        (expected_beam_phasor_native_order[None, :] + tau[:, None] * beam_slope_per_s[None, :])
        * temporal_phase[:, None]
    )
    # E_beam,z = -E_native,y for the project transform.
    native_ey_line = -beam_samples
    shape = (electric_time_s.size, 3, 5, 3)
    native_ey = np.broadcast_to(native_ey_line[:, None, :, None], shape).astype(np.float32)

    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": 1,
                "array_axis_order": "time,x,y,z",
                "serialized_drive_frequency_hz": frequency_hz,
                "port_power_w": 1.8e6,
                "simulation_timestep_s": 1.0e-13,
                "solver": "synthetic XFdtd",
                "source_run_id": "cathode-origin-unit-test",
            }
        )
        handle.create_dataset("/time/E_s", data=electric_time_s).attrs["units"] = "s"
        handle.create_dataset("/time/H_s", data=magnetic_time_s).attrs["units"] = "s"
        for quantity, units in (("E", "V/m"), ("H", "A/m")):
            for component in "xyz":
                label = quantity + component
                values = native_ey if label == "Ey" else np.zeros(shape, dtype=np.float32)
                dataset = handle.create_dataset(
                    f"/fields/{quantity}/{component}",
                    data=values,
                    chunks=(1, 1, 5, 1),
                )
                dataset.attrs["units"] = units
                dataset.attrs["native_coordinate_group"] = f"/coordinates/native/{label}"
                dataset.attrs["cathode_centered_coordinate_group"] = (
                    f"/coordinates/cathode_centered/{label}"
                )
                for axis_index, axis_name in enumerate("xyz"):
                    native = native_axes[axis_name]
                    centered = native - origin[axis_index]
                    handle.create_dataset(
                        f"/coordinates/native/{label}/{axis_name}_m", data=native
                    ).attrs["units"] = "m"
                    handle.create_dataset(
                        f"/coordinates/cathode_centered/{label}/{axis_name}_m", data=centered
                    ).attrs["units"] = "m"

    order = np.argsort(beam_z_native_order)
    return beam_z_native_order[order], expected_beam_phasor_native_order[order]


def test_physical_origin_check_uses_only_exact_time_normal_e_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "cathode_step.h5"
    expected_z, expected_phasor = _make_cathode_step_fixture(source)
    reads: list[tuple[str, str, object]] = []
    original_read = XFdtdH5.read_component

    def recording_read(self, quantity, component, selection):
        reads.append((quantity, component, selection))
        return original_read(self, quantity, component, selection)

    monkeypatch.setattr(XFdtdH5, "read_component", recording_read)
    check = check_xfdtd_cathode_origin(source)

    # The signed permutation selects native Ey, applies its minus sign, fixes
    # the component's own staggered x/z coordinates, and reads no other field.
    assert len(reads) == 1
    assert reads[0][:2] == ("E", "y")
    selection = reads[0][2]
    assert selection.time == slice(0, 9)
    assert selection.x == 1
    assert selection.y == slice(0, 5)
    assert selection.z == 1
    assert check.normal_native_component == "Ey"
    assert check.native_to_beam_normal_sign == -1.0
    assert check.fixed_native_indices == {"x": 1, "z": 1}
    np.testing.assert_allclose(check.beam_transverse_offset_m, [30.0e-6, -40.0e-6])
    np.testing.assert_allclose(check.beam_z_m, expected_z, atol=1.0e-15)
    np.testing.assert_allclose(
        check.normal_field_phasor_Vpm, expected_phasor, rtol=2.0e-6, atol=2.0e-4
    )
    assert check.logical_bytes_read == 9 * 5 * np.dtype(np.float32).itemsize

    assert check.material_sample_beam_z_m == pytest.approx(-50.0e-6)
    assert check.vacuum_sample_beam_z_m == pytest.approx(50.0e-6)
    assert check.transition_midpoint_m == pytest.approx(0.0, abs=1.0e-15)
    assert check.transition_resolution_m == pytest.approx(100.0e-6)
    assert check.field_suppression_ratio == pytest.approx(1.0e-4, rel=3.0e-6)
    assert check.transition_brackets_zero
    assert check.material_field_is_suppressed
    assert check.vacuum_signal_is_resolved
    assert check.origin_physically_bracketed
    assert check.time_sample_count == 9
    assert check.fit_global_nrmse < 1.0e-6
    assert check.transition_detection == "active_threshold_rising_edge"
    # The compact report is directly serializable without embedding line arrays.
    assert json.loads(json.dumps(check.as_dict()))["status"] == "pass"


def test_physical_origin_check_finds_shifted_transition_and_rejects_declared_zero(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shifted_origin.h5"
    _make_cathode_step_fixture(source)
    shifted_origin = np.array(XFD_TD_CATHODE_TO_BEAM.origin_native_m, copy=True)
    shifted_origin[1] -= 150.0e-6
    shifted_transform = FrameTransform(
        shifted_origin,
        XFD_TD_CATHODE_TO_BEAM.R_target_from_native,
    )

    check = check_xfdtd_cathode_origin(source, transform=shifted_transform)

    # The physical step is still detected, but it now lies 150 um material-side
    # of the claimed beam origin and therefore does not bracket z=0.
    assert check.transition_midpoint_m == pytest.approx(-150.0e-6)
    assert check.transition_resolution_m == pytest.approx(100.0e-6)
    assert check.field_suppression_ratio == pytest.approx(1.0e-4, rel=3.0e-6)
    assert not check.transition_brackets_zero
    assert check.material_field_is_suppressed
    assert check.vacuum_signal_is_resolved
    assert not check.origin_physically_bracketed
    assert check.as_dict()["status"] == "fail"

