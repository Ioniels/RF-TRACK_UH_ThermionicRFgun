from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from rf_gun.fieldmaps.artifact import (
    ArtifactValidationError,
    canonical_json,
    canonical_json_sha256,
    load_axisymmetric_artifact,
    read_axisymmetric_artifact,
    write_axisymmetric_artifact,
)
from rf_gun.fieldmaps.coordinates import MU0_HPM, FrameTransform, XFD_TD_CATHODE_TO_BEAM
from rf_gun.fieldmaps.harmonic import HarmonicFitConfig
from rf_gun.fieldmaps.models import AxisymmetricRFField
from rf_gun.fieldmaps.pipeline import ReductionConfig, reduce_xfdtd_to_axisymmetric
from rf_gun.fieldmaps.sections import extract_axis_aligned_section, extract_component_block
from rf_gun.fieldmaps.xfdtd_h5 import (
    H5Selection,
    XFdtdH5,
    inspect_xfdtd_h5,
    validate_xfdtd_h5,
)


def _make_xfdtd_fixture(path: Path) -> np.ndarray:
    shape = (5, 4, 3, 3)
    frame, x, y, z = np.indices(shape)
    data = (1000 * frame + 100 * x + 10 * y + z).astype(np.float32)
    origin = np.array([0.0, 0.013, 0.346])
    axes = {
        "x": np.arange(shape[1], dtype=float),
        "y": np.arange(shape[2], dtype=float) * 2.0,
        "z": np.arange(shape[3], dtype=float),
    }
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": 1,
                "array_axis_order": "time,x,y,z",
                "serialized_drive_frequency_hz": 2.865e9,
                "port_power_w": 1.8e6,
                "simulation_timestep_s": 1.9e-13,
                "solver": "synthetic XFdtd",
                "source_run_id": "unit-test",
            }
        )
        handle.create_dataset("/time/E_s", data=np.arange(shape[0]) * 1.0e-10).attrs["units"] = "s"
        handle.create_dataset(
            "/time/H_s", data=np.arange(shape[0]) * 1.0e-10 + 0.5e-13
        ).attrs["units"] = "s"
        for quantity, units in (("E", "V/m"), ("H", "A/m")):
            for component in "xyz":
                label = quantity + component
                dataset = handle.create_dataset(
                    f"/fields/{quantity}/{component}",
                    data=data,
                    chunks=(1, 2, 3, 1),
                    compression="gzip",
                    fletcher32=True,
                )
                dataset.attrs["units"] = units
                dataset.attrs["native_coordinate_group"] = f"/coordinates/native/{label}"
                dataset.attrs["cathode_centered_coordinate_group"] = (
                    f"/coordinates/cathode_centered/{label}"
                )
                for index, axis in enumerate("xyz"):
                    centered = axes[axis] * 1.0e-3
                    handle.create_dataset(
                        f"/coordinates/cathode_centered/{label}/{axis}_m", data=centered
                    ).attrs["units"] = "m"
                    handle.create_dataset(
                        f"/coordinates/native/{label}/{axis}_m", data=centered + origin[index]
                    ).attrs["units"] = "m"
    return data


def test_xfdtd_reader_inspects_metadata_and_only_permits_bounded_reads(tmp_path: Path):
    path = tmp_path / "synthetic.h5"
    data = _make_xfdtd_fixture(path)
    manifest = inspect_xfdtd_h5(path)
    assert manifest.array_axes == ("time", "x", "y", "z")
    assert manifest.components["Ey"].shape == data.shape
    np.testing.assert_allclose(manifest.cathode_origin_native_m, [0.0, 0.013, 0.346])
    assert validate_xfdtd_h5(path).valid

    with XFdtdH5(path) as reader:
        selection = H5Selection(slice(0, 5), slice(0, 4), slice(0, 3), slice(0, 1))
        estimate = reader.estimate_read("E", "x", selection)
        assert estimate.selected_shape == (5, 4, 3, 1)
        np.testing.assert_array_equal(reader.read_component("E", "x", selection), data[..., :1])
        with pytest.raises(ValueError, match="full-volume"):
            reader.read_component(
                "E", "x", H5Selection(slice(0, 5), slice(0, 4), slice(0, 3), slice(0, 3))
            )
        with pytest.raises(ValueError, match="explicit start and stop"):
            reader.read_component(
                "E", "x", H5Selection(slice(None), slice(0, 1), slice(0, 3), slice(0, 2))
            )


@pytest.mark.parametrize(
    ("coordinate_path", "replacement"),
    [
        (
            "/coordinates/cathode_centered/Ex/x_m",
            np.array([0.0, 1.1e-3, 1.9e-3, 3.0e-3]),
        ),
        (
            "/coordinates/cathode_centered/Ex/z_m",
            np.array([0.0, 0.8e-3, 2.0e-3]),
        ),
    ],
    ids=("locally_warped_axis", "mispaired_yee_axis"),
)
def test_xfdtd_rejects_nonrigid_native_to_centered_coordinate_pairs(
    tmp_path: Path,
    coordinate_path: str,
    replacement: np.ndarray,
):
    path = tmp_path / "corrupt_coordinates.h5"
    _make_xfdtd_fixture(path)
    # Both corruptions remain finite and strictly increasing, and their
    # pointwise offset has the same median as the valid rigid translation.  A
    # median-only origin check therefore accepted them before the pointwise
    # coordinate-frame invariant was added.
    with h5py.File(path, "r+") as handle:
        handle[coordinate_path][...] = replacement

    with pytest.raises(ValueError, match="not related by one rigid translation"):
        inspect_xfdtd_h5(path)
    report = validate_xfdtd_h5(path)
    assert not report.valid
    assert any(
        issue.code == "open_or_schema"
        and "not related by one rigid translation" in issue.message
        for issue in report.errors
    )


def test_axis_aligned_section_reads_only_bracketing_planes(tmp_path: Path):
    path = tmp_path / "synthetic.h5"
    _make_xfdtd_fixture(path)
    with XFdtdH5(path) as reader:
        section = extract_axis_aligned_section(
            reader,
            "E",
            "y",
            fixed_axis="z",
            coordinate_m=0.5e-3,
            coordinate_frame="cathode_centered",
        )
    assert section.free_axes == ("x", "y")
    assert section.values.shape == (5, 4, 3)
    frame, x, y = np.indices((5, 4, 3))
    expected = 1000 * frame + 100 * x + 10 * y + 0.5
    np.testing.assert_allclose(section.values, expected)


def test_component_block_uses_physical_bounds_and_preserves_yee_axes(tmp_path: Path):
    path = tmp_path / "synthetic.h5"
    _make_xfdtd_fixture(path)
    with XFdtdH5(path) as reader:
        block = extract_component_block(
            reader,
            "H",
            "z",
            x_bounds_m=(1.0e-3, 2.0e-3),
            y_bounds_m=(2.0e-3, 2.0e-3),
            z_bounds_m=(1.0e-3, 1.0e-3),
            guard_cells=0,
        )
    assert block.quantity == "H"
    assert block.values.shape[0] == 5
    assert block.values.shape[1:] == (2, 1, 1)
    np.testing.assert_allclose(block.x_m, [1.0e-3, 2.0e-3])


def _artifact_field() -> AxisymmetricRFField:
    r = np.linspace(0.0, 2.0e-3, 3)
    z = np.linspace(0.0, 4.0e-3, 5)
    Z, R = np.meshgrid(z, r, indexing="ij")
    base = (1.0 + R + 2.0 * Z) * np.exp(0.3j)
    return AxisymmetricRFField(
        2.86541722e9,
        r,
        z,
        base,
        2.0 * base,
        1.0e-3 * base,
        1.0e-4 * base,
        metadata={"representation": "m0", "reference_time_s": 99e-9},
    )


def _make_axisymmetric_xfdtd_fixture(path: Path) -> tuple[FrameTransform, complex]:
    frequency = 1.0e9
    e_time = np.arange(9, dtype=float) / (4.0 * frequency)
    h_time = e_time + 0.05e-12
    reference = 0.5 * (e_time[0] + h_time[-1])
    x_axis = np.linspace(-2.0e-3, 2.0e-3, 5)
    y_axis = np.array([-4.0, -3.0, -2.0, -1.0, 0.5, 1.5]) * 1.0e-3
    z_axis = np.linspace(-2.0e-3, 2.0e-3, 5)
    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    common_phase = 1.0 + 0.2j
    component_phasors = {
        "Ex": 2.0e6 * X * common_phase,
        "Ey": -(5.0e6 - 1.0e8 * Y) * common_phase,
        "Ez": 2.0e6 * Z * common_phase,
        "Hx": (-4.0 * Z / MU0_HPM) * common_phase,
        "Hy": np.zeros_like(X, dtype=np.complex128),
        "Hz": (4.0 * X / MU0_HPM) * common_phase,
    }
    # Deliberately make the metal side zero so interpolation through y=0 would be wrong.
    for values in component_phasors.values():
        values[:, Y[0, :, 0] > 0.0, :] = 0.0
    origin = np.array([0.0, 0.013, 0.346])
    rotation = XFD_TD_CATHODE_TO_BEAM.R_target_from_native
    transform = FrameTransform(origin, rotation)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": 1,
                "array_axis_order": "time,x,y,z",
                "serialized_drive_frequency_hz": frequency,
                "port_power_w": 1.8e6,
                "simulation_timestep_s": 1.0e-13,
                "solver": "synthetic XFdtd",
                "source_run_id": "axisymmetric-unit-test",
            }
        )
        handle.create_dataset("/time/E_s", data=e_time).attrs["units"] = "s"
        handle.create_dataset("/time/H_s", data=h_time).attrs["units"] = "s"
        for label, phasor in component_phasors.items():
            quantity = label[0]
            component = label[1].lower()
            times = e_time if quantity == "E" else h_time
            temporal_phase = np.exp(1j * 2.0 * np.pi * frequency * (times - reference))
            data = np.real(temporal_phase[:, None, None, None] * phasor[None, ...]).astype(np.float32)
            dataset = handle.create_dataset(
                f"/fields/{quantity}/{component}",
                data=data,
                chunks=(1, 2, 3, 3),
                compression="gzip",
                fletcher32=True,
            )
            dataset.attrs["units"] = "V/m" if quantity == "E" else "A/m"
            dataset.attrs["native_coordinate_group"] = f"/coordinates/native/{label}"
            dataset.attrs["cathode_centered_coordinate_group"] = (
                f"/coordinates/cathode_centered/{label}"
            )
            for index, (axis_name, centered) in enumerate(
                zip("xyz", (x_axis, y_axis, z_axis), strict=True)
            ):
                handle.create_dataset(
                    f"/coordinates/cathode_centered/{label}/{axis_name}_m", data=centered
                ).attrs["units"] = "m"
                handle.create_dataset(
                    f"/coordinates/native/{label}/{axis_name}_m", data=centered + origin[index]
                ).attrs["units"] = "m"
    return transform, common_phase


def test_out_of_core_pipeline_recovers_axisymmetric_e_and_b_with_surface_limit(tmp_path: Path):
    path = tmp_path / "axisymmetric.h5"
    transform, common_phase = _make_axisymmetric_xfdtd_fixture(path)
    r = np.array([0.0, 1.0e-3])
    z = np.arange(4, dtype=float) * 1.0e-3
    theta = 2.0 * np.pi * np.arange(16) / 16.0
    vacuum_mask = np.ones((z.size, r.size), dtype=bool)
    vacuum_mask[-1, -1] = False
    config = ReductionConfig(
        r_m=r,
        z_m=z,
        theta_rad=theta,
        vacuum_mask=vacuum_mask,
        transform=transform,
        harmonic=HarmonicFitConfig(1.0e9, include_offset=False),
        longitudinal_block_points=2,
        beam_core_radius_m=1.0e-3,
        surface_sample_count=3,
        m_max=3,
    )
    result = reduce_xfdtd_to_axisymmetric(path, config)
    expected_ez = np.broadcast_to(
        (5.0e6 + 1.0e8 * z)[:, None] * common_phase, (4, 2)
    ).copy()
    expected_ez[~vacuum_mask] = 0.0
    np.testing.assert_allclose(result.field.Ez_Vpm, expected_ez, rtol=2e-6)
    np.testing.assert_allclose(result.field.Er_Vpm[:, 0], 0.0, atol=1e-8)
    expected_er_edge = np.full(4, 2.0e3 * common_phase)
    expected_er_edge[-1] = 0.0
    np.testing.assert_allclose(result.field.Er_Vpm[:, 1], expected_er_edge, rtol=2e-6)
    np.testing.assert_allclose(result.field.Btheta_T[:, 0], 0.0, atol=1e-12)
    expected_btheta_edge = np.full(4, 4.0e-3 * common_phase)
    expected_btheta_edge[-1] = 0.0
    np.testing.assert_allclose(result.field.Btheta_T[:, 1], expected_btheta_edge, rtol=2e-6)
    np.testing.assert_allclose(result.field.Bz_T, 0.0, atol=1e-10)
    assert result.report.surface["applied"] is True
    assert result.report.symmetry["combined_e_nonaxisymmetric_fraction"] < 1.0e-6
    assert result.report.symmetry["combined_b_nonaxisymmetric_fraction"] < 1.0e-6
    assert result.report.symmetry["beam_core"]["combined_b_nonaxisymmetric_fraction"] < 1.0e-6


def test_artifact_round_trip_and_payload_digest(tmp_path: Path):
    path = tmp_path / "field.h5"
    field = _artifact_field()
    payload = write_axisymmetric_artifact(
        path,
        field,
        transform=XFD_TD_CATHODE_TO_BEAM,
        quality={
            "status": "qualified",
            "gates": {"synthetic_round_trip": {"passed": True}},
            "symmetry": {"B_non_m0": 0.12},
        },
        provenance={"source_sha256": "0" * 64, "processing_config": {"theta_count": 32}},
        preview={"signed_cut_Ez": field.Ez_Vpm[:, :2]},
        vacuum_mask=np.ones(field.shape, dtype=bool),
        aperture_radius_m=np.full(field.z_m.shape, field.r_m[-1]),
    )
    artifact = read_axisymmetric_artifact(path, verify="payload")
    assert artifact.payload_sha256 == payload
    assert artifact.quality["status"] == "qualified"
    np.testing.assert_allclose(artifact.field.Ez_Vpm, field.Ez_Vpm, rtol=2e-7)
    np.testing.assert_allclose(
        artifact.transform.R_target_from_native, XFD_TD_CATHODE_TO_BEAM.R_target_from_native
    )
    assert artifact.vacuum_mask is not None and np.all(artifact.vacuum_mask)
    assert load_axisymmetric_artifact(path).frequency_hz == field.frequency_hz


@pytest.mark.parametrize(
    "omitted_path",
    ["/fields/Bz_T", "/grid/vacuum_mask", "/preview/signed_cut_Ez"],
)
def test_artifact_rejects_incomplete_digest_table_even_if_aggregate_is_recomputed(
    tmp_path: Path,
    omitted_path: str,
):
    path = tmp_path / "incomplete_digest_table.h5"
    field = _artifact_field()
    write_axisymmetric_artifact(
        path,
        field,
        transform=XFD_TD_CATHODE_TO_BEAM,
        quality={"status": "qualified", "gates": {"synthetic": {"passed": True}}},
        provenance={"source_sha256": "c" * 64},
        preview={"signed_cut_Ez": field.Ez_Vpm[:, :2]},
        vacuum_mask=np.ones(field.shape, dtype=bool),
    )
    with h5py.File(path, "r+") as handle:
        table = json.loads(str(handle.attrs["payload_digests_json"]))
        del table[omitted_path]
        metadata_sha = str(handle.attrs["metadata_sha256"])
        handle.attrs["payload_digests_json"] = canonical_json(table)
        handle.attrs["payload_sha256"] = canonical_json_sha256(
            {"dataset_digests": table, "metadata_sha256": metadata_sha}
        )

    with pytest.raises(ArtifactValidationError, match="does not exactly cover"):
        read_axisymmetric_artifact(path, verify="payload")


def test_artifact_preserves_requested_complex128_field_precision(tmp_path: Path):
    path = tmp_path / "field_c128.h5"
    field = _artifact_field()
    assert field.Ez_Vpm.dtype == np.complex128
    write_axisymmetric_artifact(
        path,
        field,
        transform=XFD_TD_CATHODE_TO_BEAM,
        quality={"status": "qualified", "gates": {"synthetic": {"passed": True}}},
        provenance={"source_sha256": "a" * 64},
    )
    with h5py.File(path, "r") as handle:
        assert handle["/fields/Ez_Vpm"].dtype == np.dtype("<c16")
    loaded = read_axisymmetric_artifact(path, verify="payload")
    assert loaded.field.Ez_Vpm.dtype == np.complex128
    np.testing.assert_array_equal(loaded.field.Ez_Vpm, field.Ez_Vpm)


def test_artifact_rejects_analysis_only_by_default_and_detects_modified_payload(tmp_path: Path):
    path = tmp_path / "analysis.h5"
    field = _artifact_field()
    write_axisymmetric_artifact(
        path,
        field,
        transform=XFD_TD_CATHODE_TO_BEAM,
        quality={"status": "analysis_only"},
        provenance={"source_sha256": "f" * 64},
    )
    with pytest.raises(ArtifactValidationError, match="requires 'qualified'"):
        load_axisymmetric_artifact(path)
    read_axisymmetric_artifact(path, require_qualified=False)
    with h5py.File(path, "r+") as handle:
        handle["/fields/Ez_Vpm"][0, 0] += np.complex64(1.0 + 0.0j)
    with pytest.raises(ArtifactValidationError, match="payload digest mismatch"):
        read_axisymmetric_artifact(path, require_qualified=False, verify="payload")


@pytest.mark.parametrize(
    "quality",
    [
        {"status": "qualified"},
        {"status": "qualified", "gates": {}},
        {"status": "qualified", "gates": {"failed": {"passed": False}}},
        {"status": "qualified", "gates": {"not_boolean": {"passed": "true"}}},
    ],
)
def test_artifact_writer_rejects_unsubstantiated_qualified_status(
    tmp_path: Path,
    quality: dict,
):
    with pytest.raises(ArtifactValidationError, match="qualified artifact|qualified status"):
        write_axisymmetric_artifact(
            tmp_path / "invalid_qualified.h5",
            _artifact_field(),
            transform=XFD_TD_CATHODE_TO_BEAM,
            quality=quality,
            provenance={"source_sha256": "b" * 64},
        )
