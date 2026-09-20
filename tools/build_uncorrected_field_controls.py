#!/usr/bin/env python3
"""Build matching raw-amplitude controls from a corrected compact artifact.

Undo only the recorded real multiplier, then requalify and regenerate each
tail. This preserves the parent's spatial reduction and cathode boundary
treatment, unlike reusing a control from an earlier reducer revision. The
inverse operation has the rounding error of the parent's stored precision.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rf_gun.fieldmaps import (
    PRODUCTION_QUALITY_GATE_VERSION,
    extend_axisymmetric_with_zero_tail,
    prepare_fill_treatment,
    production_quality_gates,
    qualify_axisymmetric_field,
    qualify_evanescent_tail,
    read_axisymmetric_artifact,
    resolve_production_quality_status,
    write_axisymmetric_artifact,
)
from rf_gun.fieldmaps.tail import extend_axisymmetric_with_modal_tail
from rf_gun.fieldmaps.artifact import canonical_json


def build_controls(source: Path, output_dir: Path, *, overwrite: bool = False) -> list[dict]:
    names = {"modal": "cavity_axisymmetric_m0_rftrack.h5", "zero": "cavity_axisymmetric_m0_zero_tail_control.h5"}
    destinations = [output_dir / name for name in names.values()]
    if any(path.resolve() == source.resolve() for path in destinations):
        raise ValueError("control destinations must differ from the corrected source")
    for path in destinations:
        if not overwrite and (path.exists() or path.with_name(path.name + ".report.json").exists()):
            raise FileExistsError(path)
    parent = read_axisymmetric_artifact(source, verify="payload", require_qualified=True)
    field = parent.field
    if not field.metadata.get("fill_correction_applied"):
        raise ValueError("source has no recorded applied fill correction to undo")
    factor = float(field.metadata["fill_correction"]["correction_factor"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("invalid parent multiplier")
    if "target_power_w" in field.metadata:
        raise ValueError("source must retain its source power, without subsequent power scaling")
    if parent.vacuum_mask is None or parent.aperture_radius_m is None:
        raise ValueError("source requires a vacuum mask and aperture")
    quality = deepcopy(dict(parent.quality))
    measured_end = quality["tail_extension"]["metrics"]["measured_z_max_m"]
    count = int(np.count_nonzero(field.z_m <= measured_end + 1e-12))
    components = ("Er_Vpm", "Ez_Vpm", "Btheta_T", "Bz_T")
    metadata = dict(field.metadata)
    for key in ("fill_correction", "fill_treatment", "tail_qualification", "tail_model"):
        metadata.pop(key, None)
    metadata.update({
        "fill_correction_applied": False,
        "spatial_support": quality["field_reduction"]["support"],
        "amplitude_control_parent_payload_sha256": parent.payload_sha256,
        "removed_fill_factor": factor,
    })
    measured = replace(field, z_m=field.z_m[:count], metadata=metadata,
                       **{key: getattr(field, key)[:count] / factor for key in components})
    measured, fill = prepare_fill_treatment(
        measured, quality["field_reduction"]["temporal"], policy="off",
        simulated_duration_s=field.metadata["fill_correction"]["simulated_duration_s"],
    )
    config = deepcopy(parent.provenance["processing_config"])
    tail_start = config["tail_fit_start_m"]
    tail = qualify_evanescent_tail(measured, target_z_max_m=float(field.z_m[-1]), fit_start_z_m=tail_start)
    previous_numerical = quality["numerical_measured_support"]
    core = previous_numerical["maxwell_beam_core"]
    numerical = qualify_axisymmetric_field(
        measured, parent.vacuum_mask[:count], tail_start_z_m=tail_start,
        wall_guard_cells=core["wall_guard_cells"], axis_guard_cells=core["axis_guard_cells"],
        beam_core_radius_m=core["radius_m"],
    )
    gates = production_quality_gates(quality["field_reduction"], numerical, tail.as_dict(), quality["cathode_origin"])
    gates["source_integrity"] = deepcopy(quality["gates"]["source_integrity"])
    gates["fill_treatment"] = {
        "passed": True, "policy": "off", "steady_state_verified": False,
        "gate_version": PRODUCTION_QUALITY_GATE_VERSION, "name": "fill_treatment",
        "metric": "explicit raw-amplitude control from payload-verified parent",
    }
    status = resolve_production_quality_status("qualified", gates)
    quality.update({"status": status, "gates": gates, "failed_gates": [],
                    "fill_transient": fill, "numerical_measured_support": numerical,
                    "tail_extension": tail.as_dict()})
    config.update({"fill_treatment": fill, "amplitude_control_parent_payload_sha256": parent.payload_sha256})
    records = []
    for policy, name in names.items():
        if policy == "modal":
            extended = extend_axisymmetric_with_modal_tail(measured, float(field.z_m[-1]), tail.model, require_aligned=True)
        else:
            extended = extend_axisymmetric_with_zero_tail(measured, float(field.z_m[-1]))
        arrays = {key: np.where(parent.vacuum_mask, getattr(extended, key), 0) for key in components}
        extended = replace(extended, metadata={**extended.metadata, "tail_qualification": tail.as_dict()}, **arrays)
        np.testing.assert_allclose(extended.z_m, field.z_m, rtol=0, atol=2e-12)
        # Restoring the factor must recover the parent's measured spatial field,
        # including its cathode treatment; tolerate only stored floating precision.
        for key in components:
            reference = getattr(field, key)[:count]
            np.testing.assert_allclose(getattr(extended, key)[:count] * factor, reference,
                                       rtol=3e-7, atol=1e-12 * max(1.0, float(abs(reference).max())))
        quality["variant"] = "modal" if policy == "modal" else "zero_tail_control"
        config["tail_policy"] = policy
        provenance = deepcopy(dict(parent.provenance))
        provenance.pop("processing_config_sha256", None)
        provenance.update({"processing_config": config, "created_by": "tools/build_uncorrected_field_controls.py",
                           "created_utc": datetime.now(timezone.utc).isoformat(),
                           "amplitude_control_parent_path": str(source.resolve()),
                           "amplitude_control_parent_payload_sha256": parent.payload_sha256})
        destination = output_dir / name
        digest = write_axisymmetric_artifact(destination, extended, transform=parent.transform,
                    quality=quality, provenance=provenance, vacuum_mask=parent.vacuum_mask,
                    aperture_radius_m=parent.aperture_radius_m, overwrite=overwrite)
        verified = read_axisymmetric_artifact(destination, verify="payload", require_qualified=True)
        assert verified.payload_sha256 == digest
        record = {"artifact_path": str(destination.resolve()), "artifact_payload_sha256": digest,
                  "status": status, "quality": deepcopy(quality),
                  "source": parent.provenance.get("source_manifest"), "grid_plan": config["grid_plan"],
                  "parent_payload_sha256": parent.payload_sha256, "removed_fill_factor": factor}
        destination.with_name(destination.name + ".report.json").write_text(
            json.dumps(json.loads(canonical_json(record)), indent=2, allow_nan=False) + "\n"
        )
        records.append(record)
        print(f"Wrote and verified {destination}: {digest}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_controls(args.source, args.output_dir, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
