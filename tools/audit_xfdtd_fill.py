#!/usr/bin/env python3
"""Audit filling with bounded raw traces; never load a full field volume.

Run from the repository root:
  .venv/bin/python tools/audit_xfdtd_fill.py

The free-start profiles demonstrate identifiability limits. They are not fits
of the source ramp, measurements of Q, or a confidence interval on steady state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rf_gun.fieldmaps.harmonic import HarmonicFitConfig, fit_fixed_frequency_phasor
from rf_gun.fieldmaps.steady_state import fit_fill_transient
from rf_gun.fieldmaps.xfdtd_h5 import inspect_xfdtd_h5


def audit(source: Path) -> dict:
    manifest = inspect_xfdtd_h5(source)
    frequency = manifest.drive_frequency_hz
    duration = float(manifest.root_attributes["simulation_duration_s"])
    e_time, h_time = manifest.electric_time_s, manifest.magnetic_h_time_s
    reference = float((min(e_time[0], h_time[0]) + max(e_time[-1], h_time[-1])) / 2)
    config = HarmonicFitConfig(frequency, reference_time_s=reference, include_linear_envelope=True)
    # Native Ey is beam -Ez; native Hz is beam By. Constant sign does not affect growth.
    probes = {
        "Ey": [(0, .1), (1, .1), (0, 5), (1, 5), (0, 10), (1, 10), (0, 20), (1, 20), (0, 25)],
        "Hz": [(1, 5), (10, 5), (1, 10), (10, 10), (1, 20), (10, 20)],
        "Ex": [(1, .1), (1, 5), (10, 5), (1, 10), (10, 10), (1, 20)],
    }
    rows, profiles = [], []
    origin = manifest.cathode_origin_native_m
    with h5py.File(source, "r") as handle:
        for label, positions in probes.items():
            axes = [handle[f"coordinates/native/{label}/{a}_m"][:] for a in "xyz"]
            time = e_time if label.startswith("E") else h_time
            for radius_mm, z_mm in positions:
                native = origin + np.array([radius_mm, -z_mm, 0]) * 1e-3
                indices = tuple(int(np.argmin(abs(axis - value))) for axis, value in zip(axes, native))
                values = np.asarray(handle[f"fields/{label[0]}/{label[1].lower()}"][(slice(None),) + indices], dtype=float)
                fit = fit_fixed_frequency_phasor(time, values, config)
                gamma = complex(fit.phasor_slope_per_s / fit.phasor) / frequency
                half_growth = []
                for subset in (slice(0, 9), slice(8, 17)):
                    half = fit_fixed_frequency_phasor(time[subset], values[subset], config)
                    half_growth.append(float((half.phasor_slope_per_s / half.phasor).real / frequency))
                estimate = fit_fill_transient(gamma.real, frequency, duration, reference_time_s=reference)
                rows.append({
                    "native_component": label, "requested_radius_mm": radius_mm,
                    "requested_beam_z_mm": z_mm, "native_indices": list(indices),
                    "actual_beam_z_mm": float((origin[1] - axes[1][indices[1]]) * 1e3),
                    "phasor_amplitude": float(abs(fit.phasor)),
                    "signed_amplitude_drift_per_cycle": gamma.real,
                    "phase_drift_deg_per_cycle": float(np.rad2deg(gamma.imag)),
                    "linear_envelope_nrmse": fit.global_nrmse,
                    "half_window_amplitude_drifts_per_cycle": half_growth,
                    "ideal_step_at_zero_factor": estimate.correction_factor,
                })
                if label == "Ey" and radius_mm == 0 and z_mm == 10:
                    theta = 2 * np.pi * frequency * (time - reference)
                    # Forced and homogeneous amplitudes are independent; this
                    # removes the unverified zero-field-at-t=0 constraint.
                    for tau_ns in (40, 80, 120, 157, 207, 400, 1000):
                        decay = np.exp(-(time - reference) / (tau_ns * 1e-9))
                        design = np.column_stack((np.ones(time.size), np.cos(theta), np.sin(theta),
                                                  decay * np.cos(theta), decay * np.sin(theta)))
                        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
                        nrmse = float(np.std(design @ coefficients - values) / np.std(values))
                        profiles.append({
                            "assumed_tau_ns": tau_ns, "nrmse": nrmse,
                            "steady_to_reference_amplitude": float(abs(coefficients[1] - 1j * coefficients[2]) / abs(fit.phasor)),
                        })
    legacy_drift = 0.0024985923464796414  # exact archived report value; unsigned
    sensitivity = []
    for start_ns in (0, 2, 5, 10, 20):
        fit = fit_fill_transient(legacy_drift, frequency, duration,
                                 reference_time_s=reference, drive_start_time_s=start_ns * 1e-9)
        sensitivity.append({"assumed_step_start_ns": start_ns,
                            "factor": fit.correction_factor, "tau_ns": fit.tau_s * 1e9})
    return {
        "source": str(source.resolve()), "source_size_bytes": manifest.file_size_bytes,
        "frequency_hz": frequency, "reported_port_power_w": manifest.source_power_w,
        "port_evaluation_frequency_hz": manifest.root_attributes.get("port_evaluation_frequency_hz"),
        "power_normalization_verified": False, "steady_state_verified": False,
        "reference_time_s": reference, "simulated_duration_s": duration,
        "saved_window_duration_s": float(max(e_time[-1], h_time[-1]) - min(e_time[0], h_time[0])),
        "archived_unsigned_drift_per_cycle": legacy_drift,
        "legacy_end_time_factor": fit_fill_transient(legacy_drift, frequency, duration).correction_factor,
        "step_start_sensitivity_using_archived_drift": sensitivity,
        "raw_trace_fits": rows,
        "free_start_decay_profiles_at_axis_10mm": profiles,
        "interpretation": "Local growth is measured. Q and the steady-state factor depend on source-history and single-mode assumptions; profile values are not confidence limits.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("field_maps/cavity_E_H_full_volume.h5"))
    parser.add_argument("--output", type=Path, default=Path("docs/field_map_fill_audit.json"))
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("source and output must be distinct")
    report = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Audited {len(report['raw_trace_fits'])} raw traces; report: {args.output}")
    print(f"Legacy factor: {report['legacy_end_time_factor']:.9f}")
    print(f"Time-corrected factor, same unsigned drift: {report['step_start_sensitivity_using_archived_drift'][0]['factor']:.9f}")
    print(report["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
