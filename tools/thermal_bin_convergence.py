#!/usr/bin/env python
"""Thermal macro-time-bin convergence study (the 20/10/5 ns pilot).

``run_back_bombardment_macropulse.py --thermal-bin-ns`` sets the macro-time bin width of the
layered cathode thermal solve.  The project plan calls for a 20/10/5 ns comparison before
adopting a production bin, but nothing tabulated the result, so the chosen bin has never been
shown to be converged.

This driver re-solves the *same* captured back-bombardment events at several bin widths and
reports how the peak temperature rise moves.  It implements no physics: it invokes the
production study once per bin and reads back what that study wrote, so the numbers it reports
are exactly the ones a production run would produce at each bin.

The Richardson-style estimate assumes monotone first-order convergence in the bin width; it is
reported as an indication, not as a proof, and is suppressed when the sequence is not monotone.

usage:
  thermal_bin_convergence.py --run-dir outputs/runs/<run> [--bins-ns 20 10 5]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

#: Repository root: this tool lives in tools/, one level below the entry points it drives.
REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "run_back_bombardment_macropulse.py"


def _peak_temperature_record(study_h5: Path) -> dict[str, Any]:
    """Pull the peak temperature and supporting series out of a completed study file.

    The study writes the ``back_bombardment_macropulse_v1`` schema, whose thermal histories
    live in the ``thermal/`` group as ``T_max_K``, ``T_center_K`` and friends.
    """

    preferred = "thermal/T_max_K"
    record: dict[str, Any] = {}
    with h5py.File(study_h5, "r") as handle:
        series: dict[str, dict[str, float]] = {}
        for name in ("thermal/T_max_K", "thermal/T_center_K", "thermal/T_flat_mean_K",
                     "thermal/T_bevel_mean_K", "thermal/T_area_average_K"):
            if name not in handle:
                continue
            values = np.asarray(handle[name][()], dtype=np.float64).ravel()
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            series[name] = {
                "max": float(np.max(finite)),
                "final": float(finite[-1]),
                "initial": float(finite[0]),
            }
        if not series:
            raise RuntimeError(
                f"no thermal/T_*_K history found in {study_h5}; the study schema may have "
                "changed (expected back_bombardment_macropulse_v1)"
            )
        key = preferred if preferred in series else max(
            series.items(), key=lambda kv: kv[1]["max"]
        )[0]
        record["series"] = series
        record["peak_dataset"] = key
        record["peak_temperature_k"] = series[key]["max"]
        record["initial_temperature_k"] = series[key]["initial"]
        record["peak_rise_k"] = series[key]["max"] - series[key]["initial"]
        record["schema_version"] = str(handle.attrs.get("schema_version", ""))
    return record


def run_one(run_dir: Path, bin_ns: float, out_root: Path, duration_us: float | None,
            initial_temperature_k: float, extra: list[str]) -> dict[str, Any]:
    out_dir = out_root / f"bin_{bin_ns:g}ns"
    cmd = [
        sys.executable, str(DRIVER),
        "--source-mode", "load_run",
        "--run-dir", str(run_dir),
        "--thermal-bin-ns", str(bin_ns),
        "--initial-temperature-uniform-k", str(initial_temperature_k),
        "--output", str(out_dir),
    ]
    if duration_us is not None:
        cmd += ["--macropulse-duration-us", str(duration_us)]
    cmd += extra
    print(f"\n=== thermal bin {bin_ns:g} ns ===", flush=True)
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        raise RuntimeError(f"bin {bin_ns} ns failed (rc={proc.returncode}):\n{tail}")

    candidates = sorted(out_dir.rglob("*macropulse*.h5")) or sorted(out_dir.rglob("*.h5"))
    if not candidates:
        raise RuntimeError(f"bin {bin_ns} ns produced no study HDF5 under {out_dir}")
    record = _peak_temperature_record(candidates[0])
    record.update({"bin_ns": float(bin_ns), "study_h5": str(candidates[0])})
    print(f"  peak T = {record['peak_temperature_k']:.4f} K  "
          f"[{record['peak_dataset']}]", flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="completed transport run holding back_bombardment_events.h5")
    ap.add_argument("--bins-ns", type=float, nargs="+", default=[20.0, 10.0, 5.0])
    ap.add_argument("--macropulse-duration-us", type=float, default=None)
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/thermal_bin_convergence"))
    ap.add_argument("--initial-temperature-uniform-k", type=float, default=1650.0,
                    help="uniform cathode start temperature forwarded to the study driver")
    ap.add_argument("--tolerance-k", type=float, default=0.5,
                    help="convergence claim threshold on the finest-pair peak-T difference")
    ap.add_argument("extra", nargs="*",
                    help="additional arguments forwarded verbatim to the study driver")
    args = ap.parse_args(argv)

    events = args.run_dir / "back_bombardment_events.h5"
    if not events.is_file():
        raise SystemExit(f"no back_bombardment_events.h5 in {args.run_dir}")

    bins = sorted({float(b) for b in args.bins_ns}, reverse=True)
    args.output.mkdir(parents=True, exist_ok=True)

    results = [
        run_one(args.run_dir, b, args.output, args.macropulse_duration_us,
                float(args.initial_temperature_uniform_k), list(args.extra))
        for b in bins
    ]

    peaks = [r["peak_temperature_k"] for r in results]
    print("\n" + "=" * 70)
    print(f"{'bin (ns)':>10} {'peak T (K)':>14} {'delta vs finest (K)':>22}")
    print("-" * 70)
    finest = peaks[-1]
    for record, peak in zip(results, peaks):
        print(f"{record['bin_ns']:>10.4g} {peak:>14.4f} {peak - finest:>22.4f}")
    print("=" * 70)

    summary: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "bins_ns": bins,
        "peak_temperature_k": peaks,
        "finest_bin_ns": bins[-1],
        "tolerance_k": float(args.tolerance_k),
        "initial_temperature_uniform_k": float(args.initial_temperature_uniform_k),
        "results": results,
    }

    if len(peaks) >= 2:
        last_change = abs(peaks[-1] - peaks[-2])
        summary["finest_pair_abs_change_k"] = last_change
        summary["converged"] = bool(last_change <= float(args.tolerance_k))
        print(f"finest-pair change: {last_change:.4f} K "
              f"(tolerance {args.tolerance_k:g} K) -> "
              f"{'CONVERGED' if summary['converged'] else 'NOT CONVERGED'}")

    if len(peaks) >= 3:
        d1, d2 = peaks[1] - peaks[0], peaks[2] - peaks[1]
        # Only meaningful when the sequence is monotone and still shrinking.
        if d1 != 0.0 and abs(d2) < abs(d1) and d1 * d2 > 0.0:
            extrapolated = peaks[2] + d2 * d2 / (d1 - d2)
            summary["richardson_extrapolated_peak_k"] = float(extrapolated)
            summary["richardson_remaining_error_k"] = float(abs(extrapolated - peaks[2]))
            print(f"Richardson estimate: {extrapolated:.4f} K "
                  f"(remaining error ~{abs(extrapolated - peaks[2]):.4f} K)")
        else:
            summary["richardson_extrapolated_peak_k"] = None
            print("Richardson estimate skipped: sequence is not monotonically converging.")

    out_json = args.output / "thermal_bin_convergence.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_json}")
    return 0 if summary.get("converged", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
