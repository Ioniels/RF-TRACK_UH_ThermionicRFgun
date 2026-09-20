#!/usr/bin/env python
"""Build the Study I / I-bis field-grid convergence artifact ladder.

``KOA_slurm_scripts/study_i*_field_grid.slurm`` consume one immutable artifact per grid step
from ``field_maps/grid_convergence/``.  Nothing built them, so those studies could not run at
all.  This is that missing recipe: it calls ``prepare_xfdtd_fieldmap.py`` once per (dr, dz)
pair, naming each output exactly as the launchers expect.

On the ladder itself
--------------------
The historical ladder ran dr = 3.2 - 23.8 um and dz = 10.4 - 77.5 um.  Every one of those
steps is finer than the ~100 um XFdtd source cell, so the whole ladder only re-interpolated
the same solver samples at finer spacing: it could not converge to anything the source did not
already contain, and it never included the production grid.  The default ladder here instead
brackets the production grid (dr = 100 um, dz = 50 um) in sqrt(2) steps, from finer than the
source cell out to clearly under-resolved, so the study can actually show where the tracked
observables start to degrade.

Each build is a full out-of-core pass over the 24.8 GB source (order 30 minutes), so the ten
default steps are several hours.  Use ``--dry-run`` first, and ``--only`` to build a subset.

usage:
  build_grid_convergence_artifacts.py --dry-run
  build_grid_convergence_artifacts.py --only 3 4 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

#: Repository root: this tool lives in tools/, one level below the entry points it drives.
REPO = Path(__file__).resolve().parents[1]
PREPARE = REPO / "prepare_xfdtd_fieldmap.py"

#: sqrt(2) ladder bracketing the production grid (dr=100 um, dz=50 um) at index 3.
DEFAULT_DR_UM = [35.355339, 50.0, 70.710678, 100.0, 141.421356,
                 200.0, 282.842712, 400.0, 565.685425, 800.0]
DEFAULT_DZ_UM = [17.677670, 25.0, 35.355339, 50.0, 70.710678,
                 100.0, 141.421356, 200.0, 282.842712, 400.0]

#: The launchers interpolate the name with '%f' formatting -- six decimals, matching
#: cavity_axisymmetric_m0_dr${DR_UM}um_dz${DZ_UM}um_rftrack.h5 in the SLURM scripts.
NAME_TEMPLATE = "cavity_axisymmetric_m0_dr{dr:.6f}um_dz{dz:.6f}um_rftrack.h5"


def artifact_name(dr_um: float, dz_um: float) -> str:
    return NAME_TEMPLATE.format(dr=dr_um, dz=dz_um)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "field_maps" / "grid_convergence")
    ap.add_argument("--dr-um", type=float, nargs="+", default=DEFAULT_DR_UM)
    ap.add_argument("--dz-um", type=float, nargs="+", default=DEFAULT_DZ_UM)
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="build only these ladder indices")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and validate each configuration without reducing")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--fill-correction", choices=("auto", "off", "single-pole"), default="auto")
    ap.add_argument("--fill-start-time-ns", type=float, default=None)
    args = ap.parse_args(argv)
    if args.fill_correction == "single-pole" and args.fill_start_time_ns is None:
        ap.error("single-pole requires --fill-start-time-ns as an explicit assumption")
    if args.fill_start_time_ns is not None and args.fill_correction != "single-pole":
        ap.error("--fill-start-time-ns requires --fill-correction single-pole")

    if len(args.dr_um) != len(args.dz_um):
        raise SystemExit("--dr-um and --dz-um must have the same length")

    steps = list(enumerate(zip(args.dr_um, args.dz_um)))
    if args.only is not None:
        wanted = set(args.only)
        steps = [s for s in steps if s[0] in wanted]
        if not steps:
            raise SystemExit(f"--only {sorted(wanted)} selected no ladder step")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ladder steps: {len(steps)}   output: {args.out_dir}")
    for index, (dr, dz) in steps:
        print(f"  [{index}] dr={dr:.6f} um  dz={dz:.6f} um  -> {artifact_name(dr, dz)}")

    failures: list[int] = []
    for index, (dr, dz) in steps:
        out_path = args.out_dir / artifact_name(dr, dz)
        if args.skip_existing and out_path.is_file() and not args.dry_run:
            print(f"\n[{index}] exists, skipping: {out_path.name}")
            continue
        cmd = [
            sys.executable, str(PREPARE),
            "--output", str(out_path),
            "--dr-um", f"{dr:.6f}",
            "--dz-um", f"{dz:.6f}",
            "--overwrite",
            "--fill-correction", args.fill_correction,
        ]
        if args.fill_start_time_ns is not None:
            cmd.extend(["--fill-start-time-ns", str(args.fill_start_time_ns)])
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"\n=== [{index}] dr={dr:.6f} um dz={dz:.6f} um ===", flush=True)
        started = time.time()
        proc = subprocess.run(cmd, cwd=REPO)
        elapsed = time.time() - started
        if proc.returncode != 0:
            print(f"  FAILED (rc={proc.returncode}) after {elapsed:.1f}s", flush=True)
            failures.append(index)
        else:
            print(f"  ok in {elapsed:.1f}s", flush=True)

    if failures:
        print(f"\nfailed ladder steps: {failures}")
        return 1
    print("\nall requested ladder steps completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
