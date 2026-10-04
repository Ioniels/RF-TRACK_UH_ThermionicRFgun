"""I/O helpers for simulation outputs."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .constants import c, q_e
from .diagnostics import build_screen_summary_from_phase_space, info_get_first
from .particle_tags import T_COL, ID_COL, plane_crossing_xy

LOST_COLUMNS = ["x", "px", "y", "py", "z", "pz", "t", "mass", "q", "N", "id"]

# Full Bunch6dT identifier string: X Px Y Py Z Pz then mass, charge, N, t0, id.
# Units (RF-Track manual Table 2.2): X/Y/Z [mm], Px/Py/Pz [MeV/c], m [MeV/c^2],
# Q [e+], N [# real particles per macro-particle], t0 [mm/c], id [#].
OPENPMD_PHASE_FMT = "%X %Px %Y %Py %Z %Pz %m %Q %N %t0 %id"

#: Bump when a field is added, removed, or its meaning changes in a way that would break a
#: reader written against an earlier version (implementation guide Sec. 15.3: "include a schema
#: version so later changes remain readable"). Independent of each other since the two files can
#: evolve separately (e.g. a new hardcoded_parameters key doesn't touch run_results.json's shape).
RUN_CONFIG_SCHEMA_VERSION = 1
#: 2: openpmd_exit_beam is the fixed-s exit-plane beam with plane-crossing x, y.
RUN_RESULTS_SCHEMA_VERSION = 2


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2, sort_keys: bool = True) -> Path:
    """Write JSON atomically: serialize to a temporary file in the same directory, then
    `os.replace()` it onto `path`. A reader (or a concurrent run/process) can never observe a
    partially written file -- either the old content or the fully new content, never a truncated
    write from a crash/kill mid-`json.dump`."""
    path = Path(path)
    # Each writer owns its temporary file, including concurrent writes to the
    # same destination. A serialization failure leaves the previous file intact.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            json.dump(to_json_safe(payload), stream, indent=indent, sort_keys=sort_keys, allow_nan=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return path


def to_json_safe(value: Any) -> Any:
    """Recursively sanitize objects for strict JSON (no NaN/Inf)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    return str(value)




#: Column layout of screen snapshots (`rf_gun.simulation.SCREEN_PHASE_FMT`:
#: "%X %Px %Y %Py %Z %Pz %id %t %E %K %x %y"); arrays from older runs stop after K_MeV.
SCREEN_EXTENDED_COLUMNS = [
    "x_mm", "px_MeV_c", "y_mm", "py_MeV_c", "z_mm", "pz_MeV_c", "id", "t_mm_c", "E_MeV", "K_MeV",
    "x_cross_mm", "y_cross_mm",
]


def save_screen_distributions_hdf5(
    output_dir: Path,
    z_snaps: Sequence[float],
    M_snaps: Sequence[np.ndarray] | None,
    I_snaps: Sequence[Any] | None,
    *,
    n_initial: int | None = None,
    q_total_C: float = 0.0,
    species: str = "electron",
    filename_stem: str = "screen",
    extra_attrs: dict[str, Any] | None = None,
    robust_summaries: Sequence[dict[str, Any]] | None = None,
) -> list[Path]:
    """Save every per-screen phase-space snapshot as its own openPMD-beamphysics HDF5 file.

    Unlike `save_beam_openpmd` (used for the final, aperture-clipped exit beam), this keeps every
    macroparticle RF-Track reports at the screen -- no forward-only or aperture filtering -- so
    each file reflects exactly what was present at that z, including backward-going or off-axis
    particles; downstream analysis can still filter by `%id`/position as needed (see
    `rf_gun.particle_tags`). File naming mirrors `save_beam_openpmd`'s `Bout_*.h5` convention:
    one screen index + z position per file, plus (optionally) run-identifying tags such as
    cathode temperature / space-charge / beam-loading passed via `filename_stem`.

    Metadata useful for later inspection (screen index, z, particle counts, transmission, mean/
    sigma pz, and any RF-Track `Screen.get_info()` readback) is written as HDF5 root attributes
    alongside the openPMD ParticleGroup data, so a single file is self-describing.
    """
    try:
        from pmd_beamphysics import ParticleGroup  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "openPMD-beamphysics is required to save screens in HDF5 format. "
            "Install it with 'pip install openpmd-beamphysics'."
        ) from exc

    if not z_snaps or M_snaps is None:
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n0 = int(n_initial) if n_initial is not None else 0
    if n0 <= 0 and len(M_snaps) > 0:
        first = np.asarray(M_snaps[0])
        if first.ndim == 2:
            n0 = int(first.shape[0])

    n_real_per_macro = (abs(float(q_total_C)) / q_e) / float(n0) if (n0 > 0 and q_total_C) else 0.0

    precomputed = [dict(x) for x in robust_summaries] if robust_summaries is not None else []

    saved_paths: list[Path] = []
    n_prev = n0
    n_screens = min(len(z_snaps), len(M_snaps))
    for i in range(n_screens):
        z_m = float(z_snaps[i])
        M = np.asarray(M_snaps[i], dtype=float)
        if M.ndim != 2 or M.shape[0] == 0 or M.shape[1] < 6:
            n_prev = 0
            continue

        n = int(M.shape[0])
        pz_MeVc = M[:, 5]
        weight_C = n_real_per_macro * q_e if n_real_per_macro > 0 else float(q_e)
        pg = screen_particle_group(M, weight_C, species=species)

        if i < len(precomputed):
            summary = dict(precomputed[i])
        else:
            summary = build_screen_summary_from_phase_space(
                M, screen_index=i, z_m=z_m, n_initial=n0, n_previous=n_prev,
            )
        n_prev = int(summary.get("N", n))

        info_i = I_snaps[i] if (I_snaps is not None and i < len(I_snaps)) else None

        file_path = out_dir / f"{filename_stem}_{i:04d}_z{z_m * 1e3:+.3f}mm.h5"
        pg.write(str(file_path))

        import h5py

        with h5py.File(str(file_path), "a") as h5f:
            attrs: dict[str, Any] = {
                "screen_index": int(i),
                "z_m": float(z_m),
                "n_particles": int(n),
                "n_initial": int(n0),
                "transmission_from_initial": float(n) / n0 if n0 > 0 else float("nan"),
                "mean_pz_MeV_c": float(np.nanmean(pz_MeVc)) if n else float("nan"),
                "sigma_pz_MeV_c": float(np.nanstd(pz_MeVc)) if n else float("nan"),
            }
            if info_i is not None:
                attrs["rftrack_info_transmission"] = to_json_safe(
                    info_get_first(info_i, ["transmission", "Transmission"])
                )
            if extra_attrs:
                attrs.update(extra_attrs)
            for key, value in attrs.items():
                if value is None:
                    continue
                try:
                    h5f.attrs[str(key)] = value
                except (TypeError, ValueError):
                    h5f.attrs[str(key)] = str(value)

        saved_paths.append(file_path)

    return saved_paths


def save_lost_particles_json(output_dir: Path, lost_table: np.ndarray | None) -> Path | None:
    if lost_table is None:
        return None
    arr = np.asarray(lost_table)
    if arr.ndim != 2:
        return None

    payload = {
        "columns": LOST_COLUMNS,
        "n_particles": int(arr.shape[0]),
        "lost_particles": arr.tolist(),
    }
    out_path = Path(output_dir) / "lost_particle_diagnostics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2, sort_keys=True)
    return out_path


def screen_particle_group(M: np.ndarray, weight_C: float, *, species: str = "electron") -> Any:
    """ParticleGroup of a screen array as RF-Track reports it: fixed-time back-projected x, y and
    z relative to the plane (see `plane_crossing_xy`); columns x, Px, y, Py, z, Pz[, id, t]."""
    from pmd_beamphysics import ParticleGroup

    M = np.asarray(M, dtype=float)
    n = int(M.shape[0])
    if M.shape[1] > T_COL:
        pid = M[:, ID_COL].astype(np.int64)
        t_mm_c = M[:, T_COL]
    else:
        pid = np.arange(n, dtype=np.int64)
        t_mm_c = np.zeros(n, dtype=float)
    return ParticleGroup(data={
        "x": M[:, 0] * 1e-3,
        "y": M[:, 2] * 1e-3,
        "z": M[:, 4] * 1e-3,
        "px": M[:, 1] * 1e6,
        "py": M[:, 3] * 1e6,
        "pz": M[:, 5] * 1e6,
        "t": t_mm_c * 1e-3 / c,
        "status": np.ones(n, dtype=int),
        "weight": np.full(n, float(weight_C), dtype=float),
        "id": pid,
        "species": str(species),
    })


def ensure_exit_screen(z_screens_m: Sequence[float] | None, z_exit_m: float, *, tol_m: float = 1e-6) -> list[float]:
    """Screen positions [m] with one at the exit plane z_exit_m, appended if none lies within tol_m."""
    z = [float(v) for v in (z_screens_m or [])]
    if not any(abs(v - float(z_exit_m)) <= tol_m for v in z):
        z.append(float(z_exit_m))
    return z


def save_exit_plane_openpmd(
    output_path: Path,
    M: np.ndarray,
    weight_C: float,
    keep: np.ndarray,
    *,
    s_out_m: float,
    selection: str,
    species: str = "electron",
    extra_attrs: dict[str, Any] | None = None,
) -> Path:
    """Write the beam crossing the plane s = s_out_m (fixed-s frame: z = s_out_m, t = arrival time).

    `M` is the full screen array at s_out_m (`SCREEN_PHASE_FMT`, or the legacy layout without
    %x %y, corrected exactly); `keep` selects the alive, not-trailing particles, and non-finite or
    pz <= 0 rows are always dropped. `weight_C` is the charge per macroparticle.
    """
    from pmd_beamphysics import ParticleGroup

    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[1] <= T_COL:
        raise ValueError("Exit-plane export needs a screen array with at least x, Px, y, Py, z, Pz, id, t.")
    x_mm, y_mm, source = plane_crossing_xy(M)
    keep = np.asarray(keep, dtype=bool) & (M[:, 5] > 0.0)
    keep &= np.all(np.isfinite(np.column_stack([x_mm, y_mm, M[:, [1, 3, 5, T_COL]]])), axis=1)
    if not np.any(keep):
        raise ValueError(f"No particles cross s = {s_out_m} m with the requested selection.")
    n = int(keep.sum())
    pg = ParticleGroup(data={
        "x": x_mm[keep] * 1e-3,
        "y": y_mm[keep] * 1e-3,
        "z": np.full(n, float(s_out_m)),
        "px": M[keep, 1] * 1e6,
        "py": M[keep, 3] * 1e6,
        "pz": M[keep, 5] * 1e6,
        "t": M[keep, T_COL] * 1e-3 / c,
        "status": np.ones(n, dtype=int),
        "weight": np.full(n, float(weight_C), dtype=float),
        "id": M[keep, ID_COL].astype(np.int64),
        "species": str(species),
    })
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pg.write(str(out_path))
    attrs = {**(extra_attrs or {}), "s_out_m": float(s_out_m), "Q_total_C": float(pg.charge),
             "selection": str(selection), "frame": "fixed_s", "transverse_source": source}
    import h5py

    with h5py.File(str(out_path), "a") as h5:
        for key, value in attrs.items():
            try:
                h5.attrs[str(key)] = value
            except (TypeError, ValueError):
                h5.attrs[str(key)] = str(value)
    return out_path


def save_beam_openpmd(
    output_path: Path,
    bunch: Any,
    *,
    which: str = "good",
    forward_only: bool = True,
    aperture_radius_mm: float | None = None,
    species: str = "electron",
    reference_time_s: float = 0.0,
    phase_fmt: str = OPENPMD_PHASE_FMT,
    extra_attrs: dict[str, Any] | None = None,
) -> Path:
    """Save an RF-Track ``Bunch6dT`` beam as an openPMD-beamphysics HDF5 file.

    The exit beam of a fixed-time (``Bunch6dT``) tracking run is a snapshot at a
    single laboratory time, so all particles share the same time coordinate while
    their longitudinal position ``z`` varies. This is written as an
    openPMD-beamphysics ``ParticleGroup`` (t = constant snapshot).

    Unit conversions (RF-Track -> openPMD-beamphysics):
    - X, Y, Z [mm]        -> x, y, z [m]        (x1e-3)
    - Px, Py, Pz [MeV/c]  -> px, py, pz [eV/c]  (x1e6)
    - N [# real particles per macro] -> weight [C] (x elementary charge)

    Parameters
    ----------
    output_path:
        Destination ``.h5`` file. Parent directories are created if needed.
    bunch:
        RF-Track ``Bunch6dT`` object (e.g. ``result.Bout``).
    which:
        Particle selection passed to ``get_phase_space`` (``"good"`` keeps only
        surviving particles, ``"all"`` also includes lost ones).
    forward_only:
        If True (default), keep only forward-going particles, i.e. those with
        finite coordinates, ``pz > 0`` and ``z > 0``. This drops backward-going
        or stationary electrons (negative or null longitudinal momentum), matching
        the "transmitted" masking used elsewhere in the analysis.
    aperture_radius_mm:
        If given, additionally drop particles with ``sqrt(x^2+y^2) > aperture_radius_mm``
        (applied after the ``forward_only`` filter). Use this when saving a beam at a
        z where a physical aperture restricts the transverse acceptance but was not
        enforced as a live RF-Track boundary (RF-Track apertures act on the whole
        ``Volume``, not a sub-span -- see ``rf_gun/aperture.py``) -- without this, the
        saved "exit beam" would include particles that would in reality have been
        clipped by the collimator wall.
    species:
        openPMD species name (default ``"electron"``).
    reference_time_s:
        Laboratory time [s] assigned to every particle in the snapshot.
    phase_fmt:
        RF-Track identifier string; must yield the 11 columns of
        ``OPENPMD_PHASE_FMT`` in order.
    extra_attrs:
        Optional metadata written to the HDF5 root group attributes.

    Returns
    -------
    Path
        The path of the written HDF5 file.
    """
    try:
        from pmd_beamphysics import ParticleGroup
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "openPMD-beamphysics is required to save beams in openPMD format. "
            "Install it with 'pip install openpmd-beamphysics'."
        ) from exc

    if bunch is None or not hasattr(bunch, "get_phase_space"):
        raise ValueError("bunch must be an RF-Track Bunch6dT with get_phase_space().")

    M = np.asarray(bunch.get_phase_space(phase_fmt, which), dtype=float)
    if M.ndim != 2 or M.shape[0] == 0:
        raise ValueError(
            f"No particles to save (selection={which!r}); the beam is empty."
        )
    if M.shape[1] < 11:
        raise ValueError(
            f"Expected 11 columns from phase_fmt={phase_fmt!r}, got {M.shape[1]}."
        )

    n_selected = int(M.shape[0])
    if forward_only:
        finite = np.all(np.isfinite(M[:, :6]), axis=1)
        forward = finite & (M[:, 5] > 0.0) & (M[:, 4] > 0.0)
        M = M[forward]
        if M.shape[0] == 0:
            raise ValueError(
                "No forward-going particles to save "
                f"(pz>0 & z>0) out of {n_selected} selected."
            )

    if aperture_radius_mm is not None:
        n_before_aperture = int(M.shape[0])
        r_mm = np.sqrt(M[:, 0] ** 2 + M[:, 2] ** 2)
        within = np.isfinite(r_mm) & (r_mm <= float(aperture_radius_mm))
        M = M[within]
        if M.shape[0] == 0:
            raise ValueError(
                f"No particles within aperture_radius_mm={aperture_radius_mm!r} to save "
                f"out of {n_before_aperture} forward-going particles."
            )

    x_mm, px_MeVc, y_mm, py_MeVc, z_mm, pz_MeVc = M[:, :6].T
    n_real, pid = M[:, 8], M[:, 10]

    n = int(M.shape[0])
    weight = np.abs(n_real) * float(q_e)  # macro-charge magnitude [C]
    # Guard against zero/degenerate weights (openPMD requires positive weights).
    if not np.all(np.isfinite(weight)) or np.all(weight <= 0.0):
        weight = np.full(n, float(q_e), dtype=float)

    data = {
        "x": x_mm * 1e-3,
        "y": y_mm * 1e-3,
        "z": z_mm * 1e-3,
        "px": px_MeVc * 1e6,
        "py": py_MeVc * 1e6,
        "pz": pz_MeVc * 1e6,
        "t": np.full(n, float(reference_time_s), dtype=float),
        "status": np.ones(n, dtype=int),
        "weight": weight,
        "id": pid.astype(np.int64),
        "species": str(species),
    }

    pg = ParticleGroup(data=data)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pg.write(str(out_path))

    if extra_attrs:
        import h5py

        with h5py.File(str(out_path), "a") as h5:
            for key, value in extra_attrs.items():
                try:
                    h5.attrs[str(key)] = value
                except (TypeError, ValueError):
                    h5.attrs[str(key)] = str(value)

    return out_path


#: Bump when validation.json's schema changes.
VALIDATION_SCHEMA_VERSION = 1


def build_validation_report(checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Assemble a `validation.json` payload from a caller-supplied `{check_name: {"passed": bool,
    ...}}` mapping (Section 9: "Create a machine-readable validation.json per run"). `status` is
    `"ok"` only if every check's `"passed"` is `True`; otherwise `"failed"`, naming the failing
    checks. This does not itself decide what to check -- callers assemble `checks` from whatever
    gates are meaningful at their pipeline stage (phase calibration validity, particle-ID
    uniqueness, charge/energy closure, convergence, ...); this function only aggregates and
    standardizes the report shape so every run's validation.json looks the same.
    """
    failed = sorted(name for name, result in checks.items() if not bool(result.get("passed", False)))
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "timestamp_local": datetime.now().isoformat(),
        "status": "ok" if not failed else "failed",
        "failed_checks": failed,
        "checks": checks,
    }


def save_validation_report(run_dir: Path, checks: dict[str, dict[str, Any]], filename: str = "validation.json") -> Path:
    """Build (`build_validation_report`) and atomically write `validation.json`. Returns the
    written path regardless of overall status -- callers decide whether a `"failed"` status should
    also abort the run/skip a completion marker; this function only records the outcome."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    report = build_validation_report(checks)
    return atomic_write_json(run_dir / filename, report)


def save_run_config(
    run_dir: Path,
    *,
    run_name: str,
    source: str,
    hardcoded_parameters: dict[str, Any],
    derived_parameters: dict[str, Any],
    filename: str = "run_config.json",
) -> Path:
    """Write everything that describes *what this run was set up to do* -- every input parameter
    (cavity/field-map, solver/finesse, cathode/emission, beam-loading, aperture, deflection,
    screen/particle-count settings) plus the small set of values derived from them before
    tracking even starts (grid sizes, phase-scan crest/Veff/R-over-Q, effective length) -- to one
    small JSON file, with no per-particle or per-time-sample data anywhere in it.

    Shared by the notebook (`UH_gun_tracking_demo.ipynb`) and any batch/CLI script (e.g.
    `run_thermionic_tm010.py`), so a run started either way produces the same `run_config.json`
    shape. Pairs with `save_run_results` (the *outcome* of the run) and, when the deflection
    magnet's back-bombardment analysis is enabled, `save_back_bombardment_energy_map` (the one
    piece of per-bin, not per-particle, array data this project produces) -- three files instead
    of the previous sprawl of `run_metadata.json`/`run_summary.json`/`beam_summary.json`/
    `particle_classes_summary.json`/`progress_stats.json`/`B0_timing.json`, which duplicated the
    same handful of derived quantities across four overlapping files and (via an incomplete
    per-particle-array exclusion list) sometimes ballooned to tens of MB each.

    Parameters
    ----------
    run_dir:
        Destination directory (created if needed); the file is written to `run_dir/filename`.
    run_name:
        Human-readable run identifier (e.g. the timestamped run-directory name).
    source:
        Where this run was launched from, e.g. `"notebook:UH_gun_tracking_demo.ipynb"` or
        `"script:run_thermionic_tm010.py"` -- so a later reader can tell the two apart.
    hardcoded_parameters, derived_parameters:
        Caller-assembled nested dicts (see either call site for the expected grouping); passed
        through as-is other than JSON sanitization.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "timestamp_local": datetime.now().isoformat(),
        "source": str(source),
        "hardcoded_parameters": hardcoded_parameters,
        "derived_parameters": derived_parameters,
    }
    return atomic_write_json(run_dir / filename, payload)


def save_run_results(
    run_dir: Path,
    *,
    run_name: str,
    source: str,
    results: dict[str, Any],
    output_files: dict[str, Any],
    filename: str = "run_results.json",
) -> Path:
    """Write everything this run *found* -- R/Q and Veff actually used, peak current/current
    density, the beam-property curves vs z (transmission, Twiss, beam size -- one row per screen,
    not per particle), particle classification counts, aperture and back-bombardment summaries,
    the openPMD exit-beam summary, and the paths to every other output file this run wrote -- to
    one JSON file. See `save_run_config` for the companion "what this run was set up to do" file
    and the reasoning for this split.

    Parameters
    ----------
    run_dir, run_name, source:
        Same meaning as in `save_run_config`.
    results, output_files:
        Caller-assembled nested dicts (see either call site for the expected grouping); passed
        through as-is other than JSON sanitization. Screen/curve entries in `results` must already
        be per-screen (or per-time-sample) summaries, never a raw per-particle phase-space array --
        see `rf_gun.simulation.thermo_info_summary` for the corresponding thermo_info filter.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RUN_RESULTS_SCHEMA_VERSION,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "timestamp_local": datetime.now().isoformat(),
        "source": str(source),
        "results": results,
        "output_files": output_files,
    }
    return atomic_write_json(run_dir / filename, payload)
