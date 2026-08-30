"""I/O helpers for simulation outputs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .constants import c, q_e
from .diagnostics import build_screen_summary_from_phase_space, info_get_first

SCREEN_COLUMNS = ["x_mm", "px_MeV_c", "y_mm", "py_MeV_c", "z_mm", "pz_MeV_c"]
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
RUN_RESULTS_SCHEMA_VERSION = 1


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


def save_screen_distributions_json(
    output_dir: Path,
    z_snaps: Sequence[float],
    M_snaps: Sequence[np.ndarray] | None,
    I_snaps: Sequence[Any] | None,
    *,
    mode: str = "summary",
    n_initial: int | None = None,
    robust_summaries: Sequence[dict[str, Any]] | None = None,
) -> int:
    """Save per-screen JSON in summary or full mode."""
    if not z_snaps:
        return 0
    mode_norm = str(mode).strip().lower()
    if mode_norm not in ("summary", "full"):
        raise ValueError(f"Unknown screen JSON mode: {mode}")

    out_dir = Path(output_dir) / "screen_distributions_json"
    out_dir.mkdir(parents=True, exist_ok=True)

    precomputed = [dict(x) for x in robust_summaries] if robust_summaries is not None else []

    n0 = int(n_initial) if n_initial is not None else 0
    if n0 <= 0 and M_snaps is not None and len(M_snaps) > 0:
        first = np.asarray(M_snaps[0])
        if first.ndim == 2:
            n0 = int(first.shape[0])

    saved = 0
    n_prev = n0
    for i, z_m in enumerate(z_snaps):
        M_i = None
        if M_snaps is not None and i < len(M_snaps):
            M_i = np.asarray(M_snaps[i], dtype=float)

        if i < len(precomputed):
            summary = dict(precomputed[i])
        else:
            summary = build_screen_summary_from_phase_space(
                M_i,
                screen_index=i,
                z_m=float(z_m),
                n_initial=n0,
                n_previous=n_prev,
            )
        n_prev = int(summary["N"])

        info_i = I_snaps[i] if (I_snaps is not None and i < len(I_snaps)) else None
        raw_info = {
            "transmission": info_get_first(info_i, ["transmission", "Transmission"]),
            "mean_pz": info_get_first(info_i, ["mean_Pz", "mean_P", "mean_pz"]),
            "sigma_pz": info_get_first(info_i, ["sigma_Pz", "sigma_P", "sigma_pz"]),
        }

        payload: dict[str, Any] = {
            "screen_index": int(i),
            "z_m": float(z_m),
            "mode": mode_norm,
            "summary": summary,
            "rftrack_raw_info": raw_info,
        }

        if mode_norm == "full" and M_snaps is not None and i < len(M_snaps):
            arr = np.asarray(M_snaps[i])
            payload["columns"] = SCREEN_COLUMNS
            payload["n_particles"] = int(arr.shape[0]) if arr.ndim == 2 else 0
            payload["phase_space"] = arr.tolist() if arr.ndim == 2 and arr.size else []

        file_path = out_dir / f"screen_{i:04d}_z_{float(z_m):.6f}m.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(to_json_safe(payload), f, indent=2, sort_keys=True)
        saved += 1

    return saved


#: Column layout of the extended phase-space format used for screen snapshots throughout this
#: project (`rf_gun.simulation.EXTENDED_PHASE_FMT`: "%X %Px %Y %Py %Z %Pz %id %t %E %K").
SCREEN_EXTENDED_COLUMNS = [
    "x_mm", "px_MeV_c", "y_mm", "py_MeV_c", "z_mm", "pz_MeV_c", "id", "t_mm_c", "E_MeV", "K_MeV",
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
        from pmd_beamphysics import ParticleGroup
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
        x_mm, px_MeVc, y_mm, py_MeVc, z_mm, pz_MeVc = (
            M[:, 0], M[:, 1], M[:, 2], M[:, 3], M[:, 4], M[:, 5],
        )
        if M.shape[1] >= 8:
            pid = M[:, 6].astype(np.int64)
            t_mm_c = M[:, 7]
        else:
            pid = np.arange(n, dtype=np.int64)
            t_mm_c = np.zeros(n, dtype=float)

        weight = (
            np.full(n, n_real_per_macro * q_e, dtype=float)
            if n_real_per_macro > 0
            else np.full(n, float(q_e), dtype=float)
        )

        data = {
            "x": x_mm * 1e-3,
            "y": y_mm * 1e-3,
            "z": z_mm * 1e-3,
            "px": px_MeVc * 1e6,
            "py": py_MeVc * 1e6,
            "pz": pz_MeVc * 1e6,
            "t": t_mm_c * 1e-3 / c,
            "status": np.ones(n, dtype=int),
            "weight": weight,
            "id": pid,
            "species": str(species),
        }
        pg = ParticleGroup(data=data)

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

    x_mm, px_MeVc, y_mm, py_MeVc, z_mm, pz_MeVc, _mass, _q, n_real, _t0, pid = (
        M[:, 0], M[:, 1], M[:, 2], M[:, 3], M[:, 4], M[:, 5],
        M[:, 6], M[:, 7], M[:, 8], M[:, 9], M[:, 10],
    )

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
        "run_dir": str(run_dir.resolve()),
        "timestamp_local": datetime.now().isoformat(),
        "source": str(source),
        "hardcoded_parameters": hardcoded_parameters,
        "derived_parameters": derived_parameters,
    }
    out_path = run_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2, sort_keys=True)
    return out_path


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
        "run_dir": str(run_dir.resolve()),
        "timestamp_local": datetime.now().isoformat(),
        "source": str(source),
        "results": results,
        "output_files": output_files,
    }
    out_path = run_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2, sort_keys=True)
    return out_path


def save_back_bombardment_energy_map(
    run_dir: Path,
    energy_map: dict[str, Any] | None,
    *,
    filename: str = "back_bombardment_energy_map.npz",
) -> Path | None:
    """Write the 2D kinetic-energy-density map deposited by back-bombarding electrons at the
    cathode plane (z=0) -- the dict returned by
    `rf_gun.plotting.back_bombardment.plot_back_bombardment_energy_density` (`xedges`, `yedges`,
    `density_J_per_mm2`, `total_J`) -- to its own small binary file, independent of
    `run_config.json`/`run_results.json` (this is per-bin array data, not a per-run scalar, so it
    doesn't belong in either) and of `figures/` (this is data, not a rendering of it -- read it
    back directly with `np.load(...)`, no re-plotting needed).

    Returns `None` (writes nothing) when `energy_map` is `None`, e.g. no particle in the run had a
    physically plausible back-bombardment reconstruction (`BackBombardmentData.n_valid == 0`).
    """
    if energy_map is None:
        return None
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / filename
    np.savez_compressed(
        out_path,
        xedges=np.asarray(energy_map["xedges"], dtype=float),
        yedges=np.asarray(energy_map["yedges"], dtype=float),
        density_J_per_mm2=np.asarray(energy_map["density_J_per_mm2"], dtype=float),
        total_J=np.asarray(float(energy_map["total_J"])),
    )
    return out_path


def save_back_bombardment_events_hdf5(
    run_dir: Path,
    data: "BackBombardmentData",
    *,
    filename: str = "back_bombardment_events.h5",
    extra_attrs: dict[str, Any] | None = None,
) -> Path | None:
    """Write one row per cathode-heating-relevant back-bombardment impact --
    {x, y, t, E, K, px, py, pz, weight (real electrons), surface_id} -- for COMSOL/TIO or any
    downstream tool needing the event list itself, not just a binned 2D map
    (`save_back_bombardment_energy_map`).

    Only `data.heating_relevant` rows are written (face + chamfer, see `classify_impact_surface`);
    holder/cavity-wall impacts are excluded, matching `rf_gun.plotting.back_bombardment`'s heating
    figures. Returns `None` if there are no heating-relevant rows.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "h5py is required to save back-bombardment events in HDF5 format. "
            "Install it with 'pip install h5py'."
        ) from exc

    from .back_bombardment import kinetic_energy_joules

    v = np.asarray(data.heating_relevant, dtype=bool)
    n = int(np.sum(v))
    if n == 0:
        return None

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / filename

    weight_electrons = np.full(n, float(data.weight_per_macroparticle))

    with h5py.File(str(out_path), "w") as h5f:
        h5f.create_dataset("x_mm", data=data.x_hit_mm[v])
        h5f.create_dataset("y_mm", data=data.y_hit_mm[v])
        h5f.create_dataset("t_s", data=data.t_hit_s[v])
        h5f.create_dataset("E_total_MeV", data=data.E_total_MeV[v])
        h5f.create_dataset("K_MeV", data=data.K_MeV[v])
        h5f.create_dataset("px_MeV_c", data=data.px_MeVc[v])
        h5f.create_dataset("py_MeV_c", data=data.py_MeVc[v])
        h5f.create_dataset("pz_MeV_c", data=data.pz_MeVc[v])
        h5f.create_dataset("weight_electrons", data=weight_electrons)
        h5f.create_dataset("K_joules_weighted", data=kinetic_energy_joules(data)[v])
        h5f.create_dataset(
            "surface_id", data=np.asarray(data.surface_id[v], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        h5f.attrs["n_events"] = n
        h5f.attrs["n_cathode_face"] = int(data.n_cathode_face)
        h5f.attrs["n_cathode_chamfer"] = int(data.n_cathode_chamfer)
        h5f.attrs["n_excluded_geometry"] = int(data.n_excluded_geometry)
        h5f.attrs["weight_per_macroparticle_electrons"] = float(data.weight_per_macroparticle)
        h5f.attrs["columns"] = "x_mm, y_mm, t_s, E_total_MeV, K_MeV, px_MeV_c, py_MeV_c, pz_MeV_c, weight_electrons, surface_id"
        if extra_attrs:
            for key, value in extra_attrs.items():
                if value is None:
                    continue
                try:
                    h5f.attrs[str(key)] = value
                except (TypeError, ValueError):
                    h5f.attrs[str(key)] = str(value)

    return out_path
