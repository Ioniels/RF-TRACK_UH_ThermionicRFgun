"""Batch plotting entry points for run outputs."""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..back_bombardment import BackBombardmentData
from ..constants import c, ME_MEV
from ..particle_tags import ParticleTags, build_particle_tags, ID_COL, lost_ids_from_lost_table
from ..beam_properties import compute_beam_properties, transmission_curves
from .style import COLOR_LOST, COLOR_PRIMARY, COLOR_SECONDARY
from .back_bombardment import (
    plot_back_bombardment_energy_density,
    plot_back_bombardment_phase_space,
    plot_back_bombardment_power_density_vs_time,
    plot_back_bombardment_screen_reach,
)
from .emission import plot_emission_history, plot_j_vs_n
from .evolution import plot_beam_moments_evolution, plot_beam_twiss_evolution
from .phase_space import (
    EXTENDED_PHASE_FMT_DEFAULT,
    plot_phase_space,
    plot_spectra,
    render_screen_phase_space_figure,
)


PHASE_SPACE_COLUMNS = ["x_mm", "px_MeV_c", "y_mm", "py_MeV_c", "z_mm", "pz_MeV_c"]
#: Column names for the 4 optional trailing columns of `EXTENDED_PHASE_FMT_DEFAULT`, appended to
#: `PHASE_SPACE_COLUMNS` (by count, not by inspecting the caller's actual phase_fmt string) when a
#: caller doesn't supply explicit `phase_space_columns`.
_EXTENDED_EXTRA_COLUMNS = ["id", "t_mm_c", "E_MeV", "K_MeV"]


def _save_figure(fig, output_dir: Path, stem: str, *, formats: Sequence[str] = ("png", "eps")) -> list[str]:
    saved: list[str] = []
    fmts = [str(fmt).strip().lower() for fmt in formats if str(fmt).strip()]
    if not fmts:
        fmts = ["png"]
    for fmt in fmts:
        out_path = output_dir / f"{stem}.{fmt}"
        if fmt == "png":
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(out_path, format=fmt, bbox_inches="tight")
        saved.append(out_path.name)
    return saved


def _capture_current_figure(save_name: str, output_dir: Path, *, formats: Sequence[str] = ("png", "eps")) -> list[str]:
    """Save whatever figure the preceding plot call left open -- or nothing, cleanly, if it left
    none. Several plotting functions in this project (`plot_j_vs_n` when the emission law isn't
    "unified"; every `plot_back_bombardment_*` when there's no particle with a physically
    plausible reconstruction) intentionally print a note and return without creating a figure.
    `plt.gcf()` does not report that "no figure" state -- called with none open, it silently
    creates and returns a brand new blank one, which this function would then happily save,
    producing a confusing all-white PNG (confirmed: this is exactly how `emission_j_vs_n.png`
    ended up blank in every RD_schottky-emission-law run). Checking `plt.get_fignums()` first
    tells the two cases apart.
    """
    import matplotlib.pyplot as plt

    if not plt.get_fignums():
        return []
    fig = plt.gcf()
    saved = _save_figure(fig, output_dir, save_name, formats=formats)
    plt.close(fig)
    return saved


def _save_figure_data(stem: Path, data: Any, *, data_format: str) -> Path:
    """Write the numeric data behind a figure to `{stem}.npz` or `{stem}.json`."""
    fmt = str(data_format).strip().lower()
    if fmt == "npz":
        arrays = {k: np.asarray(v) for k, v in dict(data).items() if v is not None}
        out_path = stem.with_suffix(".npz")
        np.savez_compressed(out_path, **arrays)
        return out_path
    if fmt == "json":
        from ..io import to_json_safe

        out_path = stem.with_suffix(".json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(to_json_safe(data), f, indent=2, sort_keys=True)
        return out_path
    raise ValueError(f"Unknown data_format: {data_format!r} (expected 'npz' or 'json')")


class FigureCapture:
    """Holder yielded by `capture_figures`; set `.data` inside the `with` block when the value to
    save is only available from the plotting call's return (e.g. `plot_back_bombardment_energy_density`
    returns its histogram dict) rather than known upfront."""

    def __init__(self, data: Any = None):
        self.data = data


@contextlib.contextmanager
def capture_figures(
    name: str,
    output_dir: Path | str | None,
    *,
    formats: Sequence[str] = ("png",),
    dpi: int = 200,
    data: Any = None,
    data_format: str = "npz",
):
    """Save every figure shown (via ``plt.show()``) inside this ``with`` block, before Jupyter's
    inline backend destroys it -- and the numeric data behind that figure too.

    Notebook cells that call a plotting helper ending in ``plt.show()`` -- e.g. `field_maps`,
    `plot_spectra`, the back-bombardment figures -- never return the `Figure` object, so it can't
    be saved from the caller after the fact: `%matplotlib inline`'s own `show()` (unlike the
    default backend's) destroys every open figure right after displaying it. This wraps `plt.show`
    for the duration of the block instead, so each figure is written to disk immediately before
    that happens -- no change needed inside the plotting helpers themselves, and no dependency on
    IPython's `InlineBackend.close_figures` config.

    The data behind the figure -- typically a ``dict[str, array-like]`` of the exact x/y (and any
    color/weight) arrays passed into the plot call -- is written alongside it as `{name}.npz`
    (default; one array per key, `np.load(...)['key']` to read back) or, with `data_format="json"`
    (via `rf_gun.io.to_json_safe`, for small/mixed-type payloads), as `{name}.json`. This is what
    makes a saved figure independently reproducible/replottable later without re-running the
    simulation. Two ways to supply it:

    - Known before the plot call: pass `data=...` directly.
    - Only available from the plot call's return value: bind the context with ``as``, and set
      `.data` on it inside the block::

          with rg.capture_figures("name", FIGURES_DIR) as cap:
              cap.data = rg.plot_some_histogram(...)  # returns a dict of the binned data

    Either way, the data is written once, at the end of the block (after the figure itself).

    A no-op (yields a `FigureCapture` that discards `.data`, saves nothing) when `output_dir` is
    `None` -- the intended behavior for a `SAVE_DATA=False` run in the notebook: wrap every plot
    call in ``with rg.capture_figures("name", FIGURES_DIR):`` and pass `FIGURES_DIR=None` when
    saving is disabled, rather than threading an extra `if SAVE_DATA:` through every plotting cell.
    """
    import matplotlib.pyplot as plt

    holder = FigureCapture(data)

    if output_dir is None:
        yield holder
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_show = plt.show
    counter = {"i": 0}

    def _patched_show(*args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            suffix = "" if counter["i"] == 0 else f"_{counter['i']}"
            for fmt in formats:
                fig.savefig(out_dir / f"{name}{suffix}.{fmt}", dpi=dpi, bbox_inches="tight")
            counter["i"] += 1
        return original_show(*args, **kwargs)

    plt.show = _patched_show
    try:
        yield holder
    finally:
        plt.show = original_show
        if holder.data is not None:
            _save_figure_data(out_dir / name, holder.data, data_format=data_format)


def plot_class_conditioned_histograms(
    initial_M: np.ndarray,
    final_M: np.ndarray,
    lost_table: np.ndarray | None = None,
    t0_mm_c: np.ndarray | None = None,
):
    """Initial distributions split by final class: transmitted/backward/lost.

    `lost` is an id-based match against `lost_table` (RF-Track's own lost-particle table, see
    `rf_gun.particle_tags.lost_ids_from_lost_table`) when `initial_M` carries an `%id` column
    (index `ID_COL`); otherwise `lost` is shown only as a count annotation, since there's no way
    to pick its rows out of `initial_M` by identity.
    """
    import matplotlib.pyplot as plt

    Mi = np.asarray(initial_M)
    Mf = np.asarray(final_M)
    if Mi.ndim != 2 or Mi.shape[0] == 0 or Mi.shape[1] < 6:
        return None

    n = min(Mi.shape[0], Mf.shape[0]) if Mf.ndim == 2 else 0
    if n <= 0:
        return None

    zf = np.asarray(Mf[:n, 4], dtype=float)
    pzf = np.asarray(Mf[:n, 5], dtype=float)
    transmitted = np.isfinite(zf) & np.isfinite(pzf) & (zf > 0.0) & (pzf > 0.0)
    backward = np.isfinite(zf) & np.isfinite(pzf) & ((zf <= 0.0) | (pzf < 0.0))

    lost = np.zeros(n, dtype=bool)
    n_lost_total = int(np.asarray(lost_table).shape[0]) if lost_table is not None and np.asarray(lost_table).ndim == 2 else 0
    if n_lost_total > 0 and Mi.shape[1] > ID_COL:
        lost_ids = lost_ids_from_lost_table(lost_table)
        if lost_ids:
            ids0 = Mi[:n, ID_COL].astype(np.int64)
            lost = np.isin(ids0, list(lost_ids))

    pz0 = np.asarray(Mi[:n, 5], dtype=float)
    r0 = np.sqrt(np.asarray(Mi[:n, 0], dtype=float) ** 2 + np.asarray(Mi[:n, 2], dtype=float) ** 2)
    t0_ns = None
    if t0_mm_c is not None:
        t0_arr = np.asarray(t0_mm_c, dtype=float).reshape(-1)
        if t0_arr.size >= n:
            from ..constants import c

            t0_ns = (t0_arr[:n] * 1e-3 / c) * 1e9

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    bins_pz = np.histogram_bin_edges(1e3 * pz0[np.isfinite(pz0)], bins=60)
    axes[0].hist(1e3 * pz0, bins=bins_pz, alpha=0.4, label="all", color="tab:blue")
    axes[0].hist(1e3 * pz0[transmitted], bins=bins_pz, alpha=0.6, label="transmitted", color=COLOR_PRIMARY)
    axes[0].hist(1e3 * pz0[backward], bins=bins_pz, alpha=0.6, label="backward/returned", color=COLOR_SECONDARY)
    if np.any(lost):
        axes[0].hist(1e3 * pz0[lost], bins=bins_pz, alpha=0.6, label="lost", color=COLOR_LOST)
    elif n_lost_total > 0:
        axes[0].text(0.98, 0.95, f"lost={n_lost_total}", transform=axes[0].transAxes, ha="right", va="top")
    axes[0].set_xlabel("initial pz [keV/c]")
    axes[0].set_ylabel("counts")
    axes[0].grid(alpha=0.3)

    if t0_ns is not None:
        bins_t0 = np.histogram_bin_edges(t0_ns[np.isfinite(t0_ns)], bins=60)
        axes[1].hist(t0_ns, bins=bins_t0, alpha=0.4, label="all", color="tab:blue")
        axes[1].hist(t0_ns[transmitted], bins=bins_t0, alpha=0.6, label="transmitted", color=COLOR_PRIMARY)
        axes[1].hist(t0_ns[backward], bins=bins_t0, alpha=0.6, label="backward/returned", color=COLOR_SECONDARY)
        if np.any(lost):
            axes[1].hist(t0_ns[lost], bins=bins_t0, alpha=0.6, label="lost", color=COLOR_LOST)
        axes[1].set_xlabel("initial t0 [ns]")
        axes[1].set_ylabel("counts")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "t0 unavailable", ha="center", va="center")

    bins_r = np.histogram_bin_edges(r0[np.isfinite(r0)], bins=60)
    axes[2].hist(r0, bins=bins_r, alpha=0.4, label="all", color="tab:blue")
    axes[2].hist(r0[transmitted], bins=bins_r, alpha=0.6, label="transmitted", color=COLOR_PRIMARY)
    axes[2].hist(r0[backward], bins=bins_r, alpha=0.6, label="backward/returned", color=COLOR_SECONDARY)
    if np.any(lost):
        axes[2].hist(r0[lost], bins=bins_r, alpha=0.6, label="lost", color=COLOR_LOST)
    axes[2].set_xlabel("initial radius [mm]")
    axes[2].set_ylabel("counts")
    axes[2].grid(alpha=0.3)

    axes[0].legend(frameon=False)
    fig.suptitle("Initial distributions by final class")
    fig.tight_layout()
    return fig


def save_beam_phase_space_json(
    output_path: Path,
    M: np.ndarray,
    *,
    phase_space_columns: Sequence[str] | None = None,
    label: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save one phase-space matrix to JSON with explicit schema fields."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2:
        arr = np.zeros((0, 6), dtype=float)

    if phase_space_columns is not None:
        cols = list(phase_space_columns)
    else:
        n_extra = max(0, arr.shape[1] - len(PHASE_SPACE_COLUMNS))
        cols = list(PHASE_SPACE_COLUMNS) + _EXTENDED_EXTRA_COLUMNS[:n_extra]

    payload = {
        "schema_version": 1,
        "label": str(label) if label is not None else None,
        "phase_space_columns": cols,
        "particle_count": int(arr.shape[0]),
        "coordinate_system": "Bunch6dT phase space: X Px Y Py Z Pz"
        + (" (+ id, t, E, K when present)" if arr.shape[1] > len(PHASE_SPACE_COLUMNS) else ""),
        "timing_note": "Creation time t0 is stored separately from Z in Bunch6dT and is not one of the 6 core phase-space columns.",
        "phase_space": arr.tolist(),
    }
    if extra_metadata:
        payload.update(dict(extra_metadata))

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def save_screen_phase_space_batch(
    output_dir: Path,
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    *,
    B0=None,
    tags: ParticleTags | None = None,
    phase_fmt: str = EXTENDED_PHASE_FMT_DEFAULT,
    exclude_backward_losses: bool = True,
    exclude_lost: bool = True,
    n_macroparticles: int | None = None,
    style=None,
    show_colorbar: bool = False,
    save_json: bool = True,
    figure_formats: Sequence[str] = ("png",),
    timing_log: bool = False,
    thermo_info: dict | None = None,
) -> dict[str, Any]:
    """Save non-interactive phase-space figures for B0 and every screen.

    `Bout` is intentionally not included as a frame here -- see `rf_gun.plotting.phase_space`'s
    module docstring for why (it is a fixed-time snapshot with a spread of z among its particles,
    so it has no single z to plot against); the last screen serves as the final "exit-like" frame.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    fig_dir = output_dir / "screen_phase_space_frames"
    json_dir = output_dir / "screen_phase_space_json"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if save_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    frame_idx = 0

    def _z_tag(z_mm: float) -> str:
        return f"{z_mm:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")

    def _save_single(
        label: str,
        stem: str,
        *,
        M_local: np.ndarray,
        z_local: float | None = None,
    ) -> None:
        nonlocal frame_idx
        t0 = time.perf_counter()
        z_mm_local = (1e3 * float(z_local)) if z_local is not None else None
        t_render_start = time.perf_counter()
        fig = render_screen_phase_space_figure(
            np.asarray(M_local, dtype=float),
            label=label,
            z_mm=z_mm_local,
            tags=tags,
            exclude_backward_losses=exclude_backward_losses,
            exclude_lost=exclude_lost,
            style=style,
            show_colorbar=show_colorbar,
            n_macroparticles=n_macroparticles,
            thermo_info=thermo_info,
        )
        t_render_s = float(time.perf_counter() - t_render_start)

        t_save_start = time.perf_counter()
        fig_files = _save_figure(fig, fig_dir, stem, formats=figure_formats)
        t_save_s = float(time.perf_counter() - t_save_start)
        plt.close(fig)

        json_file = None
        t_json_s = 0.0
        if save_json:
            t_json_start = time.perf_counter()
            json_path = save_beam_phase_space_json(
                json_dir / f"{stem}.json",
                np.asarray(M_local, dtype=float),
                label=label,
            )
            json_file = json_path.name
            t_json_s = float(time.perf_counter() - t_json_start)

        t_total_s = float(time.perf_counter() - t0)
        timing = {
            "render_s": t_render_s,
            "save_figure_s": t_save_s,
            "save_json_s": t_json_s,
            "total_s": t_total_s,
        }
        manifest.append(
            {
                "frame_index": int(frame_idx),
                "label": label,
                "z_mm": float(z_local) if z_local is not None else None,
                "figure_files": fig_files,
                "json_file": json_file,
                "timing": timing,
            }
        )
        if bool(timing_log):
            print(
                f"[screen-frame {frame_idx:04d}] {label}: "
                f"render={t_render_s:.3f}s save={t_save_s:.3f}s json={t_json_s:.3f}s total={t_total_s:.3f}s"
            )
        frame_idx += 1
        plt.close("all")

    if B0 is not None:
        M0 = np.array(B0.get_phase_space(phase_fmt, "all"), copy=True)
        _save_single("B0", f"frame_{frame_idx:04d}_B0", M_local=M0)

    z_mm = 1e3 * np.asarray(z_snaps, dtype=float) if z_snaps is not None else np.asarray([], dtype=float)
    n_screens = min(len(M_snaps), int(z_mm.size))
    for i in range(n_screens):
        z_i = float(z_mm[i])
        stem = f"frame_{frame_idx:04d}_screen_{i+1:03d}_z{_z_tag(z_i)}mm"
        _save_single(
            f"screen_{i+1}",
            stem,
            M_local=np.asarray(M_snaps[i], dtype=float),
            z_local=float(z_snaps[i]),
        )

    manifest_path = output_dir / "screen_phase_space_manifest.json"
    timing_summary = {
        "render_s_total": float(sum(float(rec.get("timing", {}).get("render_s", 0.0)) for rec in manifest)),
        "save_figure_s_total": float(sum(float(rec.get("timing", {}).get("save_figure_s", 0.0)) for rec in manifest)),
        "save_json_s_total": float(sum(float(rec.get("timing", {}).get("save_json_s", 0.0)) for rec in manifest)),
        "total_s_total": float(sum(float(rec.get("timing", {}).get("total_s", 0.0)) for rec in manifest)),
    }
    if manifest:
        n_frames = float(len(manifest))
        timing_summary["total_s_mean"] = float(timing_summary["total_s_total"] / n_frames)
        timing_summary["total_s_max"] = float(max(float(rec.get("timing", {}).get("total_s", 0.0)) for rec in manifest))
    else:
        timing_summary["total_s_mean"] = 0.0
        timing_summary["total_s_max"] = 0.0

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"frames": manifest, "timing_summary": timing_summary}, f, indent=2)

    if bool(timing_log):
        print(
            "[screen-frame timing-summary] "
            f"frames={len(manifest)} render_total={timing_summary['render_s_total']:.3f}s "
            f"save_total={timing_summary['save_figure_s_total']:.3f}s "
            f"json_total={timing_summary['save_json_s_total']:.3f}s "
            f"total={timing_summary['total_s_total']:.3f}s"
        )

    return {
        "frame_count": int(len(manifest)),
        "figure_dir": str(fig_dir),
        "json_dir": str(json_dir) if save_json else None,
        "manifest_file": str(manifest_path),
        "timing_summary": timing_summary,
    }


def save_run_figures(
    output_dir: Path,
    B0,
    Bout,
    transport_phase_deg: float,
    thermo_info: dict[str, Any],
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    *,
    tags: ParticleTags | None = None,
    phase_fmt: str = EXTENDED_PHASE_FMT_DEFAULT,
    exclude_backward_losses: bool = True,
    exclude_lost: bool = True,
    n_macroparticles: int | None = None,
    mass_MeV: float = ME_MEV,
    lost_table: np.ndarray | None = None,
    back_bombardment_data: BackBombardmentData | None = None,
    back_bombardment_cathode_radius_mm: float | None = None,
) -> dict[str, Any]:
    """Generate and save a standard figure bundle for one run.

    `tags` (`rf_gun.particle_tags.ParticleTags`) drives every figure in this bundle -- pass one
    built via `build_particle_tags` (from `Bout` plus RF-Track's own lost-particle table) so this
    bundle's tagging is identical to any other output (JSON summaries, the notebook) built from
    the same run. If not supplied, falls back to backward-tagging only (from `Bout`'s own
    reliable absolute z/pz), with no lost tagging. The beam-properties table and transmission
    curves are always computed on the forward-going + dynamic-aperture-surviving population,
    matching `rf_gun.beam_properties.compute_beam_properties`.

    `back_bombardment_data` (from `rf_gun.compute_back_bombardment`), when given, adds the 4
    back-bombardment figures (phase space, screen reach, cathode energy-density map, power
    density vs time -- matching the notebook's back-bombardment cell) to the bundle;
    `back_bombardment_cathode_radius_mm` is required alongside it (only the power-density figure
    needs it, to normalize deposited energy by the cathode's nominal area).

    Returns `{"saved_figures": [...], "back_bombardment_energy_map": {...} | None}` -- the energy
    map is the exact dict `plot_back_bombardment_energy_density` returned (`xedges`, `yedges`,
    `density_J_per_mm2`, `total_J`), for the caller to persist independently (see
    `rf_gun.save_back_bombardment_energy_map`) since it's per-bin array data, not a figure.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    M0 = np.array(B0.get_phase_space(phase_fmt, "all"), copy=True)
    Bout_M = np.array(Bout.get_phase_space(phase_fmt, "all"), copy=True)

    if tags is None:
        tags = build_particle_tags(Bout_M, lost_table)

    if M_snaps:
        M_exit = np.asarray(M_snaps[-1], dtype=float)
        plot_phase_space(
            B0,
            M_exit,
            transport_phase_deg,
            tags=tags,
            exclude_backward_losses=exclude_backward_losses,
            exclude_lost=exclude_lost,
            phase_fmt=phase_fmt,
            thermo_info=thermo_info,
        )
        saved += _capture_current_figure("initial_phase_space_x_px", output_dir)

    plot_spectra(
        Bout,
        transport_phase_deg,
        B0=B0,
        thermo_info=thermo_info,
        tags=tags,
        phase_fmt=phase_fmt,
    )
    saved += _capture_current_figure("screen_spectra", output_dir)

    if M_snaps and z_snaps:
        table = compute_beam_properties(M_snaps, z_snaps, tags, mass_MeV)

        # Macroparticle count only -- every macroparticle represents an equal share of the real
        # bunch (no per-particle weight column anywhere in the phase-space format), so a
        # macroparticle-count ratio already *is* the real transmission fraction. `Q_total_C /
        # q_e` (the *real*, charge-weighted electron count) is many orders of magnitude larger
        # than the macroparticle count and must never be used as this denominator -- confirmed
        # empirically to silently produce a "transmission" far too small (e.g. 1e-5% instead of
        # ~40%) when it was used here previously.
        if n_macroparticles is not None and int(n_macroparticles) > 0:
            n_initial = int(n_macroparticles)
        else:
            n_initial = int(M0.shape[0]) if M0.ndim == 2 else 0
        transmission = transmission_curves(M_snaps, z_snaps, tags, n_initial) if n_initial > 0 else None

        plot_beam_moments_evolution(table)
        saved += _capture_current_figure("beam_moments_evolution", output_dir)

        plot_beam_twiss_evolution(table, transmission=transmission)
        saved += _capture_current_figure("beam_twiss_evolution", output_dir)

    if thermo_info:
        plot_emission_history(thermo_info, show_components=True)
        saved += _capture_current_figure("emission_history", output_dir)

        plot_j_vs_n(thermo_info)
        saved += _capture_current_figure("emission_j_vs_n", output_dir)

    fig_cls = plot_class_conditioned_histograms(
        M0,
        Bout_M,
        lost_table=lost_table,
        t0_mm_c=np.asarray(thermo_info.get("t_emit_s", []), dtype=float) * c * 1e3 if thermo_info.get("t_emit_s", None) is not None else None,
    )
    if fig_cls is not None:
        saved += _save_figure(fig_cls, output_dir, "initial_class_conditioned_histograms")
        plt.close(fig_cls)

    back_bombardment_energy_map: dict[str, Any] | None = None
    if back_bombardment_data is not None:
        plot_back_bombardment_phase_space(back_bombardment_data)
        saved += _capture_current_figure("back_bombardment_phase_space", output_dir)

        plot_back_bombardment_screen_reach(back_bombardment_data, M_snaps, z_snaps)
        saved += _capture_current_figure("back_bombardment_screen_reach", output_dir)

        back_bombardment_energy_map = plot_back_bombardment_energy_density(back_bombardment_data)
        saved += _capture_current_figure("back_bombardment_energy_density", output_dir)

        plot_back_bombardment_power_density_vs_time(
            back_bombardment_data,
            cathode_radius_mm=float(back_bombardment_cathode_radius_mm),
        )
        saved += _capture_current_figure("back_bombardment_power_density_vs_time", output_dir)

    return {
        "saved_figures": saved,
        "back_bombardment_energy_map": back_bombardment_energy_map,
    }
