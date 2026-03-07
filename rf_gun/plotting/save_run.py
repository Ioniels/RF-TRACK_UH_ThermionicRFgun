"""Batch plotting entry points for run outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..diagnostics import build_screen_summaries
from ..constants import c
from .emission import plot_emission_history, plot_j_vs_n
from .evolution import plot_evolution, plot_twiss_evolution, plot_emittance_evolution, plot_transmission_evolution
from .phase_space import plot_phase_space, plot_spectra, plot_screen_phase_space_slider


PHASE_SPACE_COLUMNS = ["x_mm", "px_MeV_c", "y_mm", "py_MeV_c", "z_mm", "pz_MeV_c"]


def _save_figure(fig, output_dir: Path, stem: str) -> list[str]:
    png_path = output_dir / f"{stem}.png"
    eps_path = output_dir / f"{stem}.eps"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    return [png_path.name, eps_path.name]


def _capture_current_figure(save_name: str, output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    if fig is None:
        return []
    saved = _save_figure(fig, output_dir, save_name)
    plt.close(fig)
    return saved


def plot_class_conditioned_histograms(
    initial_M: np.ndarray,
    final_M: np.ndarray,
    lost_table: np.ndarray | None = None,
    t0_mm_c: np.ndarray | None = None,
):
    """Initial distributions split by final class: transmitted/backward/lost."""
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
    axes[0].hist(1e3 * pz0[transmitted], bins=bins_pz, alpha=0.6, label="transmitted", color="tab:green")
    axes[0].hist(1e3 * pz0[backward], bins=bins_pz, alpha=0.6, label="backward/returned", color="tab:red")
    if lost_table is not None and np.asarray(lost_table).ndim == 2 and np.asarray(lost_table).shape[0] > 0:
        axes[0].text(0.98, 0.95, f"lost={int(np.asarray(lost_table).shape[0])}", transform=axes[0].transAxes, ha="right", va="top")
    axes[0].set_xlabel("initial pz [keV/c]")
    axes[0].set_ylabel("counts")
    axes[0].grid(alpha=0.3)

    if t0_ns is not None:
        bins_t0 = np.histogram_bin_edges(t0_ns[np.isfinite(t0_ns)], bins=60)
        axes[1].hist(t0_ns, bins=bins_t0, alpha=0.4, label="all", color="tab:blue")
        axes[1].hist(t0_ns[transmitted], bins=bins_t0, alpha=0.6, label="transmitted", color="tab:green")
        axes[1].hist(t0_ns[backward], bins=bins_t0, alpha=0.6, label="backward/returned", color="tab:red")
        axes[1].set_xlabel("initial t0 [ns]")
        axes[1].set_ylabel("counts")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "t0 unavailable", ha="center", va="center")

    bins_r = np.histogram_bin_edges(r0[np.isfinite(r0)], bins=60)
    axes[2].hist(r0, bins=bins_r, alpha=0.4, label="all", color="tab:blue")
    axes[2].hist(r0[transmitted], bins=bins_r, alpha=0.6, label="transmitted", color="tab:green")
    axes[2].hist(r0[backward], bins=bins_r, alpha=0.6, label="backward/returned", color="tab:red")
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
) -> Path:
    """Save one phase-space matrix to JSON with explicit schema fields."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2:
        arr = np.zeros((0, 6), dtype=float)

    cols = list(phase_space_columns) if phase_space_columns is not None else list(PHASE_SPACE_COLUMNS)
    payload = {
        "schema_version": 1,
        "label": str(label) if label is not None else None,
        "phase_space_columns": cols,
        "particle_count": int(arr.shape[0]),
        "phase_space": arr.tolist(),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def save_screen_phase_space_batch(
    output_dir: Path,
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    info_snaps: Sequence[Any] | None = None,
    *,
    B0=None,
    Bout=None,
    phase_fmt: str = "%X %Px %Y %Py %Z %Pz",
    clean_e: bool = True,
    clean_except_zpz: bool = False,
    n_real_ref: float | None = None,
    n_macroparticles: int | None = None,
    style=None,
    highlight_mode: str | None = None,
    show_colorbar: bool = False,
    save_json: bool = True,
) -> dict[str, Any]:
    """Save non-interactive phase-space figures for B0, screens and Bout."""
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
        info_local: Any = None,
        B0_local=None,
        Bout_local=None,
    ) -> None:
        nonlocal frame_idx
        plot_screen_phase_space_slider(
            M_snaps=[np.asarray(M_local, dtype=float)] if M_local is not None else [],
            z_snaps=[float(z_local)] if z_local is not None else [],
            info_snaps=[info_local] if info_local is not None else None,
            clean_e=clean_e,
            n_real_ref=n_real_ref,
            n_macroparticles=n_macroparticles,
            style=style,
            highlight_mode=highlight_mode,
            show_colorbar=show_colorbar,
            B0=B0_local,
            Bout=Bout_local,
            phase_fmt=phase_fmt,
            clean_except_zpz=clean_except_zpz,
        )
        fig_files = _capture_current_figure(stem, fig_dir)
        json_file = None
        if save_json:
            json_path = save_beam_phase_space_json(
                json_dir / f"{stem}.json",
                np.asarray(M_local, dtype=float),
                label=label,
            )
            json_file = json_path.name
        manifest.append(
            {
                "frame_index": int(frame_idx),
                "label": label,
                "z_mm": float(z_local) if z_local is not None else None,
                "figure_files": fig_files,
                "json_file": json_file,
            }
        )
        frame_idx += 1
        plt.close("all")

    if B0 is not None:
        M0 = np.array(B0.get_phase_space(phase_fmt, "all"), copy=True)
        _save_single("B0", f"frame_{frame_idx:04d}_B0", M_local=M0, B0_local=B0)

    z_mm = 1e3 * np.asarray(z_snaps, dtype=float) if z_snaps is not None else np.asarray([], dtype=float)
    n_screens = min(len(M_snaps), int(z_mm.size))
    for i in range(n_screens):
        info_i = info_snaps[i] if (info_snaps is not None and i < len(info_snaps)) else None
        z_i = float(z_mm[i])
        stem = f"frame_{frame_idx:04d}_screen_{i+1:03d}_z{_z_tag(z_i)}mm"
        _save_single(
            f"screen_{i+1}",
            stem,
            M_local=np.asarray(M_snaps[i], dtype=float),
            z_local=float(z_snaps[i]),
            info_local=info_i,
        )

    if Bout is not None:
        Mf = np.array(Bout.get_phase_space(phase_fmt, "all"), copy=True)
        _save_single("Bout", f"frame_{frame_idx:04d}_Bout", M_local=Mf, Bout_local=Bout)

    manifest_path = output_dir / "screen_phase_space_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"frames": manifest}, f, indent=2)

    return {
        "frame_count": int(len(manifest)),
        "figure_dir": str(fig_dir),
        "json_dir": str(json_dir) if save_json else None,
        "manifest_file": str(manifest_path),
    }


def save_run_figures(
    output_dir: Path,
    B0,
    Bout,
    transport_phase_deg: float,
    thermo_info: dict[str, Any],
    M_snaps: Sequence[np.ndarray],
    z_snaps: Sequence[float],
    I_snaps: Sequence[Any],
    *,
    phase_fmt: str = "%X %Px %Y %Py %Z %Pz",
    clean_e: bool = False,
    show_zle0: bool = True,
    n_real_ref: float | None = None,
    n_macroparticles: int | None = None,
    lost_table: np.ndarray | None = None,
) -> list[str]:
    """Generate and save a standard figure bundle for one run."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    plot_phase_space(B0, Bout, transport_phase_deg, clean_e=clean_e, show_zle0=show_zle0, phase_fmt=phase_fmt)
    saved += _capture_current_figure("initial_phase_space_x_px", output_dir)

    plot_spectra(Bout, transport_phase_deg, B0=B0, thermo_info=thermo_info, clean_e=clean_e, show_zle0=show_zle0, phase_fmt=phase_fmt)
    saved += _capture_current_figure("screen_spectra", output_dir)

    if M_snaps and z_snaps:
        plot_evolution(M_snaps, z_snaps, info_snaps=I_snaps, clean_e=clean_e)
        saved += _capture_current_figure("longitudinal_evolution", output_dir)

        plot_twiss_evolution(M_snaps, z_snaps, info_snaps=I_snaps, clean_e=clean_e)
        saved += _capture_current_figure("twiss_evolution", output_dir)

        plot_emittance_evolution(z_snaps, info_snaps=I_snaps)
        saved += _capture_current_figure("emittance_evolution", output_dir)

    if z_snaps and I_snaps:
        plot_transmission_evolution(
            z_snaps,
            I_snaps,
            n_real_ref=n_real_ref,
            n_macroparticles=n_macroparticles,
        )
        saved += _capture_current_figure("screen_transmission", output_dir)

    if thermo_info:
        plot_emission_history(thermo_info, show_components=True)
        saved += _capture_current_figure("emission_history", output_dir)

        plot_j_vs_n(thermo_info)
        saved += _capture_current_figure("emission_j_vs_n", output_dir)

    # Summary screen diagnostics figure
    if z_snaps:
        summaries = build_screen_summaries(z_snaps, I_snaps, M_snaps if M_snaps else None)
        if summaries:
            z_mm = np.asarray([1e3 * rec["z_m"] for rec in summaries], dtype=float)
            n_vals = np.asarray([rec.get("N", np.nan) for rec in summaries], dtype=float)
            trans = np.asarray([rec.get("transmission", np.nan) for rec in summaries], dtype=float)
            fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
            axes[0].plot(z_mm, n_vals, "o-", ms=3)
            axes[0].set_ylabel("screen N")
            axes[0].grid(alpha=0.3)
            axes[1].plot(z_mm, trans, "o-", ms=3, color="tab:green")
            axes[1].set_ylabel("transmission")
            axes[1].set_xlabel("z [mm]")
            axes[1].grid(alpha=0.3)
            fig.suptitle("screen summary diagnostics")
            fig.tight_layout()
            saved += _save_figure(fig, output_dir, "screen_summary")
            plt.close(fig)

    M0 = np.array(B0.get_phase_space(phase_fmt, "all"), copy=True)
    Mf = np.array(Bout.get_phase_space(phase_fmt, "all"), copy=True)
    fig_cls = plot_class_conditioned_histograms(
        M0,
        Mf,
        lost_table=lost_table,
        t0_mm_c=np.asarray(thermo_info.get("t_emit_s", []), dtype=float) * c * 1e3 if thermo_info.get("t_emit_s", None) is not None else None,
    )
    if fig_cls is not None:
        saved += _save_figure(fig_cls, output_dir, "initial_class_conditioned_histograms")
        plt.close(fig_cls)

    return saved
