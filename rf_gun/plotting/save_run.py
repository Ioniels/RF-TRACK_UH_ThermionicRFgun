"""Batch plotting entry points for run outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..diagnostics import build_screen_summaries
from ..constants import c
from .emission import plot_emission_history, plot_j_vs_n
from .evolution import plot_evolution, plot_twiss_evolution, plot_transmission_evolution
from .phase_space import plot_phase_space, plot_spectra


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
