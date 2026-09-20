"""Field map plots and on-axis phase diagnostics."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..aperture import aperture_radius_profile_mm
from ..deflection_field import DEFAULT_B_PK_PER_A_T, DEFAULT_W_MM, DEFAULT_Z_P_MM, b0_deflection_T
from .style import DEFAULT_PLOT_STYLE, PlotStyleConfig, add_aperture_curve, add_reference_lines


def field_maps(
    xy: Dict[str, np.ndarray],
    yz: Dict[str, np.ndarray],
    t_ns: np.ndarray,
    t_crest: float,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    Ez_grid: np.ndarray,
    lambda_m: float,
    *,
    Er_grid: Optional[np.ndarray] = None,
    Bt_grid: Optional[np.ndarray] = None,
    Bz_grid: Optional[np.ndarray] = None,
    z_end_m: Optional[float] = None,
    measured_z_max_m: Optional[float] = None,
    aperture_delta_mm: Optional[float] = None,
    style: PlotStyleConfig | None = None,
    show_colorbar: bool = True,
    density_cmap=None,
    xy_percentile: float = 65.0,
    phase_deg: float = 0.0,
    source_label: str = "Remcom XFdtd",
    tail_label: str = "qualified modeled tail (not solver sampled)",
):
    """Plot the processed axisymmetric fields actually supplied to RF-Track.

    ``xy``, ``yz``, ``t_ns`` and ``t_crest`` remain positional only for compatibility with older
    notebook/script calls; raw solver-axis panels are deliberately no longer plotted.  Mixing
    those panels with the transformed RF-Track fields obscured that they use different coordinate
    conventions and, more importantly, could show a component that was not used for tracking.

    Complex E and B phasors are evaluated at one common ``phase_deg``.  ``Ez``/``Bz`` are extended
    evenly across the signed-radius display, while transverse ``Er``/``Bt`` are extended oddly,
    giving the physical Cartesian cut implied by an axisymmetric vector field.  B is displayed in
    mT and E in MV/m so each panel has a readable, explicitly independent color scale. When
    ``measured_z_max_m`` is supplied, the downstream modeled continuation is shaded and labelled
    on every panel so it cannot be mistaken for solver-sampled volume data.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    del xy, yz, t_ns, t_crest, xy_percentile  # retained compatibility-only arguments
    style = DEFAULT_PLOT_STYLE if style is None else style
    cmap_diverging = plt.get_cmap("RdBu_r") if density_cmap is None else density_cmap
    phase = np.exp(1j * np.deg2rad(float(phase_deg)))

    ez = np.asarray(Ez_grid)
    if ez.ndim != 2:
        raise ValueError(f"field_maps: Ez_grid must be 2D, got shape {ez.shape}")
    for name, grid in (("Er_grid", Er_grid), ("Bt_grid", Bt_grid), ("Bz_grid", Bz_grid)):
        if grid is not None and np.shape(grid) != ez.shape:
            raise ValueError(f"field_maps: {name} shape {np.shape(grid)} does not match Ez_grid {ez.shape}")

    panels = [(ez, +1, r"E_z", 1.0e-6, "MV/m")]
    if Er_grid is not None:
        panels.append((np.asarray(Er_grid), -1, r"E_r", 1.0e-6, "MV/m"))
    if Bt_grid is not None:
        panels.append((np.asarray(Bt_grid), -1, r"B_\theta", 1.0e3, "mT"))
    if Bz_grid is not None:
        panels.append((np.asarray(Bz_grid), +1, r"B_z", 1.0e3, "mT"))

    n_panels = len(panels)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(13.5, 3.5 * n_panels + 1.0), constrained_layout=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    # Exclude r=0 from the reflected half so the actual on-axis sample appears exactly once.
    r_neg = -r_grid[:0:-1]
    r_full = np.concatenate([r_neg, r_grid])
    extent_full = [z_grid[0] * 1e3, z_grid[-1] * 1e3, r_full[0] * 1e3, r_full[-1] * 1e3]

    z_end_mm = float(z_end_m) * 1e3 if z_end_m is not None else None
    measured_z_max_mm = (
        float(measured_z_max_m) * 1e3 if measured_z_max_m is not None else None
    )
    if measured_z_max_mm is not None and not (
        extent_full[0] - 1.0e-9 <= measured_z_max_mm <= extent_full[1] + 1.0e-9
    ):
        raise ValueError(
            "field_maps: measured_z_max_m must lie inside the displayed z-grid support"
        )
    lambda_quarter_mm = lambda_m / 4 * 1e3
    aperture_r_mm = (
        aperture_radius_profile_mm(z_grid * 1e3, float(aperture_delta_mm))
        if aperture_delta_mm is not None
        else None
    )

    def _panel(ax, field_grid, parity, symbol, scale, unit):
        at_phase = np.real(np.asarray(field_grid) * phase) * scale
        field_full = np.concatenate([parity * at_phase[:, :0:-1], at_phase], axis=1)
        # A low percentile is the right default for panels whose amplitude is spread over the
        # domain, but it destroys panels like B_z whose content is a thin, high-amplitude wall
        # layer over a near-zero bulk: the 98.5th percentile there sat ~67x below the true
        # maximum, so the only visible feature was uniform saturation. Take the gentler of a
        # high percentile and the true maximum, and say so on the panel when clipping remains.
        abs_full = np.abs(field_full)
        peak = float(np.max(abs_full)) if abs_full.size else 1.0
        vmax = float(np.percentile(abs_full, 99.5)) if abs_full.size else 1.0
        if not (np.isfinite(vmax) and vmax > 0.0):
            vmax = 1.0
        if np.isfinite(peak) and peak > 0.0 and peak < 20.0 * vmax:
            vmax = peak          # no pathological outlier: show the component honestly
        clipped = np.isfinite(peak) and peak > 1.001 * vmax
        im = ax.imshow(
            field_full.T,
            aspect="auto",
            origin="lower",
            extent=extent_full,
            cmap=cmap_diverging,
            norm=colors.Normalize(vmin=-vmax, vmax=vmax),
        )
        add_reference_lines(
            ax,
            cathode_z_mm=0.0,
            z_end_mm=z_end_mm,
            lambda_quarter_mm=lambda_quarter_mm,
            halo=True,
        )
        if aperture_r_mm is not None:
            add_aperture_curve(ax, z_grid * 1e3, aperture_r_mm)
        if measured_z_max_mm is not None and measured_z_max_mm < extent_full[1] - 1.0e-9:
            ax.axvspan(
                measured_z_max_mm,
                extent_full[1],
                facecolor="white",
                edgecolor="0.25",
                hatch="////",
                alpha=0.24,
                linewidth=0.0,
                label=str(tail_label),
                zorder=3,
            )
            ax.axvline(
                measured_z_max_mm,
                color="0.2",
                ls="--",
                lw=1.0,
                label="measured-volume end",
                zorder=4,
            )
        ax.axhline(0, color="black", ls=":", lw=0.8, alpha=0.4)
        ax.set_xlabel(r"$z$ (mm)", fontsize=12)
        ax.set_ylabel(r"signed radius (mm)", fontsize=12)
        clip_note = (
            rf"; colour clipped at {vmax:.3g} {unit}, peak {peak:.3g} {unit}" if clipped else ""
        )
        ax.set_title(
            rf"Field used by RF-Track: $\Re\{{{symbol}\,e^{{i\phi}}\}}$ at "
            rf"$\phi={float(phase_deg):.1f}^\circ${clip_note}",
            fontsize=13,
        )
        ax.tick_params(labelsize=10)
        if bool(show_colorbar):
            fig.colorbar(im, ax=ax, location="right", pad=0.015, fraction=0.025).set_label(
                rf"$\Re({symbol})$ ({unit})", fontsize=11,
            )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right", frameon=True, facecolor="white", framealpha=0.75, fontsize=9)
        return im

    for ax, panel in zip(axes, panels):
        _panel(ax, *panel)

    fig.suptitle(
        # cmr10 (the Computer-Modern face selected in plotting/style.py) has no U+2014
        # glyph, so an em-dash here renders as a missing-character box. Use a colon.
        f"EM FDTD Solver ({source_label}) field maps used by RF-Track: axisymmetric projection",
        fontsize=14,
    )

    plt.show()
    return fig


def axis_phase(
    Ez_axis: np.ndarray,
    z_grid: np.ndarray,
    Ez0_phasor_axis: complex,
    emission_phase_start: float,
    emission_phase_range: float,
    lambda_m: float,
    *,
    z_end_m: Optional[float] = None,
) -> Tuple[float, float]:
    """Auto phase from on-axis phasor and plot Ez(z) at a dense, evenly-spaced sweep of phases.

    Each curve is colored by `cos(phase offset from crest)` on a red-blue diverging colormap
    (`RdBu`): blue (+1) at the crest -- the phase of maximum acceleration -- through white (0) at
    the +/-90 deg transition, to red (-1) at 180 deg from crest -- maximum deceleration. A single
    colorbar encodes phase instead of a per-curve legend entry, which would not scale to a dense
    sweep.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    phi_opt = -np.angle(Ez_axis[np.argmax(np.abs(Ez_axis))])
    phi_zero_deg = (90.0 - np.rad2deg(np.angle(Ez0_phasor_axis))) % 360.0
    phi_crest_deg = (phi_zero_deg + 90.0) % 360.0
    transport_phase_deg = (phi_zero_deg + float(emission_phase_start)) % 360.0

    print(f"Auto phase: Ez0 crosses 0 at phi approx {phi_zero_deg:.2f} deg")
    print(f"Auto crest phase at cathode: phi approx {phi_crest_deg:.2f} deg")
    print(
        f"Transport phase (t=0): phi = {transport_phase_deg:.2f} deg "
        f"(zero-crossing reference + start shift {float(emission_phase_start):.1f} deg)"
    )
    print(f"Emission window: {float(emission_phase_range):.1f} deg")

    offsets_deg = np.arange(-180.0, 180.0 + 1e-9, 15.0)
    phases_deg_plot = [np.rad2deg(phi_opt) + d for d in offsets_deg]

    cmap_phase = plt.get_cmap("RdBu")
    norm_phase = Normalize(vmin=-1.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for deg, offset in zip(phases_deg_plot, offsets_deg):
        phi = np.deg2rad(deg)
        Ez_phase = np.real(Ez_axis * np.exp(1j * phi))
        color = cmap_phase(norm_phase(np.cos(np.deg2rad(offset))))
        ax.plot(z_grid * 1e3, Ez_phase, lw=1.3, color=color, alpha=0.9)

    z_end_mm = float(z_end_m) * 1e3 if z_end_m is not None else None

    add_reference_lines(
        ax,
        cathode_z_mm=0.0,
        z_end_mm=z_end_mm,
        lambda_quarter_mm=lambda_m / 4 * 1e3,
        halo=False,
    )

    sm = cm.ScalarMappable(cmap=cmap_phase, norm=norm_phase)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Accelerating $\\leftrightarrow$ Decelerating  ($\\cos\\Delta\\phi$ from crest)", fontsize=11)

    ax.set_xlabel(r"$z$ (mm)", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Re}(E_z)\ (r=0)$ (V/m)", fontsize=12)
    ax.set_title(r"On-axis $E_z$ field at selected phases", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return float(transport_phase_deg), float(phi_zero_deg)


def plot_deflection_field_profile(
    z_mm: np.ndarray,
    configured_current_A: float,
    applied_current_A: float,
    B_pk_per_A_T: float = DEFAULT_B_PK_PER_A_T,
    z_p_mm: float = DEFAULT_Z_P_MM,
    w_mm: float = DEFAULT_W_MM,
) -> Optional[Dict[str, np.ndarray]]:
    """Plot the deflection field profile actually applied to a run.

    Callers must resolve `applied_current_A` themselves as `configured_current_A if
    deflection_enabled else 0.0` (e.g. `VolumeBuildParams.applied_deflection_current_A`) -- this
    function never re-derives "applied" from "configured" so a caller cannot accidentally plot a
    configured-but-unapplied current as though it were part of the run. When the magnet is disabled
    (`applied_current_A == 0.0` but `configured_current_A != 0.0`), this prints a short
    not-applicable note and returns `None` instead of drawing a curve for a field that had no
    effect on this run; when both are zero (magnet genuinely off with no configured current), the
    same applies. A plot is produced only when a nonzero field was actually applied.
    """
    import matplotlib.pyplot as plt

    if applied_current_A == 0.0:
        reason = (
            f"deflection magnet disabled (configured I={configured_current_A:.4g} A, not applied)"
            if configured_current_A != 0.0
            else "deflection magnet disabled (no configured current)"
        )
        print(f"Deflection field profile: skipped -- {reason}.")
        return None

    z_mm = np.asarray(z_mm, dtype=float)
    B_T = b0_deflection_T(z_mm, applied_current_A, B_pk_per_A_T, z_p_mm, w_mm)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(z_mm, B_T * 1e3, lw=1.8, color="tab:purple")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(r"$z$ (mm, from cathode)", fontsize=12)
    ax.set_ylabel(r"$B_x$ (mT)", fontsize=12)
    ax.set_title(f"Deflection field profile (applied I = {applied_current_A:.4g} A)", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {"z_mm": z_mm, "Bx_T": B_T, "applied_current_A": np.array(float(applied_current_A))}
