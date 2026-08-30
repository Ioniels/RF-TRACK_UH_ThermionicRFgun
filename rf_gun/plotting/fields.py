"""Field map plots and on-axis phase diagnostics."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..aperture import aperture_radius_profile_mm
from .style import (
    DEFAULT_PLOT_STYLE,
    PlotStyleConfig,
    add_aperture_curve,
    add_reference_lines,
    get_recentered_diverging_cmap,
)


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
    z_end_m: Optional[float] = None,
    aperture_delta_mm: Optional[float] = None,
    style: PlotStyleConfig | None = None,
    show_colorbar: bool = True,
    density_cmap=None,
    xy_percentile: float = 65.0,
):
    """Plot raw field maps (top row: r-z cavity view, x-z waveguide/iris view) and the RF-Track
    (r, z) grid (bottom, one full-width panel per field component), each with its own colorbar.

    Both field components are signed, so top-row panels use a diverging colormap (`RdBu_r`). The
    cavity view (`ax_xy`) uses a tighter normalization (`xy_percentile`, default 65th percentile
    of |value|) than the waveguide view (98.5th percentile), since most of its structure sits at
    lower field magnitude; values beyond that range are clipped to a darker off-scale color
    instead of saturating (see `get_recentered_diverging_cmap`).

    `aperture_delta_mm`, when given, overlays the dynamic aperture's R(z) profile (see
    `rf_gun.aperture.aperture_radius_profile_mm`) as +/-R(z) curves on both bottom panels -- this
    is the field map RF-Track actually uses, so it's the natural place to show the physical
    channel it's paired with during tracking.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import matplotlib.colors as colors

    style = DEFAULT_PLOT_STYLE if style is None else style
    cmap_diverging = plt.get_cmap("RdBu_r") if density_cmap is None else density_cmap
    cmap_xy = get_recentered_diverging_cmap(base="RdBu_r")

    i_snap = int(np.argmin(np.abs(t_ns - t_crest)))

    has_er = Er_grid is not None
    n_bottom_rows = 2 if has_er else 1
    # constrained_layout reserves space for each panel's own colorbar without overlapping the axes.
    fig = plt.figure(figsize=(14, 7.5 + 3.6 * n_bottom_rows), constrained_layout=True)
    gs = fig.add_gridspec(1 + n_bottom_rows, 2, width_ratios=[1.0, 1.10], height_ratios=[0.92] + [1.15] * n_bottom_rows)

    verts_xy = xy["vertices"]
    tri_xy = xy["facets"]
    Ux = verts_xy[:, 0]
    Vy = verts_xy[:, 1]
    Fx = np.asarray(xy["Ez"])[:, i_snap]

    triang_xy = mtri.Triangulation(Ux, Vy, triangles=tri_xy)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel(r"$r$ (mm)", fontsize=12)
    ax_xy.set_ylabel(r"$z$ (mm)", fontsize=12)
    ax_xy.tick_params(labelsize=10)

    verts_yz = yz["vertices"]
    tri_yz = yz["facets"]
    Uy = verts_yz[:, 1]
    Vz = verts_yz[:, 2]
    Fy = np.asarray(yz["Ez"])[:, i_snap]

    triang_yz = mtri.Triangulation(Vz, Uy, triangles=tri_yz)
    ax_yz = fig.add_subplot(gs[0, 1])

    vmax_xy = float(np.percentile(np.abs(Fx), xy_percentile)) if Fx.size else 1.0
    vmax_xy = vmax_xy if vmax_xy > 0 else 1.0
    norm_xy = colors.Normalize(vmin=-vmax_xy, vmax=vmax_xy)

    vmax_yz = float(np.percentile(np.abs(Fy), 98.5)) if Fy.size else 1.0
    vmax_yz = vmax_yz if vmax_yz > 0 else 1.0
    norm_yz = colors.Normalize(vmin=-vmax_yz, vmax=vmax_yz)

    cf_xy = ax_xy.tripcolor(triang_xy, Fx, cmap=cmap_xy, norm=norm_xy, shading="gouraud")
    cf_yz = ax_yz.tripcolor(triang_yz, Fy, cmap=cmap_diverging, norm=norm_yz, shading="gouraud")
    ax_yz.set_aspect("equal", adjustable="box")
    ax_yz.set_xlabel(r"$x$ (mm)", fontsize=12)
    ax_yz.set_ylabel(r"$z$ (mm)", fontsize=12)
    ax_yz.tick_params(labelsize=10)
    if bool(show_colorbar):
        fig.colorbar(cf_xy, ax=ax_xy, location="right", pad=0.02, fraction=0.046, extend="both").set_label(
            r"$\Re(E_z)$ (V/m)", fontsize=11
        )
        fig.colorbar(cf_yz, ax=ax_yz, location="right", pad=0.02, fraction=0.046).set_label(
            r"$\Re(E_z)$ (V/m)", fontsize=11
        )

    x_lo, x_hi = np.percentile(Ux, [0.5, 99.5])
    y_lo_xy, y_hi_xy = np.percentile(Vy, [0.5, 99.5])
    x_lo_yz, x_hi_yz = np.percentile(Vz, [0.5, 99.5])
    z_lo_yz, z_hi_yz = np.percentile(Uy, [0.5, 99.5])

    ax_xy.set_xlim(float(x_lo), float(x_hi))
    ax_xy.set_ylim(float(y_lo_xy), float(y_hi_xy))
    ax_yz.set_xlim(float(x_lo_yz), float(x_hi_yz))
    ax_yz.set_ylim(float(z_lo_yz), float(z_hi_yz))

    fig.suptitle(
        "EM FDTD Solver (Remcom - XFdtd) field maps: cavity, iris and waveguide (no electron beam)",
        fontsize=13,
    )

    r_neg = -r_grid[::-1]
    r_full = np.concatenate([r_neg, r_grid[1:]])
    Ez_full = np.concatenate([Ez_grid[:, ::-1], Ez_grid[:, 1:]], axis=1)
    extent_full = [z_grid[0] * 1e3, z_grid[-1] * 1e3, r_full[0] * 1e3, r_full[-1] * 1e3]

    z_end_mm = float(z_end_m) * 1e3 if z_end_m is not None else None
    lambda_quarter_mm = lambda_m / 4 * 1e3
    aperture_r_mm = (
        aperture_radius_profile_mm(z_grid * 1e3, float(aperture_delta_mm))
        if aperture_delta_mm is not None
        else None
    )

    def _bottom_panel(ax, field_full, title, cmap, norm, cbar_label):
        im = ax.imshow(
            np.real(field_full.T),
            aspect="auto",
            origin="lower",
            extent=extent_full,
            cmap=cmap,
            norm=norm,
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
        ax.axhline(0, color="black", ls=":", lw=0.8, alpha=0.4)
        ax.set_xlabel(r"$z$ (mm)", fontsize=12)
        ax.set_ylabel(r"$r$ (mm)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=10)
        if bool(show_colorbar):
            fig.colorbar(im, ax=ax, location="right", pad=0.015, fraction=0.025).set_label(cbar_label, fontsize=11)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right", frameon=True, facecolor="white", framealpha=0.75, fontsize=9)
        return im

    # Ez is signed, like Er below it -- use the same diverging colormap and a symmetric
    # (zero-centered) normalization rather than a sequential density colormap (which would neither
    # show sign nor be centered on zero; see `rf_gun.plotting.style`'s module docstring for the
    # density-vs-signed-field colormap distinction this project draws).
    vmax_ez = float(np.percentile(np.abs(np.real(Ez_full)), 98.5)) if Ez_full.size else 1.0
    vmax_ez = vmax_ez if vmax_ez > 0 else 1.0
    ax_rf_ez = fig.add_subplot(gs[1, :])
    _bottom_panel(
        ax_rf_ez, Ez_full, r"Field map used by RF-Track: $\Re(E_z)$", cmap_diverging,
        colors.Normalize(vmin=-vmax_ez, vmax=vmax_ez), r"$\Re(E_z)$ (V/m)",
    )

    if has_er:
        Er_full = np.concatenate([Er_grid[:, ::-1], Er_grid[:, 1:]], axis=1)
        vmax_er = float(np.percentile(np.abs(np.real(Er_full)), 98.5)) if Er_full.size else 1.0
        vmax_er = vmax_er if vmax_er > 0 else 1.0
        ax_rf_er = fig.add_subplot(gs[2, :])
        _bottom_panel(
            ax_rf_er,
            Er_full,
            r"RF-Track grid: $\Re(E_r)$",
            cmap_diverging,
            colors.Normalize(vmin=-vmax_er, vmax=vmax_er),
            r"$\Re(E_r)$ (V/m)",
        )

    plt.show()


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
