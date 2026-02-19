"""Field map plots and on-axis phase diagnostics."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def field_maps(
    xy: Dict[str, np.ndarray],
    yz: Dict[str, np.ndarray],
    t_ns: np.ndarray,
    t_crest: float,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    Ez_grid: np.ndarray,
    lambda_m: float,
):
    """Plot raw field maps and RF-Track grid."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import matplotlib.colors as colors

    i_snap = int(np.argmin(np.abs(t_ns - t_crest)))

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    verts_xy = xy["vertices"]
    tri_xy = xy["facets"]
    Ux = verts_xy[:, 0]
    Vy = verts_xy[:, 1]
    Fx = np.asarray(xy["Ez"])[:, i_snap]

    triang_xy = mtri.Triangulation(Ux, Vy, triangles=tri_xy)
    ax_xy = fig.add_subplot(gs[0, 0])
    p_xy = float(np.percentile(np.abs(Fx), 99)) if Fx.size else 1.0
    p_xy = p_xy if p_xy > 0 else 1.0
    norm_xy = colors.TwoSlopeNorm(vcenter=0.0, vmin=-p_xy, vmax=p_xy)
    levels_xy = np.linspace(-p_xy, p_xy, 257)
    cf_xy = ax_xy.tricontourf(triang_xy, Fx, levels=levels_xy, cmap="coolwarm", norm=norm_xy)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x [mm]")
    ax_xy.set_ylabel("y [mm]")
    ax_xy.set_title("Raw mesh: XY Ez at crest")
    plt.colorbar(cf_xy, ax=ax_xy, label="Ez [V/m]")

    verts_yz = yz["vertices"]
    tri_yz = yz["facets"]
    Uy = verts_yz[:, 1]
    Vz = verts_yz[:, 2]
    Fy = np.asarray(yz["Ez"])[:, i_snap]

    triang_yz = mtri.Triangulation(Vz, Uy, triangles=tri_yz)
    ax_yz = fig.add_subplot(gs[0, 1])
    p_yz = float(np.percentile(np.abs(Fy), 99)) if Fy.size else 1.0
    p_yz = p_yz if p_yz > 0 else 1.0
    norm_yz = colors.TwoSlopeNorm(vcenter=0.0, vmin=-p_yz, vmax=p_yz)
    levels_yz = np.linspace(-p_yz, p_yz, 257)
    cf_yz = ax_yz.tricontourf(triang_yz, Fy, levels=levels_yz, cmap="coolwarm", norm=norm_yz)
    ax_yz.set_aspect("equal", adjustable="box")
    ax_yz.set_xlabel("z [mm]")
    ax_yz.set_ylabel("y [mm]")
    ax_yz.set_title("Raw mesh (rotated): YZ Ez at crest")
    plt.colorbar(cf_yz, ax=ax_yz, label="Ez [V/m]")

    r_neg = -r_grid[::-1]
    r_full = np.concatenate([r_neg, r_grid[1:]])
    Ez_full = np.concatenate([Ez_grid[:, ::-1], Ez_grid[:, 1:]], axis=1)

    ax_rf = fig.add_subplot(gs[1, :])
    extent_full = [z_grid[0] * 1e3, z_grid[-1] * 1e3, r_full[0] * 1e3, r_full[-1] * 1e3]
    im = ax_rf.imshow(
        np.real(Ez_full.T),
        aspect="auto",
        origin="lower",
        extent=extent_full,
        cmap="plasma",
    )
    ax_rf.axvline(0, color="white", ls="--", lw=1, alpha=0.5, label="Cathode (z=0)")
    ax_rf.axvline(lambda_m / 4 * 1e3, color="cyan", ls="--", lw=1, alpha=0.7, label="lambda/4")
    ax_rf.axhline(0, color="white", ls=":", lw=0.8, alpha=0.4)
    ax_rf.set_xlabel("z [mm]")
    ax_rf.set_ylabel("r [mm]")
    ax_rf.set_title("RF-Track field map: Re(Ez)")
    legend = ax_rf.legend(frameon=False, loc="upper right", fontsize=16)
    for text in legend.get_texts():
        text.set_color("white")
    ax_rf.text(
        0,
        r_full[-1] * 1e3 * 0.93,
        "Cathode (z=0)",
        color="white",
        fontsize=16,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.15, edgecolor="none"),
    )
    ax_rf.text(
        lambda_m / 4 * 1e3,
        r_full[-1] * 1e3 * 0.93,
        "lambda/4",
        color="white",
        fontsize=16,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.15, edgecolor="none"),
    )
    plt.colorbar(im, ax=ax_rf, label="Ez [V/m]")

    plt.tight_layout()
    plt.show()


def axis_phase(
    Ez_axis: np.ndarray,
    z_grid: np.ndarray,
    Ez0_phasor_axis: complex,
    emission_phase_start: float,
    emission_phase_range: float,
    lambda_m: float,
) -> Tuple[float, float]:
    """Auto phase from on-axis phasor and plot Ez(z)."""
    import matplotlib.pyplot as plt

    phi_opt = -np.angle(Ez_axis[np.argmax(np.abs(Ez_axis))])
    phi_zero_deg = (90.0 - np.rad2deg(np.angle(Ez0_phasor_axis))) % 360.0
    transport_phase_deg = (phi_zero_deg + float(emission_phase_start)) % 360.0

    print(f"Auto phase: Ez0 crosses 0 at phi approx {phi_zero_deg:.2f} deg")
    print(
        f"Transport phase (t=0): phi = {transport_phase_deg:.2f} deg "
        f"(start shift {float(emission_phase_start):.1f} deg)"
    )
    print(f"Emission window: {float(emission_phase_range):.1f} deg")

    offsets_deg = [0, 30, 60, 90, 120, 150, 180]
    phases_deg_plot = [np.rad2deg(phi_opt) + d for d in offsets_deg]
    for extra_phase in (phi_zero_deg, transport_phase_deg):
        if not any(np.isclose(extra_phase, p, atol=1e-3) for p in phases_deg_plot):
            phases_deg_plot.append(extra_phase)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for deg in phases_deg_plot:
        phi = np.deg2rad(deg)
        Ez_phase = np.real(Ez_axis * np.exp(1j * phi))
        ax.plot(z_grid * 1e3, Ez_phase, lw=1.5, label=f"phi = {deg:.1f} deg")

    ax.axvline(0, color="red", ls="--", lw=1, alpha=0.6, label="Cathode")
    ax.axvline(
        lambda_m / 4 * 1e3,
        color="blue",
        ls="--",
        lw=1,
        alpha=0.6,
        label=f"lambda/4 = {lambda_m/4*1e3:.2f} mm",
    )
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Re{Ez(r=0, z, phi)} [V/m]")
    ax.set_title("On-axis field at selected phases")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return float(transport_phase_deg), float(phi_zero_deg)
