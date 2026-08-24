"""Beam-properties evolution plots along z.

Both figures are driven entirely by the table returned by
`rf_gun.beam_properties.compute_beam_properties` (a `list[dict]`, one row per screen, already
computed on the forward-going + aperture-surviving population -- see that module's docstring) --
neither figure recomputes anything from `M_snaps` itself. This replaces the project's former
`plot_evolution`/`plot_twiss_evolution`/`plot_emittance_evolution`/`plot_transmission_evolution`
(the last of which read RF-Track's per-screen `Screen.get_info()` "transmission" field; the
three-curve transmission panel here is row-count-based instead, via
`rf_gun.beam_properties.transmission_curves`).

Color convention (see `rf_gun.plotting.style`'s module docstring): `_pair` panels (x-plane vs
y-plane) are always blue (solid circle) vs red (dashed square); `_single` panels are blue for
mean-type quantities, red for sigma/spread-type quantities; the 2-curve transmission panel is
blue (backward + forward, i.e. survived the dynamic aperture) / red (forward only, the more
restrictive subset of that same population).
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from .style import COLOR_PRIMARY, COLOR_SECONDARY


def _col(table: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row.get(key, np.nan) for row in table], dtype=float)


def _pair(ax, z_mm, table, key_x, key_y, label_x, label_y, ylabel, title):
    ax.plot(z_mm, _col(table, key_x), "o-", ms=3, color=COLOR_PRIMARY, label=label_x)
    ax.plot(z_mm, _col(table, key_y), "s--", ms=3, color=COLOR_SECONDARY, label=label_y)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)


def _single(ax, z_mm, table, key, ylabel, title, kind="mean"):
    color = COLOR_PRIMARY if kind == "mean" else COLOR_SECONDARY
    ax.plot(z_mm, _col(table, key), "o-", ms=3, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_beam_moments_evolution(table: Sequence[Dict[str, Any]]):
    """Figure 1: raw moments (mean, RMS size, RMS momentum spread) vs z, 4 rows x 2 columns.

    Row 1: mean x/y | mean ToF. Row 2: mean px/py | mean pz.
    Row 3: sigma x/y | sigma ToF. Row 4: sigma px/py | sigma pz.

    ToF (`%t`), not `%Z`, is used for the longitudinal moment panel -- a screen's own `%Z` is not
    a lab-frame position (see `rf_gun.beam_properties`'s module docstring).
    """
    import matplotlib.pyplot as plt

    if not table:
        print("No beam-properties rows available.")
        return

    z_mm = _col(table, "z_mm")
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True)

    _pair(axes[0, 0], z_mm, table, "mean_x", "mean_y", "$x$", "$y$",
          r"$\langle x\rangle,\langle y\rangle\,(\mathrm{mm})$", r"$\langle x\rangle,\langle y\rangle$ vs $z$")
    _single(axes[0, 1], z_mm, table, "mean_t_ns",
            r"$\langle\mathrm{ToF}\rangle\,(\mathrm{ns})$", r"$\langle\mathrm{ToF}\rangle$ vs $z$", kind="mean")

    _pair(axes[1, 0], z_mm, table, "mean_px", "mean_py", "$p_x$", "$p_y$",
          r"$\langle p_x\rangle,\langle p_y\rangle\,(\mathrm{MeV}/c)$", r"$\langle p_x\rangle,\langle p_y\rangle$ vs $z$")
    _single(axes[1, 1], z_mm, table, "mean_pz",
            r"$\langle p_z\rangle\,(\mathrm{MeV}/c)$", r"$\langle p_z\rangle$ vs $z$", kind="mean")

    _pair(axes[2, 0], z_mm, table, "sigma_x", "sigma_y", "$\\sigma_x$", "$\\sigma_y$",
          r"$\sigma_x,\sigma_y\,(\mathrm{mm})$", r"$\sigma_x,\sigma_y$ vs $z$")
    _single(axes[2, 1], z_mm, table, "sigma_t_ns",
            r"$\sigma_{\mathrm{ToF}}\,(\mathrm{ns})$", r"$\sigma_{\mathrm{ToF}}$ vs $z$", kind="sigma")

    _pair(axes[3, 0], z_mm, table, "sigma_px", "sigma_py", "$\\sigma_{p_x}$", "$\\sigma_{p_y}$",
          r"$\sigma_{p_x},\sigma_{p_y}\,(\mathrm{MeV}/c)$", r"$\sigma_{p_x},\sigma_{p_y}$ vs $z$")
    _single(axes[3, 1], z_mm, table, "sigma_pz",
            r"$\sigma_{p_z}\,(\mathrm{MeV}/c)$", r"$\sigma_{p_z}$ vs $z$", kind="sigma")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$z\,(\mathrm{mm})$")

    fig.suptitle("Beam moments vs $z$", y=0.995)
    plt.tight_layout()
    plt.show()


def plot_beam_twiss_evolution(
    table: Sequence[Dict[str, Any]],
    transmission: Dict[str, np.ndarray] | None = None,
):
    """Figure 2: Twiss, dispersion, emittance, transmission, energy/time vs z, 4 rows x 3 columns.

    Row 1: alpha x/y | beta x/y | gamma x/y.
    Row 2: mean x/y | dispersion Dx/Dy | dispersion D'x/D'y.
    Row 3: geometric emittance x/y | normalized emittance x/y | transmission (3 curves).
    Row 4: energy spread (sigma_E) | time-of-flight spread (sigma_t) | mean kinetic energy.

    `transmission`, if given, is `rf_gun.beam_properties.transmission_curves`'s return dict
    (`z_mm`, `not_lost`, `forward_and_surviving`); the row-3-col-3 panel is left blank (with a
    note) if not supplied.
    """
    import matplotlib.pyplot as plt

    if not table:
        print("No beam-properties rows available.")
        return

    z_mm = _col(table, "z_mm")
    fig, axes = plt.subplots(4, 3, figsize=(16.5, 13), sharex=True)

    _pair(axes[0, 0], z_mm, table, "alpha_x", "alpha_y", r"$\alpha_x$", r"$\alpha_y$",
          r"$\alpha_x,\alpha_y$", r"$\alpha_x,\alpha_y$ vs $z$")
    _pair(axes[0, 1], z_mm, table, "beta_x", "beta_y", r"$\beta_x$", r"$\beta_y$",
          r"$\beta_x,\beta_y\,(\mathrm{mm})$", r"$\beta_x,\beta_y$ vs $z$")
    _pair(axes[0, 2], z_mm, table, "gamma_x", "gamma_y", r"$\gamma_x$", r"$\gamma_y$",
          r"$\gamma_x,\gamma_y\,(\mathrm{mm}^{-1})$", r"$\gamma_x,\gamma_y$ vs $z$")

    _pair(axes[1, 0], z_mm, table, "mean_x", "mean_y", "$x$", "$y$",
          r"$\langle x\rangle,\langle y\rangle\,(\mathrm{mm})$", r"$\langle x\rangle,\langle y\rangle$ vs $z$")
    _pair(axes[1, 1], z_mm, table, "disp_x", "disp_y", "$D_x$", "$D_y$",
          r"$D_x,D_y\,(\mathrm{mm})$", r"$D_x,D_y$ vs $z$")
    _pair(axes[1, 2], z_mm, table, "disp_px", "disp_py", "$D_x'$", "$D_y'$",
          r"$D_x',D_y'\,(\mathrm{rad})$", r"$D_x',D_y'$ vs $z$")

    _pair(axes[2, 0], z_mm, table, "emitt_x_geom", "emitt_y_geom", r"$\varepsilon_x$", r"$\varepsilon_y$",
          r"$\varepsilon_{x,y}\,(\mathrm{mm}\cdot\mathrm{rad})$", "Geometric emittance vs $z$")
    _pair(axes[2, 1], z_mm, table, "emitt_x_norm", "emitt_y_norm", r"$\varepsilon_x$", r"$\varepsilon_y$",
          r"$\varepsilon_{x,y}\,(\mathrm{mm}\cdot\mathrm{mrad})$", "Normalized emittance vs $z$")
    if transmission is not None and len(transmission.get("z_mm", [])):
        t_mm = np.asarray(transmission["z_mm"], dtype=float)
        axes[2, 2].plot(t_mm, 100.0 * np.asarray(transmission["not_lost"]), "o-", ms=3, color=COLOR_PRIMARY, label="backward + forward")
        axes[2, 2].plot(t_mm, 100.0 * np.asarray(transmission["forward_and_surviving"]), "s--", ms=3, color=COLOR_SECONDARY, label="forward only")
        axes[2, 2].set_ylabel(r"$T\,(\%)$")
        axes[2, 2].legend(frameon=False, fontsize=8)
    else:
        axes[2, 2].text(0.5, 0.5, "transmission not supplied", ha="center", va="center", transform=axes[2, 2].transAxes)
    axes[2, 2].set_title("Transmission vs $z$")
    axes[2, 2].grid(alpha=0.3)

    _single(axes[3, 0], z_mm, table, "sigma_E", r"$\sigma_E\,(\mathrm{MeV})$", r"$\sigma_E$ vs $z$", kind="sigma")
    _single(axes[3, 1], z_mm, table, "sigma_t_ns", r"$\sigma_{\mathrm{ToF}}\,(\mathrm{ns})$", r"$\sigma_{\mathrm{ToF}}$ vs $z$", kind="sigma")
    _single(axes[3, 2], z_mm, table, "mean_K", r"$\langle K\rangle\,(\mathrm{MeV})$", r"$\langle K\rangle$ vs $z$", kind="mean")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$z\,(\mathrm{mm})$")

    fig.suptitle("Twiss, dispersion, emittance, transmission vs $z$", y=0.995)
    plt.tight_layout()
    plt.show()
