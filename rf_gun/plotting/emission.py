"""Emission diagnostics plots."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .style import EMISSION_MODEL_COLORS


def _as_1d(arr):
    if arr is None:
        return None
    out = np.asarray(arr, dtype=float).reshape(-1)
    return out


def plot_emission_history(
    thermo_info: Dict[str, Any],
    show_components: bool = True,
    comparison_models: Optional[Sequence[str]] = None,
):
    """Emission history at cathode with model and sampled emission PDF.

    `comparison_models` overlays other emission kernels' J(t) on the current-density panel for
    visual comparison; only `thermo_info["emission_law"]` drives the actual macroparticle
    creation-time histogram.

    Top panel: if `thermo_info["Ez_corrected_t"]` is present (a converged Emission Fields
    Iteration spatial source), plots it against `Ez_t` (both raw signed field, directly
    comparable). Otherwise plots `F_t` (rectified extraction field) or raw `Ez_t` alone. `F_t` is
    never plotted next to `Ez_corrected_t`: rectified vs. signed would make an unchanged field
    look reversed.
    """
    import matplotlib.pyplot as plt

    from ..emission_models import evaluate_emission_model

    t_s = _as_1d(thermo_info.get("t_s", None))
    if t_s is None:
        print("Emission history: no time samples available.")
        return

    Ez_t = _as_1d(thermo_info.get("Ez_t", None))
    Ez_corrected_t = _as_1d(thermo_info.get("Ez_corrected_t", None))
    F_t = _as_1d(thermo_info.get("F_t", None))
    J_t = _as_1d(thermo_info.get("J_Apm2_t", None))
    J_th_t = _as_1d(thermo_info.get("J_th_Apm2_t", None))
    J_fe_t = _as_1d(thermo_info.get("J_fe_Apm2_t", None))
    I_t = _as_1d(thermo_info.get("I_A_t", None))
    area_m2 = thermo_info.get("area_m2", None)
    n_t = _as_1d(thermo_info.get("n_t", None))
    t_emit_s = _as_1d(thermo_info.get("t_emit_s", None))

    do_components = bool(show_components and J_th_t is not None and J_fe_t is not None)
    nrows = 3 if do_components else 2

    t_ns = np.asarray(t_s) * 1e9
    fig, axes = plt.subplots(nrows, 1, figsize=(8.8, 2.55 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    have_corrected = Ez_corrected_t is not None and Ez_corrected_t.size == t_ns.size
    if have_corrected and Ez_t is not None:
        # Ez_t/Ez_corrected_t share one sign convention (raw signed field) and are directly
        # comparable; F_t below is rectified and would look sign-reversed next to either.
        axes[0].plot(t_ns, Ez_t, lw=1.6, color="tab:blue", label="external (RF only)")
        axes[0].plot(t_ns, Ez_corrected_t, lw=1.6, ls="--", color="tab:red",
                      label="corrected (RF+SC+mirror)")
        axes[0].set_ylabel(r"$E_z\,(\mathrm{V\,m^{-1}})$")
        axes[0].set_title(r"Cathode field")
    elif F_t is not None:
        axes[0].plot(t_ns, F_t, lw=1.6, label="F(t)")
        axes[0].set_ylabel(r"$F\,(\mathrm{V\,m^{-1}})$")
        axes[0].set_title(r"Cathode extraction field")
    elif Ez_t is not None:
        axes[0].plot(t_ns, Ez_t, lw=1.6, color="tab:blue", label="Ez(t)")
        axes[0].set_ylabel(r"$E_z\,(\mathrm{V\,m^{-1}})$")
        axes[0].set_title(r"Cathode field")
    handles0, labels0 = axes[0].get_legend_handles_labels()
    if handles0:
        axes[0].legend(handles0, labels0, frameon=False, loc="best")
    axes[0].grid(alpha=0.3)

    if J_t is not None:
        J_cm2 = np.asarray(J_t) * 1e-4
        ln1 = axes[1].plot(t_ns, J_cm2, lw=1.9, color="tab:blue", label=r"$J(t)$")
        axes[1].set_ylabel(r"$J\,(\mathrm{A\,cm^{-2}})$")

        j_finite = J_cm2[np.isfinite(J_cm2)]
        j_pos = j_finite[j_finite > 0.0]
        if j_pos.size > 0:
            j_min = float(np.min(j_pos))
            j_max = float(np.max(j_pos))
            if np.isfinite(j_min) and np.isfinite(j_max) and j_max > j_min:
                axes[1].set_ylim(0.8 * j_min, j_max)

        scale_I_per_J = None
        if area_m2 is not None:
            scale_I_per_J = float(area_m2) * 1e4  # I[A] = J[A/cm^2] * area[cm^2]
        elif I_t is not None and np.size(I_t) == np.size(J_cm2):
            denom = np.maximum(np.abs(J_cm2), 1e-30)
            ratio = np.asarray(I_t, dtype=float) / denom
            ratio = ratio[np.isfinite(ratio)]
            if ratio.size:
                scale_I_per_J = float(np.median(ratio))

        if scale_I_per_J is not None and np.isfinite(scale_I_per_J) and scale_I_per_J > 0.0:
            ax_right = axes[1].secondary_yaxis(
                "right",
                functions=(
                    lambda y: y * scale_I_per_J,
                    lambda y: y / scale_I_per_J,
                ),
            )
            ax_right.set_ylabel(r"$I\,(\mathrm{A})$")

        if t_emit_s is not None and t_emit_s.size > 0:
            t_emit_ns = t_emit_s * 1e9
            t0 = float(np.nanmin(t_ns))
            t1 = float(np.nanmax(t_ns))
            in_window = t_emit_ns[np.isfinite(t_emit_ns)]
            if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
                in_window = in_window[(in_window >= t0) & (in_window <= t1)]

            if in_window.size > 0:
                bins = int(np.clip(np.sqrt(in_window.size), 30, 140))
                p_hist, edges = np.histogram(in_window, bins=bins, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                widths = np.diff(edges)
                p_max = float(np.max(p_hist)) if p_hist.size else 0.0
                j_max = float(np.max(j_pos)) if j_pos.size else 0.0
                if p_max > 0.0 and j_max > 0.0:
                    scale_pdf = j_max / p_max
                    axes[1].bar(
                        centers,
                        p_hist * scale_pdf,
                        width=widths,
                        alpha=0.25,
                        color="tab:blue",
                        edgecolor="none",
                        label=f"PDF {int(in_window.size)} RF-Track macroparticles",
                        zorder=0,
                    )

        from ..emission_models import canonical_emission_model_name, EMISSION_MODEL_PLOT_LABELS

        law = str(thermo_info.get("emission_law", "")).strip()
        try:
            canon_law = canonical_emission_model_name(law)
            title = f"Emission model: {EMISSION_MODEL_PLOT_LABELS[canon_law]}"
        except (KeyError, ValueError):
            canon_law = None
            title = "Emission model"
        axes[1].set_title(title)

        if comparison_models and F_t is not None:
            T_K = thermo_info.get("cathode_T_K", None)
            phi_eV = thermo_info.get("work_function_eV", None)
            if T_K is None or phi_eV is None:
                print("Emission history: comparison_models requested but cathode_T_K/work_function_eV "
                      "not in thermo_info (older saved run) -- skipping overlay.")
            else:
                F_pos = np.maximum(np.asarray(F_t, dtype=float), 0.0)
                for m in comparison_models:
                    try:
                        canon_m = canonical_emission_model_name(m)
                    except ValueError:
                        canon_m = m
                    if canon_law is not None and canon_m == canon_law:
                        continue  # already plotted as the solid production-model curve
                    try:
                        res = evaluate_emission_model(m, F_pos, float(T_K), float(phi_eV))
                    except NotImplementedError:
                        continue
                    color = EMISSION_MODEL_COLORS.get(canon_m)
                    label = EMISSION_MODEL_PLOT_LABELS.get(canon_m, m)
                    ls = "-." if canon_m == "murphygood1956_SchottkyNordheim_integral" else "--"
                    axes[1].plot(t_ns, np.asarray(res.J_Apm2) * 1e-4, ls=ls, lw=1.3, color=color, label=label, alpha=0.85)

        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(handles, labels, frameon=False, loc="upper right")
    axes[1].grid(alpha=0.3)

    if do_components:
        comp_row = 2
        axes[comp_row].plot(t_ns, np.asarray(J_th_t) * 1e-4, lw=1.5, ls="--", label=r"$J_{\mathrm{th}}(t)$")
        axes[comp_row].plot(t_ns, np.asarray(J_fe_t) * 1e-4, lw=1.5, ls=":", label=r"$J_{\mathrm{fe}}(t)$")
        if n_t is not None and n_t.size == t_ns.size:
            axn = axes[comp_row].twinx()
            axn.plot(t_ns, n_t, lw=1.2, color="tab:purple", alpha=0.8, label=r"$n(t)$")
            axn.set_ylabel(r"$n$", color="tab:purple")
            axn.tick_params(axis="y", labelcolor="tab:purple")
        axes[comp_row].set_ylabel(r"$J\,(\mathrm{A\,cm^{-2}})$")
        axes[comp_row].set_title(r"Unified-law components")
        axes[comp_row].legend(frameon=False, loc="upper right")
        axes[comp_row].grid(alpha=0.3)

    axes[-1].set_xlabel(r"$t\,(\mathrm{ns})$")

    if area_m2 is not None and J_t is not None and I_t is not None:
        I_from_J = np.asarray(J_t) * float(area_m2)
        denom = np.maximum(np.abs(I_from_J), 1e-30)
        rel_err = np.nanmax(np.abs(np.asarray(I_t) - I_from_J) / denom)
        if np.isfinite(rel_err) and rel_err > 1e-3:
            print(f"Warning: I(t) and J(t)*area differ by up to {rel_err:.2e}")

    plt.tight_layout()
    plt.show()


def plot_j_vs_n(thermo_info: Dict[str, Any]):
    """Scatter plot of J vs n over time (log-log)."""
    import matplotlib.pyplot as plt

    J_t = _as_1d(thermo_info.get("J_Apm2_t", None))
    n_t = _as_1d(thermo_info.get("n_t", None))
    if J_t is None or n_t is None:
        print("J vs n: missing J_t or n_t.")
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.scatter(n_t, np.asarray(J_t) * 1e-4, s=14, alpha=0.7, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$J\,(\mathrm{A\,cm^{-2}})$")
    ax.set_title(r"Emission map: $J$ vs $n$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def _stable_ylim(values_by_model: Dict[str, np.ndarray], pad_fraction: float = 0.15, min_rel_range: float = 2.0e-3):
    """Robust y-limits from only the finite, stability-flag-clear points across every model, so
    one model's flagged-unstable excursion can't dominate the shared axis and flatten the rest --
    see plot_emission_model_sensitivities' module note on why a single model used to visually
    swamp the others.

    `min_rel_range` floors the y-range at this fraction of the panel's central value: a
    sensitivity that is genuinely (near-)constant across the whole model set (e.g. S_phi/S_T in a
    thermionic-dominated regime, where every model should reduce to the same closed form) can
    still carry residual floating-point-level noise (~1e-5 absolute, ~1e-6 relative here) after
    every other numerical fix in this module -- letting matplotlib's default autoscale zoom in to
    fit *that* would visually amplify genuinely negligible noise into what looks like a dramatic
    feature. This floor keeps the axis from zooming in past the point where remaining variation is
    physically meaningful, without hiding a real, larger excursion (which still expands the range
    normally, since the floor is a *minimum*, not a fixed span).
    """
    finite_vals = [v[np.isfinite(v)] for v in values_by_model.values()]
    finite_vals = [v for v in finite_vals if v.size]
    if not finite_vals:
        return None
    allv = np.concatenate(finite_vals)
    lo, hi = np.percentile(allv, [1.0, 99.0])
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    center = 0.5 * (lo + hi)
    min_range = min_rel_range * max(abs(center), 1e-300)
    if (hi - lo) < min_range:
        lo, hi = center - 0.5 * min_range, center + 0.5 * min_range
    pad = pad_fraction * (hi - lo)
    return (lo - pad, hi + pad)


def plot_emission_model_sensitivities(
    F_Vpm: np.ndarray,
    T_K: float,
    phi_eV: float,
    models: Optional[Sequence[str]] = None,
    F_populated_range: Optional[tuple] = None,
    j_floor_Apm2: float = 0.0,
):
    """2x2 emission-sensitivity figure (implementation guide Sec. 4.3):
    A) J(F) for every enabled model, log scale;
    B) S_F = d ln J / d ln F;
    C) S_T = d ln J / d ln T;
    D) S_phi = -phi * d ln J / d phi.

    Points flagged unstable under finite-difference step-doubling (any of S_F/S_T/S_phi, see
    rf_gun.emission_sensitivity.compute_log_sensitivities) are excluded from each model's *line*
    (so a single bad point can't draw a spike connecting two otherwise-smooth stretches) and from
    the panels' shared y-autoscale (so one model's excursion can't flatten the others onto the
    same axis) -- they're still shown, as small black markers, so the instability itself stays
    visible rather than silently vanishing.
    """
    import matplotlib.pyplot as plt

    from ..emission_models import EMISSION_MODEL_PLOT_LABELS, canonical_emission_model_name
    from ..emission_sensitivity import compute_log_sensitivities

    models = list(models) if models is not None else [
        "RDSchottky", "jensen2014_RDSchottky_MurphyGood_additive",
    ]
    F = np.asarray(F_Vpm, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    sens_by_model = {m: compute_log_sensitivities(m, F, T_K, phi_eV, j_floor_Apm2=j_floor_Apm2) for m in models}

    # Stable-only view of each panel, used both for line-breaking and for shared y-autoscale.
    S_F_stable, S_T_stable, S_phi_stable = {}, {}, {}

    for m in models:
        canon = canonical_emission_model_name(m)
        color = EMISSION_MODEL_COLORS.get(canon)
        label = EMISSION_MODEL_PLOT_LABELS.get(canon, m)
        s = sens_by_model[m]
        J = s["J_Apm2"]
        unstable_any = s["unstable_any"]
        mask = np.isfinite(J) & (J > 0.0)

        def _break_at_unstable(arr):
            out = np.array(arr, dtype=float, copy=True)
            out[unstable_any] = np.nan
            return out

        S_F_stable[m] = _break_at_unstable(s["S_F"])
        S_T_stable[m] = _break_at_unstable(s["S_T"])
        S_phi_stable[m] = _break_at_unstable(s["S_phi"])

        style = dict(color=color, lw=1.8, label=label) if color else dict(lw=1.8, label=label)
        axA.plot(F[mask], J[mask], **style)
        axB.plot(F[mask], S_F_stable[m][mask], **style)
        axC.plot(F[mask], S_T_stable[m][mask], **style)
        axD.plot(F[mask], S_phi_stable[m][mask], **style)

        flagged = mask & unstable_any
        if np.any(flagged):
            marker_kwargs = dict(marker="x", color=color or "k", ms=5, zorder=5, ls="none")
            axB.plot(F[flagged & s["unstable_F"]], s["S_F"][flagged & s["unstable_F"]], **marker_kwargs)
            axC.plot(F[flagged & s["unstable_T"]], s["S_T"][flagged & s["unstable_T"]], **marker_kwargs)
            axD.plot(F[flagged & s["unstable_phi"]], s["S_phi"][flagged & s["unstable_phi"]], **marker_kwargs)

    for ax, values_by_model in ((axB, S_F_stable), (axC, S_T_stable), (axD, S_phi_stable)):
        ylim = _stable_ylim(values_by_model)
        if ylim is not None:
            ax.set_ylim(*ylim)

    if F_populated_range is not None:
        F_min, F_peak, F_max = F_populated_range
        for j, ax in enumerate((axA, axB, axC, axD)):
            if np.isfinite(F_min) and np.isfinite(F_max):
                # Thin dotted boundary lines rather than a solid shaded band, which visually
                # competed with the curves ("grey sharding") without adding information the
                # boundary lines alone don't already convey.
                label = "Populated field range" if j == 0 else None
                ax.axvline(F_min, color="0.6", ls=":", lw=1.1, alpha=0.8, label=label)
                ax.axvline(F_max, color="0.6", ls=":", lw=1.1, alpha=0.8)
            if np.isfinite(F_peak):
                ax.axvline(F_peak, color="k", ls="--", lw=1.0, alpha=0.6, label="Peak field" if j == 0 else None)

    axA.set_yscale("log")
    axA.set_xscale("log")
    axA.set_ylabel(r"$J\,(\mathrm{A\,m^{-2}})$")
    axA.set_title("Current density")
    axA.legend(frameon=False, fontsize=8, loc="best")

    for ax, ylabel, title in (
        (axB, r"$S_F=\partial\ln J/\partial\ln F$", "Field sensitivity"),
        (axC, r"$S_T=\partial\ln J/\partial\ln T$", "Temperature sensitivity"),
        (axD, r"$S_\Phi=-\Phi\,\partial\ln J/\partial\Phi$", "Work-function sensitivity"),
    ):
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    for ax in (axA, axB, axC, axD):
        ax.set_xlabel(r"$F\,(\mathrm{V\,m^{-1}})$")
        ax.grid(alpha=0.3)

    fig.suptitle(rf"Emission-model sensitivities: $T={T_K:.0f}\,$K, $\Phi={phi_eV:.2f}\,$eV")
    plt.show()
    return sens_by_model
