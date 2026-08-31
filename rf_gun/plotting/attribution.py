"""Frozen-source physics attribution figure (see rf_gun.frozen_source_attribution)."""
from __future__ import annotations

from typing import Any

import numpy as np


def plot_frozen_source_attribution(result: Any):
    """2x2 bar-chart comparison of exit-beam diagnostics across
    rf_gun.frozen_source_attribution.run_frozen_source_attribution's cases: transmission, mean
    longitudinal momentum, momentum spread, and transverse beam size at the exit screen. Since
    every case tracks the identical fixed-seed source (see that module's docstring), any
    difference between bars is attributable purely to the transport physics that case's label
    names, not to a different macroparticle realization.

    Also prints (does not plot -- there's nothing to compare visually) each case's total emitted
    charge as a consistency check: these should all match to floating-point precision, since none
    of space charge / mirror charges / beam loading affect the emission current itself. A
    mismatch would mean the frozen-source assumption was violated somewhere upstream and should be
    investigated before trusting the bars.
    """
    import matplotlib.pyplot as plt

    labels = list(result.labels)
    n = len(labels)
    x = np.arange(n)

    def _get(key, default=np.nan):
        return [float(s.get(key, default)) if s.get(key, default) is not None else np.nan for s in result.exit_summaries]

    transmission_pct = [100.0 * v if np.isfinite(v) else np.nan for v in _get("transmission_from_initial")]
    mean_pz = _get("mean_pz_MeV_c")
    sigma_pz = _get("sigma_pz_MeV_c")
    sigma_x_um = [1e3 * v if np.isfinite(v) else np.nan for v in _get("sigma_x_mm")]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    bar_kwargs = dict(color="tab:blue", alpha=0.85, edgecolor="black", linewidth=0.5)
    axA.bar(x, transmission_pct, **bar_kwargs)
    axA.set_ylabel("Transmission from initial (%)")
    axA.set_title("Exit transmission")

    axB.bar(x, mean_pz, **bar_kwargs)
    axB.set_ylabel(r"$\langle p_z\rangle\,(\mathrm{MeV}/c)$")
    axB.set_title("Mean longitudinal momentum")

    axC.bar(x, sigma_pz, **bar_kwargs)
    axC.set_ylabel(r"$\sigma_{p_z}\,(\mathrm{MeV}/c)$")
    axC.set_title("Momentum spread")

    axD.bar(x, sigma_x_um, **bar_kwargs)
    axD.set_ylabel(r"$\sigma_x\,(\mathrm{\mu m})$")
    axD.set_title("Transverse beam size at exit")

    for ax in (axA, axB, axC, axD):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Frozen-source physics attribution (same source, varying transport physics)")
    plt.show()

    Q_pC = np.asarray(result.emitted_charge_C) * 1e12
    Q_spread_pC = float(np.nanmax(Q_pC) - np.nanmin(Q_pC)) if Q_pC.size else 0.0
    print(f"Emitted charge by case (pC): {dict(zip(labels, np.round(Q_pC, 6)))}")
    if Q_spread_pC > 1e-6 * max(1.0, float(np.nanmax(np.abs(Q_pC)))):
        print(
            f"Warning: emitted charge differs across cases by {Q_spread_pC:.3e} pC -- the "
            "frozen-source assumption (identical B0 across cases) may not actually hold; "
            "investigate before trusting the bars above."
        )
