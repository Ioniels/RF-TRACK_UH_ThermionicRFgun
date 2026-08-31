"""Temperature-dependent effective work function models for a LaB6 <100> thermionic cathode,
transcribed from manual_references/LaB6_100_work_function_models.md (a literature synthesis
covering Swanson & Dickinson 1976, Nishitani et al. 1980, Swanson/Gesley/Davis 1981, Gesley &
Swanson 1984, Bulyga & Solonovich 1989, and Liu et al. 2017).

Central distinction from that note (do not conflate the two):
    phi_surf = E_vac - E_F           microscopic/true surface work function (UPS/FERP/DFT)
    phi_eff(T) = k_B T ln(A_R T^2/J_0(T))   effective thermionic work function, tied to the
                                             Richardson constant A_R used to extract it

These models all produce phi_eff(T) [eV] for use directly in the Richardson-Dushman-Schottky (or
any other) emission law here -- they are zero-field values (the source thermionic datasets were
extrapolated to zero field), so do not double-apply Schottky lowering into phi_eff itself; let the
emission model's own Delta_phi(F) term handle that separately (see the note's Sec. 4).

None of the three models below is a single author's published closed-form law for LaB6(100) over
the full 1000-2000 K range (the note is explicit about this for M1/M2); names are therefore
physics-descriptive rather than single-author-attributed, except where a model is anchored
directly to one dataset (constant_phi_eff_liu2017).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

#: Liu et al. 2017 (Vacuum 143, 245) single-crystal LaB6(100) thermionic effective work function,
#: the note's recommended baseline anchor; A_R=120.4 A/cm^2/K^2 is the (material-specific, measured)
#: Richardson constant used to derive it, and must be paired with phi_eff, not phi_DFT, when using
#: RDSchottky/jensen2014_RDSchottky_MurphyGood_additive. This is deliberately a separate
#: constant from rf_gun.constants.A_RICH (the generic theoretical free-electron Richardson
#: constant, ~120.17 A/cm^2/K^2): the two agree to ~0.2% here, which is coincidental (any real
#: material's measured A_R can differ from the free-electron value by far more than this), not a
#: bug to "fix" by merging them -- but evaluate_emission_model()'s A_R_Apm2K2 still defaults to
#: A_RICH regardless of which work_function_temperature_model is selected. A caller pairing
#: constant_phi_eff/linear_tcwf/piecewise_surface_evolution (all anchored to this same Liu 2017
#: point) with a materially different downstream use of A_R should pass A_R_Apm2K2=
#: LIU_2017_A_R_APM2K2 explicitly for full internal consistency; the 0.2% default mismatch is
#: negligible next to this model's other uncertainties (e.g. the +/-0.10 eV work-function-model
#: uncertainty in WORK_FUNCTION_MODEL_UNCERTAINTY_EV, whose effect on J is exponential, not linear).
LIU_2017_PHI_EFF_EV = 2.66
LIU_2017_A_R_APM2K2 = 120.4e4  # 120.4 A/cm^2/K^2 -> A/m^2/K^2

#: Bulyga & Solonovich 1989's temperature-coefficient-of-work-function (TCWF) scale, used in
#: Model 1 to give the constant anchor a smooth T-dependence.
BULYGA_ALPHA_PHI_EV_PER_K = 1.8e-4


def phi_eff_constant(T_K, phi_ref_eV: float = LIU_2017_PHI_EFF_EV):
    """Model 0: temperature-independent effective work function, anchored to the Liu et al. 2017
    LaB6(100) thermionic measurement (note Sec. 2, Model 0) -- the note's preferred baseline.

    `T_K` may be a scalar or an array (e.g. a per-cathode-cell temperature map T(x,y)); the return
    shape follows `T_K` so callers never need to special-case scalar vs. spatially-resolved use
    (a plain Python float back for scalar input, an ndarray back for array input).
    """
    T = np.asarray(T_K, dtype=float)
    result = T * 0.0 + float(phi_ref_eV)
    return result if T.ndim > 0 else float(result)


def phi_eff_linear_tcwf(
    T_K,
    phi_ref_eV: float = LIU_2017_PHI_EFF_EV,
    T_ref_K: float = 1773.0,
    alpha_eV_per_K: float = BULYGA_ALPHA_PHI_EV_PER_K,
):
    """Model 1: smooth linear temperature-coefficient-of-work-function (TCWF) model (note Sec. 2,
    Model 1) -- Liu 2017's effective-work-function anchor extended with Bulyga & Solonovich's TCWF
    slope. Not a single published formula; a constructed smooth model for the 1000-2000 K range.
    `T_K` may be scalar or array (see phi_eff_constant)."""
    T = np.asarray(T_K, dtype=float)
    result = float(phi_ref_eV) + float(alpha_eV_per_K) * (T - float(T_ref_K))
    return result if T.ndim > 0 else float(result)


def phi_eff_piecewise_surface_evolution(T_K):
    """Model 2: piecewise surface-evolution/empirical model (note Sec. 2, Model 2), blending the
    Swanson et al. 1981 prolonged-anneal low-T branch, the Bulyga & Solonovich ~1573 K
    stoichiometry-break transition, and the Liu et al. 2017 high-T branch. A phenomenological
    sensitivity model for a cathode whose effective emitting surface evolves with temperature, not
    an intrinsic band-structure T-dependence -- the note's own explicit caveat.
    `T_K` may be scalar or array (see phi_eff_constant)."""
    T = np.asarray(T_K, dtype=float)
    phi_1573 = 2.52 - 2.0e-4 * (1573.0 - 1600.0)
    phi_1673 = 2.668 + 6.11e-4 * (1673.0 - 1773.0)
    result = np.select(
        [T <= 1573.0, T < 1673.0],
        [
            2.52 - 2.0e-4 * (T - 1600.0),
            phi_1573 + (phi_1673 - phi_1573) / 100.0 * (T - 1573.0),
        ],
        default=2.668 + 6.11e-4 * (T - 1773.0),
    )
    return result if T.ndim > 0 else float(result)


WORK_FUNCTION_MODEL_NAMES = ("constant_phi_eff", "linear_tcwf", "piecewise_surface_evolution")

#: Recommended modeling uncertainty (1-sigma-ish band), per the note's own recommendation
#: throughout Sec. 2 -- not applied automatically, but exposed for sensitivity scans.
WORK_FUNCTION_MODEL_UNCERTAINTY_EV = {
    "constant_phi_eff": 0.10,
    "linear_tcwf": 0.10,
    "piecewise_surface_evolution": 0.10,
}


def evaluate_work_function_eV(model: str, T_K: float, **params: Any) -> float:
    """phi_eff(T) [eV] for `model` in WORK_FUNCTION_MODEL_NAMES; `params` overrides that model's
    default constants (e.g. phi_ref_eV, alpha_eV_per_K for linear_tcwf)."""
    if model == "constant_phi_eff":
        return phi_eff_constant(T_K, **params)
    if model == "linear_tcwf":
        return phi_eff_linear_tcwf(T_K, **params)
    if model == "piecewise_surface_evolution":
        if params:
            raise ValueError("piecewise_surface_evolution takes no override parameters")
        return phi_eff_piecewise_surface_evolution(T_K)
    raise ValueError(f"Unknown work_function_temperature_model {model!r}; expected one of {WORK_FUNCTION_MODEL_NAMES}")
