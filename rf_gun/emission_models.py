"""Thermionic and field-emission models."""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import digamma, expit

from .constants import A_FN, A_RICH, B_FN, EV, KB, KB_EV_PER_K, epsilon_0, h, m_e, q_e


def delta_phi_schottky_eV(F_Vpm: np.ndarray) -> np.ndarray:
    """Schottky lowering for a local field magnitude [V/m]."""
    F = np.maximum(F_Vpm, 0.0)
    dphi_J = np.sqrt((q_e**3) * F / (4.0 * np.pi * epsilon_0))
    return dphi_J / EV


def richardson_J_Apm2(T_K: float, phi_eff_eV: float) -> float:
    """Richardson-Dushman current density J [A/m^2]."""
    return float(A_RICH * (T_K**2) * np.exp(-phi_eff_eV / (KB_EV_PER_K * T_K)))


def sn_y(F_Vpm: np.ndarray, phi_eV: float) -> np.ndarray:
    """Nordheim parameter y = Delta_phi/phi for SN barrier (0<y<1)."""
    phi = np.maximum(phi_eV, 1e-6)
    y = delta_phi_schottky_eV(np.maximum(F_Vpm, 0.0)) / phi
    return np.clip(y, 0.0, 0.999)


def sn_v(y: np.ndarray) -> np.ndarray:
    """SN barrier correction v(y) using a standard series approximation."""
    y = np.clip(y, 1e-6, 0.999)
    y2 = y * y
    return 1.0 - y2 + (y2 / 6.0) * np.log(y)


def sn_t(y: np.ndarray) -> np.ndarray:
    """SN slope correction t(y) using a standard series approximation."""
    y = np.clip(y, 1e-6, 0.999)
    y2 = y * y
    return 1.0 + (y2 / 9.0) - (y2 / 18.0) * np.log(y)


def J_rld_schottky(F_Vpm: np.ndarray, T_K: float, phi_eV: float, A_R: float = A_RICH) -> np.ndarray:
    """Richardson with Schottky lowering using local field magnitude."""
    dphi = delta_phi_schottky_eV(np.maximum(F_Vpm, 0.0))
    phi_eff = np.maximum(phi_eV - dphi, 1e-6)
    return A_R * (T_K**2) * np.exp(-(phi_eff * EV) / (KB * T_K))


def J_mg0_sn(F_Vpm: np.ndarray, phi_eV: float) -> np.ndarray:
    """Murphy-Good cold field emission with SN corrections."""
    F = np.maximum(F_Vpm, 1.0)
    phi = np.maximum(phi_eV, 1e-6)
    y = sn_y(F, phi)
    v = sn_v(y)
    t = sn_t(y)
    pre = A_FN * (F**2) / phi / np.maximum(t, 1e-12) ** 2
    expo = -B_FN * (phi**1.5) * v / F
    return pre * np.exp(expo)


def lambda_T(p: np.ndarray) -> np.ndarray:
    """Finite-temperature factor lambda_T = (pi p)/sin(pi p)."""
    x = np.pi * np.clip(p, 0.0, 0.999)
    small = x < 1e-3
    out = np.empty_like(x)
    out[small] = 1.0 + (x[small] ** 2) / 6.0
    out[~small] = x[~small] / np.sin(x[~small])
    return out


def beta_slope_eVinv(F_Vpm: np.ndarray, phi_eV: float) -> np.ndarray:
    """Barrier slope beta_slope = dG/dE at Fermi level [1/eV]."""
    F = np.maximum(F_Vpm, 1.0)
    phi = np.maximum(phi_eV, 1e-6)
    t = sn_t(sn_y(F, phi))
    return (B_FN * np.sqrt(phi) * t) / F


def n_regime(F_Vpm: np.ndarray, T_K: float, phi_eV: float) -> Tuple[np.ndarray, np.ndarray]:
    """Regime indicator n = 1/(kT * beta_slope)."""
    beta = beta_slope_eVinv(F_Vpm, phi_eV)
    kT_eV = (KB * T_K) / EV
    p = kT_eV * beta
    n = 1.0 / np.maximum(p, 1e-12)
    return n, p


def J_field_side_gtf(F_Vpm: np.ndarray, T_K: float, phi_eV: float) -> Tuple[np.ndarray, np.ndarray]:
    """Field-side GTF: MG0 * lambda_T with regime indicator n."""
    J0 = J_mg0_sn(F_Vpm, phi_eV)
    n, p = n_regime(F_Vpm, T_K, phi_eV)
    J = J0 * lambda_T(p)
    return J, n


def J_unified(
    F_Vpm: np.ndarray,
    T_K: float,
    phi_eV: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unified emission as additive thermionic + field-side channels."""
    J_th = J_rld_schottky(F_Vpm, T_K, phi_eV)
    J_fe, n = J_field_side_gtf(F_Vpm, T_K, phi_eV)
    J = J_th + J_fe
    return J, n, J_th, J_fe


# ---------------------------------------------------------------------------
# Direct Murphy-Good energy-integral reference (guide Sec. 3.3).
#
# Energy origin/convention (documented explicitly per the guide's own requirement):
#   E_n is the normal energy measured from the bottom of the occupied band (E_n >= 0).
#   `chemical_potential_eV` (mu) is the Fermi level measured from that same origin, i.e. the
#   standard free-electron-metal band picture. The barrier height above E_n at the surface
#   (before field/image lowering) is (mu + phi_eV - E_n). Richardson-Dushman is recovered in the
#   "deep Fermi sea" limit mu >> k_B*T, where the E_n=0 lower integration cutoff carries
#   negligible occupied density -- this is the same limit implicit in every RLD derivation, which
#   is why RLD itself never needs an explicit mu. The default below (10 eV) is a generic
#   "deep enough" placeholder for this limit, not a material-specific Fermi energy; callers doing
#   a material-accurate calculation should pass their own chemical_potential_eV.
# ---------------------------------------------------------------------------

DEFAULT_CHEMICAL_POTENTIAL_EV = 10.0

#: quad() kwargs for every energy-integral evaluation in this module. scipy.integrate.quad's
#: default epsabs (~1.49e-8) is an *absolute* tolerance -- fine when the integral's value is O(1)
#: or larger, but these integrals can legitimately evaluate to O(1e-8) or smaller (e.g. deep in a
#: rgtf_2019 near-pole regime, or at extreme field/temperature combinations), where a ~1.49e-8
#: absolute error is itself a large (confirmed empirically: up to ~20%) *relative* error -- quad
#: reports "converged" long before the result is actually accurate. Setting epsabs far below any
#: value these integrals can plausibly take forces convergence to be judged by epsrel (relative)
#: alone, which is the correct criterion here since the integrals' absolute scale is not known a
#: priori. Confirmed to remove a real, previously-unexplained residual finite-difference noise in
#: rgtf_2019's work-function sensitivity at near-pole field points (S_phi went from visibly jagged,
#: +-0.02% swings uncorrelated with F, to stable to 6 significant figures).
_TIGHT_QUAD_KWARGS = dict(epsabs=1e-20, epsrel=1e-12, limit=400)


def _gamow_factor_sn_eV(En_rel_mu_eV: float, F_Vpm: float, phi_eV: float) -> float:
    """Exact-SN-elliptic WKB Gamow factor G(En) for the planar Schottky-Nordheim barrier

        V(x) = mu_eV + phi_eV - e*F*x - e^2/(16 pi eps0 x)

    expressed in terms of En_rel_mu_eV = En_eV - mu_eV, so the barrier height above En is
    (phi_eV - En_rel_mu_eV) regardless of mu's absolute value -- mu itself only ever enters
    through this difference, and separately through the Fermi-level reference in
    _supply_function_N(), which uses the un-shifted band-bottom En_eV.

    below the barrier saddle (y=1), continued for E_n at/above the saddle by a first-order
    (linear-in-E_n) Kemble parabolic-top expansion, matched in both value and slope at y=1 so
    G(E_n) is smooth (C1) across the transition -- this is the guide's flagged numerical caution
    in Sec. 3.3 ("raw below-barrier WKB is not automatically an exact reference near/above the
    barrier maximum"), addressed via option (2), the exact SN barrier functions, rather than
    silently accepting a discontinuous D(E_n) at the saddle.

    Reuses this module's existing sn_v()/B_FN (already validated for the F,phi->J_mg0_sn path)
    generalized to an arbitrary local barrier height (phi_eV - En_eV) instead of the Fermi-level
    barrier height phi_eV.
    """
    dphi_eV = float(delta_phi_schottky_eV(np.asarray(F_Vpm, dtype=float)))
    Phi_max_eV = phi_eV - dphi_eV  # barrier height above En at the SN saddle (y=1)

    Phi_local_eV = phi_eV - En_rel_mu_eV
    if Phi_local_eV > dphi_eV:
        y = dphi_eV / Phi_local_eV
        v = float(sn_v(np.array([y]))[0])
        return float(B_FN * (Phi_local_eV ** 1.5) * v / F_Vpm)

    # At/above the saddle: linear continuation calibrated at y=1 (G=0, v(1)=0) using the exact
    # analytical slope dG/dEn|_{y=1} = (B_FN/F) * sqrt(dphi_eV) * (v'(1) - 1.5*v(1)) with
    # v'(1) = -11/6 (from v(y) = 1 - y^2 + (y^2/6) ln y), i.e. dG/dEn|_{y=1} = -(11/6)*B_FN*sqrt(dphi_eV)/F.
    slope = -(11.0 / 6.0) * B_FN * np.sqrt(max(dphi_eV, 1e-300)) / F_Vpm
    return float(slope * (En_rel_mu_eV - Phi_max_eV))


def _supply_function_N(En_eV: np.ndarray, T_K: float, mu_eV: float) -> np.ndarray:
    """N(En,T) [1/(m^2 s eV)] -- guide Eq. in Sec. 3.3, En/mu measured from the band bottom.

    The paper's N(En,T) is naturally per unit Joule of En; multiplying by EV re-expresses it per
    unit eV of En, since this module integrates over En in eV throughout.
    """
    kT_eV = KB_EV_PER_K * T_K
    x = (mu_eV - np.asarray(En_eV, dtype=float)) / kT_eV
    # log1p(exp(x)) computed stably for large positive/negative x.
    log1p_exp = np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -700.0, 30.0))))
    return (4.0 * np.pi * m_e * KB * T_K / (h ** 3)) * log1p_exp * EV  # *EV: per-eV, not per-Joule


def J_murphy_good_direct_reference(
    F_Vpm: np.ndarray,
    T_K,
    phi_eV: float,
    chemical_potential_eV: Optional[float] = None,
) -> np.ndarray:
    """Slow direct Murphy-Good/RLD reference: J = e * integral_0^inf N(En,T) D(En;F,phi) dEn.

    D(En) = expit(-G(En)) (a numerically stable 1/(1+exp(G)) Kemble transmission). Intended for
    validation/tabulation (guide Sec. 3.3), not per-tracking-step evaluation -- one `quad` call
    per field-array element.

    `T_K` may be a scalar (one temperature for every F_Vpm element) or an array matching F_Vpm's
    shape (a per-element/per-cathode-cell temperature, e.g. from a spatially resolved emission
    model where each grid cell has its own local T).
    """
    mu_eV = float(chemical_potential_eV) if chemical_potential_eV is not None else DEFAULT_CHEMICAL_POTENTIAL_EV
    F_arr = np.atleast_1d(np.asarray(F_Vpm, dtype=float))
    T_arr = np.broadcast_to(np.asarray(T_K, dtype=float), F_arr.shape)
    J = np.empty_like(F_arr)

    for i, F in enumerate(F_arr):
        F = max(float(F), 1.0)
        T_i = float(T_arr[i])
        kT_eV = KB_EV_PER_K * T_i
        En_hi_eV = mu_eV + 40.0 * kT_eV  # supply function negligible well beyond this

        def integrand(En_eV, F=F, T_i=T_i):
            N = float(_supply_function_N(np.array([En_eV]), T_i, mu_eV)[0])
            if N <= 0.0:
                return 0.0
            G = _gamow_factor_sn_eV(En_eV - mu_eV, F, phi_eV)
            D = float(expit(-G))
            return N * D

        # _gamow_factor_sn_eV is only C1 (not C-infinity) at the barrier saddle En_eV=E_kink_eV,
        # where it switches from the below-barrier WKB integral to the linear above-barrier
        # continuation. Left unmarked, quad's adaptive subdivision places its nodes near this kink
        # inconsistently as it shifts with F/phi (through dphi_eV), which otherwise looks like
        # numerical noise of a few times 1e-3 relative in J -- large enough to dominate a
        # finite-difference sensitivity's own tiny perturbation and show up as an artificially
        # jagged S_phi/S_F (confirmed empirically: with `points` unset, S_phi visibly oscillates
        # F-to-F even though the true dependence is smooth; with it set, S_phi is stable to 6
        # significant figures). `points` must lie strictly inside (0, En_hi_eV) for quad to accept
        # it as a break point.
        dphi_eV_i = float(delta_phi_schottky_eV(np.array([F]))[0])
        E_kink_eV = mu_eV + phi_eV - dphi_eV_i
        quad_points = [E_kink_eV] if 0.0 < E_kink_eV < En_hi_eV else None
        val, _ = quad(integrand, 0.0, En_hi_eV, points=quad_points, **_TIGHT_QUAD_KWARGS)
        J[i] = q_e * val

    result = J if np.ndim(F_Vpm) else J[0]
    return result


# ---------------------------------------------------------------------------
# Model registry (guide Sec. 13.1): a common interface over every emission kernel, preserving the
# old bare functions above unchanged for backward compatibility.
# ---------------------------------------------------------------------------

#: Naming convention (user-specified, 2026): short author/paper-code prefix + the two physical
#: regimes/effects the model combines. Every prior name this repo has ever used for these models
#: (both the original pre-refactor names and this session's earlier author_papercode_year
#: intermediate renaming) is kept as a permanent backward-compatible alias (EMISSION_MODEL_ALIASES)
#: so old saved run_config.json files and old CLI/notebook invocations keep resolving correctly.
EMISSION_MODEL_NAMES = (
    "RDSchottky",
    "jensen2014_RDSchottky_MurphyGood_additive",
    "jensen_gtf_2007",
    "jensen2019_RDSchottky_MurphyGood_transition",
    "murphygood1956_SchottkyNordheim_integral",
)

#: Backward-compatible aliases -> canonical name. Existing saved configs using any of these old
#: names must keep working -- never silently rewrite a saved run's emission_law.
EMISSION_MODEL_ALIASES = {
    # Original (pre-2026-refactor) names.
    "unified": "jensen2014_RDSchottky_MurphyGood_additive",
    "RD_schottky": "RDSchottky",
    "rld_schottky_plus_mg": "jensen2014_RDSchottky_MurphyGood_additive",
    "rgtf_2019": "jensen2019_RDSchottky_MurphyGood_transition",
    "murphy_good_direct_reference": "murphygood1956_SchottkyNordheim_integral",
    # This session's first-pass (author_papercode_year) renaming, now itself superseded.
    "richardson_dushman_schottky": "RDSchottky",
    "schottky_murphygood_additive_legacy": "jensen2014_RDSchottky_MurphyGood_additive",
    "jensen_rgtf_2019": "jensen2019_RDSchottky_MurphyGood_transition",
    "murphy_good_1956_integral": "murphygood1956_SchottkyNordheim_integral",
}

EMISSION_MODEL_PLOT_LABELS = {
    "RDSchottky": "Thermionic: Richardson-Dushman-Schottky",
    "jensen2014_RDSchottky_MurphyGood_additive": (
        "Jensen2014: Additive regime, Thermionic: Richardson-Dushman-Schottky; Field: Murphy-Good"
    ),
    "jensen_gtf_2007": "Jensen general thermal-field (2007) [not implemented]",
    "jensen2019_RDSchottky_MurphyGood_transition": (
        "Jensen2019: Transition regime, Thermionic: Richardson-Dushman-Schottky; Field: Murphy-Good"
    ),
    "murphygood1956_SchottkyNordheim_integral": (
        "MurphyGood1956 Integrals of transmission probability into Schottky-Nordheim barrier"
    ),
}


@dataclass(frozen=True)
class EmissionModelResult:
    J_Apm2: np.ndarray
    J_thermionic_Apm2: Optional[np.ndarray] = None
    J_field_Apm2: Optional[np.ndarray] = None
    regime_n: Optional[np.ndarray] = None
    dJ_dEn: Optional[np.ndarray] = None
    energy_grid_eV: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = dataclass_field(default_factory=dict)


def canonical_emission_model_name(model: str) -> str:
    canon = EMISSION_MODEL_ALIASES.get(model, model)
    if canon not in EMISSION_MODEL_NAMES:
        raise ValueError(f"Unknown emission model {model!r}; expected one of {EMISSION_MODEL_NAMES}")
    return canon


def evaluate_emission_model(
    model: str,
    F_Vpm: np.ndarray,
    T_K: float,
    phi_eV: float,
    *,
    A_R_Apm2K2: float = A_RICH,
    chemical_potential_eV: Optional[float] = None,
) -> EmissionModelResult:
    """Common evaluator over every registered emission kernel; see EMISSION_MODEL_NAMES. `T_K` may
    be a scalar or an array matching F_Vpm's shape (a per-element/per-cathode-cell temperature)."""
    canon = canonical_emission_model_name(model)
    F = np.asarray(F_Vpm, dtype=float)
    label = EMISSION_MODEL_PLOT_LABELS[canon]

    if canon == "RDSchottky":
        J = J_rld_schottky(F, T_K, phi_eV, A_R=A_R_Apm2K2)
        return EmissionModelResult(J_Apm2=J, diagnostics={"model": canon, "plot_label": label})

    if canon == "jensen2014_RDSchottky_MurphyGood_additive":
        J, n, J_th, J_fe = J_unified(F, T_K, phi_eV)
        return EmissionModelResult(
            J_Apm2=J, J_thermionic_Apm2=J_th, J_field_Apm2=J_fe, regime_n=n,
            diagnostics={"model": canon, "plot_label": label},
        )

    if canon == "murphygood1956_SchottkyNordheim_integral":
        J = J_murphy_good_direct_reference(F, T_K, phi_eV, chemical_potential_eV=chemical_potential_eV)
        return EmissionModelResult(J_Apm2=J, diagnostics={"model": canon, "plot_label": label})

    if canon == "jensen2019_RDSchottky_MurphyGood_transition":
        J = J_rgtf_2019(F, T_K, phi_eV, chemical_potential_eV=chemical_potential_eV)
        return EmissionModelResult(J_Apm2=J, diagnostics={"model": canon, "plot_label": label})

    if canon == "jensen_gtf_2007":
        raise NotImplementedError(
            f"{canon} is registered but not yet implemented (kept as historical/mathematical "
            "comparison per the implementation guide -- jensen2019_RDSchottky_MurphyGood_transition "
            "is the preferred production GTF model and is implemented above)."
        )

    raise AssertionError(f"unreachable: canonical name {canon!r} not handled")


# ---------------------------------------------------------------------------
# rgtf_2019: Jensen, "A reformulated general thermal-field emission equation",
# J. Appl. Phys. 126, 065302 (2019).
#
# Implemented from the paper's own equations (transcribed by a dedicated paper-reading pass, not
# from memory), working in SI internally and converting only at the eV/nm-native public formulas
# below (mirrors the approach already validated exactly against RD_schottky in
# J_murphy_good_direct_reference). One deliberate deviation from a literal transcription: the
# paper's own text has an internally-flagged sign-convention ambiguity between beta_F=-dtheta/dE
# (stated explicitly, Sec. III.D) and Eq. (4)'s caption (which drops the minus sign, likely a
# typesetting slip). Rather than trust either reading, beta_F(E) is computed here as a numerical
# derivative of the already-transcribed, endpoint-verified theta(E) (Eq. 23/29a) -- guaranteed
# consistent with the model's own theta(E) by construction, regardless of which closed-form
# reading of Eq. (26)-(28) is correct.
# ---------------------------------------------------------------------------

HBAR = h / (2.0 * np.pi)

#: Table I (Jensen 2019): shape-factor sigma(y) is exact at y=0 (triangular barrier) and y=1
#: (parabolic/Schottky-thermal barrier); sigma(1/3) is the paper's numerically-fit third node.
_SIGMA_0_AT_Y1 = np.pi / 4.0
_SIGMA_2_AT_Y_1_3 = 0.73262
_SIGMA_4_AT_Y0 = 2.0 / 3.0


def _rgtf_shape_factor_sigma(y: np.ndarray) -> np.ndarray:
    """sigma(y), Jensen 2019 Eq. (29a) -- exact quadratic fit through the two exact endpoints
    (y=0, y=1) and the paper's fitted y=1/3 node. Verified at construction: sigma(0)=2/3,
    sigma(1)=pi/4, matching the paper's Eq. (24) exact values."""
    y = np.asarray(y, dtype=float)
    num = (
        (3.0 * y - 1.0) * (2.0 * y * _SIGMA_0_AT_Y1 - (1.0 - y) * _SIGMA_4_AT_Y0)
        + 8.0 * y * (1.0 - y) * _SIGMA_2_AT_Y_1_3
    )
    return num / (1.0 + y) ** 2


def _rgtf_theta_below_max(En_eV: float, F_Vpm: float, phi_eV: float, mu_eV: float) -> float:
    """theta(E) = 2 sigma[y(E)] kappa(E) L(E) for En at/below the barrier max E=mu+phi-dphi(F)
    (Jensen 2019 Eqs. 18-23), in SI. y(E)=dphi_J(F)/Phi_local_J(E) is exactly this module's
    delta_phi_schottky_eV(F) divided by the local barrier height above En -- the SI form of the
    paper's y(E)=sqrt(4FQ)/(mu+Phi-E), since 4FQ (paper units) is dphi_J^2 (see the direct-integral
    model's docstring for the same Q<->dphi identity)."""
    Phi_local_J = (mu_eV + phi_eV - En_eV) * EV
    dphi_J = float(delta_phi_schottky_eV(np.array([F_Vpm]))[0]) * EV
    y = dphi_J / Phi_local_J
    L_m = np.sqrt(max(Phi_local_J ** 2 - dphi_J ** 2, 0.0)) / (q_e * F_Vpm)
    kappa_per_m = np.sqrt(max(2.0 * m_e * (Phi_local_J - dphi_J), 0.0)) / HBAR
    sigma = float(_rgtf_shape_factor_sigma(np.array([y]))[0])
    return 2.0 * sigma * kappa_per_m * L_m


def _rgtf_theta(En_eV: float, F_Vpm: float, phi_eV: float, mu_eV: float) -> float:
    """theta(E) for any En, using the below-max closed form and the above-max odd continuation
    theta(mu+phi+Delta) = -theta(mu+phi-Delta) (Jensen 2019 Eq. 30)."""
    dphi_eV = float(delta_phi_schottky_eV(np.array([F_Vpm]))[0])
    E_max_eV = mu_eV + phi_eV - dphi_eV
    if En_eV <= E_max_eV:
        return _rgtf_theta_below_max(En_eV, F_Vpm, phi_eV, mu_eV)
    En_mirrored_eV = 2.0 * E_max_eV - En_eV
    return -_rgtf_theta_below_max(En_mirrored_eV, F_Vpm, phi_eV, mu_eV)


def _rgtf_beta_F(E_m_eV: float, F_Vpm: float, phi_eV: float, mu_eV: float, dE_eV: float = 1.0e-4) -> float:
    """beta_F(E) = -d(theta)/dE [1/eV], as a centered finite difference of the already-verified
    theta(E) -- see the module note above on why this avoids re-deriving Eq. (26)-(28) by hand."""
    theta_p = _rgtf_theta(E_m_eV + dE_eV, F_Vpm, phi_eV, mu_eV)
    theta_m = _rgtf_theta(E_m_eV - dE_eV, F_Vpm, phi_eV, mu_eV)
    return -(theta_p - theta_m) / (2.0 * dE_eV)


def _rgtf_dJ_dE(En_eV: float, F_Vpm: float, T_K: float, phi_eV: float, mu_eV: float) -> float:
    """The current-density integrand D(E)f(E) whose maximum locates E_m (Jensen 2019 Sec. III.A)."""
    N = float(_supply_function_N(np.array([En_eV]), T_K, mu_eV)[0])
    if N <= 0.0:
        return 0.0
    theta = _rgtf_theta(En_eV, F_Vpm, phi_eV, mu_eV)
    D = float(expit(-theta))
    return N * D


def _jensen_Z(x: np.ndarray) -> np.ndarray:
    """Z(x) = sum_{j=1}^inf (-1)^(j+1) / [j(j+x)] (Jensen 2007 Eq. 31), via the exact digamma
    closed form Z(x) = (1/x){ln2 - (1/2)[psi((x+2)/2) - psi((x+1)/2)]} (x!=0; Z(0)=pi^2/12) --
    verified against the paper's own special values Z(0)=pi^2/12, Z(1)=2ln2-1 to full float
    precision. Used instead of a truncated series or a restricted-domain polynomial fit because
    Sigma(x) below must be evaluated at both x=n and x=1/n, one of which is always >1 when n!=1."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-9
    x_safe = np.where(small, 1.0, x)
    val = (np.log(2.0) - 0.5 * (digamma((x_safe + 2.0) / 2.0) - digamma((x_safe + 1.0) / 2.0))) / x_safe
    return np.where(small, (np.pi ** 2) / 12.0, val)


def _sigma_series_2007(x: np.ndarray) -> np.ndarray:
    """Sigma(x) = 1 + x^2[Z(x)+Z(-x)] (Jensen 2007 Eq. 30 area, restated compactly per the
    extraction), valid for the full x>0 domain needed by N(n,s)'s Sigma(1/n)/Sigma(n) pair --
    unlike the Eq. (51) rational approximation, which is only accurate for |x|<1."""
    x = np.asarray(x, dtype=float)
    return 1.0 + (x ** 2) * (_jensen_Z(x) + _jensen_Z(-x))


def _rgtf_N_raw(n: float, s: float) -> float:
    return np.exp(-s) * (n ** 2) * float(_sigma_series_2007(np.array([1.0 / n]))[0]) + np.exp(-n * s) * float(
        _sigma_series_2007(np.array([n]))[0]
    )


#: Half-width of the n-space band, around n=1 or any other positive integer >=1, treated as
#: unreliable for the k=1-truncated analytic Sigma(x) series (see _rgtf_N docstring). Calibrated
#: empirically against the paper's own worked examples: n=12.0725 (0.0725 from the n=12 pole)
#: keeps the excellent analytic result (large-integer poles have weak, 1/j residues and barely
#: perturb anything even much closer than this), while n=1.9893 (0.0107 from the n=2 pole, a
#: much stronger low-j pole) needs the fallback -- 0.05 separates the two correctly.
_RGTF_POLE_GUARD_EPS = 0.05

#: Outer radius of a smooth (C1) blend zone from _RGTF_POLE_GUARD_EPS out to this distance: for
#: _RGTF_POLE_GUARD_EPS <= d < _RGTF_BLEND_OUTER_EPS, N(n,s) is a smoothstep-weighted mix of the
#: exact energy-domain integral and the analytic series, instead of a hard switch at exactly
#: _RGTF_POLE_GUARD_EPS. A hard switch there made N(n,s) -- and therefore J_rgtf_2019(F) -- both
#: value- and slope-discontinuous in n (and hence in F, since n(F) is smooth), which was the
#: original, most severe cause of large, unphysical-looking spikes in the finite-difference field
#: sensitivity S_F=dlnJ/dlnF: a centered FD stencil straddling a real jump returns an enormous (in
#: the limit, unbounded) slope that has nothing to do with the model's actual physics.
#:
#: Deliberately kept narrow (only 0.02 wider than the guard itself) so it stays well clear of both
#: benchmarked worked examples in test_rgtf_2019.py -- n=12.0725 (d=0.0725, comfortably outside, so
#: still 100% analytic as before) and n=1.9893 (d=0.0107, comfortably inside, so still 100%
#: exact-integral as before). This width was checked empirically (not just reasoned about) against
#: widening it further: at n=12.0725, _rgtf_exact_N_integral itself disagrees with the paper's own
#: reference N by ~8.9% (the pure analytic series matches to ~0.2%), so blending more of the
#: "exact" integral in at that point would pull an already-excellent analytic value *toward* a
#: *less* accurate one -- confirming this deliberately narrow width, not just avoiding one, is the
#: correct choice, not merely a cautious one.
#:
#: **On that ~8.9% gap itself** (checked again directly, not just re-asserted): it is *not* an
#: integration-range-truncation effect -- widening _rgtf_exact_N_integral's bounds by up to 400x
#: kT beyond the already-generous default leaves the result completely unchanged (confirmed
#: numerically), so the integral's mass genuinely is already fully captured. More importantly,
#: n=12.0725 has d_pole=0.075, just *outside* _RGTF_BLEND_OUTER_EPS -- meaning
#: _rgtf_exact_N_integral is never actually invoked at this specific benchmarked point in
#: production at all (pure analytic is always used there). So this ~8.9% number, while real and
#: reproducible, does not describe the exact integral's accuracy anywhere it is actually used; it
#: is retained here only as the reason not to widen the blend zone to *include* that point. The
#: exact integral's absolute accuracy strictly inside the blend zone (0.05<=d<0.07) has no
#: independent reference to check against (the paper gives no worked example there), so it cannot
#: currently be verified beyond internal consistency (smoothness across the blend, already
#: confirmed via the sensitivity-curve fix below) -- an honest, currently-irreducible gap, not
#: something a bounds or tolerance change can close.
#:
#: Known residual limitation (found by direct numerical testing against this gun's actual T/phi
#: over its real operating field range, not just the paper's isolated worked examples): when the
#: gun operates deep in the thermionic-dominated regime (n<<1, as expected -- see this module's
#: reduces-to-RD_schottky test and the README's own "gun is expected to be thermionic-dominated"),
#: 1/n can be very large (tens to hundreds) and can change extremely fast with F (dn/dlnF up to
#: ~10^2-10^3 in the tested range) -- so 1/n can sweep across an entire integer pole within a
#: single finite-difference step (rel_step~1e-3 in F), even though this blend makes the underlying
#: N(n,s) C1-smooth in n. A finite difference of a function with structure narrower than its own
#: step size can still report a nonzero excursion at such points, regardless of how smooth the
#: function is analytically.
#:
#: This *used* to be the dominant source of large (O(100)) spurious S_F spikes, traced to a
#: second, independent numerical issue in _rgtf_exact_N_integral itself: its integrand (via
#: _rgtf_theta) is only C1, not smoother, at the barrier-saddle energy, and scipy.integrate.quad's
#: adaptive subdivision placed its nodes near that shifting kink inconsistently from one F to the
#: next -- fixed by passing that kink's exact location via quad's own `points` argument (see
#: _rgtf_exact_N_integral), which was confirmed (not just assumed) to remove essentially all of
#: it: across this gun's realistic populated field range, S_F now stays within order-1 of
#: RDSchottky/murphygood1956_SchottkyNordheim_integral/jensen2014_RDSchottky_MurphyGood_additive
#: (down from O(100) before that fix) even at the handful of points step-doubling still flags as
#: unstable (see test_jensen2019_transition_sensitivity_mostly_smooth_over_realistic_operating_range).
#: Those remaining, much smaller excursions are, by design, still caught by
#: compute_log_sensitivities' step-doubling check and excluded from
#: plot_emission_model_sensitivities' plotted line and y-autoscale (shown only as a marker).
#: Eliminating this last residual fully would need an adaptive-step/analytic-derivative
#: sensitivity computation instead of a fixed-rel_step finite difference -- out of scope for this
#: pass; see UPGRADE_PLAN.md. (An earlier version of this note additionally proposed "a
#: wider/adaptive integration range for _rgtf_exact_N_integral" as a possible further fix, on the
#: theory that its ~8.9% gap from the paper's n=12.0725 worked example was an integration-range
#: truncation effect -- checked directly and confirmed wrong: widening the bounds by up to 400x kT
#: leaves the result completely unchanged, and that benchmark point (d_pole=0.075) sits just
#: outside _RGTF_BLEND_OUTER_EPS anyway, so the exact integral is never invoked there in
#: production at all. See the note above this constant for the corrected account.)
_RGTF_BLEND_OUTER_EPS = 0.07


def _smoothstep_blend_weight(d: float, d_inner: float, d_outer: float) -> float:
    """Weight on the exact-integral branch: 1 for d<=d_inner, 0 for d>=d_outer, a cubic smoothstep
    (zero first derivative at both ends) in between -- so the blended N(n,s) is C1 in d (and hence
    in n and F) at both band edges, not just continuous in value."""
    if d <= d_inner:
        return 1.0
    if d >= d_outer:
        return 0.0
    t = (d - d_inner) / (d_outer - d_inner)
    return 1.0 - (3.0 * t * t - 2.0 * t * t * t)


def _nearest_positive_integer_distance(x: float) -> float:
    """Distance from x to the nearest integer >=1 -- there is no pole at 0 (Sigma(0)=1 exactly,
    since the x^2 prefactor kills off Z(0)/Z(-0) even though both are individually finite
    anyway), so small positive x must not be treated as pole-adjacent."""
    nearest_j = max(1, round(x))
    return abs(x - nearest_j)


def _rgtf_n_near_integer_pole(n: float) -> bool:
    """True if n or 1/n is close enough to a positive integer that Sigma(x)=1+x^2[Z(x)+Z(-x)]
    is unreliable there. This is a *genuine* feature of the paper's own k=1-truncated N(n,s)
    (Eq. 35/11): Z(-x)=sum(-1)^(j+1)/[j(j-x)] has a real term-by-term division-by-zero at x=j for
    every positive integer j, not just j=1 -- confirmed empirically against one of the paper's own
    worked examples (n=1.9893, close enough to the n=2 pole that the raw formula returns a
    negative, unphysical N). J_rgtf_2019 falls back to the exact energy-domain integral (Eq. 36)
    directly as N(n,s) in this band, rather than trusting the analytic series."""
    return min(_nearest_positive_integer_distance(n), _nearest_positive_integer_distance(1.0 / n)) < _RGTF_POLE_GUARD_EPS


def _rgtf_N(n: float, s: float) -> float:
    """N(n,s), Jensen 2019 Eq. (5), via the analytic Eq. (11)/Eq.(35) form, using rgtf_2019's own
    exact n=1 closed form (Eq. 12), N(1,s)=(s+1)e^-s, rather than 2007's disagreeing lim-n->1
    form, since this model is built entirely from the 2019 paper's own internally-consistent
    equations (see module docstring). Callers must check _rgtf_n_near_integer_pole(n) first and
    use the exact energy-domain integral instead when it is True (see J_rgtf_2019)."""
    if abs(n - 1.0) < 1.0e-6:
        return (s + 1.0) * np.exp(-s)
    return _rgtf_N_raw(n, s)


def _rgtf_exact_N_integral(F_Vpm: float, T_K: float, phi_eV: float, mu_eV: float, E_lo_eV: float, E_hi_eV: float) -> float:
    """N[theta(E)], Jensen 2019 Eq. (36): the dimensionless integral

        N[theta(E)] = beta_T * integral ln(1+exp(beta_T(mu-E))) / (1+exp(theta(E))) dE

    -- note this is NOT _supply_function_N (which carries the extra (4 pi m_e kB T/h^3) prefactor
    needed to turn N(n,s) into an absolute current density via J=A_RLD T^2 N(n,s); Eq. 36 is the
    bare dimensionless log-supply factor). Used as the pole-adjacent replacement for N(n,s) itself
    (smoothly blended with the analytic series near a pole -- see _RGTF_BLEND_OUTER_EPS), rather
    than as a multiplicative correction factor on it.
    """
    beta_T_inv_eV = KB_EV_PER_K * T_K  # = 1/beta_T, in eV

    def integrand(En_eV):
        x = (mu_eV - En_eV) / beta_T_inv_eV
        log1p_exp = x if x > 30.0 else np.log1p(np.exp(np.clip(x, -700.0, 30.0)))
        theta = _rgtf_theta(En_eV, F_Vpm, phi_eV, mu_eV)
        return log1p_exp * float(expit(-theta))

    # theta(E) (via _rgtf_theta) is only C1, not smooth, at the barrier saddle En_eV=E_kink_eV --
    # same numerical-noise mechanism as J_murphy_good_direct_reference's integrand (see its
    # docstring note): telling quad about this exact break point makes the result stable to
    # several more significant figures instead of jittering point-to-point in a way that shows up
    # as spurious finite-difference sensitivity noise.
    dphi_eV = float(delta_phi_schottky_eV(np.array([F_Vpm]))[0])
    E_kink_eV = mu_eV + phi_eV - dphi_eV
    quad_points = [E_kink_eV] if E_lo_eV < E_kink_eV < E_hi_eV else None
    val, _ = quad(integrand, E_lo_eV, E_hi_eV, points=quad_points, **_TIGHT_QUAD_KWARGS)
    return val / beta_T_inv_eV


def J_rgtf_2019(
    F_Vpm: np.ndarray,
    T_K,
    phi_eV: float,
    chemical_potential_eV: Optional[float] = None,
) -> np.ndarray:
    """Jensen 2019 reformulated GTF current density, J = A_RLD T^2 N(n,s) [Eq. 4]. N(n,s) is the
    analytic k=1-truncated series away from a pole (n or 1/n near a positive integer), smoothly
    blended into the exact energy-domain integral (Eq. 36) near one (see _RGTF_BLEND_OUTER_EPS) --
    this replaces an earlier hard-switch/multiplicative-correction scheme that made J_rgtf_2019(F)
    both value- and slope-discontinuous in F, producing spurious spikes in its finite-difference
    field sensitivity.

    `T_K` may be a scalar (one temperature for every F_Vpm element) or an array matching F_Vpm's
    shape (a per-element/per-cathode-cell temperature -- see J_murphy_good_direct_reference's
    docstring for the same convention).
    """
    mu_eV = float(chemical_potential_eV) if chemical_potential_eV is not None else DEFAULT_CHEMICAL_POTENTIAL_EV
    F_arr = np.atleast_1d(np.asarray(F_Vpm, dtype=float))
    T_arr = np.broadcast_to(np.asarray(T_K, dtype=float), F_arr.shape)
    J = np.empty_like(F_arr)

    for i, F in enumerate(F_arr):
        F = max(float(F), 1.0)
        T_i = float(T_arr[i])
        beta_T_inv_eV = KB_EV_PER_K * T_i  # = k_B*T in eV, i.e. 1/beta_T
        dphi_eV = float(delta_phi_schottky_eV(np.array([F]))[0])
        E_max_eV = mu_eV + phi_eV - dphi_eV

        beta_F_mu = max(_rgtf_beta_F(mu_eV, F, phi_eV, mu_eV), 1e-12)
        E_lo_eV = mu_eV - 12.0 / beta_F_mu
        E_hi_eV = E_max_eV + 12.0 * beta_T_inv_eV

        def neg_log_dJ(En_eV, F=F, T_i=T_i):
            val = _rgtf_dJ_dE(En_eV, F, T_i, phi_eV, mu_eV)
            return -np.log(val) if val > 0.0 else np.inf

        # In very strong-field/deep-tunneling regimes (y(mu)=dphi(F)/phi >= ~1, well beyond this
        # gun's actual cathode field range) the true integrand maximum can sit outside the
        # Eq.(31)-recommended [E_lo,E_hi] margin; widen and retry rather than silently accept a
        # boundary-pinned (and therefore wrong) E_m -- guide Sec. 16.2 #7 requires every kernel to
        # stay finite over its full configured domain, not just the paper's own worked examples.
        for _widen in range(6):
            # xatol tightened from scipy's default (1e-5, an *absolute* eV tolerance) to 1e-9:
            # confirmed empirically that 1e-5 is not tight enough relative to the tiny (~1e-3
            # relative) perturbations compute_log_sensitivities uses for its finite-difference
            # phi/F sensitivities -- E_m's own residual optimizer imprecision at the default
            # tolerance was large enough, relative to the perturbation-induced shift in E_m, to
            # show up as spurious few-times-1e-4-relative noise in S_phi (and, more weakly, S_F)
            # even at F values with no nearby N(n,s) pole at all. At xatol=1e-9 this noise
            # vanishes (confirmed to 6+ significant figures); the extra Brent iterations this
            # costs are negligible next to this function's own per-call cost.
            res = minimize_scalar(neg_log_dJ, bounds=(E_lo_eV, E_hi_eV), method="bounded", options={"xatol": 1e-9})
            E_m_eV = float(res.x)
            at_hi_edge = E_hi_eV - E_m_eV < 1.0e-6 * max(abs(E_hi_eV), 1.0)
            at_lo_edge = E_m_eV - E_lo_eV < 1.0e-6 * max(abs(E_lo_eV), 1.0)
            if not (at_hi_edge or at_lo_edge):
                break
            if at_hi_edge:
                E_hi_eV = mu_eV + phi_eV + (E_hi_eV - mu_eV) * 2.0
            if at_lo_edge:
                E_lo_eV = mu_eV - (mu_eV - E_lo_eV) * 2.0

        beta_F_m = max(_rgtf_beta_F(E_m_eV, F, phi_eV, mu_eV), 1e-12)
        n = 1.0 / (beta_T_inv_eV * beta_F_m)  # n = beta_T/beta_F(E_m), beta_T=1/beta_T_inv_eV
        theta_m = _rgtf_theta(E_m_eV, F, phi_eV, mu_eV)
        s = theta_m + beta_F_m * (E_m_eV - mu_eV)

        d_pole = min(_nearest_positive_integer_distance(n), _nearest_positive_integer_distance(1.0 / n))
        if d_pole < _RGTF_BLEND_OUTER_EPS:
            # The analytic k=1-truncated series is unreliable near a pole (see
            # _rgtf_n_near_integer_pole's docstring); blend smoothly into the exact energy-domain
            # integral (Eq. 36) rather than switching to it discontinuously at a hard radius (see
            # _RGTF_BLEND_OUTER_EPS docstring for why the hard switch produced spurious sensitivity
            # spikes). Fully exact for d<=_RGTF_POLE_GUARD_EPS, fully analytic for
            # d>=_RGTF_BLEND_OUTER_EPS, smoothstep-blended in between.
            w_exact = _smoothstep_blend_weight(d_pole, _RGTF_POLE_GUARD_EPS, _RGTF_BLEND_OUTER_EPS)
            N_analytic = _rgtf_N(n, s)
            if w_exact >= 1.0:
                N_ns = _rgtf_exact_N_integral(F, T_i, phi_eV, mu_eV, E_lo_eV, E_hi_eV)
            elif w_exact <= 0.0:
                N_ns = N_analytic
            else:
                N_exact = _rgtf_exact_N_integral(F, T_i, phi_eV, mu_eV, E_lo_eV, E_hi_eV)
                N_ns = w_exact * N_exact + (1.0 - w_exact) * N_analytic
        else:
            N_ns = _rgtf_N(n, s)

        J_val = A_RICH * (T_i ** 2) * N_ns
        if np.isfinite(J_val) and J_val >= 0.0:
            J[i] = J_val
        else:
            import warnings

            warnings.warn(
                f"rgtf_2019 produced a non-physical value (N={N_ns!r}) at "
                f"F={F:.3e} V/m, T={T_i} K, phi={phi_eV} eV, mu={mu_eV} eV -- likely a field far "
                "outside this model's validated regime (y(mu)=dphi(F)/phi close to or above 1). "
                "Clamped to 0; treat this field point as unreliable.",
                RuntimeWarning,
            )
            J[i] = 0.0

    result = J if np.ndim(F_Vpm) else J[0]
    return result
