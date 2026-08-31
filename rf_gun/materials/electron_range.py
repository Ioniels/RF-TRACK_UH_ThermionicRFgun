"""Tabata-Ito-Okabe (TIO) extrapolated electron range and this project's own derived entrance
stopping power, for back-bombardment energy deposition (implementation plan Sec. 3.4/5.3/19.3).

Sources
-------
Tabata, Ito, Okabe, Nucl. Instrum. Methods 103, 85 (1972), p.86 Eq. 6a-6e / Table 2 p.87: defines
the extrapolated range `Rex(tau)` [mass range, g/cm^2] as a function of the reduced kinetic
energy `tau = T / (m_e c^2)` (`m_e c^2 = 511 keV`) via fit coefficients `a1..a5`, themselves
functions of atomic number `Z` and mass `A` through constants `b1..b9`:

    a1 = b1 * A / Z^b2,   a2 = b3 * Z,   a3 = b4 - b5*Z,   a4 = b6 - b7*Z,   a5 = b8 / Z^b9
    b1=0.2335, b2=1.209, b3=1.78e-4, b4=0.9891, b5=3.01e-4, b6=1.468, b7=1.180e-2,
    b8=1.232, b9=0.109

    Rex(tau) = a1 * [ ln(1 + a2*tau)/a2  -  a3*tau / (1 + a4*tau^a5) ]     [g/cm^2]
    R(tau) [cm] = Rex(tau) / rho [g/cm^3]

Bakr, Phys. Rev. ST Accel. Beams 14, 060708 (2011), Table I p.060708-2 is the source of the
LaB6 composition used here: `Z_eff=40.447`, `A_eff=94.735 g/mol`, `rho0=4720 kg/m^3` (exact
match, journal version -- the FEL2011 conference companion `Bakr 2011 thpa34.pdf` gives slightly
different rounded values and is not the source of these numbers). Bakr's own Eq. (3) (p.060708-3)
restates Tabata's `a1` as `a1 = 2.335 * A/Z^1.209` -- note the leading coefficient `2.335`, a
factor of 10 above Tabata's own `b1 = 0.2335`.

**The a1 discrepancy, and why this file uses Tabata's b1=0.2335, not Bakr's printed 2.335**
(plan Sec. 19.3, "must be handled explicitly, not silently" -- do not "fix" this without
re-reading this note and re-deriving the numbers yourself):

Plan Sec. 19.3 asserts that substituting Bakr's printed coefficient (`a1 = 2.335*A/Z^1.209`,
giving `a1 ~= 2.524` for LaB6's `Z_eff`/`A_eff`) "matches the plan's already-stated reference
range table exactly" (1, 10, 100, 300, 500, 1000 keV -> 0.030, 0.623, 23.35, 126.2, 262.3,
663.5 um). Direct numerical evaluation of `R(tau)` performed while implementing this module
shows the OPPOSITE: using `a1 = 2.5239` (Bakr's printed convention) gives values consistently
~10x LARGER than that table at every energy checked (e.g. 6634 um, not 663.5 um, at 1000 keV;
0.300 um, not 0.030 um, at 1 keV -- verified at all six tabulated energies to within ~1%).
Using Tabata's original `b1 = 0.2335` instead (`a1 ~= 0.2524` for LaB6) reproduces the reference
table to better than 1% at every one of the six tabulated energies.

Plan Sec. 19.3's own prose is self-contradictory on this point: it correctly computes that
Tabata's b1 gives "an order of magnitude smaller range" than Bakr's a1, but then identifies the
LARGER (Bakr-coefficient) range as the one matching the plan's own SMALLER tabulated values --
the two statements cannot both be true, and the arithmetic above shows the "order of magnitude
smaller" branch (Tabata's b1=0.2335) is the one that actually reproduces the table. This appears
to be an arithmetic slip made during that verification pass, not a settled resolution.

Working decision for this implementation: use `a1 = b1 * A_eff / Z_eff^b2` with `b1 = 0.2335`
(Tabata's original Table 2 coefficient), because it is the only convention, of the two on record,
that reproduces the plan's own stated reference range table and this project's prior
`LaB6_heating` note. This is recorded here, and in
`rf_gun/materials/data/LaB6_stopping_Bakr_PRSTAB060708_2011.yaml`, as an open discrepancy against
Bakr's printed Eq. (3) and against plan Sec. 19.3's own attribution claim -- NOT as settled
physics. A recommended (not blocking) follow-up before any final production physics claim: cross-
check the resulting LaB6 range curve against an independent electron CSDA range source (e.g. NIST
ESTAR for a comparable effective-Z absorber), per plan Sec. 19.3.

**Units hazard (plan Sec. 19.3), covered by `tests/test_materials.py`'s unit-conversion
regression test:** Tabata's `Rex` is a *mass* range in g/cm^2, recovered as a linear thickness
only via `R = Rex / rho` with `rho` in **g/cm^3**. Bakr's Eq. (3)/Table I instead state density in
**kg/m^3** (`rho0 = 4720 kg/m3`) without reconciling the unit change. This module makes the
conversion explicit (`rho_g_cm3 = rho_kg_m3 / 1000`) so an accidental 10x (from the a1 ambiguity
above) or 1000x (from skipping this conversion) error cannot silently creep back in.

**No analytic stopping-power (dE/dR) formula exists in either source.** Bakr's Fig. 4(b)
"deposited heat power" curves come from a separate PARMELA particle-transport simulation, not
from differentiating the TIO range law. `tio_entrance_stopping_power_kev_per_um` below is
therefore this project's own numerical derivative of `tio_range_um` (central finite difference in
kinetic energy), not a value taken from Tabata or Bakr -- documented as a derived quantity in
`LaB6_stopping_Bakr_PRSTAB060708_2011.yaml`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

#: m_e c^2 in keV, used to form the reduced kinetic energy tau = T_keV / ELECTRON_REST_MASS_KEV.
ELECTRON_REST_MASS_KEV = 511.0

#: Tabata 1972 Table 2 fit constants (b1..b9), used to form a1..a5 = f(Z, A).
#: b1 is the coefficient at the center of the a1 discrepancy discussed in this module's docstring
#: -- see there before changing this value. This implementation uses Tabata's original printed
#: b1=0.2335, NOT Bakr's Eq.(3) printed coefficient 2.335 (a factor of 10 larger).
_TABATA_B1 = 0.2335
_TABATA_B2 = 1.209
_TABATA_B3 = 1.78e-4
_TABATA_B4 = 0.9891
_TABATA_B5 = 3.01e-4
_TABATA_B6 = 1.468
_TABATA_B7 = 1.180e-2
_TABATA_B8 = 1.232
_TABATA_B9 = 0.109


def _tio_coefficients(Z_eff: float, A_eff: float) -> tuple[float, float, float, float, float]:
    """Return (a1, a2, a3, a4, a5) for the TIO range law, per Tabata 1972 Eq. 6a-6e / Table 2.
    See this module's docstring for the a1 (b1) coefficient-convention discrepancy."""
    a1 = _TABATA_B1 * A_eff / Z_eff**_TABATA_B2
    a2 = _TABATA_B3 * Z_eff
    a3 = _TABATA_B4 - _TABATA_B5 * Z_eff
    a4 = _TABATA_B6 - _TABATA_B7 * Z_eff
    a5 = _TABATA_B8 / Z_eff**_TABATA_B9
    return a1, a2, a3, a4, a5


def tio_range_um(kinetic_energy_keV: Any, Z_eff: float, A_eff: float, rho_kg_m3: float) -> Any:
    """Tabata-Ito-Okabe extrapolated electron range `R(E)` in micrometers, for normal incidence
    into a homogeneous absorber of effective atomic number/mass `Z_eff`/`A_eff` and density
    `rho_kg_m3` (see this module's docstring for the exact formula, sources, and the a1
    coefficient-convention discrepancy this implementation resolves).

    `kinetic_energy_keV` may be a scalar or array-like (the return shape follows the input, same
    convention as `rf_gun.work_function_models`). Density is scaled linearly per plan Sec. 5.3:
    `R(E,T) = R(E,T0) * rho(T0)/rho(T)` -- callers wanting the T-dependent range should pass the
    density at the temperature of interest as `rho_kg_m3` directly (this function has no notion
    of temperature itself).
    """
    E = np.asarray(kinetic_energy_keV, dtype=float)
    tau = E / ELECTRON_REST_MASS_KEV

    a1, a2, a3, a4, a5 = _tio_coefficients(float(Z_eff), float(A_eff))
    rho_g_cm3 = float(rho_kg_m3) / 1000.0  # kg/m^3 -> g/cm^3 (plan Sec. 19.3 units hazard)

    term1 = np.log1p(a2 * tau) / a2
    term2 = a3 * tau / (1.0 + a4 * tau**a5)
    R_cm = (a1 / rho_g_cm3) * (term1 - term2)
    R_um = R_cm * 1.0e4

    return R_um if E.ndim > 0 else float(R_um)


def tio_entrance_stopping_power_kev_per_um(
    kinetic_energy_keV: Any,
    Z_eff: float,
    A_eff: float,
    rho_kg_m3: float,
    *,
    relative_step: float = 1.0e-4,
) -> Any:
    """Numerically differentiate `tio_range_um` with respect to kinetic energy to obtain an
    entrance stopping power `dE/dR` [keV/um] at the given incident energy.

    This is this project's own derived quantity, not a value taken from Tabata or Bakr: neither
    source gives an analytic dE/dR formula (see this module's docstring). Uses a central finite
    difference in E with a small relative step (`relative_step`, default 1e-4 of E, floored at
    1e-6 keV to stay well-posed near E -> 0).
    """
    E = np.asarray(kinetic_energy_keV, dtype=float)
    h = np.maximum(np.abs(E) * relative_step, 1.0e-6)
    E_hi = E + h
    E_lo = np.clip(E - h, 1.0e-9, None)

    R_hi = tio_range_um(E_hi, Z_eff, A_eff, rho_kg_m3)
    R_lo = tio_range_um(E_lo, Z_eff, A_eff, rho_kg_m3)
    dE = E_hi - E_lo
    dR = np.asarray(R_hi, dtype=float) - np.asarray(R_lo, dtype=float)

    S = dE / dR  # keV/um

    return S if E.ndim > 0 else float(S)
