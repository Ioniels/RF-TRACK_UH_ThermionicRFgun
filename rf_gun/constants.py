"""Physical constants and model coefficients."""
from math import pi
from scipy.constants import c, e as q_e, epsilon_0, k as KB, eV as EV, physical_constants
from scipy.constants import h, m_e

ME_MEV = physical_constants["electron mass energy equivalent in MeV"][0]
ME_KG = m_e

#: Boltzmann constant, derived from scipy's own CODATA values rather than an independently
#: hardcoded literal -- several emission-physics modules (emission_models.py,
#: emission_sensitivity.py, emission_sampling.py) each used to carry their own copy of this same
#: constant in one unit or the other; this is the single shared source now.
KB_J_PER_K = KB
KB_EV_PER_K = KB / EV

#: mm/c -> ns: (mm/c) * 1e-3 [m/mm] / c [m/s] * 1e9 [ns/s] -- shared conversion for every `%t`
#: (arrival time) column throughout the project.
MM_C_TO_NS = 1e-3 / c * 1e9

# Richardson constant [A/m^2/K^2]
A_RICH = 4 * pi * m_e * q_e * (KB ** 2) / (h ** 3) # Check if we should use another value like 2.9e5 for LaB6

# Fowler-Nordheim constants (phi in eV, F in V/m, J in A/m^2)
A_FN = 1.541434e-6
B_FN = 6.830890e9
