"""Physical constants and model coefficients."""
from math import pi
from scipy.constants import c, e as q_e, epsilon_0, k as KB, eV as EV, physical_constants
from scipy.constants import h, m_e

ME_MEV = physical_constants["electron mass energy equivalent in MeV"][0]

#: mm/c -> ns: (mm/c) * 1e-3 [m/mm] / c [m/s] * 1e9 [ns/s] -- shared conversion for every `%t`
#: (arrival time) column throughout the project (previously duplicated in `beam_properties.py`
#: and `plotting/phase_space.py`, each with its own literal speed-of-light value).
MM_C_TO_NS = 1e-3 / c * 1e9

# Richardson constant [A/m^2/K^2]
A_RICH = 4 * pi * m_e * q_e * (KB ** 2) / (h ** 3) # Check if we should use another value like 2.9e5 for LaB6

# Fowler-Nordheim constants (phi in eV, F in V/m, J in A/m^2)
A_FN = 1.541434e-6
B_FN = 6.830890e9
