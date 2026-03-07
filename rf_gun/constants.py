"""Physical constants and model coefficients."""
from math import pi
from scipy.constants import c, e as q_e, epsilon_0, k as KB, eV as EV, physical_constants
from scipy.constants import h, m_e

ME_MEV = physical_constants["electron mass energy equivalent in MeV"][0]

# Richardson constant [A/m^2/K^2]
A_RICH = 4 * pi * m_e * q_e * (KB ** 2) / (h ** 3) # Check if we should use another value like 2.9e5 for LaB6

# Fowler-Nordheim constants (phi in eV, F in V/m, J in A/m^2)
A_FN = 1.541434e-6
B_FN = 6.830890e9
