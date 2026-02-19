"""Physical constants and model coefficients."""
from scipy.constants import c, e as q_e, epsilon_0, k as KB, eV as EV, physical_constants

ME_MEV = physical_constants["electron mass energy equivalent in MeV"][0]

# Richardson constant [A/m^2/K^2]
A_RICH = 2.9e5 # For LaB6

# Fowler-Nordheim constants (phi in eV, F in V/m, J in A/m^2)
A_FN = 1.541434e-6
B_FN = 6.830890e9
