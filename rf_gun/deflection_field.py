"""Deflection magnet: horizontal external B-field via RF-Track's UserField.

Physical model (fit to the deflection magnet, z measured from the cathode,
z=0 at the cathode surface, matching this project's convention throughout):

    B0(z, I) = Bx(0, 0, z; I) = B_pk(I) / (1 + ((z - z_p) / w)^2)
    B_pk(I)  = B_PK_PER_A_T * I      # T, sign(Bx) follows sign(I)

RF-Track's native 3D Static_Magnetic_FieldMap re-derives a Maxwell-consistent
field from its input grid and does not reproduce an arbitrary hand-specified
Bx(z) profile (verified empirically: a clear input ramp came back as an almost
constant field). rft.UserField instead lets us return the exact analytic field
at each integration point, at the cost of forcing RF-Track to run
single-threaded (a hard RF-Track requirement for UserField, not a choice made
here).

Units: RF-Track's UserField.get_field(x, y, z, t) receives z in mm, which
matches this project's cathode-referenced mm convention directly -- no offset
or scale conversion is needed for z. Returned B is in tesla.
"""
from __future__ import annotations

import numpy as np

from .config import rft

DEFAULT_B_PK_PER_A_T = 0.0114  # T/A
DEFAULT_Z_P_MM = -65.815  # mm, from cathode
DEFAULT_W_MM = 46.6  # mm


def b0_deflection_T(
    z_mm,
    current_A: float,
    B_pk_per_A_T: float = DEFAULT_B_PK_PER_A_T,
    z_p_mm: float = DEFAULT_Z_P_MM,
    w_mm: float = DEFAULT_W_MM,
):
    """Horizontal on-axis deflection field Bx(0,0,z;I), in tesla."""
    z = np.asarray(z_mm, dtype=float)
    B_pk = float(B_pk_per_A_T) * float(current_A)
    return B_pk / (1.0 + ((z - float(z_p_mm)) / float(w_mm)) ** 2)


class DeflectionField(rft.UserField):
    """Horizontal dipole-like field Bx(z), uniform in x,y, for beam deflection."""

    def __init__(
        self,
        length_m: float,
        current_A: float,
        B_pk_per_A_T: float = DEFAULT_B_PK_PER_A_T,
        z_p_mm: float = DEFAULT_Z_P_MM,
        w_mm: float = DEFAULT_W_MM,
    ):
        super().__init__(float(length_m))
        self.current_A = float(current_A)
        self.B_pk_per_A_T = float(B_pk_per_A_T)
        self.z_p_mm = float(z_p_mm)
        self.w_mm = float(w_mm)

    def get_field(self, x, y, z, t):
        Bx = b0_deflection_T(z, self.current_A, self.B_pk_per_A_T, self.z_p_mm, self.w_mm)
        E = np.zeros(3)
        B = np.array([Bx, 0.0, 0.0])
        return E, B
