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
    """Horizontal dipole-like field Bx(z), uniform in x,y, for beam deflection.

    `get_field` is invoked once per integration sub-step per particle for as long as this
    field is attached -- i.e. very often, since the deflection magnet forces single-threaded
    tracking (see the module docstring), so a full run makes millions of calls into this
    method. RF-Track's UserField binding never releases the Python objects `get_field`
    returns (empirically confirmed: process RSS grows essentially linearly with elapsed
    tracking time whenever the magnet is on, unboundedly), so returning a freshly allocated
    numpy array on every call leaks memory without bound over the course of a run -- on a
    real N~1000, fine-tier run this exhausts RAM+swap and the process is SIGKILLed right
    around the point the progress bar reaches ~98%, which is exactly the "kernel crashes at
    the last step" symptom this project saw with the magnet on. Returning the SAME
    pre-allocated E/B arrays every call (mutated in place, values copied out C++-side before
    the next call) turns that unbounded per-call leak into a fixed one-time cost.
    """

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
        self._E = np.zeros(3)
        self._B = np.zeros(3)

    def get_field(self, x, y, z, t):
        B_pk = self.B_pk_per_A_T * self.current_A
        self._B[0] = B_pk / (1.0 + ((float(z) - self.z_p_mm) / self.w_mm) ** 2)
        return self._E, self._B
