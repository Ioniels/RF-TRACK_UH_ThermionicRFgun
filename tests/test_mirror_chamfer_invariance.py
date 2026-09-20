"""Guide Sec. 16.5 #5: the mirror plane must stay at z=0 regardless of delta_cathode_chamfer_mm --
the coordinate origin is attached to the cathode emission surface, not the cavity geometry, so
shifting the chamfer profile must never move the mirror plane along with it.
"""
from __future__ import annotations

import numpy as np

import rf_gun as rg


def _mirror_state_for(aperture_delta_mm: float) -> np.ndarray:
    nz, nr = 21, 5
    Er = np.zeros((nz, nr), dtype=complex)
    Ez = np.zeros((nz, nr), dtype=complex)
    p = rg.VolumeBuildParams(
        f_hz=2.856e9, map_z0_m=0.0, z_min_m=-0.001, z_max_m=0.05,
        hr_m=1e-3, hz_m=2e-3, dt_mm=0.1,
        sc_enabled=True, mirror_charge_enabled=True, mirror_z_m=0.0,
        aperture_delta_mm=aperture_delta_mm,
        sc_nx=16, sc_ny=16, sc_nz=16,
    )
    V = rg.build_volume(rg.rft, Er, Ez, 0.0, p)
    return np.asarray(V._rf_gun_sc_engine_ref.get_mirror(), dtype=float)


def test_mirror_plane_independent_of_cathode_chamfer_offset():
    for delta_mm in (-0.5, 0.0, 0.3, 1.5):
        mirror_state = _mirror_state_for(delta_mm)
        z_mirror_mm = mirror_state[0, 2]
        assert z_mirror_mm == 0.0, (
            f"mirror plane moved to z={z_mirror_mm} mm for aperture_delta_mm={delta_mm} -- "
            "it must stay at z=0 (cathode-attached), independent of the chamfer profile"
        )
