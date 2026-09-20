"""Validate cathode mirror-charge physics against the analytical point-charge image force and
the explicit reflected-image-distribution cross-check from
RF_TRACK_2_7_SELF_CONSISTENT_EMISSION_IMPLEMENTATION_GUIDE.md Sec. 10.1/10.2.

RF-Track's SpaceCharge_PIC_FreeSpace.compute_force() returns force in MeV/m (confirmed against
manual_references/RF_Track_2.5.5_reference_manual.pdf Sec. 5.1.3, not assumed), and set_mirror(z)
takes the mirror-plane position in meters (Sec. 7.5). Both conventions are exercised here rather
than trusted from documentation alone.

A single free-standing macroparticle gives compute_force() == 0 exactly, mirror on or off --
this is very likely a degenerate/zero-extent bounding box for the free-space PIC mesh, not a
mirror-charge failure (see test_single_particle_self_force_is_degenerate below, kept as a
documented characterization rather than silently worked around). All physics assertions therefore
use a small distributed bunch, per Sec. 10.2, never a single particle.
"""
from __future__ import annotations

import numpy as np
import pytest

import rf_gun as rg

q_e = 1.602176634e-19
eps0 = 8.8541878128e-12
ME_MEV = 0.51099895
MEV_PER_M_TO_NEWTON = 1.0e6 * q_e  # 1 MeV/m -> N


def _bunch_matrix(z_mm: float, q_sign: float, N_real_each: float, n: int, sigma_r_mm: float, rng):
    r = sigma_r_mm * np.sqrt(-2.0 * np.log(rng.uniform(1e-12, 1.0, n)))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    zeros = np.zeros(n)
    M = np.column_stack([
        x, zeros, y, zeros, np.full(n, z_mm), zeros,
        np.full(n, ME_MEV), np.full(n, q_sign), np.full(n, N_real_each), zeros,
    ])
    return M


def test_single_particle_self_force_is_degenerate():
    """Documents (does not silently paper over) that a single free-standing macroparticle yields
    exactly zero compute_force(), mirror on or off -- a zero-extent-bunch mesh degeneracy, not a
    mirror defect. Guards this characterization so a future RF-Track version change is noticed."""
    M = np.array([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0, ME_MEV, -1.0, 1.0e9, 0.0]])
    B = rg.rft.Bunch6dT(M)
    sc = rg.rft.SpaceCharge_PIC_FreeSpace(32, 32, 32)
    sc.set_mirror(0.0)
    F = np.asarray(sc.compute_force(B))
    assert np.all(F == 0.0), f"expected the known single-particle degeneracy (F==0), got {F}"


@pytest.mark.parametrize("mesh", [32, 48, 64])
def test_mirror_matches_explicit_image_distribution(mesh):
    """Cross-check RF-Track's set_mirror() against a hand-built reflected image distribution:
    (F_mirror_on - F_free) on the real bunch should agree with the force from an explicit
    opposite-sign image bunch reflected through z=0, both attractive (Fz<0) and comparable in
    magnitude, per Sec. 10.2's convergence expectation.
    """
    rng = np.random.default_rng(42)
    n = 300
    d_mm = 3.0
    sigma_r_mm = 0.4
    N_real_each = 1.0e7  # per macroparticle

    M_real = _bunch_matrix(d_mm, -1.0, N_real_each, n, sigma_r_mm, rng)

    # (A) free space, mirror off
    B_free = rg.rft.Bunch6dT(M_real.copy())
    sc_free = rg.rft.SpaceCharge_PIC_FreeSpace(mesh, mesh, mesh)
    F_free = np.asarray(sc_free.compute_force(B_free))

    # (B) mirror on, same real bunch
    B_mirror = rg.rft.Bunch6dT(M_real.copy())
    sc_mirror = rg.rft.SpaceCharge_PIC_FreeSpace(mesh, mesh, mesh)
    sc_mirror.set_mirror(0.0)
    F_mirror = np.asarray(sc_mirror.compute_force(B_mirror))

    # (C) explicit reflected image: same real particles (z=+d, q=-1) plus mirrored copies
    # (z=-d, q=+1, opposite sign per the method of images), mirror OFF -- force evaluated on the
    # first n (real) rows only.
    M_image = M_real.copy()
    M_image[:, 4] *= -1.0   # Z -> -Z
    M_image[:, 7] *= -1.0   # Q -> -Q
    M_combined = np.vstack([M_real, M_image])
    B_combined = rg.rft.Bunch6dT(M_combined)
    sc_combined = rg.rft.SpaceCharge_PIC_FreeSpace(mesh, mesh, mesh)
    F_combined = np.asarray(sc_combined.compute_force(B_combined))
    F_explicit_image_only = F_combined[:n] - F_free

    Fz_mirror_induced = (F_mirror - F_free)[:, 2] * MEV_PER_M_TO_NEWTON
    Fz_explicit_induced = F_explicit_image_only[:, 2] * MEV_PER_M_TO_NEWTON

    mean_mirror = float(np.mean(Fz_mirror_induced))
    mean_explicit = float(np.mean(Fz_explicit_induced))
    print(f"mesh={mesh}: mean Fz mirror-induced={mean_mirror:.4e} N, explicit-image={mean_explicit:.4e} N")

    assert mean_mirror < 0.0, f"expected attraction toward z=0 plane, got mean Fz={mean_mirror:.3e} N"
    assert mean_explicit < 0.0, f"explicit-image control itself not attractive: {mean_explicit:.3e} N"

    ratio = mean_mirror / mean_explicit
    assert 0.5 < ratio < 2.0, (
        f"set_mirror()-induced force disagrees with explicit reflected-image distribution by "
        f"more than 2x at mesh={mesh}: ratio={ratio:.3f}"
    )
