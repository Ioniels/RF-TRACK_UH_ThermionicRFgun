"""Regression tests for `rf_gun.diagnostics.classify_particle_outcomes`'s backstop-vs-aperture
split (`backstop_z_min_m`): a backstop capture must count as backward_returned, not lost, when
supplied; omitting it must reproduce the pre-backstop-aware behavior exactly (every lost_table
row counted as lost)."""
from __future__ import annotations

import numpy as np

from rf_gun.diagnostics import classify_particle_outcomes

# EXTENDED_PHASE_FMT columns: X, Px, Y, Py, Z, Pz, id, t, E, K
_N_COLS = 10


def _row(id_: int, z: float, pz: float, k: float = 1.0) -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, z, pz, float(id_), 0.0, 0.0, k]


def _make_population():
    # id=0: transmitted; id=1: backward, still present at Bout; id=2: backstop capture (lost_table
    # only); id=3: true dynamic-aperture loss (lost_table only).
    initial = np.array([_row(0, 0.0, 0.5), _row(1, 0.0, 0.4), _row(2, 0.0, 0.3), _row(3, 0.0, 0.2)])
    final = np.array([_row(0, 10.0, 0.6), _row(1, -0.5, -0.05)])
    # LOST_COLUMNS: X, Px, Y, Py, Z, Pz, T, MASS, Q, N, ID (Z in mm).
    lost_table = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, -1.0, -0.01, 0.0, 0.511, -1.0, 1.0e5, 2.0],  # backstop capture
            [5.0, 0.0, 5.0, 0.0, 5.0, 0.3, 0.0, 0.511, -1.0, 1.0e5, 3.0],  # true aperture loss
        ]
    )
    return initial, final, lost_table


def test_without_backstop_z_min_m_all_lost_table_rows_count_as_lost():
    initial, final, lost_table = _make_population()
    result = classify_particle_outcomes(initial, final, lost_table=lost_table, max_kinetic_energy_mev=None)

    assert result["transmitted"]["count"] == 1
    assert result["backward_returned"]["count"] == 1
    assert result["lost"]["count"] == 2
    assert result["lost"]["count_aperture"] == 2
    total = result["transmitted"]["count"] + result["backward_returned"]["count"] + result["lost"]["count"]
    assert total == result["n_initial"] == 4


def test_with_backstop_z_min_m_backstop_row_counts_as_backward():
    initial, final, lost_table = _make_population()
    result = classify_particle_outcomes(
        initial, final, lost_table=lost_table, max_kinetic_energy_mev=None, backstop_z_min_m=-0.002,
    )

    assert result["transmitted"]["count"] == 1
    assert result["backward_returned"]["count"] == 2
    assert result["backward_returned"]["count_bout"] == 1
    assert result["backward_returned"]["count_backstop"] == 1
    assert result["lost"]["count"] == 1
    assert result["lost"]["count_aperture"] == 1
    total = result["transmitted"]["count"] + result["backward_returned"]["count"] + result["lost"]["count"]
    assert total == result["n_initial"] == 4

    # The backstop-captured particle's own final pz (from its lost_table row) must be folded into
    # backward_returned's stats, not silently dropped.
    assert np.isclose(result["backward_returned"]["final_pz_mean"], (-0.05 + -0.01) / 2.0)


def test_backstop_z_min_m_with_no_lost_table_is_a_no_op():
    initial, final, _ = _make_population()
    result = classify_particle_outcomes(initial, final, lost_table=None, max_kinetic_energy_mev=None, backstop_z_min_m=-0.002)
    assert result["backward_returned"]["count"] == 1
    assert result["lost"]["count"] == 0
