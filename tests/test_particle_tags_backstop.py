"""Regression tests for `rf_gun.particle_tags.build_particle_tags`'s `backstop_z_min_m` split: a
cathode-backstop capture must join `backward_ids`, not `lost_ids`, when supplied; omitting it
must reproduce the pre-backstop-aware behavior (every lost_table row stays in `lost_ids`)."""
from __future__ import annotations

import numpy as np

from rf_gun.particle_tags import backstop_ids_from_lost_table, build_particle_tags

# EXTENDED_PHASE_FMT columns: X, Px, Y, Py, Z, Pz, id, t, E, K
_N_COLS = 10


def _row(id_: int, z: float, pz: float, k: float = 1.0) -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, z, pz, float(id_), 0.0, 0.0, k]


def _make_bout_and_lost_table():
    # id=0: transmitted; id=1: backward, still present at Bout; id=2: backstop capture
    # (lost_table only); id=3: true dynamic-aperture loss (lost_table only).
    Bout_M = np.array([_row(0, 10.0, 0.6), _row(1, -0.5, -0.05)])
    lost_table = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, -1.0, -0.01, 0.0, 0.511, -1.0, 1.0e5, 2.0],  # backstop capture
            [5.0, 0.0, 5.0, 0.0, 5.0, 0.3, 0.0, 0.511, -1.0, 1.0e5, 3.0],  # true aperture loss
        ]
    )
    return Bout_M, lost_table


def test_backstop_ids_from_lost_table_isolates_the_capture_row():
    _, lost_table = _make_bout_and_lost_table()
    ids = backstop_ids_from_lost_table(lost_table, backstop_z_min_m=-0.002)
    assert ids == frozenset({2})


def test_without_backstop_z_min_m_every_lost_row_stays_lost():
    Bout_M, lost_table = _make_bout_and_lost_table()
    tags = build_particle_tags(Bout_M, lost_table, max_kinetic_energy_mev=None)
    assert tags.backward_ids == frozenset({1})
    assert tags.lost_ids == frozenset({2, 3})


def test_with_backstop_z_min_m_capture_moves_to_backward():
    Bout_M, lost_table = _make_bout_and_lost_table()
    tags = build_particle_tags(Bout_M, lost_table, max_kinetic_energy_mev=None, backstop_z_min_m=-0.002)
    assert tags.backward_ids == frozenset({1, 2})
    assert tags.lost_ids == frozenset({3})


def test_backstop_z_min_m_composes_with_extra_backward_ids():
    Bout_M, lost_table = _make_bout_and_lost_table()
    tags = build_particle_tags(
        Bout_M, lost_table, max_kinetic_energy_mev=None, backstop_z_min_m=-0.002,
        extra_backward_ids=frozenset({99}),
    )
    assert tags.backward_ids == frozenset({1, 2, 99})
    assert tags.lost_ids == frozenset({3})
