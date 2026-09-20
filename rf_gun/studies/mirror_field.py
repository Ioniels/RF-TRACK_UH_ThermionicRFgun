"""Reusable mesh study for RF-Track's mirror contribution at fixed probes.

The input bunch and probe positions stay fixed across meshes. The result is a
diagnostic of this particular sample; it is not a universal convergence claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from ..cathode_fields import extract_sc_and_mirror_from_snapshot


@dataclass(frozen=True)
class MirrorMeshSweep:
    mesh_sizes: np.ndarray
    free_Ez_Vpm: np.ndarray
    with_mirror_Ez_Vpm: np.ndarray
    elapsed_s: np.ndarray

    @property
    def mirror_Ez_Vpm(self) -> np.ndarray:
        return self.with_mirror_Ez_Vpm - self.free_Ez_Vpm

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready samples; callers choose tolerances for their geometry."""
        return {
            "mesh_sizes": self.mesh_sizes.tolist(),
            "free_Ez_Vpm": self.free_Ez_Vpm.tolist(),
            "with_mirror_Ez_Vpm": self.with_mirror_Ez_Vpm.tolist(),
            "mirror_Ez_Vpm": self.mirror_Ez_Vpm.tolist(),
            "elapsed_s": self.elapsed_s.tolist(),
        }


def sweep_mirror_mesh(
    rft: Any,
    bunch_matrix: np.ndarray,
    probe_x_m: np.ndarray,
    probe_y_m: np.ndarray,
    probe_z_m: float,
    *,
    mesh_sizes: Sequence[int] = (16, 24, 32, 48, 64),
    mirror_z_m: float = 0.0,
) -> MirrorMeshSweep:
    """Evaluate free and mirrored fields on cubic PIC meshes.

    ``bunch_matrix`` uses the RF-Track snapshot layout accepted by
    ``extract_sc_and_mirror_from_snapshot``. Probe coordinates are in metres.
    RF-Track is passed explicitly and is not imported by this module.
    """
    sizes = tuple(mesh_sizes)
    if not sizes or any(
        not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2
        for n in sizes
    ):
        raise ValueError("mesh_sizes must contain integers >= 2")
    if len(set(sizes)) != len(sizes):
        raise ValueError("mesh_sizes must be distinct")
    free, mirrored, elapsed = [], [], []
    for size in sizes:
        start = perf_counter()
        e_free, e_mirror = extract_sc_and_mirror_from_snapshot(
            rft, bunch_matrix, probe_x_m, probe_y_m, probe_z_m,
            sc_nx=int(size), sc_ny=int(size), sc_nz=int(size), mirror_z_m=mirror_z_m,
        )
        elapsed.append(perf_counter() - start)
        free.append(e_free)
        mirrored.append(e_mirror)
    return MirrorMeshSweep(
        np.asarray(sizes), np.asarray(free), np.asarray(mirrored), np.asarray(elapsed),
    )
