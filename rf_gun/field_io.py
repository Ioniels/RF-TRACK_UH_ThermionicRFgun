"""Legacy planar-MAT field-map utilities.

The production E/B path is the bounded XFdtd-volume reduction in :mod:`rf_gun.fieldmaps`, which
writes a qualified cylindrical artifact consumed directly by tracking. ``load_fieldmap_mat`` is
kept only so historical electric-only planar runs remain reproducible; it is not the place to
interpret or preprocess the current volume data.
"""
from typing import Sequence

import numpy as np
import scipy.io


def mesh_edge_length_stats(vertices: np.ndarray, facets: np.ndarray) -> dict:
    """Native mesh resolution of a raw sensor mesh (`load_fieldmap_mat`'s `vertices`/`facets`),
    from its triangle edge lengths -- i.e. how finely XFdtd actually resolved the field, before
    RF-Track interpolates it onto its own `(r, z)` grid (`dr_um`/`dz_um`). XFdtd meshes are
    adaptively refined, so this is a spread (min/median/mean/max), not one number.
    """
    nan_stats = {"min_mm": float("nan"), "median_mm": float("nan"), "mean_mm": float("nan"), "max_mm": float("nan")}
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(facets, dtype=int)
    if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] == 0:
        return nan_stats
    edges = [np.linalg.norm(v[f[:, a]] - v[f[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))]
    lengths = np.concatenate(edges)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    if lengths.size == 0:
        return nan_stats
    return {
        "min_mm": float(lengths.min()),
        "median_mm": float(np.median(lengths)),
        "mean_mm": float(lengths.mean()),
        "max_mm": float(lengths.max()),
    }


def _normalize_key(k: str) -> str:
    """Remove MATLAB export artifacts (null chars, spaces)."""
    return k.replace("\x00", "").strip()


def load_fieldmap_mat(filename, verbose=False):
    """Load planar field maps from a .mat file."""
    mat_raw = scipy.io.loadmat(filename)

    mat = {}
    for k, v in mat_raw.items():
        if k.startswith("__"):
            continue
        mat[_normalize_key(k)] = v

    if verbose:
        print("Available variables:", sorted(mat.keys()))

    required = ("vertex_X", "vertex_Y", "vertex_Z", "Time_Dimension_2", "TotalField_E_X", "TotalField_E_Y", "TotalField_E_Z")
    missing = [k for k in required if k not in mat]
    if missing:
        raise KeyError(
            f"load_fieldmap_mat({filename!r}): missing expected variable(s) {missing} -- "
            f"available: {sorted(mat.keys())}"
        )

    X = np.asarray(mat["vertex_X"]).ravel()
    Y = np.asarray(mat["vertex_Y"]).ravel()
    Z = np.asarray(mat["vertex_Z"]).ravel()
    vertices = np.column_stack([X, Y, Z])

    facets = None
    if "FacetList" in mat:
        facets = np.asarray(mat["FacetList"], dtype=int) - 1

    time = np.asarray(mat["Time_Dimension_2"]).ravel()

    Ex = np.asarray(mat["TotalField_E_X"])
    Ey = np.asarray(mat["TotalField_E_Y"])
    Ez = np.asarray(mat["TotalField_E_Z"])

    return {
        "vertices": vertices,
        "facets": facets,
        "time": time,
        "Ex": Ex,
        "Ey": Ey,
        "Ez": Ez,
        "raw_keys": list(mat_raw.keys()),
        "keys": list(mat.keys()),
    }


def detect_pec_boundary_along_axis(
    fieldmap: dict,
    *,
    x_target_mm: float = 0.0,
    field_components: Sequence[str] = ("Ex", "Ey"),
    zero_floor_ratio: float = 1.0e-4,
) -> dict:
    """Locate a PEC (perfectly-conducting) boundary crossing along a planar field map's own
    y-axis at a fixed `x_target_mm` -- e.g. the cathode's emitting face, which XFdtd's solver
    represents as a metal disk that fully screens the field on its far side (see this project's
    `Y_CATHODE_MM`/`--y_cathode_mm`: the field-map y-coordinate assumed to be the cathode plane).

    Method: at the mesh column nearest `x_target_mm`, take `max(abs(component))` over every time
    sample and every requested component (`field_components`, default `("Ex", "Ey")`) as one
    scalar field-magnitude value per native y-grid row. A field-free (PEC-shadowed) region reads
    at the solver's own numerical noise floor -- many orders of magnitude below the open/vacuum
    region -- so the transition is found as the sharpest single-step drop in `log(magnitude)`
    across consecutive y rows, not a fixed absolute threshold (a fixed threshold would need
    re-tuning for a field map with different absolute field scale).

    Returns a dict: `x_actual_mm` (the mesh column actually used), `y_mm`/`magnitude` (the full
    sorted per-row arrays, for inspecting or plotting the transition), `last_open_y_mm`/
    `first_shadowed_y_mm` (the two native grid rows bracketing the boundary -- the true PEC
    surface lies somewhere in this interval; the field map's own y-resolution there is
    `first_shadowed_y_mm - last_open_y_mm`, and no finer position is resolvable from this data
    alone), `midpoint_y_mm` (a defensible point estimate absent finer grid resolution), and
    `drop_ratio` (`magnitude[first_shadowed] / magnitude[last_open]`, near-zero for a genuine PEC
    crossing -- a small value here vs. `zero_floor_ratio` is what confirms this is a real boundary
    and not just an ordinary field gradient).

    Raises `ValueError` if the column has fewer than 2 rows, or if no drop steeper than
    `zero_floor_ratio` is found anywhere along the column (i.e. there is no PEC boundary crossing
    at this `x_target_mm` in this field map -- e.g. `x_target_mm` misses the emitting disk's
    footprint entirely, or the field map genuinely has no such boundary).
    """
    x_mm = np.asarray(fieldmap["vertices"], dtype=float)[:, 0]
    y_mm = np.asarray(fieldmap["vertices"], dtype=float)[:, 1]
    ux = np.unique(x_mm)
    x_actual_mm = float(ux[np.argmin(np.abs(ux - float(x_target_mm)))])
    mask = np.isclose(x_mm, x_actual_mm, atol=1.0e-6)
    if np.sum(mask) < 2:
        raise ValueError(
            f"detect_pec_boundary_along_axis: column x={x_actual_mm!r} mm has fewer than 2 rows "
            "-- cannot locate a boundary crossing."
        )

    magnitude = np.zeros(int(np.sum(mask)), dtype=float)
    for name in field_components:
        comp = np.asarray(fieldmap[name], dtype=float)[mask, :]
        magnitude = np.maximum(magnitude, np.max(np.abs(comp), axis=1))

    order = np.argsort(y_mm[mask])
    y_sorted = y_mm[mask][order]
    mag_sorted = magnitude[order]

    with np.errstate(divide="ignore"):
        log_mag = np.log(np.maximum(mag_sorted, np.finfo(float).tiny))
    drop = -np.diff(log_mag)  # positive where magnitude falls from row i to row i+1
    i = int(np.argmax(drop))
    ratio = float(mag_sorted[i + 1] / mag_sorted[i]) if mag_sorted[i] > 0.0 else 0.0
    if ratio > float(zero_floor_ratio):
        raise ValueError(
            f"detect_pec_boundary_along_axis: no drop steeper than zero_floor_ratio="
            f"{zero_floor_ratio!r} found along x={x_actual_mm!r} mm (sharpest drop ratio was "
            f"{ratio!r} at y={y_sorted[i]!r}->{y_sorted[i + 1]!r} mm) -- no PEC boundary crossing "
            "detected at this x."
        )

    return {
        "x_actual_mm": x_actual_mm,
        "y_mm": y_sorted,
        "magnitude": mag_sorted,
        "last_open_y_mm": float(y_sorted[i]),
        "first_shadowed_y_mm": float(y_sorted[i + 1]),
        "midpoint_y_mm": float(0.5 * (y_sorted[i] + y_sorted[i + 1])),
        "drop_ratio": ratio,
    }
