"""Field map loading utilities."""
import numpy as np
import scipy.io

from .phasor import interp_cfield


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
