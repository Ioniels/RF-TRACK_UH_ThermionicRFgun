"""Field map loading utilities."""
import numpy as np
import scipy.io

from .phasor import interp_cfield


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
