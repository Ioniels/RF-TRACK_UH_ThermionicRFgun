"""COMSOL field map loading and visualization utilities."""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize, LinearSegmentedColormap


def infer_plane_axes(vertices, tol=1e-12):
    """Infer which coordinate is constant (plane normal)."""
    std = vertices.std(axis=0)
    normal_axis = int(np.argmin(std))
    normal_value = float(np.median(vertices[:, normal_axis]))
    axes = [0, 1, 2]
    axes.remove(normal_axis)
    u_axis, v_axis = axes[0], axes[1]
    if std[normal_axis] > tol:
        print(f"Warning: plane not perfectly flat. std={std[normal_axis]:.3e}")
    return u_axis, v_axis, normal_axis, normal_value

def plot_fieldmap_on_mesh(
    data,
    component="Ez",
    t_index=-1,
    title=None,
    vmin=None,
    vmax=None,
    cmap="plasma",
    nlevels=256,
    xlim=None,
    ylim=None,
):
    """Plot field component on triangular mesh."""

    cmap_ele = plt.cm.get_cmap(cmap, 256)
    new_colors = cmap_ele(np.linspace(0, 1, 256))
    new_colors[-1] = [1, 1, 1, 0]
    cmap_white = LinearSegmentedColormap.from_list("plasma_with_white", new_colors)

    verts = data["vertices"]
    tri = data["facets"]
    if tri is None:
        raise ValueError("No facets found.")

    stds = verts.std(axis=0)
    normal_axis = int(np.argmin(stds))
    axes = [0, 1, 2]
    axes.remove(normal_axis)
    u_axis, v_axis = axes

    U = verts[:, u_axis]
    V = verts[:, v_axis]
    F = np.asarray(data[component])[:, t_index]

    triang = mtri.Triangulation(U, V, triangles=tri)

    if vmin is None:
        vmin = np.min(F)
    if vmax is None:
        vmax = np.max(F)

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    levels = np.linspace(vmin, vmax, nlevels)

    plt.figure()
    cf = plt.tricontourf(
        triang, F,
        levels=levels,
        norm=norm,
        cmap=cmap_white,
        extend="both",
    )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(["x", "y", "z"][u_axis])
    plt.ylabel(["x", "y", "z"][v_axis])

    if title is None:
        t = data["time"][t_index]
        title = f"{component} at t={t:g}"

    plt.title(title)
    plt.colorbar(cf, label=component)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.show()

def _normalize_key(k: str) -> str:
    """Remove MATLAB export artifacts (null chars, spaces)."""
    return k.replace("\x00", "").strip()


def load_fieldmap_mat(filename, verbose=False):
    """Load COMSOL planar field maps from .mat file."""

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
