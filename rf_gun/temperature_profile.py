"""Static COMSOL heater surface exports, independent of RF-Track.

The callable interface takes cathode-frame millimetres (emission); ``on_grid``
takes metres (thermal solver). Interpolators are constructed once, directly on
scattered nodes, so no plotting-grid resolution enters the physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from io import StringIO
from pathlib import Path
import re

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError

__all__ = ["ComsolSurfaceTemperature", "load_comsol_surface_temperature"]


@dataclass
class ComsolSurfaceTemperature:
    """Linear T(x,y), with explicit ``error`` or ``nearest`` hull-edge policy.

    ``nearest`` extends the measured edge into unmeasured regions, including
    the chamfer if requested by the caller. It is a modelling assumption, not
    additional COMSOL data. Thermal layers all inherit the surface temperature;
    a surface-only export cannot supply a depth profile.
    """

    xy_mm: np.ndarray
    T_K: np.ndarray
    metadata: dict
    outside_hull: str = "error"
    _linear: LinearNDInterpolator = field(init=False, repr=False)
    _nearest: NearestNDInterpolator = field(init=False, repr=False)

    def __post_init__(self):
        if self.outside_hull not in ("error", "nearest"):
            raise ValueError("outside_hull must be 'error' or 'nearest'")
        self.xy_mm = np.asarray(self.xy_mm, dtype=float)
        self.T_K = np.asarray(self.T_K, dtype=float)
        if (self.xy_mm.ndim != 2 or self.xy_mm.shape[1] != 2
                or self.T_K.shape != (len(self.xy_mm),)
                or not np.all(np.isfinite(self.xy_mm))
                or not np.all(np.isfinite(self.T_K) & (self.T_K > 0))):
            raise ValueError("Expected finite (N,2) xy coordinates and positive (N,) temperatures")
        try:
            self._linear = LinearNDInterpolator(self.xy_mm, self.T_K)
        except QhullError as exc:
            raise ValueError("Temperature nodes must span a two-dimensional surface") from exc
        self._nearest = NearestNDInterpolator(self.xy_mm, self.T_K)
        self.metadata = {**self.metadata, "interpolation": "linear_scattered",
                         "outside_hull": self.outside_hull,
                         "depth_extension": "uniform_through_layers"}

    def __call__(self, x_mm, y_mm):
        x, y = np.broadcast_arrays(np.asarray(x_mm, float), np.asarray(y_mm, float))
        points = np.column_stack((x.ravel(), y.ravel()))
        if not np.all(np.isfinite(points)):
            raise ValueError("Temperature query coordinates must be finite")
        values = np.asarray(self._linear(points))
        missing = ~np.isfinite(values)
        if np.any(missing):
            if self.outside_hull == "error":
                raise ValueError(f"{missing.sum()} temperature queries outside COMSOL surface hull")
            values[missing] = self._nearest(points[missing])
        return values.reshape(x.shape)

    def on_grid(self, x_centers_m, y_centers_m, mask, n_layers):
        """Evaluate only active thermal cells; preserve NaNs outside the cathode."""
        x, y = np.meshgrid(np.asarray(x_centers_m) * 1e3,
                           np.asarray(y_centers_m) * 1e3, indexing="ij")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != x.shape:
            raise ValueError("Thermal mask must match the coordinate grid")
        result = np.full((*mask.shape, n_layers), np.nan)
        result[mask, :] = self(x[mask], y[mask])[:, None]
        return result


def load_comsol_surface_temperature(
    path: str | Path, *, outside_hull: str = "error",
    center_xy_mm: tuple[float, float] = (0.0, 0.0), rotation_deg: float = 0.0,
) -> ComsolSurfaceTemperature:
    """Load one planar ``x y z T(K)`` COMSOL text snapshot.

    Read percent-prefixed headers rather than skipping a fixed number of rows.
    Units must be declared in the export. Subtract ``center_xy_mm`` from the
    converted COMSOL coordinates, then rotate counterclockwise into the gun
    frame. No automatic recentering, rescaling, or temperature offset is applied.
    The constant source z is recorded, not interpreted as a gun z coordinate.
    Multi-time/multi-expression exports and nonplanar surfaces are rejected.
    """
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    unit_match = re.search(r"^%\s*Length unit:\s*(\S+)", text, re.MULTILINE)
    scales = {"m": 1e3, "cm": 10.0, "mm": 1.0, "um": 1e-3, "in": 25.4}
    unit = unit_match.group(1) if unit_match else None
    if unit not in scales:
        raise ValueError(f"Missing or unsupported COMSOL Length unit: {unit!r}")
    column_match = re.search(r"^%\s*x\s+y\s+z\s+T\s*\(K\)([^\r\n]*)", text, re.MULTILINE)
    if column_match is None:
        raise ValueError("Expected COMSOL columns x y z T (K)")
    data = np.loadtxt(StringIO(text), comments="%", ndmin=2)
    if data.shape[1] != 4 or data.shape[0] < 3:
        raise ValueError("Expected a single temperature snapshot with four columns x y z T")
    if not np.all(np.isfinite(data)) or np.any(data[:, 3] <= 0):
        raise ValueError("COMSOL coordinates and positive Kelvin temperatures must be finite")
    xyz_mm = data[:, :3] * scales[unit]
    if np.ptp(xyz_mm[:, 2]) > 1e-6:
        raise ValueError("COMSOL surface must be planar at constant z")
    center = np.asarray(center_xy_mm, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)) or not np.isfinite(rotation_deg):
        raise ValueError("Specify a finite xy center and rotation")
    angle = np.deg2rad(rotation_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy_mm = (xyz_mm[:, :2] - center) @ rotation.T
    unique_xy, indices, inverse = np.unique(xy_mm, axis=0, return_index=True, return_inverse=True)
    if not np.allclose(data[:, 3], data[indices, 3][inverse], rtol=0, atol=1e-8):
        raise ValueError("Conflicting temperatures at duplicate COMSOL xy nodes")
    metadata = {
        "source": "COMSOL heater surface snapshot", "path": str(path.resolve()),
        "sha256": sha256(raw).hexdigest(), "source_length_unit": unit,
        "source_temperature_unit": "K", "snapshot": column_match.group(1).strip(),
        "source_z_mm": float(xyz_mm[0, 2]), "node_count": len(unique_xy),
        "center_xy_mm": center.tolist(), "rotation_deg": float(rotation_deg),
        "T_min_K": float(data[:, 3].min()), "T_max_K": float(data[:, 3].max()),
        "coordinate_frame": "cathode x,y [mm]; translated then rotated from COMSOL",
    }
    return ComsolSurfaceTemperature(unique_xy, data[indices, 3], metadata, outside_hull)
