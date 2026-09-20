"""Typed, RF-Track-independent field-map models.

All public quantities in this module use SI units.  The complex fields follow the
convention documented in the artifact metadata; the canonical convention used by
the new pipeline is ``Re(F * exp(+1j * omega * (t - t_ref)))``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class FieldMapValidationError(ValueError):
    """Raised when a field map violates the production data contract."""


@dataclass(frozen=True)
class ValidationIssue:
    """One structured validation finding."""

    severity: str
    code: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in {"warning", "error"}:
            raise ValueError("severity must be 'warning' or 'error'")


@dataclass(frozen=True)
class ValidationReport:
    """Collection of non-fatal and fatal validation findings."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def raise_for_errors(self) -> None:
        if self.valid:
            return
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in self.errors)
        raise FieldMapValidationError(detail)


def _readonly_1d(name: str, value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise FieldMapValidationError(f"{name} must be one-dimensional, got {array.shape}")
    array.setflags(write=False)
    return array


def _readonly_complex(name: str, value: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    dtype = np.complex64 if source.dtype == np.complex64 else np.complex128
    array = np.array(source, dtype=dtype, copy=True)
    if array.ndim != 2:
        raise FieldMapValidationError(f"{name} must be two-dimensional, got {array.shape}")
    array.setflags(write=False)
    return array


def _validate_uniform_axis(name: str, values: np.ndarray) -> float:
    if values.size < 2:
        raise FieldMapValidationError(f"{name} must contain at least two coordinates")
    if not np.all(np.isfinite(values)):
        raise FieldMapValidationError(f"{name} contains non-finite coordinates")
    delta = np.diff(values)
    if np.any(delta <= 0.0):
        raise FieldMapValidationError(f"{name} must be strictly increasing")
    step = float(np.median(delta))
    tolerance = max(1.0e-15, abs(step) * 1.0e-8)
    if not np.allclose(delta, step, rtol=1.0e-8, atol=tolerance):
        spread = float(np.max(np.abs(delta - step)))
        raise FieldMapValidationError(
            f"{name} must be uniform; median step={step:.9g} m, max deviation={spread:.3g} m"
        )
    return step


@dataclass(frozen=True)
class AxisymmetricRFField:
    """Qualified 2-D complex RF field passed to RF-Track.

    Arrays use the RF-Track beam frame and have shape ``(Nz, Nr)``.  The
    representation contains exactly the components supported by the present
    axisymmetric RF-Track element.  Omitted non-axisymmetric content belongs in
    the artifact quality metadata, not in this production object.
    """

    frequency_hz: float
    r_m: np.ndarray
    z_m: np.ndarray
    Er_Vpm: np.ndarray
    Ez_Vpm: np.ndarray
    Btheta_T: np.ndarray
    Bz_T: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequency_hz", float(self.frequency_hz))
        object.__setattr__(self, "r_m", _readonly_1d("r_m", self.r_m))
        object.__setattr__(self, "z_m", _readonly_1d("z_m", self.z_m))
        for name in ("Er_Vpm", "Ez_Vpm", "Btheta_T", "Bz_T"):
            object.__setattr__(self, name, _readonly_complex(name, getattr(self, name)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        self.validate()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.z_m.size, self.r_m.size)

    @property
    def hr_m(self) -> float:
        return float(self.r_m[1] - self.r_m[0])

    @property
    def hz_m(self) -> float:
        return float(self.z_m[1] - self.z_m[0])

    @property
    def Ez0_phasor_axis(self) -> complex:
        """Longitudinal phasor at ``r=0`` and the grid point nearest ``z=0``."""

        iz = int(np.argmin(np.abs(self.z_m)))
        if abs(float(self.z_m[iz])) > max(1.0e-15, 0.25 * self.hz_m):
            raise FieldMapValidationError("z_m does not contain the cathode plane z=0")
        return complex(self.Ez_Vpm[iz, 0])

    def validate(self) -> None:
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise FieldMapValidationError("frequency_hz must be finite and positive")
        _validate_uniform_axis("r_m", self.r_m)
        _validate_uniform_axis("z_m", self.z_m)
        if self.r_m[0] != 0.0:
            raise FieldMapValidationError(f"r_m must start exactly at 0 m, got {self.r_m[0]!r}")
        expected = self.shape
        for name in ("Er_Vpm", "Ez_Vpm", "Btheta_T", "Bz_T"):
            values = getattr(self, name)
            if values.shape != expected:
                raise FieldMapValidationError(f"{name} has shape {values.shape}; expected {expected}")
            if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
                raise FieldMapValidationError(f"{name} contains non-finite values")

    def without_rf_magnetic_field(self) -> "AxisymmetricRFField":
        """Return an explicit E-only A/B-study variant with provenance preserved."""

        metadata = dict(self.metadata)
        metadata["rf_magnetic_field"] = "zeroed_for_control"
        return AxisymmetricRFField(
            frequency_hz=self.frequency_hz,
            r_m=self.r_m,
            z_m=self.z_m,
            Er_Vpm=self.Er_Vpm,
            Ez_Vpm=self.Ez_Vpm,
            Btheta_T=np.zeros(self.shape, dtype=self.Btheta_T.dtype),
            Bz_T=np.zeros(self.shape, dtype=self.Bz_T.dtype),
            metadata=metadata,
        )

    def scaled_to_power(self, target_power_w: float, source_power_w: float) -> "AxisymmetricRFField":
        """Set the target power using one common E/B scale, including on repeated calls.

        This is relative scaling within the map's recorded amplitude model; it
        does not establish that the source was calibrated or at steady state.
        """

        target = float(target_power_w)
        source = float(source_power_w)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("target_power_w must be finite and non-negative")
        if not np.isfinite(source) or source <= 0.0:
            raise ValueError("source_power_w must be finite and positive")
        recorded_source = self.metadata.get("source_power_w", source)
        if not np.isclose(float(recorded_source), source, rtol=1e-12, atol=0.0):
            raise ValueError("source_power_w disagrees with the field's recorded normalization")
        current = float(self.metadata.get("target_power_w", source))
        if not np.isfinite(current) or current < 0.0:
            raise ValueError("recorded target_power_w must be finite and non-negative")
        if current == 0.0 and target > 0.0:
            raise ValueError("cannot recover nonzero fields from a map already scaled to zero power")
        factor = float(np.sqrt(target / current)) if current > 0.0 else 1.0
        metadata = dict(self.metadata)
        metadata.update(
            {
                "source_power_w": source,
                "target_power_w": target,
                "common_field_scale": float(np.sqrt(target / source)),
            }
        )
        return AxisymmetricRFField(
            frequency_hz=self.frequency_hz,
            r_m=self.r_m,
            z_m=self.z_m,
            Er_Vpm=self.Er_Vpm * factor,
            Ez_Vpm=self.Ez_Vpm * factor,
            Btheta_T=self.Btheta_T * factor,
            Bz_T=self.Bz_T * factor,
            metadata=metadata,
        )
