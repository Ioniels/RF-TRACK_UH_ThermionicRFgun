"""Qualified evanescent-tail diagnostics and explicitly marked extensions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import AxisymmetricRFField

SPEED_OF_LIGHT_MPS = 299_792_458.0
TM01_FIRST_ZERO = 2.404825557695773


@dataclass(frozen=True)
class EvanescentTailModel:
    """Complex exponential ``exp((-alpha+i*beta)*(z-anchor))`` fit to a modal tail."""

    fit_start_z_m: float
    anchor_z_m: float
    decay_constant_per_m: float
    phase_constant_per_m: float
    log_amplitude_r_squared: float
    reference_phasor: complex
    sample_count: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.fit_start_z_m) or not np.isfinite(self.anchor_z_m):
            raise ValueError("tail fit coordinates must be finite")
        if self.anchor_z_m <= self.fit_start_z_m:
            raise ValueError("anchor_z_m must lie after fit_start_z_m")
        if not np.isfinite(self.decay_constant_per_m) or self.decay_constant_per_m <= 0.0:
            raise ValueError("decay_constant_per_m must be finite and positive")
        if not np.isfinite(self.phase_constant_per_m):
            raise ValueError("phase_constant_per_m must be finite")
        if self.sample_count < 3:
            raise ValueError("at least three samples are required for an evanescent-tail fit")

    @property
    def decay_length_m(self) -> float:
        return 1.0 / self.decay_constant_per_m

    def factor(self, delta_z_m: np.ndarray | float) -> np.ndarray:
        delta = np.asarray(delta_z_m, dtype=np.float64)
        exponent = complex(-self.decay_constant_per_m, self.phase_constant_per_m)
        return np.exp(exponent * delta)

    def integrated_reference_tail(self, end_z_m: float) -> complex:
        """Integral of the fitted reference phasor from the anchor to ``end_z_m``."""

        length = float(end_z_m) - self.anchor_z_m
        if length < 0.0:
            raise ValueError("end_z_m must not precede anchor_z_m")
        exponent = complex(-self.decay_constant_per_m, self.phase_constant_per_m)
        if length == 0.0:
            return 0.0j
        return complex(self.reference_phasor * np.expm1(exponent * length) / exponent)

    def as_metadata(self) -> dict[str, float | int | dict[str, float]]:
        return {
            "model": "single_complex_exponential",
            "fit_start_z_m": self.fit_start_z_m,
            "anchor_z_m": self.anchor_z_m,
            "decay_constant_per_m": self.decay_constant_per_m,
            "decay_length_m": self.decay_length_m,
            "phase_constant_per_m": self.phase_constant_per_m,
            "log_amplitude_r_squared": self.log_amplitude_r_squared,
            "reference_phasor": {
                "real": float(np.real(self.reference_phasor)),
                "imag": float(np.imag(self.reference_phasor)),
            },
            "sample_count": self.sample_count,
        }


def fit_evanescent_tail(
    z_m: np.ndarray,
    phasor: np.ndarray,
    *,
    fit_start_z_m: float,
    amplitude_floor_fraction: float = 1.0e-8,
) -> EvanescentTailModel:
    """Fit log-amplitude decay and residual phase slope to a complex 1-D tail."""

    z = np.asarray(z_m, dtype=np.float64)
    values = np.asarray(phasor, dtype=np.complex128)
    if z.ndim != 1 or values.shape != z.shape:
        raise ValueError("z_m and phasor must be matching 1-D arrays")
    if not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0.0):
        raise ValueError("z_m must be finite and strictly increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("phasor contains non-finite values")
    if not 0.0 <= amplitude_floor_fraction < 1.0:
        raise ValueError("amplitude_floor_fraction must lie in [0, 1)")
    amplitude = np.abs(values)
    threshold = amplitude_floor_fraction * float(np.max(amplitude))
    selected = (z >= float(fit_start_z_m)) & (amplitude > threshold)
    if np.count_nonzero(selected) < 3:
        raise ValueError("fewer than three non-zero samples remain in the requested fit window")
    z_fit = z[selected]
    values_fit = values[selected]
    relative_z = z_fit - z_fit[-1]
    amplitude_coefficients = np.polyfit(relative_z, np.log(np.abs(values_fit)), 1)
    decay = -float(amplitude_coefficients[0])
    predicted_log_amplitude = np.polyval(amplitude_coefficients, relative_z)
    observed_log_amplitude = np.log(np.abs(values_fit))
    ss_residual = float(np.sum((observed_log_amplitude - predicted_log_amplitude) ** 2))
    ss_total = float(np.sum((observed_log_amplitude - np.mean(observed_log_amplitude)) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0.0 else 1.0
    phase = np.unwrap(np.angle(values_fit))
    phase_slope = float(np.polyfit(relative_z, phase, 1)[0])
    if decay <= 0.0:
        raise ValueError(f"tail does not decay in the requested direction (fitted alpha={decay:.6g} 1/m)")
    reference = complex(
        np.exp(amplitude_coefficients[1] + 1j * float(np.polyfit(relative_z, phase, 1)[1]))
    )
    return EvanescentTailModel(
        fit_start_z_m=float(z_fit[0]),
        anchor_z_m=float(z_fit[-1]),
        decay_constant_per_m=decay,
        phase_constant_per_m=phase_slope,
        log_amplitude_r_squared=r_squared,
        reference_phasor=reference,
        sample_count=int(z_fit.size),
    )


def circular_waveguide_tm01_decay_length_m(radius_m: float, frequency_hz: float) -> float:
    """Analytic below-cutoff TM01 amplitude decay length for a circular pipe."""

    radius = float(radius_m)
    frequency = float(frequency_hz)
    if radius <= 0.0 or frequency <= 0.0:
        raise ValueError("radius_m and frequency_hz must be positive")
    transverse_wavenumber = TM01_FIRST_ZERO / radius
    free_space_wavenumber = 2.0 * np.pi * frequency / SPEED_OF_LIGHT_MPS
    argument = transverse_wavenumber**2 - free_space_wavenumber**2
    if argument <= 0.0:
        raise ValueError("TM01 is not below cutoff for this radius and frequency")
    return float(1.0 / np.sqrt(argument))


def extend_axisymmetric_with_modal_tail(
    field: AxisymmetricRFField,
    target_z_m: float,
    model: EvanescentTailModel,
    *,
    require_aligned: bool = False,
) -> AxisymmetricRFField:
    """Extend all components with one fitted modal factor and explicit provenance.

    This is intended only after a tail-quality gate has established a genuine
    single-mode exponential.  It never disguises extrapolated cells as measured.
    """

    target = float(target_z_m)
    source_end = float(field.z_m[-1])
    if target <= source_end:
        raise ValueError("target_z_m must lie beyond the measured grid")
    if abs(model.anchor_z_m - source_end) > 0.51 * field.hz_m:
        raise ValueError(
            f"tail model anchor {model.anchor_z_m:.9g} m does not match field end {source_end:.9g} m"
        )
    intervals_float = (target - source_end) / field.hz_m
    aligned_count = int(np.round(intervals_float))
    is_aligned = aligned_count >= 1 and np.isclose(
        intervals_float, aligned_count, rtol=0.0, atol=2.0e-8
    )
    if require_aligned and not is_aligned:
        raise ValueError(
            "target_z_m must align with the field's uniform z grid; plan the measured "
            "prefix and target grid together rather than silently overshooting the target"
        )
    extra_count = aligned_count if is_aligned else int(np.ceil(intervals_float))
    extra_z = source_end + field.hz_m * np.arange(1, extra_count + 1, dtype=np.float64)
    factors = model.factor(extra_z - source_end)

    def extend(values: np.ndarray) -> np.ndarray:
        tail = factors[:, None] * values[-1:, :]
        return np.concatenate((values, tail), axis=0)

    metadata = dict(field.metadata)
    metadata["spatial_support"] = {
        "measured_z_max_m": source_end,
        "requested_z_max_m": target,
        "artifact_z_max_m": float(extra_z[-1]),
        "extension": "extrapolated_single_mode_evanescent_tail",
        "tail_model": model.as_metadata(),
    }
    return AxisymmetricRFField(
        frequency_hz=field.frequency_hz,
        r_m=field.r_m,
        z_m=np.concatenate((field.z_m, extra_z)),
        Er_Vpm=extend(field.Er_Vpm),
        Ez_Vpm=extend(field.Ez_Vpm),
        Btheta_T=extend(field.Btheta_T),
        Bz_T=extend(field.Bz_T),
        metadata=metadata,
    )
