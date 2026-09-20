"""All-frame, fixed-frequency temporal phasor regression."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from numpy.lib.array_utils import normalize_axis_index


@dataclass(frozen=True)
class HarmonicFitConfig:
    """Configuration for a shared-frequency harmonic regression."""

    frequency_hz: float
    reference_time_s: float | None = None
    include_offset: bool = True
    include_linear_envelope: bool = False
    active_amplitude_fraction: float = 1.0e-3

    def __post_init__(self) -> None:
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        if self.reference_time_s is not None and not np.isfinite(self.reference_time_s):
            raise ValueError("reference_time_s must be finite when supplied")
        if not 0.0 <= self.active_amplitude_fraction < 1.0:
            raise ValueError("active_amplitude_fraction must lie in [0, 1)")


@dataclass(frozen=True)
class HarmonicFitResult:
    """Fixed-frequency fit result in the canonical ``exp(+i omega t)`` convention."""

    phasor: np.ndarray
    offset: np.ndarray
    phasor_slope_per_s: np.ndarray
    reference_time_s: float
    frequency_hz: float
    residual_rms: np.ndarray
    global_nrmse: float
    active_nrmse_p95: float
    design_rank: int
    singular_values: np.ndarray
    sample_count: int

    def evaluate(self, time_s: np.ndarray | float) -> np.ndarray:
        """Evaluate the fitted real field at one or more physical times."""

        time = np.asarray(time_s, dtype=np.float64)
        tau = time - self.reference_time_s
        phase = np.exp(1j * (2.0 * np.pi * self.frequency_hz) * tau)
        leading = (time.ndim,) + (1,) * self.phasor.ndim
        tau_b = tau.reshape(tau.shape + (1,) * self.phasor.ndim)
        phase_b = phase.reshape(phase.shape + (1,) * self.phasor.ndim)
        phasor = self.phasor.reshape((1,) * time.ndim + self.phasor.shape)
        slope = self.phasor_slope_per_s.reshape((1,) * time.ndim + self.phasor.shape)
        offset = self.offset.reshape((1,) * time.ndim + self.offset.shape)
        result = offset + np.real((phasor + slope * tau_b) * phase_b)
        if not leading[0]:
            return np.asarray(result).reshape(self.phasor.shape)
        return result

    @property
    def peak_fractional_drift_per_cycle(self) -> float:
        amplitude = np.abs(self.phasor)
        peak = float(np.max(amplitude)) if amplitude.size else 0.0
        if peak == 0.0:
            return 0.0
        period_s = 1.0 / self.frequency_hz
        return float(np.max(np.abs(self.phasor_slope_per_s)) * period_s / peak)


def _validate_samples(time_s: np.ndarray, values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray, int]:
    time = np.asarray(time_s, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if time.ndim != 1:
        raise ValueError("time_s must be one-dimensional")
    data_axis = normalize_axis_index(axis, data.ndim)
    if data.shape[data_axis] != time.size:
        raise ValueError("values length along axis must match time_s")
    if time.size < 3:
        raise ValueError("at least three samples are required for harmonic regression")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be finite and strictly increasing")
    if not np.all(np.isfinite(data)):
        raise ValueError("values contains non-finite samples")
    return time, np.moveaxis(data, data_axis, 0), data_axis


def fit_fixed_frequency_phasor(
    time_s: np.ndarray,
    values: np.ndarray,
    config: HarmonicFitConfig,
    *,
    axis: int = 0,
    metric_mask: np.ndarray | None = None,
) -> HarmonicFitResult:
    """Fit all temporal samples at a prescribed frequency.

    The model is

    ``F(t) = c0 + Re{[P0 + P1*(t-tref)] exp(+i*omega*(t-tref))}``.

    The optional ``P1`` term is a diagnostic for a slowly changing saved-field
    envelope.  No component-wise normalization is performed.
    """

    time, moved, _ = _validate_samples(time_s, values, axis)
    reference = float(np.mean(time)) if config.reference_time_s is None else float(config.reference_time_s)
    tau = time - reference
    omega = 2.0 * np.pi * float(config.frequency_hz)
    cosine = np.cos(omega * tau)
    sine = np.sin(omega * tau)
    columns: list[np.ndarray] = []
    if config.include_offset:
        columns.append(np.ones_like(time))
    columns.extend((cosine, sine))
    time_scale = max(float(np.max(np.abs(tau))), 1.0 / config.frequency_hz)
    if config.include_linear_envelope:
        scaled_tau = tau / time_scale
        columns.extend((scaled_tau * cosine, scaled_tau * sine))
    design = np.column_stack(columns)
    if time.size < design.shape[1]:
        raise ValueError(
            f"{time.size} samples cannot constrain the {design.shape[1]}-parameter harmonic model"
        )
    flat = moved.reshape(time.size, -1)
    coefficients, _, rank, singular = np.linalg.lstsq(design, flat, rcond=None)
    if rank != design.shape[1]:
        raise ValueError(
            f"harmonic design matrix is rank deficient ({rank}/{design.shape[1]}); "
            "check timestamps and frequency"
        )
    cursor = 0
    if config.include_offset:
        offset_flat = coefficients[cursor]
        cursor += 1
    else:
        offset_flat = np.zeros(flat.shape[1], dtype=np.float64)
    cos_flat = coefficients[cursor]
    sin_flat = coefficients[cursor + 1]
    cursor += 2
    phasor_flat = cos_flat - 1j * sin_flat
    if config.include_linear_envelope:
        slope_flat = (coefficients[cursor] - 1j * coefficients[cursor + 1]) / time_scale
    else:
        slope_flat = np.zeros(flat.shape[1], dtype=np.complex128)
    predicted = design @ coefficients
    residual = flat - predicted
    residual_rms_flat = np.sqrt(np.mean(residual * residual, axis=0))
    centered = flat - np.mean(flat, axis=0, keepdims=True) if config.include_offset else flat
    if metric_mask is None:
        metric_flat = np.ones(flat.shape[1], dtype=bool)
    else:
        metric = np.asarray(metric_mask, dtype=bool)
        if metric.shape != moved.shape[1:]:
            raise ValueError(
                f"metric_mask shape {metric.shape} does not match sample shape {moved.shape[1:]}"
            )
        metric_flat = metric.reshape(-1)
        if not np.any(metric_flat):
            raise ValueError("metric_mask selects no samples")
    denominator = float(np.sqrt(np.mean(centered[:, metric_flat] * centered[:, metric_flat])))
    numerator = float(np.sqrt(np.mean(residual[:, metric_flat] * residual[:, metric_flat])))
    global_nrmse = numerator / denominator if denominator > 0.0 else (0.0 if numerator == 0.0 else np.inf)
    amplitude = np.abs(phasor_flat)
    # ``metric_mask`` defines the physical region being qualified (vacuum in
    # the XFdtd reducer).  A large field in excluded metal must not raise the
    # active-cell threshold and thereby hide vacuum cells from the p95 metric.
    metric_amplitude = amplitude[metric_flat]
    peak = float(np.max(metric_amplitude)) if metric_amplitude.size else 0.0
    active = amplitude > config.active_amplitude_fraction * peak if peak > 0.0 else np.zeros_like(amplitude, dtype=bool)
    per_cell_denominator = np.sqrt(np.mean(centered * centered, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_cell_nrmse = residual_rms_flat / per_cell_denominator
    finite_active = active & metric_flat & np.isfinite(per_cell_nrmse)
    active_p95 = float(np.percentile(per_cell_nrmse[finite_active], 95.0)) if np.any(finite_active) else 0.0
    sample_shape = moved.shape[1:]
    return HarmonicFitResult(
        phasor=phasor_flat.reshape(sample_shape),
        offset=offset_flat.reshape(sample_shape),
        phasor_slope_per_s=slope_flat.reshape(sample_shape),
        reference_time_s=reference,
        frequency_hz=float(config.frequency_hz),
        residual_rms=residual_rms_flat.reshape(sample_shape),
        global_nrmse=float(global_nrmse),
        active_nrmse_p95=active_p95,
        design_rank=int(rank),
        singular_values=np.asarray(singular),
        sample_count=int(time.size),
    )


@dataclass(frozen=True)
class EHPhasorFit:
    """E and H component fits sharing frequency and absolute reference time."""

    electric: Mapping[str, HarmonicFitResult]
    magnetic_h: Mapping[str, HarmonicFitResult]
    reference_time_s: float
    frequency_hz: float


def fit_eh_phasors(
    electric_values: Mapping[str, np.ndarray],
    electric_time_s: np.ndarray,
    magnetic_h_values: Mapping[str, np.ndarray],
    magnetic_h_time_s: np.ndarray,
    config: HarmonicFitConfig,
    *,
    axis: int = 0,
) -> EHPhasorFit:
    """Fit E and staggered H samples without pretending their timestamps coincide."""

    e_time = np.asarray(electric_time_s, dtype=np.float64)
    h_time = np.asarray(magnetic_h_time_s, dtype=np.float64)
    if e_time.size == 0 or h_time.size == 0:
        raise ValueError("both electric and magnetic timestamp arrays must be non-empty")
    reference = (
        float(config.reference_time_s)
        if config.reference_time_s is not None
        else 0.5 * (float(min(e_time[0], h_time[0])) + float(max(e_time[-1], h_time[-1])))
    )
    shared = replace(config, reference_time_s=reference)
    electric = {
        name: fit_fixed_frequency_phasor(e_time, values, shared, axis=axis)
        for name, values in electric_values.items()
    }
    magnetic = {
        name: fit_fixed_frequency_phasor(h_time, values, shared, axis=axis)
        for name, values in magnetic_h_values.items()
    }
    return EHPhasorFit(
        electric=electric,
        magnetic_h=magnetic,
        reference_time_s=reference,
        frequency_hz=float(config.frequency_hz),
    )
