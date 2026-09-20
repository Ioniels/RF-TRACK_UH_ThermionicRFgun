"""RFTrack-independent figures for XFdtd field-map qualification.

These functions visualize transformed SI data and processed cylindrical fields.
They intentionally contain no legacy solver-axis panels and never import
RF-Track.  Matplotlib is imported lazily inside public plotting functions so
metadata inspection and batch field processing remain lightweight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..fieldmaps.models import AxisymmetricRFField

SPEED_OF_LIGHT_MPS = 299_792_458.0


def _finish(fig: Any, show: bool) -> Any:
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return fig


def _finite_1d(name: str, values: np.ndarray, *, minimum_size: int = 2) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum_size:
        raise ValueError(f"{name} must be a 1-D array with at least {minimum_size} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def plot_mesh_time_metadata(
    coordinate_axes_m: Mapping[str, np.ndarray],
    electric_time_s: np.ndarray,
    magnetic_time_s: np.ndarray,
    frequency_hz: float,
    *,
    title: str = "XFdtd mesh and saved-time sampling",
    show: bool = False,
) -> Any:
    """Plot mesh extents/spacings and separate E/H saved timestamps.

    ``coordinate_axes_m`` can contain every Yee-staggered component axis (for
    example ``"Ex:x"`` and ``"Hy:z"``) or just one representative x/y/z set.
    The figure reports actual nonuniform spacing rather than implying that the
    solver volume is cell-centred on one shared mesh.
    """

    import matplotlib.pyplot as plt

    if not coordinate_axes_m:
        raise ValueError("coordinate_axes_m must not be empty")
    frequency = float(frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    axes = {str(name): _finite_1d(str(name), value) for name, value in coordinate_axes_m.items()}
    e_time = _finite_1d("electric_time_s", electric_time_s)
    h_time = _finite_1d("magnetic_time_s", magnetic_time_s)
    if np.any(np.diff(e_time) <= 0.0) or np.any(np.diff(h_time) <= 0.0):
        raise ValueError("E and H timestamps must be strictly increasing")

    labels = list(axes)
    extents_mm = np.array([(value[-1] - value[0]) * 1.0e3 for value in axes.values()])
    spacing_um = [np.diff(value) * 1.0e6 for value in axes.values()]
    median_spacing = np.array([np.median(value) for value in spacing_um])
    minimum_spacing = np.array([np.min(value) for value in spacing_um])
    maximum_spacing = np.array([np.max(value) for value in spacing_um])
    x = np.arange(len(labels))

    fig, panels = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    ax = panels[0, 0]
    ax.bar(x, extents_mm, color="#4C78A8")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("coordinate span (mm)")
    ax.set_title("Native coordinate coverage")
    ax.grid(axis="y", alpha=0.25)

    ax = panels[0, 1]
    asymmetric_error = np.vstack((median_spacing - minimum_spacing, maximum_spacing - median_spacing))
    ax.errorbar(
        x,
        median_spacing,
        yerr=asymmetric_error,
        fmt="o",
        capsize=3,
        color="#E45756",
        ecolor="#7F7F7F",
    )
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("cell spacing (µm)")
    ax.set_title("Median spacing; whiskers show min–max")
    ax.grid(axis="y", alpha=0.25)

    common_start = min(float(e_time[0]), float(h_time[0]))
    ax = panels[1, 0]
    ax.plot(np.arange(e_time.size), (e_time - common_start) * 1.0e9, "o-", label="E frames")
    ax.plot(np.arange(h_time.size), (h_time - common_start) * 1.0e9, "s--", label="H frames")
    ax.set_xlabel("saved-frame index")
    ax.set_ylabel("time from first saved frame (ns)")
    ax.set_title("Separate Yee-staggered time bases")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = panels[1, 1]
    e_dt = np.diff(e_time)
    h_dt = np.diff(h_time)
    ax.plot(np.arange(1, e_time.size), e_dt * 1.0e12, "o-", label="E cadence")
    ax.plot(np.arange(1, h_time.size), h_dt * 1.0e12, "s--", label="H cadence")
    ax.set_xlabel("saved interval")
    ax.set_ylabel("saved cadence (ps)")
    ax.set_title(
        "Sampling: "
        f"{1.0 / (frequency * np.mean(e_dt)):.3g} E and "
        f"{1.0 / (frequency * np.mean(h_dt)):.3g} H samples/cycle"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=14)
    return _finish(fig, show)


def _metric_value(value: Any, key: str, default: float = 0.0) -> float:
    if isinstance(value, Mapping):
        raw = value.get(key, default)
    else:
        raw = getattr(value, key, default)
    result = float(raw)
    return result if np.isfinite(result) else 0.0


def _flatten_temporal_metrics(temporal: Mapping[str, Any]) -> list[tuple[str, float, float, float]]:
    components = temporal.get("components", temporal)
    if not isinstance(components, Mapping):
        raise ValueError("temporal diagnostics must contain a component mapping")
    rows: list[tuple[str, float, float, float]] = []
    for component, entries in components.items():
        if isinstance(entries, Mapping) or hasattr(entries, "global_nrmse"):
            iterable: Sequence[Any] = [entries]
        elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
            iterable = entries
        else:
            raise ValueError(f"unsupported temporal entry for {component!r}")
        for index, entry in enumerate(iterable):
            label = str(component) if len(iterable) == 1 else f"{component}:{index}"
            rows.append(
                (
                    label,
                    _metric_value(entry, "global_nrmse"),
                    _metric_value(entry, "active_nrmse_p95"),
                    _metric_value(entry, "peak_fractional_drift_per_cycle"),
                )
            )
    if not rows:
        raise ValueError("temporal diagnostics contain no component metrics")
    return rows


def plot_temporal_fit_drift(
    temporal_diagnostics: Mapping[str, Any],
    *,
    title: str = "All-frame harmonic-fit quality",
    show: bool = False,
) -> Any:
    """Plot fit residuals and envelope drift for every processed block/component."""

    import matplotlib.pyplot as plt

    rows = _flatten_temporal_metrics(temporal_diagnostics)
    labels = [row[0] for row in rows]
    values = np.asarray([row[1:] for row in rows], dtype=np.float64) * 100.0
    x = np.arange(len(rows))
    fig, panels = plt.subplots(3, 1, figsize=(max(9.0, 0.42 * len(rows)), 9.0), constrained_layout=True)
    specifications = (
        (0, "global fit NRMSE (%)", "Global residual"),
        (1, "active-cell p95 NRMSE (%)", "Active-cell residual"),
        (2, "peak envelope change/cycle (%)", "Linear-envelope drift"),
    )
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for ax, (column, ylabel, subtitle), color in zip(panels, specifications, colors, strict=True):
        ax.bar(x, values[:, column], color=color)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.set_xticks(x, labels if column == 2 else [""] * len(labels), rotation=60, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=14)
    return _finish(fig, show)


def _mapping_or_attribute(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def plot_symmetry_mode_spectra(
    symmetry_diagnostics: Mapping[str, Any] | Any,
    *,
    title: str = "Cylindrical-symmetry qualification",
    show: bool = False,
) -> Any:
    """Plot component non-axisymmetric fractions and signed Fourier-mode power."""

    import matplotlib.pyplot as plt

    fractions = _mapping_or_attribute(
        symmetry_diagnostics, "component_nonaxisymmetric_fraction", {}
    )
    spectra = _mapping_or_attribute(symmetry_diagnostics, "mode_power_fraction", {})
    if not isinstance(fractions, Mapping) or not fractions:
        raise ValueError("symmetry diagnostics contain no component non-axisymmetric fractions")
    if not isinstance(spectra, Mapping) or not spectra:
        raise ValueError("symmetry diagnostics contain no mode-power spectra")
    names = list(fractions)
    fraction_values = [100.0 * float(fractions[name]) for name in names]
    combined_e = float(
        _mapping_or_attribute(symmetry_diagnostics, "combined_e_nonaxisymmetric_fraction", 0.0)
    )
    combined_b = float(
        _mapping_or_attribute(symmetry_diagnostics, "combined_b_nonaxisymmetric_fraction", 0.0)
    )

    fig, panels = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    ax = panels[0]
    x = np.arange(len(names) + 2)
    all_labels = names + ["E combined", "B combined"]
    all_values = fraction_values + [100.0 * combined_e, 100.0 * combined_b]
    colors = ["#4C78A8" if name.startswith("E") else "#E45756" for name in names]
    colors += ["#72B7B2", "#B279A2"]
    ax.bar(x, all_values, color=colors)
    ax.set_xticks(x, all_labels, rotation=45, ha="right")
    ax.set_ylabel("non-$m=0$ RMS amplitude (%)")
    ax.set_title("Discarded non-axisymmetric content")
    ax.grid(axis="y", alpha=0.25)

    ax = panels[1]
    for name, spectrum in spectra.items():
        if not isinstance(spectrum, Mapping):
            continue
        pairs = sorted((int(mode), max(float(power), 1.0e-16)) for mode, power in spectrum.items())
        if pairs:
            modes, powers = zip(*pairs, strict=True)
            ax.semilogy(modes, np.asarray(powers) * 100.0, "o-", label=str(name))
    ax.set_xlabel("signed azimuthal mode $m$")
    ax.set_ylabel("fitted mode-power fraction (%)")
    ax.set_title("Fourier spectrum before $m=0$ projection")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, ncol=2)
    fig.suptitle(title, fontsize=14)
    return _finish(fig, show)


def _field_mask(field: AxisymmetricRFField, vacuum_mask: np.ndarray | None) -> np.ndarray:
    if vacuum_mask is None:
        return np.ones(field.shape, dtype=bool)
    mask = np.asarray(vacuum_mask, dtype=bool)
    if mask.shape != field.shape:
        raise ValueError(f"vacuum_mask has shape {mask.shape}; expected {field.shape}")
    return mask


def _measured_z_limit(field: AxisymmetricRFField, measured_z_max_m: float | None) -> float | None:
    if measured_z_max_m is not None:
        return float(measured_z_max_m)
    support = field.metadata.get("spatial_support")
    if isinstance(support, Mapping) and "measured_z_max_m" in support:
        return float(support["measured_z_max_m"])
    return None


def _signed_meridional(values: np.ndarray, parity: int) -> np.ndarray:
    return np.concatenate((parity * values[:, :0:-1], values), axis=1)


def plot_meridional_fields(
    field: AxisymmetricRFField,
    *,
    phase_deg: float = 0.0,
    vacuum_mask: np.ndarray | None = None,
    measured_mask: np.ndarray | None = None,
    measured_z_max_m: float | None = None,
    aperture_radius_m: np.ndarray | None = None,
    percentile: float = 99.0,
    title: str = "Axisymmetric fields selected for RF-Track",
    show: bool = False,
) -> Any:
    """Plot ``Er``, ``Ez``, ``Btheta`` and ``Bz`` at one common RF phase.

    Odd/even cylindrical parity is applied across signed radius.  Extrapolated
    cells are hatched, and a stored or explicit measured-support endpoint is
    marked on every panel.
    """

    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    if not 50.0 <= percentile <= 100.0:
        raise ValueError("percentile must lie in [50, 100]")
    mask = _field_mask(field, vacuum_mask)
    if measured_mask is None:
        measured = mask.copy()
    else:
        measured = np.array(measured_mask, dtype=bool, copy=True)
        if measured.shape != field.shape:
            raise ValueError(f"measured_mask has shape {measured.shape}; expected {field.shape}")
        measured &= mask
    measured_limit = _measured_z_limit(field, measured_z_max_m)
    if measured_limit is not None:
        measured &= field.z_m[:, None] <= measured_limit + 1.0e-15

    phase = np.exp(1j * np.deg2rad(float(phase_deg)))
    panel_data = (
        (field.Er_Vpm, -1, r"$E_r$", 1.0e-6, "MV/m"),
        (field.Ez_Vpm, +1, r"$E_z$", 1.0e-6, "MV/m"),
        (field.Btheta_T, -1, r"$B_\theta$", 1.0e3, "mT"),
        (field.Bz_T, +1, r"$B_z$", 1.0e3, "mT"),
    )
    signed_r_mm = np.concatenate((-field.r_m[:0:-1], field.r_m)) * 1.0e3
    z_mm = field.z_m * 1.0e3
    signed_mask = _signed_meridional(mask, +1).astype(bool)
    signed_measured = _signed_meridional(measured, +1).astype(bool)
    extrapolated = signed_mask & ~signed_measured

    fig, panels = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True, sharex=True, sharey=True)
    for ax, (phasor, parity, symbol, scale, unit) in zip(panels.flat, panel_data, strict=True):
        at_phase = np.real(np.asarray(phasor) * phase) * scale
        signed = _signed_meridional(at_phase, parity).T
        signed[~signed_mask.T] = np.nan
        finite = np.abs(signed[np.isfinite(signed)])
        limit = float(np.percentile(finite, percentile)) if finite.size else 1.0
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        image = ax.pcolormesh(
            z_mm,
            signed_r_mm,
            signed,
            shading="auto",
            cmap="RdBu_r",
            norm=colors.Normalize(vmin=-limit, vmax=limit),
            rasterized=True,
        )
        if np.any(extrapolated):
            ax.contourf(
                z_mm,
                signed_r_mm,
                extrapolated.T.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["////"],
            )
        if np.any(mask) and np.any(~mask):
            ax.contour(z_mm, signed_r_mm, signed_mask.T.astype(float), levels=[0.5], colors="black", linewidths=0.8)
        if measured_limit is not None and field.z_m[0] <= measured_limit <= field.z_m[-1]:
            ax.axvline(measured_limit * 1.0e3, color="black", ls="--", lw=1.2, label="measured-map end")
        if aperture_radius_m is not None:
            aperture = np.asarray(aperture_radius_m, dtype=np.float64)
            if aperture.shape != field.z_m.shape:
                raise ValueError(f"aperture_radius_m must have shape {field.z_m.shape}")
            ax.plot(z_mm, aperture * 1.0e3, "k:", lw=1.0)
            ax.plot(z_mm, -aperture * 1.0e3, "k:", lw=1.0, label="aperture")
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
        ax.set_title(f"{symbol}: $\\Re(\\hat F e^{{i\\phi}})$, {unit}")
        ax.set_xlabel("beam $z$ (mm)")
        ax.set_ylabel("signed radius (mm)")
        fig.colorbar(image, ax=ax, pad=0.015, label=unit)
    handles, labels = panels.flat[0].get_legend_handles_labels()
    if handles:
        panels.flat[0].legend(handles, labels, loc="best", framealpha=0.8)
    fig.suptitle(f"{title}; common phase $\\phi={float(phase_deg):.1f}^\\circ$", fontsize=14)
    return _finish(fig, show)


def plot_on_axis_tail_decay(
    field: AxisymmetricRFField,
    *,
    vacuum_mask: np.ndarray | None = None,
    tail_diagnostics: Mapping[str, Any] | None = None,
    tail_start_z_m: float | None = None,
    measured_z_max_m: float | None = None,
    near_zero_fraction: float = 1.0e-2,
    title: str = "On-axis field and downstream-tail decay",
    show: bool = False,
) -> Any:
    """Plot on-axis fields, radial envelopes, and remaining measured voltage."""

    import matplotlib.pyplot as plt

    if tail_diagnostics is None:
        from ..fieldmaps.quality import tail_profile_diagnostics

        tail_diagnostics = tail_profile_diagnostics(
            field,
            vacuum_mask,
            tail_start_z_m=tail_start_z_m,
            near_zero_fraction=near_zero_fraction,
            max_profile_points=max(512, field.z_m.size),
        )
    profiles = tail_diagnostics.get("profiles", {})
    z = np.asarray(profiles.get("z_m", []), dtype=np.float64)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("tail diagnostics contain no usable z profile")

    def profile(name: str) -> np.ndarray:
        values = np.asarray(profiles.get(name, []), dtype=np.float64)
        if values.shape != z.shape:
            raise ValueError(f"tail profile {name!r} is missing or has the wrong shape")
        return values

    ez_axis = profile("Ez_axis_abs_Vpm")
    er_rms = profile("Er_Vpm")
    ez_rms = profile("Ez_Vpm")
    bt_rms = profile("Btheta_T")
    bz_rms = profile("Bz_T")
    e_rms = profile("E_combined_Vpm")
    b_rms = profile("B_combined_T")
    remaining = np.asarray(
        tail_diagnostics.get("on_axis_Ez_abs_integral", {}).get("remaining_at_each_profile_z_V", []),
        dtype=np.float64,
    )
    if remaining.shape != z.shape:
        raise ValueError("tail diagnostics contain an incompatible remaining-voltage profile")

    fig, panels = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True, sharex=True)
    z_mm = z * 1.0e3
    ax = panels[0, 0]
    ax.plot(z_mm, ez_axis * 1.0e-6, label=r"$|E_z(r=0)|$")
    ax.plot(z_mm, er_rms * 1.0e-6, label=r"RMS$_r |E_r|$")
    ax.plot(z_mm, ez_rms * 1.0e-6, "--", label=r"RMS$_r |E_z|$")
    ax.set_ylabel("electric phasor amplitude (MV/m)")
    ax.set_title("Electric field")
    ax.legend(frameon=False)

    ax = panels[0, 1]
    ax.plot(z_mm, bt_rms * 1.0e3, label=r"RMS$_r |B_\theta|$")
    ax.plot(z_mm, bz_rms * 1.0e3, label=r"RMS$_r |B_z|$")
    ax.set_ylabel("magnetic phasor amplitude (mT)")
    ax.set_title("Magnetic field")
    ax.legend(frameon=False)

    ax = panels[1, 0]
    e_peak = max(float(np.max(e_rms)), np.finfo(float).tiny)
    cb_peak = max(float(np.max(SPEED_OF_LIGHT_MPS * b_rms)), np.finfo(float).tiny)
    ax.semilogy(z_mm, np.maximum(e_rms / e_peak, 1.0e-14), label=r"$|E|$/max")
    ax.semilogy(
        z_mm,
        np.maximum(SPEED_OF_LIGHT_MPS * b_rms / cb_peak, 1.0e-14),
        label=r"$c|B|$/max",
    )
    ax.axhline(float(tail_diagnostics.get("near_zero_fraction", near_zero_fraction)), color="black", ls=":", label="near-zero threshold")
    ax.set_ylabel("normalized radial RMS")
    ax.set_xlabel("beam $z$ (mm)")
    ax.set_title("Downstream decay (log scale)")
    ax.legend(frameon=False)

    ax = panels[1, 1]
    ax.plot(z_mm, remaining, color="#B279A2")
    ax.set_ylabel(r"remaining measured $\int |E_z(0,z)|\,dz$ (V)")
    ax.set_xlabel("beam $z$ (mm)")
    tail_fraction = float(
        tail_diagnostics.get("on_axis_Ez_abs_integral", {}).get("tail_fraction_of_measured", 0.0)
    )
    ax.set_title(f"Remaining on-axis magnitude; selected tail = {100.0 * tail_fraction:.3g}%")

    measured_limit = _measured_z_limit(field, measured_z_max_m)
    tail_start = float(tail_diagnostics.get("tail_start_z_m", z[-1]))

    # The diagnostic profiles are computed over the *measured* support, so their abscissa
    # normally stops at the measured-volume end. Drawing only that range made the figure end
    # at ~32.9 mm with no sign that the artifact continues to 40.589 mm -- exactly the modeled
    # continuation this panel exists to make visible. Always display out to the field's own
    # extent and shade the modeled interval, even when no profile samples land inside it.
    field_z_max = float(np.asarray(field.z_m)[-1])
    display_z_max = max(float(z[-1]), field_z_max)
    modeled_shown = measured_limit is not None and display_z_max > measured_limit + 1.0e-12

    for ax in panels.flat:
        ax.axvline(tail_start * 1.0e3, color="#7F7F7F", ls=":", lw=1.1, label="tail-fit start")
        if measured_limit is not None and measured_limit >= z[0]:
            ax.axvline(measured_limit * 1.0e3, color="black", ls="--", lw=1.1)
            if modeled_shown:
                ax.axvspan(
                    measured_limit * 1.0e3,
                    display_z_max * 1.0e3,
                    color="0.85",
                    alpha=0.45,
                    hatch="//",
                )
        ax.set_xlim(float(z[0]) * 1.0e3, display_z_max * 1.0e3)
        ax.grid(alpha=0.25, which="both")

    suptitle = title
    if modeled_shown:
        profiles_cover_tail = float(z[-1]) > measured_limit + 1.0e-12
        suptitle = (
            f"{title}\nhatched: modeled tail beyond measured volume at "
            f"{measured_limit * 1.0e3:.3f} mm"
            + ("" if profiles_cover_tail else " (no measured profile samples there)")
        )
    fig.suptitle(suptitle, fontsize=14)
    return _finish(fig, show)


__all__ = [
    "plot_mesh_time_metadata",
    "plot_meridional_fields",
    "plot_on_axis_tail_decay",
    "plot_symmetry_mode_spectra",
    "plot_temporal_fit_drift",
]
