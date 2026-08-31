"""Configurable macropulse model (implementation plan Sec. 8.1, 8.2, 8.5, 10.1, 10.2; addendum
Sec. 19.2/19.4) -- Work Package 3/4's "one-way macropulse heating" (L2_one_way) piece.

Scope: pure Python/numpy, NO RF-Track dependency. This module supplies the three pieces plan
Sec. 8 requires that live strictly between `back_bombardment_deposition.py` (one representative RF
period's deposited-energy tensor) and `thermal.py` (the already-scaled, already-time-resolved
`VolumetricHeatSourceTimeSeries` that solver consumes):

  1. `evaluate_rf_envelope` -- the prescribed macropulse envelope (plan Sec. 8.2: "a prescribed
     RF macropulse envelope"). `"top_hat"` is the only implemented value, a deliberate idealization
     confirmed by addendum Sec. 19.4 ("neither the LaB6_heating note nor the RF-Track manual
     supplies a measured or modeled fill/flat-top/decay envelope for any pulse duration ... there is
     nothing to substitute it with yet"). Any other `envelope` string is a loud `NotImplementedError`,
     never a silent top-hat fallback.
  2. `build_macropulse_time_grid` / `compute_n_rf_periods` -- the macro-time bin-edge grid (plan
     Sec. 8.1's "initial thermal bins of roughly 20-100ns are reasonable") and the diagnostic RF
     period count `N_RF = f_RF * tau_macro` (plan Sec. 8.1's own worked example: 22,848 periods at
     8us/2.856GHz) -- a print/logging value, never iterated over (this project's whole point is NOT
     tracking all 22,848 periods, plan Sec. 1).
  3. `build_macropulse_heat_source` -- multiplies `thermal.build_constant_power_heat_source_time_series`'s
     CONSTANT-power baseline by the envelope evaluated at each time bin's CENTER (documented choice
     below), giving the macropulse-scaled `VolumetricHeatSourceTimeSeries` `thermal.
     solve_xy_layered_thermal` actually consumes.
  4. `MacropulseCurrentHistory` / `build_macropulse_current_history` -- plan Sec. 8.5's reported
     current histories (`I_emit`, `I_return`, `I_transmitted`, `I_other_loss`, `I_useful`), built
     from `events.accounting`'s per-representative-period charge numbers via `I = f_RF * Q_per_period`
     (plan Sec. 8.1's own formula) and scaled by the same envelope.
  5. `validate_charge_balance` -- plan Sec. 8.5's closure check,
     `Q_emit = Q_return + Q_transmitted + Q_other_loss + Q_surviving`.

Coupling level: every quantity here is scaled ONLY by the prescribed envelope -- the underlying
per-representative-period rates (`I_emit_A`'s pre-envelope value, the deposited-power tensor before
multiplying by the envelope, etc.) are held exactly constant across the whole macropulse. There is
no temperature-to-emission or emission-to-cavity feedback anywhere in this module: this is the
literal defining property of `coupling_level="L2_one_way"` (plan Sec. 1/8.2) -- a later
`coupling_level="L3_thermal_emission_feedback"` pass (Work Package 6, explicitly deferred, addendum
Sec. 19.2) would replace the "hold constant, just multiply by envelope" step here with an actual
per-keyframe re-evaluation of the emission/return kernel.

`events.accounting`'s exact shape (built by `rf_gun.back_bombardment_events.BackBombardmentEvents`,
see that class's docstring, NOT re-derived or guessed here)::

    accounting = {
      "counts": {...},
      "charge_C": {
        "emitted": float, "transmitted": float, "other_lost": float,
        "returned_before_filter": float, "returned_after_filter": float,
      },
      "energy_J": {...},
    }

This module reads exactly `accounting["charge_C"]["emitted"/"transmitted"/"other_lost"/
"returned_after_filter"]` -- the QUALIFIED (post-filter) returned charge, matching plan Sec. 3.1's
"accept a heating event only when p_hit . n_in > 0" filter and Sec. 8.5's `I_return = f_RF *
Q_return_per_period` (the qualified/physical return, not the pre-filter candidate population).

`Q_surviving` (plan Sec. 8.5's closure equation `Q_emit = Q_return + Q_transmitted + Q_other_loss +
Q_surviving`) has no dedicated key in `accounting["charge_C"]` -- this implementation's own
documented design choice (the plan does not fix one) is to DEFINE `Q_surviving := Q_emit - (Q_return
+ Q_transmitted + Q_other_loss)` as the residual and require it to be non-negative (to tolerance):
charge accounted for as returned/transmitted/otherwise-lost can never legitimately exceed what was
emitted, so a negative `Q_surviving` is a genuine double-counting/overcounting bug in the upstream
accounting, not a rounding artifact to paper over. `Q_surviving` itself represents charge that
neither returned to the cathode, was transmitted to the output screen, nor was lost elsewhere within
this one representative RF period's snapshot (e.g. still in flight, or lost at a boundary this
accounting dict does not separately track) -- see `validate_charge_balance`'s docstring for the
exact check this performs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .back_bombardment_events import BackBombardmentEvents
from .back_bombardment_study_config import MacropulseConfig
from .thermal import VolumetricHeatSourceTimeSeries, build_constant_power_heat_source_time_series

__all__ = [
    "evaluate_rf_envelope",
    "build_macropulse_time_grid",
    "compute_n_rf_periods",
    "build_macropulse_heat_source",
    "MacropulseCurrentHistory",
    "build_macropulse_current_history",
    "validate_charge_balance",
]

#: Envelope shapes this module actually implements. Any other `MacropulseConfig.envelope` string
#: raises `NotImplementedError` (see `evaluate_rf_envelope`) -- never a silent top-hat fallback.
_IMPLEMENTED_ENVELOPES: tuple[str, ...] = ("top_hat",)


# ================================================================================================
# 1. RF macropulse envelope (plan Sec. 8.2, addendum Sec. 19.4)
# ================================================================================================


def evaluate_rf_envelope(t_s: np.ndarray, macropulse_config: MacropulseConfig) -> np.ndarray:
    """Evaluate the prescribed macropulse envelope (values in `[0,1]`) at times `t_s` (seconds,
    measured from macropulse start `t=0`).

    `macropulse_config.envelope == "top_hat"` (the only implemented value): `1.0` for
    `0 <= t <= duration_s`, `0.0` outside. This is a DELIBERATE, explicitly-labeled idealization
    (plan Sec. 8.2: "a prescribed RF macropulse envelope (initial default: documented 8us top-hat,
    pending the measured fill/decay waveform)"; addendum Sec. 19.4 confirms no measured or modeled
    fill/flat-top/decay envelope exists yet for any pulse duration -- "there is nothing to
    substitute it with yet"). A future measured/modeled envelope belongs here as a NEW named
    `envelope` value, never as a silent change to what `"top_hat"` means.

    Any other `macropulse_config.envelope` value raises `NotImplementedError` with a clear message
    -- this function never silently falls back to top-hat for an unrecognized name.
    """
    if macropulse_config.envelope not in _IMPLEMENTED_ENVELOPES:
        raise NotImplementedError(
            f"evaluate_rf_envelope: envelope={macropulse_config.envelope!r} is not implemented. "
            f"Only {list(_IMPLEMENTED_ENVELOPES)} are implemented so far (plan Sec. 8.2/19.4 -- no "
            "measured or modeled RF fill/decay envelope exists yet for any pulse duration). This "
            "function deliberately refuses to fall back to 'top_hat' silently for an unrecognized "
            "envelope name; add a new named implementation instead of changing this error."
        )
    t = np.asarray(t_s, dtype=float)
    return np.where((t >= 0.0) & (t <= macropulse_config.duration_s), 1.0, 0.0)


# ================================================================================================
# 2. Macro time grid and diagnostic RF-period count (plan Sec. 8.1)
# ================================================================================================


def build_macropulse_time_grid(
    macropulse_config: MacropulseConfig, *, thermal_bin_s: float = 50e-9
) -> np.ndarray:
    """Bin-edge macro-time grid from `0` to `macropulse_config.duration_s`, at (approximately)
    `thermal_bin_s` resolution (plan Sec. 8.1: "initial thermal bins of roughly 20-100ns are
    reasonable, but mesh/time convergence -- not this estimate -- sets the final value"; default
    50ns is a reasonable midpoint of that range, not a validated production value).

    Returns `(n_bins + 1,)` strictly increasing bin-edge seconds, `t[0] = 0.0`,
    `t[-1] = macropulse_config.duration_s` EXACTLY -- this grid never extends past `duration_s`
    (a deliberate choice, see module/`build_macropulse_heat_source` docstrings: since
    `evaluate_rf_envelope`'s top-hat is exactly 1.0 on `[0, duration_s]`, keeping every bin inside
    that closed interval makes the macropulse-integrated energy closure in
    `build_macropulse_heat_source` an EXACT identity rather than one requiring a boundary-bin
    special case). `n_bins = max(1, round(duration_s / thermal_bin_s))`, so `thermal_bin_s` is a
    target resolution, not necessarily achieved exactly (the grid is always evenly spaced over the
    fixed `[0, duration_s]` span).

    `thermal_bin_s` is a normal, caller-configurable keyword -- never hardcoded elsewhere (plan
    Sec. 8.1's own "not this estimate -- sets the final value" instruction).
    """
    if not (thermal_bin_s > 0.0):
        raise ValueError(f"thermal_bin_s must be positive, got {thermal_bin_s!r}")
    duration_s = macropulse_config.duration_s
    n_bins = max(1, int(round(duration_s / thermal_bin_s)))
    return np.linspace(0.0, duration_s, n_bins + 1)


def compute_n_rf_periods(rf_frequency_Hz: float, macropulse_config: MacropulseConfig) -> float:
    """`N_RF = f_RF * tau_macro` (plan Sec. 8.1) -- the number of RF periods spanned by the
    macropulse. Diagnostic/print value ONLY: this project's central design decision is NOT to track
    all `N_RF` periods (plan Sec. 1) -- nothing in this module iterates over this count.

    Plan Sec. 8.1's own worked example: `rf_frequency_Hz=2.856e9`, `duration_s=8e-6` ->
    `N_RF = 22848.0` exactly.
    """
    if not (rf_frequency_Hz > 0.0):
        raise ValueError(f"rf_frequency_Hz must be positive, got {rf_frequency_Hz!r}")
    return float(rf_frequency_Hz) * float(macropulse_config.duration_s)


# ================================================================================================
# 3. Envelope-scaled macropulse heat source (plan Sec. 8.1/8.2)
# ================================================================================================


def build_macropulse_heat_source(
    heat_source: "object",
    rf_frequency_Hz: float,
    macropulse_config: MacropulseConfig,
    t_grid_edges_s: np.ndarray,
) -> VolumetricHeatSourceTimeSeries:
    """Build the envelope-scaled macropulse heat source `thermal.solve_xy_layered_thermal` consumes.

    Procedure (every conversion explicit and printed, per plan Sec. 1's non-negotiable list "Every
    conversion from 'one representative RF period' to amperes, watts, or energy over the macropulse
    is explicit in metadata and independently checked"):

      1. `thermal.build_constant_power_heat_source_time_series(heat_source, 1/rf_frequency_Hz,
         t_grid_edges_s)` -- the already-existing, already-tested "obviously-correct multiplication"
         `q_layer_W = heat_source.q_layer_J * rf_frequency_Hz`, repeated identically (CONSTANT
         power) across every bin of `t_grid_edges_s`. This module does NOT reimplement that
         conversion; it reuses it verbatim.
      2. Multiply each time bin's power by that bin's envelope value, `evaluate_rf_envelope`
         evaluated at the bin's CENTER (`0.5*(t_grid_edges_s[ib] + t_grid_edges_s[ib+1])`) --
         documented choice: a piecewise-constant power source (plan/`VolumetricHeatSourceTimeSeries`'s
         own convention) has one representative instant per bin, and the bin center is the natural,
         symmetric choice (vs. an edge, which would bias a partially-illuminated boundary bin high
         or low depending on which edge was picked). For the `"top_hat"` envelope with a time grid
         that never extends past `duration_s` (`build_macropulse_time_grid`'s own documented
         choice), every bin center lies strictly inside `[0, duration_s]`, so this distinction is
         moot for the default case -- it matters only for a future non-top-hat envelope with
         partially-illuminated bins.

    `N_RF = compute_n_rf_periods(rf_frequency_Hz, macropulse_config)` is printed alongside the
    resulting total macropulse-integrated energy, so a caller can independently check
    `sum(q_layer_W * dt) == heat_source.q_layer_J * N_RF` (exact for the top-hat envelope, see
    `build_macropulse_time_grid`'s docstring) without re-deriving either number by hand.
    """
    rf_period_s = 1.0 / float(rf_frequency_Hz)
    baseline = build_constant_power_heat_source_time_series(heat_source, rf_period_s, t_grid_edges_s)

    t_grid_edges_s = np.asarray(t_grid_edges_s, dtype=float)
    t_centers_s = 0.5 * (t_grid_edges_s[:-1] + t_grid_edges_s[1:])
    envelope = evaluate_rf_envelope(t_centers_s, macropulse_config)

    q_layer_W = baseline.q_layer_W * envelope[np.newaxis, np.newaxis, np.newaxis, :]

    n_rf = compute_n_rf_periods(rf_frequency_Hz, macropulse_config)
    total_period_energy_J = float(np.sum(heat_source.q_layer_J))
    total_macropulse_energy_J = float(np.sum(q_layer_W * np.diff(t_grid_edges_s)[np.newaxis, np.newaxis, np.newaxis, :]))
    expected_energy_J = total_period_energy_J * n_rf
    print(
        "build_macropulse_heat_source: one-period deposited energy="
        f"{total_period_energy_J:.6e} J, rf_frequency_Hz={rf_frequency_Hz:.6g}, "
        f"duration_s={macropulse_config.duration_s:.6g}, N_RF={n_rf:.6g}, "
        f"envelope={macropulse_config.envelope!r} -> macropulse-integrated deposited energy="
        f"{total_macropulse_energy_J:.6e} J (expected one_period_J*N_RF={expected_energy_J:.6e} J "
        f"for a top-hat envelope spanning the whole grid)."
    )

    return VolumetricHeatSourceTimeSeries(
        x_centers_m=baseline.x_centers_m,
        y_centers_m=baseline.y_centers_m,
        layer_boundaries_um=baseline.layer_boundaries_um,
        cathode_footprint_mask=baseline.cathode_footprint_mask,
        q_layer_W=q_layer_W,
        t_grid_s=t_grid_edges_s,
        xy_cell_area_m2=baseline.xy_cell_area_m2,
    )


# ================================================================================================
# 4. Current histories (plan Sec. 8.5)
# ================================================================================================


@dataclass(frozen=True)
class MacropulseCurrentHistory:
    """Reported macropulse current histories over macro time (plan Sec. 8.5).

    `t_s`: bin CENTERS (matching `build_macropulse_heat_source`'s own envelope-evaluation
    convention), shape `(n_bins,)`.

    `I_useful_A` is defined EXPLICITLY as the transmitted/accepted current
    (`I_useful_A := I_transmitted_A`), per plan Sec. 8.5's own explicit warning: "with an explicit
    definition (normally transmitted/accepted current, not merely `I_emit - I_return`)". It is NOT
    `I_emit_A - I_return_A`.

    Because this is a `coupling_level="L2_one_way"` (module docstring) construction, every array
    here is exactly `(f_RF * Q_x_per_period) * envelope(t)` -- the underlying per-period rate is
    held constant across the whole macropulse; only the prescribed envelope varies it in time.
    """

    t_s: np.ndarray
    I_emit_A: np.ndarray
    I_return_A: np.ndarray
    I_transmitted_A: np.ndarray
    I_other_loss_A: np.ndarray
    I_useful_A: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.t_s, dtype=float)
        object.__setattr__(self, "t_s", t)
        for name in ("I_emit_A", "I_return_A", "I_transmitted_A", "I_other_loss_A", "I_useful_A"):
            arr = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, arr)
            if arr.shape != t.shape:
                raise ValueError(
                    f"MacropulseCurrentHistory.{name} has shape {arr.shape}, expected "
                    f"{t.shape} to match t_s"
                )


_REQUIRED_CHARGE_KEYS: tuple[str, ...] = ("emitted", "transmitted", "other_lost", "returned_after_filter")


def _get_required_charges(events: BackBombardmentEvents) -> dict[str, float]:
    charge = events.accounting.get("charge_C", {}) if isinstance(events.accounting, dict) else {}
    missing = [k for k in _REQUIRED_CHARGE_KEYS if k not in charge]
    if missing:
        raise ValueError(
            f"events.accounting['charge_C'] is missing required key(s) {missing} (have "
            f"{sorted(charge) if isinstance(charge, dict) else charge!r}); see "
            "rf_gun.back_bombardment_events.BackBombardmentEvents's class docstring for the "
            "expected accounting['charge_C'] shape."
        )
    return {k: float(charge[k]) for k in _REQUIRED_CHARGE_KEYS}


def build_macropulse_current_history(
    events: BackBombardmentEvents,
    macropulse_config: MacropulseConfig,
    t_grid_edges_s: np.ndarray,
) -> MacropulseCurrentHistory:
    """Build `MacropulseCurrentHistory` from `events.accounting`'s per-representative-period charge
    numbers and `events.rf_frequency_Hz` (`I = f_RF * Q_per_period`, plan Sec. 8.1's formula),
    scaled by the same envelope `build_macropulse_heat_source` uses, evaluated at the SAME bin
    centers (so `current_history.t_s` lines up exactly with a macropulse heat source built from the
    same `t_grid_edges_s`).

    Raises `ValueError` (naming the missing keys) if `events.accounting['charge_C']` does not carry
    all of `"emitted"`, `"transmitted"`, `"other_lost"`, `"returned_after_filter"` -- this function
    never guesses or defaults a missing charge to zero.
    """
    charges = _get_required_charges(events)
    f_RF = float(events.rf_frequency_Hz)

    I_emit_rep = f_RF * charges["emitted"]
    I_return_rep = f_RF * charges["returned_after_filter"]
    I_transmitted_rep = f_RF * charges["transmitted"]
    I_other_loss_rep = f_RF * charges["other_lost"]
    I_useful_rep = I_transmitted_rep  # explicit definition (plan Sec. 8.5), NOT I_emit - I_return

    t_grid_edges_s = np.asarray(t_grid_edges_s, dtype=float)
    t_centers_s = 0.5 * (t_grid_edges_s[:-1] + t_grid_edges_s[1:])
    envelope = evaluate_rf_envelope(t_centers_s, macropulse_config)

    print(
        "build_macropulse_current_history: f_RF="
        f"{f_RF:.6g} Hz -> I_emit={I_emit_rep:.6e} A, I_return={I_return_rep:.6e} A, "
        f"I_transmitted={I_transmitted_rep:.6e} A (=I_useful), I_other_loss={I_other_loss_rep:.6e} A "
        f"(per-representative-period rates, envelope={macropulse_config.envelope!r} applied in time)."
    )

    return MacropulseCurrentHistory(
        t_s=t_centers_s,
        I_emit_A=I_emit_rep * envelope,
        I_return_A=I_return_rep * envelope,
        I_transmitted_A=I_transmitted_rep * envelope,
        I_other_loss_A=I_other_loss_rep * envelope,
        I_useful_A=I_useful_rep * envelope,
    )


# ================================================================================================
# 5. Charge balance closure check (plan Sec. 8.5)
# ================================================================================================


def validate_charge_balance(events: BackBombardmentEvents, rtol: float = 1e-6) -> None:
    """Check `Q_emit = Q_return + Q_transmitted + Q_other_loss + Q_surviving` (plan Sec. 8.5)
    against `events.accounting['charge_C']`.

    `Q_surviving` has no dedicated key in `accounting['charge_C']` (module docstring) -- this
    function DEFINES `Q_surviving := Q_emit - (Q_return + Q_transmitted + Q_other_loss)` (using
    `returned_after_filter` for `Q_return`, the qualified/physical return) and requires it to be
    non-negative to relative tolerance `rtol` (of `Q_emit`): charge accounted for as returned,
    transmitted, or otherwise lost can never legitimately exceed what was emitted. A violation
    raises `ValueError` naming the actual numbers -- this is a real, non-tautological check (it
    fails exactly when the upstream accounting double-counts or over-counts charge), not merely a
    residual computed to force a trivial pass.

    Raises `ValueError` (a different, clearly-worded one) if `events.accounting['charge_C']` is
    missing any of the required keys.
    """
    charges = _get_required_charges(events)
    Q_emit = charges["emitted"]
    Q_return = charges["returned_after_filter"]
    Q_transmitted = charges["transmitted"]
    Q_other_loss = charges["other_lost"]

    accounted = Q_return + Q_transmitted + Q_other_loss
    Q_surviving = Q_emit - accounted
    tol = rtol * max(abs(Q_emit), 1e-30)
    if Q_surviving < -tol:
        raise ValueError(
            "Charge balance failed (plan Sec. 8.5: Q_emit = Q_return + Q_transmitted + "
            "Q_other_loss + Q_surviving, with Q_surviving required to be non-negative): "
            f"Q_emit={Q_emit:.9e} C, Q_return(after_filter)={Q_return:.9e} C, "
            f"Q_transmitted={Q_transmitted:.9e} C, Q_other_loss={Q_other_loss:.9e} C -> "
            f"accounted={accounted:.9e} C exceeds Q_emit by {-Q_surviving:.9e} C "
            f"(rtol={rtol:.1e}, tol={tol:.3e} C). This indicates double-counted or "
            "over-counted charge somewhere upstream in events.accounting['charge_C']."
        )
