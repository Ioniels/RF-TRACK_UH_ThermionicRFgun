"""RF-Track Volume helpers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple, Literal

import numpy as np

from .constants import ME_MEV, c
from .deflection_field import (
    DEFAULT_B_PK_PER_A_T,
    DEFAULT_W_MM,
    DEFAULT_Z_P_MM,
    DeflectionField,
)
from .aperture import DEFAULT_DELTA_CATHODE_CHAMFER_MM, build_dynamic_aperture


def _call_first_available(obj, names, *args):
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn(*args)
                return True
            except Exception:
                continue
    return False


def _set_first_available_attr(obj, names, value):
    for name in names:
        if hasattr(obj, name):
            try:
                setattr(obj, name, value)
                return True
            except Exception:
                continue
    return False


@dataclass(frozen=True)
class VolumeBuildParams:
    @staticmethod
    def from_dict(d: dict) -> "VolumeBuildParams":
        return VolumeBuildParams(**d)

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    f_hz: float
    map_z0_m: float
    z_min_m: float
    z_max_m: float
    hr_m: float
    hz_m: float
    dt_mm: float
    ode_algorithm: str = "rk2"
    ode_epsabs: float = 1e-10
    aperture_delta_mm: float = DEFAULT_DELTA_CATHODE_CHAMFER_MM
    t_max_mm: float = 2000.0
    fm_nsteps: int = 400
    fm_tt_nsteps: int = 200
    sc_enabled: bool = False
    sc_dt_mm: float = 1.0
    emission_nsteps: int = 1
    emission_range: float = 0.0
    cfx_dt_mm: float = 1.0
    beam_loading_enabled: bool = False
    bl_Q_loaded: float = 0.0
    bl_r_over_q_ohm_per_m: float = 0.0
    bl_ncells: int = 1
    bl_tinj_mode: str = "auto_from_emission"
    bl_tinj_manual_mm_c: float = 0.0
    beam_loading_verbose: bool = True
    deflection_enabled: bool = False
    deflection_current_A: float = 0.0
    deflection_B_pk_per_A_T: float = DEFAULT_B_PK_PER_A_T
    deflection_z_p_mm: float = DEFAULT_Z_P_MM
    deflection_w_mm: float = DEFAULT_W_MM


@dataclass(frozen=True)
class ScreenBuildParams:
    width_mm: float | None = None
    height_mm: float | None = None
    time_window_mm_c: float | None = None
    t0_mode: Literal["unset", "sync_to_first_crossing", "manual"] = "unset"
    t0_manual_mm_c: float = 0.0
    log: bool = False


def _configure_screen(S, screen_params: ScreenBuildParams, index: int, z_m: float):
    if screen_params.width_mm is not None:
        _call_first_available(S, ("set_width", "set_xwidth", "set_size_x"), float(screen_params.width_mm))

    if screen_params.height_mm is not None:
        _call_first_available(S, ("set_height", "set_ywidth", "set_size_y"), float(screen_params.height_mm))

    if screen_params.time_window_mm_c is not None:
        _call_first_available(
            S,
            ("set_time_window", "set_twindow", "set_dt", "set_time_width"),
            float(screen_params.time_window_mm_c),
        )

    mode = str(screen_params.t0_mode).strip().lower()
    if mode not in ("unset", "sync_to_first_crossing", "manual"):
        raise ValueError(f"Unknown screen t0 mode: {screen_params.t0_mode}")
    if mode == "manual":
        _call_first_available(S, ("set_t0", "set_ref_time", "set_reference_time"), float(screen_params.t0_manual_mm_c))

    if screen_params.log:
        tw = (
            "inf" if screen_params.time_window_mm_c is None else f"{float(screen_params.time_window_mm_c):.4g} mm/c"
        )
        w = "inf" if screen_params.width_mm is None else f"{float(screen_params.width_mm):.4g} mm"
        h = "inf" if screen_params.height_mm is None else f"{float(screen_params.height_mm):.4g} mm"
        print(
            f"Screen {index}: z={float(z_m):.6g} m | width={w} | height={h} | "
            f"time_window={tw} | t0_mode={mode}"
        )


def _attach_beam_loading_sw(rft, FM, p: VolumeBuildParams):
    if not p.beam_loading_enabled:
        return
    if p.bl_Q_loaded <= 0.0 or p.bl_r_over_q_ohm_per_m <= 0.0:
        raise ValueError(
            "Beam loading enabled but invalid parameters: "
            f"bl_Q_loaded={p.bl_Q_loaded}, bl_r_over_q_ohm_per_m={p.bl_r_over_q_ohm_per_m}."
        )
    if not hasattr(rft, "BeamLoadingSW"):
        raise RuntimeError("RF-Track binding has no BeamLoadingSW; cannot enable beam loading.")
    if not hasattr(FM, "add_collective_effect"):
        raise RuntimeError("RF field map has no add_collective_effect; cannot attach BeamLoadingSW.")

    omega = 2.0 * np.pi * float(p.f_hz)
    tau_s = 2.0 * float(p.bl_Q_loaded) / omega

    mode = str(p.bl_tinj_mode).strip().lower()
    if mode not in ("manual", "auto_from_emission"):
        raise ValueError(f"Unknown bl_tinj_mode: {p.bl_tinj_mode}")
    if mode == "auto_from_emission":
        tinj_mm_c = 0.0
    else:
        tinj_mm_c = float(p.bl_tinj_manual_mm_c)
    tinj_s = (tinj_mm_c * 1e-3) / c
    tinj_tau = tinj_s / tau_s if tau_s > 0.0 else 0.0

    Q_scalar = float(p.bl_Q_loaded)
    rQ_scalar = float(p.bl_r_over_q_ohm_per_m)
    Q_arr = np.array([Q_scalar], dtype=float)
    rQ_arr = np.array([rQ_scalar], dtype=float)

    ctor_errors = []
    bl_obj = None
    signatures = [
        lambda: rft.BeamLoadingSW(FM, Q_scalar, rQ_scalar, int(p.bl_ncells), float(ME_MEV), -1.0, float(tinj_tau)),
        lambda: rft.BeamLoadingSW(FM, Q_scalar, rQ_scalar, float(ME_MEV), -1.0, float(tinj_tau)),
        lambda: rft.BeamLoadingSW(FM, Q_scalar, rQ_scalar, int(p.bl_ncells), float(ME_MEV), -1.0),
        lambda: rft.BeamLoadingSW(FM, Q_scalar, rQ_scalar, float(ME_MEV), -1.0),
        lambda: rft.BeamLoadingSW(FM, Q_arr, rQ_arr, int(p.bl_ncells), float(ME_MEV), -1.0, float(tinj_tau)),
        lambda: rft.BeamLoadingSW(FM, Q_arr, rQ_arr, int(p.bl_ncells), float(ME_MEV), -1.0),
        lambda: rft.BeamLoadingSW(FM, Q_arr, rQ_arr, int(p.bl_ncells)),
    ]
    for make in signatures:
        try:
            bl_obj = make()
            break
        except Exception as exc:
            ctor_errors.append(str(exc))
    if bl_obj is None:
        msg = " | ".join(ctor_errors[:3])
        raise RuntimeError(f"Could not construct BeamLoadingSW with attempted signatures. Details: {msg}")

    FM.add_collective_effect(bl_obj)

    if p.beam_loading_verbose:
        print(
            "Beam loading ON | "
            f"Q_loaded={p.bl_Q_loaded:.6g}, ncells={int(p.bl_ncells)}, "
            f"f={p.f_hz:.6g} Hz, tau={tau_s:.4e} s, "
            f"tinj={tinj_mm_c:.4e} mm/c (tinj/tau={tinj_tau:.4e})"
        )


def _coerce_volume_params(p: VolumeBuildParams | dict) -> VolumeBuildParams:
    if isinstance(p, VolumeBuildParams):
        return p
    if isinstance(p, dict):
        return VolumeBuildParams.from_dict(p)
    raise TypeError(f"Volume params must be VolumeBuildParams or dict, got {type(p)}")


def build_volume(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    phi_deg: float,
    p: VolumeBuildParams,
    add_screens_z_m: Optional[Sequence[float]] = None,
    screen_params: Optional[ScreenBuildParams] = None,
):
    """Construct a Volume containing a single RF_FieldMap_2d and optional Screens."""
    p = _coerce_volume_params(p)

    # Constructor is RF_FieldMap_2d(Er, Ez, Bt, Bz, hr, hz, length, frequency, direction,
    # P_max, P_actual) -- Bt/Bz must be the literal 0.0 (no measured B-field), not map_z0_m.
    # The map's z-placement is handled below, via V.add(FM, ..., float(p.map_z0_m), ...).
    FM = rft.RF_FieldMap_2d(
        Er_grid,
        Ez_grid,
        0.0,
        0.0,
        float(p.hr_m),
        float(p.hz_m),
        -1,
        float(p.f_hz),
        +1,
        1.0,
        1.0,
    )

    if hasattr(FM, "set_tt_nsteps"):
        FM.set_tt_nsteps(int(p.fm_tt_nsteps))
    if hasattr(FM, "set_nsteps"):
        FM.set_nsteps(int(p.fm_nsteps))
    if hasattr(FM, "set_odeint_algorithm"):
        FM.set_odeint_algorithm(p.ode_algorithm)
    if hasattr(FM, "set_odeint_epsabs"):
        FM.set_odeint_epsabs(p.ode_epsabs)

    FM.set_phid(float(phi_deg))

    t0_set = _call_first_available(FM, ("set_t0", "set_ref_time", "set_reference_time"), 0.0)
    if not t0_set:
        _set_first_available_attr(FM, ("t0", "ref_time", "reference_time"), 0.0)

    _attach_beam_loading_sw(rft, FM, p)

    V = rft.Volume()
    V.add(FM, 0.0, 0.0, float(p.map_z0_m), "entrance")

    if add_screens_z_m:
        screen_cfg = screen_params if screen_params is not None else ScreenBuildParams()
        for index, z in enumerate(add_screens_z_m):
            S = rft.Screen()
            _configure_screen(S, screen_cfg, index=index, z_m=float(z))
            V.add(S, 0.0, 0.0, float(z), "entrance")

    V.dt_mm = float(p.dt_mm)
    if hasattr(V, "cfx_dt_mm"):
        V.cfx_dt_mm = float(p.cfx_dt_mm)
    else:
        if p.beam_loading_enabled:
            print("Warning: Volume has no cfx_dt_mm attribute; beam loading convergence may be unreliable.")
    if p.beam_loading_enabled and float(p.cfx_dt_mm) < float(p.dt_mm):
        print(
            f"Warning: cfx_dt_mm ({float(p.cfx_dt_mm):.4g}) < dt_mm ({float(p.dt_mm):.4g}); "
            "this may be unnecessarily expensive."
        )
    V.odeint_algorithm = p.ode_algorithm
    V.odeint_epsabs = float(p.ode_epsabs)
    V.set_s0(float(p.z_min_m))
    V.set_s1(float(p.z_max_m))

    # Dynamic radial aperture R(z): the cavity's real transverse channel (narrow cathode-side
    # chamfer, wide body, narrow exit transition) -- see rf_gun.aperture. Sampled on the same
    # z-grid as Er_grid/Ez_grid so the aperture and the field share one z-alignment.
    nz = int(np.asarray(Ez_grid).shape[0])
    z_grid_m = np.linspace(float(p.z_min_m), float(p.z_max_m), nz)
    dyn_aperture = build_dynamic_aperture(rft, z_grid_m, float(p.aperture_delta_mm))
    V.add(dyn_aperture, 0.0, 0.0, float(p.map_z0_m), "entrance")

    V.t_max_mm = float(p.t_max_mm)

    if p.deflection_enabled:
        if int(getattr(rft.cvar, "number_of_threads", 1)) != 1:
            print(
                "Deflection field (UserField) requires single-threaded RF-Track; "
                "forcing rft.cvar.number_of_threads = 1."
            )
            rft.cvar.number_of_threads = 1
        deflection_field = DeflectionField(
            float(p.z_max_m) - float(p.z_min_m),
            float(p.deflection_current_A),
            B_pk_per_A_T=float(p.deflection_B_pk_per_A_T),
            z_p_mm=float(p.deflection_z_p_mm),
            w_mm=float(p.deflection_w_mm),
        )
        # UserField subclasses must be added by reference: V.add() copies the
        # element, which severs the Python-side director binding and leaves the
        # original object to be garbage-collected once this function returns,
        # crashing tracking later. add_ref() keeps the live Python object wired
        # in, and the explicit attribute below is a belt-and-braces keep-alive.
        V.add_ref(deflection_field, 0.0, 0.0, float(p.z_min_m), "entrance")
        V._rf_gun_deflection_field_ref = deflection_field

    if p.sc_enabled:
        _call_first_available(
            V,
            (
                "set_sc_on",
                "enable_sc",
                "enable_space_charge",
                "set_space_charge",
            ),
            True,
        )
        _set_first_available_attr(V, ("sc_on", "sc_enabled", "sc_enable", "space_charge"), True)

        if hasattr(V, "sc_dt_mm"):
            V.sc_dt_mm = float(p.sc_dt_mm)
        if hasattr(V, "emission_nsteps"):
            V.emission_nsteps = int(p.emission_nsteps)
        if hasattr(V, "emission_range"):
            V.emission_range = float(p.emission_range)

    if p.beam_loading_enabled and p.beam_loading_verbose:
        print(f"Volume steps: dt_mm={float(p.dt_mm):.4g}, cfx_dt_mm={float(p.cfx_dt_mm):.4g}")

    return V


def track_volume_with_screens(
    rft,
    Er_grid: np.ndarray,
    Ez_grid: np.ndarray,
    phi_deg: float,
    p: VolumeBuildParams,
    B0,
    z_screens_m: Sequence[float],
    screen_params: Optional[ScreenBuildParams] = None,
    return_volume: bool = False,
):
    """Track once, capturing phase-space snapshots at `z_screens_m`."""
    z_screens_m = [float(z) for z in z_screens_m]
    V = build_volume(
        rft,
        Er_grid,
        Ez_grid,
        phi_deg,
        p,
        add_screens_z_m=z_screens_m,
        screen_params=screen_params,
    )
    Bout = V.track(B0)
    snaps = V.get_bunch_at_screens() if hasattr(V, "get_bunch_at_screens") else []
    if return_volume:
        return Bout, snaps, V
    return Bout, snaps


def find_Ez_axis_phasor_at_z0(Ez_grid: np.ndarray, z_grid_m: np.ndarray, z0_m: float = 0.0) -> complex:
    """Return on-axis Ez phasor at z~z0 (r=0 index)."""
    iz0 = int(np.argmin(np.abs(z_grid_m - z0_m)))
    return complex(Ez_grid[iz0, 0])
