"""RF-Track Volume helpers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Literal

import numpy as np

from .constants import ME_MEV, c
from .deflection_field import (
    DEFAULT_B_PK_PER_A_T,
    DEFAULT_W_MM,
    DEFAULT_Z_P_MM,
    DeflectionField,
)
from .aperture import (
    DEFAULT_DELTA_CATHODE_CHAMFER_MM,
    DEFAULT_CATHODE_BACKSTOP_THICKNESS_MM,
    build_dynamic_aperture,
    build_cathode_backstop,
)


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
    sc_nx: int = 32
    sc_ny: int = 32
    sc_nz: int = 32
    mirror_charge_enabled: bool = False
    mirror_z_m: float = 0.0
    mirror_charge_tolerance: Optional[float] = None
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
    #: Opt-in, default off: a thin absorbing element behind z=0 recording each backward-crossing
    #: particle's exact state via RF-Track's own particle-loss table (see
    #: `rf_gun.aperture.build_cathode_backstop`). Provisional -- not yet checked against a real run.
    cathode_backstop_enabled: bool = False
    cathode_backstop_thickness_mm: float = DEFAULT_CATHODE_BACKSTOP_THICKNESS_MM


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
    if tinj_tau <= 0.0:
        # RF-Track 2.7's BeamLoadingSW constructor rejects tinj/tau == 0 outright ("requires ...
        # nonzero charge" -- its validation message bundles several unrelated preconditions, but
        # isolated testing (tinj_mm_c=1e-9 vs 0.0, all else identical) confirmed the actual failing
        # check is tinj>0). Physically, "the bunch starts loading the cavity at t=0" (the emission
        # start, which is exactly what tinj_mm_c=0.0 means for auto_from_emission) is a legitimate
        # value, not an error -- so floor it to a numerically-negligible positive fraction of tau
        # rather than reinterpreting auto_from_emission's actual timing.
        tinj_tau = 1.0e-9

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


def build_space_charge_engine(rft, p: "VolumeBuildParams"):
    """Construct a fresh SpaceCharge_PIC_FreeSpace engine per the confirmed RF-Track 2.7 API
    (manual_references/RF_Track_2.5.5_reference_manual.pdf, Sec. 5.1.3/7.5): set_mirror(z_m) takes
    the cathode position in meters and set_mirror_charge_tolerance() must be applied and verified,
    never silently skipped, since it is requested physics.

    Scope: `set_mirror` models a single conducting plane at the cathode. It does not impose
    conducting electrostatic boundary conditions on the rest of the cavity (chamfer, main wall,
    exit nose, pipe 2) -- `rf_gun.aperture`'s `Aperture_1d` profile is a particle-loss geometry,
    not a Poisson boundary condition. A full conducting-wall response would need a separate
    axisymmetric Poisson or boundary-element solver.
    """
    sc = rft.SpaceCharge_PIC_FreeSpace(int(p.sc_nx), int(p.sc_ny), int(p.sc_nz))

    if p.mirror_charge_enabled:
        sc.set_mirror(float(p.mirror_z_m))

        if p.mirror_charge_tolerance is not None:
            setter = getattr(sc, "set_mirror_charge_tolerance", None)
            if not callable(setter):
                raise RuntimeError(
                    "Mirror-charge tolerance was requested, but the installed RF-Track "
                    "binding does not expose set_mirror_charge_tolerance()."
                )
            setter(float(p.mirror_charge_tolerance))

    return sc


def inspect_rftrack_capabilities(rft) -> dict:
    """Record the RF-Track binding's actual API surface for space charge/mirror controls, per
    RF_TRACK_2_7_SELF_CONSISTENT_EMISSION_IMPLEMENTATION_GUIDE.md Sec. 5.5. Never guess these --
    RF-Track 2.7 changes them across templated SpaceCharge_PIC variants without notice.
    """
    report: dict = {}
    report["rf_track_version"] = str(getattr(rft, "version", "unknown"))

    sc_cls = getattr(rft, "SpaceCharge_PIC_FreeSpace", None)
    report["SpaceCharge_PIC_FreeSpace_available"] = sc_cls is not None
    if sc_cls is not None:
        try:
            probe = rft.SpaceCharge_PIC_FreeSpace(4, 4, 4)
        except Exception as exc:
            report["SpaceCharge_PIC_FreeSpace_construction_error"] = str(exc)
            probe = None
        if probe is not None:
            report["set_mirror_available"] = callable(getattr(probe, "set_mirror", None))
            report["set_mirror_charge_tolerance_available"] = callable(
                getattr(probe, "set_mirror_charge_tolerance", None)
            )
            report["default_mirror_charge_tolerance"] = (
                float(probe.get_mirror_charge_tolerance())
                if callable(getattr(probe, "get_mirror_charge_tolerance", None))
                else None
            )
            report["compute_force_available"] = callable(getattr(probe, "compute_force", None))
            report["compute_field_available"] = callable(getattr(probe, "compute_field", None)) or callable(
                getattr(probe, "get_field", None)
            )

    report["Volume_set_sc_engine_available"] = callable(getattr(rft.Volume, "set_sc_engine", None))
    report["Volume_sc_dt_mm_available"] = hasattr(rft.Volume(), "sc_dt_mm")

    return report


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

    # NOTE: the manual (Sec. 4.1.4) recommends set_s0()/set_s1() only after adding every element,
    # since adding one auto-resizes s0/s1. That ordering was tried here and, on a real production
    # run (not synthetic), made the phase scan converge on a spurious crest with every particle
    # lost there (NaN mean pz) -- root cause not understood. Reverted to set_s0/set_s1 BEFORE
    # adding the dynamic aperture/deflection field, this function's original, validated order.
    s0_m = float(p.z_min_m)
    if p.cathode_backstop_enabled:
        # Extend s0 backward so the backstop's z<0 span is inside the Volume's active region.
        s0_m -= float(p.cathode_backstop_thickness_mm) * 1e-3
    V.set_s0(s0_m)
    V.set_s1(float(p.z_max_m))

    # Dynamic radial aperture R(z): narrow cathode-side chamfer, wide body, narrow exit -- see
    # rf_gun.aperture. Same z-grid and placement offset as the field map.
    nz = int(np.asarray(Ez_grid).shape[0])
    z_grid_m = np.linspace(float(p.z_min_m), float(p.z_max_m), nz)
    dyn_aperture = build_dynamic_aperture(rft, z_grid_m, float(p.aperture_delta_mm))
    V.add(dyn_aperture, 0.0, 0.0, float(z_grid_m[0]), "entrance")

    if p.cathode_backstop_enabled:
        backstop = build_cathode_backstop(rft, thickness_mm=float(p.cathode_backstop_thickness_mm))
        backstop_thickness_m = float(p.cathode_backstop_thickness_mm) * 1e-3
        V.add(backstop, 0.0, 0.0, float(z_grid_m[0]) - backstop_thickness_m, "entrance")

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
        # V.sc_dt_mm is the actual space-charge activation switch in RF-Track 2.7 (confirmed
        # against manual_references/RF_Track_2.5.5_reference_manual.pdf Sec. 5.1.1); it always
        # used the SC_engine current at track() time, which -- unless explicitly assigned, as
        # below -- silently fell back to RF-Track's own built-in SpaceCharge_PIC_FreeSpace(32,32,32)
        # with no cathode mirror. Assigning a per-Volume engine here makes the mesh and mirror
        # state explicit, saved, and immune to leaking from a prior run/engine in the same process.
        sc_engine = build_space_charge_engine(rft, p)
        if not callable(getattr(V, "set_sc_engine", None)):
            raise RuntimeError(
                "Space charge was requested, but the installed RF-Track binding's Volume has "
                "no set_sc_engine()."
            )
        V.set_sc_engine(sc_engine)
        V._rf_gun_sc_engine_ref = sc_engine

        if p.mirror_charge_enabled:
            mirror_state = sc_engine.get_mirror()
            if not np.isfinite(np.asarray(mirror_state, dtype=float)).all():
                raise RuntimeError(
                    "Cathode mirror charges were requested but SpaceCharge_PIC_FreeSpace.get_mirror() "
                    f"reports an unset mirror plane after set_mirror(): {mirror_state!r}."
                )

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
