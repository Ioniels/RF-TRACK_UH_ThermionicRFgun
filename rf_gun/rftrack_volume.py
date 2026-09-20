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
    #: A thin absorbing element behind z=0 recording each backward-crossing particle's exact state
    #: via RF-Track's own particle-loss table (see `rf_gun.aperture.build_cathode_backstop`) --
    #: this is what makes a returning electron get absorbed instead of unphysically continuing to
    #: bounce through the field region, and enables the validated v2 backstop_raycast_v1
    #: back-bombardment capture. On by default: this is real transport physics, not an optional
    #: diagnostic -- disable only for a deliberate A/B comparison against the pre-backstop behavior.
    cathode_backstop_enabled: bool = True
    cathode_backstop_thickness_mm: float = DEFAULT_CATHODE_BACKSTOP_THICKNESS_MM

    @property
    def applied_deflection_current_A(self) -> float:
        """The current actually applied to this run's tracking: `deflection_current_A` when the
        magnet is enabled, exactly `0.0` otherwise. Use this (never `deflection_current_A` alone)
        anywhere a figure, log line, or saved metadata field claims to show what field a run
        experienced -- `deflection_current_A` alone is only the *configured* value, which stays
        whatever it was set to even when the magnet is off."""
        return float(self.deflection_current_A) if self.deflection_enabled else 0.0


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
        if not _call_first_available(S, ("set_width", "set_xwidth", "set_size_x"), float(screen_params.width_mm)):
            print(f"Warning: screen {index} (z={z_m:.6g} m): requested width_mm could not be applied "
                  "(no matching setter on the installed RF-Track Screen binding).")

    if screen_params.height_mm is not None:
        if not _call_first_available(S, ("set_height", "set_ywidth", "set_size_y"), float(screen_params.height_mm)):
            print(f"Warning: screen {index} (z={z_m:.6g} m): requested height_mm could not be applied "
                  "(no matching setter on the installed RF-Track Screen binding).")

    if screen_params.time_window_mm_c is not None:
        if not _call_first_available(
            S,
            ("set_time_window", "set_twindow", "set_dt", "set_time_width"),
            float(screen_params.time_window_mm_c),
        ):
            print(f"Warning: screen {index} (z={z_m:.6g} m): requested time_window_mm_c could not be applied "
                  "(no matching setter on the installed RF-Track Screen binding).")

    mode = str(screen_params.t0_mode).strip().lower()
    if mode not in ("unset", "sync_to_first_crossing", "manual"):
        raise ValueError(f"Unknown screen t0 mode: {screen_params.t0_mode}")
    if mode == "manual":
        if not _call_first_available(S, ("set_t0", "set_ref_time", "set_reference_time"), float(screen_params.t0_manual_mm_c)):
            print(f"Warning: screen {index} (z={z_m:.6g} m): requested manual t0 could not be applied "
                  "(no matching setter on the installed RF-Track Screen binding).")

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


#: A best-effort (NOT guaranteed-safe) floor for BeamLoadingSW's tinj/tau, raised from an
#: originally-observed unsafe value of ~8.8e-6 -- see the note at its use site in
#: `_attach_beam_loading_sw` for why this project no longer relies on any single constant here
#: being sufficient: a real KOA production batch found this exact floor value, 1e-3, ALSO inside
#: RF-Track's own unsafe region at full production scale (n=300,000), despite validating safely
#: at a smaller scale (n=50,000) with the identical value -- the unsafe region is evidently not a
#: simple "tinj/tau below X" threshold. Kept as a first attempt (raising the odds of a clean,
#: physically un-floored construction) rather than removed, but `_attach_beam_loading_sw`'s own
#: post-construction finiteness check is the real safety net now: it degrades gracefully (skips
#: attaching beam loading, does not crash the run) if this floor still is not enough.
_MIN_SAFE_TINJ_TAU = 1.0e-3


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
    elif tinj_tau < _MIN_SAFE_TINJ_TAU:
        # A second, independent numerical floor, distinct from the tinj_tau<=0 case above: real
        # KOA production runs (n=300,000 macroparticles, rf_gun.simulation
        # ._resolve_beam_loading_tinj's own auto_from_emission path) were observed to reach this
        # point with a small but strictly positive tinj_tau (~8.8e-6, well above the 1e-9 floor
        # above) for which RF-Track 2.7.0's BeamLoadingSW constructs successfully but its own
        # get_TT1() then returns non-finite values -- confirmed empirically against real cluster
        # runs, not reproducible in an isolated unit test, so treated here as an RF-Track-internal
        # numerical edge case around very small tinj/tau rather than something fixable at its
        # source. _resolve_beam_loading_tinj was also changed (mean instead of min emission time)
        # to make reaching this region far less likely in the first place; this floor is the
        # second, independent safety net for whatever residually-small value still gets here.
        print(
            f"Note: tinj/tau={tinj_tau:.4e} is below the empirically-observed RF-Track "
            f"BeamLoadingSW numerical-stability floor ({_MIN_SAFE_TINJ_TAU:.1e}) -- raising it to "
            f"{_MIN_SAFE_TINJ_TAU:.1e} (still >=3 orders of magnitude below tau={tau_s:.4e} s, i.e. "
            "still 'the beam starts loading the cavity essentially immediately')."
        )
        tinj_tau = _MIN_SAFE_TINJ_TAU

    Q_scalar = float(p.bl_Q_loaded)
    rQ_scalar = float(p.bl_r_over_q_ohm_per_m)

    # RF-Track 2.7.0's BeamLoadingSW has exactly one constructor -- confirmed directly against the
    # installed binding (`help(RF_Track.BeamLoadingSW.__init__)` reports a single signature,
    # `__doc__` shows no SWIG overload block) and matches the 2.7 manual's documented
    # `BeamLoadingSW(SWS, Q, r_Q, Ncells, mass, q, tinj)`, with `tinj` already in units of the
    # filling-time constant `tau=2Q/omega` (computed above). Do not fall back to guessed
    # alternative signatures: a constructor call that "succeeds" with the wrong argument count for
    # this RF-Track version would silently misassign quantities (e.g. mass where Ncells belongs)
    # rather than fail -- version-gate explicitly instead of catching and retrying.
    rf_track_version = str(getattr(rft, "version", "unknown"))
    try:
        bl_obj = rft.BeamLoadingSW(FM, Q_scalar, rQ_scalar, int(p.bl_ncells), float(ME_MEV), -1.0, float(tinj_tau))
    except Exception as exc:
        raise RuntimeError(
            "Could not construct BeamLoadingSW(SWS, Q, r_Q, Ncells, mass, q, tinj) -- the sole "
            f"documented RF-Track 2.7 signature -- against the installed RF-Track version "
            f"{rf_track_version!r}. This binding may require a different signature; do not guess "
            f"one, update this call site for the detected version instead. Original error: {exc}"
        ) from exc

    # Verify the constructed element reports physically sane cavity parameters before trusting it
    # (manual A4: "Verify Lcell, tfill, tinj, TT1, and TT2 after construction").
    #
    # A non-finite value here is treated as "RF-Track's own BeamLoadingSW could not be safely
    # constructed for this run's calibrated (Q_loaded, r/Q, tinj) combination" -- skip attaching
    # the collective effect and continue the run WITHOUT beam loading, rather than crash it.
    # This is not a blanket "ignore numerical problems" policy: it is specific to BeamLoadingSW,
    # justified by two things established this session, not assumed here:
    #   1. A real KOA production A/B batch (all five KOA_slurm_scripts/study_* families) found
    #      this failure recurs at more than one tinj/tau value -- including this project's own
    #      empirically-chosen safety floor (_MIN_SAFE_TINJ_TAU) itself, at full production scale
    #      (n=300,000) though not at a smaller validation scale (n=50,000) with the same floored
    #      value -- meaning the unsafe region is not simply "tinj/tau below some threshold"; it
    #      depends on the exact (Q_loaded, r/Q, tinj/tau) combination in a way not fully
    #      characterized from outside RF-Track's own compiled implementation. No single floor
    #      constant can be trusted to avoid it in every case.
    #   2. tests/test_beam_loading_cross_validation.py independently found that RF-Track's own
    #      BeamLoadingSW already produces zero measurable effect on tracked dynamics for this gun
    #      geometry in RF-Track 2.7.0, regardless of whether it is attached at all. Skipping it
    #      when it cannot be safely constructed therefore does not trade away real physics
    #      fidelity for robustness -- for this project's gun, there is none being traded away.
    # If a future gun/RF-Track combination is confirmed to have a real (non-negligible) beam-
    # loading effect, this should become a hard failure again for that configuration -- see the
    # cross-validation test above before assuming this fallback is still appropriate elsewhere.
    beam_loading_safe = True
    unsafe_reason = ""
    for getter_name in ("get_Lcell", "get_tfill", "get_tinj", "get_TT1", "get_TT2"):
        getter = getattr(bl_obj, getter_name, None)
        if not callable(getter):
            raise RuntimeError(f"RF-Track {rf_track_version} BeamLoadingSW has no {getter_name}().")
        value = np.asarray(getter(), dtype=float)
        if not np.isfinite(value).all():
            beam_loading_safe = False
            unsafe_reason = (
                f"BeamLoadingSW.{getter_name}() returned non-finite value(s) after construction "
                f"(Q_loaded={Q_scalar}, r/Q={rQ_scalar}, ncells={int(p.bl_ncells)}, "
                f"tinj/tau={tinj_tau:.4e})"
            )
            break

    if not beam_loading_safe:
        print(
            f"Warning: {unsafe_reason} -- RF-Track's own BeamLoadingSW could not be safely "
            "constructed for this run's calibration. Continuing WITHOUT beam loading for this "
            "run (confirmed to have zero measurable effect on tracked dynamics for this gun "
            "geometry in RF-Track 2.7.0 -- see tests/test_beam_loading_cross_validation.py) "
            "rather than aborting the run."
        )
        return

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
    Bt_grid: Optional[np.ndarray] = None,
    Bz_grid: Optional[np.ndarray] = None,
):
    """Construct a Volume containing a single RF_FieldMap_2d and optional Screens.

    `Bt_grid`/`Bz_grid` are the azimuthal/longitudinal RF magnetic phasors in Tesla. Production
    artifact-backed runs supply both from the qualified XFdtd volume reduction; passing neither
    is retained only for the explicit E-only control and legacy planar-MAT compatibility path.
    All four arrays share the same `(z, r)` grid and complex-phasor convention.

    Must be supplied together or not at all: despite the manual's prose suggesting each of the
    four field components is independently array-or-zero, the actual bound `RF_FieldMap_2d`
    overload set (confirmed empirically -- `tests/test_build_volume_bfield_readiness.py`) only
    accepts the (Er, Ez) pair and the (Bt, Bz) pair each as a matched pair of both-arrays or
    both-literal-zero, never one array and one scalar within the same pair.
    """
    p = _coerce_volume_params(p)
    er_shape = np.shape(Er_grid)
    ez_shape = np.shape(Ez_grid)
    if len(er_shape) != 2 or er_shape != ez_shape:
        raise ValueError(
            "build_volume: Er_grid and Ez_grid must be matching 2D arrays "
            f"(got {er_shape} and {ez_shape})"
        )
    if (Bt_grid is None) != (Bz_grid is None):
        raise ValueError(
            "build_volume: Bt_grid and Bz_grid must be supplied together (both arrays) or both "
            "omitted (RF-Track's RF_FieldMap_2d binding has no overload for one array and one "
            "scalar within the same field-component pair)."
        )
    if Bt_grid is not None:
        bt_shape = np.shape(Bt_grid)
        bz_shape = np.shape(Bz_grid)
        if bt_shape != er_shape or bz_shape != er_shape:
            raise ValueError(
                "build_volume: Bt_grid and Bz_grid must match the electric-field grid shape "
                f"{er_shape} (got {bt_shape} and {bz_shape})"
            )

    # Constructor is RF_FieldMap_2d(Er, Ez, Bt, Bz, hr, hz, length, frequency, direction,
    # P_max, P_actual). For the explicit E-only control, Bt/Bz are the literal 0.0 pair because
    # the binding does not accept one array and one scalar. The map's z-placement is handled below, via
    # V.add(FM, ..., float(p.map_z0_m), ...) -- unrelated to Bt/Bz.
    if Bt_grid is not None:
        # RF-Track silently discards the magnetic map when every supplied phasor is purely
        # real: probing such a Volume returns B = 0 exactly, while E still reads back
        # correctly. Verified on RF-Track 2.7.0 -- giving any one of the four arrays a
        # nonzero imaginary part (even 1e-9) restores B at ratio +1.000000. A qualified
        # artifact always carries genuine phase, so this only bites synthetic/hand-built
        # maps, where it would otherwise look like a working E+B run with no RF B at all.
        _all_real = all(
            not np.any(np.asarray(g).imag != 0.0)
            for g in (Er_grid, Ez_grid, Bt_grid, Bz_grid)
        )
        _has_magnetic_field = any(np.any(np.asarray(g) != 0.0) for g in (Bt_grid, Bz_grid))
        if _all_real and _has_magnetic_field:
            raise ValueError(
                "build_volume: Er/Ez/Bt/Bz are all purely real phasors. RF-Track drops the "
                "magnetic map entirely in that case (B reads back as exactly zero) while "
                "still returning E, so the run would silently become E-only. Supply complex "
                "phasors, or pass no Bt_grid/Bz_grid for a deliberate E-only control."
            )

    # RF-Track's RF_FieldMap_2d takes its azimuthal magnetic argument along -e_theta, the
    # opposite of the artifact's standard right-handed convention. Verified empirically against
    # RF-Track 2.7.0: probing the built Volume on the +x axis (where e_theta = +y_hat) returns
    # By/Re(Btheta) = -1.000000 exactly at every (r, z, t), while Er, Ez and Bz all return
    # +1.000000. The artifact's stored sign is the physical one -- it satisfies the m=0 Faraday
    # relation dEr/dz - dEz/dr = -i*omega*Btheta: the Faraday family-normalized RMS over the
    # guarded vacuum is 0.1471 as stored and 1.2751 with Btheta negated (reproduce with
    # rf_gun.fieldmaps.quality.maxwell_residual_diagnostics).
    # Negate here, at the RF-Track boundary only, so the integrated field is a Maxwell solution.
    # Do NOT "fix" this with direction=-1: that also conjugates the E phasor.
    FM = rft.RF_FieldMap_2d(
        Er_grid,
        Ez_grid,
        -np.asarray(Bt_grid) if Bt_grid is not None else 0.0,
        Bz_grid if Bz_grid is not None else 0.0,
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
    # since adding one auto-resizes s0/s1. That ordering was tried previously and, on a real
    # production run, made the phase scan converge on a spurious crest with every particle lost
    # there (NaN mean pz). A direct A/B regression against the real field maps (RF field + dynamic
    # aperture + cathode backstop, SC/mirror/beam-loading/deflection/screens off, matching
    # tests/test_s0_s1_ordering.py) found the two orderings give bit-identical tracked outcomes
    # across a small phase scan -- so the divergence is not caused by this base configuration and
    # must involve one of the elements not exercised there (space charge, mirror, beam loading,
    # deflection UserField, or screens, alone or in combination, at production particle counts).
    # Keeping the known-safe order (BEFORE adding the dynamic aperture/backstop/deflection) until
    # someone reproduces the original divergence with the full production Volume and finds the
    # actual root cause -- this is not "unresolved because untested," it is "resolved for the
    # tested configuration, open for the rest."
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
    *,
    Bt_grid: Optional[np.ndarray] = None,
    Bz_grid: Optional[np.ndarray] = None,
):
    """Track once, capturing phase-space snapshots at `z_screens_m`.

    ``Bt_grid``/``Bz_grid`` are forwarded unchanged to :func:`build_volume`; keeping them
    keyword-only avoids confusing magnetic arrays with the pre-existing screen arguments.
    """
    z_screens_m = [float(z) for z in z_screens_m]
    V = build_volume(
        rft,
        Er_grid,
        Ez_grid,
        phi_deg,
        p,
        add_screens_z_m=z_screens_m,
        screen_params=screen_params,
        Bt_grid=Bt_grid,
        Bz_grid=Bz_grid,
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
