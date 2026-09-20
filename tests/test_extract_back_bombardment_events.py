"""Real RF-Track tests for `rf_gun.back_bombardment_events.extract_back_bombardment_events` --
Work Package 1's event-capture implementation (`BACK_BOMBARDMENT_MACROPULSE_IMPLEMENTATION_PLAN.md`
Sec. 3.1, 3.2, 3.3, 4.1, 10.2; addendum Sec. 19.2/19.6).

The main test below tracks a small, hand-built bunch through a real (but tiny/fast, all-zero
field) `RF_Track.Volume` with the cathode backstop AND the dynamic aperture both active -- exactly
`tests/test_backstop_loss_separation.py`'s pattern -- with known-by-construction outcomes: one
forward-transmitted particle, one dynamic-aperture loss, and two backstop-return particles (one
normal-incidence flat-face hit, one deliberately oblique bevel hit). `SimulationResult` is
constructed directly (not via `build_bunch_thermionic`/`run_transport_with_progress`) because
those build a full thermionic emission distribution; this test needs exact, individually chosen
trajectories with known ground truth instead, and `extract_back_bombardment_events` only ever
reads a `SimulationResult`-shaped object's public fields/RF-Track duck-typed methods (`.B0.get_phase_space`,
`.Bout.get_phase_space`, `.thermo_info`, `.lost_table`, `.M_snaps`/`.z_snaps`) -- it never calls
`build_bunch_thermionic` itself, so a hand-built object exercises the exact same code path a real
notebook/CLI run would.

Because the whole tracked field is zero everywhere (as in `test_backstop_loss_separation.py`), a
particle's entire flight is a mathematically exact straight line -- RF-Track's own numerical
stepping cannot introduce any curvature/position error, only *where along that exact line* the
backstop happens to register the loss (the small-residual-Z effect documented in
`rf_gun.backstop_loss_separation`). This lets the two backstop-return test particles' true bevel
intersection be computed independently by hand (plain trigonometry on the known launch position
and momentum), *without* calling `CathodeGeometry.intersect_ray` in the test itself, and compared
against `extract_back_bombardment_events`'s output -- a genuine independent check of the
geometrically correct impact location for a non-trivial oblique case, not a tautological
re-derivation via the function under test.

A second, small set of pure-Python tests (no RF-Track) directly exercises
`extract_back_bombardment_events`'s quality-flag/edge-case handling (missing `lost_table`, a
particle ID with no match in `B0`) using a minimal fake bunch object exposing only
`get_phase_space` -- legitimate because none of that logic depends on RF-Track itself once a
`lost_table` array already exists (only the ID-retrieval calls are RF-Track-duck-typed at all).
"""
from __future__ import annotations

import numpy as np
import pytest

import rf_gun as rg
from rf_gun.back_bombardment_events import (
    QUALITY_FLAG_ID_JOIN_FAILED,
    QUALITY_FLAG_RAY_NO_HIT,
    read_back_bombardment_events_h5,
    write_back_bombardment_events_h5,
)
from rf_gun.cathode_geometry import (
    CathodeGeometry,
    SURFACE_CATHODE_BEVEL,
    SURFACE_CATHODE_FLAT,
)
from rf_gun.constants import ME_MEV, q_e

rft = pytest.importorskip("RF_Track")

# Shared tiny/fast Volume geometry, identical to tests/test_backstop_loss_separation.py.
_NZ = 21
_Z_MIN_M = 0.0
_Z_MAX_M = 0.035
_BACKSTOP_THICKNESS_MM = 2.0
_F_HZ = 2.856e9


def _zero_field():
    Er = np.zeros((_NZ, 5), dtype=complex)
    Ez = np.zeros((_NZ, 5), dtype=complex)
    return Er, Ez


def _build_volume(dt_mm: float = 0.5):
    Er, Ez = _zero_field()
    p = rg.VolumeBuildParams(
        f_hz=_F_HZ, map_z0_m=_Z_MIN_M, z_min_m=_Z_MIN_M, z_max_m=_Z_MAX_M,
        hr_m=1e-3, hz_m=(_Z_MAX_M - _Z_MIN_M) / (_NZ - 1), dt_mm=dt_mm,
        cathode_backstop_enabled=True,
        cathode_backstop_thickness_mm=_BACKSTOP_THICKNESS_MM,
    )
    return rg.build_volume(rft, Er, Ez, 0.0, p)


def _make_row(x0, px0, z0, pz0, y0=0.0, py0=0.0, n_weight=1.0):
    """Mext row: X, Px, Y, Py, Z, Pz, MASS, Q, N, T0 (mm/MeV-c convention)."""
    return [x0, px0, y0, py0, z0, pz0, ME_MEV, -1.0, n_weight, 0.0]


@pytest.fixture(scope="module")
def tracked_case():
    """Builds and tracks the mixed-population bunch once per test module (tracking a real, if
    tiny, RF-Track Volume is comparatively expensive); every test below only reads from the result.

    Row 0: forward-transmitted (small transverse offset, pz>0, survives to Bout).
    Row 1: dynamic-aperture loss, forward (crosses the real R(z) profile well downstream --
      same validated (x0, px0/pz0 ratio) as test_backstop_loss_separation.py's Group B).
    Row 2: backstop return, normal incidence on the flat face (straight down the axis).
    Row 3: backstop return, deliberately OBLIQUE incidence on the bevel -- constructed by picking
      a target impact point on the bevel (r_hit=1.5mm, i.e. z_hit=flat_radius_mm-r_hit=-0.1mm for
      the default 45deg/1.4mm/0.2mm geometry) and a direction unit vector NOT anti-parallel to the
      bevel's inward normal, then solving backward (plain algebra, not `intersect_ray`) for a
      launch position/momentum that reaches exactly that point. See the module docstring for why
      this is a valid independent check despite reusing the same target-point idea as ground truth
      -- the *launch* state fed to RF-Track is a fresh, independently-solved-for point on the same
      line, not the target itself.
    """
    # ---- Row 3's independently-solved-for oblique bevel trajectory -------------------------
    r_hit_mm = 1.5  # within [flat_radius_mm=1.4, bevel_outer_radius_mm=1.6]
    z_hit_mm = 1.4 - r_hit_mm  # bevel eq. at 45deg: z = flat_radius_mm - r
    ux, uz = -0.5, -np.sqrt(3.0) / 2.0  # unit vector (ux^2+uz^2=1), NOT the bevel normal direction
    t_path_mm = 0.5  # arbitrary positive path length "before" the hit, along the same line
    x0_bevel = r_hit_mm - t_path_mm * ux
    z0_bevel = z_hit_mm - t_path_mm * uz
    p_mag = 2.0  # MeV/c, arbitrary overall momentum scale
    px0_bevel, pz0_bevel = p_mag * ux, p_mag * uz

    rows = [
        _make_row(0.05, 0.0, 0.0, 1.0),            # 0: transmitted
        _make_row(0.2, 0.15, 0.0, 1.0),             # 1: dynamic-aperture loss (forward)
        # 2: backstop, normal incidence, flat face. x0=1e-4mm (not exactly 0) -- the backstop's
        # own absorbing aperture radius is a tiny but nonzero r_absorb_m=1e-9m (see
        # rf_gun.aperture.build_cathode_backstop's own docstring: "not exactly 0 (untested
        # degenerate r<=0 comparison)"); an exactly-on-axis (r=0) particle was empirically found
        # here to fall exactly *inside* that tiny absorbing hole and sail through to Bout instead
        # of being absorbed -- a real, pre-existing, documented backstop limitation, not something
        # this test should paper over by avoiding r=0 silently without comment.
        _make_row(1.0e-4, 0.0, 3.0, -1.0),
        _make_row(x0_bevel, px0_bevel, z0_bevel, pz0_bevel),  # 3: backstop, oblique, bevel
    ]
    Mext = np.array(rows, dtype=float)
    n = Mext.shape[0]

    B0 = rft.Bunch6dT(Mext)
    V = _build_volume(dt_mm=0.05)  # fine step: keeps the transit-residual small (documentation only)
    Bout = V.track(B0)
    from rf_gun.diagnostics import to_lost_table_array
    lost_table = to_lost_table_array(V.get_lost_particles())

    Q_total_C = float(n) * 1.0 * q_e  # every row's N=1.0 real electron
    thermo_info = {
        "initial_phase_space": Mext[:, :6].copy(),  # x,px,y,py,z,pz -- exactly B0's own launch state
        "initial_t0_mm_c": np.zeros(n, dtype=float),
        "Q_total_C": Q_total_C,
    }
    sim_result = rg.SimulationResult(
        B0=B0, Bout=Bout, thermo_info=thermo_info,
        M_snaps=[], z_snaps=[], I_snaps=[], screen_summaries=[],
        lost_table=lost_table, particle_classes=None,
    )
    geometry = CathodeGeometry()
    capture_config = rg.BackBombardmentCaptureConfig(backstop_thickness_mm=_BACKSTOP_THICKNESS_MM)

    expected_bevel_hit_mm = (r_hit_mm, 0.0, z_hit_mm)
    expected_bevel_direction = (ux, 0.0, uz)

    return {
        "sim_result": sim_result,
        "geometry": geometry,
        "capture_config": capture_config,
        "n_rows": n,
        "expected_bevel_hit_mm": expected_bevel_hit_mm,
        "expected_bevel_direction": expected_bevel_direction,
    }


def _expected_bevel_cos_incidence(direction) -> float:
    """Independent hand-derived inward normal at the bevel (x>0, y=0): n_in = (-sin(theta), 0,
    -cos(theta)) for theta=45deg -- see rf_gun.cathode_geometry's own docstring for the same
    formula, reproduced here from scratch (not imported) as an independent check."""
    ux, uy, uz = direction
    theta = np.deg2rad(45.0)
    n_in = (-np.sin(theta), 0.0, -np.cos(theta))
    return ux * n_in[0] + uy * n_in[1] + uz * n_in[2]


# -------------------------------------------------------------------------------------------
# Main end-to-end test
# -------------------------------------------------------------------------------------------

def test_extract_back_bombardment_events_end_to_end(tracked_case):
    sim_result = tracked_case["sim_result"]
    events = rg.extract_back_bombardment_events(
        sim_result, tracked_case["geometry"], tracked_case["capture_config"],
        f_hz=_F_HZ, run_id="test_extract_back_bombardment_events",
    )

    # Exactly the two backstop-return particles (ids 2 and 3) produce events -- the transmitted
    # (id 0) and dynamic-aperture-lost (id 1) particles produce none.
    assert events.n_events == 2
    assert set(events.particle_id.tolist()) == {2, 3}
    assert np.all(events.quality_flags == 0), (
        f"expected both events to ray-cast cleanly with a matched emission ID, got "
        f"quality_flags={events.quality_flags.tolist()}"
    )

    by_id = {int(pid): i for i, pid in enumerate(events.particle_id.tolist())}

    # -- Row 2: normal-incidence flat-face hit, trivial ground truth (x0=1e-4mm, not exactly 0
    # -- see the fixture's comment on row 2 for why exactly r=0 is avoided) ------------------
    i_flat = by_id[2]
    assert int(events.surface_code[i_flat]) == int(SURFACE_CATHODE_FLAT)
    assert events.x_hit_m[i_flat] * 1e3 == pytest.approx(1.0e-4, abs=1e-9)
    assert events.y_hit_m[i_flat] == pytest.approx(0.0, abs=1e-12)
    assert events.z_hit_m[i_flat] == pytest.approx(0.0, abs=1e-9)
    assert events.cos_incidence[i_flat] == pytest.approx(1.0, abs=1e-9)
    assert bool(events.heats_lab6[i_flat])

    # -- Row 3: oblique bevel hit, independently-computed non-trivial ground truth ----------
    i_bevel = by_id[3]
    assert int(events.surface_code[i_bevel]) == int(SURFACE_CATHODE_BEVEL)
    x_hit_mm, y_hit_mm, z_hit_mm = tracked_case["expected_bevel_hit_mm"]
    assert events.x_hit_m[i_bevel] * 1e3 == pytest.approx(x_hit_mm, abs=1e-6)
    assert events.y_hit_m[i_bevel] * 1e3 == pytest.approx(y_hit_mm, abs=1e-6)
    assert events.z_hit_m[i_bevel] * 1e3 == pytest.approx(z_hit_mm, abs=1e-6)

    expected_cos = _expected_bevel_cos_incidence(tracked_case["expected_bevel_direction"])
    assert 0.0 < expected_cos < 1.0, "test construction error: expected a genuinely oblique hit"
    assert events.cos_incidence[i_bevel] == pytest.approx(expected_cos, abs=1e-6)
    assert events.incidence_angle_rad[i_bevel] == pytest.approx(np.arccos(expected_cos), abs=1e-6)
    assert bool(events.heats_lab6[i_bevel])

    # -- Emission-ID join: both events' emission state matches their own B0 launch row -------
    assert events.x_emit_m[i_flat] * 1e3 == pytest.approx(1.0e-4, abs=1e-9)
    assert events.z_emit_m[i_flat] * 1e3 == pytest.approx(3.0, abs=1e-9)

    # -- Charge/energy closure: Q_emit ~= Q_returned + Q_transmitted + Q_other_lost ----------
    acc = events.accounting
    Q_emit = acc["charge_C"]["emitted"]
    Q_closure = (
        acc["charge_C"]["returned_after_filter"]
        + acc["charge_C"]["transmitted"]
        + acc["charge_C"]["other_lost"]
    )
    assert Q_closure == pytest.approx(Q_emit, rel=1e-9)
    assert acc["counts"]["n_launched"] == tracked_case["n_rows"]
    assert acc["counts"]["n_transmitted"] == 1
    assert acc["counts"]["n_other_lost"] == 1
    assert acc["counts"]["n_returned_before_filter"] == 2
    assert acc["counts"]["n_returned_after_filter"] == 2
    assert acc["counts"]["n_id_join_failed"] == 0
    assert acc["counts"]["n_ray_no_hit"] == 0

    # -- validate() passes (also already ran in __post_init__) ------------------------------
    events.validate()


def test_hdf5_round_trip(tracked_case, tmp_path):
    sim_result = tracked_case["sim_result"]
    events = rg.extract_back_bombardment_events(
        sim_result, tracked_case["geometry"], tracked_case["capture_config"],
        f_hz=_F_HZ, run_id="test_hdf5_round_trip",
    )
    out_path = write_back_bombardment_events_h5(tmp_path / "back_bombardment_events.h5", events)
    loaded = read_back_bombardment_events_h5(out_path)

    from rf_gun.back_bombardment_events import EVENT_ARRAY_FIELDS
    for name in EVENT_ARRAY_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(loaded, name), dtype=float),
            np.asarray(getattr(events, name), dtype=float),
            rtol=1e-10, atol=1e-30, equal_nan=True,
            err_msg=f"field {name!r} did not round-trip exactly",
        )
    assert loaded.accounting == events.accounting
    assert loaded.event_locator == events.event_locator


def test_missing_lost_table_raises_value_error(tracked_case):
    sim_result = tracked_case["sim_result"]
    bad = rg.SimulationResult(
        B0=sim_result.B0, Bout=sim_result.Bout, thermo_info=sim_result.thermo_info,
        M_snaps=[], z_snaps=[], I_snaps=[], screen_summaries=[],
        lost_table=None, particle_classes=None,
    )
    with pytest.raises(ValueError, match="lost_table"):
        rg.extract_back_bombardment_events(
            bad, tracked_case["geometry"], tracked_case["capture_config"], f_hz=_F_HZ,
        )


def test_empty_lost_table_raises_value_error(tracked_case):
    sim_result = tracked_case["sim_result"]
    bad = rg.SimulationResult(
        B0=sim_result.B0, Bout=sim_result.Bout, thermo_info=sim_result.thermo_info,
        M_snaps=[], z_snaps=[], I_snaps=[], screen_summaries=[],
        lost_table=np.zeros((0, 11)), particle_classes=None,
    )
    with pytest.raises(ValueError, match="lost_table"):
        rg.extract_back_bombardment_events(
            bad, tracked_case["geometry"], tracked_case["capture_config"], f_hz=_F_HZ,
        )


# -------------------------------------------------------------------------------------------
# Pure-Python edge-case tests (no RF-Track tracking; a minimal fake bunch object is enough since
# extract_back_bombardment_events only ever calls .get_phase_space on B0/Bout).
# -------------------------------------------------------------------------------------------

class _FakeBunch:
    """Minimal duck-typed stand-in for an RF-Track bunch: only `.get_phase_space("%id", "all")`
    (and, for Bout, `"%N"`) are ever called by `extract_back_bombardment_events`."""

    def __init__(self, ids, weights=None):
        self._ids = np.asarray(ids, dtype=float)
        self._weights = np.asarray(weights, dtype=float) if weights is not None else None

    def get_phase_space(self, code, selection="all"):
        if code == "%id":
            return self._ids
        if code == "%N":
            if self._weights is None:
                raise RuntimeError("no weights configured on this fake bunch")
            return self._weights
        raise RuntimeError(f"_FakeBunch does not support code={code!r}")


def test_id_join_failure_is_flagged_not_dropped():
    """A backstop-candidate row whose particle ID has no match in B0's initial record must still
    produce an event, flagged with QUALITY_FLAG_ID_JOIN_FAILED and nan emission fields -- not be
    silently dropped."""
    geometry = CathodeGeometry()
    capture_config = rg.BackBombardmentCaptureConfig(backstop_thickness_mm=_BACKSTOP_THICKNESS_MM)

    # One backstop-candidate row (Pz<0, Z within the backstop band), id=999 -- deliberately absent
    # from B0's own id list below.
    lost_table = np.array([[0.0, 0.0, 0.0, 0.0, 0.05, -1.0, ME_MEV, -1.0, 1.0, 1.0, 999.0]])
    B0 = _FakeBunch(ids=[0.0, 1.0])  # ids 0,1 only -- 999 is unmatched
    Bout = _FakeBunch(ids=[0.0], weights=[1.0])

    thermo_info = {
        "initial_phase_space": np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        "initial_t0_mm_c": np.zeros(2),
        "Q_total_C": 2.0 * q_e,
    }
    sim_result = rg.SimulationResult(
        B0=B0, Bout=Bout, thermo_info=thermo_info,
        M_snaps=[], z_snaps=[], I_snaps=[], screen_summaries=[],
        lost_table=lost_table, particle_classes=None,
    )

    events = rg.extract_back_bombardment_events(sim_result, geometry, capture_config, f_hz=_F_HZ)
    assert events.n_events == 1
    assert int(events.quality_flags[0]) & int(QUALITY_FLAG_ID_JOIN_FAILED)
    assert np.isnan(events.x_emit_m[0])
    assert np.isnan(events.t_emit_rf_s[0])
    assert events.accounting["counts"]["n_id_join_failed"] == 1


def test_ray_no_hit_is_flagged_not_dropped():
    """A backstop-candidate ray that finds no physical surface (e.g. it lands far outside the
    modeled holder radius) must still produce an event, flagged with QUALITY_FLAG_RAY_NO_HIT."""
    geometry = CathodeGeometry()
    capture_config = rg.BackBombardmentCaptureConfig(backstop_thickness_mm=_BACKSTOP_THICKNESS_MM)

    # A row must be INSIDE the radial aperture to reach the ray-cast at all -- anything at or
    # beyond R(z) is now recognised as a cavity-wall absorption and given its own impact point
    # (see backstop_loss_separation.identify_aperture_wall_losses), so the previous 50 mm fixture
    # no longer exercises this path. Instead start well inside the aperture (r = 1 mm at
    # z = 0.05 mm, where R = 2.61 mm) with a near-transverse momentum, so the ray to the cathode
    # plane lands several mm out -- past holder_outer_radius_mm, with no surface accepting it.
    lost_table = np.array([[1.0, 100.0, 0.0, 0.0, 0.05, -1.0, ME_MEV, -1.0, 1.0, 1.0, 0.0]])
    B0 = _FakeBunch(ids=[0.0])
    Bout = _FakeBunch(ids=[], weights=[])

    thermo_info = {
        "initial_phase_space": np.array([[1.0, 100.0, 0.0, 0.0, 0.0, -1.0]]),
        "initial_t0_mm_c": np.zeros(1),
        "Q_total_C": 1.0 * q_e,
    }
    sim_result = rg.SimulationResult(
        B0=B0, Bout=Bout, thermo_info=thermo_info,
        M_snaps=[], z_snaps=[], I_snaps=[], screen_summaries=[],
        lost_table=lost_table, particle_classes=None,
    )

    events = rg.extract_back_bombardment_events(sim_result, geometry, capture_config, f_hz=_F_HZ)
    assert events.n_events == 1
    assert int(events.quality_flags[0]) & int(QUALITY_FLAG_RAY_NO_HIT)
    assert np.isnan(events.x_hit_m[0])
    from rf_gun.cathode_geometry import SURFACE_UNKNOWN
    assert int(events.surface_code[0]) == int(SURFACE_UNKNOWN)
    assert events.accounting["counts"]["n_ray_no_hit"] == 1


def test_cathode_hit_ids_is_a_property_and_selects_only_lab6(tracked_case):
    """`heats_lab6_mask` and `cathode_hit_ids` are PROPERTIES, not methods. Calling them with
    `()` raises `TypeError: 'numpy.ndarray' object is not callable` at runtime -- caught only
    when the attribute is actually touched, which no test previously did.

    `cathode_hit_ids` is the set fed to `ParticleTags.cathode_hit_ids` and thence to the
    backward_hit_cathode / backward_missed_cathode split, so it must contain exactly the LaB6
    strikes: turning around is not the same as bombarding the cathode.
    """
    events = rg.extract_back_bombardment_events(
        tracked_case["sim_result"],
        geometry=tracked_case["geometry"],
        capture_config=tracked_case["capture_config"],
        f_hz=_F_HZ,
    )

    mask = events.heats_lab6_mask          # property access, no call
    assert isinstance(mask, np.ndarray)
    hit_ids = events.cathode_hit_ids       # property access, no call
    assert isinstance(hit_ids, frozenset)

    expected = {int(pid) for pid in np.asarray(events.particle_id)[mask]}
    assert set(hit_ids) == expected
    assert hit_ids, "the tracked fixture deliberately includes flat- and bevel-hit particles"

    # Every id in the set really is on a LaB6 zone, and nothing else is.
    for i, pid in enumerate(np.asarray(events.particle_id)):
        on_lab6 = int(events.surface_code[i]) in (
            int(SURFACE_CATHODE_FLAT), int(SURFACE_CATHODE_BEVEL),
        )
        assert (int(pid) in hit_ids) == on_lab6
