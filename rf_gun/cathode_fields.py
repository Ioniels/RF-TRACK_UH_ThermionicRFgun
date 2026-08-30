"""Cathode field extraction: signed extraction fields, RF field sampling on the cathode, and
near-cathode space-charge/mirror field extraction via zero-weight probe particles.

Every function here documents explicitly (per the implementation guide's own requirement):
- coordinate frame (cathode at z=0, vacuum/cavity at +z, positions in meters unless noted);
- units;
- sign convention (E_n=E_z; extraction field F_ext=max(0,-E_n), verified in
  tests/test_field_sign_convention.py against real RF-Track tracking dynamics, not assumed);
- whether the returned field is signed or an extraction magnitude;
- whether mirror and beam loading are included.

RF-Track 2.7's SpaceCharge_PIC_FreeSpace exposes no arbitrary-point field query (confirmed via
rf_gun.rftrack_volume.inspect_rftrack_capabilities: compute_field/get_field absent) -- so the
near-cathode field is extracted via Tier B (guide Sec. 9): zero-weight probe particles appended to
a real bunch snapshot, using the same SpaceCharge_PIC_FreeSpace.compute_force() the tracking
itself uses. Validated in tests/test_cathode_fields.py: a zero-weight probe's measured force
matches a tiny-real-weight probe's to ~6 significant figures (the probe does not perturb the
charge mesh), and compute_force's returned force is confirmed (against the RF-Track 2.5.5
reference manual, Sec. 5.1.3) to be in MeV/m.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .constants import ME_MEV, c, q_e


#: compute_force's convention: for a Q=-1 (electron) probe, F[N] = Q*q_e*E[V/m] = -q_e*E, so
#: E[V/m] = -F[N]/q_e = -F[MeV/m]*1e6. Verified analytically in tests/test_cathode_fields.py
#: against the same explicit reflected-image-distribution cross-check used for mirror validation.
_PROBE_FORCE_MEVPM_TO_EZ_VPM = -1.0e6


def signed_normal_field(Ez_Vpm: np.ndarray) -> np.ndarray:
    """E_n = E_z, the signed field along the cathode outward normal (+z, into the cavity)."""
    return np.asarray(Ez_Vpm, dtype=float)


def extraction_field(Ez_Vpm: np.ndarray, beta_enh: float = 1.0) -> np.ndarray:
    """F_ext = beta_enh * max(0, -E_n): zero during the retarding half-cycle (E_n>0), verified
    sign convention (tests/test_field_sign_convention.py: a real RF-Track single-electron push at
    the dynamics-confirmed extraction phase has E_z<0)."""
    Ez = np.asarray(Ez_Vpm, dtype=float)
    return float(beta_enh) * np.maximum(-Ez, 0.0)


def sample_rf_field_on_cathode(
    rft,
    volume_or_element,
    x_grid_m: np.ndarray,
    y_grid_m: np.ndarray,
    t_grid_s: np.ndarray,
    z_probe_m: float,
) -> np.ndarray:
    """Signed E_z(x,y,t) [V/m] sampled from the *same configured RF-Track element* used for
    tracking (guide Sec. 6.4: "prefer querying the constructed RF-Track element ... rather than
    independently reconstructing a second field"), at z=z_probe_m>0 (a small distance inside the
    vacuum, not exactly at the cathode plane).

    `x_grid_m`, `y_grid_m` are a flat point cloud (equal length, one cathode location per pair) --
    not necessarily axisymmetric, since a laser-heated or backbombardment-heated cathode's
    emission profile need not be. The underlying RF field itself is currently always axisymmetric
    (RF-Track's RF_FieldMap_2d is an (r,z) representation), so this samples that same field at
    r=sqrt(x^2+y^2) for each point -- but every query goes through the real (x,y) coordinates
    rather than assuming y=0, so this is already correct if a genuinely non-axisymmetric field
    representation is ever substituted underneath.

    `volume_or_element` must expose get_field(x,y,z,t) -> (E,B), each a 3-vector, per the
    confirmed RF-Track 2.7 signature (manual_references/RF_Track_2.5.5_reference_manual.pdf
    Sec. 4.1.2/4.1.3): **x,y,z in mm, t in mm/c** (not meters/seconds -- verified against the
    manual's own worked example, not assumed), E returned in V/m.

    Returns an array of shape (len(x_grid_m), len(t_grid_s)).
    """
    x_grid_m = np.asarray(x_grid_m, dtype=float)
    y_grid_m = np.asarray(y_grid_m, dtype=float)
    if x_grid_m.shape != y_grid_m.shape:
        raise ValueError(f"x_grid_m shape {x_grid_m.shape} != y_grid_m shape {y_grid_m.shape}")
    t_grid_s = np.asarray(t_grid_s, dtype=float)
    out = np.empty((x_grid_m.size, t_grid_s.size), dtype=float)
    get_field = getattr(volume_or_element, "get_field", None)
    if not callable(get_field):
        raise RuntimeError(
            f"{type(volume_or_element).__name__} has no get_field(x,y,z,t); cannot sample the "
            "configured RF field on the cathode."
        )
    z_mm = float(z_probe_m) * 1.0e3
    for i, (x_m, y_m) in enumerate(zip(x_grid_m, y_grid_m)):
        x_mm = float(x_m) * 1.0e3
        y_mm = float(y_m) * 1.0e3
        for j, t_s in enumerate(t_grid_s):
            t_mm_c = float(t_s) * c * 1.0e3  # t is in mm/c (RF-Track convention), not seconds
            E, B = get_field(x_mm, y_mm, z_mm, t_mm_c)
            E_arr = np.asarray(E, dtype=float).ravel()
            if E_arr.size < 3:
                raise RuntimeError(f"get_field returned E with {E_arr.size} components; expected 3.")
            out[i, j] = float(E_arr[2])
    return out


def inspect_rftrack_field_capabilities(rft) -> Dict[str, Any]:
    """Capability report for arbitrary-point field queries (guide Sec. 9), extending
    rf_gun.rftrack_volume.inspect_rftrack_capabilities with the element-level get_field check."""
    report: Dict[str, Any] = {}
    try:
        probe_sc = rft.SpaceCharge_PIC_FreeSpace(4, 4, 4)
        report["tier_A_sc_compute_field_available"] = callable(getattr(probe_sc, "compute_field", None))
        report["tier_A_sc_get_field_available"] = callable(getattr(probe_sc, "get_field", None))
        report["tier_B_sc_compute_force_available"] = callable(getattr(probe_sc, "compute_force", None))
    except Exception as exc:
        report["tier_A_B_probe_error"] = str(exc)

    try:
        Er = np.zeros((5, 5), dtype=complex)
        Ez = np.zeros((5, 5), dtype=complex)
        FM = rft.RF_FieldMap_2d(Er, Ez, 0.0, 0.0, 1e-3, 1e-3, -1, 3e9, 1, 1.0, 1.0)
        report["element_get_field_available"] = callable(getattr(FM, "get_field", None))
        report["element_get_field_complex_available"] = callable(getattr(FM, "get_field_complex", None))
    except Exception as exc:
        report["element_get_field_error"] = str(exc)

    report["preferred_tier"] = (
        "A" if report.get("tier_A_sc_compute_field_available") else
        "B" if report.get("tier_B_sc_compute_force_available") else
        "C/D (neither Tier A nor Tier B available on this RF-Track build)"
    )
    return report


def extract_sc_field_with_probes(
    rft,
    sc_engine,
    active_bunch_matrix: np.ndarray,
    probe_x_m: np.ndarray,
    probe_y_m: np.ndarray,
    probe_z_m: float,
    probe_mass_MeV: float = ME_MEV,
) -> np.ndarray:
    """Tier B (guide Sec. 9): signed E_z [V/m] from space charge (+ mirror, if `sc_engine` has a
    mirror configured) at a cloud of (x,y) probe points on the plane z=probe_z_m, using
    zero-weight probe particles appended to the real active-bunch snapshot and the *same*
    `sc_engine` the tracking Volume uses. `probe_x_m`/`probe_y_m` need not be axisymmetric.

    `active_bunch_matrix` must be the extended 10-column [X,Px,Y,Py,Z,Pz,MASS,Q,N,T0] format (mm,
    MeV/c) already used by rf_gun.simulation.build_bunch_thermionic, describing the currently
    active (already-emitted) macroparticles.

    Probe convention: charge sign Q=-1 (electron), weight N=0 (verified in
    tests/test_cathode_fields.py: N=0 and N=1 probes at the same location agree to ~6 significant
    figures, confirming zero-weight probes measure the ambient field without perturbing the mesh).
    compute_force's return is in MeV/m (confirmed against the RF-Track 2.5.5 reference manual);
    for a Q=-1 probe, E_z = -F_z[MeV/m]*1e6 (force = charge*field, charge=-1 in probe-charge
    units).
    """
    probe_x_m = np.asarray(probe_x_m, dtype=float)
    probe_y_m = np.asarray(probe_y_m, dtype=float)
    if probe_x_m.shape != probe_y_m.shape:
        raise ValueError(f"probe_x_m shape {probe_x_m.shape} != probe_y_m shape {probe_y_m.shape}")
    n_probes = probe_x_m.size
    probe_z_mm = float(probe_z_m) * 1.0e3
    probe_rows = np.column_stack([
        probe_x_m * 1.0e3,               # X [mm]
        np.zeros(n_probes),              # Px
        probe_y_m * 1.0e3,               # Y [mm]
        np.zeros(n_probes),              # Py
        np.full(n_probes, probe_z_mm),   # Z [mm]
        np.zeros(n_probes),              # Pz
        np.full(n_probes, probe_mass_MeV),
        np.full(n_probes, -1.0),         # Q
        np.zeros(n_probes),              # N=0 -- zero-weight probe
        np.zeros(n_probes),              # T0
    ])
    combined = np.vstack([np.asarray(active_bunch_matrix, dtype=float), probe_rows])
    B = rft.Bunch6dT(combined)
    F = np.asarray(sc_engine.compute_force(B))
    F_probes_z_MeVpm = F[-n_probes:, 2]
    return F_probes_z_MeVpm * _PROBE_FORCE_MEVPM_TO_EZ_VPM


def extract_sc_and_mirror_from_snapshot(
    rft,
    active_bunch_matrix: np.ndarray,
    probe_x_m: np.ndarray,
    probe_y_m: np.ndarray,
    probe_z_m: float,
    sc_nx: int,
    sc_ny: int,
    sc_nz: int,
    mirror_z_m: Optional[float] = None,
    mirror_charge_tolerance: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper (guide Sec. 9): builds a *fresh* SpaceCharge_PIC_FreeSpace engine
    (mirror-off) and, if `mirror_z_m` is given, a second fresh engine (mirror-on), and returns
    (E_sc_free, E_sc_plus_mirror) at the probe cloud via extract_sc_field_with_probes -- so the
    pure mirror contribution can be isolated as E_mirror = E_sc_plus_mirror - E_sc_free (guide
    Sec. 9, Tier C note), without ever reusing one engine's state across the two evaluations
    (guide Sec. 5.3: "Create a fresh engine for every run").
    """
    sc_free = rft.SpaceCharge_PIC_FreeSpace(int(sc_nx), int(sc_ny), int(sc_nz))
    E_sc_free = extract_sc_field_with_probes(rft, sc_free, active_bunch_matrix, probe_x_m, probe_y_m, probe_z_m)

    if mirror_z_m is None:
        return E_sc_free, E_sc_free.copy()

    sc_mirror = rft.SpaceCharge_PIC_FreeSpace(int(sc_nx), int(sc_ny), int(sc_nz))
    sc_mirror.set_mirror(float(mirror_z_m))
    if mirror_charge_tolerance is not None:
        sc_mirror.set_mirror_charge_tolerance(float(mirror_charge_tolerance))
    E_sc_plus_mirror = extract_sc_field_with_probes(rft, sc_mirror, active_bunch_matrix, probe_x_m, probe_y_m, probe_z_m)
    return E_sc_free, E_sc_plus_mirror


@dataclass
class BeamLoadingFieldStatus:
    """guide Sec. 11.2: whether/why a self-consistent beam-induced cathode field is available."""
    available: bool
    reason: str
    diagnostics: Dict[str, Any] = dataclass_field(default_factory=dict)


def extract_beam_loading_field(bl_obj) -> BeamLoadingFieldStatus:
    """Inspect a BeamLoadingSW collective-effect object for the information needed to reconstruct
    its beam-induced field at the cathode (guide Sec. 11.2: complex mode amplitude history,
    beam-induced voltage/gradient history, or a field query at arbitrary time).

    Confirmed API surface (RF-Track 2.7.0, this project's venv): BeamLoadingSW exposes only
    get_Lcell/get_tfill/get_tinj/get_TT1/get_TT2/compute_force -- none of the mode-amplitude,
    beam-induced-voltage, or arbitrary-time field query methods guide Sec. 11.2 asks for. This
    is not just an unexplored API surface -- two direct experiments (not present in this
    function's own logic, since neither is usable, but recorded here so this isn't re-derived):

    - `Volume.get_field(x,y,z,t)` / `RF_FieldMap_2d.get_field(...)` (the call
      `sample_rf_field_on_cathode` already uses for the RF-only field) is a static field-map
      query: tracking a real, heavily-charged bunch through a Volume with a live BeamLoadingSW
      attached and re-querying `get_field()` at the same point/time produced *zero* change --
      it never consults the collective effect's accumulated state at all.
    - `BeamLoadingSW.compute_force(...)` -- unlike `SpaceCharge_PIC_FreeSpace.compute_force(bunch)`
      (used for the SC/mirror probe trick) -- requires a C++ `MatrixNd&` output parameter with no
      Python binding (`rft.MatrixNd` does not exist); every argument combination tried raises a
      SWIG "wrong number or type of arguments" error. It is listed by `dir()` but not actually
      callable from Python.
    - The RF-Track reference manual (Sec. 5.5.1 vs 5.5.2) confirms this is architectural, not an
      oversight: the richer introspection this guide section wants (`get_G()`/`get_G_steady()`,
      transient/steady beam-induced gradient) exists only for the *traveling-wave* `BeamLoading()`
      collective effect, not the *standing-wave* `BeamLoadingSW()` this TM010 cavity uses.

    None of this affects the real production transport -- `BeamLoadingSW` is attached to and
    genuinely drives that Volume's own tracking via `V.track()` (see `rftrack_volume.py`'s
    `_attach_beam_loading_sw`); only this reduced-cost pre-estimate (`rf_gun.emission_iteration`),
    which needs a field value at arbitrary times without doing full tracking, cannot fold it in.
    Per the guide's own instruction ("If the binding does not expose the required information,
    document that limitation and defer E_BL feedback rather than infer it from final particle
    trajectories"), this returns available=False rather than attempting an unsupported
    reconstruction.
    """
    candidates = (
        "get_mode_amplitude", "get_induced_voltage", "get_induced_field",
        "get_field", "compute_field", "get_beam_induced_amplitude",
    )
    found = [name for name in candidates if callable(getattr(bl_obj, name, None))]
    if found:
        return BeamLoadingFieldStatus(
            available=True,
            reason=f"BeamLoadingSW exposes {found}; reconstruction not yet implemented.",
            diagnostics={"available_methods": found},
        )
    return BeamLoadingFieldStatus(
        available=False,
        reason=(
            "This RF-Track 2.7 BeamLoadingSW exposes only get_Lcell/get_tfill/get_tinj/get_TT1/"
            "get_TT2/compute_force -- no mode-amplitude, induced-voltage, or arbitrary-time field "
            "query. E_BL feedback is deferred (guide Sec. 11.2) rather than inferred indirectly."
        ),
        diagnostics={"available_methods": []},
    )
