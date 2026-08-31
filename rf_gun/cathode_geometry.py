"""Authoritative cathode-disk surface geometry: flat emitting face, 45deg outer bevel/chamfer,
holder placeholder, and the analytic ray/surface intersection used by back-bombardment event
capture (implementation plan Sec. 3.2, 3.3, 10.1, 10.2; addendum Sec. 19.2 confirms the geometry
below as the real gun hardware).

Scope: this module has NO RF-Track dependency. It is a pure-Python/numpy analytic geometry
library. Given already-known particle states (position + momentum, as plain numpy arrays -- e.g.
already-extracted rows of RF-Track's own `Volume.get_lost_particles()` loss table),
`CathodeGeometry.intersect_ray` finds the first physical surface a ray crosses and returns the
intersection point, surface zone, inward normal, and incidence angle. Wiring this into RF-Track's
own backstop/loss-table extraction (plan Sec. 3.2 steps 1-3) is separate, future work -- see
`rf_gun/back_bombardment.py`; nothing here touches RF-Track.

Confirmed as-built hardware geometry (plan addendum Sec. 19.2), matching the existing
`rf_gun.back_bombardment` constants exactly: a 3.2mm-diameter LaB6 disk with a 45deg x 0.2mm-wide
chamfer on its outer edge, leaving a 2.8mm-diameter (1.4mm-radius) flat emitting face. This is the
real gun hardware -- distinct from the illustrative plain 3mm-diameter x 1mm-long cylinder used by
the separate `LaB6_heating` note for its own first-principles thermal demonstration (that note's
geometry is not a source for this module's defaults; see addendum Sec. 19.2).

Coordinate convention (must match `rf_gun.back_bombardment`/`rf_gun.aperture` exactly, or the
`p_hit . n_in > 0` heating-event test of plan Sec. 3.1 silently inverts):

  - z=0 is the cathode emission surface. The tracked RF field map and the vacuum beam channel
    occupy z>=0 (`rf_gun.aperture`'s dynamic-aperture profile and the field map both start at z=0
    and extend forward/downstream). The cathode's own solid body occupies z<=0, "behind" the
    emission plane -- exactly the z<0 half-space that `rf_gun.back_bombardment` calls "behind the
    cathode" for a particle that has drifted back through z=0 (see that module's docstring), and
    where `rf_gun.aperture.build_cathode_backstop` places its absorbing band.
  - The flat face (r <= flat_radius_mm) is therefore the plane z=0, with inward normal (pointing
    from vacuum into the solid) n_in = (0, 0, -1). A returning electron re-entering the solid moves
    in -z (pz_hit < 0), so p_hit . n_in = -pz_hit > 0 for a genuine return event, while a still
    forward-going/freshly emitted electron (pz_hit > 0) gives p_hit . n_in < 0 and is correctly
    rejected. This is exactly the self-consistency plan Sec. 3.2's "forward-emitted particles are
    never counted" test exercises, and it is the *only* sign choice consistent with that
    requirement -- the opposite convention (n_in = (0,0,+1)) would flag every freshly emitted
    forward-going electron as a heating hit and reject every genuine back-bombarding one.
  - The 45deg bevel is a conical surface that recedes from z=0 (at r=flat_radius_mm) to negative z
    (deeper into the solid) as r grows toward bevel_outer_radius_mm -- i.e. the chamfered rim is cut
    away from the beam side, exactly as a real machined edge-break/countersink looks: material can
    only be *removed* by chamfering, so the cut surface cannot lie in front of (at more positive z
    than) the flat face it adjoins. Its inward normal points radially inward and toward -z (see
    `_normal_in_for_surface`); it reduces continuously to the flat face's (0,0,-1) as
    `bevel_angle_deg -> 0`.

Zones beyond the LaB6 disk (holder, cavity wall) are not yet characterized as real hardware (plan
Sec. 3.3 says only that they must be "retained separately in accounting", not that their exact
shape is known). This module's `holder_outer_radius_mm` and its associated flat-plane-at-z=0
placeholder surface are a deliberately simple stand-in -- see `CathodeGeometry`'s docstring -- to
be replaced once the holder is actually characterized (mechanical drawing or CAD import).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .back_bombardment import DEFAULT_CATHODE_CHAMFER_WIDTH_MM

# --------------------------------------------------------------------------------------------
# Surface zone codes (plan Sec. 3.3's table) -- numeric in machine-readable files/arrays, with
# human-readable labels/metadata for plots and logs.
# --------------------------------------------------------------------------------------------

SURFACE_CATHODE_FLAT = np.uint8(0)
SURFACE_CATHODE_BEVEL = np.uint8(1)
SURFACE_CATHODE_SIDE = np.uint8(2)
SURFACE_HOLDER = np.uint8(10)
SURFACE_CAVITY_WALL = np.uint8(20)
SURFACE_UNKNOWN = np.uint8(255)

#: Code -> human-readable label, matching plan Sec. 3.3's table exactly.
SURFACE_LABELS: Dict[int, str] = {
    int(SURFACE_CATHODE_FLAT): "cathode_flat",
    int(SURFACE_CATHODE_BEVEL): "cathode_bevel",
    int(SURFACE_CATHODE_SIDE): "cathode_side",
    int(SURFACE_HOLDER): "holder",
    int(SURFACE_CAVITY_WALL): "cavity_wall",
    int(SURFACE_UNKNOWN): "unknown",
}

#: Code -> {label, material_owner, plot_treatment}, transcribed verbatim from plan Sec. 3.3's
#: surface-zone table (the "Material/thermal owner" and "Plot treatment" columns).
SURFACE_ZONE_INFO: Dict[int, Dict[str, str]] = {
    int(SURFACE_CATHODE_FLAT): {
        "label": "cathode_flat",
        "material_owner": "LaB6",
        "plot_treatment": "(x,y) surface map, radius R_flat",
    },
    int(SURFACE_CATHODE_BEVEL): {
        "label": "cathode_bevel",
        "material_owner": "LaB6",
        "plot_treatment": "Unwrapped (azimuth, bevel arc length) map plus projected outline",
    },
    int(SURFACE_CATHODE_SIDE): {
        "label": "cathode_side",
        "material_owner": "LaB6 if exposed",
        "plot_treatment": "Separate accounting; normally small",
    },
    int(SURFACE_HOLDER): {
        "label": "holder",
        "material_owner": "Holder material/COMSOL domain",
        "plot_treatment": "Retained in event/accounting data, excluded from LaB6-only Python source",
    },
    int(SURFACE_CAVITY_WALL): {
        "label": "cavity_wall",
        "material_owner": "Cavity",
        "plot_treatment": "Loss accounting only",
    },
    int(SURFACE_UNKNOWN): {
        "label": "unknown",
        "material_owner": "None",
        "plot_treatment": "Causes a production warning/failure above a set fraction",
    },
}

#: Default cathode LaB6 disk thickness/depth along z [mm] -- NOT sourced from any confirmed hardware
#: drawing. Neither `rf_gun/back_bombardment.py`, `run_thermionic_tm010.py`, nor `README.md` state a
#: cathode length/thickness value (checked before choosing this). The only length value found
#: anywhere in the project's references is the *separate* `LaB6_heating` note's illustrative plain
#: 1mm-long cylinder used for its own simplified first-principles demonstration (plan addendum
#: Sec. 19.2) -- not a source for the real hardware, but a physically reasonable order-of-magnitude
#: anchor. Used only for the future volumetric-deposition depth budget (plan Sec. 3.4); needs
#: confirmation against an actual mechanical drawing before any production thermal claim.
DEFAULT_CATHODE_LENGTH_MM = 1.0

#: Default radial margin [mm] added beyond `bevel_outer_radius_mm` to get a placeholder
#: `holder_outer_radius_mm`. The real holder geometry is not yet characterized -- see
#: `CathodeGeometry`'s docstring -- this value only needs to be "clearly outside the LaB6 disk" so
#: that holder-zone impacts are retained (per plan Sec. 3.3) rather than silently merged into the
#: bevel or dropped as `SURFACE_UNKNOWN`.
DEFAULT_HOLDER_MARGIN_MM = 3.0


@dataclass(frozen=True)
class CathodeGeometry:
    """Authoritative cathode-disk surface geometry: flat emitting face, 45deg outer bevel, and a
    placeholder holder boundary, plus the analytic surface classification and ray-intersection
    methods used by back-bombardment event capture (plan Sec. 3.2/3.3).

    All lengths in mm; all angles in `*_deg` fields, degrees (converted to radians internally).

    `flat_radius_mm`/`bevel_width_mm`/`bevel_angle_deg`: the confirmed as-built geometry (plan
    addendum Sec. 19.2) -- a 2.8mm-diameter (1.4mm-radius) flat face with a 45deg x 0.2mm-wide
    chamfer, matching `rf_gun.back_bombardment.DEFAULT_CATHODE_CHAMFER_WIDTH_MM` exactly (reused,
    not reinvented).

    `cathode_length_mm`: LaB6 disk depth along z, needed for the future volumetric deposition
    budget (plan Sec. 3.4) -- see `DEFAULT_CATHODE_LENGTH_MM`'s docstring for why 1.0mm was chosen
    and why it still needs hardware confirmation.

    `insertion_offset_mm`: maps to the existing `delta_cathode_chamfer_mm` concept in
    `rf_gun.aperture`/`run_thermionic_tm010.py` (cathode insertion depth relative to the vacuum
    chamber's own dynamic-aperture chamfer -- a *different* 30deg/3.734mm chamfer, not this
    module's 45deg/0.2mm cathode-disk bevel; see this module's and `rf_gun.aperture`'s module
    docstrings for the distinction). Purely a bookkeeping/metadata field here: this module's own
    geometry is always expressed in the cathode's own z=0-at-the-emission-surface frame, so
    `insertion_offset_mm` does not enter any formula below -- it exists so a `CathodeGeometry`
    instance can carry the value alongside the disk geometry for HDF5 provenance (plan Sec. 4.1:
    "flat radius, bevel width/angle, cathode length, and insertion offset" are mandatory root
    metadata).

    `holder_outer_radius_mm`: radius beyond which an impact is attributed to the (unmodeled) holder
    rather than LaB6. The existing legacy code (`rf_gun.back_bombardment.classify_impact_surface`)
    does not make this distinction at all -- it just excludes everything past the bevel's outer
    radius as `SURFACE_EXCLUDED`. Plan Sec. 3.3 requires holder impacts to be "retained separately
    in accounting" even before the holder's exact shape is characterized, so this defaults to
    `bevel_outer_radius_mm + DEFAULT_HOLDER_MARGIN_MM` -- a deliberately simple placeholder, not a
    measured value. `intersect_ray` treats the holder as an (unphysical but harmless-for-now) flat
    plane at z=0 for `bevel_outer_radius_mm < r <= holder_outer_radius_mm`; this is a simplification
    flagged here and in `intersect_ray`'s docstring, to be replaced once real holder/mount geometry
    is available.
    """

    flat_radius_mm: float = 1.4
    bevel_width_mm: float = DEFAULT_CATHODE_CHAMFER_WIDTH_MM
    bevel_angle_deg: float = 45.0
    cathode_length_mm: float = DEFAULT_CATHODE_LENGTH_MM
    insertion_offset_mm: float = 0.0
    holder_outer_radius_mm: float | None = None

    def __post_init__(self) -> None:
        if not (self.flat_radius_mm > 0.0):
            raise ValueError(f"flat_radius_mm must be positive, got {self.flat_radius_mm!r}")
        if not (self.bevel_width_mm >= 0.0):
            raise ValueError(f"bevel_width_mm must be non-negative, got {self.bevel_width_mm!r}")
        if not (0.0 <= self.bevel_angle_deg < 90.0):
            raise ValueError(f"bevel_angle_deg must be in [0, 90), got {self.bevel_angle_deg!r}")
        if not (self.cathode_length_mm > 0.0):
            raise ValueError(f"cathode_length_mm must be positive, got {self.cathode_length_mm!r}")

        bevel_outer = self.flat_radius_mm + self.bevel_width_mm
        if self.holder_outer_radius_mm is None:
            object.__setattr__(self, "holder_outer_radius_mm", bevel_outer + DEFAULT_HOLDER_MARGIN_MM)
        elif not (self.holder_outer_radius_mm > bevel_outer):
            raise ValueError(
                f"holder_outer_radius_mm ({self.holder_outer_radius_mm!r}) must exceed "
                f"bevel_outer_radius_mm ({bevel_outer!r})"
            )

    # ------------------------------------------------------------------------------------------
    # Derived scalar properties
    # ------------------------------------------------------------------------------------------

    @property
    def bevel_angle_rad(self) -> float:
        return float(np.deg2rad(self.bevel_angle_deg))

    @property
    def bevel_outer_radius_mm(self) -> float:
        """Outer radius of the 45deg chamfer -- also the LaB6 disk's own full outer radius (the
        chamfer runs all the way from the flat face out to the disk's physical edge)."""
        return self.flat_radius_mm + self.bevel_width_mm

    @property
    def flat_area_mm2(self) -> float:
        """Projected (= true, since it is flat) area of the emitting face, `pi * r^2` [mm^2]."""
        return float(np.pi * self.flat_radius_mm**2)

    @property
    def bevel_slant_width_mm(self) -> float:
        """True slant-surface width of the chamfer (the hypotenuse of the 45deg cut), i.e. the
        radial width divided by cos(bevel_angle) -- always >= `bevel_width_mm` [mm]."""
        return self.bevel_width_mm / np.cos(self.bevel_angle_rad)

    @property
    def bevel_true_area_mm2(self) -> float:
        """True (slanted, not projected-annulus) surface area of the bevel cone [mm^2].

        Plan Sec. 3.3: "Heat flux on the bevel is energy per true bevel area, not per projected
        annulus." For a conical frustum whose generator makes angle `bevel_angle_deg` with the flat
        (r,phi) plane, the lateral surface area is the projected-annulus area
        `pi*(r_out^2 - r_in^2)` divided by `cos(bevel_angle)` (equivalently, `pi*(r_in+r_out)*slant`
        with `slant = (r_out-r_in)/cos(bevel_angle)`) -- always >= the naive projected-annulus area,
        by exactly `1/cos(bevel_angle)` (`sqrt(2)` at the confirmed 45deg default); see
        `tests/test_cathode_geometry.py` for the direct unit test plan Sec. 3.3 requires.
        """
        r_in, r_out = self.flat_radius_mm, self.bevel_outer_radius_mm
        return float(np.pi * (r_out**2 - r_in**2) / np.cos(self.bevel_angle_rad))

    # ------------------------------------------------------------------------------------------
    # Bevel surface shape
    # ------------------------------------------------------------------------------------------

    def z_of_bevel(self, r_mm: np.ndarray) -> np.ndarray:
        """z [mm] of the 45deg bevel cone at radius `r_mm`, extended algebraically for any `r_mm`
        (caller masks to the physical range `flat_radius_mm <= r_mm <= bevel_outer_radius_mm`).

        `z=0` at `r=flat_radius_mm` (smoothly continuous with the flat face), receding to negative
        z (into the solid, away from the beam) as `r` grows toward `bevel_outer_radius_mm` -- see
        the module docstring for why this sign is the physically required one (chamfering only
        removes material, so the cut surface cannot sit in front of the flat face).
        """
        r = np.asarray(r_mm, dtype=float)
        return -(r - self.flat_radius_mm) * np.tan(self.bevel_angle_rad)

    # ------------------------------------------------------------------------------------------
    # Radius-only surface classification (matches the ordering of plan Sec. 3.3's zones as far as
    # radius alone can distinguish them -- `SURFACE_CATHODE_SIDE` is a single-radius cylindrical
    # wall, not a radial band, and so cannot be produced by a radius-only classifier; it is only
    # ever assigned by `intersect_ray`, which has the full 3D ray).
    # ------------------------------------------------------------------------------------------

    def classify_surface_by_radius(self, r_mm: np.ndarray) -> np.ndarray:
        """Vectorized surface-zone classification from radius alone (no z information).

        `flat_radius_mm >= r`: `SURFACE_CATHODE_FLAT`.
        `flat_radius_mm < r <= bevel_outer_radius_mm`: `SURFACE_CATHODE_BEVEL`.
        `bevel_outer_radius_mm < r <= holder_outer_radius_mm`: `SURFACE_HOLDER`.
        Anything larger, or non-finite: `SURFACE_UNKNOWN` -- the geometry beyond the placeholder
        holder radius is entirely uncharacterized (not modeled as `SURFACE_CAVITY_WALL`, which per
        plan Sec. 3.3 is a distinct loss-accounting-only zone for genuine cavity-wall losses
        elsewhere in the gun, not a catch-all for "past the holder placeholder").

        This intentionally reproduces the same face/chamfer split as the existing
        `rf_gun.back_bombardment.classify_impact_surface` for the default geometry (see
        `tests/test_cathode_geometry.py`'s direct cross-check) -- it only adds the further
        holder/unknown distinction that legacy code collapses into a single `SURFACE_EXCLUDED`.
        """
        r = np.asarray(r_mm, dtype=float)
        code = np.full(r.shape, SURFACE_UNKNOWN, dtype=np.uint8)
        finite = np.isfinite(r)
        code[finite & (r <= self.flat_radius_mm)] = SURFACE_CATHODE_FLAT
        code[finite & (r > self.flat_radius_mm) & (r <= self.bevel_outer_radius_mm)] = SURFACE_CATHODE_BEVEL
        code[finite & (r > self.bevel_outer_radius_mm) & (r <= self.holder_outer_radius_mm)] = SURFACE_HOLDER
        return code

    # ------------------------------------------------------------------------------------------
    # Inward unit normal at a point already known to lie on a given surface zone
    # ------------------------------------------------------------------------------------------

    def _normal_in_for_surface(
        self, x_mm: np.ndarray, y_mm: np.ndarray, surface_code: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inward unit normal (vacuum -> solid), for a point already on `surface_code`'s surface.

        Flat face and the flat holder placeholder both use `(0, 0, -1)` (see module docstring).
        The bevel cone's normal follows from the implicit surface `F(x,y,z) = z +
        tan(bevel_angle)*r - tan(bevel_angle)*flat_radius_mm = 0` (`r = hypot(x,y)`): its gradient
        `(tan(theta)*x/r, tan(theta)*y/r, 1)` has magnitude `sec(theta)` and points toward
        increasing `F`, i.e. toward the vacuum side (larger z at fixed r); the *inward* normal is
        its negative, `(-sin(theta)*x/r, -sin(theta)*y/r, -cos(theta))` -- which reduces exactly to
        `(0,0,-1)` as `theta -> 0`, matching the flat face continuously at `r=flat_radius_mm`.
        """
        x_mm = np.asarray(x_mm, dtype=float)
        y_mm = np.asarray(y_mm, dtype=float)
        surface_code = np.asarray(surface_code)
        nx = np.zeros(x_mm.shape, dtype=float)
        ny = np.zeros(x_mm.shape, dtype=float)
        nz = -np.ones(x_mm.shape, dtype=float)

        is_bevel = surface_code == SURFACE_CATHODE_BEVEL
        if np.any(is_bevel):
            theta = self.bevel_angle_rad
            r = np.hypot(x_mm, y_mm)
            safe_r = np.where(r > 0.0, r, 1.0)
            bx = -np.sin(theta) * x_mm / safe_r
            by = -np.sin(theta) * y_mm / safe_r
            bz = np.full(x_mm.shape, -np.cos(theta), dtype=float)
            nx = np.where(is_bevel, bx, nx)
            ny = np.where(is_bevel, by, ny)
            nz = np.where(is_bevel, bz, nz)
        return nx, ny, nz

    # ------------------------------------------------------------------------------------------
    # Analytic ray/surface intersection (plan Sec. 3.2 step 4: "ray-cast against the parameterized
    # flat face, 45deg bevel, ... and holder; select the first physical intersection along the
    # direction of travel")
    # ------------------------------------------------------------------------------------------

    def intersect_ray(
        self,
        x0_mm: np.ndarray,
        y0_mm: np.ndarray,
        z0_mm: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        pz: np.ndarray,
    ) -> "RayIntersection":
        """First physical surface each ray crosses, moving forward along `(px,py,pz)` from
        `(x0_mm,y0_mm,z0_mm)`, vectorized over N rays with plain numpy (no Python loop over rays).

        Inputs are broadcast together (so scalars mix freely with arrays); `px,py,pz` are momentum
        components in any consistent unit -- only their *direction* matters, they need not be
        normalized or even be true momenta (a unit direction vector works identically).

        Candidate surfaces, each solved analytically and filtered before comparison:
          - the flat face (`z=0`, `r<=flat_radius_mm`);
          - the placeholder holder plane (`z=0`, `bevel_outer_radius_mm<r<=holder_outer_radius_mm`
            -- see `CathodeGeometry`'s docstring for why this is a flat-plane placeholder, not a
            claim about real holder geometry);
          - the 45deg bevel cone (`flat_radius_mm<=r<=bevel_outer_radius_mm`), solved as the
            physical root of `z + tan(bevel_angle)*r - tan(bevel_angle)*flat_radius_mm = 0` after
            eliminating the spurious (outward-facing) nappe introduced by squaring away the `r`
            square root.
        A candidate is accepted only if all hold: it lies at strictly positive path length (moving
        forward, not already sitting exactly on the surface), its radius falls in that surface's
        exact band (so e.g. a ray whose straight projection to `z=0` lands in the *unmodeled gap*
        between the flat face and the receding bevel -- `flat_radius_mm < r <= bevel_outer_radius_mm`
        at `z=0`, which is empty space, not solid, since the true bevel surface there is set back to
        negative z -- correctly finds no candidate at that plane and must instead cross the tilted
        bevel cone itself, at a different (x,y,z) than the naive flat-plane projection would
        suggest; this is exactly the plan Sec. 3.2 required test for oblique rays), and its inward
        momentum is positive (`p_hit . n_in > 0`, plan Sec. 3.1) -- this alone rejects every
        forward-going/freshly emitted ray (`pz>0` on the flat/holder plane) regardless of geometry.
        Among all accepted candidates (across all three surfaces) the one at the smallest positive
        path length wins, exactly as "select the first physical intersection along the direction of
        travel" requires.

        Returns a `RayIntersection` with `hit=False` (and all other fields `nan`/`SURFACE_UNKNOWN`)
        for a ray with no accepted candidate -- e.g. it lands beyond `holder_outer_radius_mm`, or
        it never has positive inward momentum through any modeled surface.
        """
        x0, y0, z0, px, py, pz = (np.asarray(a, dtype=float) for a in (x0_mm, y0_mm, z0_mm, px, py, pz))
        x0, y0, z0, px, py, pz = np.broadcast_arrays(x0, y0, z0, px, py, pz)
        out_shape = x0.shape
        x0 = x0.reshape(-1).copy()
        y0 = y0.reshape(-1).copy()
        z0 = z0.reshape(-1).copy()
        px = px.reshape(-1).copy()
        py = py.reshape(-1).copy()
        pz = pz.reshape(-1).copy()
        n = x0.size

        p_norm = np.sqrt(px**2 + py**2 + pz**2)
        have_dir = p_norm > 0.0
        safe_norm = np.where(have_dir, p_norm, 1.0)
        ux, uy, uz = px / safe_norm, py / safe_norm, pz / safe_norm

        eps_t = 1e-9  # mm: minimum forward path length to count (excludes t<=0/on-surface starts)
        eps_r = 1e-7  # mm: radius-band edge tolerance for numerically solved roots

        best_t = np.full(n, np.inf, dtype=float)
        best_surface = np.full(n, SURFACE_UNKNOWN, dtype=np.uint8)

        def _radius_band_ok(r: np.ndarray, surf_code: int) -> np.ndarray:
            if surf_code == SURFACE_CATHODE_FLAT:
                return r <= self.flat_radius_mm + eps_r
            if surf_code == SURFACE_CATHODE_BEVEL:
                return (r >= self.flat_radius_mm - eps_r) & (r <= self.bevel_outer_radius_mm + eps_r)
            if surf_code == SURFACE_HOLDER:
                return (r > self.bevel_outer_radius_mm + eps_r) & (r <= self.holder_outer_radius_mm + eps_r)
            return np.zeros(r.shape, dtype=bool)

        def _consider(t_cand: np.ndarray, surf_code: int) -> None:
            nonlocal best_t, best_surface
            finite_t = np.isfinite(t_cand)
            # Non-finite candidates are rejected by `valid` below regardless -- substitute a safe
            # finite stand-in (0.0) before any arithmetic so a `nan`/`inf` row never produces an
            # `inf*0`/`0/0` RuntimeWarning that would otherwise drown out real numerical issues.
            t_safe = np.where(finite_t, t_cand, 0.0)
            x_c = x0 + t_safe * ux
            y_c = y0 + t_safe * uy
            surf_arr = np.full(n, surf_code, dtype=np.uint8)
            nx, ny, nz = self._normal_in_for_surface(x_c, y_c, surf_arr)
            cos_i = ux * nx + uy * ny + uz * nz
            r_c = np.hypot(x_c, y_c)
            valid = (
                have_dir
                & finite_t
                & (t_cand > eps_t)
                & (cos_i > 0.0)
                & _radius_band_ok(r_c, surf_code)
            )
            improves = valid & (t_safe < best_t)
            best_t = np.where(improves, t_safe, best_t)
            best_surface = np.where(improves, np.uint8(surf_code), best_surface)

        # Flat face and holder placeholder: both the z=0 plane, distinguished only by radius band.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_plane = np.where(uz != 0.0, -z0 / np.where(uz != 0.0, uz, 1.0), np.nan)
        _consider(t_plane, int(SURFACE_CATHODE_FLAT))
        _consider(t_plane, int(SURFACE_HOLDER))

        # Bevel cone: z(t) + tan(theta)*(R(t) - flat_radius_mm) = 0, R(t) = hypot(x(t), y(t)) >= 0.
        # Squaring both sides to remove R(t)'s square root gives a quadratic in t with two roots;
        # only the one satisfying L(t) := z(t) - tan(theta)*flat_radius_mm <= 0 (equivalently
        # L(t) = -tan(theta)*R(t) <= 0, since tan(theta)>=0 and R(t)>=0) is the physical bevel
        # nappe -- the other root is the mirror cone on the far side of the apex and is discarded.
        if self.bevel_width_mm > 0.0 and self.bevel_angle_deg > 0.0:
            A = np.tan(self.bevel_angle_rad)
            c0 = z0 - A * self.flat_radius_mm
            a_coef = uz**2 - A**2 * (ux**2 + uy**2)
            b_coef = 2.0 * (uz * c0 - A**2 * (ux * x0 + uy * y0))
            c_coef = c0**2 - A**2 * (x0**2 + y0**2)
            for t_root in _quadratic_roots(a_coef, b_coef, c_coef):
                L = uz * t_root + c0
                t_physical = np.where(L <= eps_r, t_root, np.nan)
                _consider(t_physical, int(SURFACE_CATHODE_BEVEL))

        hit = np.isfinite(best_t) & (best_t < np.inf)
        t_final = np.where(hit, best_t, np.nan)
        x_hit = np.where(hit, x0 + t_final * ux, np.nan)
        y_hit = np.where(hit, y0 + t_final * uy, np.nan)
        z_hit = np.where(hit, z0 + t_final * uz, np.nan)
        surface_code = np.where(hit, best_surface, SURFACE_UNKNOWN).astype(np.uint8)

        nx, ny, nz = self._normal_in_for_surface(
            np.where(hit, x_hit, 0.0), np.where(hit, y_hit, 0.0), surface_code
        )
        cos_incidence = np.where(hit, ux * nx + uy * ny + uz * nz, np.nan)
        incidence_angle_rad = np.where(hit, np.arccos(np.clip(cos_incidence, -1.0, 1.0)), np.nan)
        n_in_x = np.where(hit, nx, np.nan)
        n_in_y = np.where(hit, ny, np.nan)
        n_in_z = np.where(hit, nz, np.nan)

        return RayIntersection(
            hit=hit.reshape(out_shape),
            x_hit_mm=x_hit.reshape(out_shape),
            y_hit_mm=y_hit.reshape(out_shape),
            z_hit_mm=z_hit.reshape(out_shape),
            surface_code=surface_code.reshape(out_shape),
            n_in_x=n_in_x.reshape(out_shape),
            n_in_y=n_in_y.reshape(out_shape),
            n_in_z=n_in_z.reshape(out_shape),
            cos_incidence=cos_incidence.reshape(out_shape),
            incidence_angle_rad=incidence_angle_rad.reshape(out_shape),
        )


def _quadratic_roots(a: np.ndarray, b: np.ndarray, c: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized roots of `a*t^2 + b*t + c = 0`, elementwise, falling back to the linear solution
    `t=-c/b` where `abs(a)<tol` (the "on-axis" bevel-ray case, where the quadratic's leading
    coefficient vanishes identically). `nan` where no real root exists (or, in the linear branch,
    where `b` is also ~0). The second returned array is `nan` wherever the linear branch applies
    (a single root only)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    shape = np.broadcast(a, b, c).shape
    root1 = np.full(shape, np.nan, dtype=float)
    root2 = np.full(shape, np.nan, dtype=float)

    is_linear = np.abs(a) < tol
    is_quad = ~is_linear

    disc = b**2 - 4.0 * a * c
    has_real = is_quad & (disc >= 0.0)
    sdisc = np.sqrt(np.where(has_real, disc, 0.0))
    safe_a = np.where(is_quad, np.where(a != 0.0, a, 1.0), 1.0)
    root1 = np.where(has_real, (-b - sdisc) / (2.0 * safe_a), root1)
    root2 = np.where(has_real, (-b + sdisc) / (2.0 * safe_a), root2)

    lin_has_root = is_linear & (np.abs(b) > tol)
    safe_b = np.where(lin_has_root, b, 1.0)
    root1 = np.where(lin_has_root, -c / safe_b, root1)
    # root2 stays nan in the linear branch (a single root).

    return root1, root2


@dataclass(frozen=True)
class RayIntersection:
    """Result of `CathodeGeometry.intersect_ray`, one row per input ray (same shape as the inputs).

    `hit=False` rows carry `nan` in every float field and `surface_code=SURFACE_UNKNOWN` -- index
    with `hit` before using the other fields, exactly as `rf_gun.back_bombardment.BackBombardmentData`
    already does with its own `valid` mask.
    """

    hit: np.ndarray
    x_hit_mm: np.ndarray
    y_hit_mm: np.ndarray
    z_hit_mm: np.ndarray
    surface_code: np.ndarray
    n_in_x: np.ndarray
    n_in_y: np.ndarray
    n_in_z: np.ndarray
    cos_incidence: np.ndarray
    incidence_angle_rad: np.ndarray
