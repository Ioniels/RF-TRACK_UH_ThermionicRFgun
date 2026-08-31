"""Resolves the Work Package 1 blocking question in
`BACK_BOMBARDMENT_MACROPULSE_IMPLEMENTATION_PLAN.md` Sec. 3.2: can backstop-loss rows (the thin
absorbing `Aperture_1d` immediately behind the cathode plane, `rf_gun.aperture.build_cathode_backstop`)
be reliably separated from dynamic-aperture-loss rows (`rf_gun.aperture.build_dynamic_aperture`) in
RF-Track's single combined `Volume.get_lost_particles()` table?

Verdict: YES, separable -- but NOT by the plan's own literal Sec. 3.2 step-3 wording ("the
negative-z backstop band, Pz < 0"). That wording assumes a genuine backstop hit is recorded with
Z <= 0. Empirically (see `tests/test_backstop_loss_separation.py`, which reproduces every claim
below against a real, zero-field `RF_Track.Volume`) that is only true for a particle whose
*initial* state already lies at or past the z=0 boundary. A particle that must actually travel
(the physically dominant case: a real electron is emitted forward, decelerated/turned around by
the RF+space-charge field somewhere downstream, and travels back to z=0) is instead recorded by
`Volume.get_lost_particles()` at a small **positive** Z -- the last state RF-Track's own internal
event/step handling resolved before the crossing, not the crossing itself. Concretely, sweeping
`dt_mm` in [0.001, 5] mm/c, `Pz` in [-50, -0.0001] MeV/c, backstop `thickness_mm` in [0.5, 4], and
transit distance in [2, 30] mm (all with the field map and dynamic-aperture radius held at their
project defaults, `aperture_delta_mm=0`, cathode-frame `z=0`), the observed positive-side residual
ranged up to ~0.63 mm and did **not** monotonically shrink as `dt_mm` was refined -- i.e. it does
not "converge with dt_mm" in the sense the plan asks the eventual `backstop_raycast_v1` method to
demonstrate. This is treated here as an empirically-bounded, non-vanishing artifact of RF-Track's
own (undocumented, C++-internal) loss-detection bookkeeping, not something fixable from the Python
binding -- see the module-level caveat below before reusing the default slack in a different
tracking configuration.

Despite that, separation is still exact in every configuration tested here, because:

  1. Pz sign alone is NOT sufficient (a real dynamic-aperture transverse loss can also happen with
     Pz < 0 -- e.g. a particle that reversed direction upstream and is *already* transversely
     outside the real channel R(z) well downstream, at z >> 0, when the aperture check next runs;
     `test_ambiguous_pz_negative_dynamic_aperture_loss_is_not_misclassified` reproduces exactly
     this case). The Z value, not the Pz sign, is what discriminates it -- its reported Z sits at
     z >> 0 (tens of mm in the test cavity), nowhere near the backstop band + slack.
  2. Z alone is NOT sufficient either close to z=0 (a legitimate dynamic-aperture loss occurring
     immediately at emission, Pz > 0, transverse position already beyond the chamfer-start radius,
     is reported at Z==0 too -- distinguished from a backstop hit only by Pz's sign).
  3. Combined, `Pz < 0` AND `Z` within `[backstop_z_min_m, backstop_z_max_m + z_slack_m]` correctly
     classified every row in every test run here, with the slack sized comfortably above the
     largest observed residual (see `DEFAULT_Z_SLACK_M`) yet comfortably below the smallest z at
     which a *real* near-cathode dynamic-aperture violation is geometrically possible (the dynamic
     aperture's own radius profile is R1_MM=2.5275 mm at s<=0 and only grows moving into z>0 via the
     chamfer, so a genuine transverse violation within a fraction of a mm of z=0 would require a
     transverse excursion much larger than this project's ~1.4-1.6 mm cathode radius -- plausible
     only for the already-separately-flagged pathological/numerical-blowup rows described in
     `rf_gun.particle_tags.MAX_PHYSICAL_KINETIC_ENERGY_MEV`'s docstring).
  4. No row's particle ID was ever seen both lost and in `Bout`, and no ID appeared twice in the
     loss table, in any test here -- the plan's "one-return-event-per-ID" expectation held without
     needing to be separately enforced.

Caveat / when to re-validate: `DEFAULT_Z_SLACK_M` is an empirically-calibrated constant for this
project's cavity geometry, backstop thickness range, and momentum range -- NOT a physical constant
and NOT derived from first principles (RF-Track does not document the internal mechanism producing
the residual, so it could not be derived here). Re-run this module's validation methodology
(`tests/test_backstop_loss_separation.py`) -- or at minimum re-check the worst-case residual with
`_max_observed_backstop_residual_mm`-style sweeps -- before trusting the default across a
materially different `dt_mm`, `cathode_backstop_thickness_mm`, expected particle momentum range, or
`aperture_delta_mm` (a nonzero chamfer offset moves the dynamic aperture's near-z=0 radius profile
and could shrink the safety margin between the slack and a genuine chamfer violation).
"""
from __future__ import annotations

import numpy as np

#: Empirically-calibrated slack (see module docstring) added on the positive side of the cathode
#: plane (`backstop_z_max_m`, default 0.0) when classifying a loss-table row as a backstop hit.
#: Roughly 2.4x the largest positive-side residual observed in this project's validation sweep
#: (~0.63 mm, `dt_mm` in [0.001, 5] mm/c, |Pz| in [0.0001, 50] MeV/c, transit distance in [2, 30]
#: mm, backstop thickness in [0.5, 4] mm) -- see the module docstring's caveat before reusing this
#: outside that regime.
DEFAULT_Z_SLACK_M = 1.5e-3


def identify_backstop_loss_candidates(
    lost_table: np.ndarray,
    *,
    id_col: int = 10,
    z_col: int = 4,
    pz_col: int = 5,
    backstop_z_min_m: float,
    backstop_z_max_m: float = 0.0,
    z_slack_m: float = DEFAULT_Z_SLACK_M,
) -> np.ndarray:
    """Boolean mask into `lost_table`'s rows: `True` for a backstop (back-bombardment) candidate,
    `False` for a dynamic-aperture (ordinary transverse) loss.

    `lost_table` is RF-Track's own `Volume.get_lost_particles()` output, already normalized to an
    `(n, 11)` array of `[X, Px, Y, Py, Z, Pz, T, MASS, Q, N, ID]` (mm/MeV-c/mm-c convention -- see
    `rf_gun.diagnostics.to_lost_table_array`; `LOST_TABLE_ID_COL=-1`/`id_col=10` and `z_col=4`,
    `pz_col=5` are this table's own default column positions, matching `rf_gun.io.LOST_COLUMNS`).

    `backstop_z_min_m`/`backstop_z_max_m` are the backstop element's own global z span in METERS,
    cathode-frame (`backstop_z_max_m` defaults to 0.0, the cathode plane -- matching
    `rf_gun.aperture.build_cathode_backstop`'s placement convention `[z0_global -
    thickness_mm*1e-3, z0_global]`; pass `backstop_z_min_m = -thickness_mm*1e-3` for the project's
    default `z0_global=0` cathode-frame convention). Internally compared against `lost_table`'s
    `Z` column, which is in mm, not m -- converted here, not by the caller.

    `z_slack_m` (see `DEFAULT_Z_SLACK_M` and the module docstring) widens the upper (positive-z)
    edge of the backstop band to absorb RF-Track's own non-vanishing positive-side residual for a
    genuinely-backstop-absorbed particle that had to travel to reach z=0, rather than starting
    already inside the band. Do not set this to 0.0 without re-reading the module docstring --
    doing so silently reproduces the plan's original (empirically false) "Z<=0 exactly" assumption
    and will misclassify most real (transit) backstop hits as unclassified/dynamic-aperture.

    The classification rule is `Pz < 0` (moving toward the cathode/backstop, strictly -- Pz==0 is
    not a meaningful backward crossing) AND `Z` within `[backstop_z_min_m, backstop_z_max_m +
    z_slack_m]` (converted to mm) AND a finite, non-negative particle ID (`id_col`'s value; a
    negative/non-finite ID marks a row with no usable identity, see
    `rf_gun.particle_tags._ids_of`'s sentinel convention). Neither `Pz<0` nor the Z-band alone is
    sufficient -- see the module docstring's items 1-2 for the specific ambiguous rows this
    combined rule resolves that neither alone would.

    Returns an all-`False` mask of the correct length for an empty/`None`/malformed `lost_table`
    (mirrors `rf_gun.particle_tags`'s "shaped, not raising" convention for downstream boolean
    masking).
    """
    if lost_table is None:
        return np.zeros((0,), dtype=bool)
    arr = np.asarray(lost_table, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    n = arr.shape[0]
    max_col = max(id_col if id_col >= 0 else -id_col - 1, z_col, pz_col)
    if arr.shape[1] <= max_col:
        return np.zeros((n,), dtype=bool)

    z_mm = arr[:, z_col]
    pz = arr[:, pz_col]
    ids = arr[:, id_col]

    z_band_min_mm = float(backstop_z_min_m) * 1e3
    z_band_max_mm = (float(backstop_z_max_m) + float(z_slack_m)) * 1e3

    is_backward = np.isfinite(pz) & (pz < 0.0)
    in_band = np.isfinite(z_mm) & (z_mm >= z_band_min_mm) & (z_mm <= z_band_max_mm)
    valid_id = np.isfinite(ids) & (ids >= 0.0)

    return is_backward & in_band & valid_id
