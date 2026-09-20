"""Diagnostics and summary helpers shared across simulation and plotting."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .particle_tags import K_COL, MAX_PHYSICAL_KINETIC_ENERGY_MEV


def _second_moment_twiss(u: np.ndarray, pu: np.ndarray):
    """Shared second-moment computation: returns (alpha, beta, gamma, eps_geom).

    `eps_geom = sqrt(det(cov{u,pu}))` is the geometric emittance in whatever units
    `u`/`pu` are given in (e.g. mm and rad -> mm*rad). `manual_twiss_and_emittance` (below) is
    this function's only caller, added to expose `eps_geom` for the native-vs-manual comparison.
    """
    if u.size < 2 or pu.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    u0 = u - np.mean(u)
    pu0 = pu - np.mean(pu)
    s11 = float(np.mean(u0 * u0))
    s22 = float(np.mean(pu0 * pu0))
    s12 = float(np.mean(u0 * pu0))
    det = s11 * s22 - s12 * s12
    if not np.isfinite(det) or det <= 0.0:
        return np.nan, np.nan, np.nan, np.nan
    eps = np.sqrt(det)
    alpha = -s12 / eps
    beta = s11 / eps
    gamma = s22 / eps
    return float(alpha), float(beta), float(gamma), float(eps)


def dispersion_from_moments(u: np.ndarray, delta: np.ndarray) -> float:
    """D_u = Cov(u, delta) / Var(delta), mean-centered and population-normalized (ddof=0) to
    match `_second_moment_twiss`'s convention exactly. `u` can be a position (-> D_x [mm]) or a
    divergence `px/pz` (-> D_x' [rad]); `delta` is the relative momentum deviation
    `(pz - mean(pz)) / mean(pz)`.
    """
    if u.size < 2 or delta.size < 2:
        return np.nan
    u0 = u - np.mean(u)
    d0 = delta - np.mean(delta)
    var_d = float(np.mean(d0 * d0))
    if not np.isfinite(var_d) or var_d <= 0.0:
        return np.nan
    return float(np.mean(u0 * d0) / var_d)


def manual_twiss_and_emittance(M: np.ndarray, mass_MeV: float) -> Dict[str, float]:
    """Twiss/emittance for [x,px,y,py,z,pz] via plain numpy second moments.

    This is the project's sole Twiss/emittance computation. RF-Track's own native
    `Bunch6dT.get_info()` Twiss (`alpha_x`/`alpha_y`) was empirically found to be
    unreliable under real (non-negligible) x-x'/y-y' correlation -- reproduced
    against both RF-Track 2.5.4 and 2.6.3, so not a version-specific defect -- and
    a `Screen`'s own `get_info()` returns an internal `Bunch6d` (not `Bunch6dT`)
    object whose field names collide with this project's `Bunch6dT`-style lookups
    (e.g. its `sigma_px`/`sigma_py` are mrad angle spreads, not MeV/c momentum
    spreads, but a case-insensitive field lookup for "sigma_Px" matches them
    anyway). Neither RF-Track-native path is used here; see the notebook's
    beam-parameter cell and `UPGRADE_PLAN_notebook_and_architecture.md` for the
    full empirical writeup.

    `beta_x`/`beta_y` in mm, `emitt_x`/`emitt_y` (and `_norm` aliases) normalized in mm*mrad (via
    the paraxial beta*gamma = mean_pz / mass_MeV relation); `emitt_x_geom`/`emitt_y_geom` are the
    same quantity before that scaling (mm*rad). `gamma_x`/`gamma_y`/`gamma_z` = (1+alpha^2)/beta,
    from the same second-moment computation as alpha/beta (no separate formula/pass needed).

    Longitudinal (`alpha_z`/`beta_z`/`emitt_z`) uses this project's own convention
    -- `z` [mm] vs. `delta = pz / mean(pz)` [dimensionless] -- since RF-Track's own
    manual documents a different, incompletely-specified longitudinal convention
    (`emitt_z` in mm.permille, no closed-form alpha_z/beta_z formula given) that
    does not reduce to a simple unit rescaling of (z, delta). `emitt_z` is already the
    geometric value (z, delta) itself has no natural "normalized" counterpart here.
    """
    out_keys = [
        "alpha_x", "beta_x", "gamma_x", "emitt_x", "emitt_x_norm", "emitt_x_geom",
        "alpha_y", "beta_y", "gamma_y", "emitt_y", "emitt_y_norm", "emitt_y_geom",
        "alpha_z", "beta_z", "gamma_z", "emitt_z",
    ]
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] < 6:
        return {k: np.nan for k in out_keys}

    x, px, y, py, z, pz = (arr[:, i] for i in range(6))
    pz_mean = float(np.mean(pz))
    p0 = pz_mean if pz_mean > 0.0 else 1.0
    betagamma = pz_mean / float(mass_MeV) if mass_MeV > 0.0 else np.nan

    ax, bx, gx, ex = _second_moment_twiss(x, px / pz)
    ay, by, gy, ey = _second_moment_twiss(y, py / pz)
    az, bz, gz, ez = _second_moment_twiss(z, pz / p0)

    ex_norm = ex * betagamma * 1e3 if np.isfinite(ex) else np.nan
    ey_norm = ey * betagamma * 1e3 if np.isfinite(ey) else np.nan

    return {
        "alpha_x": ax, "beta_x": bx, "gamma_x": gx,
        "emitt_x": ex_norm, "emitt_x_norm": ex_norm, "emitt_x_geom": ex,
        "alpha_y": ay, "beta_y": by, "gamma_y": gy,
        "emitt_y": ey_norm, "emitt_y_norm": ey_norm, "emitt_y_geom": ey,
        "alpha_z": az, "beta_z": bz, "gamma_z": gz, "emitt_z": ez,
    }


def info_get(info: Any, key: str):
    if info is None:
        return np.nan
    if isinstance(info, dict):
        if key in info:
            return info[key]
        if key.lower() in info:
            return info[key.lower()]
        if key.upper() in info:
            return info[key.upper()]
        return np.nan
    if hasattr(info, key):
        val = getattr(info, key)
        return val() if callable(val) else val
    if hasattr(info, key.lower()):
        val = getattr(info, key.lower())
        return val() if callable(val) else val
    if hasattr(info, key.upper()):
        val = getattr(info, key.upper())
        return val() if callable(val) else val
    if hasattr(info, f"get_{key}"):
        getter = getattr(info, f"get_{key}")
        return getter() if callable(getter) else getter
    return np.nan


def info_get_first(info: Any, keys: Sequence[str]):
    for key in keys:
        val = info_get(info, key)
        try:
            fval = float(val)
        except Exception:
            continue
        if np.isfinite(fval):
            return fval
    return np.nan


def summarize_array(values: np.ndarray, *, with_span: bool = False) -> Dict[str, Any]:
    """Return robust numeric summary, preserving count and finite_count.

    `with_span` adds a `"span"` field (`max - min` over the finite values) -- useful for e.g. an
    emission-time window, where the width itself (not just min/max separately) is the quantity of
    interest.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    out: Dict[str, Any] = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if with_span:
        out["span"] = None
    if finite.size == 0:
        return out
    out["min"] = float(np.min(finite))
    out["max"] = float(np.max(finite))
    out["mean"] = float(np.mean(finite))
    out["std"] = float(np.std(finite))
    if with_span:
        out["span"] = float(np.max(finite) - np.min(finite))
    return out


def build_screen_summary_from_phase_space(
    M_screen: np.ndarray | None,
    screen_index: int,
    z_m: float,
    n_initial: int,
    n_previous: int | None = None,
) -> Dict[str, Any]:
    """Build robust per-screen summary from explicit phase-space array only."""
    arr = np.asarray(M_screen, dtype=float) if M_screen is not None else np.zeros((0, 6), dtype=float)
    if arr.ndim != 2:
        arr = np.zeros((0, 6), dtype=float)

    n_screen = int(arr.shape[0])
    n_prev = int(n_previous) if n_previous is not None else int(n_initial)

    pz = arr[:, 5] if arr.shape[1] > 5 else np.asarray([], dtype=float)
    x = arr[:, 0] if arr.shape[1] > 0 else np.asarray([], dtype=float)
    y = arr[:, 2] if arr.shape[1] > 2 else np.asarray([], dtype=float)

    pz_f = pz[np.isfinite(pz)]
    x_f = x[np.isfinite(x)]
    y_f = y[np.isfinite(y)]

    def _mean_or_none(vals: np.ndarray, scale: float = 1.0):
        if vals.size == 0:
            return None
        return float(scale * np.mean(vals))

    def _std_or_none(vals: np.ndarray, scale: float = 1.0):
        if vals.size == 0:
            return None
        return float(scale * np.std(vals))

    tr_init = (float(n_screen) / float(n_initial)) if int(n_initial) > 0 else None
    tr_prev = (float(n_screen) / float(n_prev)) if int(n_prev) > 0 else None

    summary: Dict[str, Any] = {
        "screen_index": int(screen_index),
        "z_m": float(z_m),
        "N": int(n_screen),
        "transmission_from_initial": tr_init,
        "transmission_from_previous": tr_prev,
        "mean_pz_MeV_c": _mean_or_none(pz_f),
        "sigma_pz_MeV_c": _std_or_none(pz_f),
        "mean_x_mm": _mean_or_none(x_f),
        "sigma_x_mm": _std_or_none(x_f),
        "mean_y_mm": _mean_or_none(y_f),
        "sigma_y_mm": _std_or_none(y_f),
        # No mean_z_mm/sigma_z_mm here: a Screen's own %Z is not a lab-frame position for this
        # project's wide-energy-spread thermionic beam -- it's each crossing particle's velocity
        # times its time offset from the bunch's reference particle (see rf_gun.aperture's module
        # docstring for the full derivation). Presenting its mean/std as "position" is actively
        # misleading (confirmed: values of >1 m for a screen physically at ~1 cm) -- there is no
        # substitute z-like quantity to put here; z_m above is this screen's own known, correct z.
        # Backward-compatible aliases used by existing callers.
        "transmission": tr_init,
        "mean_pz": _mean_or_none(pz_f),
        "sigma_pz": _std_or_none(pz_f),
    }
    return summary


def _masked_mean(arr: np.ndarray, mask: np.ndarray):
    if arr is None or mask.size == 0 or not np.any(mask):
        return np.nan
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def classify_particle_outcomes(
    initial: np.ndarray,
    final: np.ndarray,
    t0_mm_c: np.ndarray | None = None,
    lost_table: np.ndarray | None = None,
    id_col: int = 6,
    lost_id_col: int = -1,
    k_col: int = K_COL,
    max_kinetic_energy_mev: Optional[float] = MAX_PHYSICAL_KINETIC_ENERGY_MEV,
    backstop_z_min_m: Optional[float] = None,
    cathode_hit_ids: Optional[frozenset] = None,
):
    """Classify particles into transmitted/backward (from `final`=`Bout`'s own z/pz) and lost.

    Supplying `cathode_hit_ids` (from `BackBombardmentEvents.cathode_hit_ids()`) additionally
    splits `backward_returned` into `count_hit_cathode` / `count_missed_cathode`, giving the four
    outcomes that actually matter for this gun:

        forward, survived every aperture      -> transmitted
        forward, removed by the aperture R(z) -> lost.count_aperture
        backward, struck LaB6                 -> backward_returned.count_hit_cathode
        backward, did not strike LaB6         -> backward_returned.count_missed_cathode

    Turning around is not the same as bombarding the cathode. The LaB6 disk is 2.8 mm across the
    flat and 3.2 mm across the bevel, and the annulus out to the cavity nose bore is open vacuum,
    so a returning electron can cross the cathode plane without striking anything. Only
    `count_hit_cathode` deposits energy in the cathode. Where the rest end up is deliberately not
    subdivided here.

    "Lost" combines two disjoint sources: RF-Track's own `lost_table` (particles the dynamic
    aperture removed during tracking, id-based, see `rf_gun.aperture`/`rf_gun.particle_tags`) and
    any `final` (`Bout`) row whose kinetic energy (`%K`, column `k_col`) is non-finite or exceeds
    `max_kinetic_energy_mev` -- see `rf_gun.particle_tags.MAX_PHYSICAL_KINETIC_ENERGY_MEV`'s
    docstring for why this backstop exists (a particle can stay within the dynamic aperture's
    transverse bound yet still blow up numerically in momentum, and would otherwise be silently
    counted as "transmitted"). Pass `max_kinetic_energy_mev=None` to use only `lost_table`.
    transmitted/backward exclude both lost sources by construction.

    `backstop_z_min_m`, when given (the cathode backstop's global z span's lower edge, matching
    `rf_gun.particle_tags.build_particle_tags`'s own parameter of the same name), reclassifies
    every `lost_table` row the backstop actually captured
    (`rf_gun.backstop_loss_separation.identify_backstop_loss_candidates`) into "backward_returned"
    instead of "lost": a backstop capture physically turned around and returned toward the
    cathode -- it is the back-bombardment population, not a dynamic-aperture (transverse) loss.
    `lost`'s own `count_aperture` is the dynamic-aperture-only count in that case. Omit (the
    default) to keep every `lost_table` row counted as `lost`, matching this function's pre-
    backstop-aware behavior.
    """
    initial = np.asarray(initial)
    final = np.asarray(final)
    n0 = int(initial.shape[0]) if initial.ndim == 2 else 0

    if final.ndim != 2 or final.shape[0] == 0 or final.shape[1] < 6:
        transmitted_mask = np.zeros((0,), dtype=bool)
        backward_mask = np.zeros((0,), dtype=bool)
        unphysical_mask = np.zeros((0,), dtype=bool)
    else:
        # Column 4 of `%X %Px %Y %Py %Z %Pz ...` is `%Z` in MILLIMETRES already -- do not rescale
        # it. Every `final_z_mean_mm` below used to carry a stray `1e3 *`, so the value reported in
        # run_results.json was 1000x too large (cross-checked against the same run's openPMD Bout,
        # which writes `z_mm * 1e-3`: a KOA study_ii "fine" case reported 1383559.15 mm for a beam
        # the HDF5 puts at 1.3836 m). Sign-only uses (the forward/backward masks here) were never
        # affected; only the reported means were.
        zf_full = np.asarray(final[:, 4], dtype=float)
        pzf_full = np.asarray(final[:, 5], dtype=float)
        if max_kinetic_energy_mev is not None and final.shape[1] > k_col:
            kf_full = np.asarray(final[:, k_col], dtype=float)
            unphysical_mask = ~np.isfinite(kf_full) | (np.abs(kf_full) > float(max_kinetic_energy_mev))
        else:
            unphysical_mask = np.zeros(final.shape[0], dtype=bool)
        transmitted_mask = (
            np.isfinite(zf_full) & np.isfinite(pzf_full) & (zf_full > 0.0) & (pzf_full > 0.0) & ~unphysical_mask
        )
        backward_mask = (
            np.isfinite(zf_full) & np.isfinite(pzf_full) & ((zf_full <= 0.0) | (pzf_full < 0.0)) & ~unphysical_mask
        )

    # Pair each `final` row with its originating `initial` row by `%id`, not by position: once the
    # dynamic aperture removes any particle mid-tracking, `final`'s row order no longer lines up
    # positionally with `initial`'s (see `rf_gun.particle_tags`'s module docstring) -- a positional
    # `initial[:n_match]`/`final[:n_match]` truncation would silently pair the wrong particles'
    # initial/final state for every row from the first removal onward. `np.searchsorted` keeps
    # this vectorized (this runs at full macroparticle count, e.g. 1e5, every call).
    if (
        initial.ndim == 2 and final.ndim == 2
        and initial.shape[1] > id_col and final.shape[1] > id_col
        and initial.shape[0] > 0 and final.shape[0] > 0
    ):
        init_ids_full = initial[:, id_col].astype(np.int64)
        final_ids_full = final[:, id_col].astype(np.int64)
        order = np.argsort(init_ids_full)
        sorted_init_ids = init_ids_full[order]
        pos = np.searchsorted(sorted_init_ids, final_ids_full)
        pos = np.clip(pos, 0, sorted_init_ids.size - 1)
        found = sorted_init_ids[pos] == final_ids_full
        final_rows = np.nonzero(found)[0]
        init_rows = order[pos[final_rows]]
    else:
        final_rows = np.zeros((0,), dtype=np.int64)
        init_rows = np.zeros((0,), dtype=np.int64)

    pz0 = np.asarray(initial[init_rows, 5], dtype=float) if initial.ndim == 2 and initial.shape[1] > 5 and init_rows.size else None
    pzf = np.asarray(final[final_rows, 5], dtype=float) if final.ndim == 2 and final.shape[1] > 5 and final_rows.size else None
    zf = np.asarray(final[final_rows, 4], dtype=float) if final.ndim == 2 and final.shape[1] > 4 and final_rows.size else None
    t0 = np.asarray(t0_mm_c, dtype=float).reshape(-1)[init_rows] if t0_mm_c is not None and init_rows.size else None
    transmitted_mask_match = transmitted_mask[final_rows]
    backward_mask_match = backward_mask[final_rows]
    unphysical_mask_match = unphysical_mask[final_rows]

    n_trans = int(np.sum(transmitted_mask))
    n_back_bout = int(np.sum(backward_mask))
    n_unphysical = int(np.sum(unphysical_mask))

    lost_arr = np.asarray(lost_table, dtype=float) if lost_table is not None else np.zeros((0, 0))
    has_lost = lost_arr.ndim == 2 and lost_arr.shape[0] > 0

    def _row_stats(rows: np.ndarray) -> tuple[int, float, float, float, float]:
        """`(n, final_pz_mean, final_z_mean_mm, initial_pz0_mean, initial_t0_mean)` for a subset
        of `lost_table` rows -- final-state stats come straight from the row's own columns
        (LOST_COLUMNS order: x, px, y, py, z, pz, t, mass, q, N, id), initial-state stats by
        id-matching back to `initial`."""
        n = int(rows.shape[0])
        if n == 0 or rows.shape[1] <= max(5, abs(lost_id_col)):
            return 0, np.nan, np.nan, np.nan, np.nan
        pzf_mean = float(np.mean(rows[:, 5]))
        zf_mean_mm = float(np.mean(rows[:, 4]))
        pz0_mean = np.nan
        t0_mean = np.nan
        if initial.ndim == 2 and initial.shape[1] > id_col:
            row_ids = rows[:, lost_id_col].astype(np.int64)
            init_ids = initial[:, id_col].astype(np.int64)
            by_id = {int(pid): i for i, pid in enumerate(init_ids.tolist())}
            rows0 = [by_id[int(pid)] for pid in row_ids.tolist() if int(pid) in by_id]
            if rows0:
                pz0_mean = float(np.mean(initial[rows0, 5])) if initial.shape[1] > 5 else np.nan
                if t0_mm_c is not None:
                    t0_full = np.asarray(t0_mm_c, dtype=float).reshape(-1)
                    if t0_full.size == initial.shape[0]:
                        t0_mean = float(np.mean(t0_full[rows0]))
        return n, pzf_mean, zf_mean_mm, pz0_mean, t0_mean

    backstop_arr = np.zeros((0, lost_arr.shape[1] if lost_arr.ndim == 2 else 0))
    if has_lost and backstop_z_min_m is not None:

        # Backward is a momentum-sign question (Pz<0), not a position one -- see
        # rf_gun.particle_tags.backward_ids_from_lost_table. The narrow backstop z-band decides
        # only which rows get ray-cast for a cathode impact, not who turned around.
        pz_lost = np.asarray(lost_arr[:, 5], dtype=float)
        is_backstop_row = np.isfinite(pz_lost) & (pz_lost < 0.0)
        backstop_arr = lost_arr[is_backstop_row]
        aperture_arr = lost_arr[~is_backstop_row]
    else:
        aperture_arr = lost_arr if has_lost else np.zeros((0, 0))

    n_lost_table, lost_pzf_mean, lost_zf_mean_mm, lost_pz0_mean, lost_t0_mean = _row_stats(aperture_arr)
    n_backstop, backstop_pzf_mean, backstop_zf_mean_mm, backstop_pz0_mean, backstop_t0_mean = _row_stats(backstop_arr)

    def frac(n: int) -> float:
        return float(n / n0) if n0 > 0 else np.nan

    def combine_mean(mean_a: float, n_a: int, mean_b: float, n_b: int) -> float:
        """Weighted mean across two disjoint groups (lost_table rows vs. unphysical-K `final`
        rows) -- these describe the same "lost" bucket but come from different arrays, so their
        per-group means must be recombined by count rather than concatenated directly."""
        a_ok = n_a > 0 and np.isfinite(mean_a)
        b_ok = n_b > 0 and np.isfinite(mean_b)
        if a_ok and b_ok:
            return float((mean_a * n_a + mean_b * n_b) / (n_a + n_b))
        if a_ok:
            return float(mean_a)
        if b_ok:
            return float(mean_b)
        return np.nan

    unphys_initial_pz_mean = _masked_mean(pz0, unphysical_mask_match) if pz0 is not None else np.nan
    unphys_final_pz_mean = _masked_mean(pzf, unphysical_mask_match) if pzf is not None else np.nan
    unphys_initial_t0_mean = _masked_mean(t0, unphysical_mask_match) if t0 is not None else np.nan
    unphys_final_z_mean_mm = _masked_mean(zf, unphysical_mask_match) if zf is not None else np.nan

    n_back = n_back_bout + n_backstop
    n_lost = n_lost_table + n_unphysical

    # Split the backward population by whether it actually struck LaB6. Turning around is NOT the
    # same as bombarding the cathode: the annulus between the 3.2 mm LaB6 disk and the cavity nose
    # bore is open vacuum, so a returning electron can cross the cathode plane without hitting
    # anything. Only `hit_cathode` deposits energy in the cathode.
    n_hit_cathode: Optional[int] = None
    n_missed_cathode: Optional[int] = None
    if cathode_hit_ids is not None:
        hit_set = frozenset(int(i) for i in cathode_hit_ids)
        n_hit_cathode = 0
        if backstop_arr.ndim == 2 and backstop_arr.shape[0] and backstop_arr.shape[1] > abs(lost_id_col):
            backstop_ids = backstop_arr[:, lost_id_col].astype(np.int64)
            n_hit_cathode += int(sum(1 for pid in backstop_ids.tolist() if int(pid) in hit_set))
        if n_back_bout and final.ndim == 2 and final.shape[1] > id_col:
            bout_back_ids = final[backward_mask, id_col].astype(np.int64)
            n_hit_cathode += int(sum(1 for pid in bout_back_ids.tolist() if int(pid) in hit_set))
        n_missed_cathode = int(n_back - n_hit_cathode)

    return {
        "n_initial": n0,
        "n_final": int(final.shape[0]) if final.ndim == 2 else 0,
        "transmitted": {
            "count": n_trans,
            "fraction": frac(n_trans),
            "initial_pz_mean": _masked_mean(pz0, transmitted_mask_match) if pz0 is not None else np.nan,
            "final_pz_mean": _masked_mean(pzf, transmitted_mask_match) if pzf is not None else np.nan,
            "initial_t0_mean_mm_c": _masked_mean(t0, transmitted_mask_match) if t0 is not None else np.nan,
            "final_z_mean_mm": _masked_mean(zf, transmitted_mask_match) if zf is not None else np.nan,
        },
        "backward_returned": {
            "count": n_back,
            "fraction": frac(n_back),
            "count_bout": n_back_bout,
            "count_backstop": n_backstop,
            # None unless `cathode_hit_ids` was supplied (it comes from the back-bombardment
            # event capture, which runs after this classification in the driver).
            "count_hit_cathode": n_hit_cathode,
            "count_missed_cathode": n_missed_cathode,
            "fraction_hit_cathode": (
                frac(n_hit_cathode) if n_hit_cathode is not None else None
            ),
            "initial_pz_mean": combine_mean(
                _masked_mean(pz0, backward_mask_match) if pz0 is not None else np.nan, n_back_bout,
                backstop_pz0_mean, n_backstop,
            ),
            "final_pz_mean": combine_mean(
                _masked_mean(pzf, backward_mask_match) if pzf is not None else np.nan, n_back_bout,
                backstop_pzf_mean, n_backstop,
            ),
            "initial_t0_mean_mm_c": combine_mean(
                _masked_mean(t0, backward_mask_match) if t0 is not None else np.nan, n_back_bout,
                backstop_t0_mean, n_backstop,
            ),
            "final_z_mean_mm": combine_mean(
                _masked_mean(zf, backward_mask_match) if zf is not None else np.nan, n_back_bout,
                backstop_zf_mean_mm, n_backstop,
            ),
        },
        "lost": {
            "count": n_lost,
            "fraction": frac(n_lost),
            "count_aperture": n_lost_table,
            "count_unphysical_energy": n_unphysical,
            "initial_pz_mean": combine_mean(lost_pz0_mean, n_lost_table, unphys_initial_pz_mean, n_unphysical),
            "final_pz_mean": combine_mean(lost_pzf_mean, n_lost_table, unphys_final_pz_mean, n_unphysical),
            "initial_t0_mean_mm_c": combine_mean(lost_t0_mean, n_lost_table, unphys_initial_t0_mean, n_unphysical),
            "final_z_mean_mm": combine_mean(lost_zf_mean_mm, n_lost_table, unphys_final_z_mean_mm, n_unphysical),
        },
    }


def to_lost_table_array(raw_lost: Any):
    """Normalize RF-Track lost-particle table to ndarray with expected columns."""
    if raw_lost is None:
        return None
    arr = np.asarray(raw_lost)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return None
    if arr.shape[0] == 0:
        return np.zeros((0, 11), dtype=float)
    return np.asarray(arr, dtype=float)
