"""Tracking-domain end (`rf_gun.aperture.L_END_MM`) and the .eps size cap."""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rf_gun as rg
from rf_gun import aperture
from rf_gun.plotting.figure_io import EPS_MAX_BYTES, save_figure_formats


def test_exit_tube_end_is_the_documented_face():
    assert aperture.L_END_MM == pytest.approx(40.589)
    assert rg.exit_tube_end_z_m(0.0) == pytest.approx(40.589e-3)
    # The pipe-2 span is positive and starts where the rounded transition ends.
    assert aperture.L_PIPE2_MM == pytest.approx(aperture.L_END_MM - aperture.L_MM)
    assert aperture.L_PIPE2_MM > 0.0


def test_exit_tube_end_follows_cathode_insertion():
    """`s = z + delta`, so a recessed cathode (delta<0) pushes the tube end downstream in z."""
    for delta in (-2.0, 0.0, 1.5):
        assert rg.exit_tube_end_z_m(delta) == pytest.approx((aperture.L_END_MM - delta) * 1e-3)
    assert rg.exit_tube_end_z_m(-2.0) > rg.exit_tube_end_z_m(1.5)


def test_channel_is_the_narrow_tube_all_the_way_to_the_end():
    """R(z) must still be the pipe-2 radius immediately before the tube end -- otherwise the
    domain end would sit somewhere in the rounded transition instead of on the exit face."""
    z_mm = np.linspace(aperture.L_MM + 1e-6, aperture.L_END_MM, 200)
    R = aperture.aperture_radius_profile_mm(z_mm, 0.0)
    assert np.allclose(R, aperture.R2_MM)


def test_domain_end_is_inside_the_field_map_channel_bracket():
    """The field map's own mesh brackets the tube end between its last narrow row and the first
    opened-up one (see `L_END_MM`'s comment); the constant must lie inside that bracket."""
    assert 40.360 < aperture.L_END_MM < 42.180


def _scatter_fig(n):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    ax.scatter(rng.normal(size=n), rng.normal(size=n), s=1)
    return fig


def test_small_eps_is_written(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = save_figure_formats(fig, tmp_path, "small")
    plt.close(fig)
    assert written == ["small.png", "small.eps"]
    assert (tmp_path / "small.eps").stat().st_size <= EPS_MAX_BYTES


def test_oversized_eps_is_skipped_but_the_png_still_lands(tmp_path):
    fig = _scatter_fig(50_000)
    written = save_figure_formats(fig, tmp_path, "big", dpi=100, eps_max_bytes=50_000)
    plt.close(fig)
    assert written == ["big.png"]
    assert (tmp_path / "big.png").exists()
    assert not (tmp_path / "big.eps").exists()


def test_capture_figures_ignores_figures_left_open_by_an_earlier_block(tmp_path):
    """Under Agg, plt.show() leaves figures on the stack; a later capture block must not adopt
    one and write it out under its own name (observed: on_axis_field_profile.png came out as a
    second copy of the field-map figure)."""
    plt.close("all")
    stale, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])  # created outside any capture block, never shown

    with rg.capture_figures("mine", tmp_path, formats=("png",)):
        fig, ax2 = plt.subplots()
        ax2.plot([0, 1], [0, 1])
        plt.show()

    assert (tmp_path / "mine.png").exists()
    assert not (tmp_path / "mine_1.png").exists()
    plt.close("all")


def test_final_z_mean_is_reported_in_millimetres():
    """`%Z` (column 4) is already in mm; `final_z_mean_mm` must not rescale it.

    Regression for a stray `1e3 *` that made every run_results.json report a mean final z 1000x
    too large (a KOA study_ii "fine" case printed 1383559.15 mm for a beam its own openPMD Bout
    puts at 1.3836 m).
    """
    from rf_gun.diagnostics import classify_particle_outcomes

    # id, z, pz: two forward at z = 1000 and 2000 mm, one backward.
    initial = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    ])
    final = np.array([
        [0.0, 0.0, 0.0, 0.0, 1000.0, 0.5, 0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0, 2000.0, 0.5, 1.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0, -1.0, -0.2, 2.0, 0.0, 0.2, 0.2],
    ])
    out = classify_particle_outcomes(initial, final, max_kinetic_energy_mev=None)
    assert out["transmitted"]["count"] == 2
    assert out["transmitted"]["final_z_mean_mm"] == pytest.approx(1500.0)
    assert out["backward_returned"]["final_z_mean_mm"] == pytest.approx(-1.0)


def test_lab6_thermal_properties_extend_together_past_the_last_knot():
    """cp AND k must both survive just past 2000 K, and both must still fail past melting.

    Regression for the failure that killed all 10 KOA Study IV macropulse cases: the solver
    overshot the last tabulated knot by <= 0.3 K and raised. Giving only the cp YAML an
    `extrapolation_limit_x` relocated the crash onto `thermal.k`, because the k composite
    (`_build_lab6_k_composite_dataset`) hardcoded `extrapolation="forbidden"` and ignored its own
    upper segment's policy.
    """
    import warnings as _warnings
    import rf_gun as rg

    th = rg.load_cathode_material("LaB6", "LaB6_UH_recommended_v1").thermal

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        # Just past the last knot: the real Study IV failure point.
        for T in (2000.0032453242623, 2100.0, 2483.0):
            assert np.isfinite(float(th.cp_J_kg_K(T)))
            assert np.isfinite(float(th.k_W_m_K(T)))
        # Monotone and physically sensible in the extrapolated band.
        assert float(th.k_W_m_K(2483.0)) < float(th.k_W_m_K(2000.0))
        assert float(th.cp_J_kg_K(2483.0)) > float(th.cp_J_kg_K(2000.0))

    # Past melting the solid-phase model genuinely stops existing -- this must stay hard.
    for name in ("cp_J_kg_K", "k_W_m_K"):
        with pytest.raises(ValueError, match="physical limit"):
            getattr(th, name)(2483.1)


def test_lab6_k_composite_warns_rather_than_raising_above_the_table():
    import rf_gun as rg

    th = rg.load_cathode_material("LaB6", "LaB6_UH_recommended_v1").thermal
    with pytest.warns(RuntimeWarning, match="above the tabulated range"):
        th.k_W_m_K(2100.0)
