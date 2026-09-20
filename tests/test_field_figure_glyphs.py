"""Every glyph in the field-map figure must exist in the configured font.

`rf_gun/plotting/style.py` selects `cmr10` for the LaTeX look.  That face carries a
small glyph set, so a stray typographic character in a title (an em-dash, a prime, a
non-breaking space) silently renders as a missing-character box in the flagship
"EM FDTD Solver ... field maps used by RF-Track" figure.  Matplotlib reports that as a
warning rather than an error, so assert on the warning.
"""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rf_gun.plotting.fields import field_maps  # noqa: E402


def _draw_full_field_figure():
    """Render the production four-panel figure (Ez, Er, Btheta, Bz) on a small grid."""
    nz, nr = 24, 9
    z_grid = np.linspace(0.0, 40.589e-3, nz)
    r_grid = np.linspace(0.0, 8.0e-3, nr)
    ez = np.ones((nz, nr), dtype=complex) * 1.0e6
    er = np.ones((nz, nr), dtype=complex) * 1.0e5
    bt = np.ones((nz, nr), dtype=complex) * 1.0e-3
    bz = np.ones((nz, nr), dtype=complex) * 1.0e-6

    field_maps(
        {}, {}, np.empty(0), 0.0,
        r_grid, z_grid, ez, 0.10496934803921569,
        Er_grid=er,
        Bt_grid=bt,
        Bz_grid=bz,
        z_end_m=40.589e-3,
        measured_z_max_m=32.9487843221426e-3,
        aperture_delta_mm=0.0,
        show_colorbar=True,
        phase_deg=208.4,
        source_label="Remcom XFdtd volume artifact",
    )
    fig = plt.gcf()
    fig.canvas.draw()  # glyph lookup only happens at draw time
    return fig


def test_field_maps_figure_has_no_missing_glyphs():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = _draw_full_field_figure()
        plt.close(fig)

    missing = [str(w.message) for w in caught if "missing from font" in str(w.message)]
    assert not missing, (
        "field_maps figure requests glyphs the configured font cannot render "
        f"(they appear as boxes): {missing}"
    )


@pytest.mark.parametrize("forbidden", ["—", "–", "−", " "])
def test_field_figure_titles_avoid_glyphs_cmr10_lacks(forbidden):
    """Guard the source text directly, so the defect is caught without rendering."""
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "rf_gun" / "plotting" / "fields.py"
    ).read_text(encoding="utf-8")
    offending = [
        line.strip()
        for line in source.splitlines()
        if forbidden in line and not line.strip().startswith("#")
    ]
    assert not offending, (
        f"non-cmr10 character {forbidden!r} in a fields.py display string: {offending}"
    )


def test_production_callers_pass_an_explicit_tail_label():
    """Both artifact variants shade the same downstream band for opposite reasons.

    `measured_z_max_m` is identical in the modal artifact and the zero-tail control, so the
    figure cannot infer which it is drawing. If a caller omits `tail_label`, the control is
    labelled "qualified modeled tail (not solver sampled)" -- describing a modeled field where
    the field is actually forced to zero.
    """
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]

    driver = (repo / "run_thermionic_tm010.py").read_text(encoding="utf-8")
    assert "rg.field_maps(" in driver
    call = driver[driver.index("rg.field_maps("):]
    call = call[: call.index("\n        )\n") + 12]
    assert "tail_label=" in call, (
        "run_thermionic_tm010.py calls field_maps without tail_label; the zero-tail control "
        "would be captioned as a modeled tail"
    )

    nb = json.loads((repo / "UH_gun_tracking_demo.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    field_map_cells = [s for s in sources if "rg.field_maps(" in s]
    assert field_map_cells, "main notebook no longer draws the field-map figure"
    for src in field_map_cells:
        assert "tail_label=" in src, (
            "UH_gun_tracking_demo.ipynb calls field_maps without tail_label"
        )


def test_tail_label_actually_reaches_the_legend():
    """The parameter must change the rendered label, not just be accepted."""
    import numpy as np

    from rf_gun.plotting.fields import field_maps

    nz, nr = 16, 6
    z_grid = np.linspace(0.0, 40.589e-3, nz)
    r_grid = np.linspace(0.0, 4.0e-3, nr)
    ez = np.ones((nz, nr), dtype=complex) * 1.0e6

    fig = field_maps(
        {}, {}, np.empty(0), 0.0, r_grid, z_grid, ez, 0.105,
        z_end_m=40.589e-3, measured_z_max_m=32.9487843221426e-3,
        show_colorbar=False, tail_label="zero-tail control (field forced to zero, not modeled)",
    )
    labels = [t.get_text() for ax in fig.axes for leg in [ax.get_legend()] if leg
              for t in leg.get_texts()]
    plt.close(fig)
    assert any("zero-tail control" in lbl for lbl in labels), labels
    assert not any("qualified modeled tail" in lbl for lbl in labels), labels
