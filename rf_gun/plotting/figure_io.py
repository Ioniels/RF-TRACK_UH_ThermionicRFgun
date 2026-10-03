"""Single writer for every figure this package puts on disk, and the .eps size cap.

Why a size cap: matplotlib's EPS backend is pure vector -- a scatter of N macroparticles becomes N
path elements in the file, so a production run's `initial_phase_space_x_px.eps` grows without bound
with `--n_particles` while its PNG stays a few hundred kB (measured on a real KOA batch:
13.2 MB .eps vs 2.5 MB .png at 100k particles, and the .eps was still growing case to case). Those
files are slow to write, slow to open, and useless for the raster-preview purpose they were serving.

The rule (`EPS_MAX_BYTES`): a vector format is rendered to an in-memory buffer first and only
written to disk if it fits. Its size is not predictable before rendering -- it depends on the
element count, not the figure size or dpi -- so buffering is the only way to enforce this without
writing the oversized file first. The raster formats are always written, so a skipped .eps never
costs you the figure; you lose only the vector copy, and the skip is printed with the actual size
so it is visible in the SLURM log rather than silent.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

#: Maximum size [bytes] of a written .eps. Larger renders are skipped (see module docstring).
EPS_MAX_BYTES = 15 * 1024 * 1024

#: Formats subject to `EPS_MAX_BYTES`. Only the vector formats can blow up with element count;
#: a raster format's size is bounded by dpi and figure size, both fixed here.
SIZE_CAPPED_FORMATS = frozenset({"eps", "ps"})

#: Default format bundle for a saved run figure: raster always, vector when it fits.
DEFAULT_FIGURE_FORMATS = ("png", "pdf", "eps")


def normalize_formats(formats: Sequence[str] | None, *, fallback: Sequence[str] = ("png",)) -> list[str]:
    """Lower-case, strip and drop empties; fall back to `fallback` if nothing survives."""
    fmts = [str(fmt).strip().lower() for fmt in (formats or ()) if str(fmt).strip()]
    return fmts or [str(f).strip().lower() for f in fallback]


def save_figure_formats(
    fig,
    out_dir: Path | str,
    stem: str,
    *,
    formats: Sequence[str] = DEFAULT_FIGURE_FORMATS,
    dpi: int = 300,
    bbox_inches: str | None = "tight",
    eps_max_bytes: int = EPS_MAX_BYTES,
    verbose: bool = True,
) -> list[str]:
    """Write `fig` to `out_dir/stem.<fmt>` for each format; return the file names actually written.

    A format in `SIZE_CAPPED_FORMATS` is rendered to memory first and written only if it is at most
    `eps_max_bytes`; oversized renders are skipped (and reported when `verbose`). Every other
    format is written unconditionally, so the raster copy of a figure never depends on whether its
    vector copy fit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for fmt in normalize_formats(formats):
        out_path = out_dir / f"{stem}.{fmt}"
        save_kwargs = {"format": fmt, "dpi": dpi}
        if bbox_inches is not None:
            save_kwargs["bbox_inches"] = bbox_inches

        if fmt not in SIZE_CAPPED_FORMATS:
            fig.savefig(out_path, **save_kwargs)
            saved.append(out_path.name)
            continue

        buf = io.BytesIO()
        fig.savefig(buf, **save_kwargs)
        n_bytes = buf.tell()
        if n_bytes > int(eps_max_bytes):
            if verbose:
                print(
                    f"  Skipped {out_path.name}: {n_bytes / 1024 / 1024:.1f} MB exceeds the "
                    f"{int(eps_max_bytes) / 1024 / 1024:.0f} MB .{fmt} cap "
                    "(rf_gun.plotting.figure_io.EPS_MAX_BYTES); the raster copy was still written.",
                    flush=True,
                )
            continue
        out_path.write_bytes(buf.getvalue())
        saved.append(out_path.name)

    return saved
