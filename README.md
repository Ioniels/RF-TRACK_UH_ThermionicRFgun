# UH Gun Thermionic RF-Track Project

Thermionic electron-beam dynamics in an S-band TM010 (lambda/4) RF gun using **RF-Track** for 6D transport, space charge, beam loading, and a post-hoc physical exit aperture / deflection magnet.

## Scope

This repository provides a consolidated simulation workflow for:

- RF phasor reconstruction from measured field maps (`reconstruct` I/Q or `simplified` crest-only)
- Thermionic emission (Richardson-Dushman with Schottky lowering, or the unified thermionic+field-emission law), including surface roughness (`Ra`, `Re`)
- RF-Track transport with optional space charge, beam loading, and a deflection magnet
- A post-hoc physical exit-aperture channel (entrance/exit screens + a geometric radius cut), independent of RF-Track's own whole-Volume aperture bound
- Robust, self-describing diagnostics and exports (JSON, HDF5, openPMD-beamphysics) for single runs and SLURM parameter scans

Two entry points share the same `rf_gun` package and produce output in the same shape:

- **`UH_gun_tracking_demo.ipynb`** — the interactive reference notebook. A `SAVE_DATA` switch controls whether a run's figures/beams/screens/summary are written to disk.
- **`run_thermionic_tm010.py`** — a CLI/SLURM-friendly script for scripted single runs or parameter scans (see `run_thermionic_tm010_scanT_1400_1700.slurm` for a worked example).

## Physics Model (Current)

- Cavity frequency: `f = 2.856 GHz` (TM010)
- Axisymmetric field map used by RF-Track is built from XY/YZ measurements (I/Q phasor reconstruction or crest-only)
- Emission timing is phase-windowed (`emission_phase_start`, `emission_phase_range`) and the time-dependent emission current is sampled and injected into RF-Track
- **Bunch timing injection uses the extended `Bunch6dT` matrix with explicit `T0`**, not optional Python binding methods
- Beam-loading `R/Q` is calibrated from a fast on-axis phase scan (`veff_from_phase_scan_pz`, `r_over_q_per_m`) unless fixed explicitly
- The physical exit aperture (`APERTURE_START_M`/`APERTURE_END_M`/`APERTURE_DIAMETER_MM` in the notebook; `--aperture_enabled/--aperture_start_m/--aperture_end_m/--aperture_diameter_mm` in the script) is a separate concept from RF-Track's own whole-Volume aperture bound (`--aperture_m`): it is a post-hoc geometric cut plus explicit entrance/exit screens, used for forward/backward particle tagging and for selecting the beam saved as the openPMD exit beam
- An optional deflection magnet (`Bx(z) = B_pk(I) / (1 + ((z - z_p)/w)^2)`) can be enabled on either entry point

Important coordinate note:

- In `Bunch6dT`, `Z` is the spatial coordinate; `T0` is the particle's creation time and is stored separately from the 6D phase-space columns.
- A screen's own `%Z` is *not* a reliable lab-frame position once particles have turned backward (see `rf_gun/aperture.py`'s and `rf_gun/particle_tags.py`'s module docstrings) — longitudinal diagnostics use `%t` (arrival time) instead, and forward/backward classification is done by `%id` lookup against `Bout`'s own reliable absolute z/pz (`rf_gun.particle_tags`), not a screen's local sign.

## Repository Layout

```text
.
├── .gitignore
├── README.md
├── UH_gun_tracking_demo.ipynb
├── run_thermionic_tm010.py
├── run_thermionic_tm010_scanT_1400_1700.slurm
├── field_maps/
│   ├── XYplanarSensorData.mat
│   └── YZplanarSensorData.mat
└── rf_gun/
    ├── __init__.py
    ├── config.py                # RF-Track import, thread/version helpers
    ├── constants.py              # physical constants, unit conversions
    ├── helpers.py                 # small generic numeric/formatting utilities
    ├── rf_params.py               # R/Q, delivered power, effective length, Veff
    ├── field_io.py                # field-map loading + interpolation onto (r,z) grid
    ├── phasor.py                  # I/Q snapshot selection, phasor construction/check
    ├── emission_models.py         # Richardson-Dushman/Schottky, unified emission law
    ├── emission_sampling.py       # time-dependent emission-current sampling
    ├── deflection_field.py        # deflection magnet field model + defaults
    ├── rftrack_volume.py          # RF-Track Volume/Screen construction
    ├── simulation.py              # TrackingParams/DiagnosticsParams, transport run, phase scan
    ├── diagnostics.py             # Twiss/emittance, per-screen summaries
    ├── particle_tags.py           # %id-based forward/backward + aperture-loss tagging
    ├── aperture.py                # physical exit-aperture masks/summary
    ├── beam_properties.py         # beam properties vs. z, transmission curves
    ├── back_bombardment.py        # backward-turning-particle reconstruction at z=0
    ├── acceptance_scan.py         # trailing/core acceptance-radius scan
    ├── io.py                      # JSON/HDF5/openPMD exports, run_summary.json
    └── plotting/
        ├── __init__.py
        ├── fields.py               # field-map + on-axis phase figures
        ├── phase_scan.py           # fast phase-scan figure
        ├── emission.py             # emission-history / J-vs-n figures
        ├── phase_space.py          # phase-space panels, spectra, screen slider
        ├── evolution.py            # beam moments/Twiss evolution vs. z
        ├── back_bombardment.py     # back-bombardment figures
        ├── acceptance_scan.py      # acceptance-scan figure
        ├── save_run.py             # save_run_figures, capture_figures
        └── style.py                # shared colors/colormaps/style config
```

`.venv/` (the Python environment, including the compiled RF-Track binding) and local-only/generated folders (`outputs/`, `outputs_Koa/`, `analysis_Koa/`, `manual_references/`, `archive/`, `logs/`) are intentionally outside this layout — see `.gitignore`.

## Core Workflow

1. Load XY/YZ field maps and compute envelope diagnostics.
2. Build the RF phasor (`reconstruct` or `simplified`) and interpolate it onto an `(r, z)` grid.
3. Run a fast on-axis phase scan for effective voltage / R-over-Q calibration.
4. Run the thermionic transport with RF-Track (space charge, beam loading, deflection magnet all independently configurable).
5. Tag particles forward/backward and (if enabled) aperture-survived via `%id`, and derive per-screen/beam summaries from explicit phase-space arrays.
6. Save figures, per-screen and exit-beam distributions, and a consolidated `run_summary.json` — identically from either entry point.

## Saved Outputs

Both entry points write to a run directory named `outputs/runs/<timestamp>_T<T>K_SC<on|off>_BL<on|off>/` (notebook: only when `SAVE_DATA = True`; script: pass `--output`, or let it auto-name the same way). Inside:

- `figures/` — every diagnostic figure (`.png`/`.eps`), each with the numeric data behind it saved alongside as `.npz` (or `.json` for mixed/tabular payloads) via `rf_gun.plotting.capture_figures`, so a figure can be reproduced or re-plotted later without re-running the simulation.
- `openpmd/` — the exit beam as an openPMD-beamphysics HDF5 file (`Bout_sout<mm>mm_T<K>K_SC<on|off>_BL<on|off>.h5`), aperture-clipped when the physical exit aperture is enabled.
- `screen_distributions_hdf5/` — one openPMD-beamphysics HDF5 file per tracking screen (unfiltered raw phase space, with screen index/z/transmission as HDF5 attributes).
- `lost_particle_diagnostics.json` — particles RF-Track reports as lost during tracking.
- `run_summary.json` — every hardcoded input parameter plus every derived/output quantity (phase-scan crest, `Veff`, `R/Q`, transmission/aperture/back-bombardment statistics, per-screen summaries, output file paths), written by the shared `rf_gun.io.save_run_summary` so a run started either way reads back the same way.

The notebook additionally saves `beam_properties.csv` (the full beam-properties-vs-z table). The script additionally saves its own `beam_data.npz`, `run_metadata.json`, `progress_stats.json`, `B0.json`/`Bout.json`, `beam_summary.json`, and `particle_classes_summary.json` — plus, optionally, `screen_distributions_json/` (`--save-screen-json`) alongside the HDF5 screens.

## Batch Execution / SLURM

Quick validation run:

```bash
python run_thermionic_tm010.py --preset quick
```

CLI reference:

```bash
python run_thermionic_tm010.py --help
```

By default the run directory is auto-named the same way as the notebook's `SAVE_DATA` run, e.g. `outputs/runs/20260101_120000_T1700K_SCon_BLon/`; pass `--output` to override it explicitly.

`run_thermionic_tm010_scanT_1400_1700.slurm` is the current parameter-scan template: a SLURM **job array** (`--array=0-30`) sweeping the cathode temperature from 1400 K to 1700 K in 10 K steps, one array task per temperature, with fine space-charge/beam-loading time steps, `N = 1,000,000` particles, 10 tracking screens, the deflection magnet enabled, and the physical exit aperture + openPMD export turned on (`--aperture_enabled --save-openpmd-beam`). Use it as the template for other scans (e.g. deflection current, cathode temperature range, roughness) by varying the swept CLI argument per array task.

## Plotting/Diagnostics Defaults

Every phase-space-style figure exposes two independent, explicit knobs instead of an ad hoc parameter zoo:

- `exclude_backward_losses` — drop (vs. highlight) particles tagged backward via `%id` against `Bout`'s reliable absolute z/pz.
- `exclude_aperture_losses` — drop (vs. highlight) particles that failed the physical exit-aperture radius cut.

Both default to `True` in the saved diagnostics, so the default saved view is the transmitted-like population, with backward/aperture-lost populations available by construction (not by re-deriving a mask ad hoc) wherever a figure chooses to show them.

## RF-Track Reference

- RF-Track project page: `https://abpcomputing.web.cern.ch/codes/codes_pages/RF-Track/`
