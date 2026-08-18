# UH Gun Thermionic RF-Track Project

A thermionic electron gun: electrons are emitted from a heated cathode and accelerated by the RF
field of an S-band TM010 (quarter-wave) cavity, then transported through space charge, beam
loading, an optional steering (deflection) magnet, and an exit aperture, using **RF-Track**.

## What this simulates

- A cathode emits electrons thermionically (Richardson-Dushman with Schottky lowering, or a
  unified thermionic + field-emission law), including surface roughness.
- The emitted bunch is accelerated by the cavity's RF field, reconstructed from measured 2D field
  maps (an r-z cavity view and an x-z waveguide/iris view).
- Space charge and beam loading (the cavity's own R/Q, calibrated from a phase scan) act on the
  bunch during transport.
- An optional deflection magnet can steer the beam, e.g. to separate back-bombarding electrons
  from the main beam.
- A physical exit aperture (a real geometric channel, separate from RF-Track's own transport
  region) can clip the beam and select what is saved as the exit beam.

Two ways to run the same physics, producing the same kind of output:

- **`UH_gun_tracking_demo.ipynb`** — the interactive notebook, for exploring a single run.
- **`run_thermionic_tm010.py`** — a command-line script for scripted runs or SLURM parameter scans.

## A performance note: the deflection magnet forces single-threaded tracking

RF-Track normally tracks a bunch across several CPU threads at once. The deflection magnet is
implemented as a custom field (an RF-Track "user field"), and RF-Track only supports a custom
field on a single thread. So turning the deflection magnet on makes RF-Track track on one CPU
core instead of several, and a run takes correspondingly longer. This is a requirement of
RF-Track itself, not something this code works around — if a run needs every core, keep the
deflection magnet off.

## Physics setup

- Cavity frequency: 2.856 GHz (S-band, TM010)
- The field map is axisymmetric, built from measured XY/YZ data by phasor reconstruction (full
  I/Q, or crest-only)
- Emission timing follows a phase window; the emitted current is time-dependent and injected into
  RF-Track accordingly
- Beam-loading R/Q is calibrated from a fast on-axis phase scan unless fixed by hand
- The physical exit aperture is separate from RF-Track's own transport-region bound: it is a
  geometric radius cut applied at explicit entrance/exit screens, used both to tag particles and
  to choose what is saved as the exit beam
- Forward/backward classification uses each particle's own persistent identity, not a screen's
  local coordinate, since a screen's position value stops being a reliable lab-frame position
  once a particle has turned around

## Repository layout

```text
.
├── .gitignore
├── README.md
├── UH_gun_tracking_demo.ipynb
├── run_thermionic_tm010.py
├── run_thermionic_tm010_scanT_1400_1700.slurm
├── run_thermionic_tm010_scanT_1400_1700_fine.slurm
├── run_thermionic_tm010_scanT_1400_1700_extrafine.slurm
├── field_maps/
│   ├── XYplanarSensorData.mat
│   └── YZplanarSensorData.mat
└── rf_gun/
    ├── config.py               # connects to RF-Track
    ├── constants.py            # physical constants, unit conversions
    ├── helpers.py              # small numeric/formatting utilities
    ├── rf_params.py            # R/Q, delivered power, effective accelerating length
    ├── field_io.py             # reads field maps, interpolates onto the tracking grid
    ├── phasor.py               # builds the RF phasor from field-map snapshots
    ├── emission_models.py      # thermionic and field-emission current models
    ├── emission_sampling.py    # samples emission current in time
    ├── deflection_field.py     # steering-magnet field model
    ├── finesse_presets.py      # bundled speed/precision presets
    ├── rftrack_volume.py       # builds the RF-Track tracking volume and screens
    ├── simulation.py           # runs the tracking, reports progress
    ├── diagnostics.py          # beam size, emittance, per-screen summaries
    ├── particle_tags.py        # forward/backward and aperture tagging
    ├── aperture.py             # exit-aperture geometric cut
    ├── beam_properties.py      # beam size and transmission vs. distance
    ├── back_bombardment.py     # reconstructs electrons returning to the cathode
    ├── acceptance_scan.py      # separates the beam core from its trailing tail
    ├── io.py                   # saves results to disk
    └── plotting/               # figures (field maps, phase space, evolution, ...)
```

`.venv/` (the Python environment, including the compiled RF-Track binding) and local-only/generated
folders (`outputs/`, `outputs_Koa/`, `analysis_Koa/`, `manual_references/`, `archive/`, `logs/`)
are intentionally outside this layout — see `.gitignore`.

## Core workflow

1. Load the XY/YZ field maps and compute envelope diagnostics.
2. Build the RF phasor and interpolate it onto an (r, z) grid.
3. Run a fast on-axis phase scan to calibrate effective voltage and R/Q.
4. Track the thermionic bunch through RF-Track (space charge, beam loading, and the deflection
   magnet are each independently switchable).
5. Tag particles forward/backward and (if enabled) aperture-survived, and compute per-screen and
   whole-beam summaries.
6. Save figures, per-screen and exit-beam distributions, and a consolidated `run_summary.json` —
   the same shape from either entry point.

## Saved outputs

Both entry points write to a run directory named
`outputs/runs/<timestamp>_T<T>K_SC<on|off>_BL<on|off>/` (notebook: only when `SAVE_DATA = True`;
script: pass `--output`, or let it auto-name the same way). Inside:

- `figures/` — every diagnostic figure, each with its underlying data saved alongside so a figure
  can be reproduced later without re-running the simulation.
- `openpmd/` — the exit beam as an openPMD-beamphysics HDF5 file, aperture-clipped when the exit
  aperture is enabled.
- `screen_distributions_hdf5/` — one HDF5 file per tracking screen (full, unfiltered phase space).
- `lost_particle_diagnostics.json` — particles RF-Track reports as lost during tracking.
- `run_summary.json` — every input parameter plus every derived quantity (phase-scan crest,
  `Veff`, `R/Q`, transmission/aperture/back-bombardment statistics, per-screen summaries, output
  file paths) — written the same way from either entry point.

The notebook additionally saves `beam_properties.csv` (beam size vs. z). The script additionally
saves `beam_data.npz`, `run_metadata.json`, `progress_stats.json`, `B0.json`/`Bout.json`,
`beam_summary.json`, and `particle_classes_summary.json`.

## Batch execution / SLURM

Quick validation run:

```bash
python run_thermionic_tm010.py --preset quick
```

Full option list:

```bash
python run_thermionic_tm010.py --help
```

`run_thermionic_tm010_scanT_1400_1700.slurm` is the current parameter-scan template: a SLURM job
array (`--array=0-30`) sweeping the cathode temperature from 1400 K to 1700 K in 10 K steps, one
array task per temperature, at `N = 1,000,000` particles with space charge, beam loading, and the
exit aperture + openPMD export on, and the deflection magnet off (see the performance note above
— this keeps tracking multi-threaded). Use it as a template for other scans (deflection current,
temperature range, roughness) by varying the swept argument per array task.

### Solver finesse presets

`--finesse {extra_fine,fine,medium,coarse}` sets every numerical-resolution setting at once (field
map grid step, integration step counts, ODE tolerance, and the space-charge/beam-loading solver
step) — a trade between run time and numerical precision, independent of the physical settings
above (particle/screen counts, cathode temperature, cavity R/Q, deflection current, and which
physics is switched on). `fine` matches `run_thermionic_tm010_scanT_1400_1700.slurm`; `coarse`
matches the notebook's defaults. `run_thermionic_tm010_scanT_1400_1700_fine.slurm` and
`..._extrafine.slurm` are the same scan template at the `fine`/`extra_fine` tiers.

The notebook has the equivalent `NOTEBOOK_FINESSE_TIER` variable near the top of its configuration
cell.

## Plotting defaults

Every phase-space figure exposes two independent switches instead of an ad hoc parameter zoo:

- `exclude_backward_losses` — drop (vs. highlight) particles tagged as having turned backward.
- `exclude_aperture_losses` — drop (vs. highlight) particles that failed the exit-aperture cut.

Both default to `True` in the saved diagnostics, so the default saved view is the transmitted-like
beam, with the backward/aperture-lost populations available wherever a figure chooses to show them.

## RF-Track reference

- RF-Track project page: `https://abpcomputing.web.cern.ch/codes/codes_pages/RF-Track/`
