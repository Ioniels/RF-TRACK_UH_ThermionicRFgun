# UH Gun Thermionic RF-Track Project

A thermionic electron gun: electrons are emitted from a heated cathode and accelerated by the RF
field of an S-band TM010 (quarter-wave) cavity, then transported through space charge, cathode
mirror charges, beam loading, an optional steering (deflection) magnet, and a dynamic transverse
aperture, using **RF-Track**.

## What this simulates

- A cathode emits electrons thermionically, via a choice of emission law (Richardson-Dushman with
  Schottky lowering, an additive thermionic + finite-temperature Murphy-Good field-emission model,
  a reformulated general thermal-field model, or a direct Murphy-Good energy-integral reference),
  including surface roughness and, optionally, a temperature-dependent LaB6⟨100⟩ work function.
- Emission can be sampled jointly in space and time, (x,y,t), from the real local RF field at the
  cathode (and, optionally, from a non-uniform cathode temperature profile T(x,y) — e.g. a
  backbombardment/laser-heating pattern) rather than assumed uniform over the emitting disk.
- The emitted bunch is accelerated by the cavity's RF field, reconstructed from measured 2D field
  maps (an r-z cavity view and an x-z waveguide/iris view).
- Space charge (an explicit PIC engine, with cathode mirror charges) and beam loading (the
  cavity's own R/Q, calibrated from a phase scan) act on the bunch during transport. An optional
  "Emission Fields Iteration" study pre-converges the emission current against its own space-charge
  and mirror feedback before the full production run.
- An optional deflection magnet can steer the beam, e.g. to separate back-bombarding electrons
  from the main beam.
- A dynamic transverse aperture R(z) — the cavity's real channel shape, not a single cylinder — is
  enforced by RF-Track during tracking, so a particle is removed the instant it leaves the real
  transverse channel.

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
- The RF field map itself is axisymmetric, built from measured XY/YZ data by phasor
  reconstruction (full I/Q, or crest-only); the cathode emission model, however, samples the
  field at genuine (x,y) points rather than assuming y=0, so a non-axisymmetric emission or
  temperature profile is fully supported even though the underlying field is not
- Emission timing follows a phase window; the emitted current is time-dependent and injected into
  RF-Track accordingly
- Beam-loading R/Q is calibrated from a fast on-axis phase scan unless fixed by hand
- The dynamic transverse aperture R(z) is enforced by RF-Track's own `Aperture_1d` element during
  tracking (not a post-hoc cut applied after the fact), so particle loss location and cause are
  physically meaningful and identical between the notebook and the script
- Forward/backward classification uses each particle's own persistent identity, not a screen's
  local coordinate, since a screen's position value stops being a reliable lab-frame position
  once a particle has turned around

## Self-consistent cathode emission

- **Emission models** (`--emission_law`): `RD_schottky` (Richardson-Dushman with Schottky
  lowering), `rld_schottky_plus_mg` (additive thermionic + finite-temperature Murphy-Good field
  emission), `rgtf_2019` (a reformulated general thermal-field model, exact digamma-based series
  rather than the source paper's own rational approximation, which has genuine poles at integer
  arguments), and `murphy_good_direct_reference` (a direct WKB/Kemble energy-integral reference,
  useful to cross-check the closed-form laws). `--compare-emission-models` evaluates several at
  once for the sensitivity figure.
- **LaB6⟨100⟩ work function models** (`--work-function-temperature-model`): `constant_phi_eff`
  (Liu et al. 2017 thermionic anchor), `linear_tcwf` (that anchor extended with a
  temperature-coefficient-of-work-function slope), and `piecewise_surface_evolution` (a
  Swanson/Bulyga/Liu three-branch blend). See `manual_references/LaB6_100_work_function_models.md`
  for the underlying literature synthesis. Unset, `--phi_eff_ev` is used as a fixed value instead.
- **Cathode mirror charges and explicit space charge** (`--mirror-charges`, `--sc-nx/ny/nz`,
  `--mirror-z-m`, `--mirror-charge-tolerance`): a `SpaceCharge_PIC_FreeSpace` engine is built and
  installed explicitly (mesh size and mirror plane both under control), instead of relying on
  RF-Track's own implicit default engine.
- **Emission Fields Iteration** (`--emission-field-iteration` and its `--emission-iteration-*`
  sub-flags): an under-relaxed Picard iteration, run once between the phase scan and the full
  production run, that couples the emission current to the space-charge and cathode-mirror field
  it generates near the cathode until current, field, and emitted charge converge. Its cathode
  grid is a genuine Cartesian (x,y) grid clipped to the emitting disk, not an azimuthally-averaged
  radial one, so an asymmetric cathode temperature profile produces a genuinely asymmetric
  converged current density. `--use-converged-iteration-source` feeds its converged J(x,y,t)
  directly into the production run in place of the default prescribed source, falling back
  clearly (with a printed message) if the iteration did not converge.
- **Spatially-resolved emission sampling** (`--spatial-emission-sampling`): samples the production
  bunch jointly in (x,y,t) from the RF field alone (no space-charge/mirror feedback), instead of
  the default independent radius-uniform/on-axis-F(t) model. Superseded by
  `--use-converged-iteration-source` when both are set and the iteration converged.

**Experimental:** `VolumeBuildParams.cathode_backstop_enabled` (default off) adds a thin absorbing
element just behind z=0, giving an exact back-bombardment crossing event via RF-Track's own
particle-loss table instead of `rf_gun.back_bombardment`'s field-free-drift extrapolation from
`Bout`. Not yet wired into the back-bombardment pipeline or validated against a production run.

**Scope limitations of this coupling:**

- The cathode mirror models a single conducting plane at the cathode, not conducting boundary
  conditions on the rest of the cavity (chamfer, main wall, exit nose, pipe 2) — the dynamic
  aperture is a particle-loss geometry, not a Poisson boundary condition.
- An emission model gives the material's *available* supply current; the self-consistent fields
  (via the Emission Fields Iteration) determine how much of it is actually extracted and
  transported. A strongly virtual-cathode-limited regime may need online emission inside RF-Track's
  own time loop rather than this outer iteration.
- Schottky/Murphy-Good barrier lowering (in the emission law) is a microscopic single-electron
  image-potential effect; the cathode mirror charge is the macroscopic field of the whole emitted
  bunch reflected in the cathode conductor. These are distinct physical scales and both should be
  kept when used together.

## Repository layout

```text
.
├── .gitignore
├── README.md
├── UH_gun_tracking_demo.ipynb
├── run_thermionic_tm010.py
├── run_thermionic_tm010_scanT_1400_1700_medium.slurm
├── run_thermionic_tm010_scanT_1400_1700_fine.slurm
├── run_thermionic_tm010_scanT_1400_1700_extrafine.slurm
├── run_thermionic_tm010_emission_iteration.slurm
├── field_maps/
│   ├── XYplanarSensorData.mat
│   └── YZplanarSensorData.mat
├── tests/                       # pytest suite (see "Running tests" below)
└── rf_gun/
    ├── config.py               # connects to RF-Track
    ├── constants.py            # physical constants, unit conversions
    ├── helpers.py              # small numeric/formatting utilities
    ├── rf_params.py            # R/Q, delivered power, effective accelerating length
    ├── field_io.py             # reads field maps, interpolates onto the tracking grid
    ├── phasor.py               # builds the RF phasor from field-map snapshots
    ├── emission_models.py      # thermionic and field-emission current models
    ├── emission_sensitivity.py # analytic/numeric sensitivity of emission models
    ├── emission_sampling.py    # samples emission phase space (thermal, roughness)
    ├── work_function_models.py # LaB6<100> phi_eff(T) models
    ├── cathode_fields.py       # signed fields, RF sampling, SC/mirror probe extraction
    ├── emission_iteration.py   # self-consistent Emission Fields Iteration study
    ├── deflection_field.py     # steering-magnet field model
    ├── finesse_presets.py      # bundled speed/precision presets
    ├── rftrack_volume.py       # builds the RF-Track tracking volume, screens, SC engine
    ├── simulation.py           # runs the tracking, reports stage-by-stage progress
    ├── diagnostics.py          # beam size, emittance, per-screen summaries
    ├── particle_tags.py        # forward/backward and aperture tagging
    ├── aperture.py             # dynamic transverse aperture R(z)
    ├── beam_properties.py      # beam size and transmission vs. distance
    ├── back_bombardment.py     # reconstructs electrons returning to the cathode
    ├── acceptance_scan.py      # separates the beam core from its trailing tail
    ├── io.py                   # saves results to disk
    └── plotting/               # figures (field maps, phase space, evolution, ...)
```

`.venv/` (the Python environment, including the compiled RF-Track binding) and local-only/generated
folders (`outputs/`, `Koa outputs/`, `analysis_Koa/`, `manual_references/`, `archive/`, `logs/`)
are intentionally outside this layout — see `.gitignore`.

## Core workflow

1. Load the XY/YZ field maps and compute envelope diagnostics.
2. Build the RF phasor and interpolate it onto an (r, z) grid.
3. Run a fast on-axis phase scan to calibrate effective voltage and R/Q.
4. Optionally run the Emission Fields Iteration self-consistency study near the cathode.
5. Build the thermionic bunch (on-axis, or jointly in (x,y,t) — see "Self-consistent cathode
   emission" above) and track it through RF-Track (space charge and cathode mirror charges, beam
   loading, and the deflection magnet are each independently switchable; the dynamic transverse
   aperture is always enforced).
6. Tag particles forward/backward/lost and compute per-screen and whole-beam summaries.
7. Save figures, per-screen and exit-beam distributions, and two consolidated JSON files
   (`run_config.json`, `run_results.json`) — the same shape from either entry point.

## Saved outputs

Both entry points write to a run directory named
`outputs/runs/<timestamp>_T<T>K_SC<on|off>_BL<on|off>/` (notebook: only when `SAVE_DATA = True`;
script: pass `--output`, or let it auto-name the same way). Inside:

- `figures/` — every diagnostic figure, each with its underlying data saved alongside so a figure
  can be reproduced later without re-running the simulation.
- `openpmd/` — the exit beam as an openPMD-beamphysics HDF5 file, reflecting whatever the dynamic
  aperture let through during tracking.
- `screen_distributions_hdf5/` — one HDF5 file per tracking screen (full, unfiltered phase space).
- `lost_particle_diagnostics.json` — particles RF-Track reports as lost during tracking.
- `run_config.json` — every input parameter (cavity/field-map, solver/finesse, cathode/emission,
  beam-loading, aperture, deflection, screen/particle-count settings) plus the handful of values
  derived from them before tracking starts (grid sizes, phase-scan crest, `Veff`, `R/Q`, effective
  length) — no per-particle or per-time-sample data anywhere in it.
- `run_results.json` — everything the run *found*: `R/Q`/`Veff` actually used, peak current and
  current density, the beam-property curves vs `z` (transmission, Twiss, beam size — one row per
  screen), particle classification, aperture and back-bombardment summaries, the openPMD exit-beam
  summary, and the paths to every other output file.
- `back_bombardment_energy_map.npz` — the 2D kinetic-energy-density map deposited by
  back-bombarding electrons at the cathode plane, when the run's `Bout` has at least one particle
  behind the cathode with a physically plausible reconstruction (see `rf_gun.back_bombardment`).
  Impacts are classified by surface (`classify_impact_surface`): the flat emitting face and its
  45deg chamfer both count toward cathode heating; anything landing further out (holder/cavity
  wall, not modeled) is excluded from heating accounting entirely.
- `back_bombardment_events.h5` — the same population as a per-event list (x, y, t, E, K, px, py,
  pz, weight, surface_id), for tools (e.g. COMSOL/TIO) that need individual impacts rather than a
  binned map.
- `emission_iteration.npz` — when `--emission-field-iteration` is on: the full (x,y,t)-resolved
  field/current history across outer iterations (too large for `run_results.json`, which instead
  gets only a scalar convergence summary).

`run_config.json`/`run_results.json` are written the same way from either entry point (via
`rf_gun.save_run_config`/`rf_gun.save_run_results`). The notebook additionally saves
`beam_properties.csv` (beam size vs. z). The script additionally saves `beam_data.npz` (the full
`B0`/`Bout`/per-screen phase-space arrays, compressed binary — the efficient, non-duplicated home
for that data; there is no JSON equivalent).

## Batch execution / SLURM

Quick validation run:

```bash
python run_thermionic_tm010.py --preset quick
```

Full option list:

```bash
python run_thermionic_tm010.py --help
```

`run_thermionic_tm010_scanT_1400_1700_fine.slurm` is the current parameter-scan template: a SLURM
job array (`--array=0-30`) sweeping the cathode temperature from 1400 K to 1700 K in 10 K steps,
one array task per temperature, at `N = 100,000` particles with space charge and beam loading on,
openPMD export on, mirror charges explicitly off, and the deflection magnet off (see the
performance note above — this keeps tracking multi-threaded). Use it as a template for other
scans (deflection current, temperature range, roughness) by varying the swept argument per array
task.

`run_thermionic_tm010_emission_iteration.slurm` runs the Emission Fields Iteration
self-consistency study on its own (production tracking disabled), at ~1000 particles, medium
finesse, 12 outer iterations, a 32³ PIC mesh, and mirror charges on — for exploring how much
space-charge/mirror feedback matters before paying for it in a full production run.

### Solver finesse presets

`--finesse {extra_fine,fine,medium,coarse}` sets every numerical-resolution setting at once (field
map grid step, integration step counts, ODE tolerance, and the space-charge/beam-loading solver
step) — a trade between run time and numerical precision, independent of the physical settings
above (particle/screen counts, cathode temperature, cavity R/Q, deflection current, and which
physics is switched on). `coarse` matches the notebook's defaults.
`run_thermionic_tm010_scanT_1400_1700_fine.slurm`, `..._medium.slurm`, and `..._extrafine.slurm`
are the same scan template at the `fine`/`medium`/`extra_fine` tiers respectively — identical
except for `--finesse` and `--scan-tags`.

The notebook has the equivalent `NOTEBOOK_FINESSE_TIER` variable near the top of its configuration
cell.

## Plotting defaults

Every phase-space figure exposes two independent switches:

- `exclude_backward_losses` — drop (vs. highlight, in grayscale) particles tagged as having
  turned backward.
- `exclude_lost` — drop (vs. highlight, in green) particles removed by the dynamic transverse
  aperture during tracking.

Both default to `True`, so the default view is the transmitted-like beam, with the
backward/lost populations available wherever a figure chooses to show them. Tagging is `%id`-based
against `Bout`'s own reliable z/pz and RF-Track's own lost-particle table
(`rf_gun.particle_tags.ParticleTags`), identical between the notebook and the script.

`plot_spectra`'s output-side panels (longitudinal kinetic energy, time-of-flight, radial position)
split the forward population by aperture transmission instead of a drop/highlight switch --
forward-transmitted vs. forward-not-transmitted, colored consistently (green always means "removed
by the dynamic aperture," the same convention used everywhere else that category is shown) -- since
these are 1D distributions of the whole output bunch rather than phase-space scatter panels.

## Running tests

```bash
python -m pytest tests/
```

Covers the emission models and their registry dispatch, work-function temperature models, cathode
field extraction (signed conventions, RF sampling, space-charge/mirror probe extraction), the
Emission Fields Iteration self-consistency study, spatially-resolved emission sampling, roughness,
momentum sampling, and beam loading's zero-`tinj` edge case. Field-map-dependent tests are skipped
automatically when `field_maps/` is not present.

## RF-Track reference

- RF-Track project page: `https://abpcomputing.web.cern.ch/codes/codes_pages/RF-Track/`
