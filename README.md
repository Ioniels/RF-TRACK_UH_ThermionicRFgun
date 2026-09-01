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
  back-bombardment/laser-heating pattern) rather than assumed uniform over the emitting disk.
- The emitted bunch is accelerated by the cavity's RF field, reconstructed from measured 2D field
  maps (an r-z cavity view and an x-z waveguide/iris view) by a shared-time-basis phasor fit that
  preserves each component's measured relative amplitude/phase (see "Field and RF-Track
  correctness" below).
- Space charge (an explicit PIC engine, with cathode mirror charges) and beam loading (the
  cavity's own R/Q, calibrated from a phase scan) are wired to act on the bunch during transport
  via RF-Track's real `BeamLoadingSW` collective effect — **though a real cross-validation found
  `BeamLoadingSW` currently produces zero measurable effect on tracked dynamics for this gun
  geometry in this RF-Track 2.7.0 binding** (see `tests/test_beam_loading_cross_validation.py` and
  "Known limitations" below); this is a real RF-Track binding/configuration limitation, not
  something this codebase can fix on its own. An optional "Emission Fields Iteration" study
  pre-converges the emission current against its own space-charge and mirror feedback (and,
  opt-in, an independent causal beam-loading envelope model) before the full production run.
- An optional deflection magnet can steer the beam, e.g. to separate back-bombarding electrons
  from the main beam.
- A dynamic transverse aperture R(z) — the cavity's real channel shape, not a single cylinder — is
  enforced by RF-Track during tracking, so a particle is removed the instant it leaves the real
  transverse channel.
- A downstream back-bombardment/macropulse study predicts cathode heating from returning electrons
  over a configurable RF macropulse, using a validated v2 ray-cast event capture and a Cartesian
  asymmetric thermal solver (see "Back-bombardment and macropulse heating study" below).

Two ways to run the same physics, producing the same kind of output:

- **`UH_gun_tracking_demo.ipynb`** — the interactive notebook, for exploring a single run.
- **`run_thermionic_tm010.py`** — a command-line script for scripted runs or SLURM parameter scans.

Both entry points resolve the same shared `rf_gun` pipeline (field preparation → RF-only phase
calibration → optional emission-field self-consistency → transport → classification → output/
validation), so a given set of physical settings produces the same physics from either one.

## Scientific scope: validated vs. exploratory

| Capability | Status | Notes |
|---|---|---|
| RF field phasor reconstruction | Validated | Shared-time-basis least-squares fit, `phasor_check` diagnostics, no independent per-component renormalization. |
| Field interpolation outside native support | Validated | Zero field outside the source mesh's convex hull; isolated interior holes repaired via KD-tree nearest lookup only. Diagnostics recorded in `run_config.json`. |
| RF magnetic field (Bphi) | **Not available** | Source `.mat` maps carry only `TotalField_E_X/Y/Z` — no measured B/H component exists to include. Results are E-only; see "Known limitations." |
| Dynamic transverse aperture R(z) | Validated | Enforced live by RF-Track's own `Aperture_1d`, identical between notebook and script. |
| Space charge + cathode mirror charge | Validated | Explicit fresh `SpaceCharge_PIC_FreeSpace` engine per run; mirror-charge unit tests and mesh-convergence tests pass, though the *peak* near-cathode field is only mesh/scale-converged to order-of-magnitude (see `rf_gun/emission_iteration.py`'s `field_probe_method` docstring). |
| RF-Track `BeamLoadingSW` | **Attached correctly, but measured to have no effect** | Constructor uses the sole documented RF-Track 2.7 signature and is verified after construction; a real A/B cross-validation on this gun's geometry found it changes tracked dynamics by an amount indistinguishable from zero (`tests/test_beam_loading_cross_validation.py`). |
| RF-only phase calibration | Validated | Genuinely on-axis, cold, negligible-charge source reused unchanged at every phase; typed `PhaseCalibrationResult` with a finite-bracketed-crest requirement; an invalid calibration aborts the run rather than producing `Veff=NaN`. |
| Emission Fields Iteration (SC+mirror, +BL opt-in) | Exploratory | Pre-converges a reduced-cost self-consistency loop; validated closed-form limits, but the causal beam-loading term is not cross-checked against real `BeamLoadingSW` (which itself has no measured effect here) and near-cathode field resolution is order-of-magnitude only. |
| Deflection magnet | Validated (geometry/sign) | `UserField.get_field()` cross-checked against the analytic profile; `applied_current_A` is exactly zero whenever the magnet is disabled, independent of the configured value. |
| Back-bombardment v2 event capture | Validated | `backstop_raycast_v1` ray-casts against the parameterized cathode geometry; charge/energy closure and an unknown-surface-fraction limit are fatal checks, not warnings. |
| Back-bombardment deposition (BB0) | Exploratory baseline | Tabata-Ito-Okabe/CSDA range-energy model only; no backscatter/secondary-electron transport (BB1/BB2 are documented, unimplemented future work, not exposed as CLI choices). |
| Macropulse heating (`L2_one_way`/`top_hat`) | Exploratory | One qualified RF-period source scaled over an idealized top-hat envelope; no temperature-to-emission or cavity feedback, no measured fill/decay waveform exists yet. |
| Cathode/holder geometry for back-bombardment | Placeholder | Flat holder boundary; heating claims are restricted to the validated LaB6 footprint and the unknown/holder fraction is reported and gated. |
| Total hemispherical emissivity | Provisional | `DEFAULT_TOTAL_HEMISPHERICAL_EMISSIVITY = 0.8`, a literature order-of-magnitude value, not a measured LaB6 dataset. |
| Mount thermal boundary | Named simplification | Adiabatic (zero contact conductance) by default; a uniform `contact_h_W_m2K` is a declared simplification, not a claim of azimuthal symmetry. |

Absolute cathode temperatures, back-bombardment power, and any quantity depending on the items
above are provisional until the underlying geometry/material/waveform data are grounded. *Relative*
trends (e.g. deflection current vs. returned charge) are more robust to these limitations than
absolute values.

## Conventions

- **Coordinates:** the cathode emitting surface is `z = 0`; `z > 0` is the downstream/vacuum side
  the beam is accelerated into. Transverse `x`, `y` are Cartesian (not assumed axisymmetric) at the
  cathode, even though the RF field map itself is axisymmetric.
- **Electron charge:** `q = -1` (elementary charge sign convention used throughout RF-Track
  `Bunch6dT` construction); `rf_gun.constants.q_e` is the positive elementary charge magnitude.
- **RF phasor:** a complex phasor `E(r,z)` such that the instantaneous field at absolute phase
  `phi` (degrees) is `Re{E(r,z) * exp(j*deg2rad(phi))}`; every component (`Er`, `Ez`, real and
  imaginary parts) shares one fitted time basis (see "Field and RF-Track correctness").
- **Radial field sign:** `Er = sign(x) * Ex` on the XY sensor plane, so a positive `Er` points away
  from the axis at `x>0` and toward it at `x<0`, matching a physically continuous radial field
  through `x=0`.
- **Deflection current polarity:** `Bx(z) = B_pk_per_A_T * I`; a positive `deflection_current_A`
  gives a positive `Bx`, with `sign` following `I` directly (`rf_gun.deflection_field`).
- **Surface-normal convention (back-bombardment):** `n_in` points from vacuum into the struck
  solid; an event is only accepted when `p_hit . n_in > 0` (inward momentum).
- **Particle weights:** `Bunch6dT`'s extended-matrix `N` column is the number of real electrons
  represented by that macroparticle; `weight_electrons * q_e` gives macro-charge in Coulombs
  throughout the codebase and saved HDF5 files.
- **Units:** SI internally in every library object; CLI flags with unit suffixes (`_um`, `_mm`,
  `_K`, `_A`, `_us`, `_ns`) convert exactly once at the argument-parsing boundary.

## Field and RF-Track correctness

- **Phasor reconstruction** (`rf_gun.phasor.build_iq_phasor`/`build_crest_phasor`): fits `Er`/`Ez`
  (real and imaginary) on one shared time basis, never independently renormalizing a component to
  its own peak — preserving the measured `Er/Ez` ratio. `phasor_check` reports peak/RMS
  reconstruction error, phase lag, and independent frequency estimates (FFT, zero-crossing,
  sinusoid fit) as a diagnostic, not merely a plot.
- **Interpolation** (`rf_gun.phasor.build_field_interpolation_context`/`interp_cfield`): the source
  Delaunay triangulation and hull-membership test are built once and reused across every field
  component sharing a grid. Outside the native convex hull the field is exactly zero (never
  nearest-neighbor-extrapolated).
- **Magnetic field:** the raw `.mat` sensor files expose only `TotalField_E_X/Y/Z` (confirmed by
  direct inspection of every top-level key) — no B/H component in these files for now.
- **`BeamLoadingSW`:** constructed with the single documented RF-Track 2.7 constructor,
  `BeamLoadingSW(SWS, Q, r_Q, Ncells, mass, q, tinj)`.
- **Volume element ordering:** `V.set_s0()`/`set_s1()` are called before the dynamic
  aperture/backstop/deflection elements are added.
- **RF-only phase calibration** (`rf_gun.simulation.run_phase_scan`,
  `rf_gun.rf_params.PhaseCalibrationResult`): the calibration source is genuinely on-axis and cold
  (`x=y=px=py=0` for every particle, negligible charge — `build_bunch_on_axis_cold`).

## Self-consistent cathode emission

- **Emission models** (`--emission_law`; see `rf_gun.emission_models` module
  docstring:
  - `RDSchottky` (was `RD_schottky`, default) — *"Thermionic: Richardson-Dushman-Schottky"* —
    classical Richardson-Dushman thermionic emission with Schottky barrier lowering.
  - `jensen2014_RDSchottky_MurphyGood_additive` (was `rld_schottky_plus_mg`/`unified`) —
    *"Jensen2014: Additive regime, Thermionic: Richardson-Dushman-Schottky; Field: Murphy-Good"* —
    an additive combination of the thermionic term with a cold Murphy-Good/Schottky-Nordheim
    field-emission term; a diagnostic comparison kernel, not a production default.
  - `jensen2019_RDSchottky_MurphyGood_transition` (was `rgtf_2019`) — *"Jensen2019: Transition
    regime, Thermionic: Richardson-Dushman-Schottky; Field: Murphy-Good"* — Jensen, *J. Appl.
    Phys.* **126**, 065302 (2019), a reformulated general thermal-field model spanning the
    thermionic/field/transition regimes; exact digamma-based series, smoothly (C¹) blended into a
    direct energy-domain integral in a narrow band around each of the series' genuine poles at
    integer regime values, rather than switching between the two discontinuously.
  - `murphygood1956_SchottkyNordheim_integral` (was `murphy_good_direct_reference`) —
    *"MurphyGood1956 Integrals of transmission probability into Schottky-Nordheim barrier"* —
    Murphy & Good, *Phys. Rev.* **102**, 1464 (1956), a direct WKB/Kemble energy-integral
    reference, useful to cross-check the closed-form laws above but too slow for routine per-run
    tracking.
  - `jensen_gtf_2007` — Jensen, *J. Appl. Phys.* **102**, 024911 (2007); registered as a
    historical/mathematical comparison point but **not yet implemented** (raises
    `NotImplementedError`) — `jensen2019_RDSchottky_MurphyGood_transition` supersedes it as the
    production general thermal-field model.

- **LaB6⟨100⟩ work function models** (`--work-function-temperature-model`): `constant_phi_eff`
  (Liu et al. 2017 thermionic anchor), `linear_tcwf` (that anchor extended with a
  temperature-coefficient-of-work-function slope), and `piecewise_surface_evolution` (a
  Swanson/Bulyga/Liu three-branch blend). Unset, `--phi_eff_ev` is used as a fixed value instead.
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
  converged current density. `--emission-iteration-nx/-ny/-nt` control the real fixed-sample size
  (`nx*ny*nt` macroparticles, all fixed positions/momenta, only weights updated each iteration); a
  formerly-accepted `--emission-iteration-particles` flag did not control this and was removed
  rather than kept as a misleading no-op. A NaN/inf appearing anywhere in the current/field/charge
  state at any outer iteration invalidates the iteration immediately (preserving that first failing
  iteration's arrays for postmortem inspection, not filling the remainder with more corrupted
  history); `converged=False` results are never labeled "converged" in the resulting figures.
  `--use-converged-iteration-source` feeds its converged `J(x,y,t)` directly into the production
  run in place of the default prescribed source when it converged; **if it did not converge, the
  run continues with the prescribed source instead and this is recorded as a failed
  `emission_field_iteration_converged` check in `validation.json`, so a production run built this
  way never receives a `.run_complete` marker** — treat a run without that marker as not
  production-valid regardless of whether it otherwise finished without a Python exception.
  - `--emission-iteration-field-probe-method`: `pic_probe` (default — zero-weight probe particles
    in RF-Track's own space-charge engine, at a fixed probe distance `--emission-field-probe-z-um`)
    or `analytic_point_charge_image` (a closed-form conductor-image kernel, evaluated exactly at
    the true cathode surface). Neither method is mesh/scale-converged for the *peak* near-cathode
    field (see `rf_gun.cathode_fields.analytic_sc_and_mirror_surface_field`'s docstring) — treat
    peak near-cathode field values as order-of-magnitude, not precision, information.
  - `--emission-iteration-include-beam-loading`: folds a causal TM010 modal-envelope beam-loading
    estimate (`E_BL(x,y,t) = -chi(t)*E_RF(x,y,t)`, `rf_gun.beam_loading_envelope`) into the
    iteration's own self-consistency loop. This is *not* a reproduction of RF-Track's own internal
    `BeamLoadingSW` discretization — it is an independent, from-first-principles implementation of
    the "fundamental theorem of beam loading" (P. B. Wilson, SLAC-PUB-2884; Wangler Sec. 4.7).
    **A real cross-check against production `BeamLoadingSW` found that `BeamLoadingSW` itself
    produces zero measurable effect on tracked dynamics for this gun geometry** (confirmed
    per-particle, across a 1000x charge range and a 100x collective-effect-step range) — treat
    `E_BL` as an independently-derived order-of-magnitude estimate, not one verified against
    RF-Track's own implementation.
- **Frozen-source physics attribution** (`rf_gun.run_frozen_source_attribution`): tracks the
  *identical* fixed-seed emitted source through up to 5 transport-physics configurations (RF only /
  RF+BL / RF+SC / RF+SC+mirror / RF+SC+mirror+BL), so any difference in the exit-beam diagnostics
  is attributable purely to that transport physics. Complements, rather than replaces, the
  Emission Fields Iteration: it measures forces/transport with a *fixed* source (no emission
  feedback), and — unlike that iteration — can genuinely include beam loading, since it uses the
  real production `BeamLoadingSW` collective effect during full tracking.
- **Spatially-resolved emission sampling** (`--spatial-emission-sampling`): samples the production
  bunch jointly in (x,y,t) from the RF field alone (no space-charge/mirror feedback), instead of
  the default independent radius-uniform/on-axis-F(t) model. Superseded by
  `--use-converged-iteration-source` when both are set and the iteration converged.
- **Emission radius** (`--r_cathode_mm`, default `2.80/2=1.40mm`, the physical flat-face radius
  excluding the 0.2mm bevel/chamfer): with this physical value, space-charge/mirror fields barely
  perturb the cathode field in this gun's normal operating regime. A separate, explicitly-labeled
  bevel-inclusive stress-test configuration (`r_cathode_mm=1.6`) is documented for exercising
  SC/mirror sensitivity in the code; it is not a physical production value.

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

## Environment and installation

- Python 3.12 (the tracked venv was built with `python3.12 -m venv .venv`).
- RF-Track 2.7.0, from PyPI: `pip install rf-track==2.7.0` (CERN-licensed; requires the PyPI index
  to be reachable/authorized for your account). `python -c "import RF_Track; print(RF_Track.version)"`
  should print `2.7.0` after activation.
- Remaining dependencies: `pip install numpy scipy matplotlib pandas h5py ipykernel ipywidgets
  pytest openpmd-beamphysics pyyaml tqdm psutil requests`.
- `RF_TRACK_NO_UPDATE_CHECK=1` and an explicit `RF_TRACK_NUMBER_OF_THREADS` are recommended
  environment variables for batch/cluster use (`rf_gun.config.set_thread_environment` sets the
  latter, plus matching `OMP`/`OPENBLAS`/`MKL`/`NUMEXPR` thread variables, consistently for every
  run so BLAS and RF-Track never disagree about how many threads are available).

## Repository layout

```text
.
├── .gitignore
├── README.md
├── UH_gun_tracking_demo.ipynb
├── run_thermionic_tm010.py                # thin CLI: single-configuration production run
├── run_back_bombardment_macropulse.py     # thin CLI: back-bombardment/macropulse study
├── KOA_slurm_scripts/                     # the five production array-job scripts (see below)
│   ├── study_i_medium_field_grid.slurm
│   ├── study_i_bis_fine_field_grid.slurm
│   ├── study_ii_finesse.slurm
│   ├── study_iii_temperature.slurm
│   └── study_iv_deflection_macropulse.slurm
├── field_maps/
│   ├── XYplanarSensorData.mat
│   └── YZplanarSensorData.mat
├── tests/                       # pytest suite (untracked locally, see "Running tests")
└── rf_gun/
    ├── config.py                   # connects to RF-Track, thread-environment setup
    ├── constants.py                # physical constants, unit conversions
    ├── helpers.py                  # small numeric/formatting utilities
    ├── rf_params.py                # R/Q, delivered power, effective length, PhaseCalibrationResult
    ├── field_io.py                 # reads field maps, mesh statistics
    ├── phasor.py                   # RF phasor construction, interpolation, phasor_check
    ├── emission_models.py          # thermionic and field-emission current models
    ├── emission_sensitivity.py     # analytic/numeric sensitivity of emission models
    ├── emission_sampling.py        # samples emission phase space (thermal, roughness)
    ├── work_function_models.py     # LaB6<100> phi_eff(T) models
    ├── cathode_fields.py           # signed fields, RF sampling, SC/mirror probe extraction
    ├── emission_iteration.py       # self-consistent Emission Fields Iteration study
    ├── beam_loading_envelope.py    # causal TM010 modal-envelope beam-loading estimate
    ├── frozen_source_attribution.py  # fixed-seed, varying-transport-physics comparison
    ├── deflection_field.py         # steering-magnet field model (UserField)
    ├── finesse_presets.py          # numerical-resolution presets (independent of field grid)
    ├── rftrack_volume.py           # builds the RF-Track tracking volume, screens, SC engine
    ├── simulation.py               # runs the tracking, reports stage-by-stage progress
    ├── diagnostics.py              # beam size, emittance, per-screen summaries
    ├── particle_tags.py            # forward/backward and aperture tagging
    ├── aperture.py                 # dynamic transverse aperture R(z), cathode backstop
    ├── beam_properties.py          # beam size and transmission vs. distance
    ├── cathode_geometry.py         # flat/bevel/side/holder surface parameterization
    ├── backstop_loss_separation.py # separates backstop hits from dynamic-aperture losses
    ├── back_bombardment.py         # legacy far-downstream Bout drift reconstruction
    ├── back_bombardment_events.py  # v2 ray-cast event capture, HDF5 schema, accounting
    ├── back_bombardment_deposition.py  # BB0 TIO/CSDA volumetric deposition
    ├── back_bombardment_study_config.py  # study/capture/deposition/thermal config dataclasses
    ├── macropulse.py                # RF/thermal time grids, current histories
    ├── thermal.py                   # Cartesian layered (x,y) thermal solver
    ├── comsol_io.py                 # COMSOL source export / result import (interface only)
    ├── materials/                   # named cathode material property sets (LaB6, ...)
    ├── studies/back_bombardment_macropulse.py  # shared study orchestration
    ├── acceptance_scan.py           # separates the beam core from its trailing tail
    ├── io.py                        # saves results to disk, atomic JSON writers, validation.json
    └── plotting/                    # figures (field maps, phase space, evolution, ...)
```

`.venv/` (the Python environment, including the compiled RF-Track binding) and local-only/generated
folders (`outputs/`, `Koa outputs/`, `analysis_Koa/`, `manual_references/`, `archive/`, `logs/`,
`Upgrade_history/`) are intentionally outside this layout — see `.gitignore`.

## Core workflow

1. Load the XY/YZ field maps and compute envelope diagnostics.
2. Build the shared-time-basis RF phasor and interpolate it onto an (r, z) grid (zero outside the
   native convex hull).
3. Run the RF-only phase scan (on-axis, cold source; every other physics switch off) to calibrate
   effective voltage and R/Q; abort if the calibration is not valid.
4. Optionally run the Emission Fields Iteration self-consistency study near the cathode.
5. Build the thermionic bunch (on-axis, or jointly in (x,y,t) — see "Self-consistent cathode
   emission" above) and track it through RF-Track (space charge and cathode mirror charges, beam
   loading, and the deflection magnet are each independently switchable; the dynamic transverse
   aperture is always enforced).
6. Tag particles forward/backward/lost and compute one final per-screen and whole-beam
   classification, reused by every downstream count/plot/output.
7. Save figures, per-screen and exit-beam distributions, and validate the run before writing a
   completion marker (see "Output schema and provenance" below).

## Back-bombardment and macropulse heating study

A separate study, downstream of a completed transport run, predicts cathode heating from
electrons that return and strike the cathode ("back-bombardment") over a configurable RF
macropulse (8 µs default). It reuses one qualified representative-RF-period event capture across
cheap reprocessing of pulse duration, material set, deposition model, and thermal backend, instead
of repeating RF-Track transport for every such change.

**Probing strategy.** Production event capture (`event_locator="backstop_raycast_v1"`) enables a
thin absorbing `Aperture_1d` cathode backstop (`--cathode_backstop_enabled` on
`run_thermionic_tm010.py`, `rf_gun.aperture.build_cathode_backstop`) just behind the cathode
plane, reads RF-Track's own `Volume.get_lost_particles()` loss table, separates genuine backstop
hits from unrelated dynamic-aperture losses, and ray-casts each candidate against the
parameterized flat face/bevel/holder (`rf_gun.cathode_geometry`) to find the true surface
intersection, incidence angle, and normal. This replaces long, residual-space-charge-sensitive
extrapolation from the far-downstream `Bout` drift reconstruction, which remains available only as
`event_locator="legacy_ballistic"` for comparison.

**Validation gates (fatal, not warnings):** `rg.validate_back_bombardment_study` requires exact
launched/returned/transmitted/other charge closure, BB0 incident/deposited/escaped energy closure,
a thermal energy residual within `config.thermal.energy_residual_tol`, a coupling level matching
what was actually computed (`L2_one_way`), and an unknown-surface-fraction (`SURFACE_UNKNOWN`
ray-cast misses) below `config.capture.max_unknown_surface_fraction` (1% default) — exceeding any
of these raises rather than continuing with an unvalidated result.

**File contracts** (all schema-versioned HDF5, each rejecting an older/foreign file rather than
guessing at its content):

- `back_bombardment_events.h5` (`schema_version="back_bombardment_events_v2"`) — the immutable,
  qualified per-event record (impact position/time/momentum/energy, surface zone, incidence angle,
  quality flags) plus accounting, geometry, and provenance groups.
- `back_bombardment_heat_source.h5` — BB0 volumetric deposited-energy source (`q'''_layer(x,y,layer)`)
  for one representative RF period, keyed to its source event file's hash. BB1 (uncertainty
  scanning)/BB2 (Geant4 response library) are documented, `NotImplementedError`-raising
  placeholders and are not exposed as CLI choices until implemented.
- `back_bombardment_macropulse.h5` (`schema_version="back_bombardment_macropulse_v1"`) — macropulse
  current/power histories, the Python thermal solution's scalar time series, an optional COMSOL
  comparison, and benchmark/closure metrics.
- `study_config.json` / `study_results.json` — the resolved study configuration and scalar summary.

**Implementation status.** This delivers the material registry and LaB6 property files,
authoritative backstop/ray-cast event capture and HDF5 v2, BB0 (TIO/CSDA) deposition, the
asymmetric `python_xy_layered` Cartesian thermal solver, and the CLI/SLURM layer — a complete
`coupling_level=L2_one_way` study (one qualified source, no temperature-to-emission feedback) at
the 8 µs default. COMSOL exchange is interface-only: `rf_gun.load_comsol_thermal_result`/
`rf_gun.compare_python_comsol_thermal` exist and are wired in, but no COMSOL run exists yet, so
comparison output always reports `comsol_available=False` rather than fabricating placeholder
data. Thermal/emission/cavity feedback (L3/L4) is explicitly out of scope for this pass.

**Example commands.** In the notebook, the back-bombardment section toggles between consuming the
just-completed run in memory and loading a saved one:

```python
BB_SOURCE_MODE = "current_notebook"  # or "load_run"
BB_RUN_DIR = None                    # required only for "load_run"
bb_input = rg.resolve_back_bombardment_study_input(
    source_mode=BB_SOURCE_MODE,
    current_events=bb_events if BB_SOURCE_MODE == "current_notebook" else None,
    run_dir=BB_RUN_DIR if BB_SOURCE_MODE == "load_run" else None,
)
```

From the command line, against a completed run directory (only `--source-mode load_run` is
practical outside a live Python session):

```bash
# Default 8 us Python study from a completed run
python run_back_bombardment_macropulse.py \
  --source-mode load_run \
  --run-dir outputs/runs/production_bb \
  --thermal-backend python_xy_layered \
  --material-set LaB6_UH_recommended_v1 \
  --deposition-model BB0_TIO \
  --macropulse-duration-us 8 \
  --thermal-bin-ns 50 \
  --initial-temperature-uniform-k 1650 \
  --output outputs/runs/production_bb \
  --save-figures

# Cheap reprocessing of the same impacts for a different pulse duration / thermal bin width
python run_back_bombardment_macropulse.py \
  --source-mode load_run --run-dir outputs/runs/production_bb \
  --macropulse-duration-us 12 --thermal-bin-ns 20 --initial-temperature-uniform-k 1650 \
  --output outputs/runs/production_bb_12us
```

`--thermal-bin-ns` (macro-time thermal bin width) and `--thermal-dt-ns` (implicit-solve sub-step,
decoupled from the bin width) are CLI-level controls over what were previously library-only
parameters (`thermal_bin_s`, `ThermalConfig.dt_s`).

## Output schema and provenance

Both entry points write to a run directory named
`outputs/runs/<timestamp>_T<T>K_SC<on|off>_BL<on|off>/` (notebook: only when `SAVE_DATA = True`;
script: pass `--output`, or let it auto-name the same way). Inside:

- `screen_distributions_hdf5/` — the canonical location for every particle-distribution HDF5
  output: `B0_*.h5` (as-launched distribution), `Bout_*.h5` (final, forward-going,
  dynamic-aperture-surviving state), and, when `--save-screen-hdf5` is set, one file per tracking
  screen (full, unfiltered phase space). There is no separate `openpmd/` directory and no
  `beam_data.npz` — those were retired as redundant duplicates of exactly this content.
- `figures/` — every diagnostic figure, each with its underlying data saved alongside so a figure
  can be reproduced later without re-running the simulation.
- `run_config.json` — every input parameter (cavity/field-map, solver/finesse, cathode/emission,
  beam-loading, aperture, deflection, screen/particle-count settings) plus the values derived from
  them before tracking starts (grid sizes, phase-scan crest, `Veff`, `R/Q`, effective length, and
  `field_provenance`: source map hashes/mesh stats, phasor/interpolation diagnostics, and the
  E-only magnetic-field-source status) — no per-particle or per-time-sample data anywhere in it.
- `run_results.json` — everything the run *found*: `R/Q`/`Veff` actually used, peak current and
  current density, the beam-property curves vs `z` (transmission, Twiss, beam size — one row per
  screen), particle classification, aperture and back-bombardment summaries, the exit-beam
  summary, and the paths to every other output file.
- `validation.json` — a machine-readable pass/fail report: phase-calibration validity, particle-ID
  uniqueness in `Bout`, finiteness of every derived RF parameter, the field-interpolation
  outside-hull fraction against a threshold, and (when run) Emission Fields Iteration convergence.
  `status` is `"ok"` only if every check passed.
- `.run_complete` — written **only** when `validation.json`'s status is `"ok"`; its absence means
  at least one physics/validation gate failed, regardless of whether the process otherwise exited
  without a Python exception. SLURM scripts check for this file and fail the job if it is missing.
- `back_bombardment_energy_map.npz` / `back_bombardment_events.h5` — the legacy far-downstream
  reconstruction (see "Back-bombardment and macropulse heating study" above for the canonical v2
  path, which supersedes this for production heating claims).
- `lost_particle_diagnostics.json` — particles RF-Track reports as lost during tracking.
- `emission_iteration.npz` — when `--emission-field-iteration` is on: the full (x,y,t)-resolved
  field/current history across outer iterations (too large for `run_results.json`, which instead
  gets only a scalar convergence summary).

`run_config.json`/`run_results.json`/`validation.json` are written atomically (a temporary file
plus `os.replace`), so a killed/crashed write can never leave a half-written file in place, and are
identical in shape from either entry point. `run_dir` is recorded as given (not resolved to an
absolute, machine-specific path), so metadata stays portable across machines/accounts.

## Field-grid resolution vs. solver finesse

These are two independent axes, and neither overrides the other:

- **Field-grid resolution** (`--dr_um`/`--dz_um`, default `4.0`/`13.0`): the physical spacing of
  the interpolated `(r,z)` field grid. An explicit value always wins; selecting a `--finesse` tier
  never touches it (`rf_gun.finesse_presets.finesse_preset_dict` deliberately excludes these keys).
- **Solver finesse** (`--finesse {extra_fine,fine,medium,coarse}`): every *numerical-integration*
  resolution setting at once — field-map integration step counts, ODE tolerance, and the
  space-charge/beam-loading/phase-scan solver step sizes. A trade between run time and numerical
  precision, independent of field-grid resolution and of every physical setting (particle/screen
  counts, cathode temperature, cavity R/Q, deflection current, which physics is switched on).
  `coarse` matches the notebook's defaults; the notebook has the equivalent `NOTEBOOK_FINESSE_TIER`
  variable near the top of its configuration cell.

## KOA SLURM production suite

`KOA_slurm_scripts/` contains five array-job scripts, each calling the same
`run_thermionic_tm010.py` "everything ON except deflection" production profile (RF field with the
validated phase calibration, physical aperture + cathode backstop with v2 event capture, fresh-PIC
space charge, cathode mirror charge, `BeamLoadingSW` attached, the converged Emission Fields
Iteration source, screens, full HDF5/JSON output, and figures) and overriding only the study
variable(s), at `N_PARTICLES=300000` and a common `--seed 42`:

| Script | Array | Varies | Finesse | Field grid |
|---|---|---|---|---|
| `study_i_medium_field_grid.slurm` | 0-9 | `dr_um`/`dz_um` (geometric ladder, index 1 = current default) | medium | varies (the study variable) |
| `study_i_bis_fine_field_grid.slurm` | 0-9 | same ladder | fine | varies |
| `study_ii_finesse.slurm` | 0-3 | `coarse,medium,fine,extra_fine` | varies (the study variable) | fixed 4/13um |
| `study_iii_temperature.slurm` | 0-20 | `T_K = 1600 + 10*task_id` (1600-1800K) | fine | fixed 4/13um |
| `study_iv_deflection_macropulse.slurm` | 0-9 | deflection current (0.0-1.0A, 10 equal steps) | fine | fixed 4/13um |

Study IV additionally enables the deflection UserField and forces single-threaded tracking for
every current including 0A (the element stays attached at zero amplitude, isolating current
amplitude from code-path/threading changes), then runs the macropulse postprocessor for 10 µs in a
separate subdirectory so it cannot overwrite the transport stage's own event file.

Every script:

- resolves the repository from `SLURM_SUBMIT_DIR` (or fails clearly) rather than a hardcoded path;
- activates a documented, overridable venv path (`RFTRACK_VENV_ACTIVATE`) and fails loudly if
  RF-Track is not importable, rather than silently proceeding;
- validates `SLURM_ARRAY_TASK_ID` before indexing into any per-case list;
- prints scheduler IDs, resolved case parameters, Python/RF-Track/numpy versions, git commit and
  clean/dirty state, hostname, and the thread environment before doing any heavy work;
- writes a unique, deterministic case directory and refuses to overwrite one that already has a
  `.run_complete` (or, for Study IV's second stage, `.macropulse_complete`) marker;
- captures wall time and peak resident memory via `/usr/bin/time -v` into
  `<run_dir>/resource_usage.txt`;
- fails the SLURM job (non-zero exit) if the expected completion marker is missing, so a
  physics/validation gate failure is visible in `sacct`, not just in a log file.

**The log directory must exist before `sbatch`** (SLURM opens `--output`/`--error` before the job
body runs): run `mkdir -p logs outputs/runs` once before submitting any of these scripts.

**The resource requests (`--cpus-per-task`, `--mem`, `--time`) and array concurrency limits
(`%N`) in every script are placeholders.** They have not been tuned against real KOA pilot runs
(this repository has no cluster access to do so) — measure your own from one coarse pilot, the
current 4/13um pilot, and the one finer 3.2/10.4um pilot before submitting a full array, and update
the scripts with real values. Study IV's `--thermal-bin-ns` (10ns initial production target) is
similarly unvalidated for this study until the mandatory 20/10/5ns pilot convergence comparison
described in that script's header comment has been run.

Quick single-configuration validation run (not a SLURM script):

```bash
python run_thermionic_tm010.py --preset quick
```

Full option list: `python run_thermionic_tm010.py --help` /
`python run_back_bombardment_macropulse.py --help`.

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

The deflection field profile figure (`rf_gun.plotting.fields.plot_deflection_field_profile`) is
skipped, with a printed reason, whenever the applied current is zero — a disabled magnet with a
nonzero *configured* current never produces a plotted (nonzero-looking) field.

## Known limitations

- **E-only RF field.** No measured or independently-derived magnetic map exists for this cavity;
  all results omit the RF azimuthal `Bphi`. See "Field and RF-Track correctness."
- **`BeamLoadingSW` has no measured effect** on tracked dynamics for this gun geometry in this
  RF-Track 2.7.0 binding (confirmed by direct A/B cross-validation). The element is attached
  correctly per the documented API; the finding is about the binding's behavior, not this
  codebase's usage of it.
- **Back-bombardment geometry** uses a placeholder flat holder boundary; only the LaB6 flat+bevel
  footprint is validated for quantitative heating, and the unknown/holder charge and energy
  fractions are reported and gated separately.
- **BB0 deposition** is a CSDA/TIO range-energy baseline with no backscatter or secondary-electron
  transport — treat deposited-energy maps as a lower bound on true surface heating uncertainty.
- **Total hemispherical emissivity (0.8)** and the **adiabatic mount boundary** are named,
  literature-order-of-magnitude/simplification choices, not measured LaB6 data.
- **Macropulse heating is `L2_one_way`/`top_hat`**: one qualified RF-period source scaled over an
  idealized top-hat envelope, with no temperature-to-emission or cavity feedback and no measured
  fill/decay waveform. Do not present it as validated absolute heating.
- **Emission Fields Iteration near-cathode fields** are order-of-magnitude, not precision,
  information — neither the PIC-probe nor the analytic-image method is mesh/scale-converged for
  the peak near-cathode field.
- **KOA SLURM resource sizing** is unvalidated placeholder data (see "KOA SLURM production suite").

## Running tests

```bash
python -m pytest tests/
```

Covers the emission models and their registry dispatch, work-function temperature models, cathode
field extraction (signed conventions, RF sampling, space-charge/mirror probe extraction), the
Emission Fields Iteration self-consistency study (including NaN-abort behavior and its
causal-envelope `include_beam_loading` feedback), the causal beam-loading envelope module, a
real-field-map cross-validation against production `BeamLoadingSW`
(`test_beam_loading_cross_validation.py` — documents/guards the finding that it has no measurable
effect on tracked dynamics in this binding), the RF-only phase calibration (on-axis source,
`PhaseCalibrationResult` validity gates), the `Volume.set_s0()/set_s1()` element-ordering
regression, the deflection field's `UserField` query against its analytic profile, back-bombardment
event capture/macropulse study validation gates (including the unknown-surface-fraction fatal
check), the frozen-source physics attribution helper's structural guarantees, spatially-resolved
emission sampling, roughness, and momentum sampling. Field-map-dependent tests are skipped
automatically when `field_maps/` is not present.

## References

- RF-Track project page: `https://abpcomputing.web.cern.ch/codes/codes_pages/RF-Track/`
- Murphy, E. L. & Good, R. H. Jr. "Thermionic Emission, Field Emission, and the Transition Region."
  *Phys. Rev.* **102**, 1464 (1956).
- Jensen, K. L. "Exchange-correlation, dipole, and image charge potentials for electron sources:
  Temperature and field variation of the tunneling barrier." *J. Appl. Phys.* **102**, 024911
  (2007).
- Jensen, K. L. et al. "A reformulated general thermal-field emission equation." *J. Appl. Phys.*
  **126**, 065302 (2019).
- Liu, H. et al. (2017) — LaB6⟨100⟩ thermionic work-function anchor.
- Wilson, P. B. "Fundamentals of RF Superconductivity and Beam Loading." SLAC-PUB-2884 (1982).
- Wangler, T. *RF Linear Accelerators*, 2nd ed., Sec. 4.7 (beam loading).
- Bakr, M. et al. "Electron beam energy deposition study for LaB6 thermionic cathode." *Phys. Rev.
  ST Accel. Beams* **14**, 060708 (2011) (Tabata-Ito-Okabe/CSDA stopping-power coefficients for
  LaB6 back-bombardment deposition, BB0).
