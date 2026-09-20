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
- The emitted bunch is accelerated by a compact, qualified axisymmetric E+B artifact prepared
  once from the cathode-centred 3D Remcom XFdtd E/H volume. The out-of-core treatment fits every
  saved time sample on one RF basis, converts H to B in vacuum, diagnoses azimuthal modes, and
  exports the cylindrical m=0 fields used by RF-Track (see "XFdtd field-map preparation" below).
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

Two ways to run the same tracking physics, producing the same kind of output:

- **`UH_gun_tracking_demo.ipynb`** — the interactive notebook, for exploring a single run.
- **`run_thermionic_tm010.py`** — a command-line script for scripted runs or SLURM parameter scans.

For code ownership, notebook use, reusable diagnostics and development checks,
see the [repository guide](docs/repository_guide.md). Expanded notebook equations
are in the [physics reference](docs/notebook_physics_reference.md).

The expensive source-map treatment is deliberately separate: **`XFdtd_field_map_preprocessing.ipynb`**
documents the analysis and **`prepare_xfdtd_fieldmap.py`** creates the immutable artifact. Both
tracking entry points then resolve the same shared `rf_gun` pipeline (verified artifact load →
RF-only phase calibration → optional emission-field self-consistency → transport → classification
→ output/validation), so a given artifact and set of physical settings produce the same physics.

## Scientific scope: validated vs. exploratory

| Capability | Status | Notes |
|---|---|---|
| XFdtd E/H volume reduction | Qualified artifact gate | All E and H snapshots are fitted on one fixed-frequency basis in bounded slabs; H is converted to B only in the guarded vacuum mask. Source, processing and payload digests are recorded and verified before production loading. |
| RF field outside measured support | Qualified modal extension | The common six-component volume ends at 32.9488 mm. Its below-cutoff pipe tail passes endpoint, decay, phase and missing-voltage gates before the explicitly marked extension to the 40.589 mm tracking end. |
| RF magnetic field (Btheta/Bz) | Available in production | The new XFdtd volume supplies H. Production uses its cylindrical m=0 B projection with `--rf-magnetic-field artifact`; `zero` is an explicit E-only control. |
| Absolute RF normalization | Explicit common E/B scaling | The solver fields are recorded at 1.8 MW. Tracking scales every E and B phasor by the same `sqrt(P_target/1.8 MW)` factor; the default and KOA baseline explicitly use 1.0 MW, consistent with the delivered-power and R/Q calculation. |
| Cylindrical-symmetry approximation | Declared production policy | E and B azimuthal spectra are reported; B is noticeably less axisymmetric. Non-m=0 content is diagnostic, not a qualification veto, and RF-Track still receives only the m=0 projection. |
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
| LaB6 `cp`/`k` above the tables | Extrapolated with a warning | Both continue their last tabulated segment from 2000 K to the 2483 K melting point (`extrapolation: linear_to_limit`), warning each time; past melting both still fail hard, since the solid-phase model genuinely stops being defined there. Before this, a <0.3 K overshoot past the last knot killed all 10 KOA Study IV macropulse cases. |
| Mount thermal boundary | Named simplification | Adiabatic (zero contact conductance) by default; a uniform `contact_h_W_m2K` is a declared simplification, not a claim of azimuthal symmetry. |

Absolute cathode temperatures, back-bombardment power, and any quantity depending on the items
above are provisional until the underlying geometry/material/waveform data are grounded. *Relative*
trends (e.g. deflection current vs. returned charge) are more robust to these limitations than
absolute values.

## Conventions

- **Coordinates:** the cathode emitting surface is `z = 0`; `z > 0` is the downstream/vacuum side
  the beam is accelerated into. The XFdtd-to-beam transform and cathode-centred origin are stored
  in the artifact. Transverse `x`, `y` remain Cartesian for particles and cathode physics; only the
  RF field supplied to RF-Track is projected to cylindrical m=0.
- **Electron charge:** `q = -1` (elementary charge sign convention used throughout RF-Track
  `Bunch6dT` construction); `rf_gun.constants.q_e` is the positive elementary charge magnitude.
- **RF phasor:** the artifact uses `Re(F exp(+i 2 pi f (t-t_ref)))`; every E and H component is
  fitted at the same frequency/reference time before cylindrical projection, preserving their
  relative amplitude and phase.
- **Cylindrical vectors:** positive `Er` points away from the axis and positive `Btheta` follows
  the right-handed beam-frame convention. Axis regularity is enforced (`Er=Btheta=0` at `r=0`).
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

- **Phasor reconstruction:** the XFdtd reducer fits all saved E and H samples at the solver drive
  frequency on one shared convention, without per-component peak normalization. Residual, drift
  and phase diagnostics are stored in the artifact report.
- **Vacuum and metal:** staggered Yee components are collocated before projection. The analytic
  aperture plus guard cells define the vacuum-quality region; fields are zero in metal, and H is
  converted to B with `B=mu0 H` only for vacuum samples.
- **Axisymmetry:** Fourier-mode diagnostics retain the evidence that the magnetic field is less
  axisymmetric than the electric field. This is shown in the preprocessing notebook, while the
  production contract deliberately exports only cylindrical-vector m=0 (`Er`, `Ez`, `Btheta`,
  `Bz`) for RF-Track.
- **Artifact integrity:** `run_thermionic_tm010.py --field-artifact ...` accepts only a `qualified`
  artifact, verifies its complete payload digest, and takes the frequency, r/z mesh, aperture
  profile and E/B phasors from it. `--rf-magnetic-field zero` is retained solely as a controlled
  E-only comparison. The ordinary loader additionally requires the `modal` tail variant; a
  zero-tail artifact is accepted only with explicit `--field-tail-policy zero-control`.
- **Power normalization:** the artifact preserves the solver's absolute 1.8 MW normalization.
  `--rf-forward-power-w` applies one common square-root power factor to `Er`, `Ez`, `Btheta`, and
  `Bz`. The same requested forward power is then used for delivered-power and BeamLoadingSW/R/Q
  calibration, preventing field amplitudes and reported RF parameters from referring to different
  powers. The historical `--bl_p_fwd_w` spelling remains a compatibility alias.
- **Legacy path retired:** omitting `--field-artifact` is now a hard error. The planar `.mat`
  files contain E only and are not a qualified input, so falling back to them silently would
  produce an E-only run against unqualified data. `--allow-legacy-mat-fieldmap` is the explicit
  opt-in if that path is ever needed.
- **`BeamLoadingSW`:** constructed with the single documented RF-Track 2.7 constructor,
  `BeamLoadingSW(SWS, Q, r_Q, Ncells, mass, q, tinj)`.
- **Volume element ordering:** `V.set_s0()`/`set_s1()` are called before the dynamic
  aperture/backstop/deflection elements are added.
- **RF-only phase calibration** (`rf_gun.simulation.run_phase_scan`,
  `rf_gun.rf_params.PhaseCalibrationResult`): the calibration source is genuinely on-axis and cold
  (`x=y=px=py=0` for every particle, negligible charge — `build_bunch_on_axis_cold`).

## XFdtd field-map preparation

> `docs/upgrade_2026_09_axisymmetric_eb.md` records what this upgrade changed, which earlier
> results it invalidates, and what is still open. Read it before comparing against any result
> produced under `KOA results/`.

The raw `field_maps/cavity_E_H_full_volume.h5` file is 24.8 GB, so neither production tracking
nor the main notebook reads it. The dedicated preprocessing code streams bounded slabs, extracts
important cross-sections, fits the E/H time series, checks mesh/origin/orientation, Maxwell and
axis regularity diagnostics, quantifies azimuthal modes, applies the physical-vacuum mask, and
writes `field_maps/cavity_axisymmetric_m0_rftrack.h5` atomically with source/configuration/payload
digests.

```bash
python prepare_xfdtd_fieldmap.py \
  --source field_maps/cavity_E_H_full_volume.h5 \
  --output field_maps/cavity_axisymmetric_m0_rftrack.h5 \
  --tail-policy modal \
  --fill-correction single-pole --fill-start-time-ns 0
```

The common six-component measured support ends at `z=32.9488 mm`, 7.6402 mm before the physical
tracking end. This is already deep in the exit-pipe evanescent tail: at the endpoint the maxima are
0.1431% of the map peak for E and 0.0207% for B, while time-averaged energy per unit length is
`4.02e-8` of its peak. The fitted decay length is 1.052860 mm versus 1.053112 mm for below-cutoff
circular-waveguide TM01 (`R^2=0.9999694`); the qualified modal extension to `z=40.589 mm` adds
approximately 0.0113% of the measured on-axis voltage integral. The archived v1 values
were 0.1965 kV and 1736.56 kV; the current JSON report contains the updated amplitudes. The extension is
explicitly marked in provenance, and a zero-tail artifact can be written in the same source pass
for a regression control. That control cannot be selected accidentally: production defaults
require artifact variant `modal`; tracking the control requires `--field-tail-policy zero-control`
and its distinct digest.

**The exported frames are not a converged solution.** Raw traces confirm approximately
0.25% amplitude growth per RF cycle. The original 2.127810x multiplier assumed an ideal
step-driven cavity and used the run end instead of the phasor reference time. Correcting the
reference alone gives 2.098003x; using signed E/H amplitude growth gives **2.087319x** in the
full-source rebuild. The active maps therefore have **1.903% lower field amplitude** than the
previous corrected maps. This remains a provisional model, not an independently measured
steady-state calibration. Previous artifacts are preserved in `field_maps/fill_audit_v1_archive/`.

`--fill-correction auto` now preserves the samples and produces an analysis-only artifact for
this transient. Select `--fill-correction single-pole --fill-start-time-ns 0` to make the existing
approximation explicit, or `--fill-correction off` for a raw-amplitude tracking control. The
notebook shares the same treatment. Policy, assumptions and applied factor enter the processing
digest. The nominal 1.8 MW port-power convention remains unverified. See
[the fill audit](docs/field_map_fill_audit.md) for evidence and reproduction commands.

`XFdtd_field_map_preprocessing.ipynb` presents the source metadata, cross-sections, time fit,
E-versus-B symmetry spectra, quality checks, selected m=0 maps and artifact round trip. In
particular it preserves the magnetic non-axisymmetry as a visible diagnostic; it does not change
the declared axisymmetric RF-Track production approximation.

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
- The tracking domain ends at the downstream face of the last narrow tube section
  (`rf_gun.aperture.L_END_MM = 40.589 mm` from the start of the chamfer, shifted by
  `--delta_cathode_chamfer_mm`; `rf_gun.aperture.exit_tube_end_z_m`). This is where the modelled
  structure stops, so it is where `Bout`/`s_out` and the last screen sit. It replaces an earlier
  `λ/4 + 7.5 mm = 33.742 mm` rule, which had no geometric meaning and stopped every run 6.8 mm
  short of the real exit. `--z_max` overrides it; `--ext_zmax` (default 0) extends past it, and
  the runner warns when either does, since beyond that face there is no channel for the field map
  to represent and the dynamic aperture keeps applying a tube radius that no longer exists.
  The measured six-component XFdtd volume itself ends at 32.9488 mm; production artifacts bridge
  only the remaining 7.6402 mm with the qualified, provenance-marked evanescent modal tail
  described above.
- With `--n_screens N`, the last screen sits exactly on that face, so every run carries an
  exit-face diagnostic. `Bout` does **not**: it is a fixed-*time* snapshot at `--t_max_mm`
  (default 2000 mm/c), and tracking does not stop at the domain end — surviving particles drift
  field-free to roughly 1–1.5 m. Use the last screen, not `Bout`, for exit-beam emittance, energy
  spread and transmission; `s_out_m` in the openPMD metadata names the end of the structure, not
  the position of the particles in the file.
- An emission model gives the material's *available* supply current; the self-consistent fields
  (via the Emission Fields Iteration) determine how much of it is actually extracted and
  transported. A strongly virtual-cathode-limited regime may need online emission inside RF-Track's
  own time loop rather than this outer iteration.
- Schottky/Murphy-Good barrier lowering (in the emission law) is a microscopic single-electron
  image-potential effect; the cathode mirror charge is the macroscopic field of the whole emitted
  bunch reflected in the cathode conductor. These are distinct physical scales and both should be
  kept when used together.

## Environment and installation

- Python 3.12 in the local `.venv`; the environment itself is not versioned.
- RF-Track 2.7.0, from PyPI: `pip install rf-track==2.7.0` (CERN-licensed; requires the PyPI index
  to be reachable/authorized for your account). `python -c "import RF_Track; print(RF_Track.version)"`
  should print `2.7.0` after activation.
- Project, notebook and test dependencies: `python -m pip install -e '.[notebook,test]'`.
  `pyproject.toml` declares the NumPy 2.x minimum and includes the material datasets in builds.
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
├── pyproject.toml                         # packaging + pytest testpaths/pythonpath settings
├── XFdtd_field_map_preprocessing.ipynb    # dedicated 24.8 GB volume analysis/qualification
├── prepare_xfdtd_fieldmap.py              # out-of-core H5 -> qualified compact artifact
├── run_thermionic_tm010.py                # thin CLI: single-configuration production run
├── run_back_bombardment_macropulse.py     # thin CLI: back-bombardment/macropulse study
├── verify_koa_run.py                      # fail-closed completion/identity gate for the launchers
├── docs/
│   ├── field_map_pipeline.md              # artifact contract, tail policy, fill correction
│   └── upgrade_2026_09_axisymmetric_eb.md # what the E+B upgrade changed and what it invalidates
├── tools/                                 # standalone studies; nothing in production depends on them
│   ├── thermal_bin_convergence.py         # 20/10/5 ns thermal macro-bin convergence pilot
│   └── build_grid_convergence_artifacts.py  # builds the Study I field-grid artifact ladder
├── KOA_slurm_scripts/                     # the six production array-job scripts (see below)
│   ├── study_i_medium_field_grid.slurm
│   ├── study_i_bis_fine_field_grid.slurm
│   ├── study_ii_finesse.slurm
│   ├── study_iii_temperature.slurm
│   ├── study_iii_bis_temperature_1500_1700K.slurm
│   └── study_iv_deflection_macropulse.slurm
├── field_maps/
│   ├── cavity_E_H_full_volume.h5           # local 24 GB XFdtd source, not tracked
│   ├── cavity_axisymmetric_m0_rftrack.h5   # generated qualified tracking artifact
│   ├── cavity_axisymmetric_m0_zero_tail_control.h5  # explicit zero-tail A/B control
│   ├── uncorrected_control/                # same reduction without the fill correction
│   ├── grid_convergence/                   # Study I ladder (empty until built; see tools/)
│   └── XYplanarSensorData.mat / YZplanarSensorData.mat  # retired legacy E-only inputs
├── tests/                       # pytest regression and physics-quality suite
└── rf_gun/
    ├── config.py                   # connects to RF-Track, thread-environment setup
    ├── constants.py                # physical constants, unit conversions
    ├── helpers.py                  # small numeric/formatting utilities
    ├── fieldmaps/                  # RF-Track-independent H5 reduction/quality/artifact code
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
folders (`outputs/`, `KOA results/`, `Koa outputs/`, `analysis_Koa/`, `manual_references/`,
`archive/`, `logs/`,
`Upgrade_history/`) are intentionally outside this layout — see `.gitignore`.

## Core workflow

1. Load and fully digest-verify the qualified compact field artifact; its frequency, mesh,
   aperture profile, modal-tail provenance, and m=0 E/B phasors are immutable run inputs.
2. Run the RF-only phase scan (on-axis, cold source; every other physics switch off) to calibrate
   effective voltage and R/Q; abort if the calibration is not valid.
3. Optionally run the Emission Fields Iteration self-consistency study near the cathode.
4. Build the thermionic bunch (on-axis, or jointly in (x,y,t) — see "Self-consistent cathode
   emission" above) and track it through RF-Track (space charge and cathode mirror charges, beam
   loading, and the deflection magnet are each independently switchable; the dynamic transverse
   aperture is always enforced).
5. Tag particles forward/backward/lost and compute one final per-screen and whole-beam
   classification, reused by every downstream count/plot/output.
6. Save figures, per-screen and exit-beam distributions, and validate the run before writing a
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
  can be reproduced later without re-running the simulation. Each figure is written as `.png`
  (always) and `.eps` (only if the render comes in at or under 15 MB —
  `rf_gun.plotting.figure_io.EPS_MAX_BYTES`). The EPS backend is pure vector, so a
  many-particle scatter or a fine field-map pcolormesh produces a file tens of MB large and slow
  to open, for no gain over the PNG; those are skipped with a note in the log, and the PNG is
  written regardless. Every figure in the project goes through one writer
  (`rf_gun.plotting.figure_io.save_figure_formats`), so this rule holds for the notebook, both
  runner scripts and the screen-frame batch alike.
- `run_config.json` — every input parameter (cavity/field-map, solver/finesse, cathode/emission,
  beam-loading, aperture, deflection, screen/particle-count settings) plus the values derived from
  them before tracking starts (grid sizes, phase-scan crest, `Veff`, `R/Q`, effective length, and
  `field_provenance`: artifact file/payload/source/processing digests, quality report, support,
  grid ownership and selected artifact/zero magnetic-field mode) — no per-particle or
  per-time-sample data anywhere in it.
- `run_results.json` — everything the run *found*: `R/Q`/`Veff` actually used, peak current and
  current density, the beam-property curves vs `z` (transmission, Twiss, beam size — one row per
  screen), particle classification, aperture and back-bombardment summaries, the exit-beam
  summary, and the paths to every other output file.
- `validation.json` — a machine-readable pass/fail report: phase-calibration validity, particle-ID
  uniqueness in `Bout`, finiteness of every derived RF parameter, the field-interpolation
  outside-hull fraction against a threshold, requested B0/Bout, screen, figure, lost-particle and
  event products, and (when run) Emission Fields Iteration convergence/data. `status` is `"ok"`
  only if every applicable check passed.
- `.run_complete` — structured completion record written **only** when `validation.json`'s status
  is `"ok"`; schema v2 binds the run family/tags, canonical configuration, field file and payload
  identities, E/B power scaling and tail policy, plus SHA-256 digests of every file present in the
  completed run bundle (including `run_config.json`, `run_results.json`, `validation.json`, the v2
  event HDF5, figures, distributions and diagnostics). Scheduler-owned `resource_usage.txt` is the
  one scripted-run exception because `/usr/bin/time` finishes it only after the driver exits; it is
  deliberately excluded from the marker. The scripted entry point also binds
  its complete parser-resolved request. The notebook has no CLI parser, so it records its explicit
  notebook identity/tags and canonical hardcoded/derived parameter digest instead; it uses the same
  required-output and field-identity hash contract. Its absence means at least one
  physics/validation gate failed, regardless of whether the process otherwise exited without a
  Python exception. SLURM scripts validate the record and every bound output with
  `verify_koa_run.py`; a plain timestamp, malformed/older schema, changed artifact, changed
  code/configuration, or altered/missing output is never accepted as complete.
- `back_bombardment_events.h5` — the canonical, marker-bound v2 cathode-backstop/ray-cast event
  source consumed by `--source-mode load_run` and the macropulse study.
- `back_bombardment_energy_map.npz` — the optional legacy far-downstream reconstruction retained
  for comparison only; it is not a production heating source and is not part of the completion
  contract.
- `lost_particle_diagnostics.json` — particles RF-Track reports as lost during tracking.
- `emission_iteration.npz` — when `--emission-field-iteration` is on: the full (x,y,t)-resolved
  field/current history across outer iterations (too large for `run_results.json`, which instead
  gets only a scalar convergence summary).

`run_config.json`/`run_results.json`/`validation.json` are written atomically (a temporary file
plus `os.replace`), so a killed/crashed write can never leave a half-written file in place, and are
identical in shape from either entry point. `run_dir` is recorded as given (not resolved to an
absolute, machine-specific path), so metadata stays portable across machines/accounts.

The production script/SLURM path is the only automatic-resume path. The interactive notebook never
uses an existing marker as a cache hit: it requires a newly created, previously nonexistent run
directory, invalidates the marker before tracking, and writes a fresh schema-v2 record only after
all current validation gates and required outputs exist. Thus both entry points fail closed, while
only the scripted path compares a saved parser/scheduler/code identity before deciding to skip work.

## Field-artifact resolution vs. solver finesse

These remain independent, but the field mesh is now fixed before tracking:

- **Field-artifact resolution** (`prepare_xfdtd_fieldmap.py --dr-um/--dz-um`): the physical
  spacing of the compact `(r,z)` E/B artifact. Production tracking uses that grid verbatim.
  Runner flags `--dr_um`/`--dz_um` apply only to the legacy planar-MAT compatibility path and are
  ignored when `--field-artifact` is supplied.
- **Solver finesse** (`--finesse {extra_fine,fine,medium,coarse}`): every *numerical-integration*
  resolution setting at once — field-map integration step counts, ODE tolerance, and the
  space-charge/beam-loading/phase-scan solver step sizes. A trade between run time and numerical
  precision, independent of artifact resolution and of every physical setting (particle/screen
  counts, cathode temperature, cavity R/Q, deflection current, which physics is switched on).
  `medium` matches the notebook's defaults; the notebook has the equivalent `NOTEBOOK_FINESSE_TIER`
  variable near the top of its configuration cell.

## KOA SLURM production suite

`KOA_slurm_scripts/` contains six array-job scripts, each calling the same
`run_thermionic_tm010.py` "everything ON except deflection" production profile (RF field with the
validated phase calibration, physical aperture + cathode backstop with v2 event capture, fresh-PIC
space charge, cathode mirror charge, `BeamLoadingSW` attached, the converged Emission Fields
Iteration source, screens, full HDF5/JSON output, and figures) and overriding only the study
variable(s), at `N_PARTICLES=300000` and a common `--seed 42`. Studies II-IV pass the checked
`cavity_axisymmetric_m0_rftrack.h5` with `--rf-magnetic-field artifact`; override its location
with `RFTRACK_FIELD_ARTIFACT`. Studies I/I-bis instead require one qualified artifact per actual
mesh under `RFTRACK_FIELD_ARTIFACT_DIR`:

| Script | Array | Varies | Finesse | Field artifact |
|---|---|---|---|---|
| `study_i_medium_field_grid.slurm` | 0-9 | artifact field-grid ladder | medium | distinct resolution-matched artifact per task |
| `study_i_bis_fine_field_grid.slurm` | 0-9 | artifact field-grid ladder | fine | distinct resolution-matched artifact per task |
| `study_ii_finesse.slurm` | 0-3 | `coarse,medium,fine,extra_fine` | varies (the study variable) | fixed qualified artifact |
| `study_iii_temperature.slurm` | 0-20 | `T_K = 1600 + 10*task_id` (1600-1800K) | fine | fixed qualified artifact |
| `study_iii_bis_temperature_1500_1700K.slurm` | 0-20 | `T_K = 1500 + 10*task_id` (1500-1700K) | fine | fixed qualified artifact |
| `study_iv_deflection_macropulse.slurm` | 0-9 | deflection current (0.0-1.0A, 10 equal steps) | fine | fixed qualified artifact |

The old per-run `--dr_um/--dz_um` tracking controls are retired: an immutable artifact owns its
mesh. Study I/I-bis therefore map every task to
`field_maps/grid_convergence/cavity_axisymmetric_m0_dr<DR>um_dz<DZ>um_rftrack.h5` (base directory
overridable with `RFTRACK_FIELD_ARTIFACT_DIR`) and perform full qualification/payload verification
plus an exact check of the stored `r/z` spacing before creating an output directory. A missing or
wrongly spaced task artifact fails rather than running duplicate physics under a different label.

**Paths.** The scripts assume this KOA install, and nothing else in the repo does:

```
/home/nbidault/rf-track/                              repo root
/home/nbidault/rf-track/run_thermionic_tm010.py       transport driver
/home/nbidault/rf-track/run_back_bombardment_macropulse.py
/home/nbidault/rf-track/verify_koa_run.py              completion/identity verifier
/home/nbidault/rf-track/rf_gun/                       the Python package
/home/nbidault/rf-track/KOA_slurm_scripts/            the scripts themselves
/home/nbidault/rf-track/field_maps/cavity_axisymmetric_m0_rftrack.h5
/home/nbidault/venvs/rftrack/bin/activate             the virtualenv
```

Override with `RFTRACK_REPO_DIR` / `RFTRACK_VENV_ACTIVATE` to run from a clone anywhere else, and
with `RFTRACK_FIELD_ARTIFACT` when the qualified artifact is stored elsewhere. The
root is located by searching `$SLURM_SUBMIT_DIR`, its parent, then that default path, accepting
only a directory holding **both** `run_thermionic_tm010.py` and `rf_gun/__init__.py`; `$REPO_DIR`
is then prepended to `PYTHONPATH` so `import rf_gun` never depends on the working directory. It
cannot be derived from the script's own path — SLURM executes a spooled copy on the compute node,
so `$0` points into `/var/spool/slurmd/`, and `SLURM_SUBMIT_DIR` is the only submit-side location
SLURM preserves.

**Reading the `.err` file.** RF-Track writes several alarming-but-benign lines to stderr from its
C++ side — most notably `error: autophase failed to find an on-crest accelerating phase.` Under
Jupyter these are invisible (IPython does not capture an extension module's C-level fd 2), so they
appear for the first time when SLURM captures fd 2 into a `.err` file, and read as a crashed run.
They are not: **every one of the 53 completed runs in the first KOA batch emitted that autophase
line as the first line of its `.err` file and finished normally.** It is RF-Track's own crest
finder, whose result this project never uses — the field map's phase and reference time are set
explicitly (`FM.set_phid()` + `FM.set_t0(0.0)`) from the run's own RF-only phase-scan calibration.
`run_thermionic_tm010.py` now prints a labelled note to stderr at startup so it lands directly
above the lines it explains. **A run's success signal is `.run_complete` in its output directory,
not the contents of `.err`.**

**Work function — every script uses the same treatment.** Each passes `--phi_eff_ev`, resolved at
submit time from `rf_gun.work_function_models.LIU_2017_PHI_EFF_EV` (2.66 eV — Liu et al. 2017,
single-crystal LaB6(100), this repo's recommended baseline) rather than duplicating a literal,
together with `--work-function-temperature-model linear_tcwf` (Model 1: that anchor carried across
temperature by Bulyga & Solonovich's TCWF slope, α = 1.8e-4 eV/K about T_ref = 1773 K). The T-model
overrides the scalar at run time and takes its own `phi_ref_eV` from the very same constant, so the
two agree by construction and `--phi_eff_ev` survives in `run_config.json` as the recorded
zero-slope anchor. Over the operating window: 2.611 eV at 1500 K, 2.629 at 1600, 2.638 at 1650,
2.647 at 1700 — i.e. ×1.46 down to ×1.09 the emitted current of a flat 2.66 eV.

This matters because `run_thermionic_tm010.py`'s own `--phi_eff_ev` default is **2.1 eV**, which is
not anchored to anything here, and at 1650 K it gives ~51× the emitted current density of 2.66 eV
(~420 A/cm² off the 1.4 mm flat face, roughly 40× what LaB6 actually delivers). Every KOA batch
predating this change silently inherited that default, so results in `outputs/runs/study_i*`,
`study_ii_*`, `study_iii_temperature/` and `study_iv_*` are 2.1 eV constant-φ runs and are **not**
comparable case-for-case against anything produced afterwards. Change the work-function treatment
in all six scripts or none — mixing them silently breaks cross-study comparability.

Study III-bis is the temperature scan re-run over the physically realistic LaB6 window at the
corrected work function (~1.6 A/cm² at 1500 K to ~18.7 A/cm² at 1700 K). It is not a subset of
Study III's output despite the overlapping 1600–1700 K range — only the temperature window and the
work function differ, but the latter changes the emitted charge by ~51×.

Study IV's stage 1 is resumable so a failed stage 2 can be retried without repeating hours of
tracking. Reuse requires an exact match of all current parser-resolved transport arguments,
run-family/stable scan tags, git/code and RF-Track identity, field file/payload, magnetic/tail
modes, common power normalization, validation, and hashed required outputs. Stage 2 has its own
atomic structured marker binding its fully resolved thermal request and output hashes to the exact
transport marker/config/event-HDF5 hashes. A changed event file, thermal bin, material setting,
transport, or code therefore invalidates reuse. The existing
`study_iv_deflection_macropulse/` tree is 2.1 eV and predates the artifact identity, so it will be
rejected — move it aside or point the study at a fresh subtree before resubmitting.

Study IV additionally enables the deflection UserField and forces single-threaded tracking for
every current including 0A (the element stays attached at zero amplitude, isolating current
amplitude from code-path/threading changes), then runs the macropulse postprocessor for 10 µs in a
separate subdirectory so it cannot overwrite the transport stage's own event file.

Every script:

- resolves the repository from `SLURM_SUBMIT_DIR` (or fails clearly) rather than a hardcoded path;
- checks that its selected qualified field artifact exists before activating the expensive run,
  passes it explicitly, and lets the runner verify qualification and the complete payload digest;
- activates a documented, overridable venv path (`RFTRACK_VENV_ACTIVATE`) and fails loudly if
  RF-Track is not importable, rather than silently proceeding;
- validates `SLURM_ARRAY_TASK_ID` before indexing into any per-case list;
- prints scheduler IDs, resolved case parameters, Python/RF-Track/numpy versions, git commit and
  clean/dirty state, a content digest of the active simulation code/launcher, hostname, and the
  thread environment before doing any heavy work; those identities are persisted in run tags;
- passes the selected forward power explicitly (`RFTRACK_RF_FORWARD_POWER_W`, default 1 MW) so
  both field scaling and beam-loading calibration use one visible value;
- writes a unique, deterministic case directory and skips it only after `verify_koa_run.py`
  establishes an exact structured-marker identity match. Legacy/plain/malformed/stale markers and
  pre-existing incomplete directories fail closed and are never overwritten automatically;
- captures wall time and peak resident memory via `/usr/bin/time -v` into
  `<run_dir>/resource_usage.txt`;
- fails the SLURM job (non-zero exit) if the expected completion marker is missing, so a
  physics/validation gate failure is visible in `sacct`, not just in a log file.

**The log directory must exist before `sbatch`** (SLURM opens `--output`/`--error` before the job
body runs): run `mkdir -p logs outputs/runs` once before submitting any of these scripts.

**The resource requests (`--cpus-per-task`, `--mem`, `--time`) and array concurrency limits
(`%N`) in every script are placeholders.** They have not been tuned against real KOA pilot runs
(this repository has no cluster access to do so) — measure your own from representative tracking
pilots before submitting a full array, and update the scripts with real values. Field-artifact
mesh convergence is a separate preprocessing-plus-tracking comparison as described above. Study
IV's `--thermal-bin-ns` (10ns initial production target) is
similarly unvalidated for this study until the mandatory 20/10/5ns pilot convergence comparison
described in that script's header comment has been run.

Quick single-configuration validation run (not a SLURM script):

```bash
python run_thermionic_tm010.py \
  --field-artifact field_maps/cavity_axisymmetric_m0_rftrack.h5 \
  --rf-magnetic-field artifact \
  --field-tail-policy modal \
  --rf-forward-power-w 1.0e6 \
  --preset quick
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

- **Axisymmetric RF approximation.** The new volume contains E and H, but the magnetic field is
  noticeably less axisymmetric than the electric field. Production deliberately tracks the m=0
  `Er/Ez/Btheta/Bz` projection; discarded modes remain quantified in the preprocessing report and
  notebook, rather than being silently treated as zero.
- **Modeled downstream field tail.** The final 7.6402 mm to the 40.589 mm structure end is not in
  the supplied volume. The production artifact uses a qualified below-cutoff modal extension from
  the already small field at 32.9488 mm; its approximately 0.0113% contribution to the
  measured on-axis total is small but modeled, and the zero-tail artifact remains the required
  regression control.
- **Absolute field amplitude rests on a provisional fill model.** The saved four-cycle window
  establishes local growth, but not the actual drive history, Q or final amplitude. The current
  explicit step model corrects the original epoch and drift-definition errors; it remains
  conditional. Nonlinear emission, capture and heating mean that relative beam trends can also
  change with the assumed gradient. A converged, power-calibrated source would retire this.
- **Tangential-E surface artefact beyond the cathode edge.** For r greater than about 2 mm — in
  the cathode-to-holder annular gap, outside the 1.6 mm LaB6 edge — `z=0` is not a conductor
  plane, so the one-sided surface limit is the wrong model there and leaves a spurious tangential
  field. The emission and back-bombardment surfaces (r <= 1.6 mm) are unaffected.
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
- **Stored `KOA results/` predate this upgrade** and were produced on RF-Track 2.7.1. They are
  superseded rather than a baseline: see `docs/upgrade_2026_09_axisymmetric_eb.md` section 5.
  Baselines must be local-vs-local.

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

The E+B upgrade added coverage for the RF magnetic-field sign convention at the RF-Track boundary
(`test_rf_magnetic_sign_convention.py`), the fill-transient extrapolation
(`test_fill_transient_correction.py`), the Yee on-plane surface stencil, fail-closed quality-status
resolution, relative-path handling in the completion verifier, rim/non-finite deposition
accounting, and a guard against missing glyphs in the field-map figure. The fill audit adds
reference-time, signed-growth, model-policy and repeated-scaling checks. Current local status:
**506 passed, none skipped**. Both notebooks are clean; their previous executed copies were archived.
The cleanup also checks concurrent JSON writes, shared hashing and matching tail controls;
legacy-schema tests now create portable fixtures instead of requiring an old local run.
See the [cleanup record](docs/repository_cleanup.md) for validation scope and preserved backups.

Keep source, tests, packaging metadata and documentation together when committing changes.
Generated outputs, environments and local audit copies are excluded through `.gitignore`.

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
