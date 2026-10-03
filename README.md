# UH thermionic RF gun with RF-Track

Simulation of the University of Hawai'i thermionic electron gun: a LaB6 cathode in an S-band
TM010 cavity, tracked with CERN's RF-Track. The `rf_gun` package loads a qualified field map
derived from a Remcom XFdtd volume, emits electrons thermionically from the cathode, tracks them
with space charge, cathode mirror charge, beam loading, the real transverse aperture and an
optional deflection magnet, and records which electrons return to the cathode and how much they
heat it over an RF macropulse. The gun exit distribution feeds the alpha-magnet and downstream
beamline repositories; the back-bombardment heat source feeds the COMSOL 3D thermal model.

## Physics scope

- **Emission.** Richardson-Dushman with Schottky lowering (default), Jensen's 2019 general
  thermal-field model, the additive thermionic plus Murphy-Good model, and a direct Murphy-Good
  energy-integral reference. Surface roughness and optional LaB6(100) work-function temperature
  models (constant, linear TCWF slope, piecewise surface evolution). The bunch is sampled either
  uniformly over the 1.40 mm flat face or jointly in (x, y, t) from the local RF field, with a
  uniform cathode temperature or the measured COMSOL profile T(x, y).
- **RF field.** Axisymmetric m = 0 electric and magnetic phasors (Er, Ez, Btheta, Bz) reduced
  from the XFdtd E/H volume recorded at 1.8 MW. Tracking scales E and B together by
  sqrt(P / 1.8 MW); the default forward power is 1.0 MW. An RF-only phase scan with a cold on-axis
  source calibrates the effective voltage and R/Q before every production run.
- **Transport.** RF-Track `Volume` with an explicit `SpaceCharge_PIC_FreeSpace` engine and
  cathode mirror plane, `BeamLoadingSW`, the cavity channel R(z) as a live `Aperture_1d`, a thin
  cathode backstop for event capture, and a deflection magnet as a `UserField`. The domain ends
  at the exit tube face, 40.589 mm from the cathode. The CLI writes the gun output
  `Bout_sout*.h5` at that plane (fixed s, from the exit screen: forward, alive, not trailing) and
  keeps the final fixed-time snapshot, about 1.2 m downstream, as `Bsnapshot_t*.h5`.
- **Emission fields iteration.** An optional under-relaxed Picard loop that converges the emitted
  current density J(x, y, t) against the space-charge and mirror field it creates near the
  cathode, and can feed the converged source into the production run.
- **Back-bombardment and heating.** Returning electrons are located by ray-casting RF-Track's
  loss table against the parameterized cathode (flat face, bevel, holder); each event records
  position, time, momentum, energy, zone and incidence angle, with exact charge and energy
  closure. Deposition uses the Tabata-Ito-Okabe/CSDA range model; a layered Cartesian thermal
  solver propagates the heat over a top-hat macropulse (8 us default) with one-way coupling.
  Energy-balance checks are mandatory outputs of every run.
- **Limits.** `BeamLoadingSW` shows no measurable effect for this geometry in RF-Track 2.7;
  the absolute field amplitude rests on a provisional cavity-filling estimate; deposition has no
  backscatter; emissivity 0.8 and an adiabatic mount are assumptions. Relative trends are more
  robust than absolute temperatures.

## Conventions

- Cathode surface at z = 0, beam towards +z; x, y Cartesian for particles and cathode physics.
  The XFdtd-to-beam transform (x, y, z) = (X, Z, -Y) and the cathode origin are stored in the
  field artifact. Positive Er points outward; Btheta follows the right-handed beam frame.
- SI units inside the library. CLI flags with a unit suffix (`_um`, `_mm`, `_K`, `_A`, `_us`,
  `_ns`) convert once at the parser. Saved distributions use openPMD-beamphysics units.
- RF phasor `Re(F exp(+i 2 pi f (t - t_ref)))`, with E and B fitted at one frequency and
  reference time. The field frequency comes from the artifact, not `--f_hz`. The launch phase
  is `phi_zero + --emission_phase_start (45 deg) + --phase_deg (0)`, with `phi_zero` from the
  RF-only phase scan; `FM.set_phid()` and `FM.set_t0(0)` are set from it, not from autophase.
- Electron charge q = -1; a macroparticle's `N` column is the number of electrons it represents
  and `N * q_e` its charge in coulombs, in memory and in every HDF5 file.
- Back-bombardment surface normal `n_in` points from vacuum into the solid; an event is accepted
  only when `p_hit . n_in > 0`.

## Installation

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[notebook]'
```

RF-Track 2.7.2 is pinned (`pyproject.toml`, extra `rftrack`). It is a proprietary CERN code
distributed from <https://gitlab.cern.ch/rf-track>; follow the download instructions there and
install the Python wheel `RF-Track==2.7.2` into the same environment. `python -c "import
RF_Track"` prints the banner with the version. For batch runs set `RF_TRACK_NO_UPDATE_CHECK=1`
and a thread count (`--threads`, or `rf_gun.config.set_thread_environment`, which also aligns
the BLAS thread variables).

## Field maps from XFdtd

The source `field_maps/cavity_E_H_full_volume.h5` (24.8 GB, E and H at 2.865417 GHz, 1.8 MW)
is reduced once, out of core, into a small immutable artifact that notebooks and batch jobs
load. Versioned artifacts: `field_maps/cavity_axisymmetric_m0_rftrack.h5` (4.3 MB, with its
`.report.json`) and the zero-tail sensitivity control `cavity_axisymmetric_m0_zero_tail_control.h5`
(3.8 MB, selected only with `--field-tail-policy zero-control`). These derived maps are versioned
as reference inputs; the raw RemCom XFdtd sets are not. The field-map manifest is documented
separately and available from Niels.

```bash
python prepare_xfdtd_fieldmap.py \
  --source field_maps/cavity_E_H_full_volume.h5 \
  --output field_maps/cavity_axisymmetric_m0_rftrack.h5 \
  --tail-policy modal --fill-correction single-pole --fill-start-time-ns 0
```

The reduction inspects the HDF5 layout, verifies the source SHA-256, plans a uniform (r, z) grid
(100 um by 50 um) with the analytic vacuum mask, fits every saved frame at the drive frequency
in bounded slabs, co-locates the staggered Yee components on 32 azimuths, converts H to B in
vacuum, keeps the m = 0 projection, checks Maxwell residuals, axis regularity and symmetry, and
extends the field from the measured end (32.9488 mm) to the tracking end with a qualified
below-cutoff modal tail. The artifact stores grids, phasors, transform, mask, aperture profile,
tail policy, all quality reports and digests; production loading requires
`quality.status == "qualified"` and a matching payload digest. The saved frames still grow by
about 0.25 % per cycle, so the fill correction (factor 2.087) is a model, not a power
calibration. `XFdtd_field_map_preprocessing.ipynb` shows the diagnostics; it only re-reads the
volume when `RUN_REDUCTION = True`. The planar `.mat` maps are retired (E only, not versioned)
and reachable only through `--allow-legacy-mat-fieldmap`. The grid-convergence ladder for KOA
Study I is built by `tools/build_grid_convergence_artifacts.py` into `field_maps/grid_convergence/`.

## COMSOL inputs

- `data/comsol_cathode_surface_temperature_t600s.txt`: COMSOL 6.3 export of the heater-only
  cathode surface temperature at t = 600 s (columns x y z T(K), length unit inches, 3862 nodes,
  1613 to 1620 K). `rf_gun.load_comsol_surface_temperature` interpolates linearly on the
  scattered nodes; `outside_hull="nearest"` extends the measured edge over the chamfer. In the
  notebook, `CATHODE_TEMPERATURE_MODE` selects `"comsol"` or `"uniform"` (`T_CATHODE_K`).
- `rf_gun.comsol_io` exports the deposited heat source for the COMSOL thermal model and reads a
  COMSOL thermal result back for comparison with the Python solver.

## Running

`UH_gun_tracking_demo.ipynb` runs top to bottom in a fresh kernel: field check, phase
calibration, emission source (optional iteration), tracking, return and heating diagnostics,
save and validate. With `SAVE_DATA = True` it creates a new run directory; the notebook is
versioned without outputs. The scripts run the same pipeline:

```bash
# Smoke test (1000 particles, coarse finesse)
python run_thermionic_tm010.py --field-artifact field_maps/cavity_axisymmetric_m0_rftrack.h5 \
  --rf-magnetic-field artifact --field-tail-policy modal --rf-forward-power-w 1.0e6 --preset quick

# Production run, then the 8 us heating study on its events
python run_thermionic_tm010.py --field-artifact field_maps/cavity_axisymmetric_m0_rftrack.h5 \
  --rf-magnetic-field artifact --finesse fine --n_particles 300000 --seed 42 \
  --phi_eff_ev 2.66 --work-function-temperature-model linear_tcwf --output outputs/tmp/gun_T1650K
python run_back_bombardment_macropulse.py --source-mode load_run --run-dir outputs/tmp/gun_T1650K \
  --material-set LaB6_UH_recommended_v1 --deposition-model BB0_TIO --macropulse-duration-us 8 \
  --thermal-bin-ns 50 --initial-temperature-uniform-k 1650 --output outputs/tmp/gun_T1650K --save-figures
```

RF frequency, cathode temperature, work function (2.66 eV, LaB6(100), Liu et al. 2017) and
macropulse duration default to the validated entries of `parameters/UH_reference_parameters.yaml`,
a copy of the Orchestrator master file. `--finesse {coarse,medium,fine,extra_fine}` sets
every numerical-resolution parameter at once; `medium` matches the notebook. Full option lists:
`--help` on either script.

## Outputs

Convention: `outputs/tmp/` holds scratch runs and is cleaned periodically; `outputs/ref/` holds
reference runs, distributions handed to other repositories and figures used in papers, and is
never cleaned automatically. Both are ignored by Git. Without `--output`, runs are named
`outputs/runs/<timestamp>_<T tag>_SC<on|off>_BL<on|off>/`; move them to `tmp` or `ref` once
validated. A run directory contains:

| File | Content |
|---|---|
| `screen_distributions_hdf5/B0_*.h5`, `Bout_*.h5`, `screen_*_z*.h5` | openPMD-beamphysics `ParticleGroup` files: launched bunch, final forward surviving bunch, one full phase space per screen (last screen on the exit face, 40.589 mm) |
| `run_config.json` | every input parameter plus pre-tracking derived values (crest phase, Veff, R/Q, grids, field provenance and digests) |
| `run_results.json` | what the run found: peak current and current density, beam properties vs z, particle classification, aperture and back-bombardment summaries, paths of all other outputs |
| `validation.json` | pass/fail report (phase calibration, particle-ID uniqueness, finite RF parameters, field hull fraction, required products, iteration convergence); `status == "ok"` required for `.run_complete` |
| `.run_complete` | completion record with SHA-256 digests of every output; its absence means a gate failed |
| `beam_properties.csv` | transmission, Twiss parameters and beam sizes per screen |
| `back_bombardment_events.h5` | per-event record, schema `back_bombardment_events_v2`, with accounting, geometry and provenance groups |
| `back_bombardment_macropulse/` | `back_bombardment_heat_source.h5` (BB0 deposited energy per cell and layer for one RF period), `back_bombardment_macropulse.h5` (current and power histories, thermal time series, COMSOL comparison); CLI runs also write `study_config.json` and `study_results.json` |
| `lost_particle_diagnostics.json`, `emission_iteration.npz` | RF-Track loss table; full field and current history of the emission iteration |
| `figures/` | every figure as 300 dpi PNG and PDF (EPS when under 15 MB) with its data alongside (`.npz`/`.json`) |

JSON files are written atomically and have the same shape from the notebook and the script.

## KOA SLURM templates

`KOA_slurm_scripts/` holds six array-job templates that call `run_thermionic_tm010.py` with the
full production physics (phase calibration, aperture and backstop, space charge and mirror,
beam loading, converged iteration source, screens, HDF5 and figures) at 300000 particles,
seed 42, 2.66 eV with `linear_tcwf`, and vary one quantity each:

| Script | Array | Varies |
|---|---|---|
| `study_i_medium_field_grid.slurm`, `study_i_bis_fine_field_grid.slurm` | 0-9 | field-grid ladder (one artifact per task) |
| `study_ii_finesse.slurm` | 0-3 | finesse tier |
| `study_iii_temperature.slurm`, `study_iii_bis_temperature_1500_1700K.slurm` | 0-20 | cathode temperature, 10 K steps |
| `study_iv_deflection_macropulse.slurm` | 0-9 | deflection current 0 to 1 A, then the macropulse study |

Submission is manual. Resource requests (`--cpus-per-task`, `--mem`, `--time`, `%N`) are
placeholders to size from a pilot. Paths default to `/home/nbidault/rf-track` and
`/home/nbidault/venvs/rftrack`; override with `RFTRACK_REPO_DIR`, `RFTRACK_VENV_ACTIVATE`,
`RFTRACK_FIELD_ARTIFACT`, `RFTRACK_RF_FORWARD_POWER_W`.

```bash
mkdir -p logs outputs/runs
sbatch KOA_slurm_scripts/study_ii_finesse.slurm
python verify_koa_run.py --help   # completion and identity check used by the scripts
```

The first line of every `.err` file, RF-Track's "autophase failed" message, is benign: the
phase is set from the project's own scan. The success signal is `.run_complete` in the run
directory, which `verify_koa_run.py` checks before a task is skipped as already done.

## References

- RF-Track: A. Latina, CERN, <https://gitlab.cern.ch/rf-track>.
- Murphy and Good, Phys. Rev. 102, 1464 (1956); Jensen et al., J. Appl. Phys. 126, 065302 (2019).
- Liu et al. (2017), LaB6(100) thermionic work function.
- Wilson, SLAC-PUB-2884 (1982) and Wangler, RF Linear Accelerators, 2nd ed., Sec. 4.7 (beam loading).
- Bakr et al., Phys. Rev. ST Accel. Beams 14, 060708 (2011), LaB6 stopping-power coefficients.
