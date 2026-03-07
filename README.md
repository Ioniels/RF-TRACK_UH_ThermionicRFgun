# UH Gun RF-Track Beam Dynamics Simulation

Thermionic cathode electron beam tracking in a TM010 λ/4 RF cavity at 2.856 GHz using RF-Track.

## Overview

This project simulates electron beam dynamics from a heated thermionic cathode through an RF cavity (S-band - λ/4 - 1 MW - 1 MeV) using RF-Track (developed at CERN: link to RF-Track github). The simulation code includes:

- Helpers for thermionic emission model self-consistent as a function of T, includes cathode surface roughness
- Loads field maps from an FDTD (3D Yee-Cell) EM solver and phasor builder for evolution vers time during transients
- Particle tracking from RF-Track includes space-charge and beam loading

To be included next:
- load cathode surface temperature from heating simulation
- e- beam back-bombardment and beam-beam interactions
- corrector magnet for e- beam back-bombardment
- Ionization and recombination with neutrals (vacuum pressure impact)

The main parameters used for the simulations are currently according to the UH Linac microwave gun.

### Physics

**TM010 Cavity:**
- Resonant frequency: 2.856 GHz
- Mode: TM010 (transverse magnetic, axially symmetric)
- Length: λ/4
- Accelerating gradient determined by field maps

**Thermionic Emission:**
- Hot cathode model: DC electron emission
- Electrons emitted when local Ez > 0
- Phase-averaged sampling simulates continuous emission

## Project Structure

```
.
├── UH_gun_tracking_demo.ipynb  # Main analysis notebook
├── run_thermionic_tm010.py     # Batch-friendly non-notebook runner
├── run_thermionic_tm010.slurm  # Slurm job script for cluster runs
├── config.py                   # RF-Track setup
├── utils.py                    # Helper functions
├── load_fieldmap_mat.py        # Field map loader
├── field_maps/                 # Field map data
│   ├── XYplanarSensorData.mat
│   └── YZplanarSensorData.mat
└── archive/                    # Previous notebook versions
```

## Workflow

### 1. Field Map Processing
- Load field maps (XY and YZ planes)
- Analyze temporal envelope via Ez_rms(t) spline fit
- Select I/Q snapshots ~90° apart for phasor construction
- Transform to axisymmetric (r,z) coordinates

### 2. RF-Track Setup
- Build complex phasor field map from I/Q snapshots
- Interpolate to regular (r,z) grid
- Configure RF_FieldMap_2d with f = 2.856 GHz
- Set integration parameters (RK2, dt, aperture)

### 3. Tracking
- **Phase scan:** Test single particles at various RF phases
- **DC emission:** Sample uniform phase distribution (48 phases × N particles)
- Volume tracking with space-charge (if enabled)

### 4. Analysis
- Energy spectrum and phase correlation
- Phase space distributions (x-px, y-py, z-pz)
- Comparison with theoretical energy gain

## Key Parameters

**Cavity:**
- `F_HZ = 2.856e9` - RF frequency
- `Y_CATHODE_MM = 13.0` - Cathode position in solver frame
- `R_MAX_M = 0.010` - Radial extent
- `NR = 4000`, `NZ = 10000` - Field map resolution

**Beam:**
- `R_CATHODE_MM = 3.14/2` - Emission radius
- `THERMAL_PT_MEVC = 0.0` - Transverse thermal momentum
- `PZ_INIT_MEVC = 1e-4` - Initial longitudinal momentum
- `N_PHASES = 48`, `N_PER_PHASE = 3` - Sampling

**Tracking:**
- `DT_MM = 0.2` - Integration step
- `APERTURE_M = 0.010` - Circular aperture
- `ODE_ALGORITHM = "rk2"` - Integrator
- `ODE_EPSABS = 1e-6` - Error tolerance

## Usage

```python
# Run the notebook
jupyter notebook UH_gun_tracking.ipynb
```

All tunable parameters are clearly defined at the top of the notebook under "Configuration" cells.

### Batch runner (recommended for laptop/cluster)

```bash
python run_thermionic_tm010.py --preset quick --output outputs/smoke_quick
```

### Re-run with full knobs (copy/paste)

```bash
python run_thermionic_tm010.py \
	--output outputs/manual_rerun \
	--threads 6 \
	--phase_deg 0.0 \
	--n_particles 100000 \
	--f_hz 2.856e9 \
	--y_cathode_mm 12.75 \
	--r_max_m 0.01 \
	--dr_um 4.0 \
	--dz_um 13.0 \
	--z_min 0.0 \
	--ext_zmax 0.0075 \
	--dt_mm 0.1 \
	--sc_dt_mm 0.2 \
	--emission_nsteps 100 \
	--emission_range 10.0 \
	--fm_nsteps 100 \
	--fm_tt_nsteps 100 \
	--cfx_dt_mm 0.1 \
	--ode_algorithm rk2 \
	--ode_epsabs 1e-6 \
	--aperture_m 0.01 \
	--sc_enabled \
	--beam_loading \
	--bl_q0 4000 \
	--bl_qext 3500 \
	--bl_p_fwd_w 1.0e6 \
	--bl_r_over_q_ohm_per_m 1.0 \
	--bl_ncells 1 \
	--bl_tinj_mode auto_from_emission \
	--bl_tinj_manual_mm_c 0.0 \
	--n_z_snap 100 \
	--screen_t0_mode unset \
	--screen_t0_manual_mm_c 0.0 \
	--r_cathode_mm 1.57 \
	--emission_scale 1.0 \
	--no-use_const_pz \
	--pz_init_mevc 4.0e-3 \
	--ra_um 1.0 \
	--re_um 10.0 \
	--emission_law RD_schottky \
	--t_cathode_k 1700.0 \
	--phi_eff_ev 2.1 \
	--beta_f 1.0 \
	--emission_phase_start 0.0 \
	--emission_phase_range 180.0 \
	--poll_interval_s 0.5 \
	--save-figures \
	--save-screen-json
```

For the complete CLI surface, run:

```bash
python run_thermionic_tm010.py --help
```

The transport progress implementation is intentionally **elapsed-only** for clarity and stability.
There is no segmented transport mode in the current code path.

## Helper Functions

**utils.py:**
- `kinetic_energy()` - Compute Ek from momenta
- `select_iq_snapshots()` - Find optimal I/Q time indices
- `build_iq_phasor()` - Construct complex field phasor
- `sample_disk()` - Uniform disk distribution
- `theoretical_energy_gain()` - Analytical ΔW for TM010
- `cavity_wavelength()` - Wavelength parameters

**load_fieldmap_mat.py:**
- `load_fieldmap_mat()` - Load .mat field-map files
- `plot_fieldmap_on_mesh()` - Visualize raw field maps

## References

- **RF-Track:** https://abpcomputing.web.cern.ch/codes/codes_pages/RF-Track/
- **Manual:** `RF_Track_reference_manual.pdf`
- **TM010 cavity physics:** Cylindrical cavity resonators, Jackson Ch. 8

## Notes

**Coordinate Transformation:**
- Source field maps use (x, y, z) with y = vertical
- RF-Track uses axisymmetric (r, z) with z = beam direction
- Mapping: r = |x|, z = y_cathode - y, Er = sign(x)·Ex, Ez = Ey

**Phasor Convention:**
- Field evolves as Re{E_hat · exp(j·2πf·t + jφ)}
- I/Q snapshots at 0° and 90° construct E_hat
- RF phase φ set via FM.set_phid()

**Thermionic Model:**
- Simulates DC beam from hot cathode
- Samples 48 phases uniformly in [0, 2π)
- Each phase gets independent tracking
- Results combined for phase-averaged statistics
