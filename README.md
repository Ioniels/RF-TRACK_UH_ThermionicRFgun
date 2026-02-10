# UH Gun RF-Track Beam Dynamics Simulation

Thermionic cathode electron beam tracking in a TM010 λ/2 RF cavity at 2.856 GHz using RF-Track.

## Overview

This project simulates electron beam dynamics from a heated thermionic cathode through an RF photoinjector cavity. The cavity operates in TM010 mode at f = 2.856 GHz with a half-wavelength (λ/2) geometry.

### Physics

**TM010 Cavity:**
- Resonant frequency: 2.856 GHz
- Mode: TM010 (transverse magnetic, axially symmetric)
- Length: λ/2 ≈ 52.5 mm
- Accelerating gradient determined by COMSOL field maps

**Thermionic Emission:**
- Hot cathode model: DC electron emission
- Electrons emitted when local Ez > 0
- Phase-averaged sampling simulates continuous emission

## Project Structure

```
.
├── UH_gun_tracking.ipynb       # Main analysis notebook
├── config.py                   # RF-Track setup
├── utils.py                    # Helper functions
├── load_fieldmap_mat.py        # COMSOL field map loader
├── field_maps/                 # COMSOL simulation data
│   ├── XYplanarSensorData.mat
│   └── YZplanarSensorData.mat
└── archive/                    # Previous notebook versions
```

## Workflow

### 1. Field Map Processing
- Load COMSOL field maps (XY and YZ planes)
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

## Helper Functions

**utils.py:**
- `kinetic_energy()` - Compute Ek from momenta
- `select_iq_snapshots()` - Find optimal I/Q time indices
- `build_iq_phasor()` - Construct complex field phasor
- `sample_disk()` - Uniform disk distribution
- `theoretical_energy_gain()` - Analytical ΔW for TM010
- `cavity_wavelength()` - Wavelength parameters

**load_fieldmap_mat.py:**
- `load_fieldmap_mat()` - Load COMSOL .mat files
- `plot_fieldmap_on_mesh()` - Visualize raw field maps

## References

- **RF-Track:** https://abpcomputing.web.cern.ch/codes/codes_pages/RF-Track/
- **Manual:** `RF_Track_reference_manual.pdf`
- **TM010 cavity physics:** Cylindrical cavity resonators, Jackson Ch. 8

## Notes

**Coordinate Transformation:**
- COMSOL uses (x, y, z) with y = vertical
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
