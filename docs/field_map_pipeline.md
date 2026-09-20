# XFdtd field-map qualification and RF-Track artifact

The 24.8 GB XFdtd file is a source dataset, not a simulation input.  It is
inspected and reduced once, out of core, into a small, immutable, qualified
axisymmetric artifact.  RF-Track notebooks and batch jobs load that artifact;
they do not reopen or refit the volume map.

The dedicated, executable walkthrough is
`XFdtd_field_map_preprocessing.ipynb`.  The command-line entry point for a
repeatable production build is `prepare_xfdtd_fieldmap.py` (use `--help` for
the exact options implemented in the current checkout).

## Files and fixed physical conventions

- Source: `field_maps/cavity_E_H_full_volume.h5`
- Production artifact: `field_maps/cavity_axisymmetric_m0_rftrack.h5`
- Source SHA-256:
  `095ead03c4d20133e55d63c39bc61608304c54ea965d09d8571edd3046072ab7`
- Source drive frequency: 2.865417220 GHz
- Source power: 1.8 MW
- Tracking-domain end: 40.589 mm downstream of the cathode
- Tail-fit start: 30 mm
- Nominal reduction grid: 100 um radially and 50 um longitudinally, with 32
  uniformly spaced azimuths.  Mesh convergence is the ten-step sqrt(2) ladder built by
  `tools/build_grid_convergence_artifacts.py`, which brackets this production grid;
  `field_maps/grid_convergence/` is empty until that ladder is built.

Coordinates and field components are transformed into the cathode-centred
beam frame before cylindrical projection.  For this export,
`(x_beam, y_beam, z_beam) = (X_native, Z_native, -Y_native)`.  The cathode
origin is read from the coordinate metadata and checked consistently across
all six staggered components; it is not inferred from a field maximum.

XFdtd stores magnetic-field intensity **H** in A/m, not magnetic flux density
**B**.  The reducer co-locates each component using its own Yee coordinates
and timestamps, transforms the vector, and applies `B = mu0 H`.  This
conversion is used only inside the analytic vacuum/aperture mask.  Values in
metal and outside the selected vacuum channel are zeroed and excluded from
symmetry, norm, and Maxwell diagnostics.  Electric and magnetic components
retain one common physical normalization; components are never normalized
independently.

## Why the volume end is acceptable only with an explicit tail policy

The common six-component source support ends at 32.9488 mm, 7.6402 mm before
the tracking-domain end.  This geometric gap is real.  A bounded, all-17-frame
check of the exit-pipe region nevertheless shows that the supplied fields are
already very small and follow the expected below-cutoff decay:

Values below describe the **archived v1** artifact after its 2.127810x estimate,
at the last aligned measured plane (32.9411 mm). For the current amplitude, use
the shipped JSON report; normalized fractions and decay lengths are unchanged:

| quantity at the measured end | value | fraction of map peak |
| --- | ---: | ---: |
| maximum electric-field magnitude | 0.1854 MV/m | 0.1431% |
| maximum magnetic-flux-density magnitude | 0.02244 mT | 0.0207% |
| time-averaged field energy per unit length | 7.90e-7 J/m | 4.02e-8 |

The on-axis tail fitted from 30 mm has a 1.052860 mm amplitude decay length,
compared with 1.053112 mm for the analytic circular-pipe TM01 decay.  Its
log-amplitude fit has R-squared 0.9999694 and less than 0.002 degrees of phase
change. In the current v2 artifact, extending this fitted mode to 40.589 mm
contributes 0.192774 kV of missing on-axis voltage, 0.0113% of the measured
1703.514 kV. The matching uncorrected control in `field_maps/uncorrected_control/`
reports approximately 0.0924 kV for the same tail; these differ by the current
2.087319x multiplier, within inverse-scaling roundoff.

The production artifact therefore uses a **marked modal tail**, never an
unlabelled extrapolation.  Its metadata records the last measured coordinate,
fit interval, fitted and analytic decay lengths, endpoint fractions, missing
voltage, and extrapolated region.  An artifact with an explicit zero-field
tail is produced or evaluated as the A/B control.  Beam observables should be
insensitive to modal versus zero tail within the declared convergence
tolerance; if they are not, the source volume is not long enough for the
requested tracking study.

## Filling is measured; the final amplitude is model dependent

The source stopped at 100 ns and its 17 frames span only four RF periods.
Independent raw-trace fits confirm roughly 0.25% amplitude growth per cycle.
The former automatic 2.127810x multiplier assumed one on-resonance mode and an
ideal step from zero at t=0. It also evaluated a slope measured at the 99.301 ns
phasor reference at the 100 ns run end. Correcting the epoch alone gives 2.098003x.

The current reducer separates signed amplitude growth, phase rotation and drift
that cannot be described by one spatial envelope. `--fill-correction auto`
preserves samples and marks a drifting export analysis-only. The explicit
`--fill-correction single-pole --fill-start-time-ns 0` option retains a provisional
step-drive estimate using the correct reference and signed E/H growth. The
full-source rebuild gives **2.087318827x**, a **1.903% amplitude reduction** from
v1. The exact factor is stored as `quality.fill_transient.applied_correction_factor`.
`--fill-correction off` is an explicit uncorrected tracking control and never
requires an admissible exponential inversion. The notebook uses the same function.

Numerical qualification does not verify the source-power convention or the
steady-state amplitude. Fill policy, reference/start times, assumptions and the
applied multiplier are recorded in field metadata and the processing digest.
The old wall-loss plausibility argument did not include reflected power and did
not establish a calibration. The 1.8 MW supplier port setting remains unchanged
pending clarification of its peak-versus-cycle-average convention.

See [field_map_fill_audit.md](field_map_fill_audit.md) for raw evidence,
identifiability limits, and reproduction commands. Seven amplitude time constants
leave 0.091% of an ideal step-fill deficit; five leave 0.67%.

## Cylindrical symmetry is a declared model choice

The full-volume samples are Fourier analysed around the beam axis before any
projection.  Electric and magnetic non-axisymmetric fractions and signed
azimuthal-mode spectra are retained in the quality record.  The magnetic field
is visibly less axisymmetric than the electric field; the preprocessing
notebook deliberately shows this result.

For the present RF-Track studies, that diagnostic does not veto the requested
model.  The production representation is always the cylindrical vector
`m = 0` projection containing complex `Er`, `Ez`, `Btheta`, and `Bz` phasors.
Discarded `m != 0` content is documented, not silently interpreted as zero in
the original solver result.  A future 3-D tracking study must start from the
volume source again rather than treating the axisymmetric artifact as a 3-D
map.

## Out-of-core processing sequence

1. Inspect the HDF5 schema, component shapes/chunks/units, coordinate metadata,
   and separate E/H timestamp arrays without reading a 4-D field dataset.
2. Validate the schema and, by default, stream the whole 24.8 GB file once to
   verify the supplier SHA-256.  `--no-verify-source-sha256` is only for a
   deliberate repeat analysis after that exact source digest has already been
   confirmed and recorded.
3. Plan a uniform `(z, r)` grid on the common six-component support and build
   the analytic cavity/pipe vacuum mask.  The same grid spacing continues to
   the 40.589 mm tracking endpoint.
4. Read bounded longitudinal slabs.  Fit all saved frames at the fixed drive
   frequency, including offset and linear-envelope terms used to report fit
   residual and temporal drift.  E and H keep their distinct saved times.
5. Co-locate the staggered fields on 32 cylindrical azimuths, apply the beam
   transform and vacuum mask, quantify Fourier modes, and retain the vector
   `m = 0` projection.
6. Check field norms, axis regularity, vacuum Maxwell residuals, temporal fit,
   cylindrical symmetry, endpoint decay, and omitted voltage.  Apply either
   the qualified modal tail or the explicit zero-tail control.
7. Atomically write the compact artifact and immediately reload it with full
   payload-digest verification.

The notebook sets `RUN_REDUCTION = False` by default so opening or running it
cannot accidentally launch a lengthy read of the volume.  With that setting,
it loads an existing artifact and plots its saved diagnostics.  Set the switch
only for an intentional interactive build; routine and cluster builds should
use `prepare_xfdtd_fieldmap.py`.

## Immutable artifact contract

An artifact contains:

- uniform SI grids `r_m`, `z_m` and complex SI phasors `Er_Vpm`, `Ez_Vpm`,
  `Btheta_T`, `Bz_T`;
- the native-to-beam transform, cathode origin, physical vacuum mask, and
  aperture-radius profile;
- drive frequency, source power, phasor convention, measured and extrapolated
  support, reduction choices, tail policy, and all quality reports;
- the source SHA-256 and a canonical processing-configuration digest; and
- individual numeric-dataset digests plus one canonical payload digest.

Production loading requires `quality.status == "qualified"` and verifies the
payload digest. Ordinary production also requires the `modal` artifact variant;
the separately written `zero_tail_control` variant requires an explicit
`--field-tail-policy zero-control` request. The artifact digest is the field-map identity that belongs in
run provenance and completion markers.  Changing the mesh, transform,
harmonic fit, mask, symmetry projection, tail policy, or source file creates a
different artifact; do not overwrite results while retaining an old digest.

The main simulation notebook may plot the four common-phase meridional maps,
including the azimuthal `Btheta` map, from this compact artifact.  It must not
recompute the raw-map treatment.  Likewise, SLURM jobs receive the artifact
path (and, for a deliberate magnetic-field A/B study, an explicit request to
zero its B components) rather than a MAT or raw HDF5 path.

The stored fields retain the supplier's nominal source setting (1.8 MW); the
absolute power convention is not independently verified.
Tracking chooses an explicit forward power and applies one common
`sqrt(P_target/P_source)` amplitude factor to all four E/B phasors. This target
power is also the forward-power input used in delivered-power and R/Q/beam-
loading calibration; E and B are never independently renormalized.
