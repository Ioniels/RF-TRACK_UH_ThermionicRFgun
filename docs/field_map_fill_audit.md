# Audit of the filling correction

The corrected whole-volume estimate is **2.087318827x**, replacing 2.127810232x:
the active fields are **1.90296% lower in amplitude**. Both use an assumed ideal
step drive from zero at t=0 and a single on-resonance cavity mode. The saved data
do not independently determine those assumptions, loaded Q, or the final
amplitude. The factor remains a provisional model, not a measured calibration.

The raw source SHA-256 is
`095ead03c4d20133e55d63c39bc61608304c54ea965d09d8571edd3046072ab7`.
The original corrected artifact had payload SHA-256
`ec4435c3bfbaefa56bfc43214fab425b38c73640f7eae654f0c86157bc764b65`.

## What was checked directly

`tools/audit_xfdtd_fill.py` reads 21 bounded native E/H traces around the
cathode, axis, and off-axis cavity interior. It fits all 17 frames using each
family's own timestamps and a common reference time. The numerical evidence is
saved in [field_map_fill_audit.json](field_map_fill_audit.json).

The common reference is 99.301361935 ns; the solver stops at 100.000062421 ns.
The saved window is only about 1.3945 ns, approximately four RF periods.
Representative on-axis longitudinal fields grow by 0.246–0.249% per cycle.
This confirms a transient. A small sinusoidal-fit residual does not establish
steady state because the fit explicitly permits a changing envelope.

### Reference-time error

The old reduction reports a drift at the phasor reference time but the old
correction evaluates the fill equation at the run end. Holding its unsigned
drift value fixed, changing only the evaluation time changes the factor:

| Time used | Assumed multiplier |
| --- | ---: |
| Run end, old calculation | 2.127810232 |
| Actual phasor reference | 2.098002648 |

That is a 1.40% amplitude difference before correcting the drift definition.
Version 2 uses the signed E/H-family projections and the actual reference time.
The full-source rebuild gives signed growth of 0.002496898 per cycle for E and
0.002487374 for H. Their mean implies a multiplier of **2.087318827**, an amplitude
time constant of 152.264 ns and loaded Q of 1370.677, conditional on the model.
The exact result is in the rebuilt artifact's `quality.fill_transient`.

For the assumed envelope `A(t) = A_inf * (1 - exp(-(t-t_start)/tau))`, the
measured logarithmic amplitude growth is `g = (dA/dt)/A`. At the phasor reference,
`g * elapsed = u / (exp(u) - 1)`, where `u = elapsed/tau` and
`elapsed = t_reference - t_start`. The amplitude multiplier is then
`1 / (1 - exp(-u))`. An unsigned complex slope also contains phase rotation and
spatial changes, so it is not the `g` required by this equation.

### What four cycles cannot identify

At the on-axis probe near z=10 mm, fit a constant forced phasor plus an
independently scaled decaying homogeneous phasor, retaining a DC offset. This
removes the unverified initial-condition constraint. Different assumed decay
times give almost indistinguishable residuals:

| Assumed decay time | Normalized RMS residual | Final / sampled amplitude |
| --- | ---: | ---: |
| 40 ns | 0.00015225 | 1.2854 |
| 157 ns | 0.00015332 | 2.1201 |
| 207 ns | 0.00015345 | 2.4768 |
| 400 ns | 0.00015365 | 3.8537 |

These are examples of model non-identifiability, **not confidence limits** or
equally plausible physical histories. The source waveform and independent Q
could exclude some of them; neither is determined by this short saved window.
Half-window fits also expose sensitivity of the local slope to residual signal.

Even retaining the ideal-step model, the assumed start time matters. With the
archived drift and the correct reference, start times of 0, 5 and 10 ns imply
factors of 2.0980, 1.9121 and 1.7642. A delayed step is only a sensitivity example;
it is not a fitted model of XFDTD's actual startup ramp.

## Implementation changes

- The reducer now stores `Re(<P,P'>/<P,P>)/f` as signed amplitude growth and
  `Im(<P,P'>/<P,P>)/f` as phase rotation. The old norm `||P'||/(f||P||)` remains
  useful as a temporal quality diagnostic but cannot supply a fill inversion.
- The E and H projections are combined with equal family weight, without
  mixing V/m and A/m. Relative agreement, phase rotation and nonseparable
  spatial drift are checked before applying one real envelope multiplier.
  The 25% consistency guard is a model-admissibility check, not an uncertainty.
- The step model is evaluated at the phasor reference minus the explicitly
  assumed drive start. Negative growth is not treated as convergence.
- Applying a correction twice, or to a field with a different frequency or
  reference epoch, is rejected. Locally stationary diagnostics serialize
  without non-standard JSON infinities and do not claim verified steady state.
- The old `uncorrected_control` files were not pure amplitude controls: their
  cathode-plane Er and Bz differed from the corrected artifact's spatial fields.
  `tools/build_uncorrected_field_controls.py` now undoes only the recorded scale
  of a verified parent and requalifies/rebuilds its tails. This retains the same
  cathode treatment, with inverse-scaling roundoff documented in provenance.
- CLI and preprocessing notebook use the same fill-treatment function.
  Previously the notebook could rebuild a raw-amplitude map at the production
  path while the CLI applied the multiplier.
- Fill policy, start time, fitted result and **actually applied** factor are
  included in artifact metadata, quality, and the processing-configuration
  digest. The raw control never needs a successful exponential inversion.
- Repeated `scaled_to_power` calls now set the requested target instead of
  multiplying an already-scaled field by a second source-to-target factor.
- The RF-Track all-real guard now checks exact imaginary zeros and permits
  explicit zero-B controls. Its previous absolute tolerance rejected valid
  complex maps with small imaginary parts.

## Using the current maps

The default `--fill-correction auto` no longer invents a final amplitude. It
preserves samples and produces an `analysis_only` artifact if the saved window
drifts. This is intentionally different from the old automatic 2.1278x boost.

To make the existing approximate model explicit:

```bash
.venv/bin/python prepare_xfdtd_fieldmap.py \
  --output field_maps/provisional_step_model.h5 \
  --fill-correction single-pole --fill-start-time-ns 0
```

To build a deliberate raw-amplitude tracking control, use `--fill-correction off`.
In the notebook, the corresponding choices are `FILL_POLICY` and
`FILL_START_TIME_NS`. The grid-convergence builder accepts the same CLI options.
Use a separate grid-convergence output directory when changing amplitude models;
do not reuse artifacts merely because their grid spacings match.

Numerical `qualified` status for an explicit model/control is **not** evidence
of physical steady-state calibration. Tracking reports that limitation and
preserves it in run provenance. New and old artifact digests identify different
inputs; do not combine their results as if only random particle samples changed.

## Rebuilt artifacts and validation

The complete 24.8 GB source was processed with the explicit `single-pole` policy
and start time 0 ns. The rebuilt modal and zero-tail maps are active at
`field_maps/cavity_axisymmetric_m0_rftrack.h5` and
`field_maps/cavity_axisymmetric_m0_zero_tail_control.h5`. Matching raw-amplitude
controls are active in `field_maps/uncorrected_control/`. The previous artifacts
and reports are preserved in `field_maps/fill_audit_v1_archive/`, with file hashes
in its `archive_manifest.json`; rebuild copies remain in `field_maps/fill_audit_v2/`.

The new modal payload SHA-256 is
`3bc29b6ea2ede431d03ecd1ad52038fb8f6ae0ccbb2eef762537a3c172655c3e`.
[The artifact comparison](field_map_fill_artifact_comparison.json) records the
old/new factors and component-wise spatial residuals after rescaling. The
largest residual is 1.06e-7 of its component's peak, consistent with the
complex64 measured-field precision. The new modal tail contributes 0.192774 kV
to a measured complex-voltage magnitude of 1703.514 kV (0.011316%).

All four maps pass numerical qualification and payload verification. Actual
RF-Track 2.7.0 field queries at four grid locations and three RF phases per map
(48 probes total) reproduce the expected electric and magnetic phasors after
scaling to 1 MW. The tracking loader also verifies the active modal maps, power
scaling, geometry and provisional-amplitude provenance.

At the time of this fill audit, the full test suite reported **498 passed, 2 skipped, 1 failed**. The same failure
was present before this audit: `test_notebook_is_clean_and_all_code_cells_parse`
requires the main tracking notebook to have no execution counts or saved
outputs. Its existing user results were preserved; all code cells in both
notebooks parsed successfully. The later [repository cleanup](repository_cleanup.md)
archived the executed notebooks and cleared their working copies; its validation
record supersedes this historical test count.

## Absolute power and cavity parameters remain unresolved

The source records 1.8 MW and a 3 GHz port evaluation frequency, versus a
2.865417220 GHz drive. Remcom's [modal-waveguide documentation](https://support.remcom.com/xfdtd/reference/excitations/modal-waveguide.html)
describes its power setting as peak instantaneous available power. The exported
metadata do not state whether the supplier already converted it to RF-cycle
average incident power. No factor-of-two power change is justified without that
information, so this audit does not apply one.

The tracking defaults Q0=4000 and Qext=3500 imply QL=1866.67 and an amplitude
decay time of 207.36 ns. The older fill inversion implied QL=1418.05 and
157.53 ns; the corrected conditional estimate gives QL=1370.68 and 152.26 ns.
None is an independent measurement from these frames. Do not
silently replace the physical cavity parameters with the inferred ones.

The previous wall-loss plausibility argument was incomplete. Small wall loss
compared with incident power can also result from reflection. A valid energy
balance requires incident, reflected and outgoing powers, all losses, and
dU/dt during filling. Scaling a field cannot verify its absolute calibration.

The [XFDTD waveform documentation](https://support.remcom.com/xfdtd/reference/definitions/waveform-editor.html)
describes a ramped sinusoidal startup. Obtain the actual waveform/ramp and port
power history, plus ringdown or independently determined loaded Q, to constrain
the model. A converged field export with calibrated power remains the definitive
replacement. In an ideal single-pole fill, 5 time constants leave 0.67% amplitude
error; 7 leave 0.091%. See also [Jensen's cavity treatment](https://arxiv.org/pdf/1601.05230).

The missing downstream tail remains a separate, relatively small modeled
contribution. A global multiplier preserves field ratios and symmetry fractions,
but **does not preserve nonlinear beam trends**: emission, capture, return
fractions and heating can change when the absolute gradient changes.
