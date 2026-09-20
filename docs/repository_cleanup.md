# Repository cleanup

The cleanup consolidates duplicated workflows while preserving the active
field artifacts and existing simulation results. A source snapshot, including
both notebooks with their original outputs, is saved at
`archive/repo_cleanup_20260915T003018Z/source_before_cleanup.tar.gz`.
The [machine-readable record](repository_cleanup.json) lists affected files and
notebook sizes. The [repository guide](repository_guide.md) describes where new
code and diagnostics belong.

## Shared implementations

- `rf_gun.fieldmaps.tracking` now owns artifact validation, aperture/domain
  checks, power scaling and provenance for the notebook and CLI. Compatibility
  aliases preserve the former private CLI entry points.
- `rf_gun.fieldmaps.assessment` applies the selected fill treatment and evaluates
  the numerical/tail/production gates once. CLI and preprocessing notebook use
  this same function.
- `rf_gun.provenance` owns bounded file hashing and canonical hashes of run
  records. Artifact reports share the artifact serializer for complex and NumPy
  values. Scientific serialization remains strict; run summaries retain their
  existing policy of converting nonfinite values to JSON null.
- The mirror-field mesh sweep moved from a test loop into
  `rf_gun.studies.mirror_field`. Its sign/spread assertions remain independent
  regression checks. Three tests share one Gaussian bunch input builder.
- Two schema-rejection tests now create an unversioned HDF5 fixture instead of
  depending on a hardcoded local run directory. These checks no longer skip
  when historical user outputs are absent.

An AST scan found no remaining exact function-body duplicates among Python
functions of six or more lines. Similar analytic checks, distinct physical
controls and campaign scripts were retained where their differences matter.

## Notebook clarity and correctness

The tracking notebook shrank from 4,502,576 bytes to 110,932 bytes after archiving
and clearing outputs. Its longest code cell shrank from 469 lines to 184 lines.
Configuration, derived RF quantities, results and completion checks now have
separate cells; beam diagnostics appear together before the heating study.
Expanded physical explanations moved to [a reference document](notebook_physics_reference.md).

The fill display reports the applied policy and provisional status, including
raw-amplitude controls, without calling local stationarity verified convergence.
Preprocessing evaluates tail diagnostics after fill treatment, displays a fresh
reduction instead of a stale artifact, writes matching controls when requested,
and rejects tail comparisons whose measured fields differ. Expensive writes
check destination collisions before reduction. Both working notebooks are clean.

## Maintenance fixes

Unused imports and dead assignments were removed across the package, scripts
and tests. JSON output writes now use separate temporary files per writer and
clean up failed writes, avoiding races on a shared `.tmp` file.

The package declares the NumPy 2.0 minimum required by its existing APIs,
notebook dependencies and static-check dependency. Material YAML and CSV files
are included in wheels. Generated build directories and local field-audit
copies are ignored; active maps and controls remain visible to version control.

## Validation

- Repository Python and both notebooks pass Pyflakes checks.
- All 13 preprocessing code cells executed in review mode. The tracking
  notebook executed 14 startup/field-diagnostic cells with saving disabled.
  Full beam transport and heating were not rerun for this cleanup.
- The shared assessment reproduces the active multiplier, all four measured
  fields, qualification and tail metrics from the matching raw control.
- An offline wheel build includes all 11 material files. Extracting the wheel
  outside the checkout successfully imports the new modules and loads the
  recommended material. The local venv lacks setuptools; this packaging check
  used system setuptools 68.1.2 directly, without changing the environment.
- Final full pytest: **506 passed, none skipped**, in 93.14 seconds. The previous
  saved-output notebook failure is resolved, and the two previously skipped
  schema checks now use portable fixtures.

Run completion checks bind results to code identity. This refactor changes that
identity even when the numerical result is preserved, so existing completion
markers are not evidence that a run used the cleaned code.
