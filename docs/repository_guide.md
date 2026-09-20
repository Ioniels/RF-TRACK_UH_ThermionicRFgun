# Repository guide

Keep physics and reusable diagnostics in `rf_gun/`. Entry-point scripts parse
requests and manage runs; notebooks expose the choices and display the results.
Tests provide independent reference cases and failure checks.

## Where work belongs

| Location | Role |
| --- | --- |
| `rf_gun/fieldmaps/` | Source inspection, reduction, fill treatment, quality gates, tails and artifact handoff |
| `rf_gun/fieldmaps/assessment.py` | Shared post-reduction assessment used by CLI and notebook |
| `rf_gun/fieldmaps/tracking.py` | Shared qualified-artifact loading, geometry checks, power scaling and run provenance |
| `rf_gun/provenance.py` | Bounded file hashing and canonical hashes of JSON-native run records |
| `rf_gun/studies/` | Reusable studies over model inputs, including heating and mirror-mesh sensitivity |
| `rf_gun/plotting/` | Figure construction and saving; no second implementation of the physics |
| `prepare_xfdtd_fieldmap.py` | Field-preparation CLI |
| `run_thermionic_tm010.py` | Single tracking-run CLI |
| `run_back_bombardment_macropulse.py` | Heating-study CLI using captured events |
| `verify_koa_run.py` | Completion and input-identity checks for cluster runs |
| `KOA_slurm_scripts/` | Distinct campaign settings and scheduling; their differences are intentional |
| `tools/` | Optional audit, artifact-generation and convergence drivers |
| `tests/helpers.py` | Shared synthetic inputs used only by tests |
| `field_maps/` | Source inputs, active processed fields and explicit physical controls |
| `outputs/`, `archive/` | Local results and preserved historical copies; ignored by Git |

Compatibility aliases for the former private CLI artifact helpers remain
available. New code should import `load_qualified_artifact_for_tracking` and
`artifact_tracking_provenance` from `rf_gun.fieldmaps`.

## Notebooks

- [Tracking](../UH_gun_tracking_demo.ipynb): configure, verify the field, calibrate
  RF phase, track, inspect beam diagnostics, compute heating, then validate/save.
- [Preprocessing](../XFdtd_field_map_preprocessing.ipynb): review the existing
  artifact by default. Reduction, writing, overwriting and figure saving have
  separate switches. A rebuild assesses its own fields before displaying tail
  values and never silently replaces them with an older on-disk artifact.
- [Physics reference](notebook_physics_reference.md): expanded equations and
  explanations moved out of the tracking notebook.

Use a fresh kernel and run top to bottom. Working notebooks ship without
execution counts or outputs; save scientific results in run directories. The
executed notebooks that preceded this cleanup are preserved in
`archive/repo_cleanup_20260915T003018Z/source_before_cleanup.tar.gz`.

The zero-tail comparison requires the same measured E/B fields and grid in both
maps. Different fill multipliers or cathode treatments invalidate a tail-only
comparison and now raise an error.

## Reusing a former test-only diagnostic

The mirror-mesh sweep is now available without running pytest:

```python
from rf_gun.studies.mirror_field import sweep_mirror_mesh

sweep = sweep_mirror_mesh(
    rft, bunch_matrix, probe_x_m, probe_y_m, probe_z_m,
    mesh_sizes=(16, 24, 32, 48, 64), mirror_z_m=0.0,
)
mirror_fields = sweep.mirror_Ez_Vpm
record = sweep.as_dict()
```

The bunch and probes stay fixed across meshes; probe coordinates use metres.
The returned free/mirrored fields and runtimes are evidence for that sample.
The regression tests retain their independent sign and spread assertions.
Analytic reference models, synthetic COMSOL writers and test-specific inputs
remain in tests so production code does not depend on its own test oracle.

## Development checks

Install the project with `python -m pip install -e '.[notebook,test]'` after
installing the supported RF-Track binding separately. The declared NumPy minimum
is 2.0 because the source uses `np.trapezoid` and `numpy.lib.array_utils`.
Material YAML and CSV files are included in package builds.

```bash
python -m pytest -q
python -m pyflakes rf_gun tools tests prepare_xfdtd_fieldmap.py \
  run_thermionic_tm010.py run_back_bombardment_macropulse.py verify_koa_run.py
```

The notebook contract tests check both notebooks for clean outputs and valid
Python. The transport notebook also retains its completion-marker checks.
Scientific field controls and legacy reference inputs are retained: similar
file names do not make them interchangeable. Local fill-audit archive/staging
copies are ignored rather than mixed with active artifacts in Git.
