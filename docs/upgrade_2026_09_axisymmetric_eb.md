# Upgrade: axisymmetric E+B field maps (campaign `axisymmetric_eb_v1`)

Historical v1 figures and payload digests below describe the original upgrade.
The subsequent [fill audit](field_map_fill_audit.md) supersedes its amplitude claims;
consult the current artifact report for its exact multiplier and digest.

September 2026.  This upgrade replaced the legacy planar `.mat` electric-only field maps with a
qualified, immutable axisymmetric artifact reduced from a 24.8 GB Remcom XFdtd volume export
that contains **both** E and H.  It also fixed several defects that predate it, three of which
change results.

Read `docs/field_map_pipeline.md` for the artifact contract itself; this document is the record
of what changed, what it invalidates, and what is still open.

---

## 1. What the upgrade delivers

| | before | after |
| --- | --- | --- |
| field source | planar `.mat` slices, E only | 24.8 GB XFdtd volume, E and H |
| magnetic field in tracking | absent | `B_theta`, `B_z` threaded end to end |
| representation | `abs(x)` plane folding | cylindrical-vector `m=0` projection |
| tracking grid | 8504 x 2596 = 22.1 M points | 341 x 813 = 277 k points (79.6x smaller) |
| amplitude calibration | assumed steady state at 1.8 MW | extrapolated to steady state (see 3.3) |
| downstream coverage | full domain assumed | measured to 32.94 mm + qualified modal tail |
| provenance | nominal git revision | source + payload digests, gate versions, fail-closed markers |

The reduction is out of core: the 4-D volume is never materialized, full-volume reads are
refused outright, and each Cartesian component is read on its own staggered Yee grid and its own
timestamps, then co-located only *after* harmonic fitting.

**Production policy remains axisymmetric.**  Non-axisymmetric content is measured and reported,
never used to veto the artifact.  For this export the discarded non-`m=0` RMS amplitude is
**3.20% (E)** and **9.42% (B)** — the magnetic field is materially less axisymmetric than the
electric field, which is a property of the cavity worth knowing, not a defect.

---

## 2. Reproducing the current artifact

```bash
.venv/bin/python prepare_xfdtd_fieldmap.py \
  --output field_maps/cavity_axisymmetric_m0_rftrack.h5 \
  --also-write-zero-control field_maps/cavity_axisymmetric_m0_zero_tail_control.h5 \
  --fill-correction single-pole --fill-start-time-ns 0 \
  --overwrite
```

The build that produced the shipped artifact took **22.8 minutes** (`elapsed_s` in the report),
including a full stream-and-verify of the 24.8 GB source checksum (the default;
`--no-verify-source-sha256` requires an explicit recorded prior-confirmation flag).

| | |
| --- | --- |
| source SHA-256 | `095ead03c4d20133e55d63c39bc61608304c54ea965d09d8571edd3046072ab7` |
| artifact payload SHA-256 | `ec4435c3bfbaefa56bfc43214fab425b38c73640f7eae654f0c86157bc764b65` |
| grid | 341 r x 813 z, dr 100.04 um, dz 49.99 um |
| frequency | 2.865417220 GHz |

Controls kept deliberately, so each modelling decision has an A/B:

- `field_maps/cavity_axisymmetric_m0_zero_tail_control.h5` — identical, tail zeroed instead of
  modelled.
- `field_maps/uncorrected_control/` — identical, without the fill correction of section 3.3.
- `--rf-magnetic-field zero` / `RFTRACK_RF_MAGNETIC_FIELD=zero` — E-only tracking control.

---

## 3. Defects fixed that change results

### 3.1 The azimuthal magnetic field reached RF-Track with the wrong sign

`RF_FieldMap_2d` takes its azimuthal magnetic argument along **`-e_theta`**, opposite to the
right-handed convention the artifact stores.  Probing the built volume on the +x axis (where
`e_theta = +y_hat`) returned `B_y / Re(B_theta) = -1.000000` exactly at every `(r, z, t)`, while
`E_r`, `E_z` and `B_z` all returned `+1.000000`.

The artifact's sign is the physical one: it satisfies the `m=0` Faraday relation
`dEr/dz - dEz/dr = -i*omega*B_theta` (Faraday family-normalized RMS over the guarded vacuum: **0.1471 as stored, 1.2751 negated** --
reproduce with `rf_gun.fieldmaps.quality.maxwell_residual_diagnostics`).
`build_volume` now negates `B_theta` exactly once, at the RF-Track boundary.

Left unfixed, the transverse RF Lorentz force would have had the wrong sign — `q(E_r + beta_z c
B_theta)` instead of `q(E_r - beta_z c B_theta)` — on **every** run with `--rf-magnetic-field
artifact`.  Do not "fix" this with `direction=-1`: that also conjugates the E phasor.

A second RF-Track behaviour found alongside it: **RF-Track silently discards the magnetic map
when every supplied phasor is purely real.**  Such a volume returns `B = 0` exactly while E
still reads back correctly, so a run looks like a working E+B run with no RF B at all.  Giving
any one array a nonzero imaginary part (even `1e-9`) restores it.  Real artifacts always carry
genuine phase, so this only bites synthetic maps; `build_volume` now refuses that case rather
than allowing it.  Both behaviours are pinned by `tests/test_rf_magnetic_sign_convention.py`.

### 3.2 Tangential E grew toward the cathode, violating the PEC boundary condition

Yee staggering places the tangential-E and normal-H nodes **exactly on** the conductor plane
(measured: beam z = -1.216 um, 1.2% of a 100 um cell), where those quantities are zero by
boundary condition.  The stencil selector used a bare sign test and discarded that node,
forcing a ~99 um linear extrapolation through strongly curved field.

The result was a tangential field on the emission plane *larger* than half a cell downstream:

| r (mm) | `abs(Er)` at z=0 / at z=50 um, before | after |
| ---: | ---: | ---: |
| 0.5 | 0.15 | 0.01 |
| 1.0 | **2.68** | 0.22 |
| 1.5 | **2.54** | 0.36 |

Those values sat exactly on the thermionic emission plane and the back-bombardment landing
plane.  `one_sided_surface_limit` now admits a sample within a small fraction of a cell of the
surface and snaps its offset to zero so it anchors the fit.

This is component-aware for free: normal E — the emission-driving field — straddles the
boundary with nodes at -51 um and +49 um, so at the 0.1-cell default it stays **strictly
one-sided** and is never averaged across the PEC (which would halve it).  Verified: `Ez(r=0,
z=0)` is bit-identical before and after.

Known remaining limitation: the artefact persists at r ~ 2 mm, **outside** the 1.6 mm cathode
edge, in the cathode-to-holder annular gap.  There z = 0 is not a conductor plane at all, so the
one-sided model is wrong in kind rather than imprecise.  Fixing it needs the holder geometry and
changes nothing for emission or back-bombardment.

### 3.3 Filling correction: original implementation superseded by audit

The source is still filling, but the old 2.127810x correction was conditional on
an ideal step drive and zero initial field. It was not a measured steady-state
calibration. Its drift was evaluated at the wrong epoch and was an unsigned
complex-slope norm. The current implementation fixes both and applies the
single-pole model only through an explicit choice. The notebook and CLI now share
that treatment. See [field_map_fill_audit.md](field_map_fill_audit.md).

The archived v1 multiplier, tau=157.5 ns and QL~1418 describe that old assumed
model. The source does not independently establish these parameters. The previous
wall-loss plausibility argument omitted reflected power and is withdrawn.
The source's nominal 1.8 MW setting also needs a peak-versus-cycle-average check.

---

## 4. Defects fixed that do not change physics

- **`verify_koa_run.py` crashed on a relative `--run-dir`** — which is how every launcher passes
  it — so the SLURM completion/resume gate died with a `ValueError` instead of returning a
  verdict.  The fix added the missing `.resolve()` at the one comparison that lacked it.
  (`verify_koa_run.py` is itself untracked, so this cannot be confirmed by diff; all three
  `run_path.resolve()` call sites in `verify_transport_completion` are load-bearing today.)
  This is new schema-v2 code, so the path had never been exercised against a real
  relative-path run.
- **Quality resolution was fail-open**: an empty or truncated gate mapping resolved to
  `qualified`, i.e. absence of evidence read as a pass.  It now requires the full required gate
  set at the current gate version.
- **`axis_Er_zero` / `axis_Btheta_zero` were vacuous**: the axis values are hard-zeroed, then the
  gate compared those forced zeros against a threshold they could not fail.  They now gate the
  *pre-forcing* projection residual,
  which actually measures `m=0` consistency (currently 4.8e-19 and 2.8e-18).
- **The legacy `.mat` path was still the production default**, so a bare
  `run_thermionic_tm010.py` silently ran unqualified, B-free data.  Now a hard error, with
  `--allow-legacy-mat-fieldmap` as a deliberate escape hatch.
- **The E-only and zero-tail controls were unreachable from automation** (hardcoded in all six
  launchers).  Now `RFTRACK_RF_MAGNETIC_FIELD` / `RFTRACK_FIELD_TAIL_POLICY`, defaults unchanged,
  failing closed on an unrecognised value.
- **Back-bombardment events with a non-finite hit position** were binned by `searchsorted`, which
  clips NaN to a fixed corner cell and dumped the whole event energy there.  Now accounted
  separately and named in the energy-closure equation.
- **Rim-reassigned energy was invisible**: conserved, but nothing recorded how much of the
  deposit had been moved to a neighbouring cell.  Now reported per run.
- **Two copies of the `bb0_energy_closure` block had diverged** (driver vs study module); a test
  now fails if they do again.
- Figure fixes: a missing-glyph box in the field-map title (`cmr10` has no U+2014), a `B_z` panel
  whose colour limit sat 67x below its own maximum so it showed only saturation, a hatch label
  that mislabelled the zero-tail control, a tail-decay figure that truncated at measured support
  and hid the modal continuation it exists to show, and a colour-inverted vacuum-mask figure.

---

## 5. What this invalidates

The KOA results under `KOA results/runs/` (Study III-bis 1500-1700 K, Study IV deflection)
**predate every defect in section 3** and were E-only.  They should be treated as superseded
rather than as a baseline to difference against:

- the `B_theta` sign error (3.1) affected no E-only run, but any comparison against a new E+B run
  is meaningless unless the new run has the fix;
- the original v1 fill correction (3.3) rescaled every gradient by 2.128x; its audited successor remains a provisional model;
- the thermal rim-energy loss (fixed earlier in this upgrade) was ~50-59 W over a 10 us
  macropulse, **larger than the ~18 W total difference Study IV was trying to resolve**.

They were also produced on **RF-Track 2.7.1** while this workstation has **2.7.0**, so
`verify-transport` will not re-verify them here and any local rerun is tagged 2.7.0.  Comparing a
fresh local run against the stored numbers would confound the field-map changes with a solver
version change.  **Baselines must be local-vs-local.**

---

## 6. First pilot result: what the RF magnetic field actually does

Paired runs at 1650 K — identical seed, particle count, finesse, power and artifact; only
`--rf-magnetic-field` differs, so the *same* launch population is tracked through both fields.

| fate | E+B | E-only | change |
| --- | ---: | ---: | --- |
| transmitted | 5363 | 5450 | **-1.60%** |
| lost on aperture | **146** | **66** | **+121%, 5.5 sigma** |
| backward returned (back-bombardment) | 4491 | 4484 | +0.16% (noise) |

Emission current, current density and crest phase are **bit-identical** between the arms — the
correct null, since emission depends on the normal E at the cathode and the phase scan runs
on-axis where `B_theta -> 0`.  It confirms B enters transport and nothing else.

**The RF magnetic field acts on aperture losses, not on back-bombardment.**  `B_theta` adds a net
radial deflection that pushes marginal particles onto the aperture, while returning particles
come back near the axis where `B_theta` vanishes.

Consequence for Study IV: the standing worry was that omitting the intrinsic RF B had biased the
back-bombardment heating results.  This measurement says it largely did not — the effect lands on
transmission instead.  Transmission and exit-charge numbers do need rebaselining.

Data: `outputs/pilots/rf_magnetic_field_1650K/` (N1500 and N10000, both arms).

---

## 7. Open items

1. **Re-run Study III-bis and Study IV** with the corrected artifact, locally, for a
   local-vs-local baseline (section 5).
2. **Study IV at 0 / 1 A** with the fixed thermal mask, to revisit whether the deflector reduces
   total cathode heating — the previous ~18 W signal was smaller than the energy the mask bug was
   dropping.
3. **Thermal macro-bin convergence** is tabulated (20/10/5 ns -> 1667.2505 / 1667.2985 /
   1667.3226 K, monotone, finest-pair change 0.024 K, Richardson limit 1667.3467 K, so even 20 ns
   leaves the 20 ns bin 0.096 K below that limit and the 5 ns bin 0.024 K below it) — but on a
   2k-particle smoke capture.  Re-run on a production capture:
   `.venv/bin/python tools/thermal_bin_convergence.py --run-dir <run>`.
4. **Study I / I-bis cannot run**: the ten per-grid artifacts do not exist.
   `tools/build_grid_convergence_artifacts.py` is the recipe (~4 h for the full ladder).  Note the
   historical ladder (dr 3.2-23.8 um, dz 10.4-77.5 um) was **entirely finer than the ~100 um
   source cell**, so it only re-interpolated the same solver samples and never included the
   production grid; the new default ladder brackets the production grid in sqrt(2) steps instead.
5. **Request a converged solver run** to retire the model-based fill correction of section 3.3.
6. **`tests/` is untracked.**  `.gitignore` says the regression suite "must be versioned", and it
   is not ignored — it has simply never been `git add`ed.  485 collected tests (483 pass,
   2 skip) live outside version control, and so does most of this upgrade's own surface:
   `prepare_xfdtd_fieldmap.py`, `verify_koa_run.py`, `pyproject.toml`, `rf_gun/fieldmaps/`,
   `tools/`, `docs/` and `XFdtd_field_map_preprocessing.ipynb` are all untracked.
7. The r ~ 2 mm surface artefact outside the cathode edge (section 3.2), if the annular gap ever
   matters.

---

## 8. Verification status

Every claim above was checked by running the thing, not by reading it.

| | |
| --- | --- |
| test suite | 483 passed, 2 skipped (348 observed when this handover began; `tests/` is untracked, so no baseline is recorded in history) |
| `XFdtd_field_map_preprocessing.ipynb` | 13/13 cells, 8 figures, 0 failures |
| `UH_gun_tracking_demo.ipynb` | 24/24 cells, 18 figures, 0 failures (5k particles, 8 us macropulse) |
| SLURM code path | `verify-artifact 0`, `transport 0`, `verify-transport 0` (`EXACT_MATCH`) |
| artifact gates | 10/10 pass; Faraday 0.132 / 0.20, Ampere 0.00066 / 0.20, temporal fits ~5e-4 / 0.02 |

Executed notebooks are preserved under `outputs/notebook_execution/`.
