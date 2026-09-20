# Notebook physics reference

Expanded explanations supporting the tracking notebook. Configuration and diagnostics remain in the notebook; the package contains the model implementations.


## Thermionic emission

**Beam parameters:**


**Thermionic model summary (used below):**

The cathode's local extraction field sets a Schottky-lowered effective work function
$\phi_{\mathrm{eff}} = \phi - \Delta\phi_{\mathrm{Schottky}}(F)$, computed from the signed local
cathode field at $z=0$ at the chosen RF phase. `EMISSION_LAW` (set below) selects which emission
kernel turns $(\phi_{\mathrm{eff}}, T, F)$ into a current density $J(F,T,\phi)$. Every kernel is
registered in `rg.EMISSION_MODEL_PLOT_LABELS`; the "Emission-model comparison" cell later in this
notebook evaluates several of them side by side over this run's own populated field range.

- **`RDSchottky`** (default) -- *"Thermionic: Richardson-Dushman-Schottky"*. Classical
  Richardson-Dushman thermionic emission with Schottky barrier lowering:
  $$J = A_R T^2 \exp\!\left[-\frac{\phi-\Delta\phi_{\mathrm{Schottky}}(F)}{k_B T}\right],\qquad
  \Delta\phi_{\mathrm{Schottky}}(F) = \frac{1}{e}\sqrt{\frac{e^3 F}{4\pi\epsilon_0}}.$$
  Simplest and fastest kernel; the production default unless a benchmark shows another model is
  materially better over this gun's actual operating field range.
- **`jensen2014_RDSchottky_MurphyGood_additive`** (alias: `unified`) -- *"Jensen2014: Additive
  regime, Thermionic: Richardson-Dushman-Schottky; Field: Murphy-Good"*. An additive combination
  of the thermionic term above with a cold Murphy-Good/Schottky-Nordheim field-emission term times
  a finite-temperature enhancement factor:
  $$J = J_{\mathrm{RLD\text{-}S}}(F,T,\phi) + J_{\mathrm{MG},0}(F,\phi)\,\lambda_T(p),\qquad
  \lambda_T(p) = \frac{\pi p}{\sin(\pi p)}.$$
  Kept as a diagnostic comparison kernel, not a production default.
- **`jensen2019_RDSchottky_MurphyGood_transition`** -- *"Jensen2019: Transition regime, Thermionic:
  Richardson-Dushman-Schottky; Field: Murphy-Good"*. Kevin L. Jensen, "A reformulated general
  thermal-field emission equation," *J. Appl. Phys.* **126**, 065302 (2019)
  (`manual_references/Thermionic Emission Models/Jensen 2019.pdf`). One closed-form current
  density spanning the thermionic, field, and thermal-field transition regimes,
  $J = A_{\mathrm{RLD}} T^2 N(n,s)$, with $n$ the ratio of thermal to field energy scales that
  distinguishes the two regimes. This repository transcribes the paper's own exact digamma-based
  series (rather than its rational approximation, which has genuine poles at integer $n$) and
  smoothly blends it into a direct energy-domain integral in a narrow band around each pole, so
  the current density stays numerically smooth in $F$ (see the `rf_gun.emission_models` module
  docstring) -- the modern, preferred fast general thermal-field model for comparison against the
  thermionic baseline.
- **`murphygood1956_SchottkyNordheim_integral`** -- *"MurphyGood1956 Integrals of transmission
  probability into Schottky-Nordheim barrier"*. E. L. Murphy and R. H. Good, Jr., "Thermionic
  Emission, Field Emission, and the Transition Region," *Phys. Rev.* **102**, 1464 (1956)
  (`manual_references/Thermionic Emission Models/Murphy 1956- manuscript.pdf`). A direct,
  deliberately slow reference,
  $$J = e\int_0^{\infty} N(E_n,T)\,D(E_n;F,\phi)\,dE_n,$$
  the free-electron supply function times a WKB/Kemble transmission probability through the exact
  Schottky-Nordheim barrier, integrated numerically rather than approximated in closed form. Used
  to validate/tabulate the faster kernels above, not for routine per-run tracking.
- *(`jensen_gtf_2007` -- Kevin L. Jensen, "General formulation of thermal, field, and
  photoinduced electron emission," J. Appl. Phys. **102**, 024911 (2007) -- is registered as a
  historical/mathematical comparison point but not yet implemented;
  `jensen2019_RDSchottky_MurphyGood_transition` supersedes it as the production general
  thermal-field model.)*

- $I(t) = J(t) \pi R^2$ using the emission radius, with $J(t)$ the chosen emission law evaluated
  at the time-dependent extraction field.
- Emission times $t_0$ are sampled from the time-dependent rate $I(t)$ derived from $E_z(t)$ at
  the cathode.
- The total emitted charge is distributed across all macroparticles.

This means the effective current and time span are set by $T$, $\phi_{\mathrm{eff}}(t)$ (optionally
itself a function of $T$ -- see `WORK_FUNCTION_TEMPERATURE_MODEL` below), and the RF phase via the
on-axis field, while the temporal profile follows $I(t)$.

## RF calibration and tracking

## 3. RF-Track simulation

This section performs an RF-only phase calibration, optionally constructs a self-consistent
thermionic source, performs one configured transport, then classifies and saves the result.
Only the calibration's `Veff`/`R/Q` feed beam loading; every other quantity here (emission
current/phase, transport, classification) is diagnostic until the final classification step below.

### RF-only phase calibration

The calibration source is truly on-axis and cold: every calibration particle has
`x=y=px=py=0` and a negligible charge, reused unchanged at every scanned phase (no resampling
between phase points). Space charge, cathode mirror charges, beam loading, the cathode backstop,
and the deflection magnet are all forced off for this scan regardless of what the rest of the
notebook has configured for production.

`PHASE_SCAN` sweeps *relative* phase (offset from the cathode's own zero-crossing, added to
`TRANSPORT_PHASE_DEG` to get the *absolute* phase RF-Track actually tracks at); the scan can
contain lost/non-finite points (a strongly decelerating phase loses every calibration particle),
so the result is only accepted if a crest exists with finite scan neighbors on both sides (a
resolved local maximum, not an isolated finite point surrounded by invalid samples) and represents
a net forward energy gain. An invalid calibration raises immediately — before `Veff`, `R/Q`, or any
beam-loading-dependent quantity is computed — rather than continuing with a degraded scan.

Effective voltage follows the relativistic conversion
$K(p) = \sqrt{p^2c^2 + m_e^2c^4} - m_ec^2$, $V_\mathrm{eff} = \Delta K / e$ between the crest's
mean exit momentum and the launch momentum; `(R/Q) = V_\mathrm{eff}^2/(P_\mathrm{del} \cdot Q_0)`
uses the cavity's *unloaded* $Q_0$ (not the loaded $Q_L$), since $P_\mathrm{del}$ is the
wall-dissipated power (`rf_gun.rf_params.r_over_q_per_m`'s docstring has the full derivation).

### Automatic transport phase

`TRANSPORT_PHASE_DEG` is **overwritten** by the on-axis field-profile cell above to the cathode's
own zero-crossing phase plus `EMISSION_PHASE_START` — not the nominal value assigned when the
variable was first declared in the configuration cell. This is distinct from the phase-scan's own
relative-phase axis and from the accelerating crest phase found by the calibration above; print the
resolved value before trusting a specific number.

### Thermionic source

The production launch model samples: transverse position over the emitting disk (uniform, or the
converged Emission Fields Iteration's areal profile when available and converged); transverse
thermal momenta `(px, py)`; longitudinal emission time from the Richardson-Dushman-Schottky (or
selected alternative) current, using the local normal extraction field at the cathode surface for
Schottky lowering; and the flux-weighted **Maxwell-Boltzmann** normal-momentum distribution (`pz`).
Particle weights represent real electron counts (`N` column of the RF-Track extended matrix), not
equal macro-charges. If the optional Emission Fields Iteration ran but did not converge, the
production source falls back to this prescribed model — printed clearly, and, for any *production*
run launched through `KOA_slurm_scripts/`, recorded as a failed `validation.json` check that
withholds the `.run_complete` marker (this notebook is exploratory and may proceed regardless; a
production run must not).

### Main RF-Track transport

The payload-verified artifact's cylindrical-vector $m=0$ electric and magnetic phasors,
the dynamic aperture, the cathode backstop, space charge (fresh PIC engine),
cathode mirror charge, `BeamLoadingSW` (attached correctly, though a real cross-validation found it
has no measurable effect on tracked dynamics for this gun geometry — see `README.md`), and the
deflection magnet (forces single-threaded tracking whenever enabled) are each independently
switchable in the configuration cells above; the dynamic transverse aperture is always enforced.
Tracking screens (`N_Z_SNAP`) are diagnostic snapshots along `z`; `Bout` is the fixed-time terminal
state RF-Track reports at the end of tracking — a different semantic than any single screen's
z-plane, not a drop-in substitute for one. `%id`-based tagging (`rf_gun.particle_tags`) uses
`Bout`'s own reliable z/pz plus RF-Track's own lost-particle table; the final classification used
for every count/plot/output below is recomputed once from those tags after the acceptance scan
(see the "Particle classification" cell), not from any earlier, now-stale classification.

The RF magnetic field is present: the artifact's axisymmetric $B_\theta$ and $B_z$ maps are
passed through the phase calibration, every emission-field iteration case, and the production
transport. `RF_MAGNETIC_FIELD="zero"` is retained only as an explicit E-only control. The
full-volume non-axisymmetry diagnostics remain in the dedicated preprocessing notebook and in
the artifact quality record; production tracking deliberately uses their stored $m=0$ projection.

### Outputs and checks

`screen_distributions_hdf5/` holds every particle-distribution HDF5 (`B0`, `Bout`, and per-screen
files); the v2 back-bombardment event capture (cathode backstop always on); a figure/data
manifest under `figures/`; and `run_config.json`/`run_results.json`/`validation.json`. A run is
only production-valid when `validation.json`'s status is `"ok"` and the `.run_complete` marker
exists — qualified artifact status with full payload verification, finite calibration and derived
RF parameters, unique particle IDs, and (when run) emission-iteration convergence are all checked
there; see `README.md`'s "Output schema and provenance" section for the full list.


## Emission iteration

### Emission Fields Iteration

A reduced-cost self-consistency study run before the full production transport: a small
fixed source sample is tracked through the near-cathode region with space charge and
mirror charges; the resulting normal field at the cathode feeds back into the emission
current, under-relaxed until current, field, and charge converge. The cathode grid is a
genuine 2D (x,y) grid, so an asymmetric `CATHODE_TEMPERATURE_K` profile gives an
asymmetric J(x,y,t) -- not an azimuthal average.

`EMISSION_ITERATION_INCLUDE_BL=True` folds a causal TM010 modal beam-loading estimate
(`E_BL(x,y,t) = -chi(t)*E_RF(x,y,t)`, same `Q_L`/`R/Q`/`Veff` as the production
`BeamLoadingSW` attach) into the self-consistency loop. **A real cross-validation found
`BeamLoadingSW` itself produces zero measurable effect on tracked dynamics for this gun
geometry in RF-Track 2.7.0** (`tests/test_beam_loading_cross_validation.py`) -- a binding
limitation, not something fixable here -- so treat `E_BL` as an independent estimate, not
one verified against RF-Track's own implementation.

`EMISSION_ITERATION_FIELD_PROBE_METHOD`: `"pic_probe"` (default) uses zero-weight probe
particles in RF-Track's space-charge engine at a fixed probe distance; alternatively
`"analytic_point_charge_image"` evaluates a closed-form conductor-image kernel exactly at
the cathode surface (no probe distance, no PIC mesh).

If this cell converges, the production run below samples (x,y,t) from its converged
J(x,y,t) surface (`rf_gun.build_bunch_thermionic_spatial`) instead of the default on-axis
`EMISSION_LAW` path; if not, it falls back to the on-axis source with a printed message.

The submodel-comparison figure below always shows both "space charge only" and "space
charge + mirror" (running a second iteration with the opposite `MIRROR_CHARGES` setting),
regardless of which one the production run actually uses.


## Back-bombardment and heating

## 5. Back-bombardment and macropulse heating study (8 µs default)

Returning electrons are captured by `rg.extract_back_bombardment_events`: a thin absorbing
backstop just behind the cathode plane (`rf_gun.aperture.build_cathode_backstop`) records each
returning particle's own loss-table state, which is then ray-cast against the true 3D cathode
geometry (`rg.CathodeGeometry.intersect_ray`) to resolve which surface -- flat face, 45° LaB6
bevel, or metal holder (unmodeled) -- it struck. The bevel's physical area (not its z=0 projection,
which understates it by `1/cos(45°)`) is used for areal quantities. Full derivation and open
items: `BACK_BOMBARDMENT_MACROPULSE_IMPLEMENTATION_PLAN.md` (Sec. 3, 8, 11-14, addendum Sec. 19).

**Incident vs. deposited energy.** Every event carries incident kinetic energy at impact; how much
is actually deposited in the LaB6 is model-dependent. `BB0_TIO` (a CSDA/Tabata extrapolated-range
law truncating incident energy at the local electron range) is the only implemented model here --
`BB1`/`BB2` (uncertainty bands, full angular/energy response) are future work.

**One RF period vs. the macropulse.** RF-Track tracks a single representative cycle at full
fidelity; that qualified event set is then treated as a periodic source and accumulated onto a
coarser macro-time grid over `MACROPULSE_DURATION_US` (8 µs default) as a declared top-hat
idealization -- no measured RF fill/decay envelope exists yet.

**Python vs. COMSOL.** `python_xy_layered` (fast, `(x,y)`-resolved, depth-layered) is the only
thermal backend that runs here. The `comsol_3d` exchange interface
(`rg.export_comsol_heat_source`/`rg.load_comsol_thermal_result`) exists but no COMSOL run has been
performed, so `COMSOL_RESULTS=None` and the figures omit COMSOL curves rather than fabricate them.

**Two source modes.** `BB_SOURCE_MODE="current_notebook"` consumes this run's just-captured events
directly; `"load_run"` loads a completed run's `back_bombardment_events.h5` so the thermal/
macropulse parameters can be re-explored without re-running RF-Track. Both resolve to the same
`rg.BackBombardmentStudyInput`.

**Materials.** Properties come from the registry set `LaB6_UH_recommended_v1`
(`rg.load_cathode_material`; Ivashchenko 2018, Adachi 2009, Tanaka Debye parameters, Sun, Williams
LBL-27907, Kowalczyk 2014, corrected Tabata/Bakr TIO coefficients -- plan Sec. 5/addendum Sec.
19.3). Benchmark values (`Kowalczyk_RSI084905_2013`, `Kowalczyk_PRSTAB120402_2014`) use their own
historical 5 µs configuration, never rescaled to 8 µs and presented as validation.

**Scope.** Only `coupling_level="L2_one_way"` (qualified source -> macropulse heating, no
temperature-to-emission feedback) is implemented; `L3`/`L4` feedback loops are out of scope here.


## Beam properties

### Beam properties vs. z

`rf_gun.beam_properties.compute_beam_properties` builds one row per screen, computed **only** on
the forward-going, aperture-surviving population at that screen (via the `%id`-based `tags` built
above) -- the same population used everywhere else in this section.

For each screen, with $x' = p_x/p_z$, $y' = p_y/p_z$ and $\langle\cdot\rangle$ the mean-centered,
population second moment:

$$\epsilon_x = \sqrt{\langle x^2\rangle\langle x'^2\rangle - \langle x x'\rangle^2}, \qquad
\alpha_x = -\frac{\langle x\,x'\rangle}{\epsilon_x}, \qquad
\beta_x = \frac{\langle x^2\rangle}{\epsilon_x}, \qquad
\gamma_x = \frac{1+\alpha_x^2}{\beta_x}$$

(and equivalently for $y$), with normalized emittance $\epsilon_{x,n} = \epsilon_x\,(\bar p_z/m_ec^2)\times10^3$ [mm mrad].

Dispersion, with $\delta = (p_z-\bar p_z)/\bar p_z$:

$$D_x = \frac{\mathrm{Cov}(x,\delta)}{\mathrm{Var}(\delta)}\ \mathrm{[mm]}, \qquad
D_x' = \frac{\mathrm{Cov}(x',\delta)}{\mathrm{Var}(\delta)}\ \mathrm{[rad]}$$

(and equivalently for $y$). Energy/time columns come from the extended phase-space format's
`%E`/`%t`/`%K`: $\sigma_E$ (energy spread), $\sigma_t$ (time-of-flight spread), $\langle K\rangle$
(mean kinetic energy).

`rf_gun.beam_properties.transmission_curves` reports two curves vs. $z$, both as a fraction of
$N_\mathrm{initial}$: backward + forward (particles that survive the dynamic aperture, in either
direction) and forward only (that same surviving population, backward-tagged particles removed).

Every longitudinal quantity above (`mean_t_ns`, `sigma_t_ns`, $\alpha_t,\beta_t,\gamma_t,\epsilon_t$) uses ToF (`%t`), not `%Z`: a screen's own `%Z` is not a lab-frame position (see the aperture-diagnostics cell above), while `%t` is reliable at every screen -- so the same (ToF, $p_z/\bar p_z$) convention that used to read `z` now reads ToF throughout, including the phase-space panels.



## Run records

## 6. Run config + results (hardcoded + derived parameters, key outputs)

Everything needed to reproduce or re-interpret this run later, without re-reading the notebook,
split into two files under `RUN_DIR` (each a plain dict of JSON-safe types):

- `run_config.json` -- every hardcoded input parameter above, plus the handful of quantities
  derived from them *before* tracking starts (grid sizes, phase-scan crest phase, `Veff`, `R/Q`).
- `run_results.json` -- everything the run *found*: peak current/current density, the
  beam-property curves vs `z` (transmission, Twiss, beam size), particle classification,
  aperture/back-bombardment/openPMD summaries, and the paths of every other file this run saved.

So future analysis can do just:

```python
import json
cfg = json.load(open("outputs/runs/<run-name>/run_config.json"))
res = json.load(open("outputs/runs/<run-name>/run_results.json"))
cfg["derived_parameters"]["phase_scan"]["crest_phase_deg"]
res["results"]["peak_current_density_A_cm2"]
```
