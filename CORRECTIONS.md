# Corrections and Consolidation Notes

This document records the consolidated corrections applied to the current RF-Track thermionic TM010 workflow.

## 1. Module Organization and Imports

- The RF parameter helpers were moved to `rf_gun/rf_params.py`.
- Root-level `rf_params.py` was removed.
- Notebook and batch workflow now import from `rf_gun.rf_params`.

Physics impact:

- No model change; this is a code-organization cleanup for a single consistent RF-Track code path.

## 2. Thermionic Timing Injection in RF-Track

- Launch timing now uses the extended `Bunch6dT` constructor with explicit `T0` in the input matrix.
- Legacy dependence on optional Python methods for setting timing was removed from the production path.

Physics impact:

- Injected emission-time distribution is explicit and reproducible.
- Backward/returned and loss channels are now represented consistently with timing-driven launch dynamics.

## 3. Robust Screen and Beam Summaries

- Screen diagnostics are built from explicit phase-space arrays at each screen.
- Transmission and moments are derived from actual particle rows, not fragile summary fields.
- RF-Track-native summary values are stored separately for traceability.

Physics impact:

- Screen transmission, `N`, and momentum moments are self-consistent for campaign post-analysis.

## 4. JSON Export Reliability

- Output serialization now sanitizes non-finite values (`NaN`, `inf`) to JSON-safe values.
- New timing-focused export added: `B0_timing.json`.
- Particle-class summary export added: `particle_classes_summary.json`.

Physics impact:

- Timing and class diagnostics are explicit and robust for downstream data reduction.

## 5. Plotting and Diagnostic Defaults

- Consolidated defaults for saved diagnostics:
  - `clean_e = True`
  - `clean_except_zpz = True`
  - `show_zle0 = True`

Physics impact:

- Transmitted-like view is preserved across most plots while retaining backward/reflected populations in `z-pz` diagnostics.

## 6. Runtime/Progress Path

- Transport progress path was unified to elapsed/proxy mode (no segmented runtime mode).
- Batch and Slurm behavior now follow one stable implementation.

## 7. Git/Data Hygiene

- Heavy/local folders are explicitly ignored:
  - `outputs_Koa/`
  - `manual_references/`

This prevents accidental large-data commits and keeps repository history focused on physics code and reproducible configuration.

