# UH Gun Thermionic RF-Track Project

Thermionic electron-beam dynamics in an S-band TM010 (lambda/4) RF gun using **RF-Track** for 6D transport, space charge, and beam loading.

## Scope

This repository provides a consolidated simulation workflow for:

- RF phasor reconstruction from measured field maps
- thermionic emission (Richardson-Dushman with Schottky lowering)
- surface roughness on emitter (`Ra`, `Re`)
- RF-Track transport with optional space charge and beam loading
- robust physics-oriented diagnostics and JSON exports for campaign post-analysis

## Physics Model (Current)

- Cavity frequency: `f = 2.856 GHz` (TM010)
- Axisymmetric field map used by RF-Track is built from XY/YZ measurements
- Emission timing is phase-windowed (`emission_phase_start`, `emission_phase_range`)
- Time-dependent emission current is sampled and injected into RF-Track
- **Bunch timing injection uses extended `Bunch6dT` matrix with explicit `T0`**, not optional Python binding methods

Important coordinate note:

- In `Bunch6dT`, `Z` is spatial coordinate
- `T0` is particle creation time and is stored separately from the 6D phase-space columns

## Project Layout

```text
.
├── UH_gun_tracking_demo.ipynb
├── run_thermionic_tm010.py
├── run_thermionic_tm010.slurm
├── run_thermionic_tm010_campaignPart1.slurm
├── run_thermionic_tm010_campaignPart2.slurm
├── run_thermionic_tm010_campaignPart3.slurm
├── run_thermionic_tm010_campaignPart4.slurm
├── rf_gun/
│   ├── simulation.py
│   ├── rf_params.py
│   ├── diagnostics.py
│   ├── io.py
│   └── plotting/
├── field_maps/
├── outputs/
├── outputs_Koa/              # local heavy data (git-ignored)
└── manual_references/        # local references (git-ignored)
```

## Core Workflow

1. Load XY/YZ field maps and compute envelope diagnostics.
2. Build RF phasor (`reconstruct` or `simplified`) and interpolate on `(r,z)` grid.
3. Run fast phase scan for effective voltage / R-over-Q calibration.
4. Run thermionic transport with RF-Track (SC/BL configurable).
5. Export robust summaries and phase-space diagnostics.

## Robust Outputs (Current Standard)

Per run, the batch workflow produces JSON outputs designed for campaign statistics:

- `run_metadata.json`
- `beam_summary.json`
- `progress_stats.json`
- `B0.json`, `Bout.json`
- `B0_timing.json`
- `particle_classes_summary.json`
- `lost_particle_diagnostics.json` (when enabled)
- `screen_distributions_json/`

Screen summaries are derived from explicit phase-space arrays (counts and moments), with RF-Track-native fields kept separately for traceability.

## Batch Execution

Quick validation run:

```bash
python run_thermionic_tm010.py --preset quick --output outputs/smoke_quick
```

Campaign production runs are launched through the `run_thermionic_tm010_campaignPart*.slurm` scripts.

CLI reference:

```bash
python run_thermionic_tm010.py --help
```

## Plotting/Diagnostics Defaults

Saved diagnostics use explicit defaults aligned with transport interpretation:

- `clean_e = True`
- `clean_except_zpz = True`
- `show_zle0 = True`

This preserves backward/reflected populations in `z-pz` while keeping other panels focused on transmitted-like particles.

## Data and Git Policy

Heavy or local-only folders are intentionally ignored:

- `outputs/`
- `outputs_Koa/`
- `manual_references/`
- `archive/`

## RF-Track Reference

- RF-Track project page: `https://abpcomputing.web.cern.ch/codes/codes_pages/RF-Track/`

