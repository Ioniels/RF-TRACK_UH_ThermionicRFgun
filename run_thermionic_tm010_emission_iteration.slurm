#!/bin/bash
#SBATCH --job-name=tm010_emission_iteration
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Emission Fields Iteration study (guide Sec. 14.3): a small, fast exploratory run of the
# self-consistent space-charge+mirror emission-field iteration (rf_gun.emission_iteration), NOT
# a production scan -- --n_particles stays at 1000 so the whole script runs in minutes. Don't add
# --emission-field-iteration to the 100,000-particle scanT_*.slurm scripts until this study shows
# it's worth the added cost there.
#
# This array scans the relaxation factor omega0 (0.20, 0.25, 0.30, 0.35), one value per task --
# swap RELAXATION_VALUES for a scan over mirror-charge tolerance, PIC mesh, or probe distance.

set -eo pipefail

cd /home/nbidault/rf-track

mkdir -p logs outputs/runs

source /home/nbidault/venvs/rftrack/bin/activate

export PYTHONUNBUFFERED=1

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

PY=python
SCRIPT=run_thermionic_tm010.py

RELAXATION_VALUES=(0.20 0.25 0.30 0.35)
OMEGA0="${RELAXATION_VALUES[$SLURM_ARRAY_TASK_ID]}"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${STAMP}_emission_iteration_omega${OMEGA0}"
OUT_DIR="outputs/runs/${RUN_NAME}"

echo "============================================================"
echo "OMEGA0=${OMEGA0}"
echo "OUT_DIR=${OUT_DIR}"

${PY} ${SCRIPT} \
  --output "${OUT_DIR}" \
  --run-family thermionic_tm010_emission_iteration \
  --scan-tags emission_iteration relaxation_scan \
  --f_hz 2.856e9 \
  --xy_fieldmap field_maps/XYplanarSensorData.mat \
  --yz_fieldmap field_maps/YZplanarSensorData.mat \
  --y_cathode_mm 12.75 \
  --r_max_m 0.0340145 \
  --ext_zmax 0.0075 \
  --z_min 0.0 \
  --r_cathode_mm 1.57 \
  --emission_scale 1.0 \
  --no-use_const_pz \
  --pz_init_mevc 4.0e-3 \
  --emission_law RD_schottky \
  --phi_eff_ev 2.1 \
  --beta_f 1.0 \
  --ra_um 0 \
  --re_um 0 \
  --phasor_mode reconstruct \
  --finesse medium \
  --phase_scan_min 0 \
  --phase_scan_max 360 \
  --emission_phase_range 180.0 \
  --emission_phase_start 0.0 \
  --phase_deg 0.0 \
  --delta_cathode_chamfer_mm 0.0 \
  --ode_algorithm rk2 \
  --sc_enabled \
  --sc-nx 32 --sc-ny 32 --sc-nz 32 \
  --mirror-charges \
  --mirror-z-m 0.0 \
  --emission_range 10.0 \
  --no-beam_loading \
  --no-deflection_enabled \
  --n_particles 1000 \
  --no-screens \
  --t_cathode_k 1700 \
  --emission-field-iteration \
  --emission-iteration-particles 1000 \
  --emission-iteration-finesse medium \
  --emission-iteration-max-iterations 12 \
  --emission-iteration-relaxation "${OMEGA0}" \
  --emission-iteration-current-tolerance 1.0e-2 \
  --emission-iteration-field-tolerance 1.0e-2 \
  --emission-iteration-charge-tolerance 1.0e-2 \
  --emission-iteration-z-max-mm 2.0 \
  --no-emission-iteration-include-beam-loading \
  --save-figures \
  --no-save-screen-hdf5 \
  --no-save-screen-json \
  --no-save-openpmd-beam \
  --save-lost-particles

echo "Emission Fields Iteration study (omega0=${OMEGA0}) completed successfully."
