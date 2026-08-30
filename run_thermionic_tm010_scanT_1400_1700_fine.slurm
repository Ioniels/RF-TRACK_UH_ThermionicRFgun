#!/bin/bash
#SBATCH --job-name=tm010_scanT_1400_1700_fine
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --array=0-30
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Temperature scan: T = 1400-1700 K in 10 K steps (31 runs), one per array task
# (SLURM_ARRAY_TASK_ID 0..30 -> T = 1400 + 10*task_id). `--finesse fine`. N=100,000 particles,
# 2 z-snap screens. Deflection magnet OFF: it forces single-threaded tracking, which would leave
# --cpus-per-task=32 mostly idle.

# Mirror charges are explicitly OFF (validated, tests/test_mirror_charge.py) -- must be enabled
# deliberately rather than inherited from a CLI default.

set -eo pipefail

cd /home/nbidault/rf-track

mkdir -p logs outputs/runs

source /home/nbidault/venvs/rftrack/bin/activate

export PYTHONUNBUFFERED=1

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "RF_TRACK_NUMBER_OF_THREADS=${RF_TRACK_NUMBER_OF_THREADS:-unset}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset}"
echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-unset}"

PY=python
SCRIPT=run_thermionic_tm010.py

# T = 1400 + 10 * task_id, task_id in [0, 30] -> T in [1400, 1700]
T_CATHODE_K=$(( 1400 + 10 * SLURM_ARRAY_TASK_ID ))

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${STAMP}_T${T_CATHODE_K}K_SCon_BLon"
OUT_DIR="outputs/runs/${RUN_NAME}"

echo "============================================================"
echo "T_CATHODE_K=${T_CATHODE_K}"
echo "OUT_DIR=${OUT_DIR}"

${PY} ${SCRIPT} \
  --output "${OUT_DIR}" \
  --run-family thermionic_tm010_scanT_1400_1700 \
  --scan-tags scanT_1400_1700 finesse_fine beam_loading space_charge \
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
  --finesse fine \
  --phase_scan_min 0 \
  --phase_scan_max 360 \
  --emission_phase_range 180.0 \
  --emission_phase_start 0.0 \
  --phase_deg 0.0 \
  --screen_t0_mode unset \
  --screen_t0_manual_mm_c 0.0 \
  --no-screen_log \
  --delta_cathode_chamfer_mm 0.0 \
  --ode_algorithm rk2 \
  --sc_enabled \
  --sc-nx 32 --sc-ny 32 --sc-nz 32 \
  --no-mirror-charges \
  --emission_range 10.0 \
  --beam_loading \
  --bl_q0 4000 \
  --bl_qext 3500 \
  --bl_p_fwd_w 1.0e6 \
  --bl_r_over_q_ohm_per_m 1.0 \
  --calibrate-bl-r-over-q \
  --bl_ncells 1 \
  --bl_tinj_mode auto_from_emission \
  --bl_tinj_manual_mm_c 0.0 \
  --no-deflection_enabled \
  --n_particles 100000 \
  --n_z_snap 2 \
  --t_cathode_k "${T_CATHODE_K}" \
  --save-figures \
  --save-screen-hdf5 \
  --no-save-screen-json \
  --save-openpmd-beam \
  --save-lost-particles

echo "Run T=${T_CATHODE_K}K completed successfully."
