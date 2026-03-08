#!/bin/bash
#SBATCH --job-name=tm010_campaign_p1
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

cd /home/nbidault/rf-track

mkdir -p logs outputs

source /home/nbidault/venvs/rftrack/bin/activate

export PYTHONUNBUFFERED=1

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "RF_TRACK_NUMBER_OF_THREADS=${RF_TRACK_NUMBER_OF_THREADS:-unset}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset}"
echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-unset}"

PY=python
SCRIPT=run_thermionic_tm010.py
OUT_ROOT=outputs

fmt_num() {
  python - "$1" <<'PY'
import sys
x = float(sys.argv[1])
s = f"{x:g}"
print(s.replace('-', 'm').replace('.', 'p'))
PY
}

run_case() {
  local name="$1"
  shift
  local out_dir="${OUT_ROOT}/${name}"
  echo "============================================================"
  echo "RUN: ${name}"
  echo "OUT: ${out_dir}"
  ${PY} ${SCRIPT} \
    --output "${out_dir}" \
    --progress-bar \
    --progress-notebook-mode auto \
    "$@"
}

COMMON_BASE=(
  --run-family thermionic_tm010_campaign
  --f_hz 2.856e9
  --xy_fieldmap field_maps/XYplanarSensorData.mat
  --yz_fieldmap field_maps/YZplanarSensorData.mat
  --y_cathode_mm 12.75
  --r_max_m 0.01
  --dr_um 4.0
  --dz_um 13.0
  --ext_zmax 0.0075
  --z_min 0.0
  --r_cathode_mm 1.57
  --emission_scale 1.0
  --no-use_const_pz
  --pz_init_mevc 4.0e-3
  --emission_law RD_schottky
  --phi_eff_ev 2.1
  --beta_f 1.0
  --phasor_mode reconstruct
  --phase_scan_min 0
  --phase_scan_max 360
  --phase_scan_n 90
  --phase_scan_n_part 20
  --emission_phase_range 180.0
  --emission_phase_start 0.0
  --phase_deg 0.0
  --screen_t0_mode unset
  --screen_t0_manual_mm_c 0.0
  --no-screen_log
  --dt_mm 0.01
  --aperture_m 1.0
  --ode_algorithm rk2
  --ode_epsabs 1e-6
  --fm_nsteps 200
  --fm_tt_nsteps 200
  --save-figures
  --save-screen-json
  --screen-json-mode summary
  --save-beam-json
  --save-beam-summary
  --save-lost-particles
)

BL_COMMON=(
  --beam_loading
  --bl_q0 4000
  --bl_qext 3500
  --bl_p_fwd_w 1.0e6
  --bl_r_over_q_ohm_per_m 1.0
  --calibrate-bl-r-over-q
  --bl_ncells 1
  --bl_tinj_mode auto_from_emission
  --bl_tinj_manual_mm_c 0.0
)

# Run 1: baseline
run_case "thermionic_baseline_T1700_SC0_BL0_N100000_NZ3_RA0_RE0" \
  "${COMMON_BASE[@]}" \
  --scan-tags baseline \
  --n_particles 100000 \
  --n_z_snap 3 \
  --t_cathode_k 1700 \
  --ra_um 0 \
  --re_um 0 \
  --no-sc_enabled \
  --no-beam_loading

# Run 2: temperature scan with space charge only
for T in $(python - <<'PY'
import numpy as np
for x in np.linspace(1600, 1900, 10):
    print(f"{x:.6g}")
PY
); do
  T_TAG=$(fmt_num "$T")
  run_case "thermionic_scanT_T${T_TAG}_SC1_BL0_scdt0p01_N100000_NZ3_RA0_RE0" \
    "${COMMON_BASE[@]}" \
    --scan-tags scanT spacecharge_only \
    --n_particles 100000 \
    --n_z_snap 3 \
    --t_cathode_k "$T" \
    --sc_enabled \
    --sc_dt_mm 0.01 \
    --emission_nsteps 200 \
    --emission_range 10.0 \
    --no-beam_loading \
    --ra_um 0 \
    --re_um 0
done

# Run 3: space-charge dt scan
for SCDT in $(python - <<'PY'
import numpy as np
for x in np.linspace(0.01, 0.2, 10):
    print(f"{x:.8g}")
PY
); do
  SCDT_TAG=$(fmt_num "$SCDT")
  run_case "thermionic_scanSCdt_T1900_SC1_BL0_scdt${SCDT_TAG}_N100000_NZ3_RA0_RE0" \
    "${COMMON_BASE[@]}" \
    --scan-tags scanSCdt spacecharge_only \
    --n_particles 100000 \
    --n_z_snap 3 \
    --t_cathode_k 1900 \
    --sc_enabled \
    --sc_dt_mm "$SCDT" \
    --emission_nsteps 200 \
    --emission_range 10.0 \
    --no-beam_loading \
    --ra_um 0 \
    --re_um 0
done
