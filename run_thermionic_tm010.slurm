#!/bin/bash
#SBATCH --job-name=tm010_thermionic
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

# Optional: activate project venv if present
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export RF_TRACK_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_ID="${SLURM_JOB_ID}"
OUT_DIR="outputs/slurm_${RUN_ID}"
mkdir -p "$OUT_DIR"

# ---------------- Simulation knobs (edit here) ----------------
N_PART=100000
DT_MM=0.1
SC_DT_MM=0.2
EMISSION_NSTEPS=100
EMISSION_RANGE=10.0
N_Z_SNAP=100
EMISSION_PHASE_START=0.0
EMISSION_PHASE_RANGE=180.0
T_CATHODE_K=1700.0
PHI_EFF_EV=2.1
BETA_F=1.0

# Beam loading toggle (true/false)
USE_BEAM_LOADING=false

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "RF_TRACK_NUMBER_OF_THREADS=$RF_TRACK_NUMBER_OF_THREADS"
echo "Output directory: $OUT_DIR"

python run_thermionic_tm010.py \
  --threads "$SLURM_CPUS_PER_TASK" \
  --output "$OUT_DIR" \
  --n_particles "$N_PART" \
  --dt_mm "$DT_MM" \
  --sc_dt_mm "$SC_DT_MM" \
  --emission_nsteps "$EMISSION_NSTEPS" \
  --emission_range "$EMISSION_RANGE" \
  --n_z_snap "$N_Z_SNAP" \
  --emission_phase_start "$EMISSION_PHASE_START" \
  --emission_phase_range "$EMISSION_PHASE_RANGE" \
  --t_cathode_k "$T_CATHODE_K" \
  --phi_eff_ev "$PHI_EFF_EV" \
  --beta_f "$BETA_F" \
  --sc_enabled \
  --save-screen-json \
  --save-figures \
  $( [[ "$USE_BEAM_LOADING" == "true" ]] && echo "--beam_loading" || echo "--no-beam_loading" )

# Quick validation example (uncomment for smoke tests):
# python run_thermionic_tm010.py --preset quick --threads "$SLURM_CPUS_PER_TASK" --output "$OUT_DIR/quick"
