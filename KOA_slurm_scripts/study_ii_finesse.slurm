#!/bin/bash
#SBATCH --job-name=study_ii_finesse
#SBATCH --partition=shared
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Study II: solver-finesse convergence. Field grid is HELD FIXED at the default 4/13um (finesse
# never touches field-grid resolution, see rf_gun/finesse_presets.py) -- only integration/PIC/
# beam-loading numerical-resolution settings vary across the four tiers.
#
# IMPORTANT -- create the log directory before `sbatch`ing this script (see study_i's script for
# the full explanation): `mkdir -p logs outputs/runs`.
#
# IMPORTANT -- the resource requests and `%2` array concurrency limit above are PLACEHOLDERS; size
# them from real KOA pilot measurements (Phase 4) before submitting the full array, especially
# since extra_fine (index 3) is the most expensive tier by construction.

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set -- submit this script with sbatch from the repository root}"
cd "$REPO_DIR"
mkdir -p logs outputs/runs

VENV_ACTIVATE="${RFTRACK_VENV_ACTIVATE:-$REPO_DIR/.venv/bin/activate}"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "FATAL: venv activation script not found at $VENV_ACTIVATE (set RFTRACK_VENV_ACTIVATE)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_ACTIVATE"

PY=python
if ! "$PY" -c "import RF_Track" >/dev/null 2>&1; then
  echo "FATAL: RF-Track 2.7 is not importable in this environment ($VENV_ACTIVATE)." >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "FATAL: SLURM_ARRAY_TASK_ID is not set -- submit with 'sbatch --array=0-3'." >&2
  exit 1
fi
TASK_ID="$SLURM_ARRAY_TASK_ID"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 3 )); then
  echo "FATAL: SLURM_ARRAY_TASK_ID=$TASK_ID out of range [0,3] for Study II." >&2
  exit 1
fi

FINESSE_LIST=(coarse medium fine extra_fine)
FINESSE="${FINESSE_LIST[$TASK_ID]}"

N_PARTICLES=300000
SEED=42
T_CATHODE_K=1650
RUN_DIR="outputs/runs/study_ii_finesse/case_${TASK_ID}_${FINESSE}"

echo "=== Study II: solver-finesse convergence ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
echo "Hostname: $(hostname)"
echo "Case: index=$TASK_ID finesse=$FINESSE (field grid held at CLI default 4/13um)"
echo "Python: $($PY --version 2>&1)"
echo "RF-Track: $($PY -c 'import RF_Track; print(getattr(RF_Track, "version", "unknown"))' 2>&1 | head -1)"
echo "numpy: $($PY -c 'import numpy; print(numpy.__version__)')"
GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
if git -C "$REPO_DIR" status --porcelain 2>/dev/null | grep -q .; then GIT_STATE=dirty; else GIT_STATE=clean; fi
echo "Git commit: $GIT_COMMIT ($GIT_STATE)"
echo "RF_TRACK_NUMBER_OF_THREADS=${RF_TRACK_NUMBER_OF_THREADS:-unset} OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Run directory: $RUN_DIR"

if [[ -f "$RUN_DIR/.run_complete" ]]; then
  echo "Case already complete at $RUN_DIR -- refusing to overwrite. Remove it manually to rerun."
  exit 0
fi
mkdir -p "$RUN_DIR"

/usr/bin/time -v -o "$RUN_DIR/resource_usage.txt" "$PY" run_thermionic_tm010.py \
  --n_particles "$N_PARTICLES" \
  --seed "$SEED" \
  --t_cathode_k "$T_CATHODE_K" \
  --finesse "$FINESSE" \
  --sc_enabled \
  --mirror-charges \
  --beam_loading \
  --cathode_backstop_enabled \
  --emission-field-iteration \
  --use-converged-iteration-source \
  --n_screens 3 \
  --save-screen-hdf5 \
  --save-openpmd-beam \
  --save-figures \
  --no-deflection_enabled \
  --output "$RUN_DIR"

if [[ ! -f "$RUN_DIR/.run_complete" ]]; then
  echo "FATAL: run did not produce .run_complete -- a physics/validation gate failed; see $RUN_DIR/validation.json." >&2
  exit 1
fi
echo "Case complete: $RUN_DIR"
