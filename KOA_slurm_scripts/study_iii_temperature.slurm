#!/bin/bash
#SBATCH --job-name=study_iii_temperature
#SBATCH --partition=shared
#SBATCH --array=0-20%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Study III: uniform cathode-temperature scan, T_K = 1600 + 10*SLURM_ARRAY_TASK_ID, giving all 21
# temperatures from 1600K through 1800K inclusive. Fine finesse, default 4/13um field grid,
# everything ON except deflection (Section 3 of the upgrade brief), 300,000 particles.
#
# IMPORTANT -- create the log directory before `sbatch`ing this script:
#     mkdir -p logs outputs/runs
#
# IMPORTANT -- the resource requests and `%2` array concurrency limit above are PLACEHOLDERS; size
# them from real KOA pilot measurements (Phase 4) before submitting the full 0-20 array.

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
  echo "FATAL: SLURM_ARRAY_TASK_ID is not set -- submit with 'sbatch --array=0-20'." >&2
  exit 1
fi
TASK_ID="$SLURM_ARRAY_TASK_ID"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 20 )); then
  echo "FATAL: SLURM_ARRAY_TASK_ID=$TASK_ID out of range [0,20] for Study III." >&2
  exit 1
fi

T_CATHODE_K=$(( 1600 + 10 * TASK_ID ))

N_PARTICLES=300000
SEED=42
RUN_DIR="outputs/runs/study_iii_temperature/case_${TASK_ID}_T${T_CATHODE_K}K"

echo "=== Study III: uniform cathode-temperature scan ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
echo "Hostname: $(hostname)"
echo "Case: index=$TASK_ID T_cathode_K=$T_CATHODE_K"
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
  --finesse fine \
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
