#!/bin/bash
#SBATCH --job-name=study_i_bis_field_grid
#SBATCH --partition=shared
#SBATCH --array=0-9%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Study I-bis: field-grid interpolation convergence, fine finesse.
#
# IMPORTANT -- create the log directory before `sbatch`ing this script: SLURM opens
# --output/--error before the job body runs, so a missing `logs/` directory makes the job fail
# immediately with no captured output at all.
#     mkdir -p logs outputs/runs
#
# IMPORTANT -- the resource requests and `%2` array concurrency limit above are PLACEHOLDERS.
# RF_GUN_REPOSITORY_UPGRADE_INSTRUCTIONS.md Phase 4 requires sizing these from one coarse pilot,
# the current 4/13um (index 1) pilot, and the one finer 3.2/10.4um (index 0) pilot BEFORE
# submitting the full 0-9 array -- do not trust these numbers, measure your own on KOA first
# (wall time and peak RSS are written to <run_dir>/resource_usage.txt by every case; aggregate
# those from the pilot cases to set real values here).
#
# Ladder: index 1 is exactly the current 4/13um grid (second finest); only index 0 is finer.
# Field-grid resolution is independent of finesse (see rf_gun/finesse_presets.py) -- this array
# only varies dr_um/dz_um; every other physics/numerical switch is held fixed at "everything ON
# except deflection" (Section 3 of the upgrade brief).

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set -- submit this script with sbatch from the repository root (SLURM sets this automatically), or export it yourself if running interactively}"
cd "$REPO_DIR"
mkdir -p logs outputs/runs

# Documented environment activation path -- override with RFTRACK_VENV_ACTIVATE if this account's
# venv lives somewhere else. No silent fallback to a hardcoded per-user path.
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
  echo "FATAL: SLURM_ARRAY_TASK_ID is not set -- submit with 'sbatch --array=0-9' (already set in this script's #SBATCH header; do not run this script directly)." >&2
  exit 1
fi
TASK_ID="$SLURM_ARRAY_TASK_ID"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 9 )); then
  echo "FATAL: SLURM_ARRAY_TASK_ID=$TASK_ID out of range [0,9] for Study I-bis." >&2
  exit 1
fi

# Geometric grid ladder (index 1 = current default 4/13um).
DR_UM_LIST=(3.200000 4.000000 5.000000 6.250000 7.812500 9.765625 12.207031 15.258789 19.073486 23.841858)
DZ_UM_LIST=(10.400000 13.000000 16.250000 20.312500 25.390625 31.738281 39.672852 49.591064 61.988830 77.486038)
DR_UM="${DR_UM_LIST[$TASK_ID]}"
DZ_UM="${DZ_UM_LIST[$TASK_ID]}"

N_PARTICLES=300000
SEED=42
T_CATHODE_K=1650
RUN_DIR="outputs/runs/study_i_bis_fine_field_grid/case_${TASK_ID}_dr${DR_UM}_dz${DZ_UM}"

echo "=== Study I-bis: field-grid convergence, fine finesse ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
echo "Hostname: $(hostname)"
echo "Case: index=$TASK_ID dr_um=$DR_UM dz_um=$DZ_UM"
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
  --dr_um "$DR_UM" --dz_um "$DZ_UM" \
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
