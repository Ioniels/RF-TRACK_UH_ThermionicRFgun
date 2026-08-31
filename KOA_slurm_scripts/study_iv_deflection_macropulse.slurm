#!/bin/bash
#SBATCH --job-name=study_iv_deflection
#SBATCH --partition=shared
#SBATCH --array=0-9%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Study IV: deflection-current scan (10 currents, 0.0-1.0 A) plus 10us back-bombardment macropulse
# heating, for every current. Fine finesse, default 4/13um field grid, uniform T=1650K, everything
# ON INCLUDING the deflection UserField and cathode backstop, 300,000 particles. The deflection
# UserField forces single-threaded RF-Track tracking for EVERY case, including 0A (the element
# stays attached at zero amplitude so 0A isolates current amplitude from code-path/threading
# changes, not from a different Volume configuration) -- this is why --cpus-per-task is only 4
# here (extra cores mostly go unused during tracking; the old 32-CPU requests in the pre-upgrade
# scripts did not help UserField cases for exactly this reason).
#
# Two stages per case, each gated behind its own completion marker so a failed stage never
# re-triggers (or silently skips) the other: (1) production transport with
# --cathode_backstop_enabled, writing back_bombardment_events.h5 (v2 schema); (2) the macropulse
# postprocessor, reading that same run directory via --source-mode load_run and writing into a
# SEPARATE subdirectory so it cannot overwrite the transport's own events file.
#
# IMPORTANT -- create the log directory before `sbatch`ing this script:
#     mkdir -p logs outputs/runs
#
# IMPORTANT -- --thermal-bin-ns below (10ns) is this brief's *initial production target*, not yet
# validated for this specific study. RF_GUN_REPOSITORY_UPGRADE_INSTRUCTIONS.md Sec. 8.5 requires a
# MANDATORY 20/10/5ns pilot convergence comparison before trusting it: rerun case 0 (and ideally
# case 9) with RFTRACK_THERMAL_BIN_NS=20 and =5 (see below), compare temperature/hotspot metrics,
# and only then decide whether 10ns is adequate, or 5ns is required, or 20ns was already
# indistinguishable (in which case document the margin and keep 10ns as the requested fine scan).
#
# IMPORTANT -- the resource requests and `%2` array concurrency limit above are PLACEHOLDERS; size
# them from real KOA pilot measurements (Phase 4), specifically the 0A and 1.0A endpoints, before
# submitting the full 0-9 array.

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
  echo "FATAL: SLURM_ARRAY_TASK_ID is not set -- submit with 'sbatch --array=0-9'." >&2
  exit 1
fi
TASK_ID="$SLURM_ARRAY_TASK_ID"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 9 )); then
  echo "FATAL: SLURM_ARRAY_TASK_ID=$TASK_ID out of range [0,9] for Study IV." >&2
  exit 1
fi

# Exact currents (A) -- 0.0 through 1.0 in 9 equal steps, given literally to avoid any
# floating-point drift from computing them with shell arithmetic.
CURRENT_A_LIST=(0.0000000000 0.1111111111 0.2222222222 0.3333333333 0.4444444444 0.5555555556 0.6666666667 0.7777777778 0.8888888889 1.0000000000)
CURRENT_A="${CURRENT_A_LIST[$TASK_ID]}"

N_PARTICLES=300000
SEED=42
T_CATHODE_K=1650
THERMAL_BIN_NS="${RFTRACK_THERMAL_BIN_NS:-10}"
MACROPULSE_DURATION_US=10.0

TRANSPORT_DIR="outputs/runs/study_iv_deflection_macropulse/case_${TASK_ID}_I${CURRENT_A}A/transport"
MACROPULSE_DIR="outputs/runs/study_iv_deflection_macropulse/case_${TASK_ID}_I${CURRENT_A}A/macropulse_bin${THERMAL_BIN_NS}ns"

echo "=== Study IV: deflection-current scan + 10us macropulse heating ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
echo "Hostname: $(hostname)"
echo "Case: index=$TASK_ID deflection_current_A=$CURRENT_A thermal_bin_ns=$THERMAL_BIN_NS"
echo "Python: $($PY --version 2>&1)"
echo "RF-Track: $($PY -c 'import RF_Track; print(getattr(RF_Track, "version", "unknown"))' 2>&1 | head -1)"
echo "numpy: $($PY -c 'import numpy; print(numpy.__version__)')"
GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
if git -C "$REPO_DIR" status --porcelain 2>/dev/null | grep -q .; then GIT_STATE=dirty; else GIT_STATE=clean; fi
echo "Git commit: $GIT_COMMIT ($GIT_STATE)"
echo "RF_TRACK_NUMBER_OF_THREADS=${RF_TRACK_NUMBER_OF_THREADS:-unset} OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Transport directory:  $TRANSPORT_DIR"
echo "Macropulse directory: $MACROPULSE_DIR"

# --- Stage 1: production transport -------------------------------------------------------------
if [[ -f "$TRANSPORT_DIR/.run_complete" ]]; then
  echo "Transport already complete at $TRANSPORT_DIR -- skipping stage 1."
else
  mkdir -p "$TRANSPORT_DIR"
  /usr/bin/time -v -o "$TRANSPORT_DIR/resource_usage.txt" "$PY" run_thermionic_tm010.py \
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
    --deflection_enabled \
    --deflection_current_A "$CURRENT_A" \
    --threads 1 \
    --output "$TRANSPORT_DIR"

  if [[ ! -f "$TRANSPORT_DIR/.run_complete" ]]; then
    echo "FATAL: transport did not produce .run_complete -- a physics/validation gate failed; see $TRANSPORT_DIR/validation.json." >&2
    exit 1
  fi
  echo "Transport complete: $TRANSPORT_DIR"
fi

# --- Stage 2: macropulse heating postprocessor (separate subdirectory) ------------------------
if [[ -f "$MACROPULSE_DIR/.macropulse_complete" ]]; then
  echo "Macropulse postprocessing already complete at $MACROPULSE_DIR -- skipping stage 2."
else
  mkdir -p "$MACROPULSE_DIR"
  /usr/bin/time -v -o "$MACROPULSE_DIR/resource_usage.txt" "$PY" run_back_bombardment_macropulse.py \
    --source-mode load_run \
    --run-dir "$TRANSPORT_DIR" \
    --thermal-backend python_xy_layered \
    --material-set LaB6_UH_recommended_v1 \
    --deposition-model BB0_TIO \
    --macropulse-duration-us "$MACROPULSE_DURATION_US" \
    --thermal-bin-ns "$THERMAL_BIN_NS" \
    --initial-temperature-uniform-k "$T_CATHODE_K" \
    --output "$MACROPULSE_DIR" \
    --save-figures

  if [[ ! -f "$MACROPULSE_DIR/study_results.json" ]]; then
    echo "FATAL: macropulse postprocessing did not produce study_results.json." >&2
    exit 1
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$MACROPULSE_DIR/.macropulse_complete"
  echo "Macropulse postprocessing complete: $MACROPULSE_DIR"
fi

echo "Case complete: index=$TASK_ID I=${CURRENT_A}A"
