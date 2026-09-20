#!/bin/bash
#SBATCH --job-name=study_i_field_grid
#SBATCH --partition=shared
#SBATCH --array=0-9%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Study I: field-grid interpolation convergence, medium finesse.
#
# Each array task requires its own qualified artifact at the corresponding physical dr/dz. Set
# RFTRACK_FIELD_ARTIFACT_DIR to the directory containing the immutable convergence artifacts; the
# default is field_maps/grid_convergence below the repository root. Missing, unqualified, wrongly
# spaced, or non-modal artifacts fail before a run directory is created.
#
# IMPORTANT -- create the log directory before `sbatch`ing this script: SLURM opens
# --output/--error before the job body runs, so a missing `logs/` directory makes the job fail
# immediately with no captured output at all.
#     mkdir -p logs outputs/runs
#
# IMPORTANT -- the resource requests and `%2` array concurrency limit above are PLACEHOLDERS.
# RF_GUN_REPOSITORY_UPGRADE_INSTRUCTIONS.md Phase 4 requires sizing these from representative
# artifact-backed pilots -- do not trust these numbers, measure your own on KOA first
# (wall time and peak RSS are written to <run_dir>/resource_usage.txt by every case; aggregate
# those from the pilot cases to set real values here).
#
# The grid ladder remains independent of solver finesse. The artifact owns the actual grid; the
# verifier requires its stored nominal dr/dz to match this task and its measured hr/hz to match
# the endpoint-fitted realized spacing recorded by preprocessing.

set -euo pipefail

# --- Repository layout ---------------------------------------------------------------------
# Expected KOA install:
#   /home/nbidault/rf-track/                              <- repo root ($REPO_DIR)
#   /home/nbidault/rf-track/run_thermionic_tm010.py       <- transport driver
#   /home/nbidault/rf-track/run_back_bombardment_macropulse.py
#   /home/nbidault/rf-track/rf_gun/                       <- the Python package
#   /home/nbidault/rf-track/KOA_slurm_scripts/            <- this script
#   /home/nbidault/venvs/rftrack/bin/activate             <- the virtualenv
# Override either location with RFTRACK_REPO_DIR / RFTRACK_VENV_ACTIVATE; nothing else is
# hardcoded, so the same script runs unchanged from a clone anywhere else.
#
# The repo root CANNOT be derived from this script's own path: SLURM executes a spooled COPY of
# the batch script on the compute node (/var/spool/slurmd/.../slurm_script), so $0 and
# BASH_SOURCE point there rather than at KOA_slurm_scripts/. SLURM_SUBMIT_DIR -- wherever
# `sbatch` was invoked -- is the only submit-side location SLURM preserves, hence the candidate
# search below. It accepts submission from the repo root, from KOA_slurm_scripts/ inside it, or
# from anywhere at all (e.g. `sbatch ~/rf-track/KOA_slurm_scripts/<this>.slurm` while sitting in
# $HOME) as long as the default install path exists.
RFTRACK_REPO_DIR_DEFAULT="/home/nbidault/rf-track"
_submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -n "${RFTRACK_REPO_DIR:-}" ]]; then
  _repo_candidates=("$RFTRACK_REPO_DIR")
else
  _repo_candidates=("$_submit_dir" "$_submit_dir/.." "$RFTRACK_REPO_DIR_DEFAULT")
fi

# A candidate must carry BOTH the driver and the package -- matching on the driver alone would
# happily accept a directory holding a stray copy of the script with no rf_gun/ beside it.
REPO_DIR=""
for _cand in "${_repo_candidates[@]}"; do
  if [[ -f "$_cand/run_thermionic_tm010.py" && -f "$_cand/rf_gun/__init__.py" ]]; then
    REPO_DIR="$(cd "$_cand" && pwd)"
    break
  fi
done
if [[ -z "$REPO_DIR" ]]; then
  echo "FATAL: could not locate the rf-track repository root." >&2
  echo "       Tried: ${_repo_candidates[*]}" >&2
  echo "       A valid root contains BOTH run_thermionic_tm010.py and rf_gun/__init__.py." >&2
  echo "       Set RFTRACK_REPO_DIR=/path/to/rf-track and resubmit." >&2
  exit 1
fi
cd "$REPO_DIR"
mkdir -p logs outputs/runs
echo "Repo root: $REPO_DIR"
IDENTITY_HELPER="$REPO_DIR/verify_koa_run.py"
if [[ ! -f "$IDENTITY_HELPER" ]]; then
  echo "FATAL: completion verifier not found at $IDENTITY_HELPER." >&2
  exit 1
fi

# Every convergence point has a distinct compact artifact. Individual artifact-path overrides are
# intentionally unsupported here because they make it too easy to relabel one mesh as another.
FIELD_ARTIFACT_DIR="${RFTRACK_FIELD_ARTIFACT_DIR:-$REPO_DIR/field_maps/grid_convergence}"

# This account's real KOA venv location -- override with RFTRACK_VENV_ACTIVATE if it moves.
VENV_ACTIVATE="${RFTRACK_VENV_ACTIVATE:-/home/nbidault/venvs/rftrack/bin/activate}"
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

# Make $REPO_DIR/rf_gun importable explicitly instead of relying on it happening to be the
# process's current directory. `cd "$REPO_DIR"` above is enough for `python run_thermionic_tm010.py`
# (Python puts the SCRIPT's directory on sys.path) but not something to lean on for the
# `python -c 'import rf_gun'` probes below, whose sys.path[0] is the cwd -- and which is disabled
# outright by PYTHONSAFEPATH / `python -P`. One export removes the whole class of problem.
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PY" -c "import rf_gun" >/dev/null 2>&1; then
  echo "FATAL: the rf_gun package at $REPO_DIR/rf_gun is not importable. Its own import-time" >&2
  echo "       RF-Track version gate (>=2.7, see rf_gun/config.py) is the usual cause; rerun" >&2
  echo "       '$PY -c \"import rf_gun\"' by hand from $REPO_DIR for the actual message." >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

# Effective work function for the LaB6(100) cathode. Resolved at runtime from
# rf_gun.work_function_models.LIU_2017_PHI_EFF_EV (Liu et al. 2017, Vacuum 143, 245 -- the
# single-crystal LaB6(100) thermionic phi_eff, and this repo's own recommended baseline anchor)
# rather than duplicated here as a literal, so the script and the constant can never drift apart.
#
# This MUST be passed explicitly: run_thermionic_tm010.py's own --phi_eff_ev default is 2.1 eV,
# which every KOA batch before this one silently inherited. 2.1 eV is not anchored to anything in
# this repo, and at 1650 K it gives ~51x the emitted current density of 2.66 eV (5.44 vs
# 0.106 MA/m^2 from rf_gun's own RD_schottky at 30 MV/m) -- i.e. ~420 A/cm^2 off a 1.4 mm-radius
# LaB6 cathode, roughly 40x what the material actually delivers at that temperature.
#
# --work-function-temperature-model linear_tcwf (Model 1) is passed alongside it: Liu 2017's
# anchor carried across temperature by Bulyga & Solonovich's TCWF slope
# (alpha = 1.8e-4 eV/K about T_ref = 1773 K). It OVERRIDES --phi_eff_ev at run time and takes its
# own phi_ref_eV default from the very same LIU_2017_PHI_EFF_EV constant, so the two agree by
# construction and --phi_eff_ev remains as the recorded zero-slope anchor in run_config.json.
# Effect over the LaB6 operating window (phi_eff, and emitted current vs. the flat 2.66 eV):
#   1500K -> 2.611 eV (x1.46)   1600K -> 2.629 eV (x1.25)
#   1650K -> 2.638 eV (x1.17)   1700K -> 2.647 eV (x1.09)
# Every script in this directory uses the SAME treatment, so runs stay comparable across studies;
# change it in one place only if you change it in all of them. Set
# piecewise_surface_evolution instead for the surface-evolution sensitivity model.
# Importing rf_gun pulls in RF-Track, which prints its copyright banner to STDOUT. Capturing the
# value from stdout is therefore a trap, and every "obvious" workaround is worse:
#   * a bare command substitution returns the number with the banner glued onto it;
#   * `| head -n 1` makes head exit early, the interpreter takes SIGPIPE writing the banner into
#     the closed pipe, and under `set -o pipefail` that kills this script AT THE ASSIGNMENT --
#     before the validation below can run, with no diagnostic at all;
#   * "just take the first line" is not even correct: with PYTHONUNBUFFERED=1 (exported above)
#     the banner is emitted BEFORE our own print, so the first line is the banner's blank line.
# So don't use stdout at all. Python writes the number to a file; stdout is discarded wholesale.
# Order-independent, buffering-independent, banner-proof.
_phi_tmp="$(mktemp "${TMPDIR:-/tmp}/rfgun_phi_eff.XXXXXX")"
"$PY" -c 'import sys, rf_gun; open(sys.argv[1], "w").write(f"{rf_gun.LIU_2017_PHI_EFF_EV:.6g}")' \
  "$_phi_tmp" >/dev/null 2>&1 || true
PHI_EFF_EV="$(cat "$_phi_tmp" 2>/dev/null || true)"
rm -f "$_phi_tmp"
PHI_EFF_EV="${PHI_EFF_EV//[[:space:]]/}"
if ! [[ "$PHI_EFF_EV" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "FATAL: could not resolve rf_gun.LIU_2017_PHI_EFF_EV from the active venv (got '$PHI_EFF_EV')." >&2
  exit 1
fi
echo "Work function: phi_eff = $PHI_EFF_EV eV (rf_gun.LIU_2017_PHI_EFF_EV, Liu et al. 2017)"

# GNU time (needed for -v/-o) isn't guaranteed to exist on every KOA compute node image --
# degrade gracefully (run the command directly, no resource_usage.txt) instead of failing the
# whole job over a missing diagnostics wrapper. 'seff $SLURM_JOB_ID' after completion reports
# peak memory/wall time from SLURM's own accounting regardless of whether this succeeds.
resolve_time_cmd() {
  local usage_file="$1"
  if command -v /usr/bin/time >/dev/null 2>&1; then
    TIME_CMD=(/usr/bin/time -v -o "$usage_file")
  else
    echo "Note: /usr/bin/time not found on this node -- running without it, no resource_usage.txt for this run (use 'seff \$SLURM_JOB_ID' after completion instead)."
    TIME_CMD=()
  fi
}

# Identity probes are success criteria, not best-effort logging.  In particular, a failed
# `git status` must never be mislabelled as a clean checkout, and an unreadable RF-Track version
# must never enter a completion marker as "unknown".
capture_git_identity() {
  local label="$1"
  local commit_variable="$2"
  local state_variable="$3"
  local sampled_commit sampled_status sampled_state
  if ! sampled_commit="$(git -C "$REPO_DIR" rev-parse --verify 'HEAD^{commit}')"; then
    echo "FATAL: Git commit probe failed while sampling $label." >&2
    return 1
  fi
  if ! [[ "$sampled_commit" =~ ^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$ ]]; then
    echo "FATAL: Git commit probe returned an invalid object ID while sampling $label: '$sampled_commit'." >&2
    return 1
  fi
  if ! sampled_status="$(git -C "$REPO_DIR" status --porcelain --untracked-files=normal)"; then
    echo "FATAL: Git worktree-status probe failed while sampling $label." >&2
    return 1
  fi
  if [[ -n "$sampled_status" ]]; then sampled_state=dirty; else sampled_state=clean; fi
  printf -v "$commit_variable" '%s' "$sampled_commit"
  printf -v "$state_variable" '%s' "$sampled_state"
}

capture_rftrack_version() {
  local label="$1"
  local output_variable="$2"
  local version_file sampled_version normalized_version
  if ! version_file="$(mktemp "${TMPDIR:-/tmp}/rfgun_rftver.XXXXXX")"; then
    echo "FATAL: could not create a temporary file for the RF-Track version probe ($label)." >&2
    return 1
  fi
  if ! "$PY" -c 'import sys, RF_Track; open(sys.argv[1], "w").write(str(getattr(RF_Track, "version", "")).strip())' \
      "$version_file" >/dev/null; then
    rm -f -- "$version_file"
    echo "FATAL: RF-Track version probe failed while sampling $label." >&2
    return 1
  fi
  if [[ ! -s "$version_file" ]]; then
    rm -f -- "$version_file"
    echo "FATAL: RF-Track version probe produced no value while sampling $label." >&2
    return 1
  fi
  sampled_version="$(<"$version_file")"
  rm -f -- "$version_file"
  normalized_version="${sampled_version,,}"
  case "$normalized_version" in
    ""|unknown|unavailable|unset|none)
      echo "FATAL: RF-Track version probe returned an unusable value while sampling $label: '$sampled_version'." >&2
      return 1
      ;;
  esac
  printf -v "$output_variable" '%s' "$sampled_version"
}

quarantine_transport_marker() {
  local marker="$1"
  local reason="$2"
  local quarantined="${marker}.invalidated_${reason}.${SLURM_JOB_ID:-nojob}.${SLURM_ARRAY_TASK_ID:-notask}.$$"
  if [[ ! -e "$marker" && ! -L "$marker" ]]; then
    echo "Note: no transport completion marker was present to quarantine at $marker." >&2
    return 0
  fi
  if [[ -e "$quarantined" || -L "$quarantined" ]]; then
    echo "FATAL: refusing to overwrite an existing quarantined marker: $quarantined" >&2
    return 1
  fi
  if ! mv -- "$marker" "$quarantined"; then
    echo "FATAL: could not quarantine transport completion marker $marker." >&2
    return 1
  fi
  echo "Transport completion marker quarantined as $quarantined" >&2
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "FATAL: SLURM_ARRAY_TASK_ID is not set -- submit with 'sbatch --array=0-9' (already set in this script's #SBATCH header; do not run this script directly)." >&2
  exit 1
fi
TASK_ID="$SLURM_ARRAY_TASK_ID"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 9 )); then
  echo "FATAL: SLURM_ARRAY_TASK_ID=$TASK_ID out of range [0,9] for Study I." >&2
  exit 1
fi

# Geometric grid ladder. Each value pair must have a separately prepared/qualified artifact.
DR_UM_LIST=(3.200000 4.000000 5.000000 6.250000 7.812500 9.765625 12.207031 15.258789 19.073486 23.841858)
DZ_UM_LIST=(10.400000 13.000000 16.250000 20.312500 25.390625 31.738281 39.672852 49.591064 61.988830 77.486038)
DR_UM="${DR_UM_LIST[$TASK_ID]}"
DZ_UM="${DZ_UM_LIST[$TASK_ID]}"

FIELD_ARTIFACT="$FIELD_ARTIFACT_DIR/cavity_axisymmetric_m0_dr${DR_UM}um_dz${DZ_UM}um_rftrack.h5"
if [[ ! -f "$FIELD_ARTIFACT" ]]; then
  echo "FATAL: task-specific qualified RF field artifact not found at $FIELD_ARTIFACT." >&2
  echo "       Prepare this exact dr=$DR_UM um, dz=$DZ_UM um artifact, or set" >&2
  echo "       RFTRACK_FIELD_ARTIFACT_DIR=/path/to/grid_convergence." >&2
  exit 1
fi
verify_exact_grid_artifact() {
  "$PY" "$IDENTITY_HELPER" verify-artifact \
    --artifact "$FIELD_ARTIFACT" \
    --tail-policy modal \
    --expected-dr-um "$DR_UM" \
    --expected-dz-um "$DZ_UM"
}
verify_exact_grid_artifact
echo "Field artifact: $FIELD_ARTIFACT"

N_PARTICLES=300000
SEED=42
T_CATHODE_K=1650
RUN_DIR="outputs/runs/study_i_medium_field_grid/case_${TASK_ID}_dr${DR_UM}_dz${DZ_UM}"

echo "=== Study I: field-grid convergence, medium finesse ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
echo "Hostname: $(hostname)"
echo "Case: index=$TASK_ID dr_um=$DR_UM dz_um=$DZ_UM"
echo "Python: $($PY --version 2>&1)"
capture_rftrack_version "before transport" RFTRACK_VERSION
echo "RF-Track: $RFTRACK_VERSION"
echo "numpy: $($PY -c 'import numpy; print(numpy.__version__)')"
capture_git_identity "before transport" GIT_COMMIT GIT_STATE
echo "Git commit: $GIT_COMMIT ($GIT_STATE)"
CODE_SHA256="$("$PY" "$IDENTITY_HELPER" code-identity --repo "$REPO_DIR" --launcher "$0")"
PYTHON_STACK_SHA256="$("$PY" "$IDENTITY_HELPER" python-stack-identity)"
echo "Code identity SHA-256: $CODE_SHA256"
echo "Python numerical-stack identity: $PYTHON_STACK_SHA256"
echo "RF_TRACK_NUMBER_OF_THREADS=${RF_TRACK_NUMBER_OF_THREADS:-unset} OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Run directory: $RUN_DIR"

RF_FORWARD_POWER_W="${RFTRACK_RF_FORWARD_POWER_W:-1000000}"
# E+B production vs an explicit E-only control, and the modal tail vs its zero-tail
# control, must both be selectable without editing this script: the A/B pilots that
# justify the production choices are otherwise unreachable from automation.
RF_MAGNETIC_FIELD="${RFTRACK_RF_MAGNETIC_FIELD:-artifact}"
FIELD_TAIL_POLICY="${RFTRACK_FIELD_TAIL_POLICY:-modal}"
case "$RF_MAGNETIC_FIELD" in artifact|zero) ;; *)
  echo "FATAL: RFTRACK_RF_MAGNETIC_FIELD must be artifact or zero (got $RF_MAGNETIC_FIELD)" >&2; exit 1 ;;
esac
case "$FIELD_TAIL_POLICY" in modal|zero-control) ;; *)
  echo "FATAL: RFTRACK_FIELD_TAIL_POLICY must be modal or zero-control (got $FIELD_TAIL_POLICY)" >&2; exit 1 ;;
esac
RUN_ARGS=(
  --field-artifact "$FIELD_ARTIFACT"
  --field-tail-policy "$FIELD_TAIL_POLICY"
  --rf-magnetic-field "$RF_MAGNETIC_FIELD"
  --rf-forward-power-w "${RFTRACK_RF_FORWARD_POWER_W:-1000000}"
  --run-family study_i_medium_field_grid
  --scan-tags
  "campaign=axisymmetric_eb_v1"
  "study=i"
  "stage=transport"
  "task_id=$TASK_ID"
  "dr_um=$DR_UM"
  "dz_um=$DZ_UM"
  "finesse=medium"
  "rf_forward_power_w=$RF_FORWARD_POWER_W"
  "git_commit=$GIT_COMMIT"
  "git_state=$GIT_STATE"
  "code_sha256=$CODE_SHA256"
  "python_stack_sha256=$PYTHON_STACK_SHA256"
  "rftrack_version=$RFTRACK_VERSION"
  "slurm_job_id=${SLURM_JOB_ID:-unset}"
  "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
  "slurm_array_task_id=$TASK_ID"
  --n_particles "$N_PARTICLES"
  --seed "$SEED"
  --t_cathode_k "$T_CATHODE_K"
  --phi_eff_ev "$PHI_EFF_EV"
  --work-function-temperature-model linear_tcwf
  --finesse medium
  --sc_enabled
  --mirror-charges
  --beam_loading
  --cathode_backstop_enabled
  --emission-field-iteration
  --use-converged-iteration-source
  --n_screens 3
  --save-screen-hdf5
  --save-openpmd-beam
  --save-figures
  --no-deflection_enabled
  --output "$RUN_DIR"
)

if [[ -e "$RUN_DIR" || -L "$RUN_DIR" ]]; then
  if [[ ! -f "$RUN_DIR/.run_complete" ]]; then
    echo "FATAL: run directory already exists without a completion marker: $RUN_DIR" >&2
    echo "       Refusing to overwrite or mix results; inspect and remove/move it manually." >&2
    exit 1
  fi
  verify_exact_grid_artifact
  if "$PY" "$IDENTITY_HELPER" verify-transport \
      --run-dir "$RUN_DIR" --artifact "$FIELD_ARTIFACT" -- "${RUN_ARGS[@]}"; then
    echo "Case already complete and exactly matches this request: $RUN_DIR"
    exit 0
  fi
  echo "FATAL: existing completion marker is invalid, stale, or belongs to another request:" >&2
  echo "       $RUN_DIR/.run_complete" >&2
  echo "       Refusing to overwrite; inspect and remove/move the directory manually." >&2
  exit 1
fi
mkdir -p "$(dirname "$RUN_DIR")"
if ! mkdir "$RUN_DIR"; then
  echo "FATAL: could not claim run directory atomically (duplicate/concurrent submission?): $RUN_DIR" >&2
  exit 1
fi

resolve_time_cmd "$RUN_DIR/resource_usage.txt"
"${TIME_CMD[@]}" "$PY" run_thermionic_tm010.py "${RUN_ARGS[@]}"

_identity_probe_failed=0
GIT_COMMIT_AFTER=""
GIT_STATE_AFTER=""
RFTRACK_VERSION_AFTER=""
CODE_SHA256_AFTER=""
PYTHON_STACK_SHA256_AFTER=""
capture_git_identity "after transport" GIT_COMMIT_AFTER GIT_STATE_AFTER || _identity_probe_failed=1
capture_rftrack_version "after transport" RFTRACK_VERSION_AFTER || _identity_probe_failed=1
if ! CODE_SHA256_AFTER="$("$PY" "$IDENTITY_HELPER" code-identity --repo "$REPO_DIR" --launcher "$0")"; then
  echo "FATAL: simulation-code identity probe failed after transport." >&2
  _identity_probe_failed=1
fi
if ! PYTHON_STACK_SHA256_AFTER="$("$PY" "$IDENTITY_HELPER" python-stack-identity)"; then
  echo "FATAL: Python numerical-stack identity probe failed after transport." >&2
  _identity_probe_failed=1
fi
if (( _identity_probe_failed != 0 )) \
    || [[ "$GIT_COMMIT_AFTER" != "$GIT_COMMIT" || "$GIT_STATE_AFTER" != "$GIT_STATE" \
          || "$RFTRACK_VERSION_AFTER" != "$RFTRACK_VERSION" \
          || "$CODE_SHA256_AFTER" != "$CODE_SHA256" \
          || "$PYTHON_STACK_SHA256_AFTER" != "$PYTHON_STACK_SHA256" ]]; then
  quarantine_transport_marker "$RUN_DIR/.run_complete" identity_change
  echo "FATAL: checkout or numerical environment changed during transport; completion marker invalidated." >&2
  exit 1
fi

if ! verify_exact_grid_artifact; then
  quarantine_transport_marker "$RUN_DIR/.run_complete" field_artifact_verification_failure
  echo "FATAL: task-specific field artifact failed its post-run grid qualification check." >&2
  exit 1
fi
if ! "$PY" "$IDENTITY_HELPER" verify-transport \
    --run-dir "$RUN_DIR" --artifact "$FIELD_ARTIFACT" -- "${RUN_ARGS[@]}"; then
  quarantine_transport_marker "$RUN_DIR/.run_complete" verification_failure
  echo "FATAL: run did not produce a valid identity-matched completion record." >&2
  exit 1
fi
echo "Case complete: $RUN_DIR"
