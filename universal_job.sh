#!/bin/bash
# Generic Slurm job wrapper for Slurminator.
#
# Responsibilities:
#   1. Parse the orchestrator-provided job arguments.
#   2. Source the user's env script.
#   3. Detect allocated GPUs and launch the provided command.
#
# The built-in multi-GPU wrapper assumes a Python entrypoint that can be
# launched with torchrun. Non-torch users should provide their own command
# wrapper or call a project-specific launcher from their env script.

# Generic fallback directives. Production values are supplied by the
# orchestrator through sbatch CLI flags and override these defaults.
#SBATCH --job-name="slurminator-job"
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1

set -euo pipefail

# srun steps must receive the environment constructed for this child job.
export SLURM_EXPORT_ENV=ALL
usage() {
  cat <<'EOF'
Usage: sbatch universal_job.sh [options]

Options:
  --env_script <path> : Environment setup script to source before running.
  --repo_path <path>  : Directory to cd into before running commands.
  --commands <string> : Quoted shell command(s) to run.
  --exp_id <id>       : Experiment ID used in log messages.
  --outdir <path>     : Output directory used by the caller.
  -h, --help          : Show this help.
EOF
}

log() {
  echo "[universal_job.sh] $*"
}

env_script=""
repo_path=""
commands=""
exp_id=""
outdir=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --env_script)
      env_script="${2:-}"
      shift 2
      ;;
    --repo_path)
      repo_path="${2:-}"
      shift 2
      ;;
    --commands=*)
      commands="${key#*=}"
      shift
      ;;
    --commands)
      commands="${2:-}"
      shift 2
      ;;
    --exp_id)
      exp_id="${2:-}"
      shift 2
      ;;
    --outdir)
      outdir="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $key" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "Slurm job info from environment:"
echo "  SLURM_JOB_NAME      = ${SLURM_JOB_NAME:-}"
echo "  SLURM_JOBID         = ${SLURM_JOBID:-${SLURM_JOB_ID:-}}"
echo "  SLURM_JOB_ACCOUNT   = ${SLURM_JOB_ACCOUNT:-}"
echo "  SLURM_JOB_PARTITION = ${SLURM_JOB_PARTITION:-}"
echo "  SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK:-}"
echo "  SLURM_MEM_PER_NODE  = ${SLURM_MEM_PER_NODE:-}"
echo "  EXP_ID              = ${exp_id:-}"
echo "  OUTDIR              = ${outdir:-}"
echo "-------------------------------------------------"

# The status callback must use the ledger row ID, not a shared display name.
# Override any ID inherited from the controller and preserve it through both
# supported container runtimes, including their clean-environment mode.
if [[ -n "$exp_id" ]]; then
  export ORCHESTRATOR_EXPERIMENT_ID="$exp_id"
  export APPTAINERENV_ORCHESTRATOR_EXPERIMENT_ID="$exp_id"
  export SINGULARITYENV_ORCHESTRATOR_EXPERIMENT_ID="$exp_id"
fi

if [[ -n "$env_script" ]]; then
  if [[ ! -f "$env_script" ]]; then
    echo "ERROR: environment script '$env_script' not found." >&2
    exit 1
  fi
  log "Sourcing environment script: $env_script"
  set +e +u
  # The env script is project-owned and may use unset variables defensively.
  # The wrapper's argument list is empty after option parsing.
  source "$env_script"
  source_rc=$?
  set -euo pipefail
  if [[ $source_rc -ne 0 ]]; then
    echo "ERROR: environment script '$env_script' returned $source_rc." >&2
    exit "$source_rc"
  fi
fi

if [[ -n "$repo_path" ]]; then
  if [[ ! -d "$repo_path" ]]; then
    echo "ERROR: repo path '$repo_path' not found." >&2
    exit 1
  fi
  log "cd $repo_path"
  cd "$repo_path"
fi

log "Working directory: $(pwd)"
if [[ -n "$commands" ]]; then
  preview="${commands:0:200}"
  suffix=""
  if (( ${#commands} > 200 )); then
    suffix="..."
  fi
  log "Command preview: ${preview}${suffix}"
else
  log "No --commands provided; exiting."
  exit 0
fi

cleanup_ipc_artifacts() {
  local phase="${1:-preflight}"
  local job_id="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"
  local owner="${USER:-$(whoami)}"

  if [[ -z "$job_id" ]]; then
    log "[$phase] SLURM_JOB_ID unset; skipping job-scoped IPC cleanup."
    return 0
  fi

  log "[$phase] Cleaning job-scoped IPC artifacts for job_id=$job_id"
  rm -rf /dev/shm/torch_"${job_id}"_* /dev/shm/torch_"${job_id}"-* 2>/dev/null || true
  rm -rf /tmp/torch_"${job_id}"_* /tmp/torch_"${job_id}"-* 2>/dev/null || true

  if [[ "${SLURMINATOR_AGGRESSIVE_SHM_CLEANUP:-0}" == "1" ]]; then
    log "[$phase] SLURMINATOR_AGGRESSIVE_SHM_CLEANUP=1; pruning stale user-owned IPC files."
    find /dev/shm -maxdepth 1 -name "torch_*" -user "$owner" -mmin +120 -delete 2>/dev/null || true
    find /dev/shm -maxdepth 1 -name "loky-*" -user "$owner" -mmin +120 -delete 2>/dev/null || true
    find /tmp -maxdepth 1 -name "torch_*" -user "$owner" -mmin +120 -delete 2>/dev/null || true
  fi
}

cleanup_on_exit() {
  local rc=$?
  trap - EXIT TERM INT
  log "Cleanup starting with exit code $rc"
  wait -n 2>/dev/null || true
  jobs -p | xargs -r kill -9 2>/dev/null || true
  cleanup_ipc_artifacts "exit"
  exit "$rc"
}

trap cleanup_on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

cleanup_ipc_artifacts "preflight"

count_csv_items() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo 0
  else
    local without_commas="${value//,/}"
    echo $(( ${#value} - ${#without_commas} + 1 ))
  fi
}

count_gpus_from_id_list() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo 0
  elif [[ "$value" == *,* ]]; then
    count_csv_items "$value"
  elif [[ "$value" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    local start="${BASH_REMATCH[1]}"
    local end="${BASH_REMATCH[2]}"
    if (( end >= start )); then
      echo $(( end - start + 1 ))
    else
      echo 0
    fi
  else
    # SLURM_JOB_GPUS/SLURM_STEP_GPUS are GPU id lists on Olivia. A plain
    # integer like "2" means one allocated physical GPU, not two GPUs.
    echo 1
  fi
}

count_gpus_from_count_value() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo 0
  elif [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value"
  elif [[ "$value" =~ ^([0-9]+)\(.*\)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$value" =~ :([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    count_csv_items "$value"
  fi
}

detect_gpu_count() {
  if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    count_gpus_from_id_list "$SLURM_STEP_GPUS"
  elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    count_gpus_from_id_list "$SLURM_JOB_GPUS"
  elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    count_gpus_from_count_value "$SLURM_GPUS_ON_NODE"
  elif [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
    count_gpus_from_count_value "$SLURM_GPUS_PER_NODE"
  elif [[ -n "${SLURM_GPUS:-}" ]]; then
    count_gpus_from_count_value "$SLURM_GPUS"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l
  elif command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showid 2>/dev/null | grep -Ec 'GPU|card|^ *[0-9]+:' || true
  else
    echo 0
  fi
}

gpu_probe_ok() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L >/dev/null 2>&1
  elif command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showid >/dev/null 2>&1
  else
    return 1
  fi
}

normalize_gpu_visibility() {
  local gpu_mask="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
  if [[ -n "$gpu_mask" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_mask"
    export ROCR_VISIBLE_DEVICES="$gpu_mask"
  fi

  local is_rocm=0
  command -v rocm-smi >/dev/null 2>&1 && is_rocm=1
  command -v rocminfo >/dev/null 2>&1 && is_rocm=1

  if [[ $is_rocm -eq 1 ]]; then
    local mask="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
    if [[ -n "$mask" ]]; then
      if [[ "$mask" == *,* || "$mask" =~ ^[0-9]+$ ]]; then
        local n_gpus="$TOTAL_GPUS"
        if [[ "$n_gpus" =~ ^[0-9]+$ && "$n_gpus" -gt 0 ]]; then
          mask="$(seq -s, 0 $((n_gpus - 1)))"
        fi
      fi
      export ROCR_VISIBLE_DEVICES="$mask"
      export HIP_VISIBLE_DEVICES="$mask"
      export APPTAINERENV_ROCR_VISIBLE_DEVICES="$mask"
      export APPTAINERENV_HIP_VISIBLE_DEVICES="$mask"
      export SINGULARITYENV_ROCR_VISIBLE_DEVICES="$mask"
      export SINGULARITYENV_HIP_VISIBLE_DEVICES="$mask"
      unset CUDA_VISIBLE_DEVICES || true
    fi
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local original_cuda_mask="$CUDA_VISIBLE_DEVICES"
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
    export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
    export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"

    local accessible_cuda_mask=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      accessible_cuda_mask="$(
        nvidia-smi -L 2>/dev/null \
          | sed -n 's/^GPU \([0-9]\+\):.*/\1/p' \
          | tr '\n' ',' \
          | sed 's/,$//'
      )"
    fi
    if [[ -n "$accessible_cuda_mask" && "$accessible_cuda_mask" != "$original_cuda_mask" ]]; then
      export CUDA_VISIBLE_DEVICES="$accessible_cuda_mask"
      export APPTAINERENV_CUDA_VISIBLE_DEVICES="$accessible_cuda_mask"
      export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$accessible_cuda_mask"
      log "Normalized CUDA_VISIBLE_DEVICES to accessible set: $accessible_cuda_mask"
    fi
  fi
}

validate_torch_runtime_gpu_visibility() {
  # Host-side nvidia-smi can see devices that are missing from the training
  # container. Probe through the Python selected by the environment script before torchrun so a
  # broken device bind fails once, clearly, before ranks or W&B are started.
  if (( LAUNCH_GPUS < 1 )) || [[ "$commands" != *torchrun* ]]; then
    return 0
  fi
  if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: torchrun command requested, but no Python runtime is available for the GPU visibility preflight." >&2
    return 1
  fi

  local torch_visible_gpus=""
  if ! torch_visible_gpus="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)"; then
    echo "ERROR: the training Python runtime could not query torch.cuda.device_count() before launch." >&2
    return 1
  fi
  if ! [[ "$torch_visible_gpus" =~ ^[0-9]+$ ]]; then
    echo "ERROR: torch.cuda.device_count() returned an invalid value: '$torch_visible_gpus'." >&2
    return 1
  fi

  log "Torch-visible GPUs in training runtime: $torch_visible_gpus"
  if (( torch_visible_gpus < LAUNCH_GPUS )); then
    echo "ERROR: launch requests $LAUNCH_GPUS rank(s), but the training runtime exposes only $torch_visible_gpus GPU(s)." >&2
    echo "ERROR: refusing to start torchrun; check the node/container GPU bind or exclude the affected node." >&2
    return 1
  fi
}

set_thread_defaults_for_multigpu() {
  if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    local cpus="${SLURM_CPUS_PER_TASK:-1}"
    if ! [[ "$cpus" =~ ^[0-9]+$ ]] || (( cpus < 1 )); then
      cpus=1
    fi
    local threads=$(( cpus / LAUNCH_GPUS ))
    if (( threads < 1 )); then
      threads=1
    fi
    export OMP_NUM_THREADS="$threads"
  fi
  export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
  log "Multi-GPU threading: OMP_NUM_THREADS=$OMP_NUM_THREADS, MKL_NUM_THREADS=$MKL_NUM_THREADS, OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
}

run_direct() {
  bash -c "$commands"
}

run_torchrun_if_python() {
  local commands_clean="${commands//\\/}"
  local cmd_prefix=""
  local cmd_tail="$commands_clean"

  if [[ "$commands_clean" =~ ^(.*&&[[:space:]]*)(python3?|python)[[:space:]] ]]; then
    cmd_prefix="${BASH_REMATCH[1]}"
    cmd_tail="${commands_clean:${#cmd_prefix}}"
  fi

  local script=""
  local script_args=""
  if [[ "$cmd_tail" =~ ^python(3)?[[:space:]]+-u[[:space:]]+([^[:space:]]+)(.*)$ ]]; then
    script="${BASH_REMATCH[2]}"
    script_args="${BASH_REMATCH[3]}"
  elif [[ "$cmd_tail" =~ ^python(3)?[[:space:]]+([^[:space:]]+)(.*)$ ]]; then
    script="${BASH_REMATCH[2]}"
    script_args="${BASH_REMATCH[3]}"
  else
    return 2
  fi
  if [[ "$script" == "-m" ]]; then
    return 2
  fi

  local nproc="$LAUNCH_GPUS"
  if ! gpu_probe_ok; then
    log "GPU probe failed; using nproc_per_node=1 for torchrun."
    nproc=1
  fi

  log "Multi-GPU Python command -> srun + torchrun (nproc_per_node=$nproc)"
  srun --kill-on-bad-exit=1 \
       --ntasks=1 \
       --cpu-bind=cores \
       bash -c "${cmd_prefix}torchrun --standalone --nnodes=1 --nproc_per_node=${nproc} ${script}${script_args}"
}

TOTAL_GPUS="$(detect_gpu_count)"
if ! [[ "$TOTAL_GPUS" =~ ^[0-9]+$ ]]; then
  TOTAL_GPUS=0
fi
log "Detected GPUs on node: $TOTAL_GPUS"
normalize_gpu_visibility
LAUNCH_GPUS="$TOTAL_GPUS"
if [[ -n "${SLURMINATOR_NPROC_PER_NODE:-}" ]]; then
  if [[ "$SLURMINATOR_NPROC_PER_NODE" =~ ^[0-9]+$ ]]; then
    LAUNCH_GPUS="$SLURMINATOR_NPROC_PER_NODE"
  else
    echo "ERROR: SLURMINATOR_NPROC_PER_NODE must be an integer, got '$SLURMINATOR_NPROC_PER_NODE'." >&2
    exit 1
  fi
fi
log "Launch process count: $LAUNCH_GPUS"
validate_torch_runtime_gpu_visibility

set +e
if (( LAUNCH_GPUS > 1 )); then
  set_thread_defaults_for_multigpu
  run_torchrun_if_python
  run_rc=$?
  if [[ $run_rc -eq 2 ]]; then
    log "Multi-GPU non-Python command -> running directly."
    run_direct
    run_rc=$?
  fi
else
  log "Single-GPU/CPU command -> running directly."
  run_direct
  run_rc=$?
fi
set -e

echo "-------------------------------------------------"
log "Command finished with exit code $run_rc."
exit "$run_rc"
