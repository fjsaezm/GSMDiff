#!/usr/bin/env bash
set -Eeuo pipefail

if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
  echo "This launcher requires Bash 5.1 or newer for wait -n -p." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-20260811}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lambda_cg_m_grid_seed${SEED}}"
REPORT_DIR="${REPORT_DIR:-reports/lambda_cg_m_grid_seed${SEED}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_STEPS="${NUM_STEPS:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
ONLY_SCHEDULE="${ONLY_SCHEDULE:-}"

GPUS=(0 3 4 7)
MEAN_FIELD_ITERATIONS=(1 5 10)
CG_ITERATIONS=(500 50)
ALL_SCHEDULE_NAMES=(
  lambda100
  lambda099
  lambda090
  lambda080
  power5_1_to_0
  power10_1_to_0
)

if [[ -n "$ONLY_SCHEDULE" ]]; then
  schedule_is_valid=false
  for schedule_name in "${ALL_SCHEDULE_NAMES[@]}"; do
    if [[ "$schedule_name" == "$ONLY_SCHEDULE" ]]; then
      schedule_is_valid=true
      break
    fi
  done
  if [[ "$schedule_is_valid" != true ]]; then
    echo "Unknown ONLY_SCHEDULE: $ONLY_SCHEDULE" >&2
    echo "Valid schedules: ${ALL_SCHEDULE_NAMES[*]}" >&2
    exit 2
  fi
  SCHEDULE_NAMES=("$ONLY_SCHEDULE")
else
  SCHEDULE_NAMES=("${ALL_SCHEDULE_NAMES[@]}")
fi

SCHEDULE_COUNT=${#SCHEDULE_NAMES[@]}
TOTAL_JOBS=$((SCHEDULE_COUNT * ${#CG_ITERATIONS[@]} * ${#MEAN_FIELD_ITERATIONS[@]}))

mkdir -p "$OUTPUT_DIR/launcher_logs" "$OUTPUT_DIR/hydra" "$OUTPUT_DIR/matplotlib_cache"

MPLCONFIGDIR="$OUTPUT_DIR/matplotlib_cache" \
"$PYTHON_BIN" -m gsmdiff.scripts.plot_lambda_schedules \
  --num-steps "$NUM_STEPS" \
  --output "$OUTPUT_DIR/lambda_schedules.png"

declare -A PID_TO_GPU=()
declare -A PID_TO_LABEL=()
declare -A PID_TO_LOG=()
ACTIVE_JOBS=0
NEXT_JOB=0

cleanup() {
  local status=$?
  if ((status != 0)); then
    for pid in "${!PID_TO_GPU[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

launch_job() {
  local job_index=$1
  local gpu=$2
  local schedule_index=$((job_index % SCHEDULE_COUNT))
  local block_index=$((job_index / SCHEDULE_COUNT))
  local cg_index=$((block_index % ${#CG_ITERATIONS[@]}))
  local mean_field_index=$((block_index / ${#CG_ITERATIONS[@]}))
  local cg_iterations=${CG_ITERATIONS[$cg_index]}
  local mean_field_iterations=${MEAN_FIELD_ITERATIONS[$mean_field_index]}

  local schedule_name=${SCHEDULE_NAMES[$schedule_index]}
  local run_name="celebahq_sech_${schedule_name}_m${mean_field_iterations}_cg${cg_iterations}_cold_seed${SEED}"
  local log_path="$OUTPUT_DIR/launcher_logs/${run_name}.log"
  local -a blend_args

  case "$schedule_name" in
    lambda100) blend_args=(blend=constant blend.value=1.0) ;;
    lambda099) blend_args=(blend=constant blend.value=0.99) ;;
    lambda090) blend_args=(blend=constant blend.value=0.9) ;;
    lambda080) blend_args=(blend=constant blend.value=0.8) ;;
    power5_1_to_0) blend_args=(blend=power_decay blend.power=5.0) ;;
    power10_1_to_0) blend_args=(blend=power_decay blend.power=10.0) ;;
    *) echo "Unknown schedule: $schedule_name" >&2; return 2 ;;
  esac

  local -a command=(
    "$PYTHON_BIN" -m gsmdiff.scripts.sample_geometric_celebahq
    distribution=hyperbolic_secant
    "${blend_args[@]}"
    "sampling.num_inference_steps=$NUM_STEPS"
    "sampling.batch_size=$BATCH_SIZE"
    "checkpoints.every_steps=$CHECKPOINT_EVERY"
    comparison.generate_diffusion_baseline=false
    "prior.cg.max_iterations=$cg_iterations"
    "prior.mean_field_iterations=$mean_field_iterations"
    prior.cg.warm_start=false
    "runtime.seed=$SEED"
    runtime.device=cuda
    runtime.deterministic=true
    "run.output_dir=$OUTPUT_DIR"
    "run.name=$run_name"
    "hydra.run.dir=$OUTPUT_DIR/hydra/$run_name"
  )

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu starting $run_name"
  CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="$OUTPUT_DIR/matplotlib_cache" \
    "${command[@]}" >"$log_path" 2>&1 &
  local pid=$!
  PID_TO_GPU[$pid]=$gpu
  PID_TO_LABEL[$pid]=$run_name
  PID_TO_LOG[$pid]=$log_path
  ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
}

for gpu in "${GPUS[@]}"; do
  if ((NEXT_JOB < TOTAL_JOBS)); then
    launch_job "$NEXT_JOB" "$gpu"
    NEXT_JOB=$((NEXT_JOB + 1))
  fi
done

while ((ACTIVE_JOBS > 0)); do
  finished_pid=""
  set +e
  wait -n -p finished_pid
  job_status=$?
  set -e

  if [[ -z "$finished_pid" || -z "${PID_TO_GPU[$finished_pid]+present}" ]]; then
    echo "Could not identify the completed worker process." >&2
    exit 1
  fi

  freed_gpu=${PID_TO_GPU[$finished_pid]}
  finished_label=${PID_TO_LABEL[$finished_pid]}
  finished_log=${PID_TO_LOG[$finished_pid]}
  unset 'PID_TO_GPU[$finished_pid]' 'PID_TO_LABEL[$finished_pid]' 'PID_TO_LOG[$finished_pid]'
  ACTIVE_JOBS=$((ACTIVE_JOBS - 1))

  if ((job_status != 0)); then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $finished_label (status $job_status)" >&2
    echo "Log: $finished_log" >&2
    tail -n 60 "$finished_log" >&2 || true
    exit "$job_status"
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $freed_gpu finished $finished_label"
  if ((NEXT_JOB < TOTAL_JOBS)); then
    launch_job "$NEXT_JOB" "$freed_gpu"
    NEXT_JOB=$((NEXT_JOB + 1))
  fi
done

echo "All $TOTAL_JOBS experiments completed successfully."
echo "Results: $OUTPUT_DIR"
echo "Lambda plot: $OUTPUT_DIR/lambda_schedules.png"

MPLCONFIGDIR="$OUTPUT_DIR/matplotlib_cache" \
"$PYTHON_BIN" -m gsmdiff.scripts.build_lambda_cg_report \
  --input-root "$OUTPUT_DIR" \
  --output-root "$REPORT_DIR" \
  --seed "$SEED" \
  --mean-field-iterations "${MEAN_FIELD_ITERATIONS[@]}" \
  --compile

echo "LaTeX report: $REPORT_DIR/report.tex"
echo "Compiled report: $REPORT_DIR/report.pdf"
