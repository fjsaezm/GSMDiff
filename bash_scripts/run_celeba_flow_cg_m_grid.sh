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
SEED_COUNT="${SEED_COUNT:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/flow_cg_m_grid_seed${SEED}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-8}"
NUM_STEPS="${NUM_STEPS:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-}"
PRIOR_SCALE="${PRIOR_SCALE:-auto}"
ONLY_FIELD="${ONLY_FIELD:-}"
CONTOUR_QUANTILE="${CONTOUR_QUANTILE:-0.98}"
CONTOUR_THRESHOLD="${CONTOUR_THRESHOLD:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-}"
MODEL_DOWNLOAD="${MODEL_DOWNLOAD:-}"
MODEL_SHA256="${MODEL_SHA256:-}"

read -r -a GPUS <<<"${GPU_IDS:-0 3 4 7}"
read -r -a MEAN_FIELD_ITERATIONS <<<"${MEAN_FIELD_VALUES:-1 5 10}"
read -r -a CG_ITERATIONS <<<"${CG_VALUES:-500 50}"
if ((${#GPUS[@]} == 0 || ${#MEAN_FIELD_ITERATIONS[@]} == 0 || ${#CG_ITERATIONS[@]} == 0)); then
  echo "GPU_IDS, MEAN_FIELD_VALUES, and CG_VALUES must not be empty." >&2
  exit 2
fi
if ((SEED_COUNT <= 0)); then
  echo "SEED_COUNT must be positive." >&2
  exit 2
fi
case "$SKIP_COMPLETED" in
  true|false) ;;
  *) echo "SKIP_COMPLETED must be true or false." >&2; exit 2 ;;
esac

# q and p name the independent coefficients of the learned and GGSM fields.
# The q1_* cases intentionally do not constrain q+p=1.
ALL_FIELD_NAMES=(
  q1_p0
  q1_p0p005
  q1_p0p01
  q1_p0p05
  q0p95_p0p05
  q1_p_linear0_to_0p1
)

if [[ -n "$ONLY_FIELD" ]]; then
  field_is_valid=false
  for field_name in "${ALL_FIELD_NAMES[@]}"; do
    if [[ "$field_name" == "$ONLY_FIELD" ]]; then
      field_is_valid=true
      break
    fi
  done
  if [[ "$field_is_valid" != true ]]; then
    echo "Unknown ONLY_FIELD: $ONLY_FIELD" >&2
    echo "Valid fields: ${ALL_FIELD_NAMES[*]}" >&2
    exit 2
  fi
  FIELD_NAMES=("$ONLY_FIELD")
else
  FIELD_NAMES=("${ALL_FIELD_NAMES[@]}")
fi

SEEDS=()
for ((seed_offset = 0; seed_offset < SEED_COUNT; seed_offset++)); do
  SEEDS+=("$((SEED + seed_offset))")
done

mkdir -p "$OUTPUT_DIR/launcher_logs" "$OUTPUT_DIR/hydra" "$OUTPUT_DIR/matplotlib_cache"

MODEL_ARGS=()
if [[ -n "$MODEL_CHECKPOINT" ]]; then
  MODEL_ARGS+=("model.checkpoint=$MODEL_CHECKPOINT")
fi
if [[ -n "$MODEL_DOWNLOAD" ]]; then
  MODEL_ARGS+=("model.download=$MODEL_DOWNLOAD")
fi
if [[ -n "$MODEL_SHA256" ]]; then
  MODEL_ARGS+=("model.sha256=$MODEL_SHA256")
fi

# Match the GGSM coefficient scale to clean samples from the same pretrained flow.
# This also downloads and verifies the public checkpoint once before parallel workers start.
if [[ "$PRIOR_SCALE" == auto ]]; then
  if [[ -z "$CALIBRATION_SAMPLES" ]]; then
    calibration_name="celeba_flow_calibration_seed${SEED}"
    CALIBRATION_SAMPLES="$OUTPUT_DIR/$calibration_name/geometric_samples.pt"
    if [[ ! -f "$CALIBRATION_SAMPLES" ]]; then
      calibration_log="$OUTPUT_DIR/launcher_logs/$calibration_name.log"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPUS[0]} starting flow-only calibration"
      CUDA_VISIBLE_DEVICES="${GPUS[0]}" MPLCONFIGDIR="$OUTPUT_DIR/matplotlib_cache" \
        "$PYTHON_BIN" -m gsmdiff.scripts.sample_geometric_flow_celeba \
        "${MODEL_ARGS[@]}" \
        distribution=hyperbolic_secant distribution.scale=1.0 \
        flow_weight=constant flow_weight.value=1.0 \
        prior_weight=constant prior_weight.value=0.0 \
        "sampling.num_inference_steps=$NUM_STEPS" \
        "sampling.batch_size=$CALIBRATION_BATCH_SIZE" \
        checkpoints.every_steps=null comparison.generate_flow_baseline=false \
        "runtime.seed=$SEED" "runtime.device=$RUNTIME_DEVICE" runtime.deterministic=true \
        "run.output_dir=$OUTPUT_DIR" "run.name=$calibration_name" \
        "hydra.run.dir=$OUTPUT_DIR/hydra/$calibration_name" \
        >"$calibration_log" 2>&1
    fi
  fi
  PRIOR_SCALE="$($PYTHON_BIN -m gsmdiff.scripts.calibrate_filter_scale \
    "$CALIBRATION_SAMPLES" --distribution hyperbolic_secant \
    --output "$OUTPUT_DIR/prior_scale_calibration.json" --print-scale-only)"
else
  echo "Using explicitly supplied PRIOR_SCALE=$PRIOR_SCALE"
fi
echo "Moment-matched hyperbolic-secant scale: $PRIOR_SCALE"

FIELD_COUNT=${#FIELD_NAMES[@]}
TOTAL_JOBS=$((FIELD_COUNT * ${#SEEDS[@]} * ${#CG_ITERATIONS[@]} * ${#MEAN_FIELD_ITERATIONS[@]}))
declare -A PID_TO_GPU=()
declare -A PID_TO_LABEL=()
declare -A PID_TO_LOG=()
ACTIVE_JOBS=0
NEXT_JOB=0
COMPLETED_JOBS=0

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
  local field_index=$((job_index % FIELD_COUNT))
  local block_index=$((job_index / FIELD_COUNT))
  local seed_index=$((block_index % ${#SEEDS[@]}))
  block_index=$((block_index / ${#SEEDS[@]}))
  local cg_index=$((block_index % ${#CG_ITERATIONS[@]}))
  local mean_field_index=$((block_index / ${#CG_ITERATIONS[@]}))
  local cg_iterations=${CG_ITERATIONS[$cg_index]}
  local mean_field_iterations=${MEAN_FIELD_ITERATIONS[$mean_field_index]}
  local run_seed=${SEEDS[$seed_index]}
  local field_name=${FIELD_NAMES[$field_index]}
  local run_name="celeba_flow_sech_${field_name}_m${mean_field_iterations}_cg${cg_iterations}_cold_seed${run_seed}"
  local log_path="$OUTPUT_DIR/launcher_logs/${run_name}.log"
  local -a field_args

  case "$field_name" in
    q1_p0)
      field_args=(flow_weight=constant flow_weight.value=1.0 prior_weight=constant prior_weight.value=0.0)
      ;;
    q1_p0p005)
      field_args=(flow_weight=constant flow_weight.value=1.0 prior_weight=constant prior_weight.value=0.005)
      ;;
    q1_p0p01)
      field_args=(flow_weight=constant flow_weight.value=1.0 prior_weight=constant prior_weight.value=0.01)
      ;;
    q1_p0p05)
      field_args=(flow_weight=constant flow_weight.value=1.0 prior_weight=constant prior_weight.value=0.05)
      ;;
    q0p95_p0p05)
      field_args=(flow_weight=constant flow_weight.value=0.95 prior_weight=constant prior_weight.value=0.05)
      ;;
    q1_p_linear0_to_0p1)
      field_args=(flow_weight=constant flow_weight.value=1.0 prior_weight=linear prior_weight.start=0.0 prior_weight.end=0.1)
      ;;
    *) echo "Unknown field configuration: $field_name" >&2; return 2 ;;
  esac

  if [[ "$SKIP_COMPLETED" == true && -f "$OUTPUT_DIR/$run_name/geometric_samples.pt" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP completed $run_name"
    COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
    return 0
  fi

  local -a command=(
    "$PYTHON_BIN" -m gsmdiff.scripts.sample_geometric_flow_celeba
    "${MODEL_ARGS[@]}"
    distribution=hyperbolic_secant "distribution.scale=$PRIOR_SCALE"
    "${field_args[@]}"
    "sampling.num_inference_steps=$NUM_STEPS"
    "sampling.batch_size=$BATCH_SIZE"
    "checkpoints.every_steps=$CHECKPOINT_EVERY"
    "image.high_pass_contours.quantile=$CONTOUR_QUANTILE"
    comparison.generate_flow_baseline=false
    "prior.cg.max_iterations=$cg_iterations"
    "prior.mean_field_iterations=$mean_field_iterations"
    prior.cg.warm_start=false
    "runtime.seed=$run_seed" "runtime.device=$RUNTIME_DEVICE" runtime.deterministic=true
    "run.output_dir=$OUTPUT_DIR" "run.name=$run_name"
    "hydra.run.dir=$OUTPUT_DIR/hydra/$run_name"
  )
  if [[ -n "$CONTOUR_THRESHOLD" ]]; then
    command+=("image.high_pass_contours.threshold=$CONTOUR_THRESHOLD")
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu starting $run_name"
  CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="$OUTPUT_DIR/matplotlib_cache" \
    "${command[@]}" >"$log_path" 2>&1 &
  local pid=$!
  PID_TO_GPU[$pid]=$gpu
  PID_TO_LABEL[$pid]=$run_name
  PID_TO_LOG[$pid]=$log_path
  ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
}

schedule_next_job() {
  local gpu=$1
  while ((NEXT_JOB < TOTAL_JOBS)); do
    local active_before=$ACTIVE_JOBS
    launch_job "$NEXT_JOB" "$gpu"
    NEXT_JOB=$((NEXT_JOB + 1))
    if ((ACTIVE_JOBS > active_before)); then
      return 0
    fi
  done
}

for gpu in "${GPUS[@]}"; do
  schedule_next_job "$gpu"
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
  COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $freed_gpu finished $finished_label"
  schedule_next_job "$freed_gpu"
done

echo "All $TOTAL_JOBS flow/GGSM experiments are available ($COMPLETED_JOBS completed or reused)."
echo "Results: $OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/prior_scale_calibration.json" ]]; then
  echo "Scale calibration: $OUTPUT_DIR/prior_scale_calibration.json"
fi
