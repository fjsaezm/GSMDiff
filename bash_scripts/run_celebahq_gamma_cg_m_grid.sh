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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gamma_cg_m_grid_seed${SEED}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-8}"
NUM_STEPS="${NUM_STEPS:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-}"
PRIOR_SCALE="${PRIOR_SCALE:-auto}"
ONLY_GUIDANCE="${ONLY_GUIDANCE:-}"
CONTOUR_QUANTILE="${CONTOUR_QUANTILE:-0.98}"
CONTOUR_THRESHOLD="${CONTOUR_THRESHOLD:-}"

read -r -a GPUS <<<"${GPU_IDS:-0 3 4 7}"
read -r -a MEAN_FIELD_ITERATIONS <<<"${MEAN_FIELD_VALUES:-1 5 10}"
read -r -a CG_ITERATIONS <<<"${CG_VALUES:-500 50}"
if ((${#GPUS[@]} == 0 || ${#MEAN_FIELD_ITERATIONS[@]} == 0 || ${#CG_ITERATIONS[@]} == 0)); then
  echo "GPU_IDS, MEAN_FIELD_VALUES, and CG_VALUES must not be empty." >&2
  exit 2
fi
ALL_GUIDANCE_NAMES=(
  gamma0
  gamma0p005
  gamma0p01
  power10_gamma0p1
)

if ((SEED_COUNT <= 0)); then
  echo "SEED_COUNT must be positive." >&2
  exit 2
fi
SEEDS=()
for ((seed_offset = 0; seed_offset < SEED_COUNT; seed_offset++)); do
  SEEDS+=("$((SEED + seed_offset))")
done

if [[ -n "$ONLY_GUIDANCE" ]]; then
  guidance_is_valid=false
  for guidance_name in "${ALL_GUIDANCE_NAMES[@]}"; do
    if [[ "$guidance_name" == "$ONLY_GUIDANCE" ]]; then
      guidance_is_valid=true
      break
    fi
  done
  if [[ "$guidance_is_valid" != true ]]; then
    echo "Unknown ONLY_GUIDANCE: $ONLY_GUIDANCE" >&2
    echo "Valid schedules: ${ALL_GUIDANCE_NAMES[*]}" >&2
    exit 2
  fi
  GUIDANCE_NAMES=("$ONLY_GUIDANCE")
else
  GUIDANCE_NAMES=("${ALL_GUIDANCE_NAMES[@]}")
fi

mkdir -p "$OUTPUT_DIR/launcher_logs" "$OUTPUT_DIR/hydra" "$OUTPUT_DIR/matplotlib_cache"

# A clean dataset tensor is preferred. If none is supplied, generate a model-matched
# diffusion-only batch. Gamma zero skips the prior, so its provisional unit scale is inert.
if [[ "$PRIOR_SCALE" == auto ]]; then
  if [[ -z "$CALIBRATION_SAMPLES" ]]; then
    calibration_name="celebahq_diffusion_calibration_seed${SEED}"
    CALIBRATION_SAMPLES="$OUTPUT_DIR/$calibration_name/geometric_samples.pt"
    if [[ ! -f "$CALIBRATION_SAMPLES" ]]; then
      calibration_log="$OUTPUT_DIR/launcher_logs/$calibration_name.log"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPUS[0]} starting scale calibration baseline"
      CUDA_VISIBLE_DEVICES="${GPUS[0]}" MPLCONFIGDIR="$OUTPUT_DIR/matplotlib_cache" \
        "$PYTHON_BIN" -m gsmdiff.scripts.sample_geometric_celebahq \
        distribution=hyperbolic_secant distribution.scale=1.0 \
        sampling.score_combination=additive guidance=constant guidance.value=0.0 \
        "sampling.num_inference_steps=$NUM_STEPS" \
        "sampling.batch_size=$CALIBRATION_BATCH_SIZE" \
        checkpoints.every_steps=null comparison.generate_diffusion_baseline=false \
        "runtime.seed=$SEED" runtime.device=cuda runtime.deterministic=true \
        "run.output_dir=$OUTPUT_DIR" "run.name=$calibration_name" \
        "hydra.run.dir=$OUTPUT_DIR/hydra/$calibration_name" \
        >"$calibration_log" 2>&1
    fi
  fi
  PRIOR_SCALE="$($PYTHON_BIN -m gsmdiff.scripts.calibrate_filter_scale \
    "$CALIBRATION_SAMPLES" --distribution hyperbolic_secant \
    --output "$OUTPUT_DIR/prior_scale_calibration.json" --print-scale-only)"
elif [[ -z "$CALIBRATION_SAMPLES" ]]; then
  echo "Using explicitly supplied PRIOR_SCALE=$PRIOR_SCALE"
fi

echo "Moment-matched hyperbolic-secant scale: $PRIOR_SCALE"

GUIDANCE_COUNT=${#GUIDANCE_NAMES[@]}
TOTAL_JOBS=$((GUIDANCE_COUNT * ${#SEEDS[@]} * ${#CG_ITERATIONS[@]} * ${#MEAN_FIELD_ITERATIONS[@]}))
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
  local guidance_index=$((job_index % GUIDANCE_COUNT))
  local block_index=$((job_index / GUIDANCE_COUNT))
  local seed_index=$((block_index % ${#SEEDS[@]}))
  block_index=$((block_index / ${#SEEDS[@]}))
  local cg_index=$((block_index % ${#CG_ITERATIONS[@]}))
  local mean_field_index=$((block_index / ${#CG_ITERATIONS[@]}))
  local cg_iterations=${CG_ITERATIONS[$cg_index]}
  local mean_field_iterations=${MEAN_FIELD_ITERATIONS[$mean_field_index]}
  local run_seed=${SEEDS[$seed_index]}
  local guidance_name=${GUIDANCE_NAMES[$guidance_index]}
  local run_name="celebahq_sech_${guidance_name}_m${mean_field_iterations}_cg${cg_iterations}_cold_seed${run_seed}"
  local log_path="$OUTPUT_DIR/launcher_logs/${run_name}.log"
  local -a guidance_args

  case "$guidance_name" in
    gamma0) guidance_args=(guidance=constant guidance.value=0.0) ;;
    gamma0p005) guidance_args=(guidance=constant guidance.value=0.005) ;;
    gamma0p01) guidance_args=(guidance=constant guidance.value=0.01) ;;
    power10_gamma0p1) guidance_args=(guidance=power_growth guidance.maximum=0.1 guidance.power=10.0) ;;
    *) echo "Unknown guidance schedule: $guidance_name" >&2; return 2 ;;
  esac

  local -a command=(
    "$PYTHON_BIN" -m gsmdiff.scripts.sample_geometric_celebahq
    distribution=hyperbolic_secant "distribution.scale=$PRIOR_SCALE"
    sampling.score_combination=additive "${guidance_args[@]}"
    "sampling.num_inference_steps=$NUM_STEPS"
    "sampling.batch_size=$BATCH_SIZE"
    "checkpoints.every_steps=$CHECKPOINT_EVERY"
    "image.high_pass_contours.quantile=$CONTOUR_QUANTILE"
    comparison.generate_diffusion_baseline=false
    "prior.cg.max_iterations=$cg_iterations"
    "prior.mean_field_iterations=$mean_field_iterations"
    prior.cg.warm_start=false
    "runtime.seed=$run_seed" runtime.device=cuda runtime.deterministic=true
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

echo "All $TOTAL_JOBS additive-guidance experiments completed successfully."
echo "Results: $OUTPUT_DIR"
echo "Scale calibration: $OUTPUT_DIR/prior_scale_calibration.json"
