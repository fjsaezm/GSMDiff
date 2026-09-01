#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gamma_metrics_seed20260811}"
REAL_IMAGES="${REAL_IMAGES:-}"
METRIC_DEVICE="${METRIC_DEVICE:-cuda}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-64}"
MINIMUM_SAMPLES="${MINIMUM_SAMPLES:-1000}"
KID_SUBSET_SIZE="${KID_SUBSET_SIZE:-1000}"
KID_SUBSETS="${KID_SUBSETS:-100}"
MEAN_FIELD_ITERATIONS="${MEAN_FIELD_ITERATIONS:-5}"
CG_ITERATIONS="${CG_ITERATIONS:-500}"
GUIDANCE_NAMES=(gamma0 gamma0p005 gamma0p01 power10_gamma0p1)

if [[ -z "$REAL_IMAGES" ]]; then
  echo "REAL_IMAGES must point to a clean CelebA-HQ image directory or NCHW .pt tensor." >&2
  exit 2
fi
if [[ ! -e "$REAL_IMAGES" ]]; then
  echo "REAL_IMAGES does not exist: $REAL_IMAGES" >&2
  exit 2
fi

metrics_dir="$OUTPUT_DIR/distribution_metrics"
mkdir -p "$metrics_dir"

for guidance_name in "${GUIDANCE_NAMES[@]}"; do
  fake_tensors=()
  run_metrics=()
  while IFS= read -r tensor_path; do
    run_name=$(basename "$(dirname "$tensor_path")")
    if [[ "$run_name" == "celebahq_sech_${guidance_name}_m${MEAN_FIELD_ITERATIONS}_cg${CG_ITERATIONS}_cold_seed"* ]]; then
      fake_tensors+=("$tensor_path")
      run_metrics+=("$(dirname "$tensor_path")/metrics.json")
    fi
  done < <(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -name geometric_samples.pt -print | sort)
  if ((${#fake_tensors[@]} == 0)); then
    echo "No generated tensors found for $guidance_name." >&2
    exit 1
  fi
  echo "Evaluating $guidance_name from ${#fake_tensors[@]} tensor batches"
  "$PYTHON_BIN" -m gsmdiff.scripts.evaluate_fid_kid \
    --real "$REAL_IMAGES" \
    --fake "${fake_tensors[@]}" \
    --output "$metrics_dir/${guidance_name}.json" \
    --device "$METRIC_DEVICE" \
    --batch-size "$METRIC_BATCH_SIZE" \
    --minimum-samples "$MINIMUM_SAMPLES" \
    --kid-subset-size "$KID_SUBSET_SIZE" \
    --kid-subsets "$KID_SUBSETS"
  "$PYTHON_BIN" -m gsmdiff.scripts.aggregate_ggsm_energy \
    "${run_metrics[@]}" \
    --output "$metrics_dir/${guidance_name}_ggsm_energy.json"
done

echo "FID/KID results: $metrics_dir"
