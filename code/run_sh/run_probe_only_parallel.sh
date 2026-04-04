#!/bin/bash
# Run probe-only metrics in parallel: one process per (model × partition), excluding ernie.
# Each job uses its own --output-dir / --per-sequence-dir / --progress-log (no collision with
# a single-job run under results/metrics/ or the ongoing ernie job).
#
# Usage:
#   bash run_probe_only_parallel.sh              # background all 10 jobs, wait at end
#   MAX_PARALLEL=2 bash run_probe_only_parallel.sh   # limit concurrency (optional)
#
# Env (same as run_probe_per_sequence.sh):
#   EMBEDDINGS_BASE, CHECKPOINT_BASE, CONDA_ENV (default: rna_probe)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EMB="${EMBEDDINGS_BASE:-$REPO_ROOT/data/embeddings}"
CKPT="${CHECKPOINT_BASE:-$REPO_ROOT/results/outputs}"
CONDA_ENV="${CONDA_ENV:-rna_probe}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"

METRICS_ROOT="$REPO_ROOT/results/metrics/parallel_probe_only"
PS_ROOT="$REPO_ROOT/results/per_sequence/parallel_probe_only"
mkdir -p "$METRICS_ROOT" "$PS_ROOT"

MODELS=(roberta rnafm rinalmo onehot rnabert)
PARTS=(ts0 new)

run_one() {
  local model="$1"
  local part="$2"
  local tag="${model}_${part}"
  local out="$METRICS_ROOT/$tag"
  local ps="$PS_ROOT/$tag"
  mkdir -p "$out" "$ps"
  echo "[start] $tag -> $out"
  conda run -n "$CONDA_ENV" python "$REPO_ROOT/code/evaluation/compute_probe_only_metrics.py" \
    --config-csv "$REPO_ROOT/configs/final_selected_config_unconstrained.csv" \
    --embeddings-base "$EMB" \
    --checkpoint-base "$CKPT" \
    --models "$model" \
    --dataset "$part" \
    --output-dir "$out" \
    --per-sequence \
    --per-sequence-dir "$ps" \
    --progress-log "$out/probe_only_progress.log"
  echo "[done]  $tag"
}

pids=()
count=0
for model in "${MODELS[@]}"; do
  for part in "${PARTS[@]}"; do
    while [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]]; do
      sleep 2
    done
    run_one "$model" "$part" &
    pids+=($!)
    count=$((count + 1))
  done
done
for pid in "${pids[@]}"; do
  wait "$pid" || exit 1
done
echo "All $count jobs finished. Metrics under $METRICS_ROOT, per-sequence under $PS_ROOT"
