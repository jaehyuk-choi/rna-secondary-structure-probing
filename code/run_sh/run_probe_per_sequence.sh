#!/bin/bash
# Probe-only micro metrics + per-sequence CSVs (unconstrained config).
# Override embedding/checkpoint roots if needed:
#   EMBEDDINGS_BASE=/path/to/embeddings CHECKPOINT_BASE=/path/to/outputs bash run_probe_per_sequence.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EMB="${EMBEDDINGS_BASE:-$REPO_ROOT/data/embeddings}"
CKPT="${CHECKPOINT_BASE:-$REPO_ROOT/results/outputs}"

exec conda run -n rna_probe python "$REPO_ROOT/code/evaluation/compute_probe_only_metrics.py" \
  --config-csv "$REPO_ROOT/configs/final_selected_config_unconstrained.csv" \
  --embeddings-base "$EMB" \
  --checkpoint-base "$CKPT" \
  --per-sequence \
  "$@"
