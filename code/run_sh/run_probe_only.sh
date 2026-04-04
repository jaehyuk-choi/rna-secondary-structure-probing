#!/bin/bash
# Unconstrained probe-only metrics → results/metrics.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python "$REPO_ROOT/code/evaluation/compute_probe_only_metrics.py" \
  --config-csv "$REPO_ROOT/configs/final_selected_config_unconstrained.csv" \
  --checkpoint-base "$REPO_ROOT/results/outputs" \
  --output-dir "$REPO_ROOT/results/metrics"
