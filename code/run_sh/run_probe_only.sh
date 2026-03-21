#!/bin/bash
# Run probe_only metrics with unconstrained config.
# Usage: bash run_probe_only.sh
# Requires: conda activate rna_probe

cd /projects/u6cg/jay/dissertations
python feb8/scripts/evaluation/compute_feb8_probe_only_metrics.py \
  --config-csv march1/final_selected_config_unconstrained.csv \
  --checkpoint-base feb8/results_updated/outputs \
  --output-dir march1
