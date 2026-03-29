#!/bin/bash
# Batch plot scripts (figures under repo).
set -e
cd "$(dirname "$0")/.."
echo "[INFO] Running all plots..."
python plotting/plot_alpha0_vs_best.py
python plotting/plot_probe_only.py
python plotting/plot_vl0_alpha_sweep.py
echo "[INFO] Done. Figures in: $(pwd)/figures/"
