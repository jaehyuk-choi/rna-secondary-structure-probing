#!/bin/bash
# Run all visualization scripts and save figures to march1/figures/
set -e
cd "$(dirname "$0")/.."
echo "[INFO] Running all plots..."
python scripts/plot_alpha0_vs_best.py
python scripts/plot_probe_only.py
python scripts/plot_vl0_alpha_sweep.py
echo "[INFO] Done. Figures in: $(pwd)/figures/"
