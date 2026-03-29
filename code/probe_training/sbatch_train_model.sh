#!/bin/bash
# ---------------------------------------------------------------------------
# Slurm job script: train probes for one RNA foundation model.
#
# Iterates over all (layer, k) combinations detected by experiment_config.py
# and calls train_probe_automated.py for each.  Existing checkpoints are
# skipped automatically.
#
# Usage:
#   sbatch sbatch_train_model.sh <model_name>
#   # e.g.  sbatch sbatch_train_model.sh rnafm
# ---------------------------------------------------------------------------
#SBATCH --job-name=train_probe
#SBATCH --output=results/slurm_logs/train_%j.out
#SBATCH --error=results/slurm_logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ---- Site-specific: override PROJECT_ROOT or CONDA_ENV if needed ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-rna_probe}"
# ------------------------------------------------------------------

MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Error: Model name required"
    echo "Usage: sbatch sbatch_train_model.sh <model_name>"
    exit 1
fi

echo "=========================================="
echo "Training probe for model: $MODEL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

cd "$PROJECT_ROOT"
source ~/.bashrc
conda activate "$CONDA_ENV"

# Detect layers via experiment_config
LAYERS=$(python3 -c "
import sys, os
sys.path.insert(0, os.path.join('$PROJECT_ROOT', 'code', 'probe_training'))
from experiment_config import detect_model_layers
layers = detect_model_layers('$MODEL')
print(' '.join(map(str, layers)))
")

K_VALUES="32 64 128"
SEED=42

echo "Model: $MODEL"
echo "Layers: $LAYERS"
echo "K values: $K_VALUES"
echo "Seed: $SEED"

TOTAL_RUNS=0
for layer in $LAYERS; do
    for k in $K_VALUES; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
done
echo "Total runs: $TOTAL_RUNS"
echo ""

RUN_COUNT=0
SKIP_COUNT=0
RESULTS_DIR="${PROJECT_ROOT}/results/outputs"

for layer in $LAYERS; do
    for k in $K_VALUES; do
        RUN_COUNT=$((RUN_COUNT + 1))

        CHECKPOINT_PATH="$RESULTS_DIR/$MODEL/layer_$layer/k_$k/seed_$SEED/best.pt"

        if [ -f "$CHECKPOINT_PATH" ]; then
            echo "=========================================="
            echo "Run $RUN_COUNT/$TOTAL_RUNS: SKIPPING $MODEL, layer=$layer, k=$k, seed=$SEED"
            echo "  (Checkpoint already exists: $CHECKPOINT_PATH)"
            echo "=========================================="
            SKIP_COUNT=$((SKIP_COUNT + 1))
            echo ""
            continue
        fi

        echo "=========================================="
        echo "Run $RUN_COUNT/$TOTAL_RUNS: Training $MODEL, layer=$layer, k=$k, seed=$SEED"
        echo "=========================================="

        python3 code/probe_training/train_probe_automated.py \
            --model "$MODEL" \
            --layer "$layer" \
            --k "$k" \
            --seed "$SEED"

        if [ $? -ne 0 ]; then
            echo "ERROR: Training failed for $MODEL, layer=$layer, k=$k"
            exit 1
        fi

        echo "Completed: $MODEL, layer=$layer, k=$k"
        echo ""
    done
done

echo "=========================================="
echo "Training summary:"
echo "  Total runs: $TOTAL_RUNS"
echo "  Skipped (already completed): $SKIP_COUNT"
echo "  Newly trained: $((TOTAL_RUNS - SKIP_COUNT))"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
