#!/bin/bash
# ---------------------------------------------------------------------------
# Master script: submit Slurm jobs for the full probe training pipeline.
#
# Steps (with dependency chaining):
#   1. Training         — one job per model (parallel across models)
#   2. Threshold sweep   — per model, after its training completes
#   3. Summarisation     — after all threshold sweeps
#   4. Final evaluation  — after summarisation
#
# Only step 1 (training) is included in this submission package.
# Steps 2–4 require sbatch_threshold_sweep_model.sh, sbatch_summarize.sh,
# and sbatch_final_eval.sh, which were part of the original HPC pipeline
# but are not included here. The corresponding Python scripts
# (threshold_sweep.py, summarize_results.py, final_evaluation.py) can be
# found in the upstream source — see SOURCE_MAP.md.
#
# Edit PROJECT_ROOT below before running.
# ---------------------------------------------------------------------------

# ---- Site-specific: override PROJECT_ROOT if needed ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
# ------------------------------------------------------------------

MODELS=("rnabert" "rnafm" "ernie" "rinalmo" "roberta" "onehot")

mkdir -p "${PROJECT_ROOT}/results/slurm_logs"

echo "=========================================="
echo "RNA Contact Prediction Probing Experiments"
echo "=========================================="
echo ""
echo "Models: ${MODELS[@]}"
echo ""

# Step 1: Training
echo "Step 1: Submitting training jobs..."
TRAIN_JOB_IDS=()
for MODEL in "${MODELS[@]}"; do
    echo "  Submitting training job for $MODEL..."
    JOB_ID=$(sbatch --parsable "${PROJECT_ROOT}/code/probe_training/sbatch_train_model.sh" "$MODEL")
    TRAIN_JOB_IDS+=($JOB_ID)
    echo "    Job ID: $JOB_ID"
done
echo "  Training jobs submitted: ${TRAIN_JOB_IDS[@]}"
echo ""

# Step 2: Threshold sweep (depends on training)
echo "Step 2: Submitting threshold sweep jobs (after training completes)..."
THRESH_JOB_IDS=()
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    TRAIN_JOB_ID="${TRAIN_JOB_IDS[$i]}"
    echo "  Submitting threshold sweep job for $MODEL (depends on $TRAIN_JOB_ID)..."
    JOB_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB_ID "${PROJECT_ROOT}/code/probe_training/sbatch_threshold_sweep_model.sh" "$MODEL")
    THRESH_JOB_IDS+=($JOB_ID)
    echo "    Job ID: $JOB_ID"
done
echo "  Threshold sweep jobs submitted: ${THRESH_JOB_IDS[@]}"
echo ""

# Step 3: Summarisation (depends on all threshold sweeps)
echo "Step 3: Submitting summarisation job (after all threshold sweeps complete)..."
ALL_THRESH_JOBS=$(IFS=:; echo "${THRESH_JOB_IDS[*]}")
SUM_JOB_ID=$(sbatch --parsable --dependency=afterok:$ALL_THRESH_JOBS "${PROJECT_ROOT}/code/probe_training/sbatch_summarize.sh")
echo "  Summarisation job ID: $SUM_JOB_ID"
echo ""

# Step 4: Final evaluation (depends on summarisation)
echo "Step 4: Submitting final evaluation job (after summarisation completes)..."
FINAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$SUM_JOB_ID "${PROJECT_ROOT}/code/probe_training/sbatch_final_eval.sh")
echo "  Final evaluation job ID: $FINAL_JOB_ID"
echo ""

echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Job dependencies:"
echo "  Training -> Threshold Sweep -> Summarisation -> Final Evaluation"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in: ${PROJECT_ROOT}/results/slurm_logs/"
