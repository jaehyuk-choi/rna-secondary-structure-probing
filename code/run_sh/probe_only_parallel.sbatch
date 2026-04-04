#!/bin/bash
#SBATCH --job-name=probe_par
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/probe_only_parallel_%A_%a.out
#SBATCH --error=results/logs/probe_only_parallel_%A_%a.err

# One array task: one model × one partition (ts0 or new). Ernie excluded (5 models × 2 = 10 tasks).
# Submit: sbatch code/run_sh/probe_only_parallel.sbatch
#
# Override paths if needed:
#   #SBATCH --export=ALL,EMBEDDINGS_BASE=/path,CHECKPOINT_BASE=/path,CONDA_ENV=rna_probe

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EMB="${EMBEDDINGS_BASE:-$REPO_ROOT/data/embeddings}"
CKPT="${CHECKPOINT_BASE:-$REPO_ROOT/results/outputs}"
CONDA_ENV="${CONDA_ENV:-rna_probe}"

METRICS_ROOT="$REPO_ROOT/results/metrics/parallel_probe_only"
PS_ROOT="$REPO_ROOT/results/per_sequence/parallel_probe_only"
mkdir -p "$METRICS_ROOT" "$PS_ROOT" "$REPO_ROOT/results/logs"

MODELS=(roberta rnafm rinalmo onehot rnabert)
PARTS=(ts0 new)

i="${SLURM_ARRAY_TASK_ID:?}"
model_idx=$((i / 2))
part_idx=$((i % 2))
model="${MODELS[$model_idx]}"
part="${PARTS[$part_idx]}"
tag="${model}_${part}"
out="$METRICS_ROOT/$tag"
ps="$PS_ROOT/$tag"
mkdir -p "$out" "$ps"

echo "Task $i: model=$model dataset=$part -> $out"

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

echo "Finished $tag"
