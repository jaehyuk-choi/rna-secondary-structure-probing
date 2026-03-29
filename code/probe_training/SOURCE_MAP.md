# Source Map: Probe Training Code

Maps each file in this directory to its upstream origin in the read-only
source tree at `/projects/public/u5cj/dissertation/jan20/`.

| Destination (this package) | Upstream source path | Notes |
|---|---|---|
| `experiment_config.py` | `experiment_config.py` | `ROOT` now configurable via `$RNA_PROBE_DATA_ROOT` |
| `train_probe_automated.py` | `training/train_probe_automated.py` | Imports rewritten for submission layout |
| `dataset_probe.py` | `utils/dataset_probe.py` | Relative import changed to absolute |
| `utils_batch.py` | `utils/utils_batch.py` | Unchanged |
| `sbatch_train_model.sh` | `training/sbatch_train_model.sh` | Hardcoded paths replaced with `$PROJECT_ROOT` / `$CONDA_ENV` |
| `run_all_experiments.sh` | `training/run_all_experiments.sh` | Paths parameterised |
| `__init__.py` | *(new)* | Package marker |

## Files already present in this repository (not re-copied)

| Existing location | Upstream equivalent | Status |
|---|---|---|
| `code/models/bilinear_probe_model.py` | `models/bilinear_probe_model.py` | Identical |
| `code/utils/evaluation.py` | `utils/evaluation.py` | Identical |

## Files intentionally excluded

| Upstream path | Reason |
|---|---|
| `training/train_probe.py` | Secondary/interactive script; not used by Slurm pipeline |
| `training/threshold_sweep.py` | Post-training analysis; not needed for training reproduction |
| `training/summarize_results.py` | Post-training aggregation |
| `training/final_evaluation.py` | Test-set evaluation (downstream of training) |
| `training/select_and_evaluate.py` | Config selection (downstream) |
| `training/*.sbatch` (other) | Threshold sweep / summarise / eval Slurm scripts |
| `training/plot_*.py` | Plotting scripts (already in `code/plotting/`) |
| `training/run_*_experiment.py` | Layer-wise and proj-dim analysis scripts |
| `training/generate_pr_curve.py` | PR curve generation |
| `training/run_paired_statistical_test.py` | Statistical tests (in `code/analysis/`) |
| `training/collect_sequence_level_results.py` | Result aggregation |
| `training/eval_baseline_contact_maps.py` | Baseline evaluation |
| `utils/evaluation_baseline.py` | Baseline-specific evaluation |
| `utils/evaluation_sequence_level.py` | Sequence-level evaluation |
| `utils/example_paired_statistical_test.py` | Example / template script |
| `results/`, `results_updated/` | Large output trees with checkpoints |
| `img/` | Generated figures |
| `*.csv` (root-level) | PR curve / threshold result artefacts |
| `baseline_evaluation/` | Baseline evaluation outputs |
