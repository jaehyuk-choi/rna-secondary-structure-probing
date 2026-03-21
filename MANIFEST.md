# MANIFEST — Included Files

This document lists every file included in the package, grouped by purpose, with the original source path and reason for inclusion.

## Dissertation

| File | Original Path | Reason |
|---|---|---|
| `dissertation/skeleton.tex` | `dissertations/skeleton.tex` | Main LaTeX source — source of truth for all tables, figures, and claims |
| `dissertation/EXPERIMENTAL_WORKFLOW.md` | `dissertations/EXPERIMENTAL_WORKFLOW.md` | Documents the full experimental pipeline |

## Code: Models

| File | Original Path | Reason |
|---|---|---|
| `code/models/bilinear_probe_model.py` | `dissertations/feb23/models/bilinear_probe_model.py` | Canonical probe model definition. Chosen over jan22/feb8 versions as the final iteration used in the dissertation pipeline. |

## Code: Preprocessing

| File | Original Path | Reason |
|---|---|---|
| `code/preprocessing/compute_structure_features.py` | `dissertations/feb8/preprocessing/compute_structure_features.py` | Computes contact maps and structure features from bpRNA annotations |
| `code/preprocessing/plot_bprna_length_boxplot.py` | `dissertations/data/plot_bprna_length_boxplot.py` | Generates the bpRNA length distribution figure (Fig: bpRNA_length_boxplot.png) |

## Code: Probe Training

| File | Original Path | Reason |
|---|---|---|
| `code/probe_training/run_split_pipeline.py` | `dissertations/feb23/scripts/run_split_pipeline.py` | Main training pipeline for probe models across splits. Chosen over feb8 version as the final iteration. |

## Code: Probe Inference

| File | Original Path | Reason |
|---|---|---|
| `code/probe_inference/generate_base_pairs.py` | `dissertations/jan22/scripts/generation/generate_base_pairs.py` | Core base pair generation from probe scores |
| `code/probe_inference/generate_vl0_base_pairs.py` | `dissertations/feb23/scripts/generate_vl0_base_pairs.py` | Generate VL0 base pairs for CPLfold alpha sweep |

## Code: Evaluation

| File | Original Path | Reason |
|---|---|---|
| `code/evaluation/evaluate_predictions.py` | `dissertations/feb8/scripts/evaluation/evaluate_predictions.py` | Core evaluation: computes F1/precision/recall for predicted structures |
| `code/evaluation/compute_feb8_probe_only_metrics.py` | `dissertations/feb8/scripts/evaluation/compute_feb8_probe_only_metrics.py` | Aggregates probe-only metrics across models |
| `code/evaluation/compute_probe_only_with_wobble.py` | `dissertations/march1/compute_probe_only_with_wobble.py` | Evaluation under canonical+wobble pairing constraints |
| `code/evaluation/select_unconstrained_best_config.py` | `dissertations/march1/select_unconstrained_best_config.py` | Selects best unconstrained configuration per model |
| `code/evaluation/build_summary_table.py` | `dissertations/march1/build_summary_table.py` | Builds summary tables from evaluation results |

## Code: Folding Integration

| File | Original Path | Reason |
|---|---|---|
| `code/folding_integration/CPLfold_inter.py` | `CPLfold_inter/CPLfold_inter.py` | CPLfold structured folding algorithm (external package) |
| `code/folding_integration/base_pair.txt` | `CPLfold_inter/base_pair.txt` | Base pair definitions for CPLfold |
| `code/folding_integration/README.md` | `CPLfold_inter/README.md` | CPLfold documentation |
| `code/folding_integration/Utils/*` | `CPLfold_inter/Utils/*` | Energy parameter and utility modules for CPLfold |
| `code/folding_integration/run_cplfold_exp.py` | `dissertations/feb25/scripts/run_cplfold_exp.py` | Runs CPLfold experiments with probe-derived priors |
| `code/folding_integration/generate_thresholded_base_pairs.py` | `dissertations/feb25/scripts/generate_thresholded_base_pairs.py` | Generates thresholded base pair files for CPLfold input |
| `code/folding_integration/aggregate_cplfold_val_ts0_new.py` | `dissertations/feb23/scripts/aggregate_cplfold_val_ts0_new.py` | Aggregates CPLfold results across VL0/TS0/NEW |

## Code: Analysis

| File | Original Path | Reason |
|---|---|---|
| `code/analysis/build_canonical_rate_table_with_baseline.py` | `dissertations/march1/scripts/` | Computes canonical pairing rates (Table: canonical_ts_new) |
| `code/analysis/compute_seq_f1_and_gt.py` | `dissertations/march1/scripts/` | Per-sequence F1 computation |
| `code/analysis/compute_wc_gu_rate_all_pairs.py` | `dissertations/march1/scripts/` | WC/GU rate computation |
| `code/analysis/onehot_pair_combo_distribution.py` | `dissertations/march1/scripts/` | Pair-type distribution (Table: pair_distribution) |
| `code/analysis/roberta_vs_others_statistical_test.py` | `dissertations/march1/scripts/` | Paired significance tests (Table: roberta_pairwise) |
| `code/analysis/compare_alpha0_vs_best.py` | `dissertations/feb25/scripts/` | Alpha=0 vs best comparison for CPLfold |
| `code/analysis/alpha0_vs_best_stats.py` | `dissertations/feb25/scripts/` | Statistical summary of alpha comparisons |
| `code/analysis/aggregate_results.py` | `dissertations/feb25/scripts/` | Aggregate CPLfold experiment results |
| `code/analysis/analyze_by_length.py` | `dissertations/feb25/scripts/` | Length-stratified analysis |

## Code: Plotting

All plotting scripts are from `dissertations/march1/scripts/` — the final canonical versions that produce the dissertation figures.

| File | Produces Figure |
|---|---|
| `plot_probe_comparison_grouped_bar.py` | `fig1_grouped_bar_f1.png` |
| `plot_f1_by_length_boxplot.py` | `f1_by_length_boxplot.png` |
| `plot_layer_wise_val_f1.py` | `layer_wise_val_f1.png` |
| `plot_k_comparison_by_model.py` | `k_comparison_val_f1_by_model.png` |
| `plot_vl0_threshold_sweep.py` | `vl0_threshold_sweep.png` |
| `plot_vl0_alpha_sweep.py` | `vl0_alpha_sweep_vienna.png`, `vl0_alpha_sweep_contrafold.png` |
| `plot_probe_contact_heatmaps.py` | `probe_contact_heatmaps_selected_updated.png`, `probe_contact_heatmaps.png` |
| `plot_probe_only.py` | Probe-only summary plots |
| `plot_pair_combo_distribution.py` | Pair combo distribution plot |
| `plot_alpha0_vs_best.py` | Alpha comparison plots |

## Code: Utils and Run Scripts

| File | Original Path | Reason |
|---|---|---|
| `code/utils/evaluation.py` | `dissertations/feb8/utils/evaluation.py` | Shared evaluation functions |
| `code/run_sh/run_all_plots.sh` | `dissertations/march1/scripts/run_all_plots.sh` | Shell script to regenerate all figures |
| `code/run_sh/run_probe_only.sh` | `dissertations/march1/run_probe_only.sh` | Run probe-only evaluation |

## Configs

| File | Original Path | Reason |
|---|---|---|
| `configs/final_selected_config_unconstrained.csv` | `dissertations/march1/final_selected_config_unconstrained.csv` | Unconstrained best configuration per model. Supports Table: probe_unconstrained_full |
| `configs/val_optimal_results.csv` | `dissertations/feb23/validation_based_optimal/val_optimal_results.csv` | Full validation-optimal results across all hyperparameters |
| `configs/best_config_val_f1.csv` | `dissertations/feb23/best_config_val_f1.csv` | Best config by VL0 F1. Supports Table: best_config_val |

## Results

### results/metrics/
| File | Supports |
|---|---|
| `final_test_metrics.csv` | Table: probe_best_full (TS0) |
| `final_new_metrics.csv` | Table: probe_best_full (NEW) |
| `final_test_metrics_wobble.csv` | Wobble constraint results |
| `final_new_metrics_wobble.csv` | Wobble constraint results |
| `unconstrained_results_summary.csv` | Table: probe_unconstrained_full |
| `probe_unconstrained_vs_best_comparison.csv` | Table: probe_unconstrained_vs_best |

### results/per_sequence/
| File | Supports |
|---|---|
| `ts_per_sequence_metrics.csv` | Tables: median_f1_by_length, appendix_f1_by_length_ts0, roberta_pairwise |
| `new_per_sequence_metrics.csv` | Tables: median_f1_by_length, appendix_f1_by_length_new, roberta_pairwise |

### results/tables/
Table-specific data files supporting dissertation tables directly.

### results/statistics/
| File | Supports |
|---|---|
| `roberta_vs_others_significance.csv` | Table: roberta_pairwise |

### results/sweeps/
| File | Supports |
|---|---|
| `layer_wise_val_f1.csv` | Figure: layer_wise_val_f1.png |
| `k_comparison_val_f1.csv` | Figure: k_comparison_val_f1_by_model.png, Table: k_val_f1 |
| `vl0_alpha_sweep_both.csv` | Figures: vl0_alpha_sweep_vienna.png, vl0_alpha_sweep_contrafold.png |

### results/folding/
| File | Supports |
|---|---|
| `alpha0_vs_best_full.csv` | Tables: alpha_vienna, alpha_contrafold |
| `alpha0_vs_best_summary.csv` | Summary of CPLfold improvements |
| `cplfold_summary.csv` | Overall CPLfold experiment summary |
| `detailed_alpha_sweep_*.csv` | Detailed alpha sweep data per backend/split |
| `results_by_length.csv` | Length-stratified CPLfold results |
| `ts0_results_summary.csv` | TS0 CPLfold summary |

## Figures

### figures/main/ (9 files — all referenced by `\includegraphics` in skeleton.tex)
| File | LaTeX Reference | Description |
|---|---|---|
| `bpRNA_length_boxplot.png` | Fig: bpRNA_length_boxplot | Dataset length distribution |
| `probe_contact_heatmaps_selected_updated.png` | Fig: probe_heatmaps_selected | Selected model heatmaps (ERNIE, RoBERTa, One-hot) |
| `fig1_grouped_bar_f1.png` | Fig: probe_unconstrained_vs_best | Unconstrained vs best F1 comparison |
| `f1_by_length_boxplot.png` | Fig: f1_by_length | F1 by sequence length |
| `layer_wise_val_f1.png` | Fig: val_f1_by_layer | Layer-wise validation F1 |
| `k_comparison_val_f1_by_model.png` | Fig: val_f1_by_k | Probe rank comparison |
| `vl0_threshold_sweep.png` | Fig: vl0_threshold_sweep | Threshold sweep on VL0 |
| `vl0_alpha_sweep_vienna.png` | Fig: vl0_alpha_vienna | Alpha sweep ViennaRNA |
| `vl0_alpha_sweep_contrafold.png` | Fig: vl0_alpha_contrafold | Alpha sweep CONTRAfold |

### figures/appendix/ (1 file)
| File | LaTeX Reference | Description |
|---|---|---|
| `probe_contact_heatmaps.png` | Fig: probe_heatmaps_bprna_rfam_21498 | Full 6-model heatmaps |

## Data

| File | Purpose |
|---|---|
| `data/metadata/ArchiveII.csv` | ArchiveII RNA structure metadata |
| `data/metadata/bpRNA.csv` | bpRNA dataset metadata |
| `data/splits/bpRNA_splits.csv` | TR0/VL0/TS0/NEW partition assignments |

## Docs

| File | Contents |
|---|---|
| `docs/BONUS_FILES.md` | CPLfold bonus file format documentation |
| `docs/DATA.md` | Data layout and pipeline documentation |
| `docs/FIGURES.md` | Figure generation guide |
| `docs/PT_AND_PROBE_STRUCTURE.md` | Pretrained model and probe structure docs |
| `docs/SCRIPTS.md` | Script usage documentation |
| `docs/STORAGE.md` | Storage layout and path documentation |
