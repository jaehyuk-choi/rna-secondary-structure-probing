# FILE SELECTION DECISIONS

Documents every major deduplication or canonical selection decision made when building this package.

---

## Probe Model Definition

**Candidates:**
- `dissertations/jan22/models/bilinear_probe_model.py`
- `dissertations/feb8/models/bilinear_probe_model.py`
- `dissertations/feb23/models/bilinear_probe_model.py`
- `dissertations/feb25/code/feb23/models/bilinear_probe_model.py` (copy)
- `dissertations/feb25/code/feb8/models/bilinear_probe_model.py` (copy)
- `dissertations/feb25/code/jan22/models/bilinear_probe_model.py` (copy)

**Chosen canonical file:** `feb23/models/bilinear_probe_model.py` → `code/models/bilinear_probe_model.py`

**Reason:** feb23 is the final standalone iteration of the probe model used in the dissertation pipeline. The feb25 directory contains copies of earlier versions for reference but does not introduce a new version.

**Excluded:** jan22 and feb8 versions are earlier iterations of the same model. feb25 copies are duplicates.

---

## Probe Training Pipeline (run_split_pipeline.py)

**Candidates:**
- `dissertations/feb8/scripts/run_split_pipeline.py`
- `dissertations/feb23/scripts/run_split_pipeline.py`
- `dissertations/feb25/code/feb8/scripts/run_split_pipeline.py` (copy)
- `dissertations/feb25/code/feb23/scripts/run_split_pipeline.py` (copy)

**Chosen canonical file:** `feb23/scripts/run_split_pipeline.py` → `code/probe_training/run_split_pipeline.py`

**Reason:** feb23 version is the final iteration used for the actual probe training runs that produced the dissertation results. feb8 is an earlier version. feb25 copies are duplicates.

---

## Evaluation Utility (evaluation.py)

**Candidates:**
- `dissertations/jan22/utils/evaluation.py`
- `dissertations/feb8/utils/evaluation.py`
- `dissertations/feb25/code/jan22/utils/evaluation.py` (copy)
- `dissertations/feb25/code/feb8/utils/evaluation.py` (copy)

**Chosen canonical file:** `feb8/utils/evaluation.py` → `code/utils/evaluation.py`

**Reason:** feb8 version is the updated evaluation utility used by the feb8 and later evaluation scripts.

---

## Probe Config Selection

**Candidates:**
- `dissertation_submission_old/configs/jan22_final_selected_config.csv`
- `dissertation_submission_old/configs/feb8_final_selected_config.csv`
- `dissertations/march1/final_selected_config_unconstrained.csv`
- `dissertations/feb23/validation_based_optimal/val_optimal_results.csv`
- `dissertations/feb23/best_config_val_f1.csv`

**Chosen canonical files:**
- `march1/final_selected_config_unconstrained.csv` → `configs/final_selected_config_unconstrained.csv`
  - Supports Table: probe_unconstrained_full
- `feb23/validation_based_optimal/val_optimal_results.csv` → `configs/val_optimal_results.csv`
  - Full validation results across all hyperparameters
- `feb23/best_config_val_f1.csv` → `configs/best_config_val_f1.csv`
  - Supports Table: best_config_val

**Reason:** The march1 unconstrained config matches the unconstrained table in the dissertation. The feb23 configs match the validation-selected best configurations (Table: best_config_val).

**Excluded:** jan22 and feb8 configs are earlier iterations that were superseded by the feb23/march1 configurations.

---

## Plotting Scripts

**Candidates:** Earlier plot scripts existed in jan22 (`plot_alpha_f1.py`), feb8 (`plot_alpha_f1.py`, `generate_length_tables_and_graphs.py`), and feb23 analysis outputs. March1 contains the complete final set of plotting scripts.

**Chosen canonical files:** All plotting scripts from `dissertations/march1/scripts/plot_*.py`

**Reason:** The march1 plotting scripts are the canonical versions that produced all dissertation figures. Earlier plotting scripts were superseded.

**Excluded:**
- `jan22/scripts/evaluation/plot_alpha_f1.py` — superseded
- `feb8/scripts/evaluation/plot_alpha_f1.py` — superseded
- `feb8/scripts/analysis/generate_length_tables_and_graphs.py` — superseded by march1 versions

---

## Figures

**Candidates per figure:**

### probe_contact_heatmaps_selected_updated.png
- `march1/figures/probe_heatmaps_bprna_rfam_21498/probe_contact_heatmaps_selected_updated.png` ← **chosen**
- `march1/figures/probe_heatmaps_bprna_rfam_21498/probe_contact_heatmaps_selected.png` (earlier version)
- `march1/figures/probe_heatmaps_bprna_rfam_21498/probe_contact_heatmaps_selected_updated.pdf` (PDF duplicate)

### fig1_grouped_bar_f1.png
- `march1/figures/probe_comparison/fig1_grouped_bar_f1.png` ← **chosen**
- `march1/figures/probe_comparison/fig1_grouped_bar_f1_lightorange.png` (cosmetic variant)
- `march1/figures/probe_comparison/fig2_grouped_bar_simple.png` (alternative layout)
- `march1/figures/probe_comparison/fig3_grouped_pair_ts0.png` through `fig8_*` (alternative visualisations)

### layer_wise_val_f1.png
- `march1/figures/layer_k_comparison/layer_wise_val_f1.png` ← **chosen**
- `march1/figures/layer_k_comparison/layer_wise_val_f1_lightorange.png` (cosmetic variant)

### f1_by_length_boxplot.png
- `march1/figures/length_boxplot/f1_by_length_boxplot.png` ← **chosen**
- `march1/figures/length_boxplot/f1_by_length_boxplot_lightorange.png` (cosmetic variant)

### vl0_threshold_sweep.png
- `march1/figures/vl0_threshold_sweep.png` ← **chosen**
- `march1/figures/vl0_threshold_sweep_lightorange.png` (cosmetic variant)

### vl0_alpha_sweep_{vienna,contrafold}.png
- `march1/figures/vl0_alpha_sweep_vienna.png` ← **chosen**
- `march1/figures/vl0_alpha_sweep_contrafold.png` ← **chosen**
- `march1/figures/vl0_alpha_sweep_both.png` (combined variant)
- `march1/figures/vl0_alpha_sweep_both_lightorange.png` (cosmetic variant)

### probe_contact_heatmaps.png (appendix)
- `march1/figures/probe_heatmaps_bprna_rfam_21498/probe_contact_heatmaps.png` ← **chosen**
- `march1/figures/probe_heatmaps/probe_contact_heatmaps.png` (different sequence set)
- `march1/figures/probe_heatmaps_bprna_rfam_22136/probe_contact_heatmaps.png` (different sequence)
- Per-model variants (`probe_contact_heatmaps_ernie.png`, etc.) — not referenced by LaTeX

**Reason:** Only the exact files referenced by `\includegraphics` in the LaTeX source are included. Lightorange variants, PDF duplicates, and alternative visualisations are excluded as cosmetic duplicates.

---

## CPLfold Results

**Candidates:**
- `dissertations/feb25/alpha0_vs_best_full.csv` (canonical CPLfold held-out results)
- `dissertations/feb25/results_thresholded_ts0_new/` (detailed sweep data)
- `dissertations/march1/data/alpha0_vs_best_full.csv` (copy in march1)
- `dissertations/march1/data/vl0_alpha_sweep_both.csv` (validation sweep data)
- Earlier results in feb8, feb23 result directories

**Chosen canonical files:**
- `feb25/alpha0_vs_best_full.csv` → `results/folding/alpha0_vs_best_full.csv`
- `feb25/results_thresholded_ts0_new/` detailed sweeps → `results/folding/detailed_alpha_sweep_*.csv`
- `march1/data/vl0_alpha_sweep_both.csv` → `results/sweeps/vl0_alpha_sweep_both.csv`

**Reason:** feb25 contains the final CPLfold held-out evaluation results (Tables: alpha_vienna, alpha_contrafold). march1 data contains the VL0 sweep data. Earlier results from feb8/feb23 are intermediate.

---

## Per-Sequence Metrics

**Candidates:**
- `dissertations/march1/data/ts_per_sequence_metrics.csv`
- `dissertations/march1/data/new_per_sequence_metrics.csv`
- `dissertation_submission_old/results/jan22/ts_per_sequence_metrics.csv`
- `dissertation_submission_old/results/jan22/new_per_sequence_metrics.csv`

**Chosen:** march1 versions — these correspond to the final pipeline with all constraint modes.

---

## Run Scripts / Slurm Scripts

**Candidates:** ~90+ shell scripts across `jan22/`, `feb8/`, `feb23/`, `feb25/`, and the old submission's `archive/slurm_scripts/`, differing mainly by model name and backend.

**Chosen:** 2 representative scripts:
- `march1/scripts/run_all_plots.sh` — regenerates all figures
- `march1/run_probe_only.sh` — runs probe-only evaluation

**Reason:** The individual model/backend Slurm scripts are highly repetitive (same template with different model/backend arguments). Including 2 representative scripts is sufficient for understanding the pipeline. The full set is documented in EXCLUDED_OR_SUPERSEDED.md.

---

## Analysis Scripts

**Candidates:**
- `jan22/scripts/analysis/analyze_base_pair_scores.py` (earlier)
- `jan22/scripts/analysis/analyze_split_base_pair_scores.py` (earlier)
- `feb8/scripts/analysis/generate_length_tables_and_graphs.py` (earlier)
- `march1/scripts/` analysis scripts (final)
- `feb25/scripts/` analysis scripts (CPLfold-specific)

**Chosen:** march1 scripts for probe analysis, feb25 scripts for CPLfold analysis.

**Reason:** march1 contains the final versions of all probe-related analysis scripts. feb25 contains the CPLfold-specific analysis scripts. Earlier jan22/feb8 versions were superseded.

---

## Documentation

**Candidates:**
- Multiple README.md files across jan22, feb8, feb23, feb25, march1
- Multiple guide files (INFERENCE_GUIDE.md, OPTIMAL_CONFIG_USAGE.md, etc.)
- Multiple analysis docs (ALPHA_F1_ANALYSIS.md, THRESHOLD_SWEEP_ANALYSIS.md, etc.)

**Chosen:** march1/docs/ files only — the final documentation set.

**Excluded:** Earlier guides and analysis docs from jan22, feb8, feb23 are superseded by march1 documentation. Slurm output files, progress logs, and intermediate guides are excluded.
