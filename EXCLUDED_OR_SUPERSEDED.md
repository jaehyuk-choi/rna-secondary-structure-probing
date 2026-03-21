# EXCLUDED OR SUPERSEDED FILES

All exclusions from the main package are intentional. This document explains what was excluded and why.

---

## 1. Date-Based Folder Organisation (Superseded)

The original repository organised work chronologically across five iteration folders:
- `dissertations/jan22/` — initial probe pipeline
- `dissertations/feb8/` — updated pipeline with validation workflow
- `dissertations/feb23/` — final probe training pipeline, CPLfold VL0 integration
- `dissertations/feb25/` — final CPLfold held-out experiments
- `dissertations/march1/` — final plotting, analysis, and figure generation

**Decision:** The canonical file from each logical function was selected regardless of which date folder it resided in. The date-based structure itself is not preserved.

---

## 2. Superseded Script Versions

### bilinear_probe_model.py (3 superseded versions)
- `jan22/models/bilinear_probe_model.py` — earliest version
- `feb8/models/bilinear_probe_model.py` — intermediate version
- `feb25/code/*/models/bilinear_probe_model.py` — copies within feb25

**Kept:** `feb23/models/bilinear_probe_model.py` (final iteration)

### run_split_pipeline.py (1 superseded)
- `feb8/scripts/run_split_pipeline.py` — earlier version

**Kept:** `feb23/scripts/run_split_pipeline.py`

### evaluation.py (1 superseded)
- `jan22/utils/evaluation.py` — earlier version

**Kept:** `feb8/utils/evaluation.py`

### plot_alpha_f1.py (2 superseded)
- `jan22/scripts/evaluation/plot_alpha_f1.py`
- `feb8/scripts/evaluation/plot_alpha_f1.py`

**Kept:** `march1/scripts/plot_vl0_alpha_sweep.py` (supersedes both)

### generate_length_tables_and_graphs.py (1 superseded)
- `feb8/scripts/analysis/generate_length_tables_and_graphs.py`

**Kept:** `march1/scripts/plot_f1_by_length_boxplot.py` + `code/analysis/analyze_by_length.py`

### analyze_base_pair_scores.py (2 superseded)
- `jan22/scripts/analysis/analyze_base_pair_scores.py`
- `jan22/scripts/analysis/analyze_split_base_pair_scores.py`

**Reason:** Superseded by march1 analysis scripts.

---

## 3. Cosmetic Figure Variants (Not Referenced by LaTeX)

All of the following are excluded because only the exact file referenced by `\includegraphics` is included:

- `fig1_grouped_bar_f1_lightorange.png` — lightorange colour variant
- `layer_wise_val_f1_lightorange.png` — lightorange colour variant
- `f1_by_length_boxplot_lightorange.png` — lightorange colour variant
- `vl0_threshold_sweep_lightorange.png` — lightorange colour variant
- `vl0_alpha_sweep_both_lightorange.png` — lightorange colour variant
- `vl0_alpha_sweep_both.png` — combined version (dissertation uses separate subfigures)
- `pair_combo_distribution.png` and `pair_combo_distribution_lightorange.png` — not referenced
- `fig1_grouped_bar.png` — earlier grouped bar variant
- `fig2_grouped_bar_simple.png` through `fig8_*` — alternative probe comparison layouts
- `probe_f1_comparison.png` — earlier comparison figure
- `probe_precision_recall.png` — not referenced by LaTeX
- All `.pdf` duplicates of `.png` figures (e.g., `probe_contact_heatmaps_selected_updated.pdf`)
- Per-model probe heatmaps (`probe_contact_heatmaps_ernie.png`, etc.) — not individually referenced
- `probe_heatmaps_bprna_rfam_22136/` — different sequence, not referenced

---

## 4. Slurm Launch Scripts (~90 scripts)

The original repository and old submission contained extensive Slurm scripts:
- `jan22/run_all_sequences.sh`, `run_all_with_evaluation.sh`, `test_ernie.sh`, etc.
- `dissertation_submission_old/archive/slurm_scripts/run_vl0_*.sh` (36 scripts)
- `dissertation_submission_old/archive/slurm_scripts/run_ts0_*.sh` (24 scripts)
- `dissertation_submission_old/archive/slurm_scripts/run_new_*.sh` (24 scripts)
- Various `run_model_*.sh` scripts in feb23, feb25

**Reason:** These scripts follow a uniform template differing only by model name and backend argument. Two representative scripts are included in `code/run_sh/`. The rest are excluded as repetitive.

---

## 5. Intermediate Result Files

### jan22 results (superseded)
- `jan22/results/` — all result files superseded by march1 metrics
- `jan22/all_runs_summary.csv`, `all_runs_summary_agg.csv` — intermediate
- `jan22/val_threshold_sweeps_agg.csv` — intermediate

### feb8 results (superseded)
- `feb8/results/`, `results_updated/`, `results_ernie_*`, `results_onehot_*`, etc. — numerous intermediate result directories
- `feb8/validation_based_results.csv` — superseded by feb23
- `feb8/ground_truth_pair_distribution*.csv` — superseded by march1 analysis

### feb23 intermediate results
- `feb23/results_vl0/`, `results_ts0/`, `results_new/` — intermediate pipeline outputs
- `feb23/results_vl0_contrafold/`, etc. — intermediate CPLfold outputs
- `feb23/results_vl0_feb8/`, `results_vl0_contrafold_feb8/` — feb8-config reruns
- `feb23/ground_truth_pair_distribution*.csv` — superseded
- `feb23/summary_table.csv` — intermediate

### feb25 intermediate outputs
- `feb25/base_pairs_thresholded/` — large directory of per-sequence base pair files (thousands of files)
- `feb25/results_thresholded_ts0_new/ts0_vienna/`, etc. — per-sequence CPLfold output directories

**Reason:** Only the final aggregated metrics and summary CSVs are needed. Per-sequence intermediate outputs are too large and not directly referenced by the dissertation.

---

## 6. Large Data Files (Too Large)

- `dissertations/data/contact_maps/` — binary .npy contact map files (thousands)
- `dissertations/data/embeddings/` — embedding files (very large)
- `feb25/base_pairs_thresholded/` — per-sequence thresholded base pair files
- Probe model checkpoint weights (`.pt` files in `jan22/models/`, `feb8/models/`, `feb23/models/`)

**Reason:** These are too large for a GitHub submission package. See `data/README.md` for instructions on regenerating them.

---

## 7. Debug / Scratch / Infrastructure Files

- `__pycache__/` directories (all iterations)
- `.pyc` compiled Python files
- Slurm output files (`slurm-*.out` in feb8, feb23)
- `nohup_wobble.out` (march1)
- Progress/resume logs (`probe_only_progress.log`, `probe_resume.log`, etc.)
- `generate_split_base_pairs.log` (jan22, feb8, feb23)
- `feb23/core` (core dump)

---

## 8. Redundant Documentation (Superseded)

- `jan22/README.md`, `feb8/README.md`, `feb23/README.md`, `feb25/README.md` — per-iteration READMEs
- `jan22/INFERENCE_GUIDE.md`, `feb8/INFERENCE_GUIDE.md`, `feb23/INFERENCE_GUIDE.md` — superseded
- `jan22/OPTIMAL_CONFIG_USAGE.md`, `feb8/OPTIMAL_CONFIG_USAGE.md`, `feb23/OPTIMAL_CONFIG_USAGE.md` — superseded
- `jan22/ALPHA_F1_ANALYSIS.md`, `feb8/ALPHA_F1_ANALYSIS.md`, `feb23/ALPHA_F1_ANALYSIS.md` — superseded
- `feb23/THRESHOLD_SWEEP_AND_EXPERIMENTAL_DESIGN.md` — superseded by march1 docs
- `feb23/FILES_TO_PUSH.md`, `feb8/FILES_TO_PUSH.md` — infrastructure only
- `feb23/RESUME_GUIDE.md` — infrastructure only
- `feb23/RNAFM_TS0_NEW_ANALYSIS.md` — intermediate analysis
- `feb23/SLURM_NHR_ANALYSIS.md` — infrastructure only
- `feb25/PY_FILES_MANIFEST.md` — intermediate
- `feb25/README_thresholded_ts0_new.md` — intermediate
- `feb25/alpha0_vs_best_stats_summary.md` — kept the CSV version instead

---

## 9. Old Submission Package

The entire `dissertation_submission_old/` directory is excluded because this package is rebuilt from scratch. Key issues with the old package:
- Date-based code organisation (`code/jan22_scripts/`, `code/feb8_scripts/`, etc.)
- Date-based result organisation (`results/jan22/`, `results/feb8/`, etc.)
- Included cosmetic figure variants
- Included ~90 Slurm scripts
- Included intermediate results alongside final ones
- Included a `notebooks/` directory (no notebooks found)

---

## 10. feb25 Code Copies

`dissertations/feb25/code/` contains full copies of code from jan22, feb8, feb23, and CPLfold_inter. These are duplicates made for deployment convenience and are excluded in favour of the originals.

- `feb25/code/jan22/` — copy of jan22 code
- `feb25/code/feb8/` — copy of feb8 code
- `feb25/code/feb23/` — copy of feb23 code
- `feb25/code/external/CPLfold_inter/` — copy of CPLfold_inter

---

## 11. Verification Scripts (Not Essential)

- `march1/scripts/verify_canonical_rate.py` — verification/debug script
- `march1/scripts/verify_heatmap_coords.py` — verification/debug script
- `feb8/scripts/utils/check_missing_files.py` — infrastructure utility

**Reason:** These are one-off verification or debugging scripts not part of the main pipeline.
