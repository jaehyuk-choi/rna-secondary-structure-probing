# Dissertation-Code Cross-Reference Audit

## A. Experiment Coverage Matrix

| # | Experiment | Dissertation section | Code location | Status |
|---|-----------|---------------------|---------------|--------|
| 1 | Probe training: 6 models x layers x k={32,64,128}, seed=42, TR0 | Sec 4.2, Table A1 | `code/probe_training/train_probe_automated.py`, `sbatch_train_model.sh` | COVERED |
| 2 | Threshold sweep on VL0: tau 0.50-0.95 x 3 pairing constraints | Sec 4.3, 5.5 | `code/evaluation/evaluate_predictions.py` (threshold sweep mode) | PARTIAL -- see note 1 |
| 3 | Best-config selection on VL0 (max F1 over layer, k, tau, mode) | Sec 4.3.2, Table 8 | `code/evaluation/select_unconstrained_best_config.py` | COVERED |
| 4 | Probe-only eval on TS0/NEW, unconstrained config | Sec 5.1, Table 5 | `code/evaluation/compute_feb8_probe_only_metrics.py` | COVERED |
| 5 | Probe-only eval on TS0/NEW, best config (incl. canonical constraints) | Sec 5.3, Table 9 | `code/evaluation/compute_feb8_probe_only_metrics.py` | COVERED |
| 6 | Canonical pairing rate (WC, WC+GU) under unconstrained decoding | Sec 5.2, Table 6 | `code/evaluation/compute_probe_only_with_wobble.py`, `code/analysis/build_canonical_rate_table_with_baseline.py` | PARTIAL -- see note 2 |
| 7 | WC+GU baseline rate among all candidate pairs | Sec 5.2, Table 6 bottom row | `code/analysis/compute_wc_gu_rate_all_pairs.py` | COVERED |
| 8 | Pair combo distribution (one-hot, RiNALMo) | Sec 5.2, Table 7 | `code/analysis/onehot_pair_combo_distribution.py` | COVERED |
| 9 | Unconstrained vs best-config comparison | Sec 5.3.1, Table 11 | Pre-computed: `results/metrics/probe_unconstrained_vs_best_comparison.csv` | COVERED |
| 10 | F1 by sequence length (boxplot + median table) | Sec 5.3.2, Table 12, Tables A2-A3 | `code/plotting/plot_f1_by_length_boxplot.py` | PARTIAL -- see note 3 |
| 11 | RoBERTa vs others statistical tests (paired t-test, Wilcoxon) | Sec 5.3.3, Table 13 | `code/analysis/roberta_vs_others_statistical_test.py` | PARTIAL -- see note 4 |
| 12 | Layer-wise VL0 F1 analysis | Sec 5.4.1, Fig 10 | `code/plotting/plot_layer_wise_val_f1.py` | PARTIAL -- see note 5 |
| 13 | Rank (k) sensitivity analysis | Sec 5.4.2, Fig 11, Table 14 | `code/plotting/plot_k_comparison_by_model.py` | PARTIAL -- see note 5 |
| 14 | Threshold sensitivity analysis on VL0 | Sec 5.4.3, Fig 12 | `code/plotting/plot_vl0_threshold_sweep.py` | COVERED |
| 15 | Generate base-pair bonus files for VL0/TS0/NEW | Sec 3.6.1, 4.4 | `code/probe_inference/generate_base_pairs.py`, `code/probe_inference/generate_vl0_base_pairs.py` | COVERED |
| 16 | CPLfold alpha sweep on VL0 (Vienna + CONTRAfold) | Sec 5.5.1, Fig 13 | `code/probe_training/run_split_pipeline.py`, `code/folding_integration/CPLfold_inter.py` | PARTIAL -- see note 6 |
| 17 | CPLfold held-out eval on TS0/NEW (Vienna + CONTRAfold) | Sec 5.5.2, Tables 15-16 | `code/probe_training/run_split_pipeline.py`, `code/analysis/compare_alpha0_vs_best.py` | COVERED |
| 18 | CPLfold VL0 summary (Appendix Table 17) | Appendix C | `code/folding_integration/aggregate_cplfold_val_ts0_new.py` | COVERED |
| 19 | Qualitative heatmap visualization (seq 21498) | Sec 5.3, Fig 7, Table 10 | `code/plotting/plot_probe_contact_heatmaps.py`, `code/analysis/compute_seq_f1_and_gt.py` | COVERED |
| 20 | bpRNA length distribution boxplot | Sec 4.1.1, Fig 4 | `code/preprocessing/plot_bprna_length_boxplot.py` | PARTIAL -- see note 7 |
| 21 | Base-pair statistics (pair rate, composition) | Sec 4.1.2, Table 3 | `code/preprocessing/compute_structure_features.py` (partial) | PARTIAL -- see note 8 |
| 22 | Contact map generation for training | Sec 3.4 | `code/preprocessing/compute_structure_features.py` | COVERED |
| 23 | Embedding extraction (all models x layers) | Sec 3.2 | Not in repo (external, expected) | N/A |

### Notes

1. **Threshold sweep (Exp 2):** `evaluate_predictions.py` can run threshold sweeps, but the script that systematically produces `val_threshold_sweep_{mode}.csv` files for every (model, layer, k) combination is not explicitly present as a standalone step. The threshold sweep data exists under `results/outputs/` (consumed by `plot_vl0_threshold_sweep.py`), so it was produced at some point. The generating code likely ran during training or as a post-training step not captured in the repo.

2. **Canonical rate table (Exp 6):** `build_canonical_rate_table_with_baseline.py` has broken paths -- it uses `MARCH1 = parents[1]` which resolves to `code/` instead of the repo root. It reads from `MARCH1 / 'final_test_metrics_wobble.csv'` (i.e., `code/final_test_metrics_wobble.csv`) which does not exist; the actual file is at `results/metrics/final_test_metrics_wobble.csv`.

3. **F1 by length (Exp 10):** `plot_f1_by_length_boxplot.py` uses `MARCH1 = parents[1]` (broken) and references `FEB8 = MARCH1.parent / 'feb8/results_updated/summary'` (external path). Would need rewiring to read from `results/per_sequence/`.

4. **RoBERTa significance (Exp 11):** `roberta_vs_others_statistical_test.py` uses `MARCH1 = parents[1]` and `FEB8 = MARCH1.parent / 'feb8/...'` (broken). Also reads `BEST_CONFIG_PATH = FEB8 / 'final_selected_config.csv'` (external). Pre-computed result exists at `results/statistics/roberta_vs_others_significance.csv`.

5. **Layer-wise and k-comparison plots (Exp 12-13):** Both use `MARCH1 = parents[1]` reading from `MARCH1 / 'data' / 'layer_wise_val_f1.csv'` (broken). Actual data is at `results/sweeps/layer_wise_val_f1.csv` and `results/sweeps/k_comparison_val_f1.csv`.

6. **VL0 alpha sweep (Exp 16):** `plot_vl0_alpha_sweep.py` looks for `results/folding/results_vl0_feb8/` and `results/folding/results_vl0_contrafold_feb8/` directories containing per-model detailed CSVs. These directories do not exist in the repo -- only the aggregated `results/sweeps/vl0_alpha_sweep_both.csv` is present. The script would fail; VL0 alpha sweep figures were generated externally.

7. **Length boxplot (Exp 20):** `plot_bprna_length_boxplot.py` uses hardcoded relative paths (`open('bpRNA_splits.csv')`, `open('bpRNA.csv')`) assuming CWD contains these files. Not portable without fixing to use REPO_ROOT.

8. **Pair statistics (Exp 21):** Table 3 (pair rates, AU/GC/GU composition) does not have a single generating script. `compute_structure_features.py` generates contact maps but not the summary statistics table. The values appear to be hand-computed or from an unlisted script.

---

## B. Figure/Table Traceability

| Figure/Table | Dissertation ref | Generating script | Input data | Status |
|-------------|-----------------|-------------------|------------|--------|
| Fig 1 (RNA structure) | Sec 2.1 | External (adapted from paper) | N/A | N/A -- not code-generated |
| Fig 2 (Timeline) | Sec 2.3 | TikZ in skeleton.tex | N/A | COVERED (inline) |
| Fig 3 (Pipeline diagram) | Sec 3.1.1 | External (diagram.pdf) | N/A | MISSING -- `dissertation/img/` does not exist |
| Fig 4 (bpRNA length boxplot) | Sec 4.1.1 | `code/preprocessing/plot_bprna_length_boxplot.py` | `data/splits/bpRNA_splits.csv`, `data/metadata/bpRNA.csv` | PARTIAL -- script has CWD-relative paths |
| Fig 5 (Max-one decoding) | Sec 3.5.1 | External (max 1.pdf) | N/A | MISSING -- `dissertation/img/` does not exist |
| Fig 7 (Probe heatmaps, selected) | Sec 5.3 | `code/plotting/plot_probe_contact_heatmaps.py` | Embeddings, checkpoints, bpRNA.csv | COVERED |
| Fig 8 (Grouped bar: unc vs best) | Sec 5.3.1 | `code/plotting/plot_probe_comparison_grouped_bar.py` | `results/metrics/probe_unconstrained_vs_best_comparison.csv` | PARTIAL -- script uses MARCH1 path |
| Fig 9 (F1 by length boxplot) | Sec 5.3.2 | `code/plotting/plot_f1_by_length_boxplot.py` | `results/per_sequence/ts_per_sequence_metrics.csv`, `new_per_sequence_metrics.csv` | PARTIAL -- broken paths |
| Fig 10 (Layer-wise VL0 F1) | Sec 5.4.1 | `code/plotting/plot_layer_wise_val_f1.py` | `results/sweeps/layer_wise_val_f1.csv` | PARTIAL -- script uses MARCH1 path |
| Fig 11 (k comparison) | Sec 5.4.2 | `code/plotting/plot_k_comparison_by_model.py` | `results/sweeps/k_comparison_val_f1.csv` | PARTIAL -- script uses MARCH1 path |
| Fig 12 (Threshold sweep) | Sec 5.4.3 | `code/plotting/plot_vl0_threshold_sweep.py` | `results/outputs/.../val_threshold_sweep_*.csv` | COVERED -- uses REPO_ROOT correctly |
| Fig 13a (VL0 alpha Vienna) | Sec 5.5.1 | `code/plotting/plot_vl0_alpha_sweep.py` | `results/folding/results_vl0_feb8/` | PARTIAL -- input dirs missing |
| Fig 13b (VL0 alpha CONTRAfold) | Sec 5.5.1 | `code/plotting/plot_vl0_alpha_sweep.py` | `results/folding/results_vl0_contrafold_feb8/` | PARTIAL -- input dirs missing |
| App Fig (Full heatmaps) | Appendix B | `code/plotting/plot_probe_contact_heatmaps.py` | Same as Fig 7 | COVERED |
| Table 1 (Model summary) | Sec 2.3 | Static (in LaTeX) | N/A | COVERED |
| Table 2 (Dataset splits) | Sec 4.1.1 | Static (in LaTeX) | N/A | COVERED |
| Table 3 (Pair statistics) | Sec 4.1.2 | No single script | N/A | PARTIAL -- no generating script |
| Table 4 (Bonus format) | Sec 3.6.1 | Static (in LaTeX) | N/A | COVERED |
| Table 5 (Unconstrained TS0/NEW) | Sec 5.1 | `compute_feb8_probe_only_metrics.py` | `results/metrics/final_test_metrics.csv`, `final_new_metrics.csv` | COVERED |
| Table 6 (Canonical rate) | Sec 5.2 | `compute_probe_only_with_wobble.py` + `build_canonical_rate_table_with_baseline.py` | `results/metrics/final_*_metrics_wobble.csv`, `results/tables/canonical_rate_baseline_table.csv` | COVERED (results exist; build script has broken paths) |
| Table 7 (Pair distribution) | Sec 5.2 | `onehot_pair_combo_distribution.py` | `results/tables/onehot_pair_combo_distribution.csv`, `rinalmo_pair_combo_distribution.csv` | COVERED |
| Table 8 (Best config per model) | Sec 5.3 | `select_unconstrained_best_config.py` | `configs/best_config_val_f1.csv` | COVERED |
| Table 9 (Best-config TS0/NEW) | Sec 5.3 | `compute_feb8_probe_only_metrics.py` | `results/metrics/final_test_metrics.csv`, `final_new_metrics.csv` | COVERED |
| Table 10 (Qualitative example) | Sec 5.3 | `code/analysis/compute_seq_f1_and_gt.py` | `results/tables/probe_heatmap_seq_f1_and_gt.txt` | COVERED |
| Table 11 (Unc vs best) | Sec 5.3.1 | Manual / `results/metrics/probe_unconstrained_vs_best_comparison.csv` | Derived from Tables 5+9 | COVERED |
| Table 12 (Median F1 by length) | Sec 5.3.2 | `plot_f1_by_length_boxplot.py` | Per-sequence metrics | PARTIAL -- broken paths |
| Table 13 (RoBERTa stat tests) | Sec 5.3.3 | `roberta_vs_others_statistical_test.py` | `results/statistics/roberta_vs_others_significance.csv` | COVERED (result file exists; script has broken paths) |
| Table 14 (k sensitivity) | Sec 5.4.2 | `plot_k_comparison_by_model.py` | `results/sweeps/k_comparison_val_f1.csv` | COVERED (result exists) |
| Table 15 (ViennaRNA TS0/NEW) | Sec 5.5.2 | `compare_alpha0_vs_best.py` | `results/folding/detailed_alpha_sweep_*.csv` | COVERED |
| Table 16 (CONTRAfold TS0/NEW) | Sec 5.5.2 | `compare_alpha0_vs_best.py` | `results/folding/detailed_alpha_sweep_*.csv` | COVERED |
| Table 17 (VL0 alpha summary) | Appendix C | `aggregate_cplfold_val_ts0_new.py` / `results/sweeps/vl0_alpha_sweep_both.csv` | VL0 sweep data | COVERED |
| Table A1 (Training config) | Appendix A | Static (in LaTeX) | N/A | COVERED |
| Tables A2-A3 (F1 by length) | Appendix B | `plot_f1_by_length_boxplot.py` | Per-sequence metrics | PARTIAL |

### Missing image directory

The dissertation references `img/` (e.g., `\includegraphics[width=\linewidth]{img/bpRNA_length_boxplot.png}`), but this directory does not exist under `dissertation/`. The generated figures are in `figures/main/` and `figures/appendix/`. The LaTeX will not compile without either:
- Creating `dissertation/img/` as a symlink to `figures/main/`, or
- Updating all `\includegraphics` paths in `skeleton.tex`.

---

## C. Gaps Requiring Action

### Critical (scripts will fail)

**C1. MARCH1 path variable in 8 scripts**

These scripts use `MARCH1 = Path(__file__).resolve().parents[1]` which resolves to `code/` instead of the repo root. All data paths are broken.

| Script | Should be | Currently resolves to |
|--------|-----------|----------------------|
| `code/plotting/plot_alpha0_vs_best.py` | `parents[2]` | `code/` |
| `code/plotting/plot_f1_by_length_boxplot.py` | `parents[2]` | `code/` |
| `code/plotting/plot_k_comparison_by_model.py` | `parents[2]` | `code/` |
| `code/plotting/plot_layer_wise_val_f1.py` | `parents[2]` | `code/` |
| `code/plotting/plot_pair_combo_distribution.py` | `parents[2]` | `code/` |
| `code/plotting/plot_probe_comparison_grouped_bar.py` | `parents[2]` | `code/` |
| `code/plotting/plot_probe_only.py` | `parents[2]` | `code/` |
| `code/analysis/build_canonical_rate_table_with_baseline.py` | `parents[2]` | `code/` |

Additionally, the data subpaths need updating from old layout names (e.g., `'data'` -> `'results/sweeps'`, `'results/metrics'`, etc.).

**C2. External path references in 2 scripts**

| Script | Broken reference |
|--------|-----------------|
| `code/analysis/roberta_vs_others_statistical_test.py` | `FEB8 = MARCH1.parent / 'feb8/results_updated/summary'` |
| `code/plotting/plot_f1_by_length_boxplot.py` | `FEB8 = MARCH1.parent / 'feb8/results_updated/summary'` |

These reference the old external `feb8/` directory tree that does not exist in this repo.

**C3. Undefined variable in `build_summary_table.py`**

`code/evaluation/build_summary_table.py` line 27: `MARCH1` is used but never defined (only `REPO_ROOT` is defined on line 10). This is a NameError at runtime.

**C4. VL0 alpha sweep input directories missing**

`code/plotting/plot_vl0_alpha_sweep.py` reads per-model CSVs from `results/folding/results_vl0_feb8/` and `results/folding/results_vl0_contrafold_feb8/`. These directories do not exist. Only aggregated `results/sweeps/vl0_alpha_sweep_both.csv` is present. The plot script needs to be rewritten to use the aggregated CSV, or the per-model VL0 sweep results need to be added to the repo.

**C5. `plot_bprna_length_boxplot.py` uses CWD-relative paths**

Opens `bpRNA_splits.csv` and `bpRNA.csv` with bare filenames (assumes CWD contains them). Needs `REPO_ROOT`-based paths.

### Important (won't block compilation but affect reproducibility)

**C6. Threshold sweep orchestration**

The dissertation describes a threshold sweep on VL0 across all (model, layer, k, mode, tau) combinations (Sec 4.2-4.3). The per-run sweep CSVs (`val_threshold_sweep_*.csv`) exist under `results/outputs/`, but no standalone script systematically generates them. `evaluate_predictions.py` can produce these, but no Slurm or shell script orchestrates running it for all 231 configurations.

**C7. Missing `dissertation/img/` directory**

The LaTeX file references `img/` paths but figures are in `figures/main/`. Either add a symlink or update the LaTeX paths.

**C8. Base-pair statistics table (Table 3)**

No script generates the pair-rate and AU/GC/GU composition table. Values appear hand-computed.

---

## D. Orphan Code

Scripts in the repo with no direct dissertation counterpart:

| Script | Purpose | Verdict |
|--------|---------|---------|
| `code/analysis/aggregate_results.py` | Aggregate CPLfold results into summary CSVs | Supporting -- feeds Tables 15-17 |
| `code/analysis/alpha0_vs_best_stats.py` | Statistical significance of alpha=0 vs best | Supporting -- not in dissertation tables but could inform discussion |
| `code/folding_integration/run_cplfold_exp.py` | Run CPLfold experiments | Supporting -- used during folding integration |
| `code/folding_integration/generate_thresholded_base_pairs.py` | Generate thresholded base-pair files for TS0/NEW | Supporting -- intermediate step for CPLfold |
| `code/evaluation/build_summary_table.py` | Build unconstrained results summary | Redundant with `select_unconstrained_best_config.py` (and has broken MARCH1 ref) |
| `code/plotting/plot_probe_only.py` | Visualize probe-only unconstrained results | Orphan -- no specific dissertation figure matches |
| `code/plotting/plot_alpha0_vs_best.py` | Visualize alpha=0 vs best alpha | Orphan -- no specific dissertation figure matches |
| `code/plotting/plot_pair_combo_distribution.py` | Bar chart of pair combo rates | Supporting -- could have produced Table 7 visual (not in final manuscript) |

None of these are harmful to include. `build_summary_table.py` is the only one that is genuinely broken and redundant.
