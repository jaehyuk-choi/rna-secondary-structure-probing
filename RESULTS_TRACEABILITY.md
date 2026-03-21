# RESULTS TRACEABILITY

Maps every major dissertation table, figure, and key numerical claim to supporting files in this package.

---

## Tables

### Table: probe_unconstrained_full (Section 5.1)
**Content:** Probe-only F1/precision/recall on TS0 and NEW under unconstrained decoding.
- **Result file:** `results/metrics/unconstrained_results_summary.csv`
- **Config file:** `configs/final_selected_config_unconstrained.csv`
- **Upstream script:** `code/evaluation/select_unconstrained_best_config.py`
- **Reproducibility:** Fully supported

### Table: canonical_ts_new (Section 5.2)
**Content:** Canonical base-pair rate (WC and CAN) under unconstrained decoding.
- **Result file:** `results/tables/canonical_rate_table_with_baseline.csv`, `.tex`
- **Upstream script:** `code/analysis/build_canonical_rate_table_with_baseline.py`
- **Supporting data:** `results/tables/wc_gu_rate_all_pairs.csv`
- **Reproducibility:** Fully supported

### Table: pair_distribution (Section 5.2)
**Content:** Distribution of predicted nucleotide pair types for one-hot and RiNALMo.
- **Result files:** `results/tables/onehot_pair_combo_distribution.csv`, `results/tables/rinalmo_pair_combo_distribution.csv`
- **Upstream script:** `code/analysis/onehot_pair_combo_distribution.py`
- **Reproducibility:** Fully supported

### Table: best_config_val (Section 5.3)
**Content:** Validation-selected best configuration for each model (layer, k, tau, mode).
- **Config file:** `configs/best_config_val_f1.csv`
- **Upstream pipeline:** `code/probe_training/run_split_pipeline.py` → `code/evaluation/`
- **Reproducibility:** Fully supported

### Table: probe_best_full (Section 5.3)
**Content:** Probe-only F1/precision/recall on TS0 and NEW under best configuration.
- **Result files:** `results/metrics/final_test_metrics.csv`, `results/metrics/final_new_metrics.csv`
- **Config file:** `configs/best_config_val_f1.csv`
- **Reproducibility:** Fully supported

### Table: qualitative_example (Section 5.3)
**Content:** Per-sequence contact recovery for RNA ID 21498.
- **Result file:** `results/tables/probe_heatmap_seq_f1_and_gt.txt`
- **Upstream script:** `code/analysis/compute_seq_f1_and_gt.py`
- **Reproducibility:** Fully supported

### Table: probe_unconstrained_vs_best (Section 5.3.1)
**Content:** Unconstrained vs best-configuration F1 comparison.
- **Result file:** `results/metrics/probe_unconstrained_vs_best_comparison.csv`
- **Reproducibility:** Fully supported

### Table: median_f1_by_length (Section 5.3.2)
**Content:** Median per-sequence F1 by length bin.
- **Derived from:** `results/per_sequence/ts_per_sequence_metrics.csv`, `results/per_sequence/new_per_sequence_metrics.csv`
- **Upstream script:** `code/analysis/analyze_by_length.py`
- **Reproducibility:** Fully supported

### Table: roberta_pairwise (Section 5.3.3)
**Content:** Paired significance tests (RoBERTa vs others).
- **Result file:** `results/statistics/roberta_vs_others_significance.csv`
- **Upstream script:** `code/analysis/roberta_vs_others_statistical_test.py`
- **Reproducibility:** Fully supported

### Table: k_val_f1 (Section 5.4.2)
**Content:** Validation F1 by probe rank k.
- **Result file:** `results/sweeps/k_comparison_val_f1.csv`
- **Reproducibility:** Fully supported

### Table: alpha_vienna (Section 5.5.2)
**Content:** Held-out F1 under ViennaRNA at alpha=0 vs alpha*.
- **Result file:** `results/folding/alpha0_vs_best_full.csv`
- **Upstream script:** `code/analysis/compare_alpha0_vs_best.py`
- **Reproducibility:** Fully supported

### Table: alpha_contrafold (Section 5.5.2)
**Content:** Held-out F1 under CONTRAfold at alpha=0 vs alpha*.
- **Result file:** `results/folding/alpha0_vs_best_full.csv`
- **Reproducibility:** Fully supported

### Table: alpha_vl0_summary (Appendix)
**Content:** VL0 alpha sweep summary for both backends.
- **Result file:** `results/sweeps/vl0_alpha_sweep_both.csv`
- **Reproducibility:** Fully supported

### Tables: appendix_f1_by_length_ts0, appendix_f1_by_length_new (Appendix)
**Content:** Mean±std F1 by length bin on TS0 and NEW.
- **Derived from:** `results/per_sequence/ts_per_sequence_metrics.csv`, `results/per_sequence/new_per_sequence_metrics.csv`
- **Reproducibility:** Fully supported

---

## Figures

### Figure: rna_structure (Section 2)
**Content:** RNA secondary structure illustration.
- **File:** NOT FOUND in repository (see REVIEW_NOTES.md)
- **LaTeX ref:** `img/rna_structure.png`
- **Status:** MISSING — must be provided manually

### Figure: pipeline_figure (Section 3)
**Content:** High-level pipeline overview.
- **File:** NOT FOUND in repository (see REVIEW_NOTES.md)
- **LaTeX ref:** `img/pipeline_figure.pdf`
- **Status:** MISSING — must be provided manually

### Figure: max_one_decoding (Section 3)
**Content:** Greedy max-one decoding illustration.
- **File:** NOT FOUND in repository (see REVIEW_NOTES.md)
- **LaTeX ref:** `img/max 1.pdf`
- **Status:** MISSING — must be provided manually

### Figure: bpRNA_length_boxplot (Section 4)
**Content:** Sequence length distribution across dataset partitions.
- **File:** `figures/main/bpRNA_length_boxplot.png`
- **Plotting script:** `code/preprocessing/plot_bprna_length_boxplot.py`
- **Status:** Complete

### Figure: probe_heatmaps_selected (Section 5.3)
**Content:** Ground-truth and probe-derived pairwise maps for RNA ID 21498 (3 models).
- **File:** `figures/main/probe_contact_heatmaps_selected_updated.png`
- **Plotting script:** `code/plotting/plot_probe_contact_heatmaps.py`
- **Status:** Complete

### Figure: probe_unconstrained_vs_best (Section 5.3.1)
**Content:** Grouped bar chart comparing unconstrained vs best F1.
- **File:** `figures/main/fig1_grouped_bar_f1.png`
- **Plotting script:** `code/plotting/plot_probe_comparison_grouped_bar.py`
- **Data:** `results/metrics/probe_unconstrained_vs_best_comparison.csv`
- **Status:** Complete

### Figure: f1_by_length (Section 5.3.2)
**Content:** Per-sequence F1 boxplot by length bin.
- **File:** `figures/main/f1_by_length_boxplot.png`
- **Plotting script:** `code/plotting/plot_f1_by_length_boxplot.py`
- **Data:** `results/per_sequence/ts_per_sequence_metrics.csv`, `new_per_sequence_metrics.csv`
- **Status:** Complete

### Figure: val_f1_by_layer (Section 5.4.1)
**Content:** Validation F1 by transformer layer.
- **File:** `figures/main/layer_wise_val_f1.png`
- **Plotting script:** `code/plotting/plot_layer_wise_val_f1.py`
- **Data:** `results/sweeps/layer_wise_val_f1.csv`
- **Status:** Complete

### Figure: val_f1_by_k (Section 5.4.2)
**Content:** Validation F1 by probe rank k.
- **File:** `figures/main/k_comparison_val_f1_by_model.png`
- **Plotting script:** `code/plotting/plot_k_comparison_by_model.py`
- **Data:** `results/sweeps/k_comparison_val_f1.csv`
- **Status:** Complete

### Figure: vl0_threshold_sweep (Section 5.4.3)
**Content:** VL0 F1 vs decoding threshold tau.
- **File:** `figures/main/vl0_threshold_sweep.png`
- **Plotting script:** `code/plotting/plot_vl0_threshold_sweep.py`
- **Status:** Complete

### Figure: vl0_alpha_sweep (Section 5.5.1)
**Content:** VL0 mean F1 vs CPLfold prior weight alpha (two subfigures).
- **Files:** `figures/main/vl0_alpha_sweep_vienna.png`, `figures/main/vl0_alpha_sweep_contrafold.png`
- **Plotting script:** `code/plotting/plot_vl0_alpha_sweep.py`
- **Data:** `results/sweeps/vl0_alpha_sweep_both.csv`
- **Status:** Complete

### Figure: probe_heatmaps_bprna_rfam_21498 (Appendix)
**Content:** Full 6-model structural probing heatmaps.
- **File:** `figures/appendix/probe_contact_heatmaps.png`
- **Plotting script:** `code/plotting/plot_probe_contact_heatmaps.py`
- **Status:** Complete

---

## Key Numerical Claims

| Claim | Source | Status |
|---|---|---|
| ERNIE-RNA F1 = 0.51 on TS0 (best config) | `results/metrics/final_test_metrics.csv` | Traceable |
| ERNIE-RNA F1 = 0.46 on NEW (best config) | `results/metrics/final_new_metrics.csv` | Traceable |
| RoBERTa outperforms RNA-FM, RiNALMo, RNABERT | `results/metrics/final_test_metrics.csv`, `final_new_metrics.csv` | Traceable |
| RoBERTa advantage p < 10^-6 | `results/statistics/roberta_vs_others_significance.csv` | Traceable |
| ERNIE canonical rate >71% | `results/tables/canonical_rate_table_with_baseline.csv` | Traceable |
| RoBERTa +82.7% relative gain from constraints | `results/metrics/probe_unconstrained_vs_best_comparison.csv` | Traceable |
| ERNIE +21.8% under CONTRAfold | `results/folding/alpha0_vs_best_full.csv` | Traceable |
| CONTRAfold yields consistent improvements | `results/folding/alpha0_vs_best_full.csv` | Traceable |
| ViennaRNA gains confined to ERNIE | `results/folding/alpha0_vs_best_full.csv` | Traceable |
| RNA-FM degrades on NEW under ViennaRNA | `results/folding/alpha0_vs_best_full.csv` | Traceable |
