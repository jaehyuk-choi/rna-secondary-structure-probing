# march1 Scripts

## `scripts/` for plotting

### `plot_alpha0_vs_best.py`

**Purpose**: visualize CPLfold results at `α=0` versus the validation-selected best `α`

**Input**: `data/alpha0_vs_best_full.csv` copied from `feb25`

**Outputs** in `figures/`
- `fig1_grouped_bar.png`: grouped bars by partition
- `fig2_pct_improvement_heatmap.png`: `%Δ` heatmap over `(model, partition)`
- `fig3_best_alpha.png`: best `α` by model and backend
- `fig4_significance_summary.png`: significance summary
- `fig5_combined_panel.png`: combined summary panel
- `fig6_line_vienna.png`: ViennaRNA TS0→NEW line plot
- `fig7_line_contrafold.png`: Contrafold TS0→NEW line plot
- `fig8_line_both.png`: combined ViennaRNA and Contrafold line plots

**Run**
```bash
cd /projects/u6cg/jay/dissertations/march1
python scripts/plot_alpha0_vs_best.py
```

---

### `plot_probe_only.py`

**Purpose**: visualize probe-only results under unconstrained decoding

**Input**: `unconstrained_results_summary.csv` or the pair `final_test_metrics.csv` and `final_new_metrics.csv`

**Outputs** in `figures/`
- `probe_f1_comparison.png`: TS0 vs. NEW F1 by model
- `probe_precision_recall.png`: TS0 precision-recall scatter

**Run**
```bash
python scripts/plot_probe_only.py
```

---

### `plot_vl0_alpha_sweep.py`

**Purpose**: plot validation-set F1 against `α` over the `0→2` sweep with step `0.02`

**Input**
- `feb23/results_vl0_feb8`
- `results_vl0_contrafold_feb8`

**Outputs** in `figures/`
- `vl0_alpha_sweep_vienna.png`
- `vl0_alpha_sweep_contrafold.png`
- `vl0_alpha_sweep_both.png`

---

### `run_all_plots.sh`

**Purpose**: run all plotting scripts in batch

**Run**
```bash
bash scripts/run_all_plots.sh
```

---

## Root-level scripts

### `select_unconstrained_best_config.py`

**Purpose**: select the best configuration for unconstrained decoding using validation F1

**Input**: probe results and configuration candidates

**Output**: `final_selected_config_unconstrained.csv`

---

### `compute_probe_only_with_wobble.py`

**Purpose**: evaluate probe-only predictions with GU wobble pairs included

**Outputs**
- `final_test_metrics_wobble.csv`
- `final_new_metrics_wobble.csv`

---

### `build_summary_table.py`

**Purpose**: merge `final_test_metrics.csv` and `final_new_metrics.csv` into `unconstrained_results_summary.csv`

**Run**: after probe-only evaluation completes
