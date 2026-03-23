# Bonus Files: Generation Details

This document records the data sources, computation logic, and execution steps for the additional analysis and visualization files generated in `march1`.

---

## 1. RoBERTa vs. Others Significance Tests

### Output files
- `figures/roberta_significance/roberta_vs_others_significance.csv`
- `figures/roberta_significance/roberta_vs_others_significance.md`

### Script
```bash
conda activate rna_probe   # scipy required
python march1/scripts/roberta_vs_others_statistical_test.py
```

### Data sources
- **Per-sequence F1**: `march1/data/ts_per_sequence_metrics.csv`, `march1/data/new_per_sequence_metrics.csv`
  - If unavailable, fall back to `feb8/results_updated/summary/ts_per_sequence_metrics.csv` and `new_per_sequence_metrics.csv`
- **Best-config filter**: `feb8/results_updated/summary/final_selected_config.csv`
  - Keep only rows whose `(model, layer, k, threshold, decoding_mode)` match the selected configuration

### Computation
1. Load only best-config rows from the per-sequence files.
2. Pivot by `seq_id` to obtain `{seq_id: {model: f1}}`.
3. For RoBERTa vs. each comparison model (ERNIE, RNAFM, RiNALMo, One-hot, RNABERT):
   - collect paired `(RoBERTa F1, other F1)` values for shared `seq_id`
   - run a paired Wilcoxon signed-rank test with `alternative='greater'`
   - run a paired t-test with `alternative='greater'`
4. Report `mean_roberta`, `mean_other`, `mean_diff`, `p_wilcoxon`, `p_ttest`, and `sig`

### Interpretation
- **p=1.0** (RoBERTa vs. ERNIE): the one-sided hypothesis `RoBERTa > other` is not supported.
- **p≈0, sig=***:** RoBERTa significantly outperforms the comparison model under both tests with `p < 0.001`.

### Dependencies
- **Per-sequence metrics** must be generated first with `compute_feb8_probe_only_metrics.py --per-sequence --per-sequence-dir march1/data`
- **scipy** is required for p-value computation; otherwise p-values are written as `nan`

---

## 2. F1 by Length Boxplot

### Output file
- `figures/length_boxplot/f1_by_length_boxplot.png`

### Script
```bash
python march1/scripts/plot_f1_by_length_boxplot.py
```

### Data sources
- **Per-sequence F1**: `march1/data/ts_per_sequence_metrics.csv`, `new_per_sequence_metrics.csv` (or the corresponding `feb8/summary` files)
- **Best-config filter**: `final_selected_config.csv`

### Computation
1. Keep only best-config rows from the per-sequence tables.
2. Assign each sequence to a length bin:
   - `<100`: `0 <= length < 100`
   - `100-200`: `100 <= length < 200`
   - `200-400`: `200 <= length < 400`
   - `400+`: `length >= 400`
3. Collect F1 values by `(partition, model, len_bin)`.
4. Draw separate TS0 and NEW boxplots with `len_bin` on the x-axis and F1 on the y-axis.

### Dependency
- Per-sequence metrics generated under the best selected configuration

---

## 3. Layer-wise and k Comparison

### Output files
- `figures/layer_k_comparison/layer_wise_val_f1.png`
- `figures/layer_k_comparison/k_comparison_val_f1.png`
- `figures/layer_k_comparison/layer_wise_table.md`
- `figures/layer_k_comparison/k_comparison_table.md`
- `data/layer_wise_val_f1.csv`
- `data/k_comparison_val_f1.csv`

### Script
```bash
python march1/scripts/plot_layer_and_k_comparison.py
```
If this script is absent, the figures must be reproduced from the logic below.

### Data sources
- **all_runs_summary**: `feb8/results_updated/summary/all_runs_summary.csv`
  - Validation F1 for each `(model, layer, k, threshold, decoding_mode)` combination
- **Best config**: `feb8/results_updated/summary/final_selected_config.csv`

### Computation

#### Layer-wise
- Hold `(k, threshold, decoding_mode)` fixed at the best configuration.
- Sweep only `layer`.
- Use the `f1` value from rows matching `(model, layer, best_k, best_threshold, best_decoding_mode)`.
- Plot validation F1 by layer.

#### k comparison
- Hold `(layer, threshold, decoding_mode)` fixed at the best configuration.
- Sweep `k` over `32, 64, 128`.
- Use the `f1` value from rows matching `(model, best_layer, k, best_threshold, best_decoding_mode)`.
- Plot validation F1 by `k`.

---

## 4. Per-sequence Metrics

### Output files
- `march1/data/ts_per_sequence_metrics.csv`
- `march1/data/new_per_sequence_metrics.csv`

### Script
```bash
python feb8/scripts/evaluation/compute_feb8_probe_only_metrics.py \
  --per-sequence \
  --per-sequence-dir /projects/u6cg/jay/dissertations/march1/data
```

### Data sources
- **Best config**: `feb8/results_updated/summary/final_selected_config.csv`
- **Checkpoint**: `feb8/results_updated/outputs/{model}/layer_{layer}/k_{k}/seed_42/best.pt`
- **Embeddings**: `data/embeddings/{model}/bpRNA/by_layer/layer_{layer}/{seq_id}.npy`
- **Ground truth**: `data/bpRNA.csv`, `data/bpRNA_splits.csv`

### Computation
1. Load the selected `(layer, k, threshold, decoding_mode)` for each model.
2. Load the `seq_id` lists for TS0 and NEW.
3. For each `(model, seq_id)` pair:
   - load embeddings
   - run the probe
   - decode pairs with the selected threshold
   - compare predictions against ground truth
4. Save rows containing `(seq_id, length, f1, canonical_rate, model, layer, k, threshold, decoding_mode)`.

### Notes
- Metrics are computed only for the selected configuration; no additional sweep is performed.
- Typical runtime is roughly 30 to 40 minutes for 6 models over TS0 and NEW.
- Background execution with `nohup` is recommended.

---

## 5. Recommended Execution Order

```text
1. Generate per-sequence metrics
   python feb8/scripts/evaluation/compute_feb8_probe_only_metrics.py --per-sequence --per-sequence-dir march1/data

2. Run the RoBERTa significance tests
   conda activate rna_probe
   python march1/scripts/roberta_vs_others_statistical_test.py

3. Plot F1 by length
   python march1/scripts/plot_f1_by_length_boxplot.py

4. Plot layer-wise and k comparisons
   python march1/scripts/plot_layer_and_k_comparison.py
```

---

## 6. File Dependency Diagram

```text
final_selected_config.csv (feb8)
        │
        ├──► all_runs_summary.csv ──► plot_layer_and_k_comparison.py ──► layer_wise_*, k_comparison_*
        │
        └──► compute_feb8_probe_only_metrics.py --per-sequence
                    │
                    └──► ts_per_sequence_metrics.csv, new_per_sequence_metrics.csv
                                    │
                                    ├──► roberta_vs_others_statistical_test.py ──► roberta_significance/*
                                    │
                                    └──► plot_f1_by_length_boxplot.py ──► f1_by_length_boxplot.png
```
