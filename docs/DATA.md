# march1 Data Files

## `data/`

### `alpha0_vs_best_full.csv`

**Source**: copied from `feb25/alpha0_vs_best_full.csv`

**Contents**: comparison of CPLfold performance at `α=0` versus the validation-selected best `α`

**Columns**
- `partition`: `TS0 Vienna`, `TS0 Contrafold`, `NEW Vienna`, `NEW Contrafold`
- `model`: `ernie`, `roberta`, `rnafm`, `rinalmo`, `onehot`, `rnabert`
- `mean(α=0)`: mean F1 at `α=0`
- `mean(best)`: mean F1 at the validation-selected `α`
- `best_α`: validation-selected optimal `α`
- `%Δ`: `(mean(best) - mean(α=0)) / mean(α=0) × 100`
- `p<0.001`, `p<0.005`, `p<0.01`: significance flags
- `p_ttest`, `p_wilcoxon`: p-values

---

## Root-level CSV files

### `unconstrained_results_summary.csv`

**Contents**: probe-only summary on TS0 and NEW under unconstrained decoding

**Columns**: `model`, `layer`, `k`, `threshold`, `decoding_mode`, `ts0_f1`, `ts0_precision`, `ts0_recall`, `new_f1`, `new_precision`, `new_recall`

---

### `final_test_metrics.csv`

**Contents**: detailed probe-only metrics for TS0, including precision, recall, F1, TP, FP, FN, and `canonical_rate`

---

### `final_new_metrics.csv`

**Contents**: detailed probe-only metrics for NEW

---

### `final_selected_config_unconstrained.csv`

**Contents**: validation-selected best configuration for unconstrained decoding, including `model`, `layer`, `k`, `threshold`, and `decoding_mode`

---

### `final_*_wobble.csv`

**Contents**: evaluation results with GU wobble pairs included
