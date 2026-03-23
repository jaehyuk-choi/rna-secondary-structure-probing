# march1 Figures

## CPLfold `α=0` vs. best `α`

### `fig1_grouped_bar.png`

Mean F1 by model across the four evaluation settings: `TS0 Vienna`, `TS0 Contrafold`, `NEW Vienna`, and `NEW Contrafold`.

- Gray bars: `α=0` baseline with no bonus
- Blue bars: validation-selected best `α`
- Larger gaps indicate stronger benefit from CPLfold bonuses

---

### `fig2_pct_improvement_heatmap.png`

Heatmap over `(model, partition)` showing `%Δ`, the relative improvement from `α=0` to the best `α`.

- Green indicates improvement
- Red indicates degradation
- Cell values are shown as percentages

---

### `fig3_best_alpha.png`

Validation-selected `α` values by model for ViennaRNA and Contrafold.

- Blue bars: ViennaRNA
- Red bars: Contrafold

---

### `fig4_significance_summary.png`

Paired t-test significance summary.

- `***`: `p < 0.001`
- `**`: `p < 0.005`
- `*`: `p < 0.01`
- `n.s.`: not significant

---

### `fig5_combined_panel.png`

Combined `2×2` panel containing selected bar plots, the `%Δ` heatmap, and the best-`α` summary.

---

### `fig6_line_vienna.png`

Line plot of mean F1 from TS0 to NEW under ViennaRNA, grouped by model.

---

### `fig7_line_contrafold.png`

Line plot of mean F1 from TS0 to NEW under Contrafold, grouped by model.

---

### `fig8_line_both.png`

Side-by-side ViennaRNA and Contrafold line plots.

---

## Validation (VL0) `α` sweep

### `vl0_alpha_sweep_vienna.png`

Mean F1 on VL0 as a function of `α` from `0` to `2` (step `0.02`) under ViennaRNA.

---

### `vl0_alpha_sweep_contrafold.png`

Mean F1 on VL0 as a function of `α` from `0` to `2` (step `0.02`) under Contrafold.

---

### `vl0_alpha_sweep_both.png`

Two-panel figure combining the ViennaRNA and Contrafold VL0 sweep curves.

---

## Probe-only figures

### `probe_f1_comparison.png`

TS0 vs. NEW F1 by model for unconstrained probe-only decoding.

---

### `probe_precision_recall.png`

Precision-recall scatter plot for TS0 under probe-only decoding.
