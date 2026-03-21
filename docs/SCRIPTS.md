# march1 스크립트 설명

## scripts/ (시각화)

### plot_alpha0_vs_best.py

**역할**: CPLfold α=0 vs best α 비교 결과 시각화

**입력**: `data/alpha0_vs_best_full.csv` (feb25에서 복사)

**출력** (figures/):
- `fig1_grouped_bar.png` - 4개 partition별 grouped bar (α=0 vs best α)
- `fig2_pct_improvement_heatmap.png` - %Δ 개선도 heatmap (model × partition)
- `fig3_best_alpha.png` - 모델별 Val-optimal α (Vienna vs Contrafold)
- `fig4_significance_summary.png` - 통계 유의성 요약 (***/**/*/n.s.)
- `fig5_combined_panel.png` - 2×2 종합 패널
- `fig6_line_vienna.png` - 꺽은선: Vienna TS0→NEW F1 (모델별)
- `fig7_line_contrafold.png` - 꺽은선: Contrafold TS0→NEW F1 (모델별)
- `fig8_line_both.png` - 꺽은선 Vienna+Contrafold 2개

**실행**:
```bash
cd /projects/u6cg/jay/dissertations/march1
python scripts/plot_alpha0_vs_best.py
```

---

### plot_probe_only.py

**역할**: Probe-only (unconstrained) 결과 시각화

**입력**: `unconstrained_results_summary.csv` 또는 `final_test_metrics.csv` + `final_new_metrics.csv`

**출력** (figures/):
- `probe_f1_comparison.png` - TS0 vs NEW F1 by model
- `probe_precision_recall.png` - Precision vs Recall scatter (TS0)

**실행**:
```bash
python scripts/plot_probe_only.py
```

---

### plot_vl0_alpha_sweep.py

**역할**: Validation (VL0) set — α sweep (0→2, step 0.02) F1 vs α 꺽은선 그래프

**입력**: feb23/results_vl0_feb8, results_vl0_contrafold_feb8

**출력** (figures/):
- `vl0_alpha_sweep_vienna.png` - Vienna
- `vl0_alpha_sweep_contrafold.png` - Contrafold
- `vl0_alpha_sweep_both.png` - 둘 다 2 subplot

---

### run_all_plots.sh

**역할**: 모든 시각화 스크립트 일괄 실행

**실행**:
```bash
bash scripts/run_all_plots.sh
```

---

## 루트 스크립트

### select_unconstrained_best_config.py

**역할**: Unconstrained decoding에서 best config 선택 (Val F1 기준)

**입력**: probe 결과, config 후보

**출력**: `final_selected_config_unconstrained.csv`

---

### compute_probe_only_with_wobble.py

**역할**: Wobble(GU) 포함 probe-only 평가

**출력**: `final_test_metrics_wobble.csv`, `final_new_metrics_wobble.csv`

---

### build_summary_table.py

**역할**: `final_test_metrics.csv` + `final_new_metrics.csv` → `unconstrained_results_summary.csv` 병합

**실행**: probe_only 완료 후
