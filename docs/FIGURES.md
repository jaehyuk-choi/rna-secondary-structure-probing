# march1 시각화 (figures/) 설명

## CPLfold α=0 vs best α

### fig1_grouped_bar.png

**내용**: 4개 partition (TS0 Vienna, TS0 Contrafold, NEW Vienna, NEW Contrafold)별로, 모델별 mean F1 비교

- **회색 막대**: α=0 (bonus 없음)
- **파란 막대**: best α (Val-optimal)
- **해석**: 막대 차이가 클수록 CPLfold bonus 효과가 큼

---

### fig2_pct_improvement_heatmap.png

**내용**: model × partition heatmap, %Δ (best α 대비 α=0 개선율)

- **색**: 녹색=개선, 빨강=악화
- **값**: +20.7% 등 퍼센트 표시
- **해석**: ernie, rnafm이 TS0/NEW에서 큰 개선

---

### fig3_best_alpha.png

**내용**: 모델별 Val-optimal α (Vienna vs Contrafold)

- **파란 막대**: Vienna
- **빨간 막대**: Contrafold
- **해석**: ernie, rnafm은 α≈1~2, rinalmo Vienna는 α=0

---

### fig4_significance_summary.png

**내용**: 통계 유의성 (paired t-test)

- ***: p<0.001
- **: p<0.005
- *: p<0.01
- n.s.: not significant

---

### fig5_combined_panel.png

**내용**: 2×2 종합 패널 (TS0 Vienna bar, NEW Vienna bar, %Δ heatmap, best α bar)

---

### fig6_line_vienna.png

**내용**: 꺽은선 그래프 — Vienna, TS0 → NEW mean F1 (모델별)

---

### fig7_line_contrafold.png

**내용**: 꺽은선 그래프 — Contrafold, TS0 → NEW mean F1 (모델별)

---

### fig8_line_both.png

**내용**: Vienna + Contrafold 꺽은선 그래프 2개 (나란히)

---

## Validation (VL0) α sweep

### vl0_alpha_sweep_vienna.png

**내용**: VL0 Vienna — α 0~2 (step 0.02)에 따른 mean F1 꺽은선 (모델별)

---

### vl0_alpha_sweep_contrafold.png

**내용**: VL0 Contrafold — α 0~2에 따른 mean F1 꺽은선 (모델별)

---

### vl0_alpha_sweep_both.png

**내용**: Vienna + Contrafold 2개 subplot

---

## Probe-only

### probe_f1_comparison.png

**내용**: TS0 vs NEW F1 by model (unconstrained probe-only)

---

### probe_precision_recall.png

**내용**: Precision vs Recall scatter (TS0, probe-only)
