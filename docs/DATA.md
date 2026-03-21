# march1 데이터 파일 설명

## data/

### alpha0_vs_best_full.csv

**출처**: feb25/alpha0_vs_best_full.csv 복사

**내용**: CPLfold α=0 vs best α 비교 (partition, model, mean F1, best_α, %Δ, p-value)

**컬럼**:
- partition: TS0 Vienna, TS0 Contrafold, NEW Vienna, NEW Contrafold
- model: ernie, roberta, rnafm, rinalmo, onehot, rnabert
- mean(α=0): α=0에서의 평균 F1
- mean(best): Val-optimal α에서의 평균 F1
- best_α: Val에서 선택된 optimal α
- %Δ: (mean(best) - mean(α=0)) / mean(α=0) × 100
- p<0.001, p<0.005, p<0.01: 유의성 플래그
- p_ttest, p_wilcoxon: p-value

---

## 루트 CSV

### unconstrained_results_summary.csv

**내용**: Probe-only (unconstrained) TS0/NEW 요약

**컬럼**: model, layer, k, threshold, decoding_mode, ts0_f1, ts0_precision, ts0_recall, new_f1, new_precision, new_recall

---

### final_test_metrics.csv

**내용**: Probe-only TS0 partition 상세 (precision, recall, F1, TP, FP, FN, canonical_rate)

---

### final_new_metrics.csv

**내용**: Probe-only NEW partition 상세

---

### final_selected_config_unconstrained.csv

**내용**: Unconstrained decoding에서 선택된 best config (model, layer, k, threshold, decoding_mode)

---

### final_*_wobble.csv

**내용**: Wobble(GU) 포함 평가 결과
