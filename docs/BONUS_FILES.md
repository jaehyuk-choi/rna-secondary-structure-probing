# 보너스 파일 생성 방법 (상세)

march1에서 생성되는 추가 분석/시각화 파일들의 **데이터 출처**, **계산 방식**, **실행 방법**을 정리합니다.

---

## 1. RoBERTa vs Others 통계 검정

### 출력 파일
- `figures/roberta_significance/roberta_vs_others_significance.csv`
- `figures/roberta_significance/roberta_vs_others_significance.md`

### 생성 스크립트
```bash
conda activate rna_probe   # scipy 필요
python march1/scripts/roberta_vs_others_statistical_test.py
```

### 데이터 출처
- **per-sequence F1**: `march1/data/ts_per_sequence_metrics.csv`, `march1/data/new_per_sequence_metrics.csv`
  - 없으면 `feb8/results_updated/summary/ts_per_sequence_metrics.csv`, `new_per_sequence_metrics.csv` 사용
- **Best config 필터**: `feb8/results_updated/summary/final_selected_config.csv`
  - per-sequence 행 중 (model, layer, k, threshold, decoding_mode)가 Best config와 일치하는 것만 사용

### 계산 방식
1. per-sequence 파일에서 Best config에 맞는 행만 로드
2. `seq_id` 기준으로 pivot: `{seq_id: {model: f1}}`
3. RoBERTa vs 각 모델(ERNIE, RNAFM, RiNALMo, One-hot, RNABERT)에 대해:
   - 같은 seq_id를 가진 sequence들의 (RoBERTa F1, other F1) 쌍 수집
   - **Paired Wilcoxon signed-rank test** (alternative='greater'): RoBERTa > other?
   - **Paired t-test** (alternative='greater')
4. mean_roberta, mean_other, mean_diff, p_wilcoxon, p_ttest, sig(***/**/*) 출력

### 해석
- **p=1.0** (RoBERTa vs ERNIE): RoBERTa가 ERNIE보다 **나쁘므로** H0(RoBERTa > other) 기각 불가. 즉, RoBERTa가 유의하게 더 좋다고 할 수 없음.
- **p≈0, sig=*****: RoBERTa가 해당 모델보다 유의하게 더 좋음 (Wilcoxon & t-test 모두 p<0.001).

### 의존성
- **per-sequence 데이터**: `compute_feb8_probe_only_metrics.py --per-sequence --per-sequence-dir march1/data` 로 먼저 생성 필요
- **scipy**: p-value 계산에 필요 (없으면 p=nan)

---

## 2. F1 by Length Boxplot

### 출력 파일
- `figures/length_boxplot/f1_by_length_boxplot.png`

### 생성 스크립트
```bash
python march1/scripts/plot_f1_by_length_boxplot.py
```

### 데이터 출처
- **per-sequence F1**: `march1/data/ts_per_sequence_metrics.csv`, `new_per_sequence_metrics.csv` (또는 feb8/summary)
- **Best config 필터**: `final_selected_config.csv`

### 계산 방식
1. per-sequence에서 Best config에 맞는 행만 로드
2. 각 sequence의 `length`로 구간 할당:
   - `<100`: 0 ≤ length < 100
   - `100-200`: 100 ≤ length < 200
   - `200-400`: 200 ≤ length < 400
   - `400+`: length ≥ 400
3. (partition, model, len_bin)별로 F1 리스트 수집
4. TS0 / NEW 각각 박스플롯: x축=len_bin, y축=F1, 모델별 색상

### 의존성
- per-sequence 데이터 (Best config 기준)

---

## 3. Layer-wise & k Comparison

### 출력 파일
- `figures/layer_k_comparison/layer_wise_val_f1.png`
- `figures/layer_k_comparison/k_comparison_val_f1.png`
- `figures/layer_k_comparison/layer_wise_table.md`
- `figures/layer_k_comparison/k_comparison_table.md`
- `data/layer_wise_val_f1.csv`
- `data/k_comparison_val_f1.csv`

### 생성 스크립트
```bash
python march1/scripts/plot_layer_and_k_comparison.py
```
*(스크립트가 없으면 아래 로직으로 별도 구현 필요)*

### 데이터 출처
- **all_runs_summary**: `feb8/results_updated/summary/all_runs_summary.csv`
  - 각 (model, layer, k, threshold, decoding_mode) 조합의 validation F1
- **Best config**: `feb8/results_updated/summary/final_selected_config.csv`

### 계산 방식

#### Layer-wise
- **고정**: Best config의 (k, threshold, decoding_mode)
- **변화**: layer만 0, 1, 2, … 로 sweep
- all_runs_summary에서 `(model, layer, best_k, best_threshold, best_decoding_mode)` 일치하는 행의 `f1` 사용
- 결과: layer별 Val F1 (line plot)

#### k comparison
- **고정**: Best config의 (layer, threshold, decoding_mode)
- **변화**: k만 32, 64, 128로 sweep
- all_runs_summary에서 `(model, best_layer, k, best_threshold, best_decoding_mode)` 일치하는 행의 `f1` 사용
- 결과: k별 Val F1 (grouped bar)

---

## 4. Per-sequence Metrics (선행 데이터)

### 출력 파일
- `march1/data/ts_per_sequence_metrics.csv`
- `march1/data/new_per_sequence_metrics.csv`

### 생성 스크립트
```bash
python feb8/scripts/evaluation/compute_feb8_probe_only_metrics.py \
  --per-sequence \
  --per-sequence-dir /projects/u6cg/jay/dissertations/march1/data
```

### 데이터 출처
- **Best config**: `feb8/results_updated/summary/final_selected_config.csv`
- **Checkpoint**: `feb8/results_updated/outputs/{model}/layer_{layer}/k_{k}/seed_42/best.pt`
- **Embeddings**: `data/embeddings/{model}/bpRNA/by_layer/layer_{layer}/{seq_id}.npy`
- **Ground truth**: `data/bpRNA.csv`, `data/bpRNA_splits.csv`

### 계산 방식
1. Best config 로드 (model별 layer, k, threshold, decoding_mode)
2. TS0 / NEW partition의 seq_id 목록 로드
3. 각 (model, seq_id)에 대해:
   - embedding 로드 → probe forward → threshold로 pair 예측
   - ground truth와 비교 → sequence-level F1, precision, recall 계산
4. (seq_id, length, f1, canonical_rate, model, layer, k, threshold, decoding_mode) 행으로 저장
   - **canonical_rate**: 예측 pair 중 Watson–Crick (AU, CG) 비율
   - **canonical_rate_wobble**: Watson–Crick + GU 비율 (decoding_mode가 canonical_wobble일 때)

### 특징
- Best config 기준으로만 계산 (다른 config sweep 없음)
- 실행 시간: 모델 6개 × (TS0 ~1305 + NEW ~5401) sequence → 약 30–40분
- nohup으로 백그라운드 실행 권장

---

## 5. 실행 순서 요약

```
1. Per-sequence 생성 (최초 1회, 또는 Best config 변경 시)
   python feb8/scripts/evaluation/compute_feb8_probe_only_metrics.py --per-sequence --per-sequence-dir march1/data

2. RoBERTa 통계 검정
   conda activate rna_probe
   python march1/scripts/roberta_vs_others_statistical_test.py

3. 길이별 박스플롯
   python march1/scripts/plot_f1_by_length_boxplot.py

4. Layer/k 비교 (plot_layer_and_k_comparison.py 존재 시)
   python march1/scripts/plot_layer_and_k_comparison.py
```

---

## 6. 파일 의존성 다이어그램

```
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
