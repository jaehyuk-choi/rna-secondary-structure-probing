# Experimental Workflow: RNA Secondary Structure Prediction with Language Model Probes

본 문서는 논문 작성용으로, 전체 실험 파이프라인의 상세한 방법론과 실행 절차를 기술한다.

**기준 디렉터리**: `dissertations/feb23/`

---

## 1. 개요

본 실험은 **사전학습된 RNA 언어 모델(RNA LM)의 임베딩**을 **구조 탐침(Structural Probe)**으로 분석하여, RNA 2차 구조 예측에 활용하는 파이프라인을 구성한다. 핵심 단계는 다음과 같다:

1. **데이터 분할**: bpRNA 데이터셋을 TR0(학습), VL0(검증), TS0(테스트), new(추가 테스트)로 분할
2. **탐침(Probe) 하이퍼파라미터 스윕 + 학습**: (layer, k, seed) 조합별로 probe를 학습하고 VL0에서 성능/손실을 기록
3. **검증(VL0) 기반 best config 선택**: 모델별로 (layer\*, k\*, decoding\_mode\*, threshold\*)를 VL0에서 선택
4. **Probe-only(=threshold sweep 스타일) 평가**: 선택된 best config로 TS0/NEW에서 folding 없이 직접 예측 성능을 측정(베이스라인)
5. **Base-pair 보너스 생성**: 선택된 best config의 threshold/decoding\_mode를 반영하여 (i, j, score) 보너스 파일 생성
6. **CPLfold α 스윕**: ViennaRNA/Contrafold 기반 CPLfold에 보너스를 주입하여 α(가중치)를 0.0~2.0 범위로 스윕
7. **검증 기반 최적 α 선택**: VL0에서 mean F1을 최대화하는 α를 선택하고, TS0/NEW에서 해당 α로 평가
8. **분석**: 길이 구간별 성능, α-F1 곡선, 통계 검정

---

## 2. 데이터

### 2.1 bpRNA 데이터셋

- **경로**: `dissertations/data/bpRNA.csv`, `dissertations/data/bpRNA_splits.csv`
- **열**: `id`, `sequence`, `base_pairs` (bpRNA.csv); `id`, `fold`, `partition`, `min_train_dist` (bpRNA_splits.csv)
- **base_pairs 형식**: `[[i1, j1], [i2, j2], ...]` (1-based 인덱스)

### 2.2 분할(Partition)

| Partition | 시퀀스 수 | 용도 |
|-----------|-----------|------|
| **TR0** | 10,814 | 탐침 학습 (training) |
| **VL0** | 1,300 | α 선택용 검증 (validation) |
| **TS0** | 1,305 | 테스트 (test) |
| **new** | 5,401 | 추가 테스트 (held-out test) |

- **Data leakage 방지**: α는 VL0에서만 선택하며, TS0/NEW는 α 선택에 사용하지 않음.
- **feb23 모델**: ernie, roberta, rnafm, rinalmo, onehot, **rnabert** (6개)

---

## 3. 구조 탐침(Structural Probe) 학습

### 3.1 모델 구조

**StructuralContactProbe** (저차원 투영):

- 입력: RNA LM 임베딩 \( h_i \in \mathbb{R}^D \)
- 투영 행렬: \( B \in \mathbb{R}^{D \times k} \) (k << D)
- 계산:
  \[
  z_i = B^\top h_i \in \mathbb{R}^k, \quad
  s_{ij} = z_i^\top z_j, \quad
  p_{ij} = \sigma(s_{ij})
  \]
- \( p_{ij} \): 위치 (i, j)가 base pair일 확률

#### (대안/비교) Bilinear probe

코드에는 full-rank bilinear probe도 포함되어 있다.

- \( W \in \mathbb{R}^{D \times D} \)
- \( s_{ij} = h_i^\top W h_j \)

구현: `dissertations/jan22/models/bilinear_probe_model.py`의 `BilinearContactProbe`, `StructuralContactProbe`.

### 3.2 학습 목적함수(연락지도/contact map 예측)

각 시퀀스 길이 \(L\)에 대해, 정답 base pair set을 contact map으로 변환한다.

- 정답 contact map: \(Y \in \{0,1\}^{L \times L}\)
  - \(Y_{ij}=1\) iff (i,j)가 정답 base pair (1-based)이고, 보통 \(i<j\) 상삼각만 사용
- probe가 출력하는 logits: \(S \in \mathbb{R}^{L \times L}\)
- 확률: \(P = \sigma(S)\)

학습은 일반적으로 binary cross entropy를 사용한다(코드 구현 세부는 run 스크립트에 따라 다를 수 있으나, 모델이 logits를 출력하고 `BCEWithLogitsLoss` 계열을 사용한다는 점은 일관됨).

\[
\mathcal{L}_{\text{BCE}}(S, Y)
= - \sum_{i<j} \left( Y_{ij}\log \sigma(S_{ij}) + (1-Y_{ij})\log (1-\sigma(S_{ij})) \right).
\]

패딩/가변 길이는 마스크로 제외한다(모델 forward에서 padding 위치 logits을 큰 음수로 마스킹 가능).

### 3.3 하이퍼파라미터 탐색

- **layer**: 임베딩 추출 레이어 (0~33)
- **k**: 투영 차원 (32, 64, 128)
- **seed**: 학습 시드(재현성/안정성 확인)
- **threshold / decoding_mode**: “학습” 하이퍼파라미터라기보다는, §3.4의 **디코딩(후처리) 선택 파라미터**로서 VL0에서 스윕하여 선택

### 3.4 검증(VL0) 기반 threshold sweep과 decoding_mode

본 파이프라인에서 threshold(및 decoding\_mode)는 “학습의 파라미터”가 아니라, **연락지도 확률 \(P_{ij}\)** 를 **base pair set**으로 변환하는 **디코딩/후처리 단계의 선택 파라미터**다.

핵심 이유:
- 단순히 \(P_{ij} \ge \tau\)인 모든 (i,j)를 채택하면 한 염기가 여러 염기와 결합하는 충돌이 발생한다.
- 따라서 본 프로젝트의 probe-only 예측은 **greedy global argmax** 방식으로 1-to-1 constraint(각 위치는 최대 1개 파트너)를 강제한다.

구현: `dissertations/jan22/utils/evaluation.py:prob_to_pairs()`

**Greedy decoding(개념)**

1. 후보 집합 \(C = \{(i,j): 1\le i<j\le L\}\)에서 시작한다.
2. decoding\_mode가 canonical이면 허용 후보만 남긴다.
   - `canonical_constrained`: AU/UA/CG/GC만 허용
   - `canonical_wobble`: AU/UA/CG/GC + GU/UG 허용
3. 아직 사용되지 않은 i와 j에 대해, \(P_{ij}\)가 최대인 (i,j)를 하나 선택한다.
4. 그 최대값이 \(\tau\) 이하이면 종료한다.
5. (i,j)를 예측 pair에 추가하고 i행과 j열(또는 i와 j의 사용)을 제거하여 충돌을 방지한다.

이 과정을 통해 threshold \(\tau\) 및 decoding\_mode에 따라 예측 pair set \(\hat{Y}(\tau,\text{mode})\)가 결정된다.

**Threshold sweep(VL0)**

각 모델 및 후보 (layer, k, seed) 설정에 대해, VL0에서 아래 grid로 \(\tau\)를 스윕한다.

\[
\tau \in \{0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95\}.
\]

산출물 예시:
- `.../val_threshold_sweep.csv`: 각 (decoding\_mode, threshold) 조합에 대한 precision/recall/F1 및 TP/FP/FN 집계
- `.../val_best_threshold.csv`: decoding\_mode별 best threshold(요약)

실제 파일 포맷(예시):
- `dissertations/jan22/results_updated/outputs/onehot/layer_0/k_32/seed_42/val_threshold_sweep.csv`
- `dissertations/jan22/results_updated/outputs/onehot/layer_0/k_32/seed_42/val_best_threshold.csv`

### 3.5 Best config 선택: (layer*, k*, decoding_mode*, threshold*)

모델 \(m\)마다(ernie/roberta/rnafm/rinalmo/onehot/rnabert 등), **서로 독립적으로** VL0만을 사용해 최종 설정을 선택한다. 선택은 크게 두 단계로 요약할 수 있다.

#### Step A: (layer, k, seed, best_epoch) 선택 (probe 학습 성능/손실 기반)

여러 후보 run(다양한 layer/k/seed)에 대해 TR0로 학습하고, VL0에서 검증 손실을 계산하여 학습 진행을 모니터링한다.

- 각 epoch \(e\)에서 VL0 검증 손실 \(\mathcal{L}_{\text{val}}^{(e)}\)를 계산하고,
  \[
  e^* = \arg\min_e \mathcal{L}_{\text{val}}^{(e)}
  \]
  로 best epoch를 정의한다(early stopping/모델 선택).
- 이후 후보 run 간 비교는 \(\min_e \mathcal{L}_{\text{val}}^{(e)}\) (=`best_val_loss`)를 사용한다.

- 선택 규칙(현재 저장된 config 기준): `min_val_loss`

요약 테이블(예시):
- `dissertations/jan22/results_updated/summary/all_runs_summary.csv`
  - `best_val_loss`, `best_epoch`, `checkpoint_path` 등 포함

#### Step B: decoding_mode 및 threshold 선택 (threshold sweep 기반)

Step A에서 선택된 run(=고정된 layer/k/seed/checkpoint)에 대해,

\[
(\text{mode}^*, \tau^*) = \arg\max_{\text{mode}, \tau} \operatorname{F1}_{\text{VL0}}(\text{mode}, \tau)
\]

로 정의되는 최적 decoding\_mode와 threshold를 선택한다.

이때 \(\operatorname{F1}_{\text{VL0}}\)는 VL0에서 TP/FP/FN을 집계한 후 계산한다(§7).

#### 최종 저장: final_selected_config.csv

모델별 최종 선택 결과는 아래 파일에 저장/재사용된다.

- `dissertations/jan22/results_updated/summary/final_selected_config.csv`
  - `selected_layer`, `selected_k`, `selected_seed`, `selected_best_epoch`
  - `selected_best_threshold`, `selected_decoding_mode`
  - `selection_rule`(현재 `min_val_loss`)

> 데이터 누수 방지  
> (layer,k,mode,threshold) 선택은 **VL0에서만** 수행하며, TS0/NEW는 선택에 사용하지 않는다.

### 3.6 Probe-only(=threshold sweep 스타일) 최종 평가: TS0/NEW

위에서 선택된 best config를 고정한 뒤, TS0와 NEW에서 **folding 없이** probe-only 예측 성능을 측정한다.

이는 CPLfold 결과와 별개로, “probe 자체가 주는 direct contact prediction 성능”을 베이스라인으로 제공하기 위함이다.

구현 예시(재계산/검증용):
- `dissertations/feb8/scripts/evaluation/compute_feb8_probe_only_metrics.py`
  - 선택된 config의 threshold/decoding\_mode로 greedy decoding을 수행하여 TS0/NEW의 precision/recall/F1 및 TP/FP/FN을 산출

### 3.7 선택 기준 및 최종 설정(요약)

- **선택 기준**: `min_val_loss` (VL0에서 검증 손실 최소)
- **최종 설정**: `jan22/results_updated/summary/final_selected_config.csv`

| Model | Layer | k | Threshold | Decoding |
|-------|-------|---|-----------|----------|
| ernie | 11 | 32 | 0.95 | unconstrained |
| roberta | 11 | 32 | 0.90 | unconstrained |
| rnafm | 11 | 32 | 0.90 | unconstrained |
| rinalmo | 25 | 64 | 0.65 | unconstrained |
| onehot | 0 | 32 | 0.70 | canonical_constrained |
| rnabert | 0 | 128 | 0.65 | unconstrained |

---

## 4. Base-pair 보너스 생성

### 4.1 알고리즘

1. 학습된 탐침 체크포인트에서 투영 행렬 B 로드
2. 각 시퀀스의 nucleotide 임베딩 \( h \in \mathbb{R}^{L \times D} \) 로드
3. 계산:
   \[
   z = h B \in \mathbb{R}^{L \times k}, \quad
   p_{ij} = \sigma(z_i^\top z_j)
   \]
4. \( p_{ij} \geq \text{threshold} \) 인 (i, j) 쌍만 추출 (i < j)
5. `base_pair_{model}_{seq_id}.txt` 형식으로 저장

> 중요: decoding_mode 반영  
> `decoding_mode`가 canonical 계열이면, 위의 “추출” 단계에서 (i,j)가 canonical mask를 통과한 경우만 보너스 후보로 포함한다.  
> 이는 probe-only 디코딩(§3.4)의 허용 후보 집합과 **동일한 제약**을 CPLfold 입력에도 적용하기 위함이다.

### 4.2 출력 형식

```
i	j	score
1	335	0.961317
3	8	0.958443
...
```

- 1-based 인덱스, 탭 구분
- score: \( p_{ij} \in [0, 1] \)

### 4.3 출력 경로 (feb23 기준)

- VL0: `feb23/vl0_base_pairs/{model}/base_pair_{model}_{seq_id}.txt`
- TS0: `feb23/ts_base_pairs/{model}/base_pair_{model}_{seq_id}.txt`
- new: `feb23/new_base_pairs/{model}/base_pair_{model}_{seq_id}.txt`

### 4.4 스크립트

- `jan22/scripts/generation/generate_base_pairs.py` (TS0, new)
- `feb23/scripts/generate_vl0_base_pairs.py` (VL0, rnabert 포함)

---

## 5. CPLfold α 스윕

### 5.1 CPLfold 개요

- **경로**: `CPLfold_inter/CPLfold_inter.py`
- **역할**: RNA 2차 구조 예측 시, base-pair 보너스 행렬을 에너지에 가산
- **스코어링**: base pair (i, j) 형성 시
  \[
  \text{score} \mathrel{+}= \text{bonus}_{ij} \times \alpha_{\text{scaled}}
  \]
- **α 스케일**:
  - ViennaRNA 모드: \( \alpha_{\text{scaled}} = \alpha \times 100 \)
  - Contrafold 모드: \( \alpha_{\text{scaled}} = \alpha \)

### 5.2 α 범위

- **범위**: 0.0 ~ 2.0
- **간격**: 0.02
- **점 개수**: 101
- **α=0**: 보너스 없음 (에너지 모델만 사용)
- **α>0**: 탐침 보너스 적용, α가 클수록 보너스 영향 증가

### 5.3 백엔드

| 백엔드 | 플래그 | 에너지 모델 |
|--------|--------|-------------|
| ViennaRNA | `--V` | Turner 37 파라미터 |
| Contrafold | (기본) | Contrafold 파라미터 |

### 5.4 실행 파이프라인

- **스크립트**: `feb23/scripts/run_split_pipeline.py`
- **입력**: partition(VL0/TS0/new), base_pairs 디렉터리, 모델 목록 (ernie, roberta, rnafm, rinalmo, onehot, rnabert)
- **처리**: (seq_id, model, α) 조합별로 CPLfold 실행 → dot-bracket 구조 예측 → ground truth와 비교

### 5.5 출력

- **파일**: `detailed_results_{model}.csv`
- **열**: `seq_id`, `model`, `alpha`, `threshold_used`, `f1`, `precision`, `recall`, `tp`, `fp`, `fn`, `predicted_count`, `energy`, `structure`

### 5.6 결과 디렉터리 (feb23)

| Partition | Vienna | Contrafold |
|-----------|--------|------------|
| VL0 | `feb23/results_vl0/` | `feb23/results_vl0_contrafold/` |
| TS0 | `feb23/results_ts0/` | `feb23/results_ts0_contrafold/` |
| new | `feb23/results_new/` | `feb23/results_new_contrafold/` |

---

## 6. 검증 기반 최적 α 선택

### 6.1 절차

1. **VL0 α 스윕**: VL0 시퀀스에 대해 α ∈ [0.0, 0.02, ..., 2.0] 각각 CPLfold 실행
2. **최적 α 선택**: 모델·백엔드별로 VL0 mean F1을 최대화하는 \( \alpha^* \) 선택
3. **TS0/NEW 평가**: 기존 `detailed_results_*.csv`에서 \( \alpha^* \)에 해당하는 행만 추출 (재실행 없음)

### 6.2 스크립트

- `feb8/scripts/validation_workflow/select_optimal_alpha_and_extract.py`
- feb23용: `--feb8-dir /projects/u6cg/jay/dissertations/feb23` 지정
- 출력: `feb23/validation_based_optimal/val_optimal_results.csv`
  - 열: `Backend`, `Model`, `Val_opt_alpha`, `Val_F1`, `TS0_F1_at_opt`, `NEW_F1_at_opt`
- **참고**: feb23은 rnabert 포함 6개 모델. 스크립트 기본값은 5개 모델이므로, rnabert 결과를 포함하려면 스크립트 수정 필요.

### 6.3 Data Leakage 방지

- α는 **VL0에서만** 선택
- TS0, new는 α 선택에 사용하지 않음

---

## 7. 평가 지표

### 7.1 정의

- **TP (True Positive)**: 예측 pair가 ground truth와 일치 (shift=1 허용)
- **FP (False Positive)**: 예측 pair가 ground truth에 없음
- **FN (False Negative)**: ground truth pair가 예측에 없음

- **Precision**: \( P = \frac{TP}{TP + FP} \)
- **Recall**: \( R = \frac{TP}{TP + FN} \)
- **F1**: \( F1 = \frac{2PR}{P + R} \)

### 7.2 Shift 허용

- `shift=1`: (i, j)가 (i±1, j) 또는 (i, j±1)과 일치하면 정답으로 인정
- 구현: `jan22/utils/evaluation.py`의 `compute_pair_metrics`, `precision_recall_f1`

### 7.3 평가 흐름

1. CPLfold 출력 dot-bracket → 예측 pair 리스트 변환
2. bpRNA ground truth pair 로드
3. shift 허용으로 TP/FP/FN 계산
4. Precision, Recall, F1 계산

---

## 8. 분석

### 8.1 길이 구간별 성능

- **구간**: <100 nt, 100–200 nt, 200–400 nt, ≥400 nt
- **지표**: α=0.0 (baseline), α=1.0 (고정), Optimal (per-sequence 최적 α)
- **통계**: paired t-test (α=0.0 vs Optimal), 유의수준 표시 (*, **, ***)

### 8.2 α-F1 곡선

- α별 mean F1, Precision, Recall 시각화
- 모델·partition·백엔드별 그래프

### 8.3 스크립트 및 출력 (feb23)

- `feb8/scripts/analysis/generate_length_tables_and_graphs.py` (feb23 기준 경로 사용 시)
- 출력: `feb23/analysis_outputs/length_tables/`
  - `length_table_{PARTITION}_{BACKEND}.txt`, `summary_table.csv`

---

## 9. 전체 파이프라인 요약

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 데이터 준비                                                           │
│    bpRNA_splits.csv → TR0, VL0, TS0, new                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. probe 하이퍼파라미터 스윕 + 학습 (jan22)                              │
│    StructuralContactProbe: z_i = B^T h_i, p_ij = σ(z_i^T z_j)           │
│    하이퍼파라미터: layer, k, seed                                        │
│    검증(VL0): val_loss 기록                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. best config 선택 (VL0만 사용)                                         │
│    (layer*,k*,epoch*) = argmin val_loss                                  │
│    (mode*,thr*) = argmax_{mode,thr} VL0 F1 (threshold sweep + greedy)    │
│    저장: final_selected_config.csv                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. probe-only 평가(베이스라인)                                           │
│    best config 고정 후 TS0/NEW에서 folding 없이 F1 측정                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. Base-pair 보너스 생성                                                 │
│    임베딩 + B → p_ij ≥ thr* (+ canonical mask) → base_pair_*.txt         │
│    출력: vl0_base_pairs/, ts_base_pairs/, new_base_pairs/                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. CPLfold α 스윕                                                        │
│    α ∈ [0.0, 0.02, ..., 2.0] × (VL0, TS0, new) × (Vienna, Contrafold)   │
│    score += bonus_ij × α_scaled                                          │
│    출력: detailed_results_{model}.csv                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. 검증 기반 최적 α                                                      │
│    VL0에서 α* = argmax mean_F1                                          │
│    TS0/NEW에서 α*로 F1 추출                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 8. 분석                                                                  │
│    길이 구간별 F1, α-F1 곡선, 통계 검정                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. 주요 파일 경로 (feb23 기준)

| 항목 | 경로 |
|------|------|
| **기준 디렉터리** | `dissertations/feb23/` |
| 데이터 분할 | `dissertations/data/bpRNA_splits.csv` |
| bpRNA 시퀀스/구조 | `dissertations/data/bpRNA.csv` |
| 임베딩 | `dissertations/data/embeddings/{model}/bpRNA/by_layer/layer_{k}/` |
| 탐침 설정 | `dissertations/jan22/results_updated/summary/final_selected_config.csv` |
| 탐침 체크포인트 | `dissertations/jan22/results_updated/outputs/{model}/layer_{k}/k_{d}/seed_42/best.pt` |
| VL0 Base-pair 생성 | `feb23/scripts/generate_vl0_base_pairs.py` |
| CPLfold 파이프라인 | `feb23/scripts/run_split_pipeline.py` |
| CPLfold 실행기 | `CPLfold_inter/CPLfold_inter.py` |
| 최적 α 선택 | `feb8/scripts/validation_workflow/select_optimal_alpha_and_extract.py` (--feb8-dir feb23) |
| 평가 유틸 | `jan22/utils/evaluation.py` |
| 분석 출력 | `feb23/analysis_outputs/length_tables/` |

---

## 11. Slurm 실행 예시 (feb23)

```bash
cd /projects/u6cg/jay/dissertations/feb23

# VL0 base-pair 생성 (ernie, roberta, rnafm, rinalmo, onehot, rnabert)
sbatch slurm/run_vl0_base_pairs.sh

# CPLfold α 스윕 (Vienna)
sbatch slurm/run_vl0_ernie.sh
sbatch slurm/run_vl0_roberta.sh
sbatch slurm/run_vl0_rnafm.sh
sbatch slurm/run_vl0_rinalmo.sh
sbatch slurm/run_vl0_onehot.sh
sbatch slurm/run_vl0_rnabert.sh

# CPLfold α 스윕 (Contrafold)
sbatch slurm/run_vl0_ernie_contrafold.sh
sbatch slurm/run_vl0_roberta_contrafold.sh
# ... (모델별)

# 최적 α 선택 및 TS0/NEW 추출 (feb23 기준)
python3 /projects/u6cg/jay/dissertations/feb8/scripts/validation_workflow/select_optimal_alpha_and_extract.py \
  --feb8-dir /projects/u6cg/jay/dissertations/feb23
```
