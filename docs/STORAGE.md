# march1 저장 구조 및 데이터 흐름

march1 폴더의 파일 저장 방식과 파이프라인을 정리한 문서입니다.

---

## 1. 디렉터리 구조

```
march1/
├── README.md
├── docs/                    # 문서
│   ├── STORAGE.md           # 이 문서 (저장 구조)
│   ├── FILE_INDEX.md        # 전체 파일 인덱스
│   ├── DATA.md              # 데이터 파일 설명
│   ├── SCRIPTS.md           # 스크립트 설명
│   └── FIGURES.md           # 그림 설명
│
├── data/                    # 정리된 입력 데이터 (외부에서 복사/생성)
│   └── alpha0_vs_best_full.csv
│
├── figures/                 # 시각화 결과 (PNG)
│   ├── fig1_grouped_bar.png
│   ├── fig2_pct_improvement_heatmap.png
│   ├── ... (CPLfold α=0 vs best)
│   ├── vl0_alpha_sweep_*.png
│   └── probe_*.png
│
├── scripts/                 # 시각화 스크립트
│   ├── plot_alpha0_vs_best.py
│   ├── plot_probe_only.py
│   ├── plot_vl0_alpha_sweep.py
│   └── run_all_plots.sh
│
├── [Config & Probe 파이프라인]
│   ├── final_selected_config_unconstrained.csv   # Config 선택 결과
│   ├── final_test_metrics.csv                    # Probe TS0 결과
│   ├── final_new_metrics.csv                     # Probe NEW 결과
│   ├── final_test_metrics_wobble.csv             # Probe TS0 (wobble)
│   ├── final_new_metrics_wobble.csv              # Probe NEW (wobble)
│   └── unconstrained_results_summary.csv         # TS0+NEW 병합 요약
│
├── [실행 스크립트]
│   ├── run_probe_only.sh
│   └── run_wobble_nohup.sh
│
├── [루트 스크립트]
│   ├── select_unconstrained_best_config.py
│   ├── compute_probe_only_with_wobble.py
│   └── build_summary_table.py
│
├── [로그/임시]
│   ├── probe_only_progress.log
│   ├── probe_only_wobble_progress.log
│   └── nohup_wobble.out
│
└── [기타]
    ├── unconstrained_results_table.md
    └── ...
```

---

## 2. 데이터 저장 흐름 (파이프라인)

### 2.1 Config 선택 → Probe 평가 (Unconstrained)

```
feb8/results_updated/outputs/     (checkpoint, val_threshold_sweep_*.csv)
        │
        ▼  select_unconstrained_best_config.py
final_selected_config_unconstrained.csv   ← model, layer, k, threshold, decoding_mode
        │
        ▼  compute_feb8_probe_only_metrics.py (feb8/scripts/evaluation/)
        │   --config-csv march1/final_selected_config_unconstrained.csv
        │   --output-dir march1
        │
        ├──► final_test_metrics.csv   (TS0 partition)
        └──► final_new_metrics.csv    (NEW partition)
        │
        ▼  build_summary_table.py
unconstrained_results_summary.csv   ← ts0_f1, new_f1, precision, recall 병합
```

### 2.2 Probe Wobble (별도 실행)

```
final_selected_config_unconstrained.csv
        │
        ▼  compute_probe_only_with_wobble.py
        │   (nohup 백그라운드: run_wobble_nohup.sh)
        │
        ├──► final_test_metrics_wobble.csv
        └──► final_new_metrics_wobble.csv
```

### 2.3 시각화

```
입력                              스크립트                      출력 (figures/)
─────────────────────────────────────────────────────────────────────────────
data/alpha0_vs_best_full.csv  →  plot_alpha0_vs_best.py    →  fig1~8_*.png
feb23/results_vl0_*           →  plot_vl0_alpha_sweep.py   →  vl0_alpha_sweep_*.png
unconstrained_results_summary  →  plot_probe_only.py        →  probe_*.png
(final_test + final_new)
```

---

## 3. 파일별 저장 역할

### 3.1 입력 (외부/선행 단계)

| 파일 | 출처 | 용도 |
|------|------|------|
| `data/alpha0_vs_best_full.csv` | feb25에서 복사 | CPLfold α=0 vs best 비교 |
| `feb8/results_updated/outputs/` | feb8 학습 결과 | checkpoint, val sweep |
| `feb23/results_vl0_*` | feb23 VL0 실험 | α sweep 시각화 |

### 3.2 Config & Probe 결과 (march1 내부 생성)

| 파일 | 생성 스크립트 | 내용 |
|------|---------------|------|
| `final_selected_config_unconstrained.csv` | select_unconstrained_best_config.py | 모델별 best (layer, k, threshold) |
| `final_test_metrics.csv` | compute_feb8_probe_only_metrics.py | TS0 partition 상세 (precision, recall, F1, TP, FP, FN) |
| `final_new_metrics.csv` | compute_feb8_probe_only_metrics.py | NEW partition 상세 |
| `final_*_wobble.csv` | compute_probe_only_with_wobble.py | Wobble(GU) 포함 평가 |
| `unconstrained_results_summary.csv` | build_summary_table.py | TS0+NEW 요약 (ts0_f1, new_f1 등) |

### 3.3 시각화 출력

| 디렉터리 | 생성 스크립트 | 파일 수 |
|----------|---------------|---------|
| `figures/` | plot_alpha0_vs_best.py | fig1~8 (8개) |
| `figures/` | plot_vl0_alpha_sweep.py | vl0_alpha_sweep_* (3개) |
| `figures/` | plot_probe_only.py | probe_* (2개) |

### 3.4 로그/임시

| 파일 | 생성 | 용도 |
|------|------|------|
| `probe_only_progress.log` | compute_feb8_probe_only_metrics.py | probe 진행 상황 (tail -f) |
| `probe_only_wobble_progress.log` | compute_probe_only_with_wobble.py | wobble 진행 상황 |
| `nohup_wobble.out` | nohup | wobble 표준출력/에러 |

---

## 4. 권장 실행 순서

1. **Config 선택**  
   `python select_unconstrained_best_config.py`  
   → `final_selected_config_unconstrained.csv` 생성

2. **Probe 평가 (TS0 + NEW)**  
   `bash run_probe_only.sh`  
   또는 TS0만: `--dataset ts0`  
   NEW만: `--dataset new` (기존 TS0 CSV 로드)

3. **요약 테이블**  
   `python build_summary_table.py`  
   → `unconstrained_results_summary.csv` 갱신

4. **시각화**  
   `bash scripts/run_all_plots.sh`  
   또는 개별: `python scripts/plot_probe_only.py` 등

5. **Wobble (선택, 백그라운드)**  
   `bash run_wobble_nohup.sh`

---

## 5. Probe-only 저장 세부사항

- **compute_feb8_probe_only_metrics.py**  
  - 모델별로 TS0 → NEW 순서 실행  
  - 각 모델 완료 시 `write_partial_csv()` 호출 → `final_test_metrics.csv`, `final_new_metrics.csv` 갱신  
  - `--dataset ts0`: TS0만 계산, 기존 NEW 유지  
  - `--dataset new`: NEW만 계산, 기존 TS0 로드  
  - `--dataset both`: 둘 다 (기본값)

- **경로**  
  - config: `--config-csv march1/final_selected_config_unconstrained.csv`  
  - 출력: `--output-dir march1`  
  - 로그: `--progress-log march1/probe_only_progress.log`
